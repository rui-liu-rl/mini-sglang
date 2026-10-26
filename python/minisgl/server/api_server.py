# =====================================================================================
# FILE MAP -- server/api_server.py
# =====================================================================================
# WHAT THIS FILE IS: the FRONT DESK of the Kitchen. A web server (FastAPI + uvicorn) that
#                    takes orders over HTTP, hands them to the card-maker, and streams the
#                    answer back word by word.
#
# ELI5: Customers walk up and say "I'd like to chat". The front desk:
#         1. checks the order form is filled in properly        (pydantic request models)
#         2. writes a TICKET NUMBER on it (uid)                 (FrontendManager.new_user)
#         3. hands it to the card-maker and instantly turns to the next customer
#         4. keeps a "light" per ticket; when new words arrive for ticket 7, light 7 turns on
#            and the desk passes those words to customer 7 while the connection stays open
#         5. if the customer walks out, tells the kitchen to cancel
#
# THE ASYNC IDEA: one waiter, many tables. `async def` + `await` means "while I wait for this
#       table's food, I'll serve another table". No threads, no locks.
#
# THREE MAIN ACTORS INSIDE THIS FILE
#   app                : the FastAPI web app with its endpoints (/generate, /v1/chat/completions ...)
#   FrontendManager    : the desk's LEDGER (ticket counter, per-ticket piles and lights)
#   listen()           : one background task, the MAIL SORTER that files replies by ticket
#
# WHO CALLS IT: server/launch.py (run_api_server).
# WHAT IT CALLS: message/* forms, utils/mp.py (async ZeroMQ queues), core.SamplingParams.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

# ELI5: the one shared ledger for the whole front desk, created in run_api_server() at the bottom.
_GLOBAL_STATE = None


# WHAT: fetch the ledger; the assert catches "used before run_api_server set it up".
def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


# WHAT: a packet from the card-reader is either one UserReply or a BatchFrontendMsg holding
#       several. Return a flat list either way.
def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


# =====================================================================================
# THE ORDER FORMS (pydantic models). FastAPI uses them to validate incoming JSON BEFORE our
# code runs, and to generate the API docs. Think of them as the printed form the customer fills in.
# =====================================================================================


# WHAT: the simple /generate endpoint's form: raw prompt text, a length cap, and a flag.
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


# WHAT: one chat message. Only the three standard roles are accepted.
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    # GOTCHA: required by the schema (so clients like the OpenAI SDK work) but never used to
    #         choose a model: the server serves the one model it loaded at startup.
    model: str

    # WHAT: give EITHER `prompt` (a plain string) OR `messages` (a chat).
    prompt: str | None = None
    messages: List[Message] | None = None

    # WHAT: default 16 new tokens if the client does not say (OpenAI's convention).
    max_tokens: int = 16
    # GOTCHA: the HTTP default is 1.0 (OpenAI's convention), while SamplingParams' own default
    #         is 0.0 (greedy). A client that omits temperature gets SAMPLING here, not greedy.
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    # GOTCHA: the fields n, stop, presence_penalty and frequency_penalty are ACCEPTED so that
    #         standard clients do not error, but nothing below reads them. They are ignored.
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


# WHAT: the shape of one entry in the GET /v1/models answer.
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# =====================================================================================
# THE LEDGER
# =====================================================================================


# ELI5: everything the front desk remembers while orders are in flight.
@dataclass
class FrontendManager:
    config: ServerArgs
    # WHAT: the out-tray towards the card-maker (TokenizeMsg, AbortMsg). Async-friendly.
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    # WHAT: the in-tray from the card-reader (UserReply). Async-friendly.
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    # WHAT: the next ticket number to hand out.
    uid_counter: int = 0
    # WHAT: has the mail-sorter task been started yet?
    initialized: bool = False
    # ELI5: per ticket, the PILE of words that arrived but have not been passed to the customer.
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    # ELI5: per ticket, the LIGHT that says "there are new words in your pile".
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    # ELI5: hand out the next ticket number and prepare an empty pile and an unlit light for it.
    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    # ELI5: THE MAIL SORTER. One endless background task for the whole server. It waits for a
    #       packet from the card-reader, then files each reply in its ticket's pile and
    #       switches that ticket's light on.
    # WHAT: `await ...get()` suspends this task (not the whole server) until a packet arrives.
    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                # WHAT: a reply for a ticket we already forgot (customer left, abort done) is
                #       silently dropped.
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    # WHY lazy: asyncio.create_task needs a RUNNING event loop, which only exists once uvicorn
    #     is serving. So the sorter is started on the first send_one, not at construction.
    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    # WHAT: send any message towards the card-maker (starting the mail sorter if needed).
    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    # ELI5: the CUSTOMER'S side of the light. "Sleep until my light turns on; then take my whole
    #       pile; hand the words over one by one; stop after the final one."
    # WHAT: an async GENERATOR (it `yield`s values). Callers write `async for ack in ...`.
    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            # WHAT: grab the entire pile and leave a fresh empty one. Several replies can have
            #       piled up while we were not looking; none are lost.
            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            # WHAT: `ack` is now the LAST reply of this batch. If it is the final one, stop.
            if ack and ack.finished:
                break

        # WHAT: tidy the books once the ticket is finished.
        del self.ack_map[uid]
        del self.event_map[uid]

    # WHAT: format for the plain /generate endpoint: one "data: <text>" line per reply, then
    #       "data: [DONE]". Not strict SSE (a blank line should separate events); the interactive
    #       shell below parses exactly this simple format.
    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    # ELI5: format for OpenAI-compatible chat streaming. Each spoonful is a JSON "chunk" shaped
    #       like OpenAI's: a `delta` holding the NEW text (and, in the first chunk only, the role).
    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            # WHAT: the very first chunk announces who is speaking.
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            # WHAT: an empty piece of text is possible (held-back bytes); then no "content" key.
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            # WHAT: the SSE rule: "data: <payload>" followed by a BLANK line ("\n\n").
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        # GOTCHA: finish_reason is always "stop" here, even when the answer was cut off by
        #         max_tokens (OpenAI would say "length").
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    # ELI5: a WATCHER wrapped around a stream. Between spoonfuls it asks "is the customer still
    #       at the counter?" If not, it cancels the order.
    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            # WHAT: start the cleanup as its OWN task, because this generator is already being
            #       torn down. Then re-raise so the cancellation continues upward.
            asyncio.create_task(self.abort_user(uid))
            raise

    # ELI5: "Ticket `uid` left. Tear up its pile and light, and tell the kitchen to stop."
    async def abort_user(self, uid: int):
        # GOTCHA/GUESS: the 0.1 s pause is not explained in the source. A likely reason is to
        #               let in-flight chunks for this ticket drain before its books are removed.
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        # WHAT: the abort travels: this desk -> card-maker -> (as AbortBackendMsg) -> scheduler,
        #       which frees the order's peg and pantry pages.
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


# WHAT: FastAPI "lifespan": code before `yield` runs at startup, code after at shutdown. Here
#       only shutdown matters: close the ZeroMQ sockets.
@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


# =====================================================================================
# THE DOORS (endpoints)
# =====================================================================================


# ELI5: door #1: "give me text for this raw prompt". Simple format, always streaming.
@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    # ELI5: take a ticket number (and make a pile + light for it).
    uid = state.new_user()
    # ELI5: hand the slip to the card-maker. SamplingParams here only sets two fields; the rest
    #       use SamplingParams' own defaults (greedy, no top-k/top-p).
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    # WHAT: returning a StreamingResponse does NOT end the request: the connection stays open
    #       and the generator keeps producing bytes as the answer is written.
    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


# WHAT: a tiny "are you there?" door that some OpenAI clients probe.
@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


# ELI5: door #2, the main one: OpenAI-compatible chat. Any client that speaks the OpenAI API
#       can point at this server.
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    # WHAT: if the client sent `messages`, the "prompt" is a LIST of {role, content} dicts and
    #       the tokenizer will apply the chat template. Otherwise it is the plain `prompt` string.
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    # ELI5: ticket number, then send the slip (with the customer's adventurousness settings).
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    # WHAT: streaming reply: return immediately with an open connection, wrapped by the watcher.
    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    # WHAT: the customer wants the whole dish at once, so wait here, gluing pieces together.
    # GOTCHA: this path has no disconnect watcher; if the client leaves, generation continues
    #         to the end.
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    # GOTCHA: the usage token counts are placeholders (all 0) and finish_reason is always "stop".
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# WHAT: door #3: which model is loaded? (Clients like to ask.) Returns the model path.
@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


# =====================================================================================
# THE INTERACTIVE SHELL (used by `python -m minisgl.shell`): chat in your terminal with no HTTP.
# =====================================================================================


# WHAT: the shell's version of the chat door. It calls the same machinery directly (no HTTP).
async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    # GOTCHA (looks like an upstream bug, unverified by running): `lambda: _abort` returns the
    #         function `_abort` itself and never CALLS it, so this background task appears to do
    #         nothing. It does not matter in the normal case (the answer finished on its own).
    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



# ELI5: a little chat program in the terminal. You type a line, the Kitchen answers, you see
#       the words appear. Commands: /exit to quit, /reset to forget the conversation.
async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        # WHAT: the chat so far as (user text, assistant text) pairs, resent in full each turn,
        #       because the model has no memory between requests.
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            # WHAT: sampling settings come from MINISGL_SHELL_* environment variables (env.py).
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            # WHAT: read the stream directly and strip the "data: " prefix and trailing newline
            #       that stream_generate puts on each line.
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        # WHY kill: the worker processes were started with daemon=False, so they would keep
        #     running after the shell ends unless we stop them explicitly here.
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


# ELI5: open for business. Build the ledger and phone lines, hire the workers, then either serve
#       HTTP forever or run the terminal shell.
def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    # WHAT: the in-tray from the card-reader is OWNED (bound) by the front desk (create=True),
    #       the out-tray towards the tokenizers is bound only in separate-tokenizer mode.
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    # ELI5: now (phone lines exist) hire everyone and wait for their ready flags.
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        # WHAT: blocks forever, serving HTTP.
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
