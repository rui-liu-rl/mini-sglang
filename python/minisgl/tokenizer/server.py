# =====================================================================================
# FILE MAP -- tokenizer/server.py
# =====================================================================================
# WHAT THIS FILE IS: the main loop of the card-maker / card-reader worker process.
#
# ELI5: Picture a small room with one clerk and an in-tray. Slips arrive of THREE kinds:
#         "make cards from this text"      (TokenizeMsg)   -> clerk makes cards, passes them on
#                                                             to the head waiter (UserMsg)
#         "turn this new card into words"  (DetokenizeMsg) -> clerk makes words, sends them to the
#                                                             front desk (UserReply)
#         "ticket 7 left"                  (AbortMsg)      -> clerk passes a cancel to the waiter
#       The same clerk does all of it (that is why one process can be both the tokenizer and the
#       detokenizer).
#
# WHO CALLS IT: server/launch.py starts it as a separate process (once or several times).
# WHAT IT CALLS: tokenize.py (TokenizeManager), detokenize.py (DetokenizeManager), and the
#                ZeroMQ queues in utils/mp.py.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


# WHAT: a packet may be one message or a Batch wrapper holding many. Always return a list so the
#       caller can treat both the same way.
def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


# WHAT: inference_mode turns off PyTorch's gradient bookkeeping (no training here), which saves
#       a little time and memory for the tensors this worker creates.
@torch.inference_mode()
def tokenize_worker(
    *,
    # WHAT: where to load the tokenizer from (the model's folder or HF repo id).
    tokenizer_path: str,
    # WHAT: the address of the in-tray this worker READS from (TokenizeMsg / AbortMsg /
    #       DetokenizeMsg all arrive here).
    addr: str,
    # WHAT: True = this worker owns (binds) the in-tray address; False = it dials in.
    create: bool,
    # WHAT: the out-tray towards the scheduler (UserMsg, AbortBackendMsg).
    backend_addr: str,
    # WHAT: the out-tray towards the front desk (UserReply).
    frontend_addr: str,
    # WHAT: "local batch size": how many messages to gather before processing. With 1 the worker
    #       handles whatever it has; it never waits to fill a bigger batch.
    local_bs: int,
    # WHAT: only used in the log prefix and the readiness message.
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    # WHAT: where to announce "I'm ready" so the launcher knows it can open for business.
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    # WHY create=False for both out-trays: the scheduler and the front desk own those addresses
    #     (they bind); this worker simply dials in.
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    # WHAT: loads the Hugging Face tokenizer (its vocabulary and chat template).
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    # ELI5: raise the "ready" flag.
    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # ELI5: wait at the in-tray until at least one slip arrives...
            pending_msg = _unwrap_msg(recv_listener.get())
            # ...then also scoop up anything else that is ALREADY waiting (never waits for more).
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # WHAT: sort the slips by kind. The assert proves nothing was left over.
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)

            # --- job 1: new cards -> words, back to the front desk -----------------------
            # WHY first: finishing words already in flight keeps streaming smooth; the extra
            #     tokenizing work can wait a moment.
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                # WHAT: a one-item batch is sent as the item itself (smaller packet).
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # --- job 2: new text -> cards, on to the head waiter -------------------------
            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            # --- job 3: cancels, passed straight on to the head waiter -------------------
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        # WHAT: Ctrl+C (or the launcher shutting down) ends the loop quietly.
        pass
