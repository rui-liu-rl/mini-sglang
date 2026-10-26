# =====================================================================================
# FILE MAP -- benchmark/client.py
# =====================================================================================
# WHAT THIS FILE IS: the shared machinery of mini-sglang's online benchmarks: a load generator that
#                    sends many chat requests to a running server through the OpenAI-style API,
#                    records WHEN each streamed chunk arrives, and turns those timestamps into
#                    the numbers from Part I chapter 5 (TTFT, TPOT, end-to-end time, throughput).
#
# ELI5: Picture a crowd of pretend customers. Each walks up to the front desk, orders, and then
#       stands with a stopwatch, writing down the clock time at which every spoonful of the answer
#       arrives. At the end we read the stopwatches:
#
#           tics = [t_start, t_first_chunk, t_2nd_chunk, ..., t_last_chunk]
#           TTFT  = t_first_chunk - t_start                 (time to first token)
#           TPOT  = each gap between consecutive chunks      (time per output token / ITL)
#           E2E   = t_last - t_start                         (whole answer)
#           throughput = tokens / (latest time - earliest time over ALL requests)
#
# THE PIECES
#       BenchmarkTrace      one planned request: when to send it, the text, the lengths
#       benchmark_one       send ONE request, stream the reply, return the timestamps
#       benchmark_one_batch send many at once (all together)
#       benchmark_trace     send many at their recorded times (replaying a real workload trace)
#       process_benchmark_results   timestamps -> printed statistics
#       read_qwen_trace / read_mooncake_trace   load two public trace file formats
#       Console / make_console      progress bars and in-flight counters
#
# WHO USES IT: benchmark/online/*.py, benchmark/offline/*.py (top-level benchmark folder).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import asyncio
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, overload

from minisgl.utils import UNSET, Unset, init_logger
from openai import AsyncOpenAI as OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

logger = init_logger(__name__)


# ELI5: one line of a workload plan: "at second T, send this text, and expect about this many
#       answer tokens". input_length is optional and used to force a prompt length.
@dataclass(frozen=True)
class BenchmarkTrace:
    timestamp: float  # unit (second)
    message: str
    output_length: int  # output length in tokens
    input_length: int | None = None  # input length in tokens, optional


# ELI5: one request's stopwatch readings, saved in a compact form.
# FORMAT (as_json): [input_len, output_len, tic0, tic1, tic2, ...], times already shifted so the
#       earliest request starts at 0. from_json reads it back (checks the two lengths are whole numbers).
@dataclass(frozen=True)
class BenchOneResult:
    tics: List[float]
    input_len: int
    output_len: int

    def as_json(self) -> List[float]:
        return [self.input_len, self.output_len] + self.tics

    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        # check raw[0] and raw[1] are integers
        assert raw[0].is_integer() and raw[1].is_integer()
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))


# WHAT: the raw output of benchmark_one: the prompt text, the lengths, and the timestamps
#       (time.perf_counter() readings, in seconds, from an arbitrary starting point).
@dataclass(frozen=True)
class RawResult:
    input_len: int | None
    output_len: int
    message: str
    tics: List[float]


# WHAT: a counter that remembers its highest value ever (used for "max requests in flight").
@dataclass
class Counter:
    current: int = 0
    history_max: int = 0

    def inc(self, n=1):
        self.current += n
        self.history_max = max(self.history_max, self.current)

    def dec(self, n=1):
        self.current -= n
        assert self.current >= 0


# ELI5: the scoreboard: four progress bars plus two live counters.
#   input_pbar    requests sent                 output_pbar   requests finished
#   prefill_pbar  requests that got their FIRST chunk (the prompt has been processed)
#   decode_pbar   chunks received after the first (decode steps)
#   inflight_counter  sent but not yet finished;  queue_counter  sent but no first chunk yet
#   (so queue_counter approximates "requests waiting for prefill")
@dataclass
class Console:
    input_pbar: tqdm
    output_pbar: tqdm
    prefill_pbar: tqdm
    decode_pbar: tqdm
    disabled: bool
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n=1):
        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n=1):
        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n=1):
        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n=1):
        self.decode_pbar.update(n)

    @contextmanager
    def inflight(self, n=1):
        self.update_input(n)
        yield
        self.update_output(n)

    @contextmanager
    def log_stats(self):
        yield
        self.input_pbar.close()
        self.output_pbar.close()
        self.prefill_pbar.close()
        self.decode_pbar.close()
        if not self.disabled:
            max_inflight = self.inflight_counter.history_max
            max_queue = self.queue_counter.history_max
            logger.info(f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}")


# WHAT: all requests' readings together; as_json / from_json turn it into plain lists so a
#       benchmark run can be saved to a file and analysed later.
@dataclass(frozen=True)
class BenchmarkResult:
    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


# WHAT: build the scoreboard. The prefill bar counts requests (one first chunk per request); the
#       decode bar counts the remaining chunks: sum of output lengths minus one per request.
#       The string juggling with n_fmt just widens the number column so the bars line up.
def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0
    n_fmt_align = 5
    prefill_tokens = num_requests
    decode_tokens = sum_output_length - prefill_tokens

    if len(str(decode_tokens)) > n_fmt_align:
        n_fmt_align = len(str(decode_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")
        BAR_FORMAT_1 = BAR_FORMAT_0

    if len(str(prefill_tokens)) < len(str(decode_tokens)):
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")

    disabled = not use_pbar
    input_pbar = tqdm(
        total=num_requests,
        desc="Requests sent",
        position=0,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    output_pbar = tqdm(
        total=num_requests,
        desc="Requests done",
        position=1,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    prefill_pbar = tqdm(
        total=prefill_tokens,
        desc="Prefill token",
        position=2,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    decode_pbar = tqdm(
        total=decode_tokens,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


# ELI5: "Invent a prompt that is exactly n tokens long." Pick n random token ids from the lower half
#       of the vocabulary, turn them into text, then re-tokenise that text; text and tokens do not
#       always round-trip, so if the count is off, top up with more random ids or cut, and try again
#       (up to 64 times, then give up). Random words mean there is NO shared prefix between prompts,
#       so the prefix cache cannot help: a harsh test.
def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer."""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            token_ids = token_ids[:n]

    raise ValueError("Failed to generate a message of the desired length.")


# ELI5: ONE pretend customer. Sends the prompt, then writes down the clock each time a streamed
#       chunk arrives.
# WHAT (in order):
#   1. request settings: ignore_eos=True (always produce exactly output_length tokens, so every run
#      does the same amount of work), top_k=1 and temperature 0.0 (deterministic greedy).
#      `input_length_override` is an internal extra field "that might work for our system"; the
#      server's request form ignores unknown fields, so today it has no effect on mini-sglang.
#   2. send with stream=True. tics[0] is the clock right after the call returns, tics[1] the first
#      chunk, and so on.
#   3. progress bookkeeping: the first chunk marks "prefill done"; later chunks are decode steps.
# GOTCHA (my reading): with the OpenAI client, create(stream=True) returns once the response
#      HEADERS arrive, which the server sends before the first token, so "TTFT" here is measured
#      from that moment, not from the instant the request was written to the socket.
# GOTCHA: every chunk is counted, including the server's closing chunk (finish_reason "stop"), so
#      a request with output_length N produces about N + 2 tics.
async def benchmark_one(
    client: OpenAI,
    prompt: str,
    output_length: int,
    model: str,
    *,
    pbar: Console | bool = True,
    extra_body: Dict[str, Any] | None = None,
    input_length: int | None = None,  # a hack to force input length
) -> RawResult:
    if isinstance(pbar, bool):
        pbar = make_console(1, output_length, use_pbar=pbar)
    with pbar.inflight(1):
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        # this is an internal kwargs that might work for our system
        if input_length is not None:
            kwargs["input_length_override"] = input_length
        kwargs.update(extra_body or {})  # can override kwargs
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=output_length,
            temperature=0.0,
            extra_body=kwargs,
        )
        tics = [time.perf_counter()]
        async for _ in response:
            tics.append(time.perf_counter())
            if len(tics) == 2:
                pbar.update_prefill()
            elif len(tics) <= output_length + 1:
                pbar.update_decode()
        return RawResult(
            input_len=input_length,
            output_len=output_length,
            message=prompt,
            tics=tics,
        )


# ELI5: send ALL the prompts at the same moment (asyncio.gather) and wait for everyone. This is
#       the "burst" workload: the server sees a wall of requests at once.
async def benchmark_one_batch(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int] | int,
    model: str,
    *,
    extra_body: Dict[str, Any] | None = None,
    input_lengths: List[int | None] | None = None,
    pbar: Console | bool = True,
) -> List[RawResult]:
    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    if isinstance(pbar, bool):
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)
    if input_lengths is None:
        l: List[int | None] = [None] * len(prompts)
        input_lengths = l  # work-around for typing bug

    tasks = [
        benchmark_one(
            client=client,
            prompt=prompt,
            output_length=output_length,
            model=model,
            pbar=pbar,
            extra_body=extra_body,
            input_length=input_length,
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True
        )
    ]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


# ELI5: replay a recorded workload. Each request sleeps until its own send time, then goes. The
#       earliest timestamp minus 1 second is the zero point, so the first request is sent about
#       1 second after the start, and the recorded spacing between arrivals is reproduced.
async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
) -> List[RawResult]:
    if isinstance(pbar, bool):
        sum_output_len = sum(msg.output_length for msg in msgs)
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1

    async def benchmark_timed(msg: BenchmarkTrace):
        target = start + msg.timestamp - offset
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await benchmark_one(
            client, msg.message, msg.output_length, model, pbar=pbar, input_length=msg.input_length
        )

    tasks = [benchmark_timed(msg) for msg in msgs]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


# ELI5: read the stopwatches. Given every request's tics, print and (if a tokenizer is supplied)
#       return the results. The two @overload lines only tell type checkers: with a tokenizer you
#       get a BenchmarkResult back, without one you get None.
# WHAT it computes (timestamps in seconds; TTFT and TPOT are printed in milliseconds):
#       first_times  each request's first gap (tics[1] - tics[0])           -> TTFT
#       accum_times  all later gaps of all requests, pooled                  -> TPOT / ITL
#       e2e_times    last tic minus first tic per request                    -> end to end (seconds)
#       For each list: average and the p50 / p90 / p99 / max, using plain index picking
#       (sorted_list[int(len * q)], no interpolation; with fewer than 100 samples p99 is simply
#       near the maximum).
#       Duration = latest tic over all requests minus earliest; throughput = tics / duration.
# GOTCHA: "tokens" in the throughput line are the number of tics, which includes each request's
#       start tic and the closing chunk (about 2 extra per request).
# GOTCHA: the closing chunk arrives immediately after the last token, so it adds one near-zero gap
#       per request to the TPOT pool, nudging TPOT slightly down (about 1/N for N output tokens).
@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    accum_times: List[float] = []
    first_times: List[float] = []
    results = [r.tics for r in raw_data]
    for tics in results:
        deltas: List[float] = []
        for i in range(len(tics) - 1):
            diff = tics[i + 1] - tics[i]
            deltas.append(diff)
        first_times.append(deltas[0])
        accum_times.extend(deltas[1:])

    e2e_times = [tics[-1] - tics[0] for tics in results]
    first_times.sort()
    accum_times.sort()
    e2e_times.sort()

    def _print_stats(times: List[float], scale: float = 1.0) -> Tuple[float, ...]:
        assert len(times) > 0
        return (
            scale * sum(times) / len(times),  # avg
            scale * times[int(len(times) * 0.5)],  # p50
            scale * times[int(len(times) * 0.9)],  # p90
            scale * times[int(len(times) * 0.99)],  # p99
            scale * max(times),  # max
        )

    def _fmt(x: float) -> str:
        if x >= 1000:
            return f"{int(x):>6}"
        elif x >= 10:
            return f"{x:>6.2f}"
        else:
            return f"{x:>6.4f}"

    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000)
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000)
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times)

    min_time = min(min(r) for r in results)
    max_time = max(max(r) for r in results)
    dur = max_time - min_time
    assert dur > 0, "Duration must be positive"

    num_tokens = sum(len(tic) for tic in results)
    num_requests = len(results)

    logger.info(f"Num requests: #{num_requests}, Num tokens: #{num_tokens}")
    logger.info(
        f"TTFT: {_fmt(avg_ttft)} ms (p50: {_fmt(p50_ttft)} ms, p90: {_fmt(p90_ttft)} ms,"
        f" p99: {_fmt(p99_ttft)} ms, max: {_fmt(max_ttft)} ms)"
    )
    logger.info(
        f"TPOT: {_fmt(avg_tpot)} ms (p50: {_fmt(p50_tpot)} ms, p90: {_fmt(p90_tpot)} ms,"
        f" p99: {_fmt(p99_tpot)} ms, max: {_fmt(max_tpot)} ms)"
    )
    logger.info(
        f"E2E:  {_fmt(avg_e2e) }  s (p50: {_fmt(p50_e2e) }  s, p90: {_fmt(p90_e2e) }  s,"
        f" p99: {_fmt(p99_e2e) }  s, max: {_fmt(max_e2e) }  s)"
    )
    logger.info(f"Duration: {_fmt(dur)} s")
    logger.info(f"Throughput: {_fmt(num_tokens / dur)} token/s, {_fmt(num_requests / dur)} req/s")

    # normalize the time to start from zero
    results = [[r - min_time for r in tics] for tics in results]
    if isinstance(tokenizer, Unset):
        return None

    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=(
                    r.input_len
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))
                ),
                output_len=r.output_len,
            )
            for r in raw_data
        ]
    )


# ELI5: load a trace file in the Qwen format: one JSON object per line with chat_id, parent_chat_id,
#       timestamp (seconds), input_length, output_length (the rest is unused here).
#       dummy=True: make ONE long random prompt and cut every request's prompt from its beginning
#       (so prompts share prefixes, favouring a prefix cache); dummy=False: a fresh random
#       prompt of the right length for each line (no sharing). `n` keeps only the first n lines.
def read_qwen_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    class JSONInput(BaseModel):
        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


# ELI5: the same for the Mooncake trace format (timestamps in MILLISECONDS, divided by 1000 here;
#       fields timestamp, input_length, output_length, hash_ids).
def read_mooncake_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    class JSONInput(BaseModel):
        timestamp: int
        input_length: int
        output_length: int
        hash_ids: List[int]  # unused for now

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


# ELI5: speed the replay up or down. Shift times so the first request is at 0, multiply by `scale`
#       (0.5 = twice as fast, 2.0 = half the request rate), and sort by time.
def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    min_tic = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_length=trace.input_length,
                output_length=trace.output_length,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )


# WHAT: ask the server which model it serves (GET /v1/models) and return the first id.
async def get_model_name(client: OpenAI) -> str:
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")
