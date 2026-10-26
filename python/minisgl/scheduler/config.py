# =====================================================================================
# FILE MAP -- scheduler/config.py
# =====================================================================================
# WHAT THIS FILE IS: the scheduler's settings, plus the addresses of its "phone lines".
#
# ELI5: Every worker in the Kitchen needs to know (a) how the head waiter should behave, and
#       (b) which phone numbers to call to reach the front desk, the card-maker and the other
#       chefs. This file is the laminated card with those settings and phone numbers.
#
# WHO USES IT: scheduler/scheduler.py, scheduler/io.py, server/launch.py and server/args.py.
# WHAT IT CALLS: engine.EngineConfig (this class EXTENDS it, so it has all engine settings too).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


# WHAT: builds a string like ".pid=12345" for the CURRENT process id.
# WHY: the socket file names below include this suffix, so two mini-sglang servers running on
#      one machine do not fight over the same /tmp files.
def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


# WHAT: frozen=True makes instances read-only after creation (nobody can quietly change a
#       setting mid-run). SchedulerConfig inherits every EngineConfig field and adds its own.
# GOTCHA: engine.py's _adjust_config() uses object.__setattr__ to change a few fields anyway;
#         its own comment calls that "dangerous". It happens once, at startup.
@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # WHAT: the most prompt tokens ("extend" tokens) one prefill pass may process. A longer
    #       prompt is cut into chunks of at most this size (chunked prefill).
    # WHY: bounds the memory and time of one forward pass so a huge prompt cannot starve
    #      everyone else or run the GPU out of memory.
    max_extend_tokens: int = 8192
    # WHAT: which prefix cache to use: "radix" (shares common prompt beginnings) or "naive"
    #       (no sharing).
    cache_type: str = "radix"
    # WHAT: True when mini-sglang is used as a Python library (llm/llm.py) instead of a server;
    #       no network queues are created in that case.
    offline_mode: bool = False

    # networking config
    # WHAT: a per-process suffix, appended to every socket address below.
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    # ELI5: three "phone lines", each a file in /tmp (a local "ipc" socket, not a network call).
    # WHAT: tokenizer -> scheduler: new requests come in on this line.
    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    # WHAT: scheduler -> detokenizer: freshly generated tokens go out on this line. (With the
    #       default shared tokenizer, new requests from the API server arrive on this same line.)
    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    # WHAT: scheduler rank 0 -> the other tensor-parallel schedulers: a broadcast line, so every
    #       GPU worker sees exactly the same requests in the same order.
    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    # WHAT: the longest forward pass the engine must be ready for (used to size communication
    #       buffers). Here it equals max_extend_tokens.
    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    # ELI5: a phone line has two ends. One side "creates" it (binds the address) and the other
    #       side "dials in" (connects). This says who creates the scheduler -> detokenizer line.
    # WHAT: True here, meaning the scheduler creates it. server/args.py OVERRIDES this to
    #       `not share_tokenizer`: in the default setup the tokenizer and detokenizer are ONE
    #       worker process that already created that address, so the scheduler must dial in.
    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
