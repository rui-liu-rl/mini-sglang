# =====================================================================================
# FILE MAP -- distributed/info.py
# =====================================================================================
# WHAT THIS FILE IS: the answer to "who am I among the GPU workers?"
#
# ELI5: With 4 GPUs there are 4 chef processes. Each wears a name tag: "rank 2 of 4". Rank 0 is
#       the PRIMARY: it talks to the outside world (reads requests, sends tokens out, prints
#       logs). Layers consult this tag to know which slice of a weight matrix is theirs.
#
# WHO CALLS IT: engine/engine.py sets it once at startup; layers (linear, embedding, attention)
#               and loggers read it.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass


# WHAT: rank = this worker's number (0-based); size = how many workers (the tensor-parallel degree).
#       frozen: the name tag cannot be edited.
@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    # WHAT: rank 0 is the primary worker.
    def is_primary(self) -> bool:
        return self.rank == 0


# ELI5: the one name tag for this process, hung on a global hook.
_TP_INFO: DistributedInfo | None = None


# WHAT: set exactly once per process (a second call raises).
def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


# WHAT: read the tag; error if it was never set (code running before Engine.__init__).
def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


# WHAT: read the tag if there is one, else None (used by the logger, which may run early).
def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
