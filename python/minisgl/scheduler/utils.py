# =====================================================================================
# FILE MAP -- scheduler/utils.py
# =====================================================================================
# WHAT THIS FILE IS: two tiny helper records used inside the scheduler.
#
# ELI5: PendingReq is an order that is still on the "waiting" clipboard: accepted at the front
#       desk, but not yet given a peg or any cooking. ScheduleResult is a small bundle for
#       reporting a scheduling decision.
#
# WHO USES IT: scheduler/prefill.py builds PendingReq; ScheduleResult is a plain record.
# WHAT IT CALLS: nothing at run time (imports are for type hints only).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


# ELI5: a waiting order. It is not a Req yet: a Req needs a peg (table_idx), a cache ticket
#       (cache_handle) and a known cached_len, and none of those exist until it is admitted.
@dataclass
class PendingReq:
    uid: int
    # SHAPE: 1-D int32 CPU tensor, the whole prompt.
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    # WHAT: set only if this prompt is long and is being prefilled in CHUNKS: it points at the
    #       partial Req from the previous chunk so the next chunk can pick up where it left off.
    chunked_req: ChunkedReq | None = None

    # WHAT: prompt length in tokens.
    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    # WHAT: how many tokens this request is allowed to generate (the client's max_tokens,
    #       already clipped by the scheduler).
    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


# WHAT: a record of "these requests were scheduled, and their outputs go to these indices".
# GOTCHA: not referenced by the current scheduler loop; kept as a plain record type.
@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
