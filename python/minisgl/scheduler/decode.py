# =====================================================================================
# FILE MAP -- scheduler/decode.py
# =====================================================================================
# WHAT THIS FILE IS: the list of orders that are "mid-meal", i.e. already prefilled and now
#                    producing one word per step.
#
# ELI5: The head waiter keeps a clipboard titled "TABLES STILL EATING". Every step, all the
#       tables on that clipboard get their next word cooked together in one pass. When a table
#       is finished it is crossed off.
#
# WHO USES IT: scheduler/scheduler.py (asks it for the next decode tray) and
#              scheduler/prefill.py (asks how much memory the running tables still need).
# WHAT IT CALLS: core.Batch / core.Req only.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    # WHAT: the KV-cache page size, needed only for the memory estimate in inflight_tokens.
    page_size: int
    # WHAT: all requests currently in the decode phase. A set (no duplicates, fast removal).
    running_reqs: Set[Req] = field(default_factory=set)

    # ELI5: "Add the tables that just got served, and cross off any that are done."
    # WHAT: merge `reqs` into the running set, keeping only those that can still decode
    #       (remain_len > 0). Called after every forward pass with that pass's requests.
    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    # WHAT: drop one request (it finished by EOS or was freed).
    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    # WHAT: cancel by id. Returns the Req so the caller can free its resources, or None if this
    #       id is not decoding (it may still be waiting or already gone).
    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    # ELI5: "How much pantry space do the tables still eating are going to need, worst case?"
    # WHAT: sum of every running request's remaining allowed tokens, plus a one-page reserve per
    #       request because the page they are writing into may be only partly used.
    # WHY: the prefill side uses this as its `reserved_size`. A new prompt is admitted only if
    #      there is room for ITS needs AND the worst-case needs of everyone already running.
    #      That is how the engine promises never to run out of KV memory half-way through a
    #      decode, so it never has to evict a running request.
    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    # WHAT: build the next decode tray, or None if nobody is eating.
    # GOTCHA: sorted by uid. In tensor parallelism every GPU worker builds its own tray from its
    #         own set; a set has no guaranteed order, so without the sort the workers would put
    #         the same requests in DIFFERENT orders and their results would no longer line up.
    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
