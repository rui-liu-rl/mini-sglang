# =====================================================================================
# FILE MAP -- scheduler/prefill.py
# =====================================================================================
# WHAT THIS FILE IS: the WAITING CLIPBOARD and the ADMISSION RULES for new orders. It decides
#                    which waiting orders may be "read" (prefilled) in the next pass, and how
#                    much of each.
#
# ELI5: New orders queue on a clipboard in arrival order. Before the head waiter lets one onto
#       the next tray she asks FOUR questions:
#         1. Is there a free numbered peg?                        (table_manager)
#         2. Does part of this order's beginning already have
#            notes in the pantry? Then don't redo that part.      (prefix cache match)
#         3. Will the pantry have room for the rest of this order
#            AND everyone already eating, in the WORST case?      (reserved_size + available_size)
#         4. Is there still room in this pass's word budget?       (token_budget)
#       If an order is too long for the budget, she reads only the first CHUNK now and keeps
#       the rest on the clipboard for the next pass ("chunked prefill").
#
# KEY IDEA 1 -- WORST-CASE RESERVATION. Admission counts, for every order, input tokens that
#       still need notes PLUS the maximum new tokens it may generate. So once admitted, an order
#       can ALWAYS finish: the pantry can never run dry mid-answer, and no running order ever
#       has to be thrown out.
# KEY IDEA 2 -- STRICT ORDER. The clipboard is first-in-first-out and admission STOPS at the
#       first order that doesn't fit (no skipping ahead). Fair, but a big order at the front can
#       hold back small ones behind it.
#
# WHO CALLS IT: scheduler/scheduler.py (PrefillManager.schedule_next_batch each iteration).
# WHAT IT CALLS: CacheManager (match/lock/available_size), TableManager (pegs), DecodeManager
#                (inflight_tokens), core.Req / core.Batch.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


# ELI5: an order whose beginning has been read but whose end has NOT. It sits on the tray
#       (so its notes get written) but it produces no word yet, and it can't join the "still
#       eating" list.
# WHAT: two overrides turn off the two things only a finished prompt may do:
class ChunkedReq(Req):
    # no sampled word exists to append (the scheduler also skips chunked requests when
    # processing results, so this is a guard)
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


# ELI5: the ADMISSION CLERK for ONE prefill pass. Created fresh each time with a word budget and
#       a "memory already promised" counter, then asked to admit orders one by one.
@dataclass
class PrefillAdder:
    # WHAT: how many more prompt tokens this pass may still process (starts at max_extend_tokens).
    token_budget: int
    # WHAT: KV-cache tokens ALREADY PROMISED: the worst case of everyone currently decoding,
    #       plus the worst case of every order admitted earlier in this same pass.
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    # ELI5: questions 1-3 above for a brand-new order. Returns (cache ticket, peg number) if
    #       the order can start, else None.
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # Q1: no free peg -> cannot admit.
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # Q2: look for a shared beginning in the prefix cache. `handle.cached_len` is how many
        #     leading tokens already have notes somewhere in the pantry.
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        # WHAT: tokens that still need NEW notes, plus the most it may generate afterwards.
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # Q3 (first look): is there room? `available_size` counts free pages plus cache entries
        #     nobody is using (those can be evicted).
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # WHY lock now: pin the matched prefix so eviction can't throw those notes away while
        #     we use them.
        self.cache_manager.lock(handle)
        # Q3 (second look): locking moves the prefix from "evictable" to "protected", which
        #     SHRINKS available_size. Check again; if it no longer fits, undo the lock.
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # Q1: take a peg.
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # ELI5: the new order's row must begin with the SAME pantry slots as the shared
            #       prefix: copy its token ids and the slot numbers into the new peg's strips.
            #       From then on this order "reads" the old notes as if they were its own.
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    # ELI5: turn a waiting order into a real Req (or a ChunkedReq) for this tray, and write the
    #       prompt tokens it will process into the peg's token strip.
    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        # WHAT: how much of the prompt still needs reading, and how much of that fits now.
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        # WHAT: if only part of the rest fits, this is a CHUNKED request for now.
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        # WHAT: spend the budget; and promise memory for the WHOLE rest of the prompt plus the
        #       full allowed output (not just this chunk).
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # WHAT: copy just this chunk's token ids into the peg's token strip (pinned memory +
        #       non_blocking = an asynchronous host-to-GPU copy).
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        # GOTCHA: input_ids is cut to cached_len + chunk_size. So for a chunked order, the Req's
        #         device_len is the END OF THIS CHUNK, not the end of the prompt.
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    # ELI5: "Can this waiting order join the tray?" Returns the Req, or None if it can't.
    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # no word budget left in this pass
        if self.token_budget <= 0:
            return None

        # WHAT: a CONTINUING chunk: this order already has a peg and a cache ticket from its
        #       earlier chunk. Reuse them and carry on from where it stopped (cached_len was
        #       advanced to the end of the previous chunk by complete_one()).
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # WHAT: a NEW order: ask questions 1-3; if it passes, build its Req.
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


# ELI5: the WAITING CLIPBOARD itself, plus the routine that builds the next prefill tray.
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    # WHAT: orders accepted but not (fully) read yet, oldest first.
    pending_list: List[PendingReq] = field(default_factory=list)

    # WHAT: put a new slip at the end of the clipboard.
    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    # ELI5: "Build the next prefill tray, or say there isn't one."
    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # WHAT: start the "memory promised" counter with the worst case of every order currently
        #       decoding, so new prompts never take space those orders will still need.
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        # WHAT: walk the clipboard from the front, admitting while orders fit.
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                # WHAT: an order that only got a partial read stays on the clipboard, remembering
                #       its progress, to be continued next pass.
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        # WHAT: new clipboard = the unfinished (chunked) orders first, then everyone who was not
        #       reached. Fully admitted orders drop off the clipboard.
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    # WHAT: cancel a waiting order. Returns its partial Req (if it was mid-chunk, because that
    #       one is holding a peg and a cache ticket that must be freed) or None.
    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    # WHAT: is anyone waiting for a first (or next) read?
    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
