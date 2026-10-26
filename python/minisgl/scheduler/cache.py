# =====================================================================================
# FILE MAP -- scheduler/cache.py
# =====================================================================================
# WHAT THIS FILE IS: the PANTRY CLERK. It owns the list of FREE pantry pages (KV-cache pages),
#                    hands pages out to orders, takes them back, and keeps the shared-prefix
#                    index (the prefix cache) in sync.
#
# ELI5: The pantry (KV cache) is a huge shelf of numbered slots. Each order needs one slot
#       per word to store its notes. The clerk keeps:
#         free_slots    : the list of pages nobody owns (a GPU tensor of page start numbers)
#         prefix_cache  : the index of notes that FINISHED or READ-ALREADY orders left behind,
#                         so new orders with the same beginning can borrow them
#       Three kinds of pantry space:
#           FREE        nobody owns it                            -> can be handed out at once
#           EVICTABLE   owned by the prefix cache, nobody using it -> can be thrown out if needed
#           PROTECTED   owned by the prefix cache AND some running order uses it -> untouchable
#       available_size = FREE + EVICTABLE  (what the clerk could produce on demand)
#
# PAGES AND SLOTS. One PAGE = page_size consecutive token slots. With the default page_size=1,
#       a page is one slot. Everything in the tables is counted in TOKEN SLOTS; free_slots
#       stores the FIRST slot of every free page, e.g. page_size=4 -> [0, 4, 8, 12, ...].
#
# WHO CALLS IT: scheduler/scheduler.py and scheduler/prefill.py.
# WHAT IT CALLS: kvcache/ (the prefix cache implementations, selected by name).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        # ELI5: at the start every page is free, so the list is 0, page_size, 2*page_size, ...
        device = page_table.device
        # SHAPE: (number of free pages,), int32, on the GPU.
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # WHAT: "radix" or "naive" prefix index, looked up by name in kvcache/__init__.py.
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        # WHAT: the SAME page_table tensor the engine owns (row = order's peg, column = position).
        self.page_table = page_table
        self.page_size = page_size

    # ELI5: "Does anyone's old notes match the beginning of this new order?"
    # WHY input_ids[: input_len - 1]: never match the LAST prompt token. At least one token must
    #     really go through the model to produce the logits for the first answer word, so the
    #     last token is always recomputed (this keeps Req's rule cached_len < device_len true).
    # WHAT: returns a MatchResult whose handle says how many leading tokens matched and gives
    #       access to their pantry slots. The match itself changes nothing about ownership.
    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    # ELI5: "How much pantry could I hand out if I had to?" = free pages + notes nobody is using.
    # WHAT: in TOKEN SLOTS.
    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    # ELI5: pin the notes behind this ticket so the clerk can't throw them away while in use.
    # WHAT: bumps reference counts along the tree path; PROTECTED space grows, EVICTABLE shrinks.
    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    # ELI5: "Give every order on the tray pantry pages for the words it is about to write."
    # WHAT: for each order, the pages that must exist for positions [cached_len, device_len):
    #       from page ceil(cached_len / page_size) up to page ceil(device_len / page_size).
    #       Often zero pages are needed (a decode step usually stays inside the current page);
    #       a new page is needed exactly when a new token crosses into a fresh page.
    #       Example, page_size=4: a decode step at device_len=5 needs page index 1 (positions
    #       4..7); at device_len=6, 7, 8 nothing; at device_len=9 it needs page index 2.
    # WHY one batch at a time: all needed pages for the whole tray are taken in a single
    #     _allocate call and written to the page table with a single GPU assignment.
    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    # ELI5: "AN ORDER IS DONE (or just finished its first read). Put its notes where they belong."
    #       There are two moments this is called:
    #         finished=False : right after a prompt was READ (prefill). Save the prompt's notes
    #                          in the shared index NOW, so a customer with the same beginning
    #                          can borrow them while this order is still being written.
    #         finished=True  : the order is over. Save everything useful (prompt AND answer, which
    #                          is what the next turn of a chat will start with) and give back
    #                          the rest.
    #
    #       Worked example (page_size = 1). A prompt of 10 tokens; 4 were borrowed from the cache
    #       when the order started (slots S0..S3, in the tree); the order computed 6 new ones
    #       (slots N4..N9).
    #       1) After prefill  (finished=False): insert 10 tokens. The tree already has the first 4,
    #          so only a new leaf for tokens 4..9 is added (holding N4..N9). Nothing to free.
    #          The order's ticket is swapped for one covering all 10 tokens and re-locked.
    #       2) It then generates 4 more tokens (slots D10..D13). When it finishes:
    #          insert 14 tokens: the tree already has 10, so a leaf for 10..13 is added. All
    #          14 slots now live in the tree, unlocked (EVICTABLE) until someone reuses them.
    #
    #       With page_size > 1 only WHOLE pages can enter the tree; the partial tail page is
    #       freed when the order finishes.
    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        # ELI5 of the table above, left to right on the order's strip of words:
        #   |<- old_handle.cached_len ->|<---- computed by THIS order ---->|
        #   |  borrowed from the tree   | [dup part][ new part ][ tail ]    |
        #   dup part  = someone else put the same words in the tree while we were busy ->
        #               ours are duplicates: free them (see the note on this in NOTES-open-questions)
        #   new part  = enters the tree as a new leaf (the tree now owns these slots)
        #   tail      = leftover that doesn't fill a whole page: freed when the order finishes
        # WHAT: what the order has fully computed so far, as keys (token ids, CPU) and values
        #       (pantry slot numbers from its row of the page table, GPU).
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        # WHAT: returns (how many leading tokens were ALREADY in the tree, a ticket for the path).
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        # WHY late: the old ticket pinned the borrowed prefix; we keep it until the insert is done
        #     so the tree can't evict those notes in between.
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    # WHAT: a self-check, run when the scheduler is idle: every page must be EITHER free OR owned
    #       by the prefix cache. If the counts don't add up, pages leaked or were counted twice.
    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    # ELI5: "Don't hand each returned page back one at a time; collect them in a basket and put
    #       the whole basket back on the shelf at the end." Many small GPU concatenations become one.
    # HOW: inside the with-block, `self._free` is swapped for a function that only collects page
    #      numbers. On exit, the swap is undone and the basket is appended to free_slots in one go.
    # GOTCHA: while the block runs, the collected pages are not yet counted in free_slots.
    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    # ELI5: "Take N pages off the shelf. If the shelf is short, first throw out some unused old
    #       notes from the shared index."
    # WHAT: eviction asks the prefix cache for (shortfall x page_size) token slots; the cache
    #       returns its LEAST RECENTLY USED unlocked leaves (it may return more than asked).
    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    # WHAT: put pages back on the shelf. `indices` are TOKEN slot numbers; taking every
    #       page_size-th element keeps just each page's first slot.
    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    # WHAT: expand page start numbers into the token slot numbers inside each page.
    #       Example, page_size=4: pages [8, 20] -> [8,9,10,11, 20,21,22,23].
    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


# ELI5: write the newly handed-out slot numbers into the right cells of the page table.
#       `allocation_info` lists, per order, (its peg row, first page index, last page index).
#       The loop builds two aligned lists (rows and positions) in pinned CPU memory, copies them
#       to the GPU once, and does ONE fancy-indexed assignment: page_table[rows, positions] = slots.
# WHAT (the example): order on peg 3 needs pages 2..3 at page_size=4 -> positions 8..15 of row 3
#       receive the 8 slot numbers just allocated.
def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
