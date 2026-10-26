# =====================================================================================
# FILE MAP -- scheduler/table.py
# =====================================================================================
# WHAT THIS FILE IS: the "peg board" that gives every running request its own numbered row.
#
# ELI5: The head waiter has a wall with N numbered pegs. When an order is accepted it gets a
#       free peg (allocate); when the order leaves, the peg is handed back (free). Behind
#       each peg are two long strips of paper, one box per word position:
#          page_table strip : "which pantry SLOT holds the notes for word #p?"
#          token_pool strip : "which token ID is word #p?"
#
# WHO USES IT: scheduler/scheduler.py (creates it), scheduler/prefill.py (allocates rows),
#              scheduler/cache.py (writes page numbers into rows).
# WHAT IT CALLS: nothing but torch.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import torch


class TableManager:
    # WHAT: max_running_reqs = how many pegs there are = the most requests that can be in
    #       flight at once. page_table is created by the Engine (it lives on the GPU) and
    #       shared here.
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        # WHAT: a plain Python list of free row numbers: [0, 1, 2, ..., max_running_reqs-1].
        self._free_slots = list(range(max_running_reqs))
        # SHAPE: (max_running_reqs + 1, aligned_max_seq_len), int32. The extra last row belongs
        #        to the dummy request used for CUDA-graph padding; it is never handed out here.
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # ELI5: the token_pool strips start out all zeros (a harmless token id), so even the
        #       dummy request, which reads from here, always sees valid numbers.
        # SHAPE: identical to page_table, but holds token ids instead of slot numbers.
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    # WHAT: how many pegs are still free.
    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    # WHAT: take a peg. list.pop() takes from the END of the list.
    # GOTCHA: no check for "empty": callers must test available_size first (prefill.py does).
    def allocate(self) -> int:
        return self._free_slots.pop()

    # WHAT: hand a peg back. Its strips are NOT cleared; the next owner overwrites what it uses.
    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
