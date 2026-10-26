# =====================================================================================
# FILE MAP -- kvcache/naive_cache.py
# =====================================================================================
# WHAT THIS FILE IS: the "no sharing" prefix cache: a stand-in that implements the index's
#                    interface but remembers nothing.
#
# ELI5: Imagine the shared index is a notice board of "these beginnings already have notes".
#       The naive board is permanently BLANK. Every lookup says "nobody has anything for you",
#       every insert is ignored. So every order computes its whole prompt from scratch, and when
#       an order ends ALL its slots are returned to the clerk (nothing is kept for others).
#       Useful as a baseline: compare it against "radix" to see what prefix sharing buys you
#       (--cache-type naive).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


# WHAT: a ticket that covers zero tokens and points at no slots.
class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # WHAT: share one empty tensor with every handle (set on the CLASS).
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    # nothing is ever stored, so there is nothing to pin
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    # "no match, ever"
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())

    # WHAT: returns cached_len 0, which makes CacheManager.cache_req free the order's own pages for
    #       the portion that is not stored (see scheduler/cache.py): with this cache, nothing is kept.
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())

    # WHAT: nothing is ever held by the cache, so evicting 0 is a no-op and asking for more is a bug.
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
