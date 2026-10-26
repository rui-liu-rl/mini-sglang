# =====================================================================================
# FILE MAP -- kvcache/base.py
# =====================================================================================
# WHAT THIS FILE IS: the job descriptions ("interfaces") of the two big jobs around the KV cache:
#                      1. BaseKVCachePool : the PANTRY SHELF itself (store and fetch K/V numbers)
#                      2. BasePrefixCache : the SHARED INDEX of which words already have notes
#                    plus the small record types they exchange.
#
# ELI5: Two different questions, two different objects:
#       "Where do the actual numbers live and how do I write/read them?"  -> BaseKVCachePool
#       "Has anyone already written notes for the beginning of this order?" -> BasePrefixCache
#       The pool knows NUMBERS but not WHO they belong to. The prefix cache knows WHO/WHAT (which
#       words map to which slots) but never touches the numbers.
#
# VOCABULARY
#   handle (BaseCacheHandle) : a TICKET describing a stored path ("the first 120 words of this
#                              order are stored, here are their slots"). cached_len = path length.
#   lock / unlock            : pin the notes behind a ticket so eviction cannot remove them
#                              (reference counting).
#   evictable_size           : stored tokens nobody is using; may be thrown out when space is needed
#   protected_size           : stored tokens some running order is using; untouchable
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


# ELI5: "the pantry shelf". Three actions plus three facts about itself.
class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    # SHAPE: returns the layer's Label shelf [num_pages, page_size, kv_heads, head_dim]  (a view)
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    # SHAPE: same for the Gift shelf.
    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    # WHAT: write k[i], v[i] into slot out_loc[i] of layer `layer_id`, for every new token i.
    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


# ELI5: a TICKET for a stored path. frozen = cannot be edited; hashable.
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    # WHAT: how many leading tokens this ticket covers.
    cached_len: int

    # SHAPE: returns the slot numbers of those cached_len tokens, in order [cached_len], int32.
    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


# WHAT: the two counters the clerk (scheduler/cache.py) uses to decide what is available.
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


# WHAT: the answer of insert_prefix.
class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


# WHAT: the answer of match_prefix. (The "cuda_handle" naming leaves room for a CPU-memory tier
#       later: "TODO: support HiCache"; see Part III chapter 3.)
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


# ELI5: "the shared index". Six actions; the radix tree and the naive no-op both implement them.
class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
