# =====================================================================================
# FILE MAP -- kvcache/__init__.py
# =====================================================================================
# WHAT THIS FILE IS: the factory counter for the two KV-cache things: the physical pool, and the
#                    prefix-cache policy (chosen by NAME through a Registry, see utils/registry.py).
#
#   create_kvcache_pool(...)   builds the pantry shelf (always MHAKVCache today;
#                              "TODO: support other variants (e.g. MLA)")
#   create_prefix_cache(...)   builds the shared index: "radix" (tree, shares beginnings) or
#                              "naive" (remembers nothing)
#
# ELI5: the sign at the counter says "Pantry shelves: ask here. Shared index: pick naive or radix."
#       The command line flag --cache-type picks the name; its allowed values come from this
#       registry (SUPPORTED_CACHE_MANAGER.supported_names()).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


# WHAT: a type description: "a function taking a device and returning a prefix cache".
class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


# ELI5: build the pantry shelf from the model's shape: how many floors, KV heads, head size.
# WHAT: num_pages here already includes the +1 dummy page added by the engine.
def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


# WHAT: look up by name and build. An unknown name raises "Unsupported Cache Manager: ...".
def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
