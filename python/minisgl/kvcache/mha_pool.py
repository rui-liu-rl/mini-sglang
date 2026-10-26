# =====================================================================================
# FILE MAP -- kvcache/mha_pool.py
# =====================================================================================
# WHAT THIS FILE IS: THE PANTRY SHELF. One big pre-allocated GPU tensor that holds the K and V
#                    numbers for every layer, every page, every slot, every head.
#
# ELI5: Picture a warehouse built before opening day. Aisles = floors (layers); each aisle has
#       thousands of numbered slots; each slot holds one word's Label and Gift for that floor.
#       Nothing here knows which order owns which slot; it only stores and fetches by number.
#
#       "MHA" = multi-head attention layout: the shelf keeps `kv_heads` heads per slot (with GQA,
#       fewer than the query heads).
#
# SHAPE:  _kv_buffer = ( 2 , num_layers , num_pages , page_size , local_kv_heads , head_dim )
#                       K/V    floor       page         slot        head           number
# SIZE:   2 x L x pages x page_size x heads x head_dim x bytes-per-number
#         e.g. Qwen3-0.6B, 21,000 slots, bf16: 2 x 28 x 21000 x 8 x 128 x 2 = 2.24 GiB.
#
# WHO CALLS IT: the attention backends (store_kv / k_cache / v_cache), created by
#               kvcache/__init__.py via engine/engine.py.
# WHAT IT CALLS: kernel.store_cache (a custom CUDA copy kernel).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


# GOTCHA: this docstring is copied from the base class; it describes the INTERFACE, not this
#         concrete class (harmless, but misleading).
class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        # WHAT: with tensor parallelism each GPU keeps only ITS KV heads (or 1 replicated head if
        #       there are more GPUs than KV heads).
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # ELI5: build the whole warehouse at once. torch.empty reserves memory without clearing it
        #       (slots are overwritten before they are ever read).
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        # WHAT: two VIEWS of the same memory: the Label half and the Gift half. No copy.
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # WHAT: the shape used when WRITING: pages and offsets merged into one "slot" axis, so
        #       the write kernel can address a slot by a single number.
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    # SHAPE: returns [num_pages, page_size, local_kv_heads, head_dim], a view of the aisle for floor `index`.
    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    # ELI5: "Put these new Labels and Gifts on their numbered slots in this floor's aisle."
    # SHAPE: k, v [N, local_kv_heads, head_dim]; out_loc [N] slot numbers.
    #        Row i of k and v goes to slot out_loc[i] of this layer.
    # NOTE: the custom kernel (kernel/csrc/jit/store.cu) uses one warp per token and copies K and V
    #       rows with wide, coalesced loads.
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
