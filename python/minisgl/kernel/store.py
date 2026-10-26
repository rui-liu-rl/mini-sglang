# =====================================================================================
# FILE MAP -- kernel/store.py
# =====================================================================================
# WHAT THIS FILE IS: the Python door to the "store_cache" GPU tool: put this step's new K and V
#                    rows onto their pantry slots.
#
# ELI5: After the model computes the Labels (K) and Gifts (V) for the new words of every order,
#       they must be written to the pantry slots the scheduler chose for them (`out_loc`). In
#       plain torch that is  k_cache[indices] = k;  v_cache[indices] = v  (two scatter ops).
#       This tool does both copies in ONE launch, with each GPU warp (32 threads) handling one
#       word's row.
#
# THE C++ SIDE: kernel/csrc/jit/store.cu.
#
# WHO CALLS IT: kvcache/mha_pool.py (MHAKVCache.store_kv), once per layer per forward pass.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# WHAT: the same dial settings as index.py (the name is reused, the config is not shared state).
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# WHAT: build (once per distinct row width in bytes) the StoreKernel variant and export its
#       `run` function to Python as `launch`.
@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


# ELI5: "Row i of k and v goes to slot indices[i] of this layer's K and V shelves."
# SHAPE: k_cache, v_cache  one layer's shelf viewed as [total_slots, kv_heads, head_dim]
#        indices           [N] slot numbers
#        k, v              [N, kv_heads, head_dim] (or [N, kv_heads * head_dim]); the new rows
# WHAT: flatten each slot to ONE row of kv_heads * head_dim numbers; element_size is that row's
#       width in bytes (example: 8 heads x 128 x 2 bytes = 2048). The kernel needs only that.
def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
