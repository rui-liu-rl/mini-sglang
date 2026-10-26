# =====================================================================================
# FILE MAP -- kernel/index.py
# =====================================================================================
# WHAT THIS FILE IS: the Python door to the "indexing" GPU tool: look up rows of a table by id.
#
# ELI5: This is the embedding lookup. Given a big table (one row of numbers per vocabulary word)
#       and a list of word ids, hand back the rows for those ids.
#
#       torch can do this with weights[indices], but with TENSOR PARALLELISM each GPU owns only some
#       rows (layers/embedding.py). This tool adds one rule: if an id is OUTSIDE my range, write a
#       row of ZEROS instead of failing. Then an all_reduce (sum) across GPUs yields the right row,
#       because exactly one GPU had a real row for each id.
#
# THE C++ SIDE: kernel/csrc/jit/index.cu. This file only chooses a good variant and launches it.
#
# WHO CALLS IT: layers/embedding.py (VocabParallelEmbedding.forward).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# WHAT: 128 threads per block, at most 1 block resident per multiprocessor, no PDL.
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# ELI5: build (once) the variant of the tool for rows that are `element_size` BYTES wide, and split
#       each row's copy into `num_splits` pieces. functools.cache = "remember the answer for
#       these exact arguments", so each distinct width is compiled only once per process.
# WHAT: "IndexKernel<element_size, num_splits, threads, occupancy, pdl>::run" is exported to Python
#       under the name `launch`.
@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


# ELI5: "For each id in `indices`, copy that row of `weights` into the answer."
# SHAPE: weights [rows_I_own, width]; indices [N] int; output [N, width] (allocated if not given).
#        vocab_range = (first row I own, how many rows I own), or None for the ordinary case.
# WHY the number of splits depends on the row width: a wide row is cut into 2 or 4 pieces that
#     different warps (groups of 32 GPU threads) copy in parallel; rows whose byte width is a
#     multiple of 2048 get 4 pieces, of 1024 get 2, otherwise 1.
#     (Example: Qwen3-0.6B bf16: width 1024 x 2 bytes = 2048 -> 4.)
def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
