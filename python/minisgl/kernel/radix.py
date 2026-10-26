# =====================================================================================
# FILE MAP -- kernel/radix.py
# =====================================================================================
# WHAT THIS FILE IS: the Python door to one tiny C++ helper, `fast_compare_key`, used by the radix
#                    prefix-cache tree.
#
# ELI5: The tree must keep asking "how many words at the start of these two lists are the same?"
#       Comparing thousands of ids in a Python loop would be slow; this does it in C++.
#
# THE C++ SIDE: kernel/csrc/src/radix.cpp (compiled ahead of time, once).
#
# WHO CALLS IT: kvcache/radix_cache.py (RadixTreeNode.get_match_len).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


# WHAT: compile (the first time) and load the module; cached afterwards.
@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


# ELI5: "Where do these two lists first disagree?"
# SHAPE: x, y are 1-D, contiguous CPU integer tensors of the SAME dtype (int32 or int64); they may
#        have different lengths. Anything else makes the C++ side raise.
# RETURNS: the number of leading entries that are equal, i.e. the index of the first difference,
#          or the length of the shorter list if one is a prefix of the other.
def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
