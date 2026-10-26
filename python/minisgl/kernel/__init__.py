# =====================================================================================
# FILE MAP -- kernel/__init__.py
# =====================================================================================
# WHAT THIS FILE IS: the shop window of the "kernel" package: the handful of custom GPU/C++ tools
#                    the rest of mini-sglang calls by name.
#
# ELI5: Python is slow at tiny repetitive jobs and PyTorch has no ready-made tool for a few things
#       an inference engine needs. So mini-sglang keeps a small toolbox of hand-written C++ / CUDA
#       tools, each wrapped in a short Python function. This file lists the toolbox:
#
#         indexing           row lookup with a "not my rows -> zeros" rule (the embedding lookup)
#         store_cache        write new K/V rows into the pantry slots chosen by the scheduler
#         fast_compare_key   how many leading numbers do two CPU lists agree on? (radix tree)
#         test_tensor        a tiny self-test of the C++ tensor helper (used by tests/)
#         init_pynccl,       the home-made wrapper around NVIDIA's NCCL library (GPU-to-GPU
#         PyNCCLCommunicator   all-reduce / all-gather), see kernel/pynccl.py
#         fused_moe_kernel_triton, moe_sum_reduce_triton
#                            the Mixture-of-Experts matrix multiply and the final sum (kernel/moe_impl.py)
#
# HOW THEY ARE BUILT: each Python wrapper builds its C++/CUDA source the first time it is used
#       (compile-on-first-call, cached afterwards). The plumbing is in kernel/utils.py. Source code
#       is under kernel/csrc/.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from .index import indexing
from .moe_impl import fused_moe_kernel_triton, moe_sum_reduce_triton
from .pynccl import PyNCCLCommunicator, init_pynccl
from .radix import fast_compare_key
from .store import store_cache
from .tensor import test_tensor

__all__ = [
    "indexing",
    "fast_compare_key",
    "store_cache",
    "test_tensor",
    "init_pynccl",
    "PyNCCLCommunicator",
    "fused_moe_kernel_triton",
    "moe_sum_reduce_triton",
]
