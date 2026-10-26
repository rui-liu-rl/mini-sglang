# =====================================================================================
# FILE MAP -- tests/kernel/test_tensor.py   (a script; needs at least 2 GPUs)
# =====================================================================================
# WHAT THIS FILE TESTS: the C++ tensor-checking toolkit (csrc/include/minisgl/tensor.h) through its
#                       self-test (csrc/src/tensor.cpp). It passes if no exception is raised.
#
# ELI5: Hand the checker two tensors that SHOULD pass, and see that it does:
#       x: a [12, 1024] int32 CPU view of a wider [12, 2048] table (so its row stride is 2048,
#          not 1024: the checker is told "any row stride" but "last dimension contiguous")
#       y: a [12, 1024] int64 tensor on GPU number 1 (the C++ test insists on CUDA device 1,
#          hence the two-GPU requirement)
#       and N = 12 is a multiple of 4, as the last check demands.
#       Change any of these (dtype, shape, device) and the C++ side should raise a clear
#       "Tensor match failed ... Root cause: ..." message.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from minisgl.kernel import test_tensor
from minisgl.utils import call_if_main
import torch


@call_if_main()
def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    test_tensor(x, y)
