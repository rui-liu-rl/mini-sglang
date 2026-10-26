# =====================================================================================
# FILE MAP -- kernel/tensor.py
# =====================================================================================
# WHAT THIS FILE IS: a self-test hook for the C++ tensor-checking helpers (csrc/include/minisgl/
#                    tensor.h). It is not used by the engine; tests/kernel/test_tensor.py calls it.
#
# ELI5: The C++ side has a small toolkit for the question "is this tensor really what I think it
#       is?" (right shape, right type, right device...). `test_tensor` pushes two tensors through
#       that toolkit so the tests can check that the answers (and the error messages) are right.
#
# THE C++ SIDE: kernel/csrc/src/tensor.cpp.
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


# WHAT: compile (first time) and load the test module; cached afterwards.
@functools.cache
def _load_test_tensor_module() -> Module:
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


# WHAT: runs the C++ checks on x and y. The C++ function `test` returns nothing (void), so the
#       `-> int` annotation below is not accurate; what matters is whether it RAISES.
#       It expects x: [N, 1024], inner stride 1, int32/float32, on the CPU;
#                  y: [N, 1024], contiguous, int32/int64, on a CUDA device;  and N % 4 == 0.
def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_test_tensor_module().test(x, y)
