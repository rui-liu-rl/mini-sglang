# =====================================================================================
# FILE MAP -- utils/arch.py
# =====================================================================================
# WHAT THIS FILE IS: "what generation of GPU stove do I have?"
#
# ELI5: NVIDIA GPUs have a "compute capability" like 8.0, 9.0 or 10.0 (SM80, SM90, SM100). Newer
#       numbers = newer tricks available in the hardware:
#           SM80  A100 (Ampere)        SM90  H100/H200 (Hopper)        SM100  B200 (Blackwell)
#       The engine uses this to pick the best attention kernels (engine.py _adjust_config).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from typing import Tuple


# WHAT: ask CUDA for (major, minor) of GPU 0, once (functools.cache remembers the answer).
#       None if there is no CUDA GPU.
# GOTCHA: the name says "version", but the value returned is the GPU's compute CAPABILITY.
@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


# WHAT: is this GPU at least (major, minor)? Tuples compare element by element, so (9, 0) >= (8, 6).
def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


# WHAT: Hopper or newer (H100, H200, B200...).
def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


# WHAT: Blackwell or newer (B200...).
def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
