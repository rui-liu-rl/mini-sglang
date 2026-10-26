# =====================================================================================
# FILE MAP -- layers/norm.py
# =====================================================================================
# WHAT THIS FILE IS: the "thermostat" layers: RMSNorm, and RMSNormFused which also does the
#                    residual (shortcut) addition.
#
# ELI5: RMSNorm rescales a vector so its typical size is 1, then multiplies each position by a
#       learned knob `weight`:   y = x / sqrt(mean(x^2) + eps) * weight
#       (Simpler than LayerNorm: no mean subtraction, no bias.)
#
#       RMSNormFused merges "add the shortcut" and "cool it" into ONE GPU kernel. The shortcut
#       (`residual`) travels beside the main value `x` through all layers:
#            first call (residual is None):   residual := x            x := norm(x)
#            later calls:                     residual := residual + x   x := norm(residual)
#       Result: the "x + sublayer(x)" additions of Pre-Norm are never written as separate steps;
#       each one is folded into the next norm. (Worked trace: Part II ch.3 code walkthrough.)
#
# WHO USES IT: models/*.py (decoder layers and final norm), models/utils.py (QK-norm).
# WHAT IT CALLS: FlashInfer's rmsnorm / fused_add_rmsnorm kernels.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        # WHY imported here: FlashInfer is a heavy optional dependency; import it only when a norm
        #     layer is actually created.
        from flashinfer import rmsnorm

        self.eps = eps
        # SHAPE: [size], the learned per-position scale ("weight"). Meta tensor until loaded.
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    # SHAPE: x [..., size] -> same shape. Returns a NEW tensor.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    # WHAT: write the result back into x (no new tensor). Used for QK-norm on the q/k views.
    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    # ELI5: "Take this layer's output x and the shortcut road. Update the road, cool it, and give
    #       me back (cooled value, road)."
    # WHAT: first call (no road yet): the cooled value is norm(x) and the road starts as the raw x.
    #       Later calls: fused_add_rmsnorm does, in one kernel and IN PLACE,
    #           residual += x      then      x = rmsnorm(residual) * weight
    #       so both returned tensors are the (modified) inputs.
    # GOTCHA: the in-place update means the tensors passed in are overwritten. Do not reuse the
    #         old x or residual afterwards.
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
