# =====================================================================================
# FILE MAP -- layers/activation.py
# =====================================================================================
# WHAT THIS FILE IS: the "gate" step of the gated MLP, as thin wrappers over FlashInfer kernels.
#
# ELI5: The MLP computes two long vectors side by side, `gate` and `up` (they arrive glued
#       together as one vector of length 2*I). This step does
#               out = activation(gate) * up        ( "and_mul" = ... AND multiply by up )
#       silu(x) = x * sigmoid(x) is a smooth version of ReLU; gelu is another smooth variant.
#       The gate decides, number by number, how much of `up` gets through.
#
# SHAPE: x [N, 2 * I] (first half = gate, second half = up) -> out [N, I].
# WHY a fused kernel: activation and multiply in one pass over memory instead of three.
#
# WHO USES IT: models/utils.py (GatedMLP), moe/fused.py (expert MLPs).
# WHAT IT CALLS: flashinfer.silu_and_mul / gelu_and_mul.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# WHAT: out=None means "allocate the result"; passing a tensor writes the result into it.
def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
