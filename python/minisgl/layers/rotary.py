# =====================================================================================
# FILE MAP -- layers/rotary.py
# =====================================================================================
# WHAT THIS FILE IS: RoPE (rotary position embedding): the layer that tells attention WHERE each
#                    word sits, by rotating its Question and Label arrows.
#
# ELI5 (recap of Part I ch.1): imagine every pair of numbers in a Question/Label vector as the
#       tip of a little arrow. For the word at seat number p, rotate the arrow in pair i by the
#       angle  p * inv_freq[i].  Low pairs spin fast (fine seat differences), high pairs spin slowly
#       (coarse differences). After rotation, how well a Question matches a Label depends only
#       on the DIFFERENCE of their seat numbers.
#
# HOW IT IS DONE EFFICIENTLY: the cos and sin of every (position, pair) are computed ONCE at
#       startup into a table, _cos_sin_cache, and FlashInfer's kernel rotates q and k IN PLACE
#       for all tokens in the batch by looking up their positions in the table.
#
#       SHAPE of the table: [max_position, rotary_dim] =  [ cos(p*f_0..p*f_{d/2-1}) | sin(...) ]
#       Example Qwen3-0.6B: 40,960 positions x 128 values (float32), about 21 MB, built once
#       (functools.cache on get_rope).
#
# LONG-CONTEXT VARIANTS (rope_scaling): to run a model on longer inputs than it was trained on,
#       slow some of the "clocks" down:
#           "llama3" : keep the fast clocks, divide the slowest ones by `factor`, blend in between.
#           "yarn"   : similar idea (interpolate low-frequency dims, keep high ones) with a ramp.
#           None/"default": plain RoPE.
#
# WHO USES IT: layers/attention.py (get_rope). WHAT IT CALLS: flashinfer RoPE kernel.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


# ELI5: owns the cos/sin table and applies the rotation. It has no learned weights (StateLessOP),
#       the table is a derived buffer (notice `_cos_sin_cache` starts with "_", so it is never saved
#       or loaded as a weight).
class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # GOTCHA: partial rotary (rotating only a part of each head) is NOT supported here.
        assert rotary_dim == head_size
        # WHAT: the "clock speed" of each pair: inv_freq[i] = 1 / base^(2i / rotary_dim).
        # SHAPE: [rotary_dim / 2], float32.
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # WHAT: optional length-extension tweak of the clock speeds (llama3 / yarn).
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        # WHAT: all seat numbers 0..max-1, then the angle table = outer product seat x speed.
        # SHAPE: t [max_pos]; freqs [max_pos, rotary_dim / 2].
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        # SHAPE: [max_pos, rotary_dim]: cos in the first half of the last dim, sin in the second.
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        # WHAT: the FlashInfer kernel supports only these head sizes.
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    # ELI5: "Rotate every token's Question and Label by that token's seat number."
    # SHAPE: positions [N] int; query [N, num_q_heads * head_dim] or [N, heads, head_dim];
    #        key likewise. Modified IN PLACE and also returned.
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


# ELI5: choose the right flavour of RoPE from the model's config (rope_scaling dict).
def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        # ELI5: Llama-3.1 style. Think of each clock by its WAVELENGTH (how many seats for a full
        #       turn). Wavelengths shorter than original_max_position / high_freq_factor (fast
        #       clocks) stay as they are. Wavelengths longer than original_max_position /
        #       low_freq_factor (the slowest clocks) are slowed by `factor`. In between, blend
        #       smoothly (the `smooth` ramp, clamped to 0..1).
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        # ELI5: YaRN style. Find which clock indices (low..high) complete a number of turns within
        #       the original training length; interpolate (slow down by `factor`) the clocks outside
        #       that band and keep the band's fast clocks, with a linear ramp (0 to 1) between.
        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


# WHAT: the GPU to build the cos/sin table on. Set by Engine via set_rope_device().
_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


# ELI5: "Give me the RoPE table for this configuration, building it only the first time."
# WHY functools.cache: every layer asks for the same table; they all share one copy. The arguments
#     must be hashable, which is why `rope_scaling` arrives as a TUPLE of (key, value) pairs
#     (attention.py converts the dict) and is turned back into a dict here.
# WHY the meta-device branch: the model skeleton is built under `with torch.device("meta")` (no
#     memory). But a RoPE table is REAL data, computed now, so it must be created on a real GPU:
#     `t = torch.tensor([])` shows what the current default device is; if it's "meta" we switch to
#     _ROPE_DEVICE for the build.
@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
