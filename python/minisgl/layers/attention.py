# =====================================================================================
# FILE MAP -- layers/attention.py
# =====================================================================================
# WHAT THIS FILE IS: the MEETING ROOM layer: takes the packed QKV vectors, tidies and rotates the
#                    Questions and Labels, and hands everything to the attention backend, which
#                    stores the new K/V in the pantry and runs the real attention kernel.
#
# ELI5: Think of this as the meeting's MODERATOR. It does not do the heavy comparing itself (the
#       backend's GPU kernel does). It prepares the cards:
#            split QKV -> (optional) tidy Q and K -> rotate Q and K by seat number -> backend.forward
#
# TENSOR-PARALLEL NOTE: each GPU owns SOME of the heads. num_qo_heads / num_kv_heads here are the
#       LOCAL counts (already divided by tp_size).
#
# WHO USES IT: models/utils.py (RopeAttn). WHAT IT CALLS: layers/rotary.py, the attention backend
#               in the global Context (attention/*), optional RMSNorms for Q/K.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


# WHAT: StateLessOP = this layer owns no weights itself (the projections and norms live in the
#       sibling layers; the q/k norm objects are kept here but belong to RopeAttn's state).
class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        # WHY the assert: GQA needs every KV head to serve a whole number of Question heads.
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        # WHAT: this GPU's heads. KV heads may be replicated if there are fewer than GPUs.
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        # WHAT: widths of the Q part and of each of the K and V parts in the packed qkv row.
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        # WHAT: the shared cos/sin table (built once for all layers).
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    # ELI5: "Prepare the cards, then let the backend run the meeting."
    # SHAPE: qkv [N, qo_dim + 2 * kv_dim]  ->  returns [N, qo_dim]   (N = tokens in the batch)
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        # 1. cut the packed row into Question | Label | Gift (views, no copy)
        # SHAPE: q [N, qo_dim], k [N, kv_dim], v [N, kv_dim]
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # 2. (Qwen3 only) normalise each head's Question and Label vector in place
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 3. rotate Q and K in place by each token's position (V is not rotated)
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        # 4. the backend (a) stores this pass's k, v into the KV cache at batch.out_loc, then
        #    (b) runs attention of q against the cached K/V of each request.
        #    GOTCHA: K is stored AFTER the rotation above, so the cache holds rotated Labels.
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
