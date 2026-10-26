# =====================================================================================
# FILE MAP -- models/utils.py
# =====================================================================================
# WHAT THIS FILE IS: the two reusable "rooms" every decoder layer is built from:
#                    RopeAttn (the meeting room) and GatedMLP (the private office), plus MoEMLP
#                    (the many-offices version).
#
# ELI5: Llama, Qwen2, Qwen3 and Mistral differ only in small settings, so their files all reuse
#       these. A decoder layer = { self_attn = RopeAttn, mlp = GatedMLP, two norms }.
#
#   RopeAttn  : x -> qkv_proj (one merged matmul) -> AttentionLayer (RoPE + backend) -> o_proj
#   GatedMLP  : x -> gate_up_proj (one merged matmul) -> silu(gate) * up -> down_proj
#   MoEMLP    : x -> router scores -> MoELayer (each word goes to its top-k experts) -> combined
#
# WHO USES IT: models/qwen3.py, qwen2.py, llama.py, mistral.py, qwen3_moe.py.
# WHAT IT CALLS: layers/ (linear, norm, attention, activation, moe).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


# ELI5: the private office with a GATE. Two long vectors are computed from each word, `gate` and
#       `up`; the gate (after a smooth squashing) decides how much of `up` passes; then a final
#       matrix returns to the hidden size.
class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        # SHAPE: weight [2 * intermediate_size / tp, hidden_size]: the gate and up matrices stacked.
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        # WHAT: pick the gate function by the model's hidden_act name.
        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        # SHAPE: weight [hidden_size, intermediate_size / tp]. Row-parallel: ends with an all_reduce.
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    # SHAPE: x [N, hidden] -> gate_up [N, 2I/tp] -> y [N, I/tp] -> output [N, hidden].
    # WHY `del`: drop the reference to the big temporary tensors as soon as they are consumed, so
    #     the GPU memory can be reused for the next step.
    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


# ELI5: the "many offices" version. A small matrix `gate` (the router) scores every word against
#       every expert; MoELayer sends each word to its top-k experts and combines their answers.
class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        # WHAT: the router is tiny, so it is REPLICATED on every GPU (no communication needed).
        # SHAPE: weight [num_experts, hidden_size].
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    # SHAPE: hidden_states [num_tokens, hidden] -> router_logits [num_tokens, num_experts]
    #        -> output [num_tokens, hidden].
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


# ELI5: the meeting room. Three steps, each a layer you've met:
#       1. qkv_proj  : one merged matrix multiply -> Questions, Labels, Gifts side by side
#       2. attn      : AttentionLayer (optional QK-norm, RoPE, KV-cache store, attention kernel)
#       3. o_proj    : mix the heads and return to hidden size (row-parallel + all_reduce)
class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        # WHAT: Qwen2 has a bias on the QKV projection; the others do not.
        has_attn_bias: bool = False,
        # WHAT: Qwen3 normalises each head's Q and K vector before RoPE.
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        # SHAPE: [(Hq + 2*Hkv) * head_dim / tp, hidden_size]
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            # WHAT: the norms act on ONE head's vector (size head_dim), shared across heads.
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        # SHAPE: [hidden_size, Hq * head_dim / tp]. Row-parallel: ends with an all_reduce.
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    # SHAPE: x [N, hidden] -> qkv [N, (Hq+2Hkv)D/tp] -> o [N, Hq*D/tp] -> output [N, hidden].
    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
