# =====================================================================================
# FILE MAP -- models/config.py
# =====================================================================================
# WHAT THIS FILE IS: mini-sglang's own simplified description of a model's SHAPE, distilled from
#                    the (messy, model-specific) Hugging Face config.json.
#
# ELI5: Every model family writes its config.json a little differently ("num_key_value_heads"
#       here, missing there; rope_theta in one place, inside another dict elsewhere). The rest of
#       the engine would hate to deal with that. So this file reads the HF config ONCE and fills
#       in a clean ModelConfig with fixed field names. Everything else uses ModelConfig.
#
# Example (Qwen3-0.6B): num_layers 28, num_qo_heads 16, num_kv_heads 8, head_dim 128,
#       hidden_size 1024, vocab_size 151936, intermediate_size 3072, hidden_act "silu",
#       tie_word_embeddings True, rope base 1,000,000.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


# WHAT: everything needed to build the RoPE table.
@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    # WHAT: how many dims of each head get rotated (here always the whole head).
    rotary_dim: int
    # WHAT: the longest position the table covers = the model's maximum context length.
    max_position: int
    # WHAT: "rope_theta": the base of the geometric clock speeds (1e4 classically, 1e6 for Qwen3).
    base: float
    # WHAT: the rope_scaling dict from the config (llama3 / yarn / ...), or None.
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    # WHAT: "qo" = query/output heads (attention's Question heads; the output has the same count).
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    # WHAT: width of the MLP's inner layer (per gate/up; the merged matrix is twice this).
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    # WHAT: "silu" or "gelu": which gate function the MLP uses.
    hidden_act: str
    tie_word_embeddings: bool
    # WHAT: MoE models only (0 for dense models).
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]

    # WHAT: crude but effective: any model_type containing "moe" (qwen3_moe) is a MoE model.
    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    # ELI5: "Read the messy HF config and fill in the clean one."
    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # WHAT: multimodal wrappers (e.g. Mistral3) nest the language model's settings in a
        #       `text_config`. Use that one, but copy over a few fields that the outer config owns
        #       (architectures, rope_theta, rope_scaling) if the inner one lacks them.
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        # WHAT: no `num_key_value_heads` means classic multi-head attention (every head has its
        #       own K/V), so kv heads = attention heads.
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        # WHAT: head_dim is explicit for some models (Qwen3: 128 even though hidden/heads would be
        #       64 for the 0.6B); otherwise derive it as hidden_size / num_heads.
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
