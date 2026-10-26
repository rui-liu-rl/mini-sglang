# =====================================================================================
# FILE MAP -- models/qwen3.py
# =====================================================================================
# WHAT THIS FILE IS: the Qwen3 recipe: how the reusable rooms (RopeAttn, GatedMLP) and the norms
#                    are stacked into floors, and how floors are stacked into the whole model.
#
# ELI5: three nested classes, from the inside out:
#   Qwen3DecoderLayer   ONE FLOOR       norm -> meeting (attention) -> norm -> private office (MLP)
#   Qwen3Model          THE BUILDING    embed -> floor x N -> final norm
#   Qwen3ForCausalLM    WITH A ROOF     building -> LM head (scores for every word)
#
# NAMES MATTER: attribute names here (model, layers, self_attn, mlp, input_layernorm,
#       post_attention_layernorm, embed_tokens, norm, lm_head) are chosen to MATCH the Hugging Face
#       checkpoint keys, so BaseOP.state_dict()/load_state_dict() line up with the weight files
#       without any renaming (the loader only merges q/k/v and gate/up).
#
# Qwen3's own specialities: QK-norm (has_qk_norm=True), no attention bias, tied input/output
#       embeddings for small sizes.
# Compare: llama.py, mistral.py (no QK norm), qwen2.py (attention bias), qwen3_moe.py (MoE MLP).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


# ELI5: ONE FLOOR.
class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        # WHAT: the two thermostats of a Pre-Norm floor (before attention, before the MLP).
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # WHAT: private (leading "_") so it is not treated as a weight; used by the profiler tag.
        self._layer_id = layer_id

    # ELI5: the floor's five lines. The shortcut addition is hidden inside the NEXT norm call
    #       (RMSNormFused), so you won't see an explicit `x + attention(x)`.
    # SHAPE: x [N, hidden]; residual None on the very first layer, then [N, hidden].
    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # cool x (and fold the previous floor's leftovers into the shortcut road)
        x, residual = self.input_layernorm.forward(x, residual)
        # the meeting
        x = self.self_attn.forward(x)
        # fold the attention output into the road, cool again
        x, residual = self.post_attention_layernorm.forward(x, residual)
        # the private office
        x = self.mlp.forward(x)
        return x, residual


# ELI5: THE BUILDING.
class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        # SHAPE: [vocab / tp, hidden] lookup table.
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # WHAT: N identical-shaped floors, named "0", "1", ... in the checkpoint.
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    # SHAPE: input_ids [N] -> hidden states [N, hidden].
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # WHAT: last thermostat; [0] picks the cooled value (the road is no longer needed).
        return self.norm.forward(x, residual)[0]


# ELI5: THE BUILDING WITH A ROOF: adds the scoreboard.
class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        # WHAT: if the checkpoint ties input and output embeddings, the LM head borrows the
        #       embedding table instead of owning a second copy.
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    # ELI5: takes NO arguments; reads the tray (token ids) from the Kitchen whiteboard.
    # SHAPE: returns logits [batch_size, vocab_size]: one row per request (last position).
    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
