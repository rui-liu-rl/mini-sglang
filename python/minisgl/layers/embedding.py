# =====================================================================================
# FILE MAP -- layers/embedding.py
# =====================================================================================
# WHAT THIS FILE IS: the two ends of the model: the INPUT lookup table (token id -> vector) and
#                    the OUTPUT scoreboard (vector -> a score per vocabulary word).
#
# ELI5: The vocabulary can be huge (151,936 words for Qwen3). With several GPUs the table is cut
#       by ROWS: GPU r owns vocabulary rows [start_r, end_r).
#
#       VocabParallelEmbedding (input side):
#         each GPU looks up the ids that fall in ITS row range; ids outside its range give zeros
#         on that GPU. Add all GPUs' results with all_reduce: exactly one GPU contributed a
#         non-zero vector per token, so the sum is the right vector.
#
#       ParallelLMHead (output side):
#         each GPU multiplies by its rows, producing scores for ITS part of the vocabulary;
#         all_gather glues the parts into the full score list.
#         It also keeps ONLY THE LAST position of each request during prefill (see forward).
#
# TIED EMBEDDINGS: many small models reuse the SAME table for input and output ("tie_word_embeddings").
#       Then the LM head owns no weights of its own and borrows the embedding's.
#
# WHO USES IT: models/*.py. WHAT IT CALLS: kernel.indexing (a custom CUDA gather), F.linear,
#               DistributedCommunicator.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        # WHAT: rows per GPU, rounded UP so the last GPU may own fewer (or padding is implied).
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        # WHAT: (first row owned, number of rows really owned) for the lookup kernel.
        self.vocab_range = (start_idx, finish_idx - start_idx)
        # SHAPE: [rows per GPU, embedding_dim]. Example Qwen3-0.6B, tp=1: [151936, 1024].
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    # SHAPE: x is [N] token ids -> returns [N, embedding_dim].
    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        # WHAT: a custom GPU gather: out[i] = weight[x[i] - start] if x[i] is in my range, else 0.
        #       With one GPU (vocab_range=None) it is an ordinary row lookup.
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    # WHAT: if tied, this head has no weights of its own. The checkpoint may still contain an
    #       "lm_head.weight" (and bias); those are removed from the heap so the "unexpected keys"
    #       check doesn't fail. If not tied, behave like any layer.
    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    # ELI5: "From each order's final vector, produce a score for every vocabulary word."
    # WHAT (in order):
    #   1. PREFILL ONLY: keep just the LAST hidden vector of each request (get_last_indices gives
    #      their row numbers in the packed list). Decode already has one row per request.
    #      This is a huge saving: scoring the whole vocabulary for EVERY prompt token would be
    #      N times more work and memory (Part II ch.3).
    #   2. logits = x @ W^T  (using the embedding's weights if tied)
    #   3. one GPU: done. Several GPUs: each has scores for its vocabulary slice, so all_gather
    #      concatenates them (along the first dimension), then the slices are rearranged to
    #      [batch, full_vocab] and any padding rows beyond the real vocabulary are cut off.
    # SHAPE: x [N, hidden] -> (after step 1) [bs, hidden] -> logits [bs, vocab_size].
    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        # SHAPE: all_gather stacks the tp slices along dim 0 -> [tp * bs, vocab_slice].
        output_tensor = self._comm.all_gather(logits)

        # WHAT: with a single request the slices can be laid side by side by a simple reshape.
        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # WHAT: general case: [tp, bs, slice] -> permute to [bs, tp, slice] -> merge the last two
        #       dims into [bs, tp * slice], then drop padding columns past num_embeddings.
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
