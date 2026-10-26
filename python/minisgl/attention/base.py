# =====================================================================================
# FILE MAP -- attention/base.py
# =====================================================================================
# WHAT THIS FILE IS: the contract every attention backend follows, plus the HybridBackend that
#                    uses one backend for prefill and another for decode.
#
# ELI5: Attention is the heavy part of a forward pass, and several GPU-kernel libraries do it.
#       They are interchangeable tools with the SAME handle. Each backend promises five things:
#           forward(q,k,v,layer_id,batch)  store this pass's K/V in the cache, run attention
#           prepare_metadata(batch)        build the index/length arrays the kernel needs (once per
#                                          batch, on the CPU, before any layer runs)
#           init_capture_graph(...)        allocate fixed buffers for CUDA graphs
#           prepare_for_capture(batch)     set up metadata for a recording
#           prepare_for_replay(batch)      refresh the fixed buffers before a replay
#
# HYBRID: prefill and decode have different shapes (many query tokens per request vs one), so the
#       best kernel can differ. "fa,fi" means FlashAttention for prefill, FlashInfer for decode.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


# ELI5: the "paperwork" a backend attaches to a Batch (stored in batch.attn_metadata).
# WHAT: the only thing every backend must expose generically: where the LAST token of each
#       request sits in the packed token list (used by the LM head in prefill).
@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    # SHAPE: q [N, Hq_local, D]; k, v [N, Hkv_local * D] (or similar); returns [N, Hq_local, D].
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


# ELI5: two specialists, one per phase; the Batch's phase decides who answers.
# WHAT: forward and prepare_metadata dispatch on batch.is_prefill. CUDA graphs are recorded only
#       for decode, so the capture/replay calls go to the decode backend.
class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
