# FILE MAP -- attention/utils.py
# WHAT: the fixed "stage" buffers shared by the attention backends for CUDA-graph recording.
# ELI5: a recorded CUDA graph remembers memory ADDRESSES, so everything its attention kernel
#       reads must live in buffers that never move. BaseCaptureData allocates them once, sized
#       for the LARGEST recorded batch (max_bs) and the longest sequence (max_seq_len):
#           seq_lens      [max_bs]       how long each request's context is (starts as all 1s)
#           positions     [max_bs]
#           cu_seqlens_k  [max_bs + 1]   cumulative key lengths: 0,1,2,...  (a placeholder)
#           cu_seqlens_q  [max_bs + 1]   cumulative query lengths: 0,1,2,... (decode: 1 per request)
#           page_table    [max_bs, max_seq_len]  the KV-slot (or page) list per request
#       Before each replay the backend copies the REAL values for this step into these buffers
#       (prepare_for_replay); the recording then reads them.
# Tag legend: see zero-to-sglang/eli5/README.md
from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    # WHAT: build the buffers; subclasses can add extra fields through **kwargs.
    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
