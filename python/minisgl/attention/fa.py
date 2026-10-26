# =====================================================================================
# FILE MAP -- attention/fa.py   (the "fa" backend = FlashAttention via sgl_kernel)
# =====================================================================================
# WHAT THIS FILE IS: the attention backend built on FlashAttention-3 (FlashAttention-4 on
#                    Blackwell) through SGLang's `sgl_kernel` package. On Hopper GPUs it is the
#                    default for PREFILL ("fa,fi").
#
# ELI5 -- what FlashAttention is: the textbook way to do attention builds the full N x N score
# table, writes it to the far warehouse (HBM), reads it back for softmax, writes again ...
# FlashAttention instead works TILE BY TILE inside the fast on-chip memory, keeping a running
# softmax, and never writes the big table out. Same answer, far fewer trips to the warehouse
# (Part I ch.3's lesson, in a kernel).
#
# HOW THIS BACKEND FEEDS IT. The kernel `flash_attn_with_kvcache` is given:
#       q               the new Questions, packed [N, Hq, D]
#       k_cache/v_cache the layer's whole pantry, [num_pages, page_size, Hkv, D]
#       page_table      per request, which PAGES hold its K/V   [bs, max_pages]
#       cache_seqlens   per request, total context length (device_len)
#       cu_seqlens_q    cumulative new-token counts (the packed-list boundaries)
#       causal=True     query j of a request sees keys up to its own position
#
#     Compared with fi.py: this backend supports real PAGES (page_size > 1): the page table is
#     built by taking every page_size-th entry of the per-token table and dividing by page_size
#     (token slot number -> page number).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


# ELI5: the paperwork for one batch.
@dataclass
class FAMetadata(BaseAttnMetadata):
    # SHAPE: all GPU int32 tensors.
    cu_seqlens_k: torch.Tensor  # [bs + 1] running sum of context lengths
    cu_seqlens_q: torch.Tensor  # [bs + 1] running sum of new-token counts
    cache_seqlens: torch.Tensor  # [bs]     context length of each request
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor  # [bs, max_pages]

    # WHAT: row of the LAST token of each request in the packed list = cu_seqlens_q[i+1] - 1.
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        # WHAT: the 1/sqrt(head_dim) scaling of the scores.
        self.scale = config.head_dim**-0.5
        # WHAT: which FlashAttention generation to call: 4 on Blackwell, else 3 (Hopper).
        self.version = 4 if is_sm100_supported() else 3

    # ELI5: store the new K/V in the pantry, then run the FlashAttention kernel.
    # SHAPE: q [N, Hq_local, D] -> returns [N, Hq_local, D].
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    # ELI5: build the paperwork. Same logic as fi.py for the lengths; the page table differs.
    # WHAT: page_table row for request i = its per-token slot list, sampled every page_size
    #       entries (the first slot of each page) up to the longest context in the batch, then
    #       divided by page_size to get PAGE numbers. All rows get the same length
    #       (max_seqlen_k // page_size); entries past a request's real length are ignored because
    #       cache_seqlens tells the kernel how far to read.
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    # WHAT: allocate the fixed CUDA-graph stage buffers. The page table here is [max_bs, pages],
    #       so max_seq_len is divided by page_size.
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    # ELI5: the recording is filmed with metadata that are VIEWS of the fixed stage buffers.
    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    # ELI5: before playing the recording, copy this step's REAL lengths and page lists onto the
    #       stage. (cu_seqlens_q is untouched: for decode it is always 0,1,2,...,bs.)
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


# WHAT: thin wrapper around sgl_kernel's flash_attn_with_kvcache with SGLang's defaults:
#       window_size (-1, -1) = unlimited context window; softcap 0.0 = off; causal=True.
#       num_splits / pack_gqa = tuning knobs left at their defaults.
# GOTCHA: a clear ImportError message is raised if sgl_kernel is missing.
def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
