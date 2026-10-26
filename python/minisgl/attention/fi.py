# =====================================================================================
# FILE MAP -- attention/fi.py   (the "fi" backend = FlashInfer)
# =====================================================================================
# WHAT THIS FILE IS: the attention backend built on the FlashInfer library. Used for DECODE on
#                    Hopper-class GPUs ("fa,fi") and for everything on older GPUs ("fi").
#
# ELI5 -- WHAT ATTENTION HAS TO DO HERE. Each request has its notes (K/V) scattered over pantry
# slots all over the shelf (the paged KV cache). The kernel must, for every request, gather its
# slots, compare the request's new Questions with those Labels, and mix the Gifts. FlashInfer is
# told WHERE everything is by a few small index arrays (the "metadata"), built on the CPU for each
# batch by prepare_metadata():
#
#     cu_seqlens_q  cumulative QUERY lengths   [0, q0, q0+q1, ...]  (how many new tokens per request)
#     cu_seqlens_k  cumulative KEY   lengths   [0, k0, k0+k1, ...]  (k = total context incl. cached)
#     indices       the pantry slot of EVERY key token, all requests concatenated ("ragged")
#     last_page_len with page_size 1 each "page" is one token, so this is all ones
#
#     Example, 2 requests: A (context 5, 5 new tokens), B (context 6, 4 new tokens, 2 cached):
#         cu_seqlens_q = [0, 5, 9]      cu_seqlens_k = [0, 5, 11]
#         indices      = [A's 5 slots | B's 6 slots]         (11 numbers)
#
# TWO-STEP PROTOCOL of FlashInfer: wrapper.plan(...) first precomputes how to split the work
# across the GPU's multiprocessors (scheduling metadata; some work happens on the CPU), then
# wrapper.run(q, kv_cache) does the math. Planning is done lazily, once per batch, by
# _initialize_metadata_once (the first layer's call), and reused by all the other layers.
#
# PAGE SIZE: this backend always tells FlashInfer "page = ONE token" (the metadata's page_size is
# the constant 1, and the assert below only checks that constant). It works with a pool of any
# page_size because forward() re-views the pool as one-token pages (_flatten_cache) and
# `indices` holds PER-TOKEN slot numbers straight from the page table. So FlashInfer's own
# multi-token paging feature is never used here; fa.py and trtllm.py do use real pages.
#
# GROUPED-QUERY NOTE: for decode, `use_tensor_cores` is switched on when 4 or more query heads
# share a KV head (num_qo_heads // num_kv_heads >= 4); with that much sharing the work per KV head
# looks like a small matrix multiply and tensor cores pay off (env override in env.py).
#
# WHO CALLS IT: layers/attention.py via ctx.attn_backend. Created by attention/__init__.py.
# WHAT IT CALLS: flashinfer's BatchPrefill/BatchDecode wrappers, the KV cache pool.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


# WHAT: smallest power of two >= n (1 for n <= 1). Used to grow a cached all-ones array in steps.
def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


# WHAT: the CUDA-graph stage buffers (attention/utils.py) with two friendlier names:
#       `one_tensor` (a vector of ones, reused as last_page_len) and `indices` (the flat slot list).
@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


# ELI5: the paperwork for one batch (stored in batch.attn_metadata).
@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    # WHAT: which arrays live on the CPU vs the GPU matters: FlashInfer's planner reads the CPU ones.
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    # WHAT: has plan() been run for this batch yet? (done by the first layer)
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    # ELI5: "Which row of the packed token list is the LAST token of each request?"
    # WHAT: cu_seqlens_q[i+1] - 1. With cu_seqlens_q = [0, 5, 9] -> [4, 8].
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # WHAT: 128 MiB of scratch space FlashInfer uses for its intermediate results.
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # WHAT: one wrapper object for prefill and one for decode. "NHD" is the memory layout of
        #       the K/V cache: [tokens, heads, head_dim].
        # WHY backend="fa2": the comment says FlashInfer's FA3 path is slow, so the FA2 path is used.
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        # WHAT: share one integer scratch buffer between the prefill and decode wrappers (saves memory).
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        # WHAT: a cached pinned-memory vector of ones (the per-request "last page length").
        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        # WHAT: marks the point where the last plan()'s asynchronous host-to-device copy was queued.
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    # ELI5: "Make the plan for this batch, but only ONCE (the first layer pays; the rest reuse)."
    # WHAT: plan() takes the CPU-side arrays and prepares the kernel's work split. Prefill and
    #       decode wrappers have different plan() signatures, hence the two branches.
    # WHY last_event.synchronize(): plan() stages data in a pinned CPU buffer and launches an
    #     async copy to the GPU. Before the NEXT plan overwrites that buffer, wait until the copy
    #     from the previous plan has finished.
    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    # WHAT: a pinned CPU vector of `bs` ones. It grows to the next power of two to avoid
    #       reallocating on every small change of batch size.
    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    # ELI5: THE MEETING. Called by every layer.
    # WHAT: 1) make sure this batch has been planned
    #       2) store this layer's new K and V in the pantry at batch.out_loc
    #       3) view the pantry's K and V for this layer as one-token "pages"
    #       4) run FlashInfer: new Questions vs all cached Labels/Gifts of each request
    # SHAPE: q [N, Hq_local, D] -> returns [N, Hq_local, D].
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    # ELI5: build the paperwork (see the example at the top) from the tray's requests.
    # WHAT:
    #   seqlens_q = extend_len per request (new tokens);  seqlens_k = device_len (whole context)
    #   cu_seqlens_k = running sum of seqlens_k (prefixed with 0)
    #   cu_seqlens_q has three cases:
    #       decode (every extend_len is 1)   -> 0,1,2,...,bs
    #       prefill with no cache hits       -> same as cu_seqlens_k (query length == key length)
    #       prefill after a prefix-cache hit -> running sum of seqlens_q
    #   indices = for every request, its first device_len slots from the page table, concatenated.
    #   The wrapper is the decode one or the prefill one depending on the phase.
    # NOTE: uses batch.padded_reqs, so dummy padding requests get paperwork too (needed by CUDA graphs).
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    # WHAT: CUDA-graph setup: create fixed buffers for the largest recorded batch. The page table
    #       buffer is viewed as 1-D because FlashInfer wants the flat "ragged" index list.
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    # WHAT: decide once whether decode should use Tensor Cores (env override, else the GQA >= 4 rule).
    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    # ELI5: while "filming" the CUDA graph for batch size bs, give FlashInfer a special
    #       graph-friendly wrapper that reads its inputs from the FIXED stage buffers.
    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    # ELI5: before PLAYING the recording, re-plan with this step's real lengths and slot lists,
    #       writing them into the fixed buffers the recording reads.
    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
