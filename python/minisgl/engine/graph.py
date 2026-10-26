# =====================================================================================
# FILE MAP -- engine/graph.py
# =====================================================================================
# WHAT THIS FILE IS: the CUDA-GRAPH RECORDER AND PLAYER for decode passes.
#
# ELI5: One decode pass of a 28-layer model launches hundreds of tiny GPU programs ("kernels"),
#       one after another. Each launch costs the CPU a few microseconds; for a small model that
#       can be as slow as the actual GPU work, so the GPU spends time waiting for the CPU.
#       FIX: film it once. At startup the chef "rehearses" a decode pass for each tray size
#       (1, 2, 4, 8, 16, ...) while the GPU RECORDS every launch into a "graph" (a video of the
#       motions). During service, to cook a decode tray the CPU just says "play recording #8":
#       ONE call replays hundreds of launches back to back.
#
# WHAT MAKES A RECORDING REPLAYABLE
#   A recording remembers memory ADDRESSES, not values. So the inputs must always be placed in
#   the same fixed buffers: GraphCaptureBuffer (input ids, KV write slots, positions) and the
#   output logits. Before each replay we COPY this pass's real values into those fixed buffers.
#
# WHY ONLY FOR DECODE: a decode pass has a fixed shape (1 new token per order, N orders). A
#   prefill pass has a different number of tokens every time, so no single recording fits.
#
# WHY PADDING: there is one recording per tray size in the list (1, 2, 4, 8, 16, 24, ...). A
#   tray of 11 orders is topped up with 5 DUMMY orders to use the size-16 recording.
#   (core.Batch.padded_reqs; the dummy order writes its K/V into a throw-away dummy page.)
#
# WHO CALLS IT: engine/engine.py (creates it; forward_batch asks can_use_cuda_graph / replay),
#               scheduler/scheduler.py (_prepare_batch calls pad_batch).
# WHAT IT CALLS: torch.cuda.CUDAGraph, the model, the attention backend.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


# ELI5: the FIXED STAGE the recording is filmed on. Actors (data) are copied onto the stage
#       before each replay; the recording always looks at the same spots.
# SHAPE: sized for the BIGGEST recorded batch (bs = max_graph_bs); smaller trays use a prefix.
@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    # WHAT: allocate the stage. logits are float32 (batch, vocab).
    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    # WHAT: point the batch's fields at the first padded_size entries of the stage buffers
    #       (views, not copies), so the model run during capture reads/writes the stage.
    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    # WHAT: before a replay, copy the REAL batch's values onto the stage.
    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


# ELI5: "Which tray sizes should I record?"
# WHAT: an explicit list wins. Otherwise: max batch size is 160 (or 256 if there is more than 80 GiB free,
#       e.g. an H200); a max below 1 disables graphs; sizes are 1, 2, 4, then 8, 16, 24, ...,
#       up to the max in steps of 8. Few recordings for tiny trays (rare, and cheap anyway), more
#       for bigger ones.
def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


# WHAT: format a byte count as GiB for logs.
def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


# WHAT: free memory (bytes) on this GPU, as reported by the CUDA driver.
def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        # WHAT: 0 means "graphs disabled".
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        # ELI5: do all the filming right now, at startup (this is why startup is slow).
        self._capture_graphs(max_seq_len, vocab_size, model)

    # ELI5: FILM one recording per tray size.
    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # WHAT: tray size -> recording.
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # WHAT: the attention backend allocates its own fixed buffers for the largest sizes.
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        # WHAT: clean the memory counters so the log lines below show what capture costs.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # ELI5: build the stage, big enough for the biggest tray.
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        # WHAT: all recordings share ONE pool of memory for their temporary tensors. They are never
        #       replayed at the same time, so they can reuse the same space.
        # WHY biggest first: the first (largest) recording sizes the shared pool; smaller ones fit in it.
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            # ELI5: the rehearsal tray is `bs` copies of the dummy order, in the decode phase.
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                # WHAT: one ordinary (un-recorded) run first. WARM-UP: lets lazy initialisation
                #       (kernel compilation, workspace allocation, ...) happen BEFORE recording,
                #       because those things must not appear inside a recording.
                self.buffer.logits[:bs] = model.forward()
                # ELI5: "camera rolling": everything launched inside this with-block is recorded
                #       into `graph` instead of being executed.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    # WHAT: graphs are used only for decode trays whose size is within the recorded range.
    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    # ELI5: "PLAY RECORDING #n."
    # WHAT: 1) copy this tray's real inputs onto the stage
    #       2) pick the recording for the padded tray size
    #       3) let the attention backend refresh its per-step metadata (page lists, lengths...)
    #          in the buffers the recording reads
    #       4) replay: ONE call launches the whole recorded pass
    #       5) return the logits rows of the REAL orders (padding rows dropped)
    # SHAPE: returns (batch.size, vocab_size) float32.
    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    # ELI5: top the tray up with blank dummy orders to a recorded size.
    # WHAT: padded size = the smallest recorded size >= the real size (only if graphs apply;
    #       otherwise no padding). Example: sizes [1,2,4,8,16,...], a tray of 11 -> 16, so 5
    #       dummy orders are appended.
    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    # WHY: a recorded graph may contain communication operations that reference NCCL objects;
    #      releasing NCCL first can leave them dangling and hang the process at exit.
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
