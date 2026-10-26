# =====================================================================================
# FILE MAP -- engine/engine.py
# =====================================================================================
# WHAT THIS FILE IS: THE CHEF'S STATION on ONE GPU. It loads the recipe book (weights), sets up
#                    the pantry (KV cache), the peg-board page table, the attention tools and
#                    the sampler; and offers ONE main action: forward_batch(tray) -> next words.
#
# ELI5: When the restaurant opens, the chef:
#         1. claims a GPU and a private "lane" (CUDA stream)
#         2. joins the other chefs if the recipe book is split over several GPUs (TP)
#         3. loads the recipe book onto the GPU                                (weights)
#         4. measures how much room is left and fills ALL of it with numbered pantry
#            shelves                                                            (KV cache pool)
#         5. hangs up the peg board                                              (page_table)
#         6. picks the attention tools and the word-picker                       (backends, sampler)
#         7. rehearses the cooking motions for tray sizes 1,2,4,8,... and records them
#            like a video so each can be replayed instantly                      (CUDA graphs)
#       During service: forward_batch(tray) cooks one pass and returns the next word for every
#       order on the tray.
#
# THE ENGINE KNOWS NOTHING ABOUT: HTTP, tokenizing, queues, finishing criteria. Those belong to
# the scheduler. The engine just "cooks the tray it is handed".
#
# MEMORY BUDGET (how num_pages is chosen, see _determine_num_pages):
#       KV budget = memory_ratio (default 0.9) x (free GPU memory BEFORE loading the model)
#                   - (memory the model weights took)
#       num_pages = KV budget // (bytes per page)
#   The remaining ~10% of the original free memory is left for temporary activations,
#   CUDA-graph buffers and fragmentation.
#
# WHO CALLS IT: scheduler/scheduler.py (creates it and calls forward_batch).
# WHAT IT CALLS: models, kvcache, attention, moe, distributed, engine/graph.py, engine/sample.py.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


# ELI5: what the chef hands back after one pass.
class ForwardOutput(NamedTuple):
    # SHAPE: (batch.size,), int32, on the GPU: the next token for each real order on the tray.
    next_tokens_gpu: torch.Tensor
    # WHAT: the same tokens copied to CPU memory (asynchronously; not valid until the event fires).
    next_tokens_cpu: torch.Tensor
    # WHAT: fires when the GPU-to-CPU copy above has finished. The scheduler waits on exactly this.
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        # WHY: the engine must be the FIRST thing to touch CUDA in this process (spawned fresh by
        #      server/launch.py), so that set_device and the memory measurements below are clean.
        assert not torch.cuda.is_initialized()
        # WHAT: record "I am GPU worker r of n" in a global the layers can query.
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # WHAT: resolve "auto" settings (attention backend, page size, MoE backend).
        _adjust_config(config)

        # ELI5: claim GPU number `rank` (one worker process per GPU) and fix the random seed
        #       (so sampling is reproducible run to run).
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        # ELI5: the chef's private LANE on the GPU; all engine work is queued in it.
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        # ELI5: hang up the whiteboard (core.Context) where every layer can find it.
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # ELI5: if there are several GPUs, make the phone connection between the chefs, and get
        #       a CPU-side group for small control messages.
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        # ELI5: STEP 3, the recipe book. First build an EMPTY SKELETON of the model on the "meta"
        #       device (shapes and dtypes only, NO memory allocated), in the model's dtype. Then
        #       load the real weights and let them REPLACE the empty placeholders (see
        #       layers/base.py: load_state_dict just swaps tensors in, no copy).
        # WHY: avoids allocating a throw-away full model before loading, which would double the
        #      peak memory use during startup.
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        # ELI5: STEP 4, the pantry. How many pages fit in what is left of the memory?
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # ELI5: STEP 5, the peg board. One row per possible running order (+1 spare row for the
        #       dummy order), one column per position; each cell will hold a pantry slot number.
        # WHY the length is the smaller of two things: a sequence can't be longer than the model
        #     allows, AND it can't be longer than the whole pantry has slots for.
        # WHY align to 32 ints = 128 bytes: rows start at nicely aligned memory addresses, which
        #     GPU kernels like.
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # ELI5: STEP 6a, the attention tools (FlashAttention / FlashInfer / TensorRT-LLM).
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        # ELI5: STEP 6b, the word-picker.
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # ELI5: STEP 7. CUDA graphs need a stand-in order to rehearse with: a "dummy request"
        #       with one token (id 0) and the LAST row of the peg board (row max_running_req).
        # WHY the row is filled with num_tokens: that number is the index of the extra DUMMY PAGE
        #     at the end of the pantry. During rehearsal and padding, the dummy order writes its
        #     K/V there, harmlessly, instead of into a real order's slots.
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    # ELI5: set up how the chefs talk to each other.
    # WHAT: two communication worlds are needed:
    #         CPU world (gloo)  : small control messages and barriers
    #         GPU world         : big tensor traffic (all-reduce of activations)
    #       PyNCCL is mini-sglang's own thin wrapper around NCCL (kernel/pynccl.py). When it is
    #       used (the default, or when there is only one GPU) torch's group is only for the CPU
    #       world, and the GPU world is set up by enable_pynccl_distributed. Otherwise torch's own
    #       NCCL backend carries GPU traffic and a second gloo group carries the CPU traffic.
    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # WHAT: the biggest tensor one all-reduce will carry = the longest forward pass
            #       (tokens) x hidden size x bytes per number.
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    # WHAT: produce {parameter name: tensor} for this GPU's shard of the model.
    #       dummy mode: random numbers (for speed/memory tests without a download).
    #       real mode: stream the checkpoint (sharded and merged per GPU, see models/weight.py)
    #       and cast to the engine's dtype.
    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    # ELI5: "How many numbered shelves can the pantry hold?"
    # WHAT (step by step, all in bytes):
    #       cache_per_page = 2 (K and V) x head_dim x (this GPU's kv heads) x page_size
    #                        x bytes per number x number of layers
    #       model_memory   = free memory before loading the model - free memory after
    #       available      = memory_ratio x free-before - model_memory
    #       num_pages      = available // cache_per_page
    #       --num-pages overrides all of this.
    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    # ELI5: "How much room is free on the GPU, being careful to ask every chef?"
    # WHAT: cleans up cached allocations first (empty_cache) so the number is honest, reads the
    #       free memory, then all-reduces the MIN over all GPU workers. Returns (min free,
    #       max free). All GPUs must have almost the same free memory: they hold identical
    #       shards, so if they differ by more than 2 GiB something is wrong and we stop.
    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        # TRICK: MIN of [free, -free] gives (min free, -max free) in one collective.
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    # ELI5: THE MAIN ACTION. "Cook this tray for one pass and tell me the next word of each order."
    # WHAT (in order):
    #   1. hang the tray on the whiteboard (with ctx.forward_batch)
    #   2. run the model: replay a recorded CUDA graph if this is a small-enough decode tray,
    #      otherwise run the model normally (prefill, or a very big decode batch)
    #   3. tick each order's bookkeeping (complete_one) -- BEFORE the result is known (see core.py)
    #   4. pick next tokens, only for the real orders (logits[: batch.size] drops padding rows)
    #   5. start an asynchronous copy of the tokens to CPU memory and record an event that
    #      fires when the copy is done
    # SHAPE: logits is (number of rows, vocab_size). For prefill, only the LAST position of each
    #        request has a row: ParallelLMHead.forward (layers/embedding.py) keeps just those
    #        positions before computing scores, so rows == batch size. A graph-replayed decode
    #        batch may include padding rows; the [: batch.size] slice drops them.
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    # WHAT: orderly teardown. Free CUDA graphs first, then communicators.
    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


# WHAT: round `num` UP to a multiple of 32. Example: 40000 -> 40000 (already a multiple),
#       100 -> 128.
def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


# ELI5: replace "auto" settings by concrete choices, based on which GPU generation we have.
def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        # GOTCHA: EngineConfig is a frozen dataclass; object.__setattr__ bypasses that on purpose.
        object.__setattr__(config, attr, value)

    # WHAT: SM100 (Blackwell) -> TensorRT-LLM kernels for everything.
    #       SM90 (Hopper, e.g. H100) -> FlashAttention for prefill and FlashInfer for decode.
    #       anything older -> FlashInfer.
    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    # WHAT: the TensorRT-LLM backend only supports page sizes 16, 32 or 64.
    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
