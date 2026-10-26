# =====================================================================================
# FILE MAP -- engine/config.py
# =====================================================================================
# WHAT THIS FILE IS: the chef's settings sheet (EngineConfig). SchedulerConfig and ServerArgs
#                    build on top of it.
#
# ELI5: Everything the chef needs to know before opening: WHICH recipe book, HOW MANY chefs
#       share it, WHAT number format, HOW MUCH of the pantry to use, WHICH tools.
#
# WHO CALLS IT: engine/engine.py (reads it), scheduler/config.py (extends it).
# WHAT IT CALLS: Hugging Face config loading (cached).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# WHAT: frozen = read-only after creation (engine.py's _adjust_config is the one deliberate
#       exception, using object.__setattr__ at startup).
@dataclass(frozen=True)
class EngineConfig:
    # WHAT: Hugging Face repo id or local folder with config.json, tokenizer and *.safetensors.
    model_path: str
    # WHAT: "I am GPU worker `rank` out of `size`" (tensor-parallel info).
    tp_info: DistributedInfo
    # WHAT: number format of weights and activations (bfloat16 / float16 / float32).
    dtype: torch.dtype
    # WHAT: how many requests may run at the same time. Sizes the page table and the token pool
    #       (one row per request).
    max_running_req: int = 256
    # WHAT: "auto", or names like "fi", "fa", "trtllm", or "prefillBackend,decodeBackend".
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # WHAT: exact list of batch sizes to record CUDA graphs for (None = derive it, see graph.py).
    cuda_graph_bs: List[int] | None = None
    # WHAT: the largest recorded batch size (None = pick from free memory; 0 or less = disabled).
    cuda_graph_max_bs: int | None = None
    # WHAT: token slots per KV page. 1 = finest. Some attention backends force a larger value.
    page_size: int = 1
    # WHAT: fraction of the free GPU memory (measured BEFORE loading weights) that weights plus
    #       KV cache may use. The rest is headroom for activations and CUDA graphs.
    memory_ratio: float = 0.9
    # WHAT: seconds to wait for the other GPU workers before giving up.
    distributed_timeout: float = 60.0
    # WHAT: load random weights instead of real ones (speed/memory tests only).
    use_dummy_weight: bool = False
    # WHAT: use mini-sglang's own NCCL wrapper for GPU-to-GPU traffic.
    use_pynccl: bool = True
    # WHAT: override the longest allowed sequence (otherwise the model's own maximum).
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # WHAT: the raw Hugging Face config.json, loaded once and remembered (cached_property).
    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    # WHAT: mini-sglang's own, simplified view of the model's shape (layers, heads, ...).
    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    # WHAT: the longest conversation (prompt + answer) allowed: the model's own maximum
    #       position unless overridden.
    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    # WHAT: the longest single forward pass; here equal to max_seq_len (SchedulerConfig lowers it
    #       to max_extend_tokens). Used to size communication buffers.
    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    # WHAT: where GPU workers rendezvous for torch.distributed. Fixed port 2333 here;
    #       ServerArgs overrides it to HTTP port + 1.
    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
