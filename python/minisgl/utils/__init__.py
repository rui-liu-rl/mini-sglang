# FILE MAP -- utils/__init__.py
# WHAT: the "drawer index". Re-exports the small shared helpers so other files can write
#       `from minisgl.utils import init_logger, div_ceil, ZmqPushQueue ...`.
# THE DRAWER CONTENTS:
#   arch.py         which GPU generation is this (SM90 = Hopper, SM100 = Blackwell)
#   hf.py           fetch tokenizer / config.json / weights from the Hugging Face hub
#   logger.py       coloured logging, plus *_rank0 variants
#   misc.py         div_even, div_ceil, align_ceil, align_down ...
#   mp.py           ZeroMQ queue wrappers (the Kitchen's phone lines)
#   registry.py     the named-options "menu board"
#   torch_utils.py  torch_dtype context and the nvtx_annotate profiler tag
# Tag legend: see zero-to-sglang/eli5/README.md
from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
from .hf import cached_load_hf_config, download_hf_weight, load_tokenizer
from .logger import init_logger
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import nvtx_annotate, torch_dtype

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_tokenizer",
    "init_logger",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
