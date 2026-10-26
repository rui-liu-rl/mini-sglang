# =====================================================================================
# FILE MAP -- utils/torch_utils.py
# =====================================================================================
# WHAT THIS FILE IS: two small PyTorch conveniences.
#
# ELI5: (1) torch_dtype: "while I build the model, make every new number this kind (bf16 ...)".
#       (2) nvtx_annotate: stick a NAME TAG on a function so that, in a GPU profiler's timeline,
#           the block of work shows up labelled "Layer_7" or "Sampler" instead of anonymous kernels.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ELI5: "Within this with-block, newly created floating-point tensors default to `dtype`."
# WHY: the model skeleton is built with torch.empty(...) calls that do not name a dtype; this makes
#      them bf16/fp16 as requested, without touching every layer.
# WHAT: always restores the previous default, even if building the model raises an error.
@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


# ELI5: a name tag for the profiler.
# WHAT: a decorator FACTORY: nvtx_annotate("Layer_{}", layer_id_field="_layer_id") returns a decorator.
#       When the decorated method runs, it is wrapped in an NVTX range with that display name.
#       If layer_id_field is given and the object has that attribute, the {} in the name is filled
#       with it ("Layer_{}" -> "Layer_7"). NVTX ranges are what Nsight Systems shows as labelled bars.
# COST: near zero when no profiler is attached.
def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        # functools.wraps keeps the original function's name and metadata.
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
