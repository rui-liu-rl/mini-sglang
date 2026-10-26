# FILE MAP -- models/base.py
# WHAT: the contract every whole-model class follows: "a model is a BaseOP whose forward() takes
#       no arguments and returns logits".
# ELI5: the model reads its inputs from the Kitchen whiteboard (core.Context.batch) instead of
#       taking them as arguments. That keeps the call identical for eager mode and for the
#       pre-recorded CUDA graphs (engine/graph.py), which record a function with no arguments.
# SHAPE: forward() -> logits [batch_size, vocab_size].
# Tag legend: see zero-to-sglang/eli5/README.md
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
