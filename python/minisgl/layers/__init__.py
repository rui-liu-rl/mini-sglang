# FILE MAP -- layers/__init__.py
# WHAT: the building-block shelf. Re-exports every layer so model files can write
#       `from minisgl.layers import RMSNormFused, VocabParallelEmbedding, ...`.
# THE SHELF:
#   base.py        BaseOP / StateLessOP / OPList  (the mini replacement for nn.Module)
#   linear.py      matrix-multiply layers in tensor-parallel flavours
#   embedding.py   input lookup table and output scoreboard (LM head)
#   norm.py        RMSNorm and the fused norm+residual
#   rotary.py      RoPE position rotation
#   attention.py   the attention moderator (calls the backend)
#   activation.py  silu_and_mul / gelu_and_mul
#   moe.py         mixture-of-experts weights + wrapper
# Tag legend: see zero-to-sglang/eli5/README.md
from .activation import gelu_and_mul, silu_and_mul
from .attention import AttentionLayer
from .base import BaseOP, OPList, StateLessOP
from .embedding import ParallelLMHead, VocabParallelEmbedding
from .linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
)
from .moe import MoELayer
from .norm import RMSNorm, RMSNormFused
from .rotary import get_rope, set_rope_device

__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "AttentionLayer",
    "BaseOP",
    "StateLessOP",
    "OPList",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "LinearColParallelMerged",
    "LinearRowParallel",
    "LinearOProj",
    "LinearQKVMerged",
    "RMSNorm",
    "RMSNormFused",
    "get_rope",
    "set_rope_device",
    "LinearReplicated",
    "MoELayer",
]
