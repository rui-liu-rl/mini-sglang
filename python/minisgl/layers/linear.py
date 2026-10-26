# =====================================================================================
# FILE MAP -- layers/linear.py
# =====================================================================================
# WHAT THIS FILE IS: the "multiply by a weight matrix" layers, in the flavours tensor parallelism
#                    needs. Most of a transformer's parameters live here.
#
# ELI5: A linear layer is a big grid of numbers W. y = x @ W^T. With SEVERAL GPUs, the grid is
#       cut into pieces so each GPU stores and multiplies only its piece. There are exactly two
#       ways to cut a grid, and they have opposite consequences:
#
#         COLUMN-PARALLEL: cut the grid by OUTPUT rows. Every GPU sees the WHOLE input and
#                          produces ITS SLICE of the output. No communication needed; the
#                          slices simply stay where they are for the next step.
#             x (full) ──▶ GPU0: W0 x = y0  \
#                      ──▶ GPU1: W1 x = y1   > y = [y0 | y1] (never gathered)
#
#         ROW-PARALLEL:    cut the grid by INPUT columns. Every GPU holds a SLICE of the input
#                          (which is exactly what the previous column-parallel layer left it
#                          with) and produces a PARTIAL SUM of the full output. One all_reduce
#                          adds the partial sums.
#             x0 ──▶ GPU0: W0 x0 = p0 \
#             x1 ──▶ GPU1: W1 x1 = p1  > y = p0 + p1   (all_reduce = "sum across GPUs")
#
#       Pair them: column-parallel THEN row-parallel needs just ONE all_reduce per block.
#         attention: qkv_proj (column)  -> attention per local head -> o_proj (row)  + all_reduce
#         MLP:       gate_up_proj (col) -> silu*up                   -> down_proj (row) + all_reduce
#
# WEIGHT SHAPE CONVENTION (PyTorch): weight is [output_size, input_size]; F.linear(x, W) = x @ W^T.
#
# WHO USES IT: models/utils.py (RopeAttn, GatedMLP, MoEMLP).
# WHAT IT CALLS: F.linear and the DistributedCommunicator (all_reduce).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    # WHAT: remembers both the FULL size (the whole model) and the LOCAL size (this GPU's slice).
    # SHAPE: weight [local_osize, local_isize]; bias [local_osize] or None.
    # NOTE: torch.empty here runs under `with torch.device("meta")` (see engine.py), so no memory
    #       is allocated until load_state_dict swaps in the real tensor.
    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ELI5: NOT cut: every GPU keeps the whole grid. Used for tiny things that are cheaper to
#       duplicate than to communicate (the mixture-of-experts "router").
class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


# ELI5: COLUMN-parallel, with several matrices stacked into one (here: the "gate" and "up"
#       matrices of the gated MLP). Each part is cut in the same way.
# SHAPE: weight [sum(output_sizes) / tp, input_size]. Example Qwen3-0.6B, tp=1:
#        output_sizes [3072, 3072] -> weight [6144, 1024].
# WHY merge: one big matrix multiply is faster than two small ones.
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


# ELI5: COLUMN-parallel for attention's Q, K and V, stacked into one matrix. They are cut BY HEAD:
#       each GPU gets its share of the Question heads and of the Label/Gift heads.
# WHAT: local_num_qo = Question heads / tp. local_num_kv = KV heads / tp, or 1 if there are
#       FEWER KV heads than GPUs (allow_replicate: several GPUs then hold a copy of the same
#       KV head).
# SHAPE: weight [(local_num_qo + 2 * local_num_kv) * head_dim, hidden_size].
#        Example Qwen3-0.6B, tp=1: (16 + 2*8) * 128 = 4096 -> [4096, 1024]. With tp=2: [2048, 1024].
class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


# ELI5: ROW-parallel: attention's OUTPUT projection. Each GPU holds the attention result of ITS
#       heads (its slice of the input) and produces a partial sum of the full hidden vector;
#       all_reduce adds them.
# SHAPE: weight [hidden_size, (num_qo_heads * head_dim) / tp]. Example Qwen3-0.6B, tp=2: [1024, 1024].
# GOTCHA: if a bias were present it would be added on EVERY GPU before the all_reduce, so the
#         final sum would contain tp_size copies of it. These layers are created with
#         has_bias=False for that reason.
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # WHAT: add the partial sums from all GPUs (in place), only if there is more than one GPU.
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


# ELI5: ROW-parallel again, for the MLP's "down" matrix. Same pattern as LinearOProj.
# SHAPE: weight [output_size, input_size / tp]. Example Qwen3-0.6B, tp=2: [1024, 1536].
class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
