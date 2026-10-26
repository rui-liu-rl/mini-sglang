# =====================================================================================
# FILE MAP -- distributed/impl.py
# =====================================================================================
# WHAT THIS FILE IS: the GPU-to-GPU "group calls". Two operations are enough for tensor
#                    parallelism:
#
#   all_reduce(x):  every GPU has a tensor x of the same shape (a PARTIAL result).
#                   Afterwards every GPU holds the ELEMENT-WISE SUM of all of them.
#                      GPU0: [1, 2]          GPU0: [4, 6]
#                      GPU1: [3, 4]   ==>    GPU1: [4, 6]        (sum = [1+3, 2+4])
#                   Used after a row-parallel matmul: each GPU computed part of the sum.
#
#   all_gather(x):  every GPU has a DIFFERENT slice. Afterwards every GPU holds all slices
#                   stacked one after another.
#                      GPU0: [a]            GPU0: [a, b]
#                      GPU1: [b]     ==>    GPU1: [a, b]
#                   Used to rebuild the full vocabulary scores from per-GPU slices.
#
# ELI5: all_reduce is "everybody write your partial answer on a whiteboard; we add them and
#       everybody copies the total". all_gather is "everybody hold up your piece of the jigsaw;
#       everybody ends up with the whole picture."
#
# TWO IMPLEMENTATIONS (plugins), the LAST one registered is used:
#       TorchDistributedImpl : torch.distributed (NCCL under the hood). The default.
#       PyNCCLDistributedImpl: mini-sglang's own thin wrapper (kernel/pynccl.py), added by
#                              enable_pynccl_distributed(). Used by default when tp > 1.
#
# WHO CALLS IT: layers/linear.py, layers/embedding.py, layers/moe.py (via DistributedCommunicator).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


# WHAT: the shared contract.
@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


# ELI5: the default phone system: PyTorch's built-in collective calls.
@dataclass
class TorchDistributedImpl(DistributedImpl):
    # WHAT: sum across GPUs, IN PLACE (x is overwritten with the total) and also returned.
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    # SHAPE: x [s0, ...] -> out [tp * s0, ...]: the GPUs' tensors stacked along dim 0, in rank order.
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


# ELI5: the in-house phone system: mini-sglang's own wrapper around NCCL (kernel/pynccl.py and
#       kernel/csrc/src/pynccl.cu). At startup it reserves ONE big buffer registered with NCCL as
#       "symmetric memory". An all_reduce of a tensor no bigger than that buffer is staged through
#       it (copy in, reduce in place, copy out; the copies are skipped if the tensor already lives
#       there), which lets NCCL use its faster symmetric-memory kernels. Bigger tensors take the
#       ordinary path. all_gather writes straight into the output tensor.
# GOTCHA: the wrapper supports float16 and bfloat16 tensors only.
@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


# ELI5: the switchboard layers call. They don't care WHICH implementation answers.
# WHAT: `plugins` is a CLASS-level list shared by all instances; the last entry wins. It starts
#       with the torch implementation; enable_pynccl_distributed() appends the PyNCCL one.
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


# WHAT: switch the switchboard to PyNCCL. Does nothing with a single GPU.
# WHY max_bytes: PyNCCL allocates its symmetric-memory buffer up front, sized for the largest
#     tensor one all_reduce is expected to carry (longest forward pass x hidden size x bytes per
#     number), capped by the MINISGL_PYNCCL_MAX_BUFFER_SIZE environment variable (default 1 GiB).
def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


# WHAT: forget all implementations. Engine.shutdown calls this after the CUDA graphs are destroyed.
def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
