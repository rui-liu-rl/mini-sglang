# =====================================================================================
# FILE MAP -- kernel/pynccl.py
# =====================================================================================
# WHAT THIS FILE IS: the Python side of mini-sglang's own NCCL wrapper ("PyNCCL"): how GPUs in a
#                    tensor-parallel group are connected so they can all_reduce / all_gather.
#
# ELI5: NCCL is NVIDIA's library for GPU-to-GPU "group calls". Before any group call, every
#       participant must hold the SAME secret group ticket (a "unique id"), made by one of them
#       and handed to the others. This file:
#         1. compiles / loads the C++ wrapper (kernel/csrc/src/pynccl.cu),
#         2. lets rank 0 create the ticket and shares it with every rank over the CPU-side (Gloo)
#            group, which is already working at that point,
#         3. builds the communicator object `PyNCCLImpl` on each rank from
#            (rank, group size, staging-buffer size, ticket).
#
# WHAT THE OBJECT CAN DO (see the stub class below): all_reduce, all_gather, and get_buffer (the
#       address of its staging buffer; nothing in this repo's Python calls it, but a tensor that
#       lives IN the buffer skips the two copies inside all_reduce). The Python type hint says
#       op="sum" only; the C++ side also knows prod, max, min and avg.
#
# WHO CALLS IT: distributed/impl.py (enable_pynccl_distributed -> init_pynccl).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    # WHAT: a typing-only description of the object's methods, so editors and type checkers know
    #       its shape. At run time (the else branch) the name is just `Any`.
    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


# WHAT: compile (first time) and load pynccl.cu, linked against the system NCCL library (-lnccl).
@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


# WHAT: declare to TVM-FFI that the C++ class registered as "minisgl.NCCLWrapper" can be created
#       from Python. __ffi_init__ calls its C++ constructor with the arguments given.
@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


# ELI5: "Connect this GPU to its tensor-parallel group."
# WHAT (in order):
#   1. cap the staging buffer: max_size_bytes = min(what the engine asked for, the env limit
#      MINISGL_PYNCCL_MAX_BUFFER_SIZE)
#   2. load the C++ module and the wrapper class
#   3. rank 0 makes the NCCL unique id; everyone receives it via broadcast_object_list over the
#      Gloo group (a CPU channel; the GPU channel does not exist yet, that's what we're building)
#   4. build the communicator from (rank, size, buffer size, id)
# GOTCHA: the broadcast is a collective call: every rank must reach it, rank 0 sends a real id,
#         the others send None as a placeholder to be overwritten.
def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
