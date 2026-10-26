# =====================================================================================
# FILE MAP -- tests/kernel/test_index.py   (a script; needs a CUDA GPU)
# =====================================================================================
# WHAT THIS FILE TESTS: the custom row-lookup kernel `indexing` (kernel/index.py, csrc/jit/index.cu)
#                       against PyTorch's F.embedding, for correctness and for speed.
#
# ELI5: Two exams. Exam 1: look up random rows of a [131,072 x 4,096] half-precision table with
#       batch sizes 1, 2, 4, ... 32,768; the answers must be bit-identical to PyTorch's, and a
#       timing table is printed (effective GB/s for each). Exam 2: the same, but pretending to be
#       one of 4 tensor-parallel GPUs that owns only rows [32,768, 65,536): ids outside that window
#       must give rows of zeros.
#
# ref_indexing is the slow, obvious reference: shift the ids by `start`, remember which fall
#       outside [0, length), look those up as row 0 (so F.embedding doesn't crash), then zero them.
#
# GOTCHA: in test_indexing_with_mask the TIMING block calls indexing(weights, indices) WITHOUT
#       vocab_range (for both sides), so it times the plain path; only the correctness check
#       above it exercises the masked kernel. Looks like a copy-paste.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel import indexing
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def ref_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if vocab_range is not None:
        start, length = vocab_range
        assert length <= weights.shape[0]
        indices = indices - start
        indices_mask = (indices < 0) | (indices >= length)
        indices[indices_mask] = 0  # set out-of-vocab indices to zero
        result = F.embedding(indices, weights)
        result[indices_mask] = 0
        return result
    else:
        return F.embedding(indices, weights)


@call_if_main(__name__)
def test_indexing():
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # first test the correctness
        result = indexing(
            weights,
            indices,
        )
        expected = ref_indexing(
            weights,
            indices,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"

        # test the perf
        MEM = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),
            baseline=lambda: ref_indexing(weights, indices),
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
        )


@call_if_main(__name__)
def test_indexing_with_mask():
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072
    TP = 4
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # start, length

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # first test the correctness
        result = indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        expected = ref_indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"

        # test the perf
        MEM = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),
            baseline=lambda: ref_indexing(weights, indices),
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
