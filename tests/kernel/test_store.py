# =====================================================================================
# FILE MAP -- tests/kernel/test_store.py   (a script; needs a CUDA GPU)
# =====================================================================================
# WHAT THIS FILE TESTS: the K/V write kernel `store_cache` (kernel/store.py, csrc/jit/store.cu)
#                       against the plain PyTorch way (k_cache[indices] = k), for correctness and speed.
#
# ELI5: Make a pretend pantry of 1,048,576 slots, each holding 128 half-precision numbers for K and
#       128 for V. For batch sizes 1, 2, 4, ... 32,768: pick that many DISTINCT random slots (the
#       kernel gives no guarantee for duplicate slots), write random K and V rows there with the
#       custom kernel, and check the shelf holds exactly those rows. Then time it against the
#       PyTorch version (compiled with torch.compile) and print effective GB/s.
#
# WHY THE SETUP IS SLIGHTLY AWKWARD ON PURPOSE: K and V are slices of ONE interleaved tensor
#       (kv_cache[:, 0, :] and [:, 1, :]), so a cache row is 256 elements apart, and the inputs k, v
#       are slices of a wider qkv tensor (512 elements apart). The kernel must handle both row
#       strides, as it does when fed real model output. Bytes moved = bs x 128 x 2 (K and V) x 2.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from minisgl.benchmark.perf import compare_memory_kernel_perf
import torch
from minisgl.kernel import store_cache
from minisgl.utils import call_if_main


@call_if_main(__name__)
def test_store_cache():
    HEAD_SIZE = 128
    NUM_TOKENS = 1048576  # 1M
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    for bs in [2**n for n in range(0, 16)]:
        # NOTE: we cannot tolerate duplicate indices in this test
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)
        qkv = torch.randn((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)
        k = qkv[:, :HEAD_SIZE]
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]
        store_cache(
            k_cache,
            v_cache,
            indices,
            k,
            v,
        )

        assert torch.all(k_cache[indices] == k), bs
        assert torch.all(v_cache[indices] == v), bs

        # 2 = k + v
        MEM = bs * HEAD_SIZE * 2 * kv_cache.element_size()

        k = k.contiguous()
        v = v.contiguous()

        @torch.compile()
        def baseline():
            k_cache[indices] = k
            v_cache[indices] = v

        compare_memory_kernel_perf(
            our_impl=lambda: store_cache(k_cache, v_cache, indices, k, v),
            baseline=baseline,
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
