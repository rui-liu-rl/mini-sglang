# =====================================================================================
# FILE MAP -- benchmark/perf.py
# =====================================================================================
# WHAT THIS FILE IS: a stopwatch for GPU functions, used by tests/kernel/* to compare a custom
#                    kernel against a baseline (usually the plain PyTorch way).
#
# ELI5: Timing GPU code with time.time() is misleading: the CPU only QUEUES work and moves on, so
#       the clock stops before the GPU is done. The right tool is a pair of CUDA events recorded
#       ON the GPU's timeline: "start" and "stop", and the GPU itself measures the time between.
#       Also, one call is too short and noisy to trust, so we repeat it many times and divide.
#
#       perf_cuda(f)                    average milliseconds per call of f
#       compare_memory_kernel_perf      run a baseline and our kernel, print latency and effective
#                                       memory BANDWIDTH (bytes moved / time), the right yardstick
#                                       for copy-like kernels (Part I chapter 3)
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from minisgl.utils import init_logger

logger = init_logger(__name__)


# ELI5: "How many milliseconds does one call of f take on the GPU?"
# WHAT (in order):
#   1. make a fresh CUDA stream (a private lane), or reuse the current one if init_stream=False
#   2. call f once (warm-up: compile, allocate, load caches)
#   3. CUDA-graph mode (cuda_graph_repetitions = N): record N back-to-back calls of f into one CUDA
#      graph. Replaying the graph removes the CPU launch overhead of each call, so the measurement
#      is the kernel itself. Otherwise just replay f directly (N = 1).
#   4. synchronise, replay once more (warm-up of the replay), then time `repetitions` replays
#      between two GPU events
#   5. return total / (N x repetitions) = milliseconds per call
# GOTCHA: a first `stream = torch.cuda.Stream()` is created and then thrown away (immediately
#         replaced in the if/else below); harmless.
def perf_cuda(
    f: Callable[[], Any],
    *,
    init_stream: bool = True,
    repetitions: int = 10,
    cuda_graph_repetitions: int | None = 10,
) -> float:
    import torch

    assert repetitions > 0
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if init_stream:
        stream = torch.cuda.Stream()
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        f()
        if N := cuda_graph_repetitions:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(N):
                    f()
            replay = g.replay
            del g
        else:
            replay = f
            N = 1

        torch.cuda.synchronize()

        replay()
        tic.record()
        for _ in range(repetitions):
            replay()
        toc.record()
        toc.synchronize()
        dur = tic.elapsed_time(toc)
        return dur / (N * repetitions)


# ELI5: "Race the baseline against our kernel and report how fast each moves memory."
# WHAT: bandwidth = memory_footprint bytes / (duration in ms x 1e6) = GB/s (decimal). `memory_footprint`
#       is the number of bytes the operation reads plus writes, supplied by the caller.
#       Prints one log line with both latencies and bandwidths and returns the two bandwidths.
# EXAMPLE: copying 100 MB in 0.5 ms is 100e6 / (0.5 x 1e6) = 200 GB/s.
def compare_memory_kernel_perf(
    *,
    baseline: Callable[[], Any],
    our_impl: Callable[[], Any],
    memory_footprint: int,  # in bytes
    description: str = " ",
    extra_kwargs: Dict[str, Any] | None = None,
    need_latency: bool = True,
) -> Tuple[float, float]:
    extra_kwargs = extra_kwargs or {}

    dur = perf_cuda(baseline, **extra_kwargs)
    bandwidth_0 = memory_footprint / (dur * 1e6)  # GB/s
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    message_0 = f"Baseline: {latency_msg}{bandwidth_0:8.3f} GB/s"

    dur = perf_cuda(our_impl, **extra_kwargs)
    bandwidth_1 = memory_footprint / (dur * 1e6)  # GB/s
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    logger.info(f"{description}{message_0} | Our Impl: {latency_msg}{bandwidth_1:8.3f} GB/s")
    return bandwidth_0, bandwidth_1
