# =====================================================================================
# FILE MAP -- benchmark/online/bench_qwen.py
# =====================================================================================
# WHAT THIS FILE IS: an ONLINE benchmark that replays a recorded real-world request trace against
#                    a RUNNING mini-sglang server (start it first: python -m minisgl --model ...,
#                    port 1919), at several request rates, and prints TTFT / TPOT / throughput.
#
# ELI5: "Replay yesterday's traffic, then play it faster and faster, and see when the kitchen
#       starts to struggle." The trace is a public file of real requests from an Alibaba service
#       (one JSON line per request: time sent, prompt length, output length).
#
# STEPS
#   1. connect to the server through the OpenAI-style client; ask it which model it serves
#   2. download the trace file once (next to this script) and read the first N = 1000 lines
#      (dummy=True: prompts are cut from one long random prompt, so they SHARE prefixes, which
#       lets a prefix cache show its worth)
#   3. for each SCALE in [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]: multiply all send-times by SCALE
#      (smaller = requests come faster) and replay; print the statistics after each pass
#
# READING THE RESULTS: as the scale gets smaller the arrival rate rises. Watch TTFT: when the
#       server cannot keep up, requests queue and TTFT climbs sharply (the "knee" of the curve).
#       (Statistics are produced by benchmark/client.py: process_benchmark_results.)
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


# WHAT: download the trace to the folder of this script the first time; reuse it afterwards.
def download_qwen_trace(url: str) -> str:
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


async def main():
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
