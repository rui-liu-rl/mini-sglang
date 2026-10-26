# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

# =====================================================================================
# FILE MAP -- benchmark/offline/bench.py
# =====================================================================================
# WHAT THIS FILE IS: an OFFLINE throughput benchmark: no HTTP server; the engine is used as a
#                    Python library (minisgl.llm.LLM) and fed a fixed batch of random requests.
#
# ELI5: Hand the kitchen 256 orders at once and time how long it takes to cook them all. The
#       answer is "output tokens per second".
#
# THE WORKLOAD (seed 0, so every run is identical):
#       256 requests; each prompt is a list of random token ids, length uniform in [100, 1024];
#       each asks for a random 100 to 1024 output tokens with ignore_eos=True (so the output length
#       is exactly what was asked, whatever the model "wants" to say); temperature 0.6.
#       On average about 560 tokens in and 560 tokens out per request, roughly 143,000 output
#       tokens in total.
#
# THE ENGINE SETTINGS ("align the hyperparameters"): max_seq_len_override 4096, up to 16,384 prompt
#       tokens per prefill pass, CUDA graphs for decode batch sizes up to 256, KV page size 256.
#
# WHAT THE NUMBER MEANS: throughput = (sum of requested output tokens) / (seconds for generate()).
#       It counts OUTPUT tokens only, not the prompt tokens processed by prefill.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        page_size=256,
    )

    # WHAT: random token ids (0 to 10000) as the prompts, so no two prompts share a prefix.
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    # WHY: the first call pays one-off costs (library setup, kernel compilation); do it before the clock starts.
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
