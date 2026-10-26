# =====================================================================================
# FILE MAP -- benchmark/online/bench_simple.py
# =====================================================================================
# WHAT THIS FILE IS: the simplest ONLINE benchmark: send a burst of requests (all at once) to a
#                    RUNNING mini-sglang server and print TTFT / TPOT / throughput.
#
# ELI5: Sixty-four customers walk in at the same moment, each with a prompt of random length (1 to
#       8,192 words) asking for a random number of answer tokens (16 to 1,024). We time everyone.
#
# STEPS
#   1. connect to the server at 127.0.0.1:1919 and ask which model it serves; load that tokenizer
#   2. in the background, build 64 random prompts (random length, random token ids); meanwhile
#      send a tiny connection test (a 100-token prompt, 2 output tokens). If fewer than three
#      timestamps come back, report failure and stop
#   3. send all 64 at once with benchmark_one_batch and print the statistics
#
# Because every prompt is random, there is no shared prefix: the prefix cache cannot help here.
# Statistics code: benchmark/client.py (process_benchmark_results). Random seed 42.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import asyncio
import random
import sys
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


async def main():
    try:
        random.seed(42)  # reproducibility

        # WHAT: build `max_bs` random prompts; `await asyncio.sleep(0)` after each lets other tasks
        #       (the connection test) run while the prompts are being generated.
        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, MAX_INPUT)
                message = generate_prompt(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        TEST_BS = [64]
        PORT = 1919
        MAX_INPUT = 8192
        # Create the async client
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
            MODEL = await get_model_name(client)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            logger.info(f"Loaded tokenizer from {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                raise e from e

            msgs = await gen_task
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            for batch_size in TEST_BS:
                try:
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL
                    )
                    process_benchmark_results(results)
                except Exception as e:
                    logger.info(f"Error with batch size {batch_size}: {e}")
                    continue
            logger.info("Benchmark completed.")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
