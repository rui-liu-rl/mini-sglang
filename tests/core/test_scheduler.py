# =====================================================================================
# FILE MAP -- tests/core/test_scheduler.py   (a script; needs a GPU and the model weights)
# =====================================================================================
# WHAT THIS FILE TESTS: the scheduler + engine on their own, with NO HTTP server and NO tokenizer
#                       process: a hand-made "front desk" talks to the scheduler directly over
#                       ZeroMQ and checks that it produces a complete answer.
#
# ELI5: Hire one head waiter and one chef (a Scheduler process on GPU 0, model Llama-3.1-8B-Instruct,
#       at most 4 orders at once, CUDA graphs for tray sizes 2, 4, 8). Then play every other
#       role yourself:
#         1. tokenise one question with the model's own tokenizer
#         2. push a UserMsg (uid 0, those token ids, max 100 new tokens) into the scheduler's in-tray
#         3. read the scheduler's out-tray: each DetokenizeMsg carries ONE new token; append it,
#            stop when `finished`
#         4. decode and log the whole text; send ExitMsg so the scheduler shuts down
#       It is a smoke test: it passes if the loop ends and the logged text is sensible.
#
# NOTE: the scheduler runs in its own spawned process and signals readiness through a queue
#       (queue.put(None) / q.get()). The test's queues use create=False: it dials in to addresses
#       the scheduler owns.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, call_if_main, init_logger
from minisgl.core import SamplingParams

logger = init_logger(__name__)


@torch.inference_mode()
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    scheduler = Scheduler(config)
    queue.put(None)
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


@call_if_main(__name__)
def main():
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[2, 4, 8],
    )

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler, args=(config, q))
    p.start()
    q.get()

    send_backend = ZmqPushQueue(
        config.zmq_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )

    recv_backend = ZmqPullQueue(
        config.zmq_detokenizer_addr,
        create=False,
        decoder=BaseTokenizerMsg.decoder,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    send_backend.put(
        UserMsg(
            uid=0,
            input_ids=ids,
            sampling_params=SamplingParams(max_tokens=100),
        )
    )

    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:
            break

    logger.info(tokenizer.decode(ids.tolist()))
    send_backend.put(ExitMsg())
