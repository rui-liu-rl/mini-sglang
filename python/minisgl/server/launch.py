# =====================================================================================
# FILE MAP -- server/launch.py
# =====================================================================================
# WHAT THIS FILE IS: the "open the restaurant" button. It reads the command line, starts every
#                    worker process, waits until they all say "ready", then opens the front desk.
#
# ELI5: The manager (this file) arrives in the morning and:
#         1. reads the day's instructions (command-line flags)
#         2. hires one head-waiter-plus-chef per GPU (scheduler processes)
#         3. hires the card-reader (and card-makers if asked) (tokenizer processes)
#         4. waits at the door until every hire has raised a "ready" flag
#         5. opens the front desk (the API server, in THIS process)
#
# PROCESS MAP (default settings, 1 GPU):
#       main process            = API server (FastAPI + uvicorn)        <- this file's process
#       minisgl-TP0-scheduler   = Scheduler + Engine on GPU 0
#       minisgl-detokenizer-0   = tokenizer AND detokenizer (shared, because --num-tokenizer 0)
#
# WHO CALLS IT: `python -m minisgl` (__main__.py) and `python -m minisgl.shell` (shell.py).
# WHAT IT CALLS: server/args.py (flags), server/api_server.py (front desk), scheduler.Scheduler,
#                tokenizer.tokenize_worker.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


# ELI5: this is what ONE scheduler worker does with its life.
# WHAT: it runs INSIDE a fresh child process (one per GPU). `args` carries this worker's rank.
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    # WHY imports inside the function: the heavy imports (torch, the scheduler) happen only in
    #     the child process that needs them, after it has started. With the "spawn" start method
    #     the child begins as a blank interpreter anyway.
    import torch
    from minisgl.scheduler import Scheduler

    # WHAT: no autograd bookkeeping for everything below (this is inference).
    with torch.inference_mode():
        # ELI5: slow part. Loads weights onto the GPU, reserves the KV pantry, records CUDA graphs.
        scheduler = Scheduler(args)
        # WHAT: wait for all GPU workers to finish loading, so everyone starts together.
        scheduler.sync_all_ranks()

        # WHAT: only rank 0 raises the "ready" flag (the launcher expects exactly one from the
        #       scheduler group).
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        # WHAT: in shell mode, hide INFO logs so they do not clutter the chat.
        if args.silent_output:
            logging.disable(logging.INFO)

        # ELI5: from now on this process just loops forever: schedule, cook, report.
        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


# ELI5: the manager's morning routine.
# WHAT: run_shell=True is used by `python -m minisgl.shell` for an interactive terminal chat
#       instead of an HTTP server.
def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    # WHAT: turn command-line flags into a frozen ServerArgs object (see args.py).
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    # ELI5: a recipe for "hire everybody and wait until they are ready". It is handed to the API
    #       server, which calls it at the right moment (after the front desk's own phone lines
    #       exist, but before the web server starts accepting customers).
    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # WHY "spawn": each child starts as a brand-new Python interpreter instead of a copy of
        #     this one. Copying a process that has touched CUDA ("fork") is unsafe, so spawn is
        #     the standard choice with GPUs. force=True overrides any earlier setting.
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        # ELI5: the "ready flag" mailbox every hire drops a note into when they are set up.
        ack_queue: mp.Queue[str] = mp.Queue()

        # --- hire one scheduler (head waiter + chef) PER GPU ---------------------------------
        for i in range(world_size):
            # WHAT: copy the settings, but with this worker's own rank (0..world_size-1).
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            # GOTCHA: daemon=False means these children are NOT killed automatically when the
            #         parent exits normally; shutdown has to be handled explicitly.
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        # --- hire the card-reader (and, by default, card-maker too) --------------------------
        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        # WHAT: `addr` is the in-tray it reads. With the default shared mode (num_tokenizer == 0)
        #       that in-tray is the detokenizer address, so TokenizeMsg from the front desk and
        #       DetokenizeMsg from the scheduler BOTH arrive here, and this ONE worker handles
        #       both jobs. `create=tokenizer_create_addr` is True in this shared mode, so this
        #       worker owns (binds) that address. (In separate mode it is False: the scheduler
        #       owns the detokenizer's in-tray and this worker dials in.)
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # --- optional extra card-makers (only if --num-tokenizer N with N > 0) --------------
        # WHAT: they all read the SAME in-tray (zmq_tokenizer_addr, owned by the front desk).
        #       A PUSH socket hands messages out round-robin to every connected PULL socket,
        #       so the N tokenizers automatically share the load.
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        # ELI5: stand at the door and count raised flags. Only when all have arrived do we return,
        #       and only then does the web server start taking customers.
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # WHAT: creates the front desk's phone lines, calls start_subprocess(), then serves HTTP
    #       (or runs the interactive shell).
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
