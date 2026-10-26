# =====================================================================================
# FILE MAP -- server/args.py
# =====================================================================================
# WHAT THIS FILE IS: the command-line flags (the "--model", "--port" ... you type) and the
#                    ServerArgs object that carries all the settings through the program.
#
# ELI5: Before opening, the manager reads a settings sheet filled in by whoever started the
#       program. This file defines the SHEET (ServerArgs), the QUESTIONS printed on it
#       (argparse flags), and the rules that clean up the answers (parse_args).
#
# LAYERS OF SETTINGS (each inherits the previous one):
#       EngineConfig    (engine/config.py)     model, dtype, memory, pages, attention backend ...
#         -> SchedulerConfig (scheduler/config.py) + chunk size, cache type, ZeroMQ addresses
#           -> ServerArgs (this file)            + host, port, number of tokenizers
#
# WHO CALLS IT: server/launch.py (parse_args), server/api_server.py (reads the fields).
# WHAT IT CALLS: argparse, torch (for dtypes), and a few validators from other packages.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    # WHAT: which network interface the HTTP server listens on. 127.0.0.1 = only this machine.
    #       Use 0.0.0.0 to accept connections from other machines (think about security first).
    server_host: str = "127.0.0.1"
    # WHAT: the HTTP port. NOTE the default is 1919, not 30000 (the SGLang lesson's port).
    server_port: int = 1919
    # WHAT: how many EXTRA tokenizer processes to start. 0 = none: the detokenizer worker also
    #       tokenizes ("shared tokenizer").
    num_tokenizer: int = 0
    # WHAT: hide INFO logs (used by the interactive shell).
    silent_output: bool = False

    # WHAT: True when there is no separate tokenizer process.
    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    # ELI5: the phone line carrying finished text from the card-reader to the front desk.
    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    # ELI5: the in-tray the front desk sends TokenizeMsg to.
    # WHAT: shared mode -> the SAME address as the detokenizer's in-tray (one worker reads both).
    #       separate mode -> its own address, minisgl_4.
    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    # WHAT: should a tokenize worker BIND (own) its in-tray address?
    #       Shared mode (default): YES. The one worker owns the in-tray and everybody else
    #       (the front desk, the scheduler) dials in to it.
    #       Separate mode: NO for every tokenize worker. Instead the front desk owns the
    #       tokenizers' in-tray (frontend_create_tokenizer_link) and the scheduler owns the
    #       detokenizer's in-tray (backend_create_detokenizer_link).
    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    # WHAT: should the SCHEDULER bind the scheduler -> detokenizer line? Only in separate mode.
    #       (In shared mode the worker already owns it, so the scheduler dials in.)
    #       This overrides SchedulerConfig's default of True.
    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    # WHAT: should the FRONT DESK bind its line towards the tokenizer? The mirror image: only in
    #       separate mode (in shared mode the worker owns it).
    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    # WHAT: the address the GPU workers use to find each other for torch.distributed.
    #       Overrides EngineConfig's fixed port 2333 with (HTTP port + 1), so two servers on
    #       one machine with different --port do not collide.
    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


# ELI5: read the settings sheet, tidy the answers, and return a ServerArgs.
def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    # WHY imported here: these packages pull in heavy dependencies; importing only when the
    #     flags are parsed keeps `import minisgl.server.args` light.
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    # --- which model -----------------------------------------------------------------
    # WHAT: a Hugging Face repo id (downloaded on first use) or a local folder. Required.
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    # WHAT: the number format of weights and math: float16/bfloat16/float32, or "auto" = follow
    #       what the model was published in (see the resolution further down).
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    # WHAT: tensor parallelism: how many GPUs one copy of the model is split across
    #       (also the number of scheduler processes started).
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    # WHAT: the size of the head waiter's peg board = the most requests running at once.
    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    # WHAT: overrides the longest allowed conversation (prompt + answer). Default comes from
    #       the model's own maximum position.
    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    # ELI5: "How much of the free GPU memory may the pantry (KV cache) take?" 0.9 = 90%.
    # WHY not 100%: leave room for temporary activations, CUDA graphs and fragmentation.
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    # WHAT: skip loading real weights; use random numbers. Handy to test speed/memory without a
    #       download. The assert guards that the default really is False.
    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    # WHAT: use PyNCCL (mini-sglang's own thin NCCL wrapper) for GPU-to-GPU communication.
    #       This flag turns it OFF (store_false). Default is on.
    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    # --- where to listen --------------------------------------------------------------
    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    # WHAT: the largest batch size for which a CUDA graph is recorded. None = pick
    #       automatically from free GPU memory. 0 would disable graphs (per docs/features.md).
    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    # WHAT: see ServerArgs.num_tokenizer.
    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    # ELI5: "How many words of one order may the chef read in a single pass?" Longer prompts are
    #       cut into pieces of at most this size (chunked prefill). Default 8192.
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    # WHAT: set the number of KV pages by hand instead of computing it from free memory.
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    # WHAT: how many token slots per KV page. 1 = finest granularity (default); larger pages
    #       waste a little at the end of each request but are friendlier to some attention kernels.
    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    # WHAT: which attention kernels to use. "fa,fi" = FlashAttention for prefill, FlashInfer for
    #       decode (a hybrid). validate_attn_backend rejects unknown names early.
    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    # WHAT: where to download the model from. Not stored in ServerArgs (deleted below).
    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    # WHAT: "radix" (shares common prompt beginnings between requests) or "naive" (no sharing).
    #       The choices come from the registry, so adding a new cache type adds a valid choice.
    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    # WHAT: interactive terminal chat instead of an HTTP server.
    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    # WHAT: turn the flags into a plain dict we can adjust before building ServerArgs.
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    # WHAT: shell mode (from the flag OR from python -m minisgl.shell) shrinks the setup to one
    #       request at a time with quiet logs: a personal chat, not a busy server.
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    # WHAT: expand "~" in a local path.
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # WHAT: with ModelScope as the source, download the files (unless a local folder was given)
    #       and replace the repo id with the local folder. With --dummy-weight, skip the big
    #       weight files and fetch only config/tokenizer files.
    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # WHAT: "auto" dtype = read the dtype the model's own config.json declares.
    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    # WHAT: convert the dtype NAME to the real torch dtype object.
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    # WHAT: this is rank 0's view; launch.py makes one copy per GPU with the right rank.
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
