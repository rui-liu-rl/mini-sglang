# =====================================================================================
# FILE MAP -- scheduler/scheduler.py
# =====================================================================================
# WHAT THIS FILE IS: THE HEAD WAITER. The endless loop that decides, every few milliseconds,
#                    "which orders get cooked in the next pass?", hands that tray to the chef
#                    (the Engine), and then tidies up with the results.
#
# ELI5: Picture the head waiter standing at the pass between the dining room and the kitchen.
#       Over and over, forever:
#         1. collect any new order slips from the mailbox                      (receive_msg)
#         2. decide the next tray: fresh orders to READ (prefill), else        (_schedule_next_batch)
#            orders underway to WRITE one more word (decode)
#         3. give the tray to the chef                                         (_forward)
#         4. read the dishes that came back, tell the card-reader the new
#            words, cross finished orders off and give their pantry space back (_process_last_data)
#
# THE BIG IDEA (OVERLAP SCHEDULING). The chef (GPU) is expensive; the waiter (CPU) is slow at
# paperwork. A naive waiter would wait for the chef to finish before doing step 4 and step 2 of
# the NEXT pass, leaving the chef idle. Instead, the real loop does:
#
#     CPU:  schedule(n)  launch(n)  process results(n-1) ......  schedule(n+1) launch(n+1) ...
#     GPU:               [============ run batch n ============]               [=== run n+1 ===]
#                                    ^ the CPU's paperwork about the PREVIOUS batch hides here
#
# That works because the next batch's input tokens live on the GPU (token_pool), where the
# previous step's sampled tokens were already written by the GPU. The CPU never has to read
# them before scheduling the next step.
#
# TWO CUDA "LANES" (streams). A GPU runs work in lanes; work in the same lane runs in order,
# work in different lanes may overlap. This file uses two:
#       self.stream        : the scheduler's lane, for small bookkeeping copies (positions, indices)
#       self.engine.stream : the engine's lane, for the heavy model forward pass
#   `engine.stream.wait_stream(self.stream)` = "engine lane: don't start until the bookkeeping
#   lane has finished everything queued so far."
#
# OBJECTS THE SCHEDULER OWNS
#   engine           the chef (engine/engine.py)
#   table_manager    the peg board: one numbered row per running order (scheduler/table.py)
#   cache_manager    the pantry clerk: hands out/takes back KV pages, talks to the prefix cache
#   prefill_manager  the waiting clipboard + admission rules for NEW orders (scheduler/prefill.py)
#   decode_manager   the "tables still eating" clipboard (scheduler/decode.py)
#   (inherited)      SchedulerIOMixin: how messages come in/out (scheduler/io.py)
#
# WHO CALLS IT: server/launch.py (_run_scheduler) and llm/llm.py (LLM inherits from it).
# WHAT IT CALLS: everything above, plus tokenizer loading (to learn the EOS token id).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

# WHAT: a pair of index tensors (a, b) used for "gather/scatter at (a[i], b[i])" on a 2-D table.
Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
# ELI5: everything the chef needs for ONE pass, bundled up. It is saved (not rebuilt later)
#       because under overlap scheduling the pass is launched now but its results are read one
#       iteration later, and by then the live data structures have moved on. ("IMA" = illegal
#       memory access, a GPU crash from touching memory that has since been reused.)
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    # WHAT: (token_mapping, positions). To read the input ids: token_pool[token_mapping, positions].
    #       token_mapping[i] is the peg (table row) that token i belongs to.
    input_tuple: Indice2D  # (token_mapping, positions)
    # WHAT: (req_mapping, seq_lens or -1). After sampling, request j's new token is written to
    #       token_pool[req_mapping[j], seq_lens[j]], i.e. at the position right after its current
    #       end. The value -1 is a sentinel for "this request produces no real token this pass"
    #       (for example a prompt chunk): the write goes to the throw-away last column.
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        # ELI5: hire the chef. This loads the model onto the GPU, reserves the KV pantry,
        #       and records CUDA graphs (slow, so it happens once at startup).
        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        # ELI5: open a second lane on the GPU for bookkeeping, and tell this thread to use it.
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        # WHAT: the peg board shares the engine's page_table (one row per running request).
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # WHAT: the pantry clerk knows how many KV pages exist, the page size, the same page table,
        #       and which prefix-cache flavour to use ("radix" or "naive").
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        # WHAT: requests that were just freed. Needed because under overlap scheduling the same
        #       finished request can show up in two consecutive batches (see _process_last_data).
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        # WHAT: the "I'm done" token id for this model.
        self.eos_token_id = self.tokenizer.eos_token_id
        # SHAPE: (max_running_req + 1, aligned_max_seq_len), int32. token_pool[row, position] = token id.
        self.token_pool = self.table_manager.token_pool
        # WHAT: the most prompt tokens one prefill pass may process (default 8192).
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        # ELI5: now set up the mailbox (ZeroMQ queues), or the offline replacements.
        super().__init__(config, self.engine.tp_cpu_group)

    # WHAT: called by the I/O mixin right before it BLOCKS waiting for a message, i.e. when the
    #       kitchen has nothing to do. A good moment for a sanity check of the pantry books.
    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    # ELI5: ONE turn of the real (overlapping) loop. See the timeline at the top of the file.
    # WHAT: `last_data` is the previous turn's (input, output) bundle, or None. It returns THIS
    #       turn's bundle, to be processed on the next turn.
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # WHAT: only sleep waiting for mail when there is truly nothing else to do: no previous
        #       batch to process, nobody waiting to prefill, nobody decoding.
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 1. collect and file new mail
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 2. choose the next tray and prepare all its index tensors (CPU + small GPU copies)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # 3. LAUNCH the pass on the engine's lane. This returns quickly: the GPU work is
            #    only queued. wait_stream makes the engine lane wait for the bookkeeping copies.
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 4. WHILE THE GPU IS BUSY with batch n, process the results of batch n-1.
        self._process_last_data(last_data)
        return ongoing_data

    # ELI5: the plain, easy-to-read loop (no overlap). Used only when
    #       MINISGL_DISABLE_OVERLAP_SCHEDULING is set. Each turn finishes its own results
    #       before starting the next batch, so the GPU idles during the paperwork.
    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    # WHAT: the process's main loop, forever. inference_mode = no gradient bookkeeping.
    # WHY the stream assert: overlap_loop assumes the current CUDA stream is the scheduler's own
    #     (set in __init__); this catches an accidental stream switch.
    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    # WHAT: orderly shutdown: finish pending GPU work, make every GPU worker arrive here
    #       together, then tear down the engine (CUDA graphs and communicators).
    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    # ELI5: "READ THE DISHES THAT CAME BACK." For every order that was on the previous tray:
    #       write down the new word, decide if the order is finished, tell the card-reader,
    #       and if finished, hand its peg and pantry pages back.
    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        # WHAT: unpack: the tray that was cooked, and its output (new tokens on the GPU, the same
        #       tokens copied to the CPU, and an event that fires when that copy is complete).
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # WHAT: CPU waits (only for that specific copy) until the tokens have really arrived.
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        # WHAT: inside this block every page-freeing call is collected and applied ONCE at the
        #       end (one torch.cat instead of many small ones). See CacheManager.lazy_free_region.
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                # WHAT: a prompt chunk produced no real token (its prompt isn't finished yet).
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                # WHAT: keep the CPU-side copy of the sequence in sync (the prefix cache will use
                #       these ids as keys).
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # WHAT: finished = out of allowed words, OR wrote the end-of-sequence token
                #       (unless this request asked to ignore EOS).
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                # WHY: batch n was scheduled BEFORE we knew batch n-1 finished this request, so the
                #      request can be in batch n as well; when batch n's results arrive here, it
                #      "finishes" a second time. `self.finished_reqs` (the set built on the previous
                #      call) remembers who was just freed so we don't free it again.
                if finished and req not in self.finished_reqs:
                    # ELI5: cross the order off the "still eating" list, return its peg, and
                    #       save its words in the prefix cache for future customers.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    # ELI5: the prompt has just been read; write it into the shared index NOW so
                    #       the next customer with the same beginning can reuse these notes.
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # WHAT: send all new tokens to the card-reader in one batch (a no-op for offline mode,
        #       see llm/llm.py).
        self.send_result(reply)

    # ELI5: "WHAT DOES THIS SLIP SAY?" Handles each kind of incoming message.
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        # WHAT: a Batch wrapper just contains more messages: handle each one.
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        # WHAT: "close the restaurant". KeyboardInterrupt makes _run_scheduler shut down cleanly.
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        # WHAT: a NEW ORDER.
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            # GOTCHA: a prompt that already fills (or exceeds) the whole allowed length is dropped
            #         and NO reply is ever sent, so an HTTP client would wait forever.
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            # WHAT: ask for more words than fit? Quietly shrink the request to what fits.
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # ELI5: put the slip on the waiting clipboard.
            self.prefill_manager.add_one_req(msg)
        # WHAT: a CANCEL. Look for the order on the waiting clipboard first, then among the tables
        #       still eating; free whatever it was holding.
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    # WHAT: give back everything a finished or cancelled request held.
    #       1) its numbered peg (table row)
    #       2) its KV pages: cache_req(finished=True) inserts the useful part (prompt + answer)
    #          into the prefix cache for reuse and frees the rest.
    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    # ELI5: "SET THE TRAY." Everything that must be ready before the chef starts.
    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # 1. pad the tray with dummy slips up to a size a CUDA graph was recorded for
        self.engine.graph_runner.pad_batch(batch)
        # 2. give every order pantry pages for the words it is about to write
        self.cache_manager.allocate_paged(batch.reqs)
        # 3. seat numbers (positions) of each word, for RoPE
        batch.positions = _make_positions(batch, self.device)
        # 4. two index lists: where to READ input ids, where to WRITE the new tokens
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        # 5. for each word, the pantry slot where its new K/V will be written
        batch.out_loc = self.engine.page_table[input_mapping]
        # 6. extra paperwork for the attention kernels (page lists, lengths, ...)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            # 7. per-request temperature / top-k / top-p tensors for the sampler
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    # ELI5: "WHICH TRAY NEXT?" Fresh orders to read, else orders to continue.
    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        # WHAT: `A or B` = "A if A is not None, else B". Prefill gets priority: new customers
        #       start sooner (lower TTFT) at the cost of briefly pausing decode for others.
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    # ELI5: "HAND THE TRAY TO THE CHEF."
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        # WHAT: gather this pass's input token ids from the GPU token pool.
        # SHAPE: 1-D, one id per token being processed (all requests concatenated).
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        # WHAT: scatter the freshly sampled tokens back into the pool at (row, next position),
        #       ready to be read as input on the next pass. Done on the GPU: no CPU round trip.
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # WHAT: these requests (those that can still decode) now count as "still eating".
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


# ELI5 for the next three helpers: they build the INDEX LISTS for a tray. Worked example:
#   request A: table row 3, cached_len 0, device_len 5   (prefill, extend_len 5)
#   request B: table row 7, cached_len 2, device_len 6   (prefill, extend_len 4)
#     positions     = [0,1,2,3,4,  2,3,4,5]        (_make_positions)
#     token rows    = [3,3,3,3,3,  7,7,7,7]        (first half of _make_input_tuple)
#     write tuple   = ([3, 7], [5, 6])             (_make_write_tuple: row and device_len)
# WHAT (this one): the position of every token to be computed, request after request.
# SHAPE: 1-D int32, length = sum of extend_len over padded_reqs (dummy slips included).
# WHY pin_memory=True + non_blocking=True: the list is built in page-locked CPU memory, which the
#     GPU can fetch asynchronously, so the copy overlaps with other work instead of stalling.
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        # WHAT: fill this request's slice with cached_len, cached_len+1, ..., device_len-1.
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


# WHAT: the READ list: for each token, (which row of the tables it belongs to, its position).
#       With it, token_pool[input_tuple] fetches input ids and page_table[input_tuple] fetches
#       the KV slot where each token's new K/V must be stored.
def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


# WHAT: the WRITE list: for each REAL request (padding excluded), (its row, the position right
#       after its current end). The sampled token for that request goes to that cell.
#       If the request can't decode any more the position is -1 (a throw-away cell).
def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
