# =====================================================================================
# FILE MAP -- scheduler/io.py
# =====================================================================================
# WHAT THIS FILE IS: the head waiter's MAILBOX: how requests arrive and results leave. It is a
#                    "mixin" (a bundle of methods) that Scheduler inherits.
#
# ELI5: Think of two mail slots on the kitchen wall:
#         IN  slot : new order slips and cancels from the card-maker
#         OUT slot : the new word for each order, to the card-reader
#       With ONE GPU this is trivial. With SEVERAL GPUs there is a catch: every GPU's head
#       waiter must see EXACTLY the same orders in EXACTLY the same sequence. So only waiter #0
#       reads the real mailbox and then ANNOUNCES each slip over a loudspeaker to the others.
#
#       Roles by GPU rank (tp_info):
#         rank 0   : reads the real mailbox, announces to the others, sends the results out
#         rank 1.. : listens to rank 0's announcements, sends nothing out
#
# LINES USED (ZeroMQ, see utils/mp.py and scheduler/config.py):
#         _recv_from_tokenizer  PULL  : from the card-maker                (rank 0 only)
#         _send_into_tokenizer  PUSH  : to the card-reader                 (rank 0 only)
#         _send_into_ranks      PUB   : rank 0 announces to the others     (rank 0, TP > 1)
#         _recv_from_rank0      SUB   : the others listen                  (rank 1.., TP > 1)
#
# OFFLINE MODE: when used as a library (llm/llm.py) there is no mailbox at all; LLM replaces
#       receive_msg / send_result with plain Python methods.
#
# WHO CALLS IT: scheduler/scheduler.py (inherits it) and llm/llm.py (overrides the offline_*).
# WHAT IT CALLS: utils/mp.py queues, message forms, torch.distributed (a CPU process group).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        # WHAT: a CPU-side (gloo) process group that spans all GPU workers. Used for tiny
        #       control messages ("how many slips?") and barriers, not for tensors.
        self.tp_cpu_group: Final = tp_cpu_group
        # ELI5: library mode: swap in the do-it-in-Python methods (defined in llm/llm.py) and
        #       leave early; no sockets are needed.
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        # WHAT: only rank 0 touches the real mailbox.
        if tp_info.is_primary():
            # WHAT: this scheduler OWNS (binds) the in-tray for new requests: create=True.
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # WHAT: the out-tray towards the card-reader. Whether this side binds or dials in
            #       depends on the tokenizer mode (see server/args.py).
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # WHAT: the default (single GPU) behaviours; replaced below if there are several GPUs.
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                # ELI5: rank 0 gets the loudspeaker.
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # ELI5: everyone else gets a headset tuned to rank 0, and sends nothing out.
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # WHAT: from now on scheduler.py just calls self.receive_msg(...) / self.send_result(...)
        #       and gets whichever behaviour fits this process.
        self.receive_msg = recv
        self.send_result = send

    # WHAT: implemented by Scheduler (idle housekeeping hook).
    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    # WHAT: implemented by llm/llm.py for library mode.
    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    # ELI5: "Everybody stop at this line until all GPU workers have arrived." A barrier.
    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    # ---------------------------------------------------------------------------------
    # ONE GPU: simply read the mailbox.
    # ---------------------------------------------------------------------------------
    # WHAT: blocking=True means "nothing else to do, so SLEEP until at least one slip arrives"
    #       (run_when_idle is called first). Then, either way, also take every slip that is
    #       ALREADY waiting without sleeping (the `empty()` peek).
    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    # ---------------------------------------------------------------------------------
    # SEVERAL GPUs, rank 0: read the mailbox AND announce every slip to the other ranks.
    # ---------------------------------------------------------------------------------
    # WHY get_raw / put_raw: rank 0 forwards the exact bytes it received. No decode and re-encode.
    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        # ELI5: scoop up everything that has already arrived, without waiting for more.
        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # ELI5: first tell everybody "this round there are N more slips". This count travels over
        #       a reliable collective (not the loudspeaker), so every rank knows exactly how
        #       many slips to wait for. Without it, a rank could read too few or too many slips
        #       and the GPU workers would drift out of sync.
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # ELI5: now announce each slip, and also process it ourselves.
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    # ---------------------------------------------------------------------------------
    # SEVERAL GPUs, rank 1 and up: listen to rank 0.
    # ---------------------------------------------------------------------------------
    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        # WHAT: if rank 0 blocked on the mailbox, so do we: it announces the first slip.
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # WHAT: receive the count rank 0 broadcast (-1 is just a placeholder that gets overwritten).
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        # WHAT: then read exactly that many announcements.
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    # ---------------------------------------------------------------------------------
    # SENDING RESULTS
    # ---------------------------------------------------------------------------------
    # WHAT (rank 0): one new token for each request that was on the tray. A single message is sent
    #       as is; several are bundled into one BatchTokenizerMsg (one packet instead of many).
    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    # WHAT (rank 1 and up): do nothing. Every rank computed the same tokens; only rank 0 reports.
    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
