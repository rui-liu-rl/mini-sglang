# =====================================================================================
# FILE MAP -- llm/llm.py
# =====================================================================================
# WHAT THIS FILE IS: a tiny "use the kitchen as a library" front door. No HTTP, no separate
#                    processes: you call llm.generate([...]) and get the finished answers.
#
# ELI5: Instead of a restaurant with a front desk and a street, this is a PRIVATE CHEF at your
#       kitchen table. You hand over a stack of order slips (a list of prompts); the chef cooks
#       them all (using the same batching, caching and scheduling as the real restaurant) and
#       hands back a stack of dishes.
#
# HOW IT WORKS: LLM IS-A Scheduler. A Scheduler normally RECEIVES requests from the network and
#       SENDS results to the detokenizer. In offline_mode the scheduler's I/O (scheduler/io.py)
#       is replaced by two methods defined HERE:
#           offline_receive_msg  : pulls the next requests out of a Python list
#           offline_send_result  : collects generated tokens into a Python dict
#       Everything in between (prefill/decode scheduling, KV cache, engine) is the unchanged real thing.
#
# WHO CALLS IT: user code and benchmark scripts (benchmark/offline/*.py).
# WHAT IT CALLS: scheduler.Scheduler (it inherits from it), message forms.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


# WHAT: raised when every request is finished, to break out of the otherwise endless
#       scheduler loop (Scheduler.run_forever never returns by itself).
class RequestAllFinished(Exception):
    pass


# WHAT: the running record of one offline request: its prompt ids and the output ids so far.
@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    # WHAT: builds a one-GPU scheduler+engine inside THIS process.
    # GOTCHA: the default dtype here is bfloat16, while the server's --dtype default is "auto"
    #         (follow the model's own config).
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        # ELI5: the stack of order slips not yet handed to the chef.
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    # WHAT: turn one prompt (a string, or an already-tokenized list of ids) into an int32 tensor.
    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    # ELI5: the replacement for "listen to the phone line". The scheduler asks "any new
    #       orders?" and we hand over slips from our stack, but no more than one prefill pass
    #       can take (prefill_budget tokens), so the stack is fed in as the chef has room.
    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # WHAT: the scheduler only asks in BLOCKING mode when it has nothing else to do. If our
        #       stack is also empty, everything is finished: stop the loop via an exception.
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            # WHAT: ids are handed out 0, 1, 2, ... in the order of the input list, so the
            #       final answers can be returned in the same order.
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    # ELI5: the replacement for "send words to the card-reader". Instead we simply jot each new
    #       token id on the right order's record.
    # WHAT: the final end-of-sequence token is not recorded (same rule as the real detokenizer).
    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    # ELI5: the one button you press. Hand in all your prompts; wait; get all the answers back.
    # SHAPE: returns a list (one entry per prompt, same order) of
    #        {"text": <decoded answer>, "token_ids": <list of generated ids>}.
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        # WHAT: start from a clean slate every call.
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        # WHAT: one SamplingParams for all prompts, or one per prompt.
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        # WHAT: run the real scheduler loop until offline_receive_msg raises RequestAllFinished.
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
