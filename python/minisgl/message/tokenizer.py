# =====================================================================================
# FILE MAP -- message/tokenizer.py
# =====================================================================================
# WHAT THIS FILE IS: the forms that travel INTO the tokenizer worker (the process that makes
#                    cards from words AND words from cards).
#
# ELI5: Three kinds of slips land on the card-maker's desk:
#         TokenizeMsg   : from the front desk, "please make cards from this text"
#         AbortMsg      : from the front desk, "ticket 7 left"
#         DetokenizeMsg : from the head waiter, "ticket 7 produced this new card"
#
# WHO USES IT: server/api_server.py creates TokenizeMsg and AbortMsg; scheduler/scheduler.py
#              creates DetokenizeMsg; tokenizer/server.py reads all three.
# WHAT IT CALLS: message/utils.py (encode/decode), core.SamplingParams (a field type).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


# WHY one "tokenizer" form family handles both directions: in the default setup a single worker
#     does tokenizing AND detokenizing, and both kinds of work arrive on the same phone line.
@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


# WHAT: several tokenizer-side messages in one packet. The scheduler uses this when more than
#       one request produced a token in the same step.
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


# ELI5: "Ticket `uid` has produced one more numbered card. If finished=True, it was the last."
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    # WHAT: ONE token id (a plain Python int), the word the sampler just picked.
    next_token: int
    finished: bool


# ELI5: "Please turn this into numbered cards."
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    # WHAT: either a plain prompt string, or a list of chat messages like
    #       [{"role": "user", "content": "..."}] that the chat template will format.
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


# ELI5: "Ticket `uid` left; drop it."
@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
