# =====================================================================================
# FILE MAP -- message/backend.py
# =====================================================================================
# WHAT THIS FILE IS: the forms that travel FROM the card-maker (tokenizer) TO the head waiter
#                    (scheduler). "Backend" here means the scheduler + engine side.
#
# ELI5: The card-maker hands the head waiter a slip: "ticket 7: here are the numbered cards, and
#       the customer's adventurousness card." Or: "ticket 7 left, cancel it." Or: "close the
#       restaurant."
#
# WHO USES IT: tokenizer/server.py creates them; scheduler/scheduler.py reads them.
# WHAT IT CALLS: message/utils.py (encode/decode), core.SamplingParams (a field type).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


# WHY SamplingParams is imported here (and not only under TYPE_CHECKING): the decoder below
#     looks classes up by NAME in this file's globals(), so SamplingParams must be a real,
#     importable name here or a UserMsg could not be rebuilt.
@dataclass
class BaseBackendMsg:
    # GOTCHA: unlike the frontend/tokenizer forms, encoder is an INSTANCE method here
    #         (msg.encoder()), while decoder is a static method. ZmqPubQueue and ZmqPushQueue
    #         are given BaseBackendMsg.encoder, which works because calling an instance method
    #         through the class and passing the object works the same way.
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


# WHAT: several backend messages in one packet.
@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


# WHAT: "stop the scheduler loop". The scheduler turns it into KeyboardInterrupt, which makes it
#       shut down cleanly.
@dataclass
class ExitMsg(BaseBackendMsg):
    pass


# ELI5: a NEW ORDER, ready to cook: ticket number, the numbered cards, the adventurousness card.
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    # SHAPE: 1-D int32 tensor on the CPU, the whole prompt as token ids.
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


# ELI5: "Ticket `uid` walked out. Stop cooking it and give back its pantry space."
@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
