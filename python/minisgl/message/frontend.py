# =====================================================================================
# FILE MAP -- message/frontend.py
# =====================================================================================
# WHAT THIS FILE IS: the forms that travel FROM the card-reader (detokenizer) BACK TO the front
#                    desk (API server).
#
# ELI5: The finished words are carried to the front desk on a small slip: "ticket 7, here are
#       the next few letters, and (yes/no) that's the end of the answer."
#
# WHO USES IT: tokenizer/server.py creates UserReply; server/api_server.py reads it.
# WHAT IT CALLS: message/utils.py for encoding and decoding.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


# ELI5: the parent form. Its encoder/decoder turn ANY frontend form into bytes and back.
# WHY decoder uses globals(): the decoder looks up the class named in the incoming packet in
#     THIS file's names. So the packet can say "UserReply" and we know which class to build.
@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


# WHAT: a bundle of several replies sent in ONE packet (cheaper than many tiny packets).
@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


# ELI5: "Ticket `uid`: here are the next letters. finished=True means that was the last bit."
@dataclass
class UserReply(BaseFrontendMsg):
    # WHAT: the request id, so the front desk finds the right waiting customer.
    uid: int
    # WHAT: only the NEW text since the last reply (not the whole answer so far).
    incremental_output: str
    # WHAT: True on the final reply; the front desk then closes the stream with [DONE].
    finished: bool
