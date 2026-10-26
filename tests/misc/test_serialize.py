# =====================================================================================
# FILE MAP -- tests/misc/test_serialize.py   (a script; CPU only)
# =====================================================================================
# WHAT THIS FILE TESTS: the message packing code (message/utils.py: serialize_type /
#                       deserialize_type) that turns the forms sent between processes into bytes-
#                       friendly dictionaries and back (see message/*.py).
#
# ELI5: Two round trips. (1) A made-up nested record A (an int, a string, a list of more A's, and
#       a tensor) is packed and unpacked; the log shows both forms so you can eyeball that they match.
#       (2) A real BatchBackendMsg holding one UserMsg is encoded and decoded the way the sockets do it.
#       Only 1-D tensors are supported by the packer (all the messages carry token-id lists).
#
# NOTE: it logs the results instead of asserting equality, so it is a demonstration/smoke test rather
#       than a strict check.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from minisgl.core import SamplingParams
import torch
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


@call_if_main()
def test_serialize_deserialize():

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)

    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)
