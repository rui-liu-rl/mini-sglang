# FILE MAP -- distributed/__init__.py
# WHAT: front door of the distributed package.
#   DistributedInfo / get_tp_info / set_tp_info / try_get_tp_info : "who am I?" (info.py)
#   DistributedCommunicator / enable_pynccl_distributed / destroy_distributed : all_reduce and
#   all_gather between GPU workers (impl.py)
# Tag legend: see zero-to-sglang/eli5/README.md
from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed
from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info

__all__ = [
    "DistributedInfo",
    "get_tp_info",
    "set_tp_info",
    "enable_pynccl_distributed",
    "DistributedCommunicator",
    "try_get_tp_info",
    "destroy_distributed",
]
