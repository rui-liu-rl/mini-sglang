# FILE MAP -- scheduler/__init__.py
# WHAT: the package's front door. Exposes the head waiter (Scheduler) and its settings
#       (SchedulerConfig).
# THE PACKAGE AT A GLANCE (read in this order):
#   scheduler.py : the endless loop (receive -> schedule -> forward -> process results)
#   prefill.py   : waiting clipboard + admission rules for new orders (chunked prefill)
#   decode.py    : the "still eating" set
#   cache.py     : the pantry clerk (free pages, prefix-cache bookkeeping)
#   table.py     : the peg board (one row per running order)
#   io.py        : the mailbox (ZeroMQ in/out, broadcast to other GPU workers)
#   config.py    : settings and phone-line addresses
#   utils.py     : PendingReq (a waiting order) and a small record type
# Tag legend: see zero-to-sglang/eli5/README.md
from .config import SchedulerConfig
from .scheduler import Scheduler

__all__ = ["Scheduler", "SchedulerConfig"]
