# =====================================================================================
# FILE MAP -- utils/misc.py
# =====================================================================================
# WHAT THIS FILE IS: a drawer of small arithmetic helpers used everywhere (divide evenly, round
#                    up, round down) plus two odd little tools.
#
# ELI5: Kitchen measuring spoons. Nothing clever, but you reach for them constantly:
#       "how many pages do 13 tokens need if a page holds 4?"  -> div_ceil(13, 4) = 4
#       "how many heads per GPU if 16 heads share 2 GPUs?"      -> div_even(16, 2) = 8
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations


# WHAT: a decorator for demo/test code at the bottom of a script. It runs the decorated function
#       immediately ONLY when the file is run as a script (name == "__main__"), and leaves the
#       function alone when the file is merely imported.
# GOTCHA: with discard=True (the default when run as main) the function is run and then the NAME is
#         rebound to None, so you cannot call it again afterwards. Rarely used in this repo.
def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


# ELI5: "Split `a` things evenly among `b` helpers, and COMPLAIN if it doesn't split evenly."
# WHAT: returns a // b after asserting b divides a. Used to split heads or matrix sizes across
#       GPUs: a model with 16 query heads on 4 GPUs gives 4 heads per GPU.
# WHY allow_replicate: key/value heads can be FEWER than GPUs (e.g. 2 KV heads on 8 GPUs). Then
#     each GPU keeps ONE copy of a KV head (several GPUs hold the same one). That is allowed only
#     if the GPU count is a multiple of the head count; the answer is 1 head per GPU.
def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


# WHAT: ceiling division using only integers. div_ceil(13, 4) = 4, div_ceil(12, 4) = 3.
def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


# WHAT: round `a` UP to a multiple of `b`. align_ceil(13, 4) = 16.
def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


# WHAT: round `a` DOWN to a multiple of `b`. align_down(13, 4) = 12. The radix cache uses it so
#       only WHOLE pages are inserted into the tree.
def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


# ELI5: a special "nothing was given" marker, different from None (which can be a real value).
class Unset:
    pass


UNSET = Unset()
