# =====================================================================================
# FILE MAP -- env.py
# =====================================================================================
# WHAT THIS FILE IS: a tidy way to read "secret switches" set through ENVIRONMENT VARIABLES.
#
# ELI5: Behind the kitchen door there are a few dials you can twist WITHOUT editing any code:
#       type `MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl ...` and the dial turns.
#       This file declares each dial, its default position, and how to read the typed value.
#
# HOW IT WORKS in three steps:
#   1. EnvVar(default, converter) = one dial with a default and a function that turns the text
#      of the variable into a Python value (int, float, bool, memory size ...).
#   2. EnvClassSingleton lists all the dials as class attributes.
#   3. When ENV is created (bottom of the file) it looks at every dial and, if the real
#      environment has MINISGL_<NAME> set, replaces the default with the converted value.
#
# USAGE ELSEWHERE:  from minisgl.env import ENV;  ENV.DISABLE_OVERLAP_SCHEDULING   or  ENV.X.value
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


# WHAT: the parent type, so the singleton can check "everything I hold is a dial".
class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    # WHAT: default_value = the dial's resting position; fn = how to parse the environment text.
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    # WHAT: called once with the full variable name (e.g. "MINISGL_SHELL_TOP_K").
    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                # GOTCHA: a value that fails to parse is silently ignored (the default stays),
                #         so a typo like MINISGL_SHELL_TOP_K=abc does nothing and says nothing.
                pass

    # GOTCHA: Python requires __bool__ to return a real bool. For boolean dials this is fine
    #         (`if ENV.DISABLE_OVERLAP_SCHEDULING:` works), but `if ENV.SOME_INT:` or an
    #         option that is still None would raise TypeError. Use `.value` for those.
    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


# WHAT: "1", "true", "yes" (any case) mean True; anything else means False.
_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


# WHAT: parse a human memory size like "512M", "2GB" or "1073741824" into a number of bytes.
#       Units are powers of 1024.
def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


# WHAT: every variable must start with this prefix, so they cannot clash with other programs.
MINISGL_ENV_PREFIX = "MINISGL_"
# WHAT: ready-made dial types: partial() pre-fills the converter so you only give the default.
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
# WHAT: a THREE-way switch: None (not set; let the code decide), True or False.
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


# ELI5: the control panel. Exactly one panel exists in the whole program (a "singleton").
class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    # WHAT: sampling settings for the interactive shell (python -m minisgl.shell).
    #       MINISGL_SHELL_MAX_TOKENS, ..._TOP_K, ..._TOP_P, ..._TEMPERATURE
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TOP_K = EnvInt(-1)
    SHELL_TOP_P = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    # WHAT: force FlashInfer to use (True) or not use (False) Tensor Cores in its decode
    #       kernels. None = let attention/fi.py decide. Its default rule: use Tensor Cores when
    #       4 or more query heads share one K/V head (num_qo_heads // num_kv_heads >= 4),
    #       because then there is enough matrix-shaped work to feed them.
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    # WHAT: set to 1 to use the plain scheduler loop instead of the overlap loop (handy for
    #       debugging: easier to reason about, but the GPU idles during CPU bookkeeping).
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # WHAT: an UPPER CAP (in bytes) on the buffer PyNCCL reserves for GPU-to-GPU transfers. The
    #       engine asks for max_forward_len * hidden_size * bytes-per-number, and
    #       kernel/pynccl.py takes the smaller of that and this cap. Default 1 GiB; "512M" or
    #       "2G" are accepted spellings.
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    # WHY: __new__ hands back the same object every time, so reading the environment and the
    #      state of every dial exist exactly once.
    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # WHAT: go through every public attribute (every dial) and let it read its environment
    #       variable named MINISGL_<attribute name>.
    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
