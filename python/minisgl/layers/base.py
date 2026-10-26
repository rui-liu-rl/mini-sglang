# =====================================================================================
# FILE MAP -- layers/base.py
# =====================================================================================
# WHAT THIS FILE IS: mini-sglang's own tiny replacement for torch.nn.Module: "BaseOP", plus two
#                    variants (StateLessOP, OPList).
#
# ELI5: Every building block of the model (a linear layer, a norm, a whole decoder layer) is an
#       "OP": a box that has some NUMBERS inside (weights) and a forward() that uses them. Boxes
#       can contain other boxes, like Russian dolls. This file teaches boxes how to
#         state_dict()      : write a flat list of all the numbers inside, each labelled with its
#                             nested name, e.g. "model.layers.3.self_attn.qkv_proj.weight"
#         load_state_dict() : the reverse: receive such a list and put each tensor in its place.
#
# WHY NOT torch.nn.Module? Simplicity and control. Here a weight is just a plain tensor attribute
#     (self.weight = torch.empty(...)). Loading a checkpoint simply REPLACES that attribute with
#     the real tensor (setattr). There are no parameters registry, hooks or autograd, none of
#     which inference needs.
#
# HOW THE NAMES WORK: any public attribute (name does not start with "_") that is a tensor is a
#       weight; any public attribute that is another BaseOP is a child to recurse into. Names are
#       joined with dots, matching the Hugging Face checkpoint key names.
#
# WHO USES IT: every file in layers/ and models/.
# WHAT IT CALLS: torch only.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# WHAT: {"dotted.name": tensor}
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


# WHAT: join a prefix and a name with a dot, but don't start with a stray dot.
def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    # WHAT: every OP must say what it computes. (BaseOP is not an ABC, so this is a convention
    #       rather than something Python enforces.)
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    # ELI5: "List every number you hold, with its full nested name."
    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            # GOTCHA: attributes starting with "_" (like _comm, _layer_id) are private and skipped.
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    # ELI5: "Here is a labelled heap of tensors. For each number-slot you own, pop the matching
    #       tensor from the heap and put it in."
    # WHAT: for each tensor attribute, take (pop) its entry from `state_dict`, check shape AND
    #       dtype match what the skeleton expected, and REPLACE the attribute (no copy). Children
    #       recurse with a longer prefix. At the top level, if anything is left in the heap, the
    #       checkpoint had a tensor the model doesn't know about: raise an error.
    # WHY pop: so "leftover keys" is detectable.
    # NOTE: a weight created on the "meta" device (no memory) has the right shape/dtype but no
    #       data; this is where it gets real data.
    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


# ELI5: a box with NO numbers inside (a pure calculation like rotary embedding). Loading and
#       listing do nothing, except that the top-level leftover check still applies.
class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


# ELI5: a numbered row of identical-shaped boxes, like the decoder layers 0..N-1.
# WHAT: its children are named by index, so layer 3's weights are "...layers.3.self_attn..."
#       exactly as in Hugging Face checkpoints. The Python list lives in `op_list`.
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
