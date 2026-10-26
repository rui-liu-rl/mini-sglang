# =====================================================================================
# FILE MAP -- message/utils.py
# =====================================================================================
# WHAT THIS FILE IS: the "envelope stuffer". It turns a message object (a dataclass) into a
#                    plain dictionary that msgpack can pack into bytes, and back again.
#
# ELI5: The rooms of the Kitchen pass notes through a mail slot, and a mail slot only takes flat
#       paper. A message object is more like a little 3-D model. serialize_type FLATTENS it to a
#       sheet of paper that says "I am a UserMsg: uid=7, input_ids=<bytes>, ...".
#       deserialize_type reads that sheet and BUILDS the 3-D model again on the other side.
#
# WHO USES IT: message/frontend.py, backend.py and tokenizer.py (each class's encoder/decoder).
# WHAT IT CALLS: numpy and torch, only to pack/unpack tensors.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


# WHAT: flatten ANY value. Containers are flattened element by element; plain numbers, strings,
#       None, bool and bytes pass through untouched; anything else is assumed to be a message
#       object or a tensor and goes through serialize_type.
# GOTCHA: a tuple becomes a list after msgpack (msgpack has no tuple type), so the receiver
#         never sees tuples.
def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


# ELI5: "Write down what kind of object this is, then write down every field."
# WHAT: returns a dict with a "__type__" entry (the class name) plus one entry per attribute.
def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    # ELI5: a tensor is written as "a pile of raw bytes + what kind of number they are".
    # SHAPE: only 1-D tensors are supported (see the assert). That is all the engine needs:
    #        token ids are a 1-D int32 tensor.
    # WHY raw bytes: msgpack can carry bytes directly, and numpy's tobytes() is a fast copy.
    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    # WHAT: the class name is the label; every instance attribute becomes a key.
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


# WHAT: the mirror image of _serialize_any. A dict that has "__type__" is a message object (or
#       tensor) to rebuild; a dict without it is just a dict; containers are rebuilt recursively.
def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


# ELI5: "Read the label, find the matching blueprint, and build the object from the fields."
# WHAT: cls_map is a dictionary {class name: class}. Callers pass their own module's globals()
#       (see backend.py), so the lookup finds the classes defined or imported in that file.
# GOTCHA: that means a message can only contain types that are visible in the decoding module's
#         globals. backend.py and tokenizer.py import SamplingParams for exactly this reason.
def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        # WHAT: "torch.int32" -> "int32", which is also numpy's name for the same type.
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # WHY .copy(): frombuffer gives a read-only VIEW of the bytes; copying makes an owned,
        #     writable array before handing it to torch.
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
