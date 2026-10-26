# FILE MAP -- moe/__init__.py
# WHAT: registers the available MoE backends by name (only "fused" today) and builds one on
#       request, using the same Registry pattern as attention/__init__.py and kvcache/__init__.py.
# Tag legend: see zero-to-sglang/eli5/README.md
from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


# WHAT: a type description: "a function with no arguments that returns a backend".
class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
