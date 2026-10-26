# =====================================================================================
# FILE MAP -- utils/registry.py
# =====================================================================================
# WHAT THIS FILE IS: a tiny "menu board" of named options.
#
# ELI5: Several parts of the Kitchen come in flavours you can choose by NAME:
#         attention backend : "fa", "fi", "trtllm"
#         prefix cache      : "radix", "naive"
#         MoE backend       : "fused"
#       Each flavour writes its name on the menu board (Registry) with a recipe to build it
#       (a factory function). To get one you ask the board: SUPPORTED_ATTENTION_BACKENDS["fi"](cfg).
#       Benefits: the command-line flag can list the valid names, an unknown name gets a clear
#       error, and adding a new flavour means ADDING one decorated function, not editing a big
#       if/elif chain.
#
# HOW IT IS USED (see attention/__init__.py):
#       SUPPORTED = Registry[BackendCreator]("Attention Backend")
#       @SUPPORTED.register("fi")
#       def create_fi_backend(config): ...
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from typing import Callable, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    # WHAT: `type` is only a human-readable label used in error messages ("Attention Backend").
    def __init__(self, type: str):
        self._registry = {}
        self._type = type

    # ELI5: returns a decorator that writes the decorated function on the board under `name`.
    # GOTCHA: the duplicate-name check happens when `register(name)` is CALLED (at import time),
    #         not when the decorator is applied.
    def register(self, name: str) -> Callable[[T], None]:
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    # WHAT: look an option up by name; a clear error if it is not on the board.
    def __getitem__(self, name: str) -> T:
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        return list(self._registry.keys())

    # WHAT: check one name or several; raise ArgumentTypeError (so argparse prints a friendly
    #       message) listing the valid names if any is unknown.
    def assert_supported(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
