# FILE MAP -- tokenizer/__init__.py
# WHAT: the package's front door. Exposes the one thing other code needs: tokenize_worker, the
#       entry point of the card-maker / card-reader process (started by server/launch.py).
# Tag legend: see zero-to-sglang/eli5/README.md
from .server import tokenize_worker

__all__ = ["tokenize_worker"]
