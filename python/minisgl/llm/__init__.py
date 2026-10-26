# FILE MAP -- llm/__init__.py
# WHAT: the package's front door. Exposes LLM, the "private chef at your kitchen table" class
#       for using mini-sglang as a Python library without any server.
# Tag legend: see zero-to-sglang/eli5/README.md
from .llm import LLM

__all__ = ["LLM"]
