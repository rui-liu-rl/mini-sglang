# FILE MAP -- server/__init__.py
# WHAT: the package's front door. Exposes launch_server, the "open the restaurant" function.
# Tag legend: see zero-to-sglang/eli5/README.md
from .launch import launch_server

__all__ = ["launch_server"]
