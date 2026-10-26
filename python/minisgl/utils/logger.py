# =====================================================================================
# FILE MAP -- utils/logger.py
# =====================================================================================
# WHAT THIS FILE IS: the Kitchen's NOTEBOOK-WRITER. init_logger("some.module") returns a logger
#                    that prints neat, colourful lines.
#
# ELI5: Every worker writes what it is doing in a notebook, in the same style:
#           [2026-10-25|18:03:11|core|rank=0] INFO     Allocating 21000 tokens for KV cache ...
#       With several GPU workers, all would write the same line, so there are special "_rank0"
#       versions (info_rank0, warning_rank0 ...) that only GPU worker 0 actually writes.
#
# KNOBS (environment variables):  LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO),
#                                 LOG_PID=1 adds the process id to every line.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

# WHAT: the log level, decided once per process on the first init_logger call.
_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    import logging
    import os
    import sys

    global _LOG_LEVEL
    # WHAT: first caller decides the level: the `level` argument, else $LOG_LEVEL, else INFO.
    if _LOG_LEVEL is None:
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    # WHAT: `suffix` is an extra label shown in the prefix (for example "FrontendAPI").
    #       strip_file keeps just the file-name part if a path was passed.
    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # Color formatter class
    # ELI5: decides how one log record becomes one line of text.
    class ColorFormatter(logging.Formatter):
        """Formatter with colors and pretty output"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            from minisgl.distributed import try_get_tp_info

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            # WHAT: once the GPU worker's rank is known, add "|core|rank=N" to the prefix.
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # ELI5: the "only worker 0 writes this" wrapper behind info_rank0 / warning_rank0 / ...
    # GOTCHA: it needs the TP rank to be set already (Engine.__init__ does it), otherwise the
    #         assert/get_tp_info raises. So the *_rank0 loggers are for code that runs after
    #         engine start.
    def _call_rank0(msg, *args, _which, **kwargs):
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    # WHAT: this block exists only for type checkers (so they know about the *_rank0 methods).
    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """Custom logger to handle the color formatter."""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        # WHAT: at run time, attach the four rank-0-only methods to the logger object.
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger
