"""
PredaCore Structured Logging — JSON logs with trace_id correlation.

Replaces scattered `logging.getLogger(__name__)` with a consistent
structured logging system. Every log line carries:
  - timestamp (ISO 8601)
  - level
  - logger name
  - message
  - trace_id (from metrics.py ContextVar)
  - extra fields (tool_name, user_id, provider, etc.)

Supports two output modes:
  - JSON (production/daemon) — machine-parseable, works with ELK/Loki
  - Pretty (development/CLI) — colored, human-readable

Usage:
    from src.common.logging_config import setup_logging, get_logger

    setup_logging(json_mode=False, level="DEBUG")  # Pretty for dev
    logger = get_logger("predacore.core")
    logger.info("Processing message", extra={"user_id": "alice", "channel": "cli"})
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


def get_trace_id() -> str:
    """Lazy wrapper to avoid dual-module loading of metrics at import time."""
    try:
        from predacore._vendor.common.metrics import get_trace_id as _get
    except Exception:  # pragma: no cover
        from predacore._vendor.common.metrics import (
            get_trace_id as _get,  # type: ignore
        )
    return _get()


# ── JSON Formatter ───────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """
    Emits each log record as a single-line JSON object.

    Output format:
    {"ts":"2026-02-21T02:30:00Z","level":"INFO","logger":"predacore.core",
     "msg":"Processing message","trace_id":"abc123","user_id":"alice"}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": get_trace_id(),
        }

        # Add file/line for errors
        if record.levelno >= logging.WARNING:
            log_entry["file"] = f"{record.filename}:{record.lineno}"

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = str(record.exc_info[1])
            log_entry["error_type"] = type(record.exc_info[1]).__name__

        # Add extra fields (user_id, channel, tool_name, provider, etc.)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_KEYS and not key.startswith("_"):
                try:
                    json.dumps(value)  # Only include JSON-serializable extras
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        return json.dumps(log_entry, ensure_ascii=False)


# Standard LogRecord attributes to exclude from extras
_STANDARD_LOG_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }
)


# ── Pretty Formatter (for CLI/dev) ──────────────────────────────────


class PrettyFormatter(logging.Formatter):
    """
    Colored, human-readable log formatter for development.

    Output:
    02:30:00 ▸ INFO   predacore.core       Processing message  [trace:abc123]
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        t = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        level = record.levelname.ljust(8)
        name = record.name[-25:].ljust(25)  # Truncate long logger names
        msg = record.getMessage()
        tid = get_trace_id()

        line = f"{self.DIM}{t}{self.RESET} ▸ {color}{level}{self.RESET} {self.DIM}{name}{self.RESET} {msg}"

        if tid:
            line += f"  {self.DIM}[trace:{tid[:8]}]{self.RESET}"

        # Append extra fields inline
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_KEYS and not k.startswith("_")
        }
        if extras:
            pairs = " ".join(f"{k}={v}" for k, v in extras.items())
            line += f"  {self.DIM}{pairs}{self.RESET}"

        # Add exception
        if record.exc_info and record.exc_info[1]:
            line += f"\n  {color}⚠ {record.exc_info[1]}{self.RESET}"

        return line


# ── Setup ────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    json_mode: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure the root logger for the entire PredaCore system.

    Args:
        level:     Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_mode: True for JSON output (production), False for pretty (dev)
        log_file:  Optional path to write logs to a file
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    if log_file:
        # File-only logging (daemon mode) — rotating to prevent unbounded growth
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)
    else:
        # Console handler (CLI/foreground mode)
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(JSONFormatter() if json_mode else PrettyFormatter())
        root.addHandler(console)

    # Suppress noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "asyncio", "grpc"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Usage:
        logger = get_logger("predacore.core")
        logger.info("Processing", extra={"user_id": "alice"})
    """
    return logging.getLogger(name)
