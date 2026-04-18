"""
Lightweight logging helpers to standardize trace_id handling and key-value logs.
"""
import json
from typing import Any, Optional

try:
    from google.protobuf.struct_pb2 import Struct
except (
    ImportError,
    AttributeError,
    TypeError,
):  # pragma: no cover - google protobuf may not be present in some scopes
    Struct = None  # type: ignore


def extract_trace_id(
    context_struct: Optional["Struct"], fallback: str | None = None
) -> str | None:
    """Safely extract trace_id from a protobuf Struct; fallback if missing.

    Args:
        context_struct: google.protobuf.Struct that may contain 'trace_id'.
        fallback: value to return if 'trace_id' not present.

    Returns:
        The trace_id string, or fallback, or None.
    """
    try:
        if (
            context_struct
            and hasattr(context_struct, "fields")
            and "trace_id" in context_struct.fields
        ):
            v = context_struct.fields["trace_id"]
            # Handle both string_value and number_value just in case
            if hasattr(v, "string_value") and v.string_value:
                return v.string_value
            if hasattr(v, "number_value") and v.number_value:
                try:
                    return str(int(v.number_value))
                except (ValueError, TypeError):
                    return str(v.number_value)
    except (AttributeError, KeyError, TypeError):
        pass
    return fallback


def kv_message(msg: str, **kv) -> str:
    """Render a compact key=value tail for logs."""
    if not kv:
        return msg
    parts = []
    for k, v in kv.items():
        try:
            parts.append(f"{k}={v}")
        except (TypeError, ValueError):
            parts.append(f"{k}=?")
    return f"{msg} | " + " ".join(parts)


def log_json(logger: Any, level: int, event: str, **fields: dict[str, Any]) -> None:
    """Emit a structured JSON log with a top-level 'event' field.

    Example:
        log_json(logger, logging.INFO, 'wil.execute', trace_id=..., tool_id=..., status='success')
    """
    try:
        payload = {"event": event, **fields}
        logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback to plain string if serialization fails
        logger.log(level, f"{event} | {fields}")
