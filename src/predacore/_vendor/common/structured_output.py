"""
Utilities for validating and versioning structured outputs from LLM tools.
"""
from __future__ import annotations

from typing import Any


def validate_and_version(
    data: Any, schema: dict[str, Any] | None = None, *, version: str = "v1"
) -> dict[str, Any]:
    """
    Validate a Python object against a JSON schema (if provided), attach output_version,
    and return a dictionary. If validation fails, wrap the raw content in an error payload.
    """
    try:
        if schema and isinstance(schema, dict) and schema:
            from jsonschema import validate

            validate(instance=data, schema=schema)
        if isinstance(data, dict):
            out = dict(data)
        else:
            out = {"raw": data}
        out.setdefault("output_version", version)
        return out
    except Exception as e:
        return {
            "error": f"schema_validation_failed: {e}",
            "raw": data,
            "output_version": version,
        }
