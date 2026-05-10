"""Wire protocol for the memory daemon UDS.

JSON-RPC over Unix Domain Socket with length-prefix framing.
Reuses the same shape as services/db_server.py / db_client.py so workers
and operators can read either with one mental model.

Frame format:
    8-byte big-endian unsigned length || UTF-8 JSON payload

Request payload:
    {"id": "<uuid>", "method": "<name>", "params": {...}}

Response payload (success):
    {"id": "<uuid>", "result": {...}}

Response payload (error):
    {"id": "<uuid>", "error": {"code": int, "message": str}}

Methods (verbs):
    embed         — embed N texts → list[vec384]
    recall        — semantic search → list[memory_row]
    bm25          — keyword search → list[memory_row]
    fuzzy         — typo-tolerant trigram → list[memory_row]
    store         — append memory (with content-hash dedup) → memory_id
    reindex_file  — re-index a file → {chunks_indexed: int}
    entity_ctx    — entity context (entity + relations + memories) → dict
    stats         — daemon stats → dict
    inflight_check — claim or wait on (tool_name, args_hash) for in-flight dedup
    inflight_publish — publish a result for an in-flight key

All methods take a "trace_id" param for end-to-end tracing.
"""
from __future__ import annotations

import asyncio
import json
import struct
import uuid
from typing import Any

# Protocol-level error codes (subset of JSON-RPC convention).
ERR_INTERNAL = -32603
ERR_INVALID_PARAMS = -32602
ERR_METHOD_NOT_FOUND = -32601
ERR_PARSE = -32700
ERR_BUDGET = -32001
ERR_CANCELLED = -32002
ERR_VALIDATION = -32003
ERR_DEDUP_AWAITED = -32004  # in-flight dedup: caller should re-call

# Maximum frame size — 16 MiB. Big enough for ~30k recalled memories or a
# bulk embed batch; small enough that a malicious client can't OOM us.
MAX_FRAME_BYTES = 16 * 1024 * 1024


def encode_frame(payload: dict[str, Any]) -> bytes:
    """Serialize payload + length prefix. Raises ValueError if too large."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {len(body)} > {MAX_FRAME_BYTES}")
    return struct.pack(">Q", len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one length-prefixed frame from a StreamReader.

    Returns None on clean EOF. Raises on protocol errors.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError:
        return None  # peer closed cleanly
    (length,) = struct.unpack(">Q", header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame size {length} exceeds MAX_FRAME_BYTES")
    body = await reader.readexactly(length)
    return json.loads(body.decode("utf-8"))


def make_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a request envelope with a fresh id."""
    return {
        "id": uuid.uuid4().hex,
        "method": method,
        "params": params or {},
    }


def make_response(req_id: str, result: Any) -> dict[str, Any]:
    return {"id": req_id, "result": result}


def make_error(req_id: str, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        err["data"] = data
    return {"id": req_id, "error": err}
