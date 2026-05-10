"""MemoryClient — async UDS client for the daemon's MemoryServer.

Used by:
  - DAF workers (skip per-worker HNSW rebuild + embedder warmup)
  - In-process callers that want shared in-flight dedup
  - Test harnesses that connect to a running daemon

Behavior:
  - Reuses a single connection per client instance (asyncio.Lock for
    request ordering — RPCs are serialized per connection).
  - On connection drop, transparently reconnects on the next call.
  - On daemon-down (socket missing), raises ConnectionError so the
    caller can fall back to local mode.

Wire protocol: services/memory_protocol.py.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from .memory_protocol import (
    encode_frame,
    make_request,
    read_frame,
)
from .memory_server import DEFAULT_SOCKET_PATH

logger = logging.getLogger(__name__)


class MemoryClientError(RuntimeError):
    """Raised on protocol-level errors (server returned an error frame)."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.data = data


class MemoryClient:
    """Async client for the memory daemon.

    Each instance owns one persistent connection. Methods are coroutines.
    Safe for concurrent use — internal lock serializes RPCs.
    """

    def __init__(
        self,
        socket_path: str | os.PathLike[str] = DEFAULT_SOCKET_PATH,
        *,
        connect_timeout: float = 1.0,
        rpc_timeout: float = 30.0,
    ) -> None:
        self._socket_path = Path(os.path.expanduser(str(socket_path)))
        self._connect_timeout = connect_timeout
        self._rpc_timeout = rpc_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def can_reach(cls, socket_path: str | os.PathLike[str] = DEFAULT_SOCKET_PATH) -> bool:
        """Cheap probe — returns True if the socket file exists.

        Doesn't open a connection. The caller's first RPC will surface
        any deeper failure.
        """
        p = Path(os.path.expanduser(str(socket_path)))
        return p.exists() and p.is_socket()

    async def _ensure_connected(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path=str(self._socket_path)),
                timeout=self._connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError) as exc:
            raise ConnectionError(f"MemoryServer unreachable at {self._socket_path}: {exc}")

    async def close(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        self._reader = None
        self._writer = None

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """One RPC. Reconnects once on dropped connection."""
        async with self._lock:
            for attempt in range(2):
                await self._ensure_connected()
                req = make_request(method, params)
                try:
                    self._writer.write(encode_frame(req))  # type: ignore[union-attr]
                    await self._writer.drain()             # type: ignore[union-attr]
                    response = await asyncio.wait_for(
                        read_frame(self._reader), timeout=self._rpc_timeout  # type: ignore[arg-type]
                    )
                except (
                    asyncio.IncompleteReadError,
                    ConnectionResetError,
                    BrokenPipeError,
                    asyncio.TimeoutError,
                ):
                    # Drop the dead connection and retry once
                    await self.close()
                    if attempt == 0:
                        continue
                    raise ConnectionError("MemoryServer connection lost")
                if response is None:
                    await self.close()
                    if attempt == 0:
                        continue
                    raise ConnectionError("MemoryServer closed connection")
                if "error" in response:
                    err = response["error"] or {}
                    raise MemoryClientError(
                        int(err.get("code", -32603)),
                        str(err.get("message", "unknown")),
                        err.get("data"),
                    )
                return response.get("result")
        raise ConnectionError("MemoryServer unreachable after retry")

    # ── High-level RPCs ───────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._call("embed", {"texts": list(texts)})

    async def recall(
        self,
        query: str,
        *,
        user_id: str = "default",
        top_k: int = 10,
        memory_types: list[str] | None = None,
        min_importance: int = 1,
        scopes: list[str] | None = None,
        team_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        project_id: str | None = None,
        verify: bool = False,
    ) -> list[tuple[dict[str, Any], float]]:
        params: dict[str, Any] = {"query": query, "user_id": user_id, "top_k": top_k}
        if memory_types is not None:
            params["memory_types"] = memory_types
        if min_importance != 1:
            params["min_importance"] = min_importance
        if scopes is not None:
            params["scopes"] = scopes
        if team_id is not None:
            params["team_id"] = team_id
        if agent_id is not None:
            params["agent_id"] = agent_id
        if session_id is not None:
            params["session_id"] = session_id
        if project_id is not None:
            params["project_id"] = project_id
        if verify:
            params["verify"] = True
        rows = await self._call("recall", params)
        return [(r, float(r.pop("_score", 0.0))) for r in rows]

    async def store(
        self,
        content: str,
        *,
        memory_type: str = "note",
        importance: int = 2,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str = "default",
        team_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        memory_scope: str = "global",
        project_id: str | None = None,
        trust_source: str = "claude_inferred",
        confidence: float = 0.7,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "user_id": user_id,
            "memory_scope": memory_scope,
            "trust_source": trust_source,
            "confidence": confidence,
        }
        if tags is not None:
            params["tags"] = tags
        if metadata is not None:
            params["metadata"] = metadata
        if team_id is not None:
            params["team_id"] = team_id
        if agent_id is not None:
            params["agent_id"] = agent_id
        if session_id is not None:
            params["session_id"] = session_id
        if project_id is not None:
            params["project_id"] = project_id
        return await self._call("store", params)

    async def reindex_file(
        self,
        path: str,
        *,
        project_id: str | None = None,
        user_id: str = "default",
        root: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"path": path, "user_id": user_id}
        if project_id is not None:
            params["project_id"] = project_id
        if root is not None:
            params["root"] = root
        return await self._call("reindex_file", params)

    async def stats(self) -> dict[str, Any]:
        return await self._call("stats", {})

    async def inflight_check(
        self,
        *,
        team_id: str,
        tool: str,
        args: Any,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return await self._call(
            "inflight_check",
            {"team_id": team_id, "tool": tool, "args": args, "timeout": timeout},
        )

    async def inflight_publish(
        self,
        *,
        claim_key: str,
        result: Any = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"claim_key": claim_key}
        if error is not None:
            params["error"] = error
        else:
            params["result"] = result
        return await self._call("inflight_publish", params)
