"""RemoteMemoryStore — thin shim that mimics UnifiedMemoryStore's async API
by delegating to the daemon's MemoryServer over UDS.

DAF workers (and any other process that doesn't want to load BGE / build
HNSW / pay the 30-60s cold-start tax) construct one of these instead of
a full UnifiedMemoryStore. The daemon owns the canonical state; this
class is purely a proxy.

Drop-in compatibility: only the methods callers actually use are
implemented. Anything missing surfaces as AttributeError immediately so
gaps are caught in development, not at runtime.

Failure mode: if the daemon goes away mid-operation, the next call
raises ConnectionError. The caller (typically SubsystemFactory) is
expected to fall back to a local UnifiedMemoryStore.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..services.memory_client import MemoryClient
from ..services.memory_server import DEFAULT_SOCKET_PATH

logger = logging.getLogger(__name__)


class RemoteMemoryStore:
    """UDS-backed proxy. Same async surface as UnifiedMemoryStore.

    Construct with the daemon socket path. ``can_connect()`` returns
    False if the daemon isn't running — caller falls back to local.
    """

    def __init__(
        self,
        socket_path: str | os.PathLike[str] = DEFAULT_SOCKET_PATH,
        *,
        rpc_timeout: float = 30.0,
    ) -> None:
        self._socket_path = Path(os.path.expanduser(str(socket_path)))
        self._client = MemoryClient(self._socket_path, rpc_timeout=rpc_timeout)

    @classmethod
    def can_connect(cls, socket_path: str | os.PathLike[str] = DEFAULT_SOCKET_PATH) -> bool:
        return MemoryClient.can_reach(socket_path)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "RemoteMemoryStore":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.close()

    # ── Reads ─────────────────────────────────────────────────────────

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
        **_extra: Any,  # absorb future kwargs without breaking
    ) -> list[tuple[dict[str, Any], float]]:
        return await self._client.recall(
            query,
            user_id=user_id,
            top_k=top_k,
            memory_types=memory_types,
            min_importance=min_importance,
            scopes=scopes,
            team_id=team_id,
            agent_id=agent_id,
            session_id=session_id,
            project_id=project_id,
            verify=verify,
        )

    async def get_all_memories(
        self,
        *,
        memory_type: str | None = None,
        limit: int = 100,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        # Implemented as a recall with an empty query falling through to
        # BM25 → keyword path on the server. For pure listings, callers
        # should typically filter by memory_type.
        params: dict[str, Any] = {"top_k": limit, "min_importance": 1}
        if memory_type is not None:
            params["memory_types"] = [memory_type]
        if user_id is not None:
            params["user_id"] = user_id
        results = await self._client.recall(query="", **params)
        return [row for row, _score in results]

    async def get_entity_context(self, name: str) -> dict[str, Any]:
        return await self._client._call("entity_ctx", {"name": name})

    # ── Writes ────────────────────────────────────────────────────────

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
        **_extra: Any,  # absorb forward-compat kwargs
    ) -> str:
        result = await self._client.store(
            content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            metadata=metadata,
            user_id=user_id,
            team_id=team_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_scope=memory_scope,
            project_id=project_id,
            trust_source=trust_source,
            confidence=confidence,
        )
        return str(result.get("id") or "")

    async def reindex_file(
        self,
        path: str,
        *,
        project_id: str | None = None,
        user_id: str = "default",
        root: str | None = None,
        **_extra: Any,
    ) -> dict[str, Any]:
        return await self._client.reindex_file(
            path, project_id=project_id, user_id=user_id, root=root
        )

    # ── Coordination (in-flight dedup) ────────────────────────────────

    async def inflight_check(
        self,
        *,
        team_id: str,
        tool: str,
        args: Any,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Claim a (tool, args) key for in-flight dedup.

        Returns:
          {"first": True,  "claim_key": "..."}      — caller runs the work
          {"first": False, "result": <cached>}      — caller uses cached value
        """
        return await self._client.inflight_check(
            team_id=team_id, tool=tool, args=args, timeout=timeout
        )

    async def inflight_publish(self, *, claim_key: str, result: Any = None) -> None:
        await self._client.inflight_publish(claim_key=claim_key, result=result)

    async def inflight_fail(self, *, claim_key: str, error: str) -> None:
        await self._client.inflight_publish(claim_key=claim_key, error=error)

    # ── Stats / passthroughs ──────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        return await self._client.stats()
