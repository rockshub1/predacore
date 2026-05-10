"""MemoryServer — UDS server that hosts the canonical memory plane.

One process (the predacore daemon) owns:
  - the BGE embedder (mmap'd weights via Candle, naturally page-cache-shared)
  - the HNSW graph (~150 MB at 100k vecs, expensive to duplicate per worker)
  - the SQLite WAL connection (single writer eliminates contention)
  - the Healer + snapshots (run-once globally)
  - the in-flight registry (dedup of identical tool calls across workers)

DAF workers and other in-process callers use MemoryClient (uds) instead of
each holding their own UnifiedMemoryStore. Cold-start drops from 30-60 s
(per-worker HNSW rebuild) to <500 ms.

Failure mode: if the daemon is unreachable, the client falls back to
local mode (constructs a UnifiedMemoryStore directly). This keeps tests
and one-shot CLI commands working.

Concurrency: aiohttp-style asyncio server with one connection per worker;
each connection serializes its own RPCs (per-worker ordering preserved).
The store itself uses asyncio semaphores (4 readers, 1 writer) so the
daemon can serve many readers concurrently.

Wire protocol: see services/memory_protocol.py.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .memory_protocol import (
    ERR_DEDUP_AWAITED,
    ERR_INTERNAL,
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    encode_frame,
    make_error,
    make_response,
    read_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "~/.predacore/memory.sock"
# Maximum concurrent connections accepted before back-pressure kicks in.
# 64 covers a 20-worker DAF pool with headroom.
MAX_CONNECTIONS = 64


def _hash_for_dedup(tool: str, args: Any) -> str:
    """Stable hash for in-flight dedup. tool+args canonicalised."""
    import json

    payload = json.dumps([tool, args], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class _InFlightRegistry:
    """Track tool calls currently executing per team_id.

    When two subagents call the same (tool, args) within the dedup
    window, the second await-s the first's result instead of running
    again. Eliminates the duplicate-work failure mode Anthropic hit.

    Window default: 60 s (configurable). Entries auto-expire.
    """

    def __init__(self, *, window_seconds: int = 60) -> None:
        self._window = window_seconds
        # key (team_id + dedup_hash) → asyncio.Future
        self._inflight: dict[str, tuple[asyncio.Future[Any], float]] = {}
        self._results: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def claim_or_wait(
        self, *, team_id: str, tool: str, args: Any, timeout: float = 30.0
    ) -> tuple[bool, Any]:
        """If first caller, return (True, None) — caller runs the work.

        If duplicate, await the in-flight future; return (False, result).
        On timeout, treat as miss (return True, None).
        """
        key = f"{team_id}:{_hash_for_dedup(tool, args)}"
        now = time.monotonic()

        async with self._lock:
            self._gc(now)
            # Recently completed?
            if key in self._results:
                value, _ = self._results[key]
                return False, value
            # Currently in flight?
            if key in self._inflight:
                fut, _ = self._inflight[key]
                # Found existing in-flight — await outside the lock
                try:
                    result = await asyncio.wait_for(asyncio.shield(fut), timeout)
                    return False, result
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    return True, None  # treat as miss; caller runs
            # Claim it
            fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self._inflight[key] = (fut, now)
            return True, key  # key returned so caller can publish result later

    async def publish(self, key: str, result: Any) -> None:
        """First caller finished — publish result to any waiters."""
        async with self._lock:
            entry = self._inflight.pop(key, None)
            self._results[key] = (result, time.monotonic())
            if entry is not None:
                fut, _ = entry
                if not fut.done():
                    fut.set_result(result)

    async def fail(self, key: str, exc: BaseException) -> None:
        """First caller errored — propagate to waiters; do not cache."""
        async with self._lock:
            entry = self._inflight.pop(key, None)
            if entry is not None:
                fut, _ = entry
                if not fut.done():
                    fut.set_exception(exc)

    def _gc(self, now: float) -> None:
        """Drop entries older than window. Called under the lock."""
        cutoff = now - self._window
        # Drop stale in-flight (failure left over)
        stale_inflight = [k for k, (_, t) in self._inflight.items() if t < cutoff]
        for k in stale_inflight:
            fut, _ = self._inflight.pop(k)
            if not fut.done():
                fut.cancel()
        # Drop stale results
        stale_results = [k for k, (_, t) in self._results.items() if t < cutoff]
        for k in stale_results:
            self._results.pop(k, None)


class MemoryServer:
    """UDS server fronting a UnifiedMemoryStore.

    Construct with an already-warmed store; the daemon owns its
    lifecycle. ``start()`` binds the socket; ``stop()`` cancels the
    accept task and closes connections.
    """

    def __init__(
        self,
        store: Any,  # UnifiedMemoryStore — duck-typed to avoid import cycle
        *,
        socket_path: str | os.PathLike[str] = DEFAULT_SOCKET_PATH,
        max_connections: int = MAX_CONNECTIONS,
        inflight_window_seconds: int = 60,
    ) -> None:
        self._store = store
        self._socket_path = Path(os.path.expanduser(str(socket_path)))
        self._max_connections = max_connections
        self._inflight = _InFlightRegistry(window_seconds=inflight_window_seconds)
        self._server: asyncio.base_events.Server | None = None
        self._sem = asyncio.Semaphore(max_connections)
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {
            "embed": self._handle_embed,
            "recall": self._handle_recall,
            "bm25": self._handle_bm25,
            "fuzzy": self._handle_fuzzy,
            "store": self._handle_store,
            "reindex_file": self._handle_reindex_file,
            "entity_ctx": self._handle_entity_ctx,
            "stats": self._handle_stats,
            "inflight_check": self._handle_inflight_check,
            "inflight_publish": self._handle_inflight_publish,
        }
        # Telemetry
        self._n_requests = 0
        self._n_errors = 0
        self._started_at = 0.0

    async def start(self) -> None:
        """Bind the socket and start accepting. Idempotent on re-call."""
        if self._server is not None:
            return
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            # Stale socket from prior run — remove it
            try:
                self._socket_path.unlink()
            except OSError:
                pass
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self._socket_path)
        )
        # Tighten permissions to user-only.
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError:
            pass
        self._started_at = time.monotonic()
        logger.info(
            "MemoryServer listening on %s (max_connections=%d)",
            self._socket_path, self._max_connections,
        )

    async def stop(self) -> None:
        """Stop accepting and clean up."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        with contextlib.suppress(OSError):
            self._socket_path.unlink()
        logger.info("MemoryServer stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Per-connection handler — reads frames, dispatches, writes responses.

        Connection-scoped errors close the connection but never crash the
        server. Frame-scoped errors return a JSON-RPC error response.
        """
        async with self._sem:
            try:
                while True:
                    try:
                        frame = await read_frame(reader)
                    except (asyncio.IncompleteReadError, ConnectionResetError):
                        return
                    except (ValueError, UnicodeDecodeError) as exc:
                        # Bad frame — close
                        logger.debug("MemoryServer bad frame: %s", exc)
                        return
                    if frame is None:
                        return
                    response = await self._dispatch(frame)
                    try:
                        writer.write(encode_frame(response))
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        return
            finally:
                with contextlib.suppress(OSError, ConnectionError):
                    writer.close()
                with contextlib.suppress(asyncio.CancelledError, OSError, ConnectionError):
                    await writer.wait_closed()

    async def _dispatch(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Route one request frame to the right handler."""
        self._n_requests += 1
        req_id = str(frame.get("id") or "")
        method = str(frame.get("method") or "")
        params = frame.get("params") or {}
        if not isinstance(params, dict):
            self._n_errors += 1
            return make_error(req_id, ERR_INVALID_PARAMS, "params must be object")
        handler = self._handlers.get(method)
        if handler is None:
            self._n_errors += 1
            return make_error(req_id, ERR_METHOD_NOT_FOUND, f"unknown method: {method!r}")
        try:
            result = await handler(params)
            return make_response(req_id, result)
        except (ValueError, TypeError, KeyError) as exc:
            self._n_errors += 1
            return make_error(req_id, ERR_INVALID_PARAMS, str(exc))
        except Exception as exc:  # catch-all: per-method boundary
            self._n_errors += 1
            logger.warning("MemoryServer handler %s raised: %s", method, exc, exc_info=True)
            return make_error(req_id, ERR_INTERNAL, str(exc))

    # ── Method handlers ───────────────────────────────────────────────

    async def _handle_embed(self, params: dict[str, Any]) -> Any:
        """Batch embed via predacore_core (Rust).

        Texts is a list of strings; returns list of 384-dim float vectors.
        """
        texts = params.get("texts") or []
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be list[str]")
        if not texts:
            return []
        # predacore_core.embed releases the GIL during inference, so this
        # plays nicely with concurrent dispatcher tasks.
        import predacore_core  # type: ignore[import-not-found]

        return predacore_core.embed(texts)

    async def _handle_recall(self, params: dict[str, Any]) -> Any:
        """Semantic recall — delegates to UnifiedMemoryStore.recall."""
        query = str(params.get("query") or "").strip()
        if not query:
            return []
        kwargs = {
            k: params[k]
            for k in (
                "user_id", "top_k", "memory_types", "min_importance",
                "scopes", "team_id", "agent_id", "session_id", "tags",
                "project_id", "show_stale", "verify",
            )
            if k in params
        }
        results = await self._store.recall(query=query, **kwargs)
        # Drop scores into the row so the wire format is uniform
        return [
            {**(row if isinstance(row, dict) else {}), "_score": float(score)}
            for row, score in results
        ]

    async def _handle_bm25(self, params: dict[str, Any]) -> Any:
        """BM25 keyword search via predacore_core."""
        query = str(params.get("query") or "").strip()
        if not query:
            return []
        results = await self._store._recall_keyword(  # noqa: SLF001 — private but stable
            query=query,
            user_id=params.get("user_id", "default"),
            top_k=int(params.get("top_k", 10)),
            project_id=params.get("project_id"),
        )
        return [
            {**(row if isinstance(row, dict) else {}), "_score": float(score)}
            for row, score in results
        ]

    async def _handle_fuzzy(self, params: dict[str, Any]) -> Any:
        """Trigram fuzzy match via predacore_core."""
        import predacore_core  # type: ignore[import-not-found]

        query = str(params.get("query") or "").strip()
        if not query:
            return []
        memories = await self._store.get_all_memories(limit=int(params.get("scan_limit", 200)))
        user_id = params.get("user_id")
        if user_id is not None:
            memories = [m for m in memories if m.get("user_id") == user_id]
        if not memories:
            return []
        contents = [m.get("content", "") for m in memories]
        raw = predacore_core.fuzzy_search(
            query, contents,
            top_k=int(params.get("top_k", 5)),
            threshold=float(params.get("threshold", 0.2)),
        )
        return [
            {**memories[idx], "_score": float(score)}
            for idx, score in raw
        ]

    async def _handle_store(self, params: dict[str, Any]) -> Any:
        """Append memory with content-hash dedup.

        Dedup window: 60s by team. Returns existing id if a recent
        identical memory exists (same content_hash + same team_id),
        otherwise inserts new and returns new id.
        """
        content = str(params.get("content") or "").strip()
        if not content:
            raise ValueError("content is empty")
        team_id = str(params.get("team_id") or "")
        # Content-hash dedup — fast SHA-256 over normalized content
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        # Try dedup-by-recent-write
        try:
            existing_id = await self._lookup_recent_dup(
                content_hash=content_hash, team_id=team_id, window_seconds=60,
            )
            if existing_id:
                return {"id": existing_id, "deduped": True}
        except Exception:  # noqa: BLE001 — dedup failure must not block writes
            pass  # fall through to write

        kwargs = {
            k: params[k]
            for k in (
                "memory_type", "importance", "tags", "metadata", "source",
                "user_id", "session_id", "agent_id", "team_id", "memory_scope",
                "project_id", "ttl_seconds", "expires_at",
                "trust_source", "confidence", "supersedes",
            )
            if k in params
        }
        memory_id = await self._store.store(content=content, **kwargs)
        return {"id": memory_id, "deduped": False, "content_hash": content_hash}

    async def _lookup_recent_dup(
        self, *, content_hash: str, team_id: str, window_seconds: int
    ) -> str | None:
        """SQL lookup for a recent memory with the same content hash.

        Uses the store's connection (read-only). Returns id or None.
        """
        # The schema has a content_hash-like field used internally for
        # superseding; if it's absent we fall back to a substring match on
        # content (lossy but acceptable for dedup).
        try:
            conn = self._store._conn  # noqa: SLF001 — internal handle
        except AttributeError:
            return None
        if conn is None:
            return None
        cutoff = time.time() - window_seconds
        try:
            cur = conn.execute(
                "SELECT id FROM memories "
                "WHERE substr(content_hash,1,16) = ? "
                "  AND (team_id = ? OR (team_id IS NULL AND ? = '')) "
                "  AND created_at > datetime(?, 'unixepoch') "
                "ORDER BY created_at DESC LIMIT 1",
                (content_hash, team_id, team_id, cutoff),
            )
            row = cur.fetchone()
            return row[0] if row else None
        except Exception:  # noqa: BLE001 — schema variance tolerated
            return None

    async def _handle_reindex_file(self, params: dict[str, Any]) -> Any:
        path = str(params.get("path") or "").strip()
        if not path:
            raise ValueError("path is empty")
        kwargs = {
            k: params[k]
            for k in ("project_id", "user_id", "root", "delete_first")
            if k in params
        }
        return await self._store.reindex_file(path, **kwargs)

    async def _handle_entity_ctx(self, params: dict[str, Any]) -> Any:
        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("name is empty")
        return await self._store.get_entity_context(name)

    async def _handle_stats(self, params: dict[str, Any]) -> Any:
        store_stats = await self._store.get_stats()
        return {
            "store": store_stats,
            "server": {
                "uptime_seconds": round(time.monotonic() - self._started_at, 2),
                "n_requests": self._n_requests,
                "n_errors": self._n_errors,
            },
        }

    async def _handle_inflight_check(self, params: dict[str, Any]) -> Any:
        """Claim or wait on (tool, args) for in-flight dedup.

        Returns either:
          {"first": True, "claim_key": "<key>"}     — caller runs the work
          {"first": False, "result": <cached>}     — caller uses cached result
        """
        team_id = str(params.get("team_id") or "")
        tool = str(params.get("tool") or "")
        args = params.get("args") or {}
        timeout = float(params.get("timeout", 30.0))
        if not tool:
            raise ValueError("tool is empty")
        first, payload = await self._inflight.claim_or_wait(
            team_id=team_id, tool=tool, args=args, timeout=timeout
        )
        if first:
            return {"first": True, "claim_key": payload}
        return {"first": False, "result": payload}

    async def _handle_inflight_publish(self, params: dict[str, Any]) -> Any:
        key = str(params.get("claim_key") or "")
        if not key:
            raise ValueError("claim_key is empty")
        if "result" in params:
            await self._inflight.publish(key, params.get("result"))
            return {"published": True}
        if "error" in params:
            await self._inflight.fail(key, RuntimeError(str(params["error"])))
            return {"failed": True}
        raise ValueError("must provide 'result' or 'error'")
