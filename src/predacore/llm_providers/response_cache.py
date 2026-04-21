"""
Local LLM response idempotency cache (Phase D — 2026-04-21).

SQLite-backed cache of ``(provider, model, messages, tools, temperature) →
response``. Hits return instantly without calling the provider API; misses
go through normally and the result is written back.

Works for every provider transparently because it lives at the router
boundary — provider adapters don't know the cache exists.

Design:
  - Opt-in only: enabled by ``PREDACORE_IDEMPOTENT=1`` env var.
  - Only caches deterministic calls (``temperature == 0.0``). Any higher
    temperature is by definition non-deterministic — a cached response
    would be statistically wrong.
  - Key is a sha256 over (provider, model, messages, tools, temperature).
    Session ID is intentionally NOT in the key so identical prompts hit
    across sessions (what you want for tests/benchmarks).
  - TTL default 24h (``PREDACORE_IDEMPOTENT_TTL_HOURS``). Expired entries
    are pruned on write.
  - Thread-safe via SQLite's own locking. No in-process mutex.
  - Same-pattern as ``agents/daf/IdempotencyCache`` for OpenClaw but
    scoped to LLM calls instead of delegated tasks.

Use cases:
  - Benchmark re-runs: re-running LongMemEval or similar stops hitting
    rate limits because identical prompts already have cached responses.
  - Test suites: deterministic tests speed up 10-100x.
  - Offline resume: daemon restart after an API outage can serve cached
    responses for prompts that already ran.

Not for:
  - Production chat with real users (temperature=0.7, unique prompts).
  - Workflows where the environment changes between requests in ways
    not captured in the prompt (e.g. tool outputs that read live files).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path.home() / ".predacore" / "cache" / "responses.db"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_responses (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_responses_expires ON llm_responses(expires_at);
"""


def is_enabled() -> bool:
    """Cache is gated by PREDACORE_IDEMPOTENT env var. Default: OFF.

    Set ``PREDACORE_IDEMPOTENT=1`` to enable. Any other value (including
    unset) keeps the cache disabled — every ``get()`` returns ``None`` and
    every ``set()`` is a no-op.
    """
    return os.environ.get("PREDACORE_IDEMPOTENT", "").strip().lower() in {"1", "true", "yes", "on"}


def _compute_key(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
) -> str:
    """Stable sha256 over the canonical inputs that determine an LLM response.

    Uses ``json.dumps(..., sort_keys=True)`` so dict key order doesn't break
    hits. Session ID is NOT hashed — identical prompts hit across sessions.
    """
    payload = {
        "provider": provider or "",
        "model": model or "",
        "messages": messages or [],
        "tools": tools or [],
        "temperature": float(temperature) if temperature is not None else 0.0,
    }
    try:
        blob = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError) as exc:
        # Fallback: non-serializable content — bail (don't cache, don't crash).
        logger.debug("cache key serialization failed: %s", exc)
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """SQLite-backed cache of LLM responses keyed by prompt fingerprint."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_CACHE_PATH
        env_ttl = os.environ.get("PREDACORE_IDEMPOTENT_TTL_HOURS", "")
        if ttl_seconds is not None:
            self.ttl_seconds = int(ttl_seconds)
        elif env_ttl:
            try:
                self.ttl_seconds = int(float(env_ttl) * 3600)
            except (TypeError, ValueError):
                self.ttl_seconds = _DEFAULT_TTL_SECONDS
        else:
            self.ttl_seconds = _DEFAULT_TTL_SECONDS
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # ``isolation_level=None`` = autocommit mode; we commit explicitly via ``executescript``.
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def get(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
    ) -> dict[str, Any] | None:
        """Return cached response or ``None``.

        Temperature must be ``0.0`` — any non-zero temperature is
        non-deterministic by definition and returns None even on hit.
        Returns None silently if the cache is disabled or the key is
        unreachable (serialization failure, DB lock, etc.).
        """
        if not is_enabled():
            return None
        # Only deterministic calls cache.
        if temperature is not None and float(temperature) != 0.0:
            return None
        key = _compute_key(provider, model, messages, tools, temperature)
        if not key:
            return None
        now = time.time()
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT response_json, expires_at FROM llm_responses WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                response_json, expires_at = row
                if float(expires_at) <= now:
                    # Expired — nothing to return; a future set() will also
                    # run purge, so we don't delete here (keeps get() fast).
                    return None
                conn.execute(
                    "UPDATE llm_responses SET hit_count = hit_count + 1 WHERE key = ?",
                    (key,),
                )
        except sqlite3.Error as exc:
            logger.debug("response cache get() failed (non-fatal): %s", exc)
            return None
        try:
            return json.loads(response_json)
        except (TypeError, ValueError) as exc:
            logger.debug("response cache payload decode failed: %s", exc)
            return None

    def set(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        response: dict[str, Any],
    ) -> None:
        """Store a response for later reuse.

        No-op when the cache is disabled, temperature is non-zero, or the
        response is unserializable (likely contains raw bytes or custom
        objects — bail rather than crash the inference loop).
        """
        if not is_enabled():
            return
        if temperature is not None and float(temperature) != 0.0:
            return
        key = _compute_key(provider, model, messages, tools, temperature)
        if not key:
            return
        try:
            response_json = json.dumps(response, default=str)
        except (TypeError, ValueError) as exc:
            logger.debug("response cache set() — payload unserializable: %s", exc)
            return
        now = time.time()
        expires = now + self.ttl_seconds
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_responses "
                    "(key, provider, model, response_json, created_at, expires_at, hit_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0)",
                    (key, provider, model, response_json, now, expires),
                )
                # Opportunistic prune of expired rows — cheap, runs only on write.
                conn.execute(
                    "DELETE FROM llm_responses WHERE expires_at <= ?", (now,),
                )
        except sqlite3.Error as exc:
            logger.debug("response cache set() failed (non-fatal): %s", exc)

    def stats(self) -> dict[str, Any]:
        """Return cache health stats for observability."""
        try:
            with self._conn() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM llm_responses"
                ).fetchone()[0]
                hits = conn.execute(
                    "SELECT COALESCE(SUM(hit_count), 0) FROM llm_responses"
                ).fetchone()[0]
                expired = conn.execute(
                    "SELECT COUNT(*) FROM llm_responses WHERE expires_at <= ?",
                    (time.time(),),
                ).fetchone()[0]
                by_provider = dict(
                    conn.execute(
                        "SELECT provider, COUNT(*) FROM llm_responses GROUP BY provider"
                    ).fetchall()
                )
        except sqlite3.Error as exc:
            return {"enabled": is_enabled(), "error": str(exc)}
        return {
            "enabled": is_enabled(),
            "db_path": str(self.db_path),
            "total_rows": total,
            "total_hits": int(hits),
            "expired_rows": expired,
            "by_provider": by_provider,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> int:
        """Delete every cache entry; returns the count deleted."""
        try:
            with self._conn() as conn:
                before = conn.execute(
                    "SELECT COUNT(*) FROM llm_responses"
                ).fetchone()[0]
                conn.execute("DELETE FROM llm_responses")
                return int(before)
        except sqlite3.Error as exc:
            logger.debug("response cache clear() failed: %s", exc)
            return 0


# Module-level shared instance — lazy, process-local.
_shared_cache: ResponseCache | None = None


def get_shared_cache() -> ResponseCache:
    """Return the process-wide singleton cache instance."""
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = ResponseCache()
    return _shared_cache


__all__ = [
    "ResponseCache",
    "get_shared_cache",
    "is_enabled",
]
