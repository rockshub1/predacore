"""
PredaCore Unified Memory Store — the single source of truth.

SQLite (schema + storage + scope filtering) layered over a Rust compute kernel
(predacore_core: vector search, BM25, fuzzy, embeddings, entity extraction).

predacore_core is a HARD dependency. No Python fallbacks. No numpy fallback.
No graceful degradation. If the Rust kernel fails, the memory system fails
loudly — this is a deliberate design choice for correctness and predictable
performance.

Tables:
    memories  — core memory entries (facts, conversations, preferences, etc.)
    entities  — named entities (people, tools, models, projects)
    relations — edges between entities (uses, prefers, part_of, etc.)
    episodes  — compressed session summaries

Vector index is an in-RAM cache only, rebuilt from SQLite on startup.
SQLite is the single source of truth — no separate vector file to corrupt.

All async methods use asyncio.to_thread() to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import predacore_core  # HARD dependency — no fallback

from ..utils.cache import TTLCache, hash_key

logger = logging.getLogger(__name__)

# Memory types that must not be evicted from the vector index cache.
# Preferences and entities are long-lived identity-defining memories.
_PROTECTED_MEMORY_TYPES = frozenset({"preference", "entity"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid4())


_VALID_MEMORY_SCOPES = frozenset({"global", "agent", "team", "scratch"})


def normalize_memory_scope(scope: str | None) -> str:
    value = str(scope or "global").strip().lower()
    return value if value in _VALID_MEMORY_SCOPES else "global"


def _coerce_metadata_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _prepare_memory_metadata(
    metadata: dict[str, Any] | None = None,
    *,
    memory_scope: str = "global",
    team_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    meta = _coerce_metadata_dict(metadata)
    meta["scope"] = normalize_memory_scope(meta.get("scope") or memory_scope)
    if team_id:
        meta["team_id"] = str(team_id)
    if agent_id:
        meta["agent_id"] = str(agent_id)
    return meta


def _memory_matches_scope(
    memory: dict[str, Any],
    scopes: list[str] | None,
    team_id: str | None = None,
    agent_id: str | None = None,
) -> bool:
    requested = [normalize_memory_scope(scope) for scope in (scopes or ["global"])]
    meta = _coerce_metadata_dict(memory.get("metadata") or {})
    actual_scope = normalize_memory_scope(meta.get("scope"))
    if actual_scope not in requested:
        return False
    if team_id and actual_scope in {"team", "scratch"}:
        return str(meta.get("team_id") or "") == str(team_id)
    if agent_id and actual_scope == "agent":
        return str(meta.get("agent_id") or "") == str(agent_id)
    return True


def future_iso_from_ttl(seconds: int | float | None) -> str | None:
    if seconds is None:
        return None
    try:
        ttl = max(int(seconds), 0)
    except (TypeError, ValueError):
        return None
    if ttl <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()


def _pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into compact binary (float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(data: bytes) -> list[float]:
    """Unpack binary-packed float32 embedding."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    importance INTEGER NOT NULL DEFAULT 2,
    source TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL DEFAULT 'default',
    embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    decay_score REAL NOT NULL DEFAULT 1.0,
    expires_at TEXT,
    session_id TEXT,
    parent_id TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    properties TEXT NOT NULL DEFAULT '{}',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    weight REAL NOT NULL DEFAULT 1.0,
    properties TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_facts TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    tools_used TEXT NOT NULL DEFAULT '[]',
    outcome TEXT,
    user_satisfaction REAL,
    created_at TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at DESC);
"""


# ---------------------------------------------------------------------------
# UnifiedMemoryStore
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Lightweight Numpy Vector Index (no gRPC, no FAISS dependency)
# ---------------------------------------------------------------------------

class _NumpyVectorIndex:
    """
    In-RAM vector index backed by predacore_core Rust SIMD cosine search.

    This is an EPHEMERAL cache. SQLite is the source of truth for embeddings
    (stored as float32 blobs). On startup, UnifiedMemoryStore rebuilds this
    index from SQLite. On eviction under memory pressure, protected types
    (preference, entity) are preserved; least-important unprotected entries
    are dropped first.

    No disk persistence. No numpy fallback. predacore_core is mandatory.
    """

    # Raised from 50K — modern hardware handles 100K × 384 float32 = ~150 MB
    MAX_VECTORS = 100_000

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._meta: list[dict] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._ids)

    def _bulk_evict(self, count: int) -> None:
        """
        Evict `count` entries. Prefers non-protected types (not preference/entity),
        falls back to oldest protected only if nothing else is available.
        Caller must hold self._lock.
        """
        if count <= 0 or not self._ids:
            return

        # First pass: find evictable (non-protected) indices, oldest first
        evictable: list[int] = [
            i for i in range(len(self._ids))
            if self._meta[i].get("type") not in _PROTECTED_MEMORY_TYPES
        ]

        to_remove: set[int]
        if len(evictable) >= count:
            to_remove = set(evictable[:count])
        else:
            # Not enough unprotected — evict all evictable, then oldest protected
            to_remove = set(evictable)
            remaining = count - len(evictable)
            protected_indices = [i for i in range(len(self._ids)) if i not in to_remove]
            to_remove.update(protected_indices[:remaining])

        # Rebuild lists, skipping removed indices (preserves order)
        self._ids = [v for i, v in enumerate(self._ids) if i not in to_remove]
        self._vecs = [v for i, v in enumerate(self._vecs) if i not in to_remove]
        self._meta = [v for i, v in enumerate(self._meta) if i not in to_remove]

        logger.info(
            "Vector index evicted %d entries (cap=%d, protected=%d)",
            len(to_remove),
            self.MAX_VECTORS,
            sum(1 for m in self._meta if m.get("type") in _PROTECTED_MEMORY_TYPES),
        )

    def _add_sync(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        """Synchronous add — used for startup rebuild from SQLite and from async add()."""
        with self._lock:
            if id in self._ids:
                idx = self._ids.index(id)
                self._vecs[idx] = vector
                self._meta[idx] = metadata or {}
                return
            if len(self._ids) >= self.MAX_VECTORS:
                # Evict 5% at a time, respecting protection
                evict_count = max(1, self.MAX_VECTORS // 20)
                self._bulk_evict(evict_count)
            self._ids.append(id)
            self._vecs.append(vector)
            self._meta.append(metadata or {})

    async def add(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        """Add or update a vector. Pure in-memory operation."""
        self._add_sync(id, vector, metadata)

    async def remove(self, id: str) -> bool:
        """Remove a vector by ID. Returns True if found and removed."""
        with self._lock:
            try:
                idx = self._ids.index(id)
            except ValueError:
                return False
            self._ids.pop(idx)
            self._vecs.pop(idx)
            self._meta.pop(idx)
            return True

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        layers: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search top-k similar vectors via predacore_core SIMD cosine.

        Raises RuntimeError if predacore_core.vector_search fails — no fallback.
        """
        with self._lock:
            if not self._ids:
                return []

            # Fetch more than top_k if we need to filter by layer, else exact top_k
            fetch_k = top_k * 3 if layers is not None else top_k
            raw_results = predacore_core.vector_search(query_vector, self._vecs, fetch_k)

            results: list[tuple[str, float]] = []
            for idx, score in raw_results:
                if layers is not None and self._meta[idx].get("layer") not in layers:
                    continue
                results.append((self._ids[idx], score))
                if len(results) >= top_k:
                    break
            return results


class UnifiedMemoryStore:
    """
    Single SQLite + vector index store for all PredaCore memory.

    All public methods are async and use asyncio.to_thread() internally
    so they never block the event loop.

    When a ``db_adapter`` is provided, all SQL operations are routed
    through it instead of a direct SQLite connection.  The in-memory
    numpy vector index is always kept in-process regardless of the
    adapter setting.
    """

    _DB_NAME = "unified_memory"

    def __init__(
        self,
        db_path: str,
        embedding_client: Any = None,
        vector_index: Any = None,
        db_adapter: Any = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embed = embedding_client
        self._vec_index = vector_index
        self._db_adapter = db_adapter
        # WAL mode permits concurrent reads; semaphore caps them at 4.
        self._db_lock = asyncio.Semaphore(4)
        # SQLite constraint: only one writer at a time.
        self._db_write_lock = asyncio.Semaphore(1)
        # Embedding cache: avoid re-embedding the same text (10 min TTL)
        self._embedding_cache = TTLCache()

        # Initialize DB
        self._conn = self._open_db()  # None when adapter is active
        if self._conn is not None:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

        # Vector index is in-RAM only. Rebuild from SQLite if not provided.
        if self._vec_index is None:
            self._vec_index = self._build_vector_index_from_sqlite()

        logger.info(
            "UnifiedMemoryStore initialized (db=%s, vectors=%d, adapter=%s)",
            self._db_path,
            self._vec_index.size if self._vec_index else 0,
            "yes" if self._db_adapter else "no",
        )

    async def __aenter__(self):
        """Support `async with UnifiedMemoryStore(...) as store:` for clean resource management."""
        return self

    async def __aexit__(self, *exc):
        """Close the store on context manager exit."""
        self.close()

    async def count(self, memory_type: str | None = None) -> int:
        """Quick count of memories, optionally filtered by type."""
        if memory_type:
            sql = "SELECT COUNT(*) FROM memories WHERE memory_type = ?"
            params: tuple = (memory_type,)
        else:
            sql = "SELECT COUNT(*) FROM memories"
            params = ()

        if self._db_adapter is not None:
            rows = await self._db_adapter.query(self._DB_NAME, sql, list(params))
            return rows[0][0] if rows else 0

        def _count():
            return self._conn.execute(sql, params).fetchone()[0]

        async with self._db_lock:
            return await self._in_thread(_count)

    async def init_schema(self) -> None:
        """Create tables via the adapter (call once at startup when adapter is used)."""
        if self._db_adapter is not None:
            await self._db_adapter.executescript(self._DB_NAME, _SCHEMA)

    def _open_db(self) -> sqlite3.Connection | None:
        if self._db_adapter is not None:
            return None
        conn = sqlite3.connect(str(self._db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Best-default pragmas for a memory-intensive local store:
        conn.execute("PRAGMA journal_mode=WAL")           # concurrent reads + crash safety
        conn.execute("PRAGMA busy_timeout=5000")          # wait up to 5s on contention
        conn.execute("PRAGMA synchronous=NORMAL")         # fast + durable enough (WAL-safe)
        conn.execute("PRAGMA foreign_keys=ON")            # referential integrity for entities/relations
        conn.execute("PRAGMA temp_store=MEMORY")          # temp tables in RAM (faster joins)
        conn.execute("PRAGMA cache_size=-64000")          # 64 MB page cache (negative = KB)
        conn.execute("PRAGMA mmap_size=268435456")        # 256 MB memory-mapped I/O
        conn.execute("PRAGMA wal_autocheckpoint=10000")   # checkpoint every 10K pages (~40 MB)
        return conn

    def _detect_embedding_dims(self) -> int:
        """Detect embedding dimensions from the client, default to 384."""
        if self._embed is not None:
            # HashingEmbeddingClient and RustEmbeddingClient have .dim directly
            if hasattr(self._embed, "dim"):
                return self._embed.dim
            # ResilientEmbeddingClient wraps a primary
            primary = getattr(self._embed, "primary", self._embed)
            if hasattr(primary, "dim"):
                return primary.dim
            # Detect by class name to avoid importing
            cls_name = type(self._embed).__name__
            if "Rust" in cls_name or "Local" in cls_name:
                return 384
            if "Hashing" in cls_name:
                return 256
        return 384  # safe default: GTE-small / all-MiniLM-L6-v2

    def _build_vector_index_from_sqlite(self) -> _NumpyVectorIndex:
        """
        Build an in-RAM vector index by reading all embeddings from SQLite.
        SQLite is the source of truth; the vector index is a hot cache.

        When the DB adapter is active, the caller must invoke
        ``rebuild_vector_index()`` asynchronously after ``init_schema()``.
        """
        dims = self._detect_embedding_dims()
        idx = _NumpyVectorIndex(dimensions=dims)

        if self._conn is None:
            # Adapter path — async rebuild must run separately
            return idx

        rows = self._conn.execute(
            "SELECT id, embedding, memory_type FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        rebuilt = 0
        for row in rows:
            mem_id = row[0]
            try:
                vec = _unpack_embedding(row[1])
            except struct.error:
                continue
            if vec and len(vec) == dims:
                idx._add_sync(mem_id, vec, {"type": row[2]})
                rebuilt += 1

        if rebuilt:
            logger.info("Rebuilt vector index from SQLite: %d vectors", rebuilt)
        return idx

    async def rebuild_vector_index(self) -> int:
        """Rebuild the in-RAM vector index from SQLite via adapter. Returns count."""
        if self._db_adapter is None or self._vec_index is None:
            return 0
        dims = self._detect_embedding_dims()
        rows = await self._db_adapter.query(
            self._DB_NAME,
            "SELECT id, embedding, memory_type FROM memories WHERE embedding IS NOT NULL",
        )
        # Clear existing index to prevent duplicates on re-rebuild
        self._vec_index._ids.clear()
        self._vec_index._vecs.clear()
        self._vec_index._meta.clear()
        rebuilt = 0
        for row in rows:
            mem_id = row[0]
            blob = row[1]
            if not blob:
                continue
            try:
                vec = _unpack_embedding(blob)
            except (struct.error, ValueError):
                continue
            if len(vec) == dims:
                self._vec_index._add_sync(mem_id, vec, {"type": row[2]})
                rebuilt += 1
        if rebuilt:
            logger.info("Rebuilt vector index via adapter: %d vectors", rebuilt)
        return rebuilt

    # ── Thread wrapper ────────────────────────────────────────────────

    async def _in_thread(self, fn, *args, **kwargs):
        """Run a sync function in a thread to avoid blocking the event loop."""
        from functools import partial

        return await asyncio.to_thread(partial(fn, *args, **kwargs))

    async def execute_raw(self, sql: str, params: tuple = ()) -> None:
        """Execute a raw SQL statement under the write lock.

        This is the public API for callers (e.g. MemoryConsolidator) that
        need to run ad-hoc SQL without touching private attributes.
        """
        if self._db_adapter is not None:
            await self._db_adapter.execute(self._DB_NAME, sql, list(params))
            return

        def _exec():
            self._conn.execute(sql, params)
            self._conn.commit()

        async with self._db_write_lock:
            async with self._db_lock:
                await self._in_thread(_exec)

    # ── Core Memory Operations ────────────────────────────────────────

    async def store(
        self,
        content: str,
        memory_type: str = "fact",
        importance: int = 2,
        source: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str = "default",
        session_id: str | None = None,
        parent_id: str | None = None,
        expires_at: str | None = None,
        memory_scope: str = "global",
        team_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Store a new memory. Returns the memory ID."""
        memory_id = _uuid()
        now = _now_iso()
        tags_json = json.dumps(tags or [])
        prepared_metadata = _prepare_memory_metadata(
            metadata,
            memory_scope=memory_scope,
            team_id=team_id,
            agent_id=agent_id,
        )
        meta_json = json.dumps(prepared_metadata)

        # Generate embedding (before DB insert — embedding is needed for the blob)
        embedding_blob = None
        embedding_vec = None
        if self._embed and content:
            try:
                embed_key = hash_key(content)
                cached_vec = self._embedding_cache.get(embed_key)
                if cached_vec is not None:
                    embedding_vec = cached_vec
                else:
                    vecs = await self._embed.embed([content])
                    if vecs and vecs[0]:
                        embedding_vec = vecs[0]
                        self._embedding_cache.set(embed_key, embedding_vec, ttl_seconds=600)
                if embedding_vec:
                    embedding_blob = _pack_embedding(embedding_vec)
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.debug("Embedding generation failed: %s", exc)

        # T1: Atomic DB insert + vector add -- DB insert first, then vector index.
        # If DB insert fails, vector index stays clean. If vector add fails,
        # we still have the embedding blob in SQLite for later index rebuild.
        _sql = """INSERT INTO memories
                   (id, content, memory_type, importance, source, tags, metadata,
                    user_id, embedding, created_at, updated_at, last_accessed,
                    access_count, decay_score, expires_at, session_id, parent_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)"""
        _params = (
            memory_id,
            content,
            memory_type,
            importance,
            source,
            tags_json,
            meta_json,
            user_id,
            embedding_blob,
            now,
            now,
            now,
            float(importance),
            expires_at,
            session_id,
            parent_id,
        )

        if self._db_adapter is not None:
            await self._db_adapter.execute(self._DB_NAME, _sql, list(_params))
        else:
            def _insert():
                self._conn.execute(_sql, _params)
                self._conn.commit()

            async with self._db_write_lock:
                async with self._db_lock:
                    await self._in_thread(_insert)

        # Add to vector index only after DB insert succeeds
        if embedding_vec and self._vec_index:
            try:
                await self._vec_index.add(
                    memory_id,
                    embedding_vec,
                    metadata={
                        "type": memory_type,
                        "user": user_id,
                        "scope": prepared_metadata.get("scope", "global"),
                        "team_id": prepared_metadata.get("team_id", ""),
                    },
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Vector index add failed (will rebuild from DB): %s", exc)
        logger.debug(
            "Stored memory %s (type=%s, importance=%d)",
            memory_id[:8],
            memory_type,
            importance,
        )
        return memory_id

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10,
        memory_types: list[str] | None = None,
        min_importance: int = 1,
        scopes: list[str] | None = None,
        team_id: str | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Recall memories by semantic similarity (predacore_core SIMD cosine).
        Empty queries or cold-start indices fall through to BM25 keyword search.
        Embedding failures propagate — no silent degradation.
        """
        # Empty query → semantic has no meaning, use keyword path
        if not query or not query.strip():
            return await self._recall_keyword(
                query, user_id, top_k, memory_types, min_importance, scopes, team_id
            )

        # No embedder configured or cold-start index → keyword path
        if not self._embed or not self._vec_index or self._vec_index.size == 0:
            return await self._recall_keyword(
                query, user_id, top_k, memory_types, min_importance, scopes, team_id
            )

        # Semantic search path — failures raise, no fallback
        embed_key = hash_key(query)
        query_vec = self._embedding_cache.get(embed_key)
        if query_vec is None:
            vecs = await self._embed.embed([query])
            if not vecs or not vecs[0]:
                raise RuntimeError(
                    f"Embedding provider returned empty for query (len={len(query)})"
                )
            query_vec = vecs[0]
            self._embedding_cache.set(embed_key, query_vec, ttl_seconds=600)

        # Semantic search via vector index
        hits = await self._vec_index.search(query_vec, top_k=top_k * 3)

        results: list[tuple[dict[str, Any], float]] = []
        if self._db_adapter is not None:
            for memory_id, score in hits:
                mem = await self._get_memory_via_adapter(memory_id)
                if mem is None:
                    continue
                if mem["user_id"] != user_id:
                    continue
                if memory_types and mem["memory_type"] not in memory_types:
                    continue
                if mem["importance"] < min_importance:
                    continue
                if not _memory_matches_scope(mem, scopes, team_id=team_id):
                    continue
                adjusted_score = score * mem["decay_score"]
                results.append((mem, adjusted_score))
                if len(results) >= top_k:
                    break
        else:
            async with self._db_lock:
                for memory_id, score in hits:
                    mem = await self._in_thread(self._get_memory_sync, memory_id)
                    if mem is None:
                        continue
                    if mem["user_id"] != user_id:
                        continue
                    if memory_types and mem["memory_type"] not in memory_types:
                        continue
                    if mem["importance"] < min_importance:
                        continue
                    if not _memory_matches_scope(mem, scopes, team_id=team_id):
                        continue
                    # Boost score by decay_score
                    adjusted_score = score * mem["decay_score"]
                    results.append((mem, adjusted_score))
                    if len(results) >= top_k:
                        break

        # Batch update access time for all retrieved memories (single SQL)
        if results:
            ids = [mem["id"] for mem, _ in results]
            now = _now_iso()
            if self._db_adapter is not None:
                placeholders = ",".join("?" * len(ids))
                await self._db_adapter.execute(
                    self._DB_NAME,
                    f"UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({placeholders})",
                    [now, *ids],
                )
            else:
                def _batch_update():
                    placeholders = ",".join("?" * len(ids))
                    self._conn.execute(
                        f"UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({placeholders})",
                        (now, *ids),
                    )
                    self._conn.commit()

                async with self._db_write_lock:
                    async with self._db_lock:
                        await self._in_thread(_batch_update)

        return results

    async def _recall_keyword(
        self,
        query: str,
        user_id: str,
        top_k: int,
        memory_types: list[str] | None = None,
        min_importance: int = 1,
        scopes: list[str] | None = None,
        team_id: str | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        BM25 keyword search using predacore_core (Rust). Used for empty queries
        and cold-start indices. Synonym expansion applied to query terms.
        """
        _sql = """SELECT * FROM memories
                     WHERE user_id = ? AND importance >= ?
                     AND (expires_at IS NULL OR expires_at > ?)
                     ORDER BY decay_score DESC LIMIT ?"""
        _params = [user_id, min_importance, _now_iso(), top_k * 10]

        if self._db_adapter is not None:
            rows = await self._db_adapter.query_dicts(self._DB_NAME, _sql, _params)
        else:
            def _search():
                return [dict(r) for r in self._conn.execute(_sql, _params).fetchall()]

            async with self._db_lock:
                rows = await self._in_thread(_search)

        if not rows:
            return []

        # Filter by memory_type and scope BEFORE BM25 (smaller corpus = faster)
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row["metadata"] = _coerce_metadata_dict(row.get("metadata") or {})
            if memory_types and row["memory_type"] not in memory_types:
                continue
            if not _memory_matches_scope(row, scopes, team_id=team_id):
                continue
            filtered.append(row)

        if not filtered:
            return []

        # Expand query terms with tech-domain synonyms via Rust
        raw_terms = [w for w in query.lower().split() if len(w) >= 2]
        if raw_terms:
            expanded = predacore_core.expand_synonyms(raw_terms)
            expanded_query = " ".join(expanded)
        else:
            expanded_query = query

        # Rust BM25 ranking (k1=1.5, b=0.75, IDF smoothing)
        contents = [r["content"] for r in filtered]
        ranked = predacore_core.bm25_search(expanded_query, contents, top_k)

        # Combine BM25 score with decay score
        results = [
            (filtered[idx], float(score) * filtered[idx]["decay_score"])
            for idx, score in ranked
        ]

        # Batch update access time
        if results:
            ids = [mem["id"] for mem, _ in results]
            now = _now_iso()
            if self._db_adapter is not None:
                placeholders = ",".join("?" * len(ids))
                await self._db_adapter.execute(
                    self._DB_NAME,
                    f"UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({placeholders})",
                    [now, *ids],
                )
            else:
                def _batch_update():
                    placeholders = ",".join("?" * len(ids))
                    self._conn.execute(
                        f"UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({placeholders})",
                        (now, *ids),
                    )
                    self._conn.commit()

                async with self._db_write_lock:
                    async with self._db_lock:
                        await self._in_thread(_batch_update)

        return results

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        if self._db_adapter is not None:
            return await self._get_memory_via_adapter(memory_id)
        async with self._db_lock:
            return await self._in_thread(self._get_memory_sync, memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        If the ID also matches an entity, cascade-delete its relations.
        """
        if self._db_adapter is not None:
            result = await self._db_adapter.execute(
                self._DB_NAME,
                "DELETE FROM memories WHERE id = ?",
                [memory_id],
            )
            deleted = result.get("rowcount", 0) > 0
            # Cascade: remove relations referencing this ID as an entity
            await self._db_adapter.execute(
                self._DB_NAME,
                "DELETE FROM relations WHERE source_entity_id = ? OR target_entity_id = ?",
                [memory_id, memory_id],
            )
            if deleted and self._vec_index:
                try:
                    await self._vec_index.remove(memory_id)
                except ValueError as exc:
                    logger.debug("Vector index removal failed for %s: %s", memory_id, exc)
            return deleted

        def _del():
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            # Cascade: remove relations referencing this ID as an entity
            self._conn.execute(
                "DELETE FROM relations WHERE source_entity_id = ? OR target_entity_id = ?",
                (memory_id, memory_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

        async with self._db_write_lock:
            async with self._db_lock:
                deleted = await self._in_thread(_del)
                if deleted and self._vec_index:
                    try:
                        await self._vec_index.remove(memory_id)
                    except ValueError as exc:
                        logger.debug("Vector index removal failed for %s: %s", memory_id, exc)
        return deleted

    async def update_access(self, memory_id: str) -> None:
        """Update access time and count for a memory."""
        if self._db_adapter is not None:
            await self._update_access_via_adapter(memory_id)
            return
        async with self._db_write_lock:
            async with self._db_lock:
                await self._in_thread(self._update_access_sync, memory_id)

    # -- Adapter helpers (async DB access) ---------------------------------

    async def _get_memory_via_adapter(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory dict via the adapter."""
        rows = await self._db_adapter.query_dicts(
            self._DB_NAME,
            "SELECT * FROM memories WHERE id = ?",
            [memory_id],
        )
        if not rows:
            return None
        d = dict(rows[0])
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        d.pop("embedding", None)
        return d

    async def _update_access_via_adapter(self, memory_id: str) -> None:
        """Update access time and count via the adapter."""
        await self._db_adapter.execute(
            self._DB_NAME,
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            [_now_iso(), memory_id],
        )

    # -- Direct-connection sync helpers ------------------------------------

    def _get_memory_sync(self, memory_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        d.pop("embedding", None)  # Don't expose raw embedding
        return d

    def _update_access_sync(self, memory_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (_now_iso(), memory_id),
        )
        self._conn.commit()

    # ── Entity Operations ─────────────────────────────────────────────

    async def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create or update an entity. Returns the entity ID."""
        if self._db_adapter is not None:
            now = _now_iso()
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT id, mention_count, properties FROM entities WHERE name = ? COLLATE NOCASE",
                [name],
            )
            if rows:
                eid = rows[0]["id"]
                existing_props = json.loads(rows[0].get("properties") or "{}")
                if properties:
                    existing_props.update(properties)
                await self._db_adapter.execute(
                    self._DB_NAME,
                    """UPDATE entities SET
                       last_seen = ?, mention_count = mention_count + 1,
                       properties = ? WHERE id = ?""",
                    [now, json.dumps(existing_props), eid],
                )
                return eid
            else:
                eid = _uuid()
                await self._db_adapter.execute(
                    self._DB_NAME,
                    """INSERT INTO entities (id, name, entity_type, properties, first_seen, last_seen, mention_count)
                       VALUES (?, ?, ?, ?, ?, ?, 1)""",
                    [eid, name, entity_type, json.dumps(properties or {}), now, now],
                )
                return eid

        def _upsert():
            now = _now_iso()
            row = self._conn.execute(
                "SELECT id, mention_count, properties FROM entities WHERE name = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            if row:
                eid = row["id"]
                existing_props = json.loads(row["properties"] or "{}")
                if properties:
                    existing_props.update(properties)
                self._conn.execute(
                    """UPDATE entities SET
                       last_seen = ?, mention_count = mention_count + 1,
                       properties = ? WHERE id = ?""",
                    (now, json.dumps(existing_props), eid),
                )
                self._conn.commit()
                return eid
            else:
                eid = _uuid()
                self._conn.execute(
                    """INSERT INTO entities (id, name, entity_type, properties, first_seen, last_seen, mention_count)
                       VALUES (?, ?, ?, ?, ?, ?, 1)""",
                    (eid, name, entity_type, json.dumps(properties or {}), now, now),
                )
                self._conn.commit()
                return eid

        async with self._db_write_lock:
            async with self._db_lock:
                return await self._in_thread(_upsert)

    async def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get entity by name (case-insensitive)."""
        if self._db_adapter is not None:
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT * FROM entities WHERE name = ? COLLATE NOCASE",
                [name],
            )
            if not rows:
                return None
            d = dict(rows[0])
            d["properties"] = json.loads(d.get("properties") or "{}")
            return d

        def _get():
            row = self._conn.execute(
                "SELECT * FROM entities WHERE name = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["properties"] = json.loads(d.get("properties") or "{}")
            return d

        async with self._db_lock:
            return await self._in_thread(_get)

    async def list_entities(
        self, entity_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List entities, optionally filtered by type."""
        if self._db_adapter is not None:
            if entity_type:
                results = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    "SELECT * FROM entities WHERE entity_type = ? ORDER BY mention_count DESC LIMIT ?",
                    [entity_type, limit],
                )
            else:
                results = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                    [limit],
                )
            for r in results:
                r["properties"] = json.loads(r.get("properties") or "{}")
            return results

        def _list():
            if entity_type:
                rows = self._conn.execute(
                    "SELECT * FROM entities WHERE entity_type = ? ORDER BY mention_count DESC LIMIT ?",
                    (entity_type, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

        async with self._db_lock:
            results = await self._in_thread(_list)
        for r in results:
            r["properties"] = json.loads(r.get("properties") or "{}")
        return results

    async def add_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str = "related_to",
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add a relation between two entities. Returns relation ID."""
        rel_id = _uuid()

        if self._db_adapter is not None:
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                """SELECT id, weight FROM relations
                   WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?""",
                [source_entity_id, target_entity_id, relation_type],
            )
            if rows:
                existing_id = rows[0]["id"]
                await self._db_adapter.execute(
                    self._DB_NAME,
                    "UPDATE relations SET weight = weight + ? WHERE id = ?",
                    [weight * 0.1, existing_id],
                )
                return existing_id
            await self._db_adapter.execute(
                self._DB_NAME,
                """INSERT INTO relations (id, source_entity_id, target_entity_id, relation_type, weight, properties, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [rel_id, source_entity_id, target_entity_id, relation_type,
                 weight, json.dumps(properties or {}), _now_iso()],
            )
            return rel_id

        def _insert():
            # Check if relation already exists
            existing = self._conn.execute(
                """SELECT id, weight FROM relations
                   WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?""",
                (source_entity_id, target_entity_id, relation_type),
            ).fetchone()
            if existing:
                # Strengthen existing relation
                self._conn.execute(
                    "UPDATE relations SET weight = weight + ? WHERE id = ?",
                    (weight * 0.1, existing["id"]),
                )
                self._conn.commit()
                return existing["id"]
            self._conn.execute(
                """INSERT INTO relations (id, source_entity_id, target_entity_id, relation_type, weight, properties, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rel_id,
                    source_entity_id,
                    target_entity_id,
                    relation_type,
                    weight,
                    json.dumps(properties or {}),
                    _now_iso(),
                ),
            )
            self._conn.commit()
            return rel_id

        async with self._db_write_lock:
            async with self._db_lock:
                return await self._in_thread(_insert)

    async def get_entity_context(self, entity_name: str) -> dict[str, Any]:
        """
        Get full context for an entity: its properties, relations, and recent memories.
        This is what the retriever uses for entity-aware context building.
        """
        entity = await self.get_entity(entity_name)
        if entity is None:
            return {"entity": None, "relations": [], "memories": []}

        eid = entity["id"]
        # Escape LIKE wildcards to prevent pattern injection
        safe_name = (
            entity_name
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        if self._db_adapter is not None:
            relations = await self._db_adapter.query_dicts(
                self._DB_NAME,
                """SELECT r.*, e1.name AS source_name, e2.name AS target_name
                   FROM relations r
                   JOIN entities e1 ON r.source_entity_id = e1.id
                   JOIN entities e2 ON r.target_entity_id = e2.id
                   WHERE r.source_entity_id = ? OR r.target_entity_id = ?
                   ORDER BY r.weight DESC LIMIT 20""",
                [eid, eid],
            )
            mem_rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                """SELECT * FROM memories
                   WHERE content LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY created_at DESC LIMIT 10""",
                [f"%{safe_name}%"],
            )
            memories = []
            for d in mem_rows:
                d.pop("embedding", None)
                d["tags"] = json.loads(d.get("tags") or "[]")
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                memories.append(d)
            return {"entity": entity, "relations": relations, "memories": memories}

        def _get_context():
            # Relations (both directions)
            rel_rows = self._conn.execute(
                """SELECT r.*, e1.name AS source_name, e2.name AS target_name
                   FROM relations r
                   JOIN entities e1 ON r.source_entity_id = e1.id
                   JOIN entities e2 ON r.target_entity_id = e2.id
                   WHERE r.source_entity_id = ? OR r.target_entity_id = ?
                   ORDER BY r.weight DESC LIMIT 20""",
                (eid, eid),
            ).fetchall()
            relations = [dict(r) for r in rel_rows]

            mem_rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE content LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY created_at DESC LIMIT 10""",
                (f"%{safe_name}%",),
            ).fetchall()
            memories = []
            for r in mem_rows:
                d = dict(r)
                d.pop("embedding", None)
                d["tags"] = json.loads(d.get("tags") or "[]")
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                memories.append(d)

            return relations, memories

        async with self._db_lock:
            relations, memories = await self._in_thread(_get_context)
        return {"entity": entity, "relations": relations, "memories": memories}

    # ── Knowledge Graph Query Methods ────────────────────────────────

    async def query_nodes(
        self,
        entity_type: str | None = None,
        name_contains: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query entities by type and/or name substring."""
        if self._db_adapter is not None:
            clauses: list[str] = []
            params: list[Any] = []
            if entity_type is not None:
                clauses.append("entity_type = ?")
                params.append(entity_type)
            if name_contains is not None:
                safe = (
                    name_contains
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                clauses.append("name LIKE ? ESCAPE '\\'")
                params.append(f"%{safe}%")
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT * FROM entities{where} ORDER BY mention_count DESC LIMIT ?"
            params.append(limit)
            rows = await self._db_adapter.query_dicts(self._DB_NAME, sql, params)
            for r in rows:
                r["properties"] = json.loads(r.get("properties") or "{}")
            return rows

        def _query():
            clauses: list[str] = []
            params: list[Any] = []
            if entity_type is not None:
                clauses.append("entity_type = ?")
                params.append(entity_type)
            if name_contains is not None:
                safe = (
                    name_contains
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                clauses.append("name LIKE ? ESCAPE '\\'")
                params.append(f"%{safe}%")
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT * FROM entities{where} ORDER BY mention_count DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        async with self._db_lock:
            results = await self._in_thread(_query)
        for r in results:
            r["properties"] = json.loads(r.get("properties") or "{}")
        return results

    async def query_edges(
        self,
        relation_type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query relations by type and/or source/target."""
        if self._db_adapter is not None:
            clauses: list[str] = []
            params: list[Any] = []
            if relation_type is not None:
                clauses.append("relation_type = ?")
                params.append(relation_type)
            if source_id is not None:
                clauses.append("source_entity_id = ?")
                params.append(source_id)
            if target_id is not None:
                clauses.append("target_entity_id = ?")
                params.append(target_id)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT * FROM relations{where} ORDER BY weight DESC LIMIT ?"
            params.append(limit)
            rows = await self._db_adapter.query_dicts(self._DB_NAME, sql, params)
            for r in rows:
                r["properties"] = json.loads(r.get("properties") or "{}")
            return rows

        def _query():
            clauses: list[str] = []
            params: list[Any] = []
            if relation_type is not None:
                clauses.append("relation_type = ?")
                params.append(relation_type)
            if source_id is not None:
                clauses.append("source_entity_id = ?")
                params.append(source_id)
            if target_id is not None:
                clauses.append("target_entity_id = ?")
                params.append(target_id)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"SELECT * FROM relations{where} ORDER BY weight DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

        async with self._db_lock:
            results = await self._in_thread(_query)
        for r in results:
            r["properties"] = json.loads(r.get("properties") or "{}")
        return results

    async def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[dict]:
        """Get neighboring entities connected by relations.

        Args:
            entity_id: The entity to find neighbors for.
            direction: "outgoing", "incoming", or "both".
            relation_type: Optional filter on relation type.

        Returns:
            List of ``{"entity": {...}, "relation": {...}}`` dicts.
        """
        if self._db_adapter is not None:
            results: list[dict] = []
            if direction in ("outgoing", "both"):
                clauses = ["r.source_entity_id = ?"]
                params: list[Any] = [entity_id]
                if relation_type is not None:
                    clauses.append("r.relation_type = ?")
                    params.append(relation_type)
                where = " AND ".join(clauses)
                rows = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    f"""SELECT r.*, e.id AS eid, e.name, e.entity_type, e.properties AS eprops,
                               e.first_seen, e.last_seen, e.mention_count
                        FROM relations r JOIN entities e ON r.target_entity_id = e.id
                        WHERE {where} ORDER BY r.weight DESC""",
                    params,
                )
                for row in rows:
                    results.append({
                        "entity": {
                            "id": row["eid"], "name": row["name"],
                            "entity_type": row["entity_type"],
                            "properties": json.loads(row.get("eprops") or "{}"),
                            "first_seen": row["first_seen"], "last_seen": row["last_seen"],
                            "mention_count": row["mention_count"],
                        },
                        "relation": {
                            "id": row["id"], "source_entity_id": row["source_entity_id"],
                            "target_entity_id": row["target_entity_id"],
                            "relation_type": row["relation_type"], "weight": row["weight"],
                            "properties": json.loads(row.get("properties") or "{}"),
                            "created_at": row["created_at"],
                        },
                    })
            if direction in ("incoming", "both"):
                clauses = ["r.target_entity_id = ?"]
                params = [entity_id]
                if relation_type is not None:
                    clauses.append("r.relation_type = ?")
                    params.append(relation_type)
                where = " AND ".join(clauses)
                rows = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    f"""SELECT r.*, e.id AS eid, e.name, e.entity_type, e.properties AS eprops,
                               e.first_seen, e.last_seen, e.mention_count
                        FROM relations r JOIN entities e ON r.source_entity_id = e.id
                        WHERE {where} ORDER BY r.weight DESC""",
                    params,
                )
                for row in rows:
                    results.append({
                        "entity": {
                            "id": row["eid"], "name": row["name"],
                            "entity_type": row["entity_type"],
                            "properties": json.loads(row.get("eprops") or "{}"),
                            "first_seen": row["first_seen"], "last_seen": row["last_seen"],
                            "mention_count": row["mention_count"],
                        },
                        "relation": {
                            "id": row["id"], "source_entity_id": row["source_entity_id"],
                            "target_entity_id": row["target_entity_id"],
                            "relation_type": row["relation_type"], "weight": row["weight"],
                            "properties": json.loads(row.get("properties") or "{}"),
                            "created_at": row["created_at"],
                        },
                    })
            return results

        def _get_neighbors():
            results: list[dict] = []
            if direction in ("outgoing", "both"):
                clauses = ["r.source_entity_id = ?"]
                params: list[Any] = [entity_id]
                if relation_type is not None:
                    clauses.append("r.relation_type = ?")
                    params.append(relation_type)
                where = " AND ".join(clauses)
                rows = self._conn.execute(
                    f"""SELECT r.*, e.id AS eid, e.name, e.entity_type, e.properties AS eprops,
                               e.first_seen, e.last_seen, e.mention_count
                        FROM relations r JOIN entities e ON r.target_entity_id = e.id
                        WHERE {where} ORDER BY r.weight DESC""",
                    params,
                ).fetchall()
                for row in rows:
                    d = dict(row)
                    results.append({
                        "entity": {
                            "id": d["eid"], "name": d["name"],
                            "entity_type": d["entity_type"],
                            "properties": json.loads(d.get("eprops") or "{}"),
                            "first_seen": d["first_seen"], "last_seen": d["last_seen"],
                            "mention_count": d["mention_count"],
                        },
                        "relation": {
                            "id": d["id"], "source_entity_id": d["source_entity_id"],
                            "target_entity_id": d["target_entity_id"],
                            "relation_type": d["relation_type"], "weight": d["weight"],
                            "properties": json.loads(d.get("properties") or "{}"),
                            "created_at": d["created_at"],
                        },
                    })
            if direction in ("incoming", "both"):
                clauses = ["r.target_entity_id = ?"]
                params = [entity_id]
                if relation_type is not None:
                    clauses.append("r.relation_type = ?")
                    params.append(relation_type)
                where = " AND ".join(clauses)
                rows = self._conn.execute(
                    f"""SELECT r.*, e.id AS eid, e.name, e.entity_type, e.properties AS eprops,
                               e.first_seen, e.last_seen, e.mention_count
                        FROM relations r JOIN entities e ON r.source_entity_id = e.id
                        WHERE {where} ORDER BY r.weight DESC""",
                    params,
                ).fetchall()
                for row in rows:
                    d = dict(row)
                    results.append({
                        "entity": {
                            "id": d["eid"], "name": d["name"],
                            "entity_type": d["entity_type"],
                            "properties": json.loads(d.get("eprops") or "{}"),
                            "first_seen": d["first_seen"], "last_seen": d["last_seen"],
                            "mention_count": d["mention_count"],
                        },
                        "relation": {
                            "id": d["id"], "source_entity_id": d["source_entity_id"],
                            "target_entity_id": d["target_entity_id"],
                            "relation_type": d["relation_type"], "weight": d["weight"],
                            "properties": json.loads(d.get("properties") or "{}"),
                            "created_at": d["created_at"],
                        },
                    })
            return results

        async with self._db_lock:
            return await self._in_thread(_get_neighbors)

    # ── Episode Operations ────────────────────────────────────────────

    async def store_episode(
        self,
        session_id: str,
        summary: str,
        key_facts: list[str] | None = None,
        entities_mentioned: list[str] | None = None,
        tools_used: list[str] | None = None,
        outcome: str | None = None,
        user_satisfaction: float | None = None,
        token_count: int = 0,
    ) -> str:
        """Store a session episode summary."""
        episode_id = _uuid()
        _sql = """INSERT INTO episodes
                   (id, session_id, summary, key_facts, entities_mentioned,
                    tools_used, outcome, user_satisfaction, created_at, token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        _params = [
            episode_id,
            session_id,
            summary,
            json.dumps(key_facts or []),
            json.dumps(entities_mentioned or []),
            json.dumps(tools_used or []),
            outcome,
            user_satisfaction,
            _now_iso(),
            token_count,
        ]

        if self._db_adapter is not None:
            await self._db_adapter.execute(self._DB_NAME, _sql, _params)
        else:
            def _insert():
                self._conn.execute(_sql, _params)
                self._conn.commit()

            async with self._db_write_lock:
                async with self._db_lock:
                    await self._in_thread(_insert)
        return episode_id

    async def get_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent episode summaries."""
        if self._db_adapter is not None:
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
                [limit],
            )
            for d in rows:
                d["key_facts"] = json.loads(d.get("key_facts") or "[]")
                d["entities_mentioned"] = json.loads(
                    d.get("entities_mentioned") or "[]"
                )
                d["tools_used"] = json.loads(d.get("tools_used") or "[]")
            return rows

        def _get():
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["key_facts"] = json.loads(d.get("key_facts") or "[]")
                d["entities_mentioned"] = json.loads(
                    d.get("entities_mentioned") or "[]"
                )
                d["tools_used"] = json.loads(d.get("tools_used") or "[]")
                results.append(d)
            return results

        async with self._db_lock:
            return await self._in_thread(_get)

    async def get_summarized_session_ids(self) -> set[str]:
        """Get set of session IDs that already have episode summaries."""
        if self._db_adapter is not None:
            rows = await self._db_adapter.query(
                self._DB_NAME,
                "SELECT DISTINCT session_id FROM episodes",
            )
            return {r[0] for r in rows}

        def _get():
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM episodes"
            ).fetchall()
            return {r["session_id"] for r in rows}

        async with self._db_lock:
            return await self._in_thread(_get)

    # ── Decay & Maintenance ───────────────────────────────────────────

    # Per-type decay rates: preferences/entities persist longer, conversations fade faster
    _DECAY_RATES: dict[str, float] = {
        "preference": 0.999,   # Near-permanent (~693 hours half-life)
        "entity": 0.998,       # Very slow decay (~346 hours half-life)
        "fact": 0.997,         # Slow decay (~231 hours half-life)
        "skill": 0.997,        # Same as fact
        "knowledge": 0.996,    # Moderate (~173 hours half-life)
        "task": 0.993,         # Faster (~99 hours half-life)
        "context": 0.990,      # Fast (~69 hours half-life)
        "conversation": 0.985, # Fastest (~46 hours half-life)
    }

    async def apply_decay(
        self,
        decay_rate: float = 0.995,
        reward_map: dict[str, float] | None = None,
    ) -> int:
        """
        Apply time-based decay to all memory scores with per-type rates.

        Preferences and entities decay very slowly; conversations fade faster.
        Base formula:
            decay_score = importance * (type_rate ** hours_since_last_access)

        If ``reward_map`` is provided (session_id -> reward factor, clamped to
        [0.5, 1.5]), memories tied to positively-reviewed sessions decay slower
        and memories tied to negatively-reviewed sessions decay faster:
            effective_rate = type_rate ** (1.0 / reward_factor)
        A reward of 1.5 makes decay ~0.5x slower; 0.5 makes it ~2x faster.
        Memories with no session_id or unknown session get the neutral 1.0.

        Returns count of memories updated.
        """
        reward_map = reward_map or {}

        def _effective_rate(base_rate: float, session_id: str | None) -> float:
            reward = reward_map.get(session_id or "", 1.0) if reward_map else 1.0
            # Clamp for safety — callers should already clamp, but be defensive
            reward = max(0.5, min(1.5, float(reward)))
            if reward == 1.0:
                return base_rate
            return base_rate ** (1.0 / reward)

        if self._db_adapter is not None:
            now_ts = time.time()
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT id, importance, last_accessed, memory_type, session_id FROM memories",
            )
            count = 0
            for row in rows:
                try:
                    la = datetime.fromisoformat(row["last_accessed"]).timestamp()
                except (ValueError, TypeError):
                    la = now_ts
                hours_since = max(0.0, (now_ts - la) / 3600.0)
                mem_type = row["memory_type"] or "fact"
                base_rate = self._DECAY_RATES.get(mem_type, decay_rate)
                rate = _effective_rate(base_rate, row.get("session_id"))
                new_score = row["importance"] * (rate ** hours_since)
                await self._db_adapter.execute(
                    self._DB_NAME,
                    "UPDATE memories SET decay_score = ? WHERE id = ?",
                    [new_score, row["id"]],
                )
                count += 1
            return count

        def _decay():
            now_ts = time.time()
            rows = self._conn.execute(
                "SELECT id, importance, last_accessed, memory_type, session_id FROM memories"
            ).fetchall()
            count = 0
            for row in rows:
                try:
                    la = datetime.fromisoformat(row["last_accessed"]).timestamp()
                except (ValueError, TypeError):
                    la = now_ts
                hours_since = max(0.0, (now_ts - la) / 3600.0)
                mem_type = row["memory_type"] or "fact"
                base_rate = self._DECAY_RATES.get(mem_type, decay_rate)
                rate = _effective_rate(base_rate, row["session_id"])
                new_score = row["importance"] * (rate ** hours_since)
                self._conn.execute(
                    "UPDATE memories SET decay_score = ? WHERE id = ?",
                    (new_score, row["id"]),
                )
                count += 1
            self._conn.commit()
            return count

        async with self._db_write_lock:
            async with self._db_lock:
                return await self._in_thread(_decay)

    async def prune_expired(self) -> int:
        """Delete expired memories. Returns count deleted."""
        now = _now_iso()

        if self._db_adapter is not None:
            rows = await self._db_adapter.query(
                self._DB_NAME,
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                [now],
            )
            ids = [r[0] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await self._db_adapter.execute(
                    self._DB_NAME,
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    ids,
                )
            if ids and self._vec_index:
                for mid in ids:
                    try:
                        await self._vec_index.remove(mid)
                    except ValueError as exc:
                        logger.debug("Vector index removal failed for %s: %s", mid, exc)
            return len(ids)

        def _prune():
            # Collect IDs first so we can clean the vector index
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})", ids
                )
                self._conn.commit()
            return ids

        async with self._db_write_lock:
            async with self._db_lock:
                deleted_ids = await self._in_thread(_prune)
                # Clean up vector index for deleted memories
                if deleted_ids and self._vec_index:
                    for mid in deleted_ids:
                        try:
                            await self._vec_index.remove(mid)
                        except ValueError as exc:
                            logger.debug("Vector index removal failed for %s: %s", mid, exc)
                return len(deleted_ids)

    async def prune_low_importance(
        self, max_memories: int = 10000, min_decay: float = 0.01
    ) -> int:
        """Prune lowest-decay memories when count exceeds max_memories."""
        if self._db_adapter is not None:
            count_rows = await self._db_adapter.query(
                self._DB_NAME, "SELECT COUNT(*) FROM memories"
            )
            total = count_rows[0][0] if count_rows else 0
            if total <= max_memories:
                return 0
            excess = total - max_memories
            rows = await self._db_adapter.query(
                self._DB_NAME,
                """SELECT id FROM memories
                   WHERE decay_score < ? AND memory_type NOT IN ('preference', 'entity')
                   ORDER BY decay_score ASC LIMIT ?""",
                [min_decay, excess],
            )
            ids = [r[0] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await self._db_adapter.execute(
                    self._DB_NAME,
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    ids,
                )
            if ids and self._vec_index:
                for mid in ids:
                    try:
                        await self._vec_index.remove(mid)
                    except ValueError as exc:
                        logger.debug("Vector index removal failed for %s: %s", mid, exc)
            return len(ids)

        def _prune():
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if total <= max_memories:
                return []
            excess = total - max_memories
            rows = self._conn.execute(
                """SELECT id FROM memories
                   WHERE decay_score < ? AND memory_type NOT IN ('preference', 'entity')
                   ORDER BY decay_score ASC LIMIT ?""",
                (min_decay, excess),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})", ids
                )
                self._conn.commit()
            return ids

        async with self._db_write_lock:
            async with self._db_lock:
                deleted_ids = await self._in_thread(_prune)
                # Clean up vector index for deleted memories
                if deleted_ids and self._vec_index:
                    for mid in deleted_ids:
                        try:
                            await self._vec_index.remove(mid)
                        except ValueError as exc:
                            logger.debug("Vector index removal failed for %s: %s", mid, exc)
                return len(deleted_ids)

    async def get_all_memories(
        self,
        memory_type: str | None = None,
        limit: int = 100,
        min_decay: float = 0.0,
        scopes: list[str] | None = None,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memories for consolidation/inspection."""
        if self._db_adapter is not None:
            if memory_type:
                rows = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    """SELECT * FROM memories WHERE memory_type = ? AND decay_score >= ?
                       ORDER BY created_at DESC LIMIT ?""",
                    [memory_type, min_decay, limit],
                )
            else:
                rows = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    """SELECT * FROM memories WHERE decay_score >= ?
                       ORDER BY created_at DESC LIMIT ?""",
                    [min_decay, limit],
                )
            for d in rows:
                d.pop("embedding", None)
                d["tags"] = json.loads(d.get("tags") or "[]")
                d["metadata"] = _coerce_metadata_dict(d.get("metadata") or "{}")
            return [d for d in rows if _memory_matches_scope(d, scopes, team_id=team_id)]

        def _list():
            if memory_type:
                rows = self._conn.execute(
                    """SELECT * FROM memories WHERE memory_type = ? AND decay_score >= ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (memory_type, min_decay, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM memories WHERE decay_score >= ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (min_decay, limit),
                ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d.pop("embedding", None)
                d["tags"] = json.loads(d.get("tags") or "[]")
                d["metadata"] = _coerce_metadata_dict(d.get("metadata") or "{}")
                results.append(d)
            return [d for d in results if _memory_matches_scope(d, scopes, team_id=team_id)]

        async with self._db_lock:
            return await self._in_thread(_list)

    # ── Stats ─────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system stats."""
        if self._db_adapter is not None:
            async def _q(sql: str) -> int:
                rows = await self._db_adapter.query(self._DB_NAME, sql)
                return rows[0][0] if rows else 0

            total = await _q("SELECT COUNT(*) FROM memories")
            by_type: dict[str, int] = {}
            type_rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT memory_type, COUNT(*) AS cnt FROM memories GROUP BY memory_type",
            )
            for row in type_rows:
                by_type[row["memory_type"]] = row["cnt"]
            entities = await _q("SELECT COUNT(*) FROM entities")
            relations = await _q("SELECT COUNT(*) FROM relations")
            episodes = await _q("SELECT COUNT(*) FROM episodes")
            return {
                "total_memories": total,
                "by_type": by_type,
                "entities": entities,
                "relations": relations,
                "episodes": episodes,
                "vector_index_size": self._vec_index.size if self._vec_index else 0,
            }

        def _stats():
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = {}
            for row in self._conn.execute(
                "SELECT memory_type, COUNT(*) AS cnt FROM memories GROUP BY memory_type"
            ).fetchall():
                by_type[row["memory_type"]] = row["cnt"]
            entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[
                0
            ]
            episodes = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            return {
                "total_memories": total,
                "by_type": by_type,
                "entities": entities,
                "relations": relations,
                "episodes": episodes,
                "vector_index_size": self._vec_index.size if self._vec_index else 0,
            }

        async with self._db_lock:
            return await self._in_thread(_stats)

    # ── Vector Index Persistence ──────────────────────────────────────

    async def save_vector_index(self) -> None:
        """
        No-op. The vector index is in-RAM only; SQLite is the source of truth.
        Retained for backward compatibility with callers that expect this method.
        """
        return None

    # ── SQLite Backup ─────────────────────────────────────────────────

    async def backup(self, dest_path: str) -> None:
        """
        Atomic backup of the memory DB to dest_path using the SQLite backup API.
        Safe to call while the store is live — backup respects WAL.
        """
        if self._conn is None:
            raise RuntimeError("backup() is only available on the direct-connection path")

        def _do_backup():
            dest = sqlite3.connect(dest_path)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

        async with self._db_write_lock:
            async with self._db_lock:
                await self._in_thread(_do_backup)
        logger.info("Memory DB backed up to %s", dest_path)

    # ── Migration ─────────────────────────────────────────────────────

    async def migrate_from_legacy(
        self,
        memory_service: Any = None,
        outcome_store: Any = None,
        kn_store: Any = None,
    ) -> dict[str, int]:
        """
        Migrate data from legacy fragmented memory systems.
        Called once on first run of unified store.
        """
        stats = {"memories": 0, "outcomes": 0, "kn_nodes": 0}

        # Migrate from MemoryService
        if memory_service is not None:
            try:
                conn = (
                    getattr(memory_service, "_conn", None)
                    or getattr(memory_service, "_get_conn", lambda: None)()
                )
                if conn:
                    rows = conn.execute("SELECT * FROM memories").fetchall()
                    for row in rows:
                        await self.store(
                            content=row["content"],
                            memory_type=row["memory_type"],
                            importance=row["importance"],
                            source="legacy_memory_service",
                            tags=json.loads(row.get("tags") or "[]"),
                            metadata=json.loads(row.get("metadata") or "{}"),
                            user_id=row["user_id"],
                        )
                        stats["memories"] += 1
                    logger.info(
                        "Migrated %d memories from MemoryService", stats["memories"]
                    )
            except (sqlite3.Error, json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("MemoryService migration failed: %s", exc)

        # Migrate from OutcomeStore
        if outcome_store is not None:
            try:
                conn = getattr(outcome_store, "_conn", None)
                if conn:
                    rows = conn.execute(
                        "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT 200"
                    ).fetchall()
                    for row in rows:
                        await self.store(
                            content=f"Task: {row.get('user_message', '')[:200]} → {row.get('success', '')}",
                            memory_type="task",
                            importance=2,
                            source="legacy_outcome_store",
                            metadata={
                                "tools_used": row.get("tools_used", ""),
                                "provider": row.get("provider_used", ""),
                                "success": row.get("success", ""),
                            },
                        )
                        stats["outcomes"] += 1
                    logger.info("Migrated %d outcomes", stats["outcomes"])
            except (sqlite3.Error, KeyError, TypeError) as exc:
                logger.warning("OutcomeStore migration failed: %s", exc)

        # Migrate from Knowledge Nexus
        if kn_store is not None:
            try:
                nodes = await kn_store.query_nodes()
                for node in nodes or []:
                    text = node.properties.get("text", "")
                    if text:
                        await self.store(
                            content=text,
                            memory_type="fact",
                            importance=2,
                            source="legacy_knowledge_nexus",
                            metadata={
                                "labels": list(node.labels),
                                "layers": list(node.layers),
                            },
                        )
                        stats["kn_nodes"] += 1
                logger.info("Migrated %d KN nodes", stats["kn_nodes"])
            except (AttributeError, KeyError, TypeError) as exc:
                logger.warning("Knowledge Nexus migration failed: %s", exc)

        await self.save_vector_index()
        logger.info("Migration complete: %s", stats)
        return stats

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the SQLite connection. Vector index is GC'd with the instance."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.debug("Error closing DB connection: %s", exc)
