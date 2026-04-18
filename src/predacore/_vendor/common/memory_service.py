"""
Prometheus Memory Service — Persistent Memory with SQLite + Semantic Recall.

Upgraded from JSON file storage to SQLite (WAL mode) for ACID guarantees,
concurrent access, and efficient querying. Embeddings are stored inline
and searched via cosine similarity (with optional numpy acceleration).

Features:
    - SQLite persistence with WAL mode (survives crashes)
    - 7 memory types: fact, conversation, task, preference, context, skill, entity
    - 4 importance levels with automatic decay
    - Semantic search via embeddings + keyword fallback
    - Per-user scoping with batch write support
    - Expiring memories (auto-cleanup)
    - Full backward compat with existing JSON data (auto-migration)

Usage:
    service = MemoryService(data_path="~/.predacore/data")
    await service.store("User prefers dark mode", user_id="alice",
                        memory_type=MemoryType.PREFERENCE,
                        importance=ImportanceLevel.HIGH)
    results = await service.recall("dark mode preference", user_id="alice")
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Try to use numpy for faster vector ops, fall back to pure Python
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ── Enums ────────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    FACT = "fact"
    CONVERSATION = "conversation"
    TASK = "task"
    PREFERENCE = "preference"
    CONTEXT = "context"
    SKILL = "skill"
    ENTITY = "entity"


class ImportanceLevel(int, Enum):
    """Importance levels for memory prioritization."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class Memory:
    """Represents a single memory item."""

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    embedding: list[float] | None = None
    source: str = ""
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "source": self.source,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        return cls(
            id=UUID(data["id"])
            if isinstance(data.get("id"), str)
            else data.get("id", uuid4()),
            content=data.get("content", ""),
            memory_type=MemoryType(data.get("memory_type", "fact")),
            importance=ImportanceLevel(data.get("importance", 2)),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            user_id=data.get("user_id", ""),
            created_at=_parse_datetime(data.get("created_at")),
            last_accessed=_parse_datetime(data.get("last_accessed")),
            access_count=data.get("access_count", 0),
            source=data.get("source", ""),
            expires_at=_parse_datetime(data.get("expires_at"))
            if data.get("expires_at")
            else None,
        )


def _parse_datetime(val: Any) -> datetime:
    """Parse a datetime from ISO string, timestamp, or return now."""
    if val is None:
        return datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return val
    if isinstance(val, int | float):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    try:
        # Handle ISO format strings (with or without timezone)
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ── SQLite Schema ────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    content      TEXT NOT NULL,
    memory_type  TEXT NOT NULL DEFAULT 'fact',
    importance   INTEGER NOT NULL DEFAULT 2,
    tags         TEXT NOT NULL DEFAULT '[]',
    metadata     TEXT NOT NULL DEFAULT '{}',
    embedding    BLOB,
    source       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(user_id, importance);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL;

-- Schema version tracking for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Insert initial version if not present
INSERT OR IGNORE INTO schema_version (version, applied_at)
    VALUES (1, datetime('now'));
"""


# ── Memory Service ───────────────────────────────────────────────────


class MemoryService:
    """
    Manages persistent memories with SQLite + semantic search.

    Features:
    - SQLite WAL mode for crash-safe persistence
    - Semantic search via embeddings (numpy-accelerated if available)
    - Keyword search fallback when embeddings unavailable
    - Automatic expired memory cleanup
    - In-memory cache for fast repeated reads
    - Backward-compatible JSON import
    - Thread-safe (connection per thread)
    - Optional DBAdapter support (skips write lock when server serializes)
    """

    _DB_NAME = "memory"

    def __init__(
        self,
        data_path: str | None = None,
        embedding_client: Any | None = None,
        write_batch_size: int = 100,
        db_adapter: Any = None,
    ):
        self._data_path = Path(data_path) if data_path else Path("data/memory")
        self._data_path.mkdir(parents=True, exist_ok=True)

        self._db_path = self._data_path / "memory.db"
        self._embedding_client = embedding_client
        self._write_batch_size = write_batch_size
        self._db_adapter = db_adapter

        # Thread-local storage for SQLite connections
        self._local = threading.local()

        # Application-level write lock — serializes write operations
        # (store, delete, update) so concurrent callers cannot interleave
        self._write_lock = threading.Lock()

        # In-memory caches (populated on first access per user)
        self._cache: dict[str, dict[UUID, Memory]] = {}
        self._vectors: dict[str, list[tuple[UUID, list[float]]]] = {}

        # Initialize database
        self._init_db()

        # Auto-migrate from JSON if legacy data exists
        self._migrate_from_json()

        # Cleanup expired memories
        self._cleanup_expired()

        total = self._count_all()
        logger.info(
            f"MemoryService initialized with {total} memories (SQLite: {self._db_path})"
        )

    # ── Database Management ──────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            # Write-ahead log for concurrent reads/writes. If another process
            # is holding an init lock, continue with the existing journal mode.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                logger.debug(
                    "Could not set journal_mode=WAL due to active lock; continuing: %s",
                    exc,
                )
            conn.execute("PRAGMA synchronous=NORMAL")  # Good durability + performance
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")  # 30s busy wait
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        retries = 1
        for attempt in range(retries + 1):
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                is_locked = "locked" in str(exc).lower()
                if not is_locked or attempt >= retries:
                    raise
                delay = 0.25 * (2**attempt)
                logger.debug(
                    "Memory DB init locked (attempt %d/%d); retrying in %.2fs",
                    attempt + 1,
                    retries + 1,
                    delay,
                )
                time.sleep(delay)

    def _count_all(self) -> int:
        """Count total memories in database."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def _cleanup_expired(self) -> None:
        """Remove expired memories from database."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        if cursor.rowcount > 0:
            conn.commit()
            logger.info(f"Cleaned up {cursor.rowcount} expired memories")

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── JSON Migration ───────────────────────────────────────────

    def _migrate_from_json(self) -> None:
        """Auto-migrate legacy JSON memory files to SQLite."""
        conn = self._get_conn()

        # Check if we already have data (skip migration)
        if self._count_all() > 0:
            return

        migrated = 0
        for user_dir in self._data_path.iterdir():
            if not user_dir.is_dir():
                continue

            memories_file = user_dir / "memories.json"
            if not memories_file.exists():
                continue

            user_id = user_dir.name
            try:
                with memories_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                for mem_data in data:
                    memory = Memory.from_dict(mem_data)
                    memory.user_id = user_id

                    # Skip expired
                    if memory.expires_at and memory.expires_at < datetime.now(
                        timezone.utc
                    ):
                        continue

                    embedding_blob = None
                    if mem_data.get("embedding"):
                        memory.embedding = mem_data["embedding"]
                        embedding_blob = _encode_embedding(memory.embedding)

                    conn.execute(
                        """INSERT OR IGNORE INTO memories
                           (id, user_id, content, memory_type, importance,
                            tags, metadata, embedding, source, created_at,
                            last_accessed, access_count, expires_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(memory.id),
                            memory.user_id,
                            memory.content,
                            memory.memory_type.value,
                            memory.importance.value,
                            json.dumps(memory.tags),
                            json.dumps(memory.metadata),
                            embedding_blob,
                            memory.source,
                            memory.created_at.isoformat(),
                            memory.last_accessed.isoformat(),
                            memory.access_count,
                            memory.expires_at.isoformat()
                            if memory.expires_at
                            else None,
                        ),
                    )
                    migrated += 1

                conn.commit()

                # Rename the old file to mark migration complete
                backup = memories_file.with_suffix(".json.migrated")
                memories_file.rename(backup)
                logger.info(
                    f"Migrated {migrated} memories for user {user_id} from JSON → SQLite"
                )

            except Exception as e:
                logger.error(f"Failed to migrate memories for user {user_id}: {e}")
                conn.rollback()

        if migrated > 0:
            logger.info(f"Total migrated: {migrated} memories from JSON → SQLite")

    # ── Core Operations ──────────────────────────────────────────

    async def store(
        self,
        content: str,
        user_id: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "",
        expires_at: datetime | None = None,
    ) -> Memory:
        """Store a new memory (persisted to SQLite immediately)."""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            user_id=user_id,
            source=source,
            expires_at=expires_at,
        )

        # Generate embedding for semantic search
        embedding_blob = None
        if self._embedding_client:
            try:
                embeddings = await self._embedding_client.embed([content])
                if embeddings and embeddings[0]:
                    memory.embedding = embeddings[0]
                    embedding_blob = _encode_embedding(memory.embedding)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Persist to SQLite
        _params = (
            str(memory.id),
            memory.user_id,
            memory.content,
            memory.memory_type.value,
            memory.importance.value,
            json.dumps(memory.tags),
            json.dumps(memory.metadata),
            embedding_blob,
            memory.source,
            memory.created_at.isoformat(),
            memory.last_accessed.isoformat(),
            memory.access_count,
            memory.expires_at.isoformat() if memory.expires_at else None,
        )
        _sql = """INSERT INTO memories
                   (id, user_id, content, memory_type, importance,
                    tags, metadata, embedding, source, created_at,
                    last_accessed, access_count, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        if self._db_adapter is not None:
            await self._db_adapter.execute(self._DB_NAME, _sql, list(_params))
        else:
            conn = self._get_conn()
            with self._write_lock:
                conn.execute(_sql, _params)
                conn.commit()

                # Update in-memory cache inside write lock to prevent stale reads
                if user_id not in self._cache:
                    self._cache[user_id] = {}
                self._cache[user_id][memory.id] = memory

                if memory.embedding:
                    if user_id not in self._vectors:
                        self._vectors[user_id] = []
                    self._vectors[user_id].append((memory.id, memory.embedding))

        logger.debug(
            f"Stored memory {memory.id} for user {user_id} [{memory_type.value}]"
        )
        return memory

    async def recall(
        self,
        query: str,
        user_id: str,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
        min_importance: ImportanceLevel = ImportanceLevel.LOW,
    ) -> list[tuple[Memory, float]]:
        """
        Recall memories using semantic search and filters.
        Returns list of (memory, score) tuples sorted by relevance.
        """
        # Load user memories from DB if not cached
        self._ensure_user_loaded(user_id)

        if user_id not in self._cache or not self._cache[user_id]:
            return []

        candidates: list[tuple[Memory, float]] = []

        # Try semantic search first
        if self._embedding_client and self._vectors.get(user_id):
            try:
                query_embedding = (await self._embedding_client.embed([query]))[0]

                for mem_id, mem_embedding in self._vectors[user_id]:
                    memory = self._cache[user_id].get(mem_id)
                    if not memory:
                        continue

                    # Embedding dimensions may differ across provider migrations.
                    if len(mem_embedding) != len(query_embedding):
                        continue

                    # Apply filters
                    if not self._passes_filters(
                        memory, memory_types, tags, min_importance
                    ):
                        continue

                    score = _cosine_similarity(query_embedding, mem_embedding)
                    candidates.append((memory, score))

            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to keyword: {e}")

        # Fallback to keyword matching
        if not candidates:
            query_lower = query.lower()
            for memory in self._cache[user_id].values():
                if not self._passes_filters(memory, memory_types, tags, min_importance):
                    continue

                content_lower = memory.content.lower()
                if query_lower in content_lower or any(
                    query_lower in tag.lower() for tag in memory.tags
                ):
                    keywords = query_lower.split()
                    score = sum(1 for kw in keywords if kw in content_lower) / max(
                        len(keywords), 1
                    )
                    candidates.append((memory, score))

        # Sort by score and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = candidates[:top_k]

        # Update access stats (batched write to DB)
        if results:
            now = datetime.now(timezone.utc).isoformat()
            for memory, _ in results:
                memory.last_accessed = datetime.now(timezone.utc)
                memory.access_count += 1
            if self._db_adapter is not None:
                for memory, _ in results:
                    await self._db_adapter.execute(
                        self._DB_NAME,
                        "UPDATE memories SET last_accessed = ?, access_count = ? WHERE id = ?",
                        [now, memory.access_count, str(memory.id)],
                    )
            else:
                conn = self._get_conn()
                with self._write_lock:
                    for memory, _ in results:
                        conn.execute(
                            "UPDATE memories SET last_accessed = ?, access_count = ? WHERE id = ?",
                            (now, memory.access_count, str(memory.id)),
                        )
                    conn.commit()

        return results

    def get(self, memory_id: UUID, user_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        self._ensure_user_loaded(user_id)
        return self._cache.get(user_id, {}).get(memory_id)

    async def delete(self, memory_id: UUID, user_id: str) -> bool:
        """Delete a memory from both SQLite and cache."""
        _sql = "DELETE FROM memories WHERE id = ? AND user_id = ?"
        _params = (str(memory_id), user_id)

        if self._db_adapter is not None:
            result = await self._db_adapter.execute(
                self._DB_NAME, _sql, list(_params)
            )
            deleted = result.get("rowcount", 0) > 0
        else:
            conn = self._get_conn()
            with self._write_lock:
                cursor = conn.execute(_sql, _params)
                conn.commit()
            deleted = cursor.rowcount > 0

        if not deleted:
            return False

        # Update cache
        if user_id in self._cache:
            self._cache[user_id].pop(memory_id, None)
        if user_id in self._vectors:
            self._vectors[user_id] = [
                (mid, vec) for mid, vec in self._vectors[user_id] if mid != memory_id
            ]

        logger.debug(f"Deleted memory {memory_id} for user {user_id}")
        return True

    def list_by_type(
        self,
        user_id: str,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[Memory]:
        """List memories by type, most recent first."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE user_id = ? AND memory_type = ?
               AND (expires_at IS NULL OR expires_at > ?)
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, memory_type.value, now, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_by_tag(
        self,
        user_id: str,
        tag: str,
        limit: int = 50,
    ) -> list[Memory]:
        """List memories containing a specific tag."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        # SQLite JSON: tags is stored as JSON array text
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE user_id = ? AND tags LIKE ?
               AND (expires_at IS NULL OR expires_at > ?)
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, f'%"{tag}"%', now, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def summarize_context(
        self,
        user_id: str,
        context_query: str,
        max_memories: int = 10,
    ) -> str:
        """Generate a context summary from relevant memories for prompt injection."""
        results = await self.recall(
            query=context_query, user_id=user_id, top_k=max_memories
        )

        if not results:
            return ""

        parts = []
        for memory, _score in results:
            prefix = f"[{memory.memory_type.value}]"
            parts.append(f"{prefix} {memory.content}")

        return "\n".join(parts)

    def get_stats(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics for a user."""
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        by_type = {}
        for row in conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories WHERE user_id = ? GROUP BY memory_type",
            (user_id,),
        ).fetchall():
            by_type[row["memory_type"]] = row["cnt"]

        by_importance = {}
        for row in conn.execute(
            "SELECT importance, COUNT(*) as cnt FROM memories WHERE user_id = ? GROUP BY importance",
            (user_id,),
        ).fetchall():
            level_name = ImportanceLevel(row["importance"]).name
            by_importance[level_name] = row["cnt"]

        with_embeddings = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ? AND embedding IS NOT NULL",
            (user_id,),
        ).fetchone()[0]

        return {
            "total": total,
            "by_type": by_type,
            "by_importance": by_importance,
            "with_embeddings": with_embeddings,
        }

    def flush(self, user_id: str | None = None) -> None:
        """Flush any pending operations. (No-op with SQLite — writes are immediate.)"""
        pass  # SQLite WAL auto-flushes; kept for API compatibility

    # ── Private Helpers ──────────────────────────────────────────

    def _ensure_user_loaded(self, user_id: str) -> None:
        """Load user's memories into cache if not already loaded."""
        if user_id in self._cache:
            return

        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            """SELECT * FROM memories WHERE user_id = ?
               AND (expires_at IS NULL OR expires_at > ?)""",
            (user_id, now),
        ).fetchall()

        self._cache[user_id] = {}
        self._vectors[user_id] = []

        for row in rows:
            memory = self._row_to_memory(row)
            self._cache[user_id][memory.id] = memory
            if memory.embedding:
                self._vectors[user_id].append((memory.id, memory.embedding))

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a SQLite row to a Memory object."""
        embedding = None
        if row["embedding"]:
            embedding = _decode_embedding(row["embedding"])

        memory = Memory(
            id=UUID(row["id"]),
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=ImportanceLevel(row["importance"]),
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            user_id=row["user_id"],
            created_at=_parse_datetime(row["created_at"]),
            last_accessed=_parse_datetime(row["last_accessed"]),
            access_count=row["access_count"],
            embedding=embedding,
            source=row["source"],
            expires_at=_parse_datetime(row["expires_at"])
            if row["expires_at"]
            else None,
        )
        return memory

    @staticmethod
    def _passes_filters(
        memory: Memory,
        memory_types: list[MemoryType] | None,
        tags: list[str] | None,
        min_importance: ImportanceLevel,
    ) -> bool:
        """Check if a memory passes the given filters."""
        if memory_types and memory.memory_type not in memory_types:
            return False
        if memory.importance.value < min_importance.value:
            return False
        if tags and not any(t in memory.tags for t in tags):
            return False
        if memory.expires_at and memory.expires_at < datetime.now(timezone.utc):
            return False
        return True


# ── Embedding Serialization ──────────────────────────────────────────


def _encode_embedding(embedding: list[float]) -> bytes:
    """Encode a float list to compact bytes for SQLite BLOB storage."""
    if HAS_NUMPY:
        return np.array(embedding, dtype=np.float32).tobytes()
    import struct

    return struct.pack(f"{len(embedding)}f", *embedding)


def _decode_embedding(blob: bytes) -> list[float]:
    """Decode bytes back to a float list."""
    if HAS_NUMPY:
        return np.frombuffer(blob, dtype=np.float32).tolist()
    import struct

    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if HAS_NUMPY:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norm) if norm > 0 else 0.0

    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ── Convenience Functions ────────────────────────────────────────────


async def remember_fact(
    service: MemoryService,
    user_id: str,
    fact: str,
    importance: ImportanceLevel = ImportanceLevel.MEDIUM,
    tags: list[str] | None = None,
) -> Memory:
    """Quick helper to store a fact."""
    return await service.store(
        content=fact,
        user_id=user_id,
        memory_type=MemoryType.FACT,
        importance=importance,
        tags=tags,
    )


async def remember_preference(
    service: MemoryService,
    user_id: str,
    preference: str,
    importance: ImportanceLevel = ImportanceLevel.HIGH,
) -> Memory:
    """Quick helper to store a user preference."""
    return await service.store(
        content=preference,
        user_id=user_id,
        memory_type=MemoryType.PREFERENCE,
        importance=importance,
        tags=["preference"],
    )
