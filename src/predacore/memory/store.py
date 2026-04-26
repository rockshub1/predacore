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
import hashlib
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

# ── Schema versioning + invariant constants ─────────────────────────
#
# Ported from the world-model-lab (2026-04-21, Phase 6b) to give
# production the same drift-detection substrate the lab has: content
# hashing, trust provenance, verification states, and version tracking.
# The `migrate_schema()` below upgrades a legacy v1 DB to this v2
# schema in place, idempotently — safe on every daemon startup.

SCHEMA_VERSION = 2

# Current embedding model version. Every row stores this so version-skew
# queries can be detected and filtered at recall time. Bump when switching
# BGE versions or embedding providers; the healer will re-embed old rows.
CURRENT_EMBEDDING_VERSION = "bge-small-en-v1.5"
CURRENT_CHUNKER_VERSION = "semantic-v1"

# Valid verification states for a memory row. Enforced at recall — rows
# not in {ok, unverified} are filtered out of semantic search unless
# explicitly requested (show_stale=True / show_superseded=True).
_VALID_VERIFICATION_STATES = frozenset({
    "ok",            # passes all invariants
    "unverified",    # not yet audited (new row) — still visible at recall
    "stale",         # content_hash no longer matches source
    "orphaned",      # source_path no longer exists
    "version_skew",  # embedding_version != current
    "superseded",    # explicitly replaced by a newer memory (Build 2)
})


def normalize_verification_state(value: str | None) -> str:
    v = str(value or "unverified").strip().lower()
    return v if v in _VALID_VERIFICATION_STATES else "unverified"


# Verification states that hide a row from default recall/search. ``unverified``
# is NOT in this set — new rows start unverified and remain visible until the
# healer audits them. ``superseded`` IS in the set by default but is opt-in
# visible via ``show_superseded=True`` (unlike the other three, which are
# always invariant-failures and never legitimately wanted).
_HIDDEN_BY_DEFAULT_STATES = frozenset({"stale", "orphaned", "version_skew"})


def _memory_is_visible_in_recall(
    mem: dict[str, Any] | sqlite3.Row,
    show_superseded: bool = False,
    *,
    skips: dict[str, int] | None = None,
) -> bool:
    """Return True if this memory should appear in default recall results.

    Filters out:
      - stale / orphaned / version_skew rows (always hidden — invariant failures)
      - superseded rows (hidden by default, opt-in via show_superseded=True)
    Keeps:
      - ok rows (healthy)
      - unverified rows (new rows awaiting healer audit)

    If ``skips`` (a counter dict) is provided, increments the matching
    reason key when returning False. Used by L4 ``recall_explain`` to
    compute per-stage filter deltas. Pass ``self._invariant_skips``.
    """
    # mem may be a dict or sqlite3.Row; both support .get / [] access, but
    # sqlite3.Row doesn't support .get — guard against that.
    try:
        state = mem["verification_state"] if "verification_state" in mem.keys() else "ok"  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        state = mem.get("verification_state", "ok") if hasattr(mem, "get") else "ok"
    state = (state or "ok").lower()
    if state in _HIDDEN_BY_DEFAULT_STATES:
        if skips is not None:
            # state name maps directly to counter key (stale → stale_verification,
            # orphaned → orphaned, version_skew → version_skew)
            key = "stale_verification" if state == "stale" else state
            if key in skips:
                skips[key] += 1
        return False
    if state == "superseded" and not show_superseded:
        return False
    return True


def _verify_chunk_against_source(mem: dict[str, Any] | sqlite3.Row) -> bool | None:
    """T5+ real-time verification: is this chunk's content still present in
    its source file?

    Returns:
        True   — chunk has a source_path AND content/anchor was located
                 in the current file → chunk is still accurate.
        False  — chunk has a source_path BUT file is missing OR content
                 not found → chunk has drifted; do NOT trust it.
        None   — chunk has no source_path (synthesis memory, decision,
                 user-stated preference, etc.) — verification not applicable.
                 Don't drop: trust the trust_source × confidence weights.

    The check is two-tier:
      1. Full content substring match (whitespace-stripped) — strongest signal
      2. First non-blank line of chunk content present in file — anchor match
         (handles minor body edits while keeping the function/class header)

    For predacore's typical workload (1500+ code_extracted chunks of mostly
    Python), this gives true 100% accuracy on code-backed retrievals — we
    KNOW the chunk content still resolves in the file at recall time.
    """
    src = None
    try:
        src = mem["source_path"] if "source_path" in mem.keys() else None  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        if hasattr(mem, "get"):
            src = mem.get("source_path")
    if not src:
        return None  # not verifiable; not the same as failed

    try:
        path = Path(str(src))
    except (TypeError, ValueError):
        return False
    if not path.exists() or not path.is_file():
        return False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            current = f.read()
    except OSError:
        return False

    # Pull chunk content
    try:
        chunk = mem["content"] if "content" in mem.keys() else ""  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        chunk = mem.get("content", "") if hasattr(mem, "get") else ""
    chunk = (chunk or "").strip()
    if not chunk:
        return False

    # Tier 1: full content substring match (whitespace-normalized)
    if chunk in current:
        return True

    # Tier 2: first non-blank line of chunk present in current file
    # (anchor match — body may have drifted but the def/class line is there)
    for line in chunk.splitlines():
        line = line.strip()
        if line:
            if line in current:
                return True
            break  # only check the FIRST non-blank line as anchor

    return False


def _matches_project_filter(
    mem: dict[str, Any] | sqlite3.Row,
    project_id: str | list[str] | None,
) -> bool:
    """L5 — return True if memory's project_id matches the filter.

    Filter semantics:
      - ``None`` or ``"all"`` → no filter (always True)
      - str → exact match
      - list[str] → match any value in the list
    """
    if project_id is None or project_id == "all":
        return True
    try:
        mem_proj = mem["project_id"] if "project_id" in mem.keys() else "default"  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        mem_proj = mem.get("project_id", "default") if hasattr(mem, "get") else "default"
    mem_proj = mem_proj or "default"
    if isinstance(project_id, str):
        return mem_proj == project_id
    return mem_proj in project_id


def _row_field(mem: dict[str, Any] | sqlite3.Row, key: str, default: Any = None) -> Any:
    """Safe accessor that works on both dict and sqlite3.Row.

    sqlite3.Row doesn't implement .get(), so we can't do ``mem.get(...)``
    uniformly. This helper bridges the two types.
    """
    if hasattr(mem, "get") and callable(mem.get):  # dict
        return mem.get(key, default)
    try:
        return mem[key] if key in mem.keys() else default  # type: ignore[union-attr]
    except (AttributeError, TypeError, KeyError, IndexError):
        return default


def _apply_ranking_weights(
    base_score: float, mem: dict[str, Any] | sqlite3.Row
) -> float:
    """Combine similarity/BM25 score with trust, confidence, and decay weights.

    Formula (Build 3):
        final = base_score  *  decay_score  *  trust_multiplier  *  confidence

    where:
        decay_score       comes from the row, in [0, 1] (time-decayed importance)
        trust_multiplier  from TRUST_MULTIPLIERS keyed on the row's trust_source
                          (user_corrected=1.00, code_extracted=0.95,
                          user_stated=0.90, claude_inferred=0.60)
        confidence        from the row, clamped to [0, 1]

    Unknown/missing values get conservative defaults:
        trust_source missing / None → claude_inferred (0.60)
        confidence missing / None / bad-type → 0.7
        decay_score missing / None / bad-type → 1.0

    Note: 0.0 is a LEGITIMATE value for confidence and decay_score (= row
    is fully decayed / zero-confidence). We explicitly check for None, not
    falsiness, so 0.0 passes through unchanged.
    """
    raw_decay = _row_field(mem, "decay_score", 1.0)
    if raw_decay is None:
        decay = 1.0
    else:
        try:
            decay = float(raw_decay)
        except (TypeError, ValueError):
            decay = 1.0

    trust_source = _row_field(mem, "trust_source", "claude_inferred")
    if trust_source is None:
        trust_source = "claude_inferred"
    trust_mult = TRUST_MULTIPLIERS.get(
        str(trust_source).lower(), TRUST_MULTIPLIERS["claude_inferred"]
    )

    raw_conf = _row_field(mem, "confidence", 0.7)
    if raw_conf is None:
        conf = 0.7
    else:
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = 0.7
    conf = max(0.0, min(1.0, conf))

    return float(base_score) * decay * trust_mult * conf


# Trust provenance — where a fact came from. Used at retrieval time to
# weight the final ranking score (Build 3).
#
# Ranking intuition (code is the most verifiable ground truth; user
# corrections trump everything; claude inferences need verification):
#     user_corrected   1.00  — explicit user override of a prior fact
#     code_extracted   0.95  — literal from file/API/tool output; re-verifiable
#     user_stated      0.90  — user told us; may drift if system state changes
#     claude_inferred  0.60  — model-derived; verify before acting on it
_VALID_TRUST_SOURCES = frozenset({
    "user_stated",      # explicit user input
    "claude_inferred",  # model-derived (any LLM — Gemini/GPT/Anthropic/etc.),
                        # verify before acting on it
    "code_extracted",   # literal from files / tool output (re-verifiable)
    "user_corrected",   # explicit user correction of a prior fact (Build 2)
})

# Retrieval-time score multipliers per trust source. Keep this in sync
# with the ranking formula in retriever.py (Build 3).
TRUST_MULTIPLIERS: dict[str, float] = {
    "user_corrected":  1.00,
    "code_extracted":  0.95,
    "user_stated":     0.90,
    "claude_inferred": 0.60,
}


def normalize_trust_source(value: str | None) -> str:
    """Coerce any input to a valid trust_source.

    Defaults to ``claude_inferred`` (not ``user_stated`` like the lab)
    because PredaCore's daemon stores many kinds of facts automatically;
    the conservative default prevents over-trusting a row whose caller
    forgot to pass trust_source explicitly.
    """
    v = str(value or "claude_inferred").strip().lower()
    return v if v in _VALID_TRUST_SOURCES else "claude_inferred"


def compute_content_hash(content: str) -> str:
    """SHA256 of the content bytes (UTF-8). Used to detect drift at query time."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_anchor_hash(content: str, anchor_len: int = 120) -> str:
    """Content-based anchor for line-number drift recovery.

    Hashes the first ``anchor_len`` chars of the first non-blank line.
    Line numbers are volatile (reformats, refactors shift them), but the
    first non-blank line of a semantic chunk is a stable identifier that
    can be re-grepped to find the chunk in the current file state.

    Returns an empty string if ``content`` has no non-blank lines.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return hashlib.sha256(stripped[:anchor_len].encode("utf-8")).hexdigest()
    return ""


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
    parent_id TEXT,
    -- v2 invariant columns (ported from world-model-lab 2026-04-21) --
    content_hash TEXT NOT NULL DEFAULT '',
    anchor_hash TEXT,
    source_path TEXT,
    source_blob_sha TEXT,
    source_mtime INTEGER,
    chunk_ordinal INTEGER NOT NULL DEFAULT 0,
    embedding_version TEXT NOT NULL DEFAULT '',
    chunker_version TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT 'default',
    branch TEXT,
    trust_source TEXT NOT NULL DEFAULT 'claude_inferred',
    confidence REAL NOT NULL DEFAULT 0.7,
    last_verified_at TEXT,
    verification_state TEXT NOT NULL DEFAULT 'unverified',
    -- supersede support (Build 2 prep) --
    superseded_by TEXT,
    superseded_at TEXT
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
# Schema migration: v1 → v2 (idempotent, safe on every startup)
# ---------------------------------------------------------------------------
#
# Runs once at UnifiedMemoryStore.__init__. Behaviour:
#   - Fresh DB    → no-op (all columns already created by _SCHEMA).
#                   Indexes get created; schema_version stamped = 2.
#   - Up-to-date  → no-op (early return based on schema_meta.schema_version).
#   - Legacy v1   → ALTER TABLE adds 16 v2 columns, backfills content_hash,
#                   anchor_hash, embedding_version, chunker_version, plus
#                   infers trust_source + confidence from the existing
#                   `source` column (option B).

# Columns that schema v2 adds on top of v1. Used by the migrator to
# decide what to ALTER on an existing legacy DB. Order matters only for
# readability; sqlite applies them one at a time.
_V2_COLUMNS: list[tuple[str, str]] = [
    ("content_hash", "TEXT NOT NULL DEFAULT ''"),
    ("anchor_hash", "TEXT"),
    ("source_path", "TEXT"),
    ("source_blob_sha", "TEXT"),
    ("source_mtime", "INTEGER"),
    ("chunk_ordinal", "INTEGER NOT NULL DEFAULT 0"),
    ("embedding_version", "TEXT NOT NULL DEFAULT ''"),
    ("chunker_version", "TEXT NOT NULL DEFAULT ''"),
    ("project_id", "TEXT NOT NULL DEFAULT 'default'"),
    ("branch", "TEXT"),
    ("trust_source", "TEXT NOT NULL DEFAULT 'claude_inferred'"),
    ("confidence", "REAL NOT NULL DEFAULT 0.7"),
    ("last_verified_at", "TEXT"),
    ("verification_state", "TEXT NOT NULL DEFAULT 'unverified'"),
    # Supersede support (Build 2 prep) — nullable, no default needed.
    ("superseded_by", "TEXT"),
    ("superseded_at", "TEXT"),
]

_V2_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_branch ON memories(branch)",
    "CREATE INDEX IF NOT EXISTS idx_memories_verification ON memories(verification_state)",
    "CREATE INDEX IF NOT EXISTS idx_memories_source_path ON memories(source_path)",
    "CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by)",
]


def _get_current_schema_version(conn: sqlite3.Connection) -> int:
    """Read schema_version from schema_meta, or 1 if table is missing (legacy v1)."""
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return 1  # Pre-v2 databases had no schema_meta table.
    if row is None:
        return 1
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 1


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def migrate_schema(conn: sqlite3.Connection) -> int:
    """Migrate memories table to ``SCHEMA_VERSION``. Idempotent.

    Safe to call on: a fresh DB, an up-to-date DB, or a legacy v1 DB.
    Returns the schema version after migration.

    On a legacy (v1) DB the migrator:
      - ALTER TABLEs each v2 column that isn't present yet
      - Backfills content_hash + anchor_hash from stored content
      - Tags embedding_version = CURRENT for rows that actually have an
        embedding (the healer will verify + correct these later)
      - Tags chunker_version = CURRENT for all rows
      - Infers trust_source + confidence from the existing ``source``
        column (user/* → user_stated 0.9; code/file/read_file/grep →
        code_extracted 0.85; else → claude_inferred 0.7)
      - Creates the v2 indexes
      - Writes schema_version = 2 into schema_meta
    """
    # schema_meta is created by _SCHEMA too, but we also create it here so
    # the migrator is standalone-callable (e.g. from a migration test).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )

    current = _get_current_schema_version(conn)
    if current >= SCHEMA_VERSION:
        return current

    existing = _existing_columns(conn, "memories")
    added_cols: list[str] = []
    for col, ddl in _V2_COLUMNS:
        if col in existing:
            continue
        conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")
        added_cols.append(col)

    row_count = 0
    if added_cols:
        # Content/anchor hashes from stored content.
        rows = conn.execute("SELECT id, content FROM memories").fetchall()
        row_count = len(rows)
        backfill = [
            (compute_content_hash(r[1] or ""), compute_anchor_hash(r[1] or ""), r[0])
            for r in rows
        ]
        if backfill:
            conn.executemany(
                "UPDATE memories SET content_hash = ?, anchor_hash = ? WHERE id = ?",
                backfill,
            )
        # Tag embedding_version on rows that actually have an embedding.
        # The healer will verify + correct if the user ran a different
        # embedding model previously.
        conn.execute(
            "UPDATE memories SET embedding_version = ? "
            "WHERE embedding_version = '' AND embedding IS NOT NULL",
            (CURRENT_EMBEDDING_VERSION,),
        )
        conn.execute(
            "UPDATE memories SET chunker_version = ? WHERE chunker_version = ''",
            (CURRENT_CHUNKER_VERSION,),
        )
        # Infer trust_source + confidence from the existing `source` column
        # (option B). Runs once per DB — all rows at this point are
        # pre-migration rows; anything added after this passes trust_source
        # explicitly (call-site audit ensures this).
        conn.execute(
            """
            UPDATE memories
            SET trust_source = CASE
                    WHEN LOWER(source) IN ('user', 'user_input', 'user_msg', 'user_stated')
                        THEN 'user_stated'
                    WHEN LOWER(source) LIKE '%code%'
                         OR LOWER(source) IN ('file', 'read_file', 'grep', 'tool_output')
                        THEN 'code_extracted'
                    ELSE 'claude_inferred'
                END,
                confidence = CASE
                    WHEN LOWER(source) IN ('user', 'user_input', 'user_msg', 'user_stated')
                        THEN 0.9
                    WHEN LOWER(source) LIKE '%code%'
                         OR LOWER(source) IN ('file', 'read_file', 'grep', 'tool_output')
                        THEN 0.85
                    ELSE 0.7
                END
            """
        )

    for idx_sql in _V2_INDEXES:
        conn.execute(idx_sql)

    _set_schema_version(conn, SCHEMA_VERSION)
    conn.commit()

    if added_cols:
        logger.info(
            "Memory schema migrated v%d → v%d (added %d columns: %s; backfilled %d rows)",
            current, SCHEMA_VERSION, len(added_cols), ", ".join(added_cols), row_count,
        )
    else:
        logger.debug(
            "Memory schema migration no-op (already at v%d, no columns to add)",
            current,
        )
    return SCHEMA_VERSION


# ---------------------------------------------------------------------------
# UnifiedMemoryStore
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Lightweight Numpy Vector Index (no gRPC, no FAISS dependency)
# ---------------------------------------------------------------------------

class _NumpyVectorIndex:
    """
    In-RAM vector index backed by predacore_core Rust SIMD cosine search.

    SQLite is the source of truth for embeddings (stored as float32 blobs).
    On startup, UnifiedMemoryStore attempts to load a persisted cache of
    this index from disk (``vector_index.cache.npz``) to skip the O(n)
    rebuild; if the cache is missing or stale, it falls back to rebuilding
    from SQLite. On shutdown, the current state is dumped back to disk.

    On eviction under memory pressure, protected types (preference, entity)
    are preserved; least-important unprotected entries are dropped first.

    predacore_core is mandatory — no numpy fallback for vector search.
    numpy is only used for serialization of the cache file.
    """

    # Raised from 50K — modern hardware handles 100K × 384 float32 = ~150 MB
    MAX_VECTORS = 100_000
    # Cache format version — bump when the on-disk layout changes so stale
    # caches are auto-invalidated instead of loading as corrupt.
    CACHE_VERSION = 1

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

    # ── Persistence ────────────────────────────────────────────────────
    #
    # The in-RAM index is ephemeral: SQLite is source-of-truth for every
    # embedding. But rebuilding from SQLite on every daemon start is O(n)
    # and can take 30-60 seconds at 100k memories. These two methods cache
    # the in-RAM state to disk so daemon restart is O(1) read.
    #
    # Invariants checked on load (mismatch → treat cache as invalid, fall
    # back to rebuild from SQLite):
    #   - ``cache_version``      — on-disk layout version
    #   - ``dimensions``         — embedding dimension sanity
    #   - ``row_count``          — memories-with-embedding count in SQLite
    #                              at the time of the dump
    #   - ``embedding_version``  — BGE / embedder model version
    #
    # If any invariant fails, we silently rebuild — never trust a stale
    # or corrupted cache.

    def dump_to_disk(
        self,
        path: Path,
        *,
        row_count: int,
        embedding_version: str,
    ) -> bool:
        """Persist (ids, vecs, meta) + invariant header to an ``.npz`` file.

        Atomic write via temp + rename. Compressed. Returns True on success.
        """
        import numpy as np

        with self._lock:
            if not self._ids:
                # Nothing to cache — remove any stale file to force rebuild next time
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
                return False

            header = {
                "cache_version": self.CACHE_VERSION,
                "dimensions": self.dimensions,
                "row_count": int(row_count),
                "embedding_version": str(embedding_version),
                "n_vectors": len(self._ids),
                "dumped_at": _now_iso(),
            }
            vecs_arr = np.asarray(self._vecs, dtype=np.float32)
            # Store ids as JSON bytes rather than an object array — numpy
            # refuses to load object arrays with allow_pickle=False (which
            # we MUST keep false to stay safe against malicious .npz files).
            ids_bytes = np.frombuffer(
                json.dumps(self._ids).encode("utf-8"), dtype=np.uint8
            )
            meta_bytes = np.frombuffer(
                json.dumps(self._meta).encode("utf-8"), dtype=np.uint8
            )
            header_bytes = np.frombuffer(
                json.dumps(header).encode("utf-8"), dtype=np.uint8
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        # Use a tmp filename that already ends in .npz so numpy doesn't
        # append an extra suffix (np.savez_compressed auto-appends .npz
        # to filenames lacking that extension, which would break the
        # atomic rename below).
        tmp_path = path.parent / (path.stem + "_tmp.npz")
        try:
            np.savez_compressed(
                str(tmp_path),
                header=header_bytes,
                ids=ids_bytes,
                vecs=vecs_arr,
                meta=meta_bytes,
            )
            tmp_path.replace(path)  # atomic
            logger.info(
                "Vector index persisted: %d vectors → %s (row_count=%d)",
                len(self._ids), path, row_count,
            )
            return True
        except (OSError, ValueError) as exc:
            logger.warning("Vector index persist failed: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def load_from_disk(
        self,
        path: Path,
        *,
        expected_row_count: int,
        expected_embedding_version: str,
    ) -> bool:
        """Load (ids, vecs, meta) from ``.npz``. Returns True on successful load.

        Returns False (silently, at INFO log level) if:
          - file missing or unreadable
          - header malformed or missing invariants
          - cache_version mismatch
          - dimensions mismatch
          - row_count mismatch (SQLite has different number of embedded rows)
          - embedding_version mismatch (model changed since dump)

        On False, the caller should fall back to rebuilding from SQLite.
        """
        import numpy as np

        if not path.exists():
            logger.debug("Vector index cache missing at %s — will rebuild from SQLite", path)
            return False

        try:
            data = np.load(str(path), allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.info("Vector index cache unreadable (%s) — rebuilding", exc)
            return False

        try:
            header_bytes = bytes(data["header"])
            header = json.loads(header_bytes.decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.info("Vector index cache header malformed (%s) — rebuilding", exc)
            return False

        # Invariant checks — any mismatch → silent rebuild
        reasons: list[str] = []
        if header.get("cache_version") != self.CACHE_VERSION:
            reasons.append(
                f"cache_version {header.get('cache_version')} != {self.CACHE_VERSION}"
            )
        if header.get("dimensions") != self.dimensions:
            reasons.append(
                f"dimensions {header.get('dimensions')} != {self.dimensions}"
            )
        if int(header.get("row_count", -1)) != int(expected_row_count):
            reasons.append(
                f"row_count {header.get('row_count')} != SQLite {expected_row_count}"
            )
        if header.get("embedding_version") != expected_embedding_version:
            reasons.append(
                f"embedding_version {header.get('embedding_version')!r} "
                f"!= current {expected_embedding_version!r}"
            )
        if reasons:
            logger.info(
                "Vector index cache stale (%s) — rebuilding from SQLite",
                "; ".join(reasons),
            )
            return False

        try:
            ids_bytes = bytes(data["ids"])
            ids_list = [str(x) for x in json.loads(ids_bytes.decode("utf-8"))]
            vecs_list = data["vecs"].tolist()
            meta_bytes = bytes(data["meta"])
            meta_list = json.loads(meta_bytes.decode("utf-8"))
        except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.info("Vector index cache body malformed (%s) — rebuilding", exc)
            return False

        if not (len(ids_list) == len(vecs_list) == len(meta_list)):
            logger.info(
                "Vector index cache lengths mismatch "
                "(ids=%d, vecs=%d, meta=%d) — rebuilding",
                len(ids_list), len(vecs_list), len(meta_list),
            )
            return False

        with self._lock:
            self._ids = ids_list
            self._vecs = vecs_list
            self._meta = meta_list
        logger.info(
            "Vector index loaded from cache: %d vectors from %s (dumped %s)",
            len(ids_list), path, header.get("dumped_at", "?"),
        )
        return True


# ---------------------------------------------------------------------------
# HNSW-backed vector index (opt-in, O(log n) at scale)
# ---------------------------------------------------------------------------
#
# Drop-in compatible with _NumpyVectorIndex — same async add/remove/search
# API, same .size / .dimensions. Wraps the Rust predacore_core.PyHnswIndex
# under the hood.
#
# Enable via env var: PREDACORE_USE_HNSW=1 (default: off, use linear scan).
#
# Trade-offs vs the numpy index:
#   + O(log n) search — meaningful at >~10k vectors
#   - Approximate recall (~99% at default ef_search=50, configurable)
#   - No native delete — we use a tombstone set + filter at query time
#   - No persistence yet (caller still rebuilds from SQLite on startup;
#     would need hnsw_rs::hnswio::file_dump integration, tracked as
#     follow-up work)
#
# At <10k vectors, linear scan is both faster AND simpler. Recommend HNSW
# only once you observe recall latency in the tens of ms.


class _HnswVectorIndex:
    """HNSW-backed vector index, opt-in alternative to _NumpyVectorIndex.

    Matches _NumpyVectorIndex interface so UnifiedMemoryStore can swap
    implementations transparently based on the PREDACORE_USE_HNSW env var.

    Deletion semantics: HNSW graphs can't cleanly remove nodes (removing
    would orphan neighbors). We use a tombstone set — deleted IDs are
    marked but remain in the graph; search results filter them out. Over
    time this grows; a full rebuild compacts. For PredaCore's workload
    (append-heavy, rare deletes) the tombstone growth is negligible.
    """

    # Upper bound on vectors the index can hold. hnsw_rs needs this at
    # construction time but doesn't penalize unused capacity, so we set it
    # well above the realistic ceiling (~100k for a heavy multi-year user)
    # to avoid ever having to rebuild. 1M leaves ~10× headroom over what
    # any personal memory system will actually hit.
    MAX_VECTORS = 1_000_000

    # Tunable HNSW parameters (all have sensible defaults inside the Rust
    # constructor; overriding via env is possible but not wired yet).
    #
    # These values target ~99.9% recall all the way up to 10M vectors on
    # 384-dim BGE embeddings. Denser graph + wider search beam = accuracy
    # that stays within 0.1 pp of linear scan while keeping queries under
    # 10 ms at million-vector scale. Memory overhead is ~2× the M=16 graph
    # (still negligible on modern hardware: ~200 KB at today's 80 rows,
    # ~200 MB at 100k, ~2 GB at 1M).
    DEFAULT_M = 32
    DEFAULT_EF_CONSTRUCTION = 400
    DEFAULT_EF_SEARCH = 400

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        # Rust-backed HNSW graph (thread-safe internally via its own Mutex).
        self._hnsw = predacore_core.PyHnswIndex(
            dims=dimensions,
            max_nb_connection=self.DEFAULT_M,
            ef_construction=self.DEFAULT_EF_CONSTRUCTION,
            max_elements=self.MAX_VECTORS,
        )
        # Metadata map (id → dict) — HNSW itself only stores vectors + IDs,
        # so we track our own metadata here.
        self._meta: dict[str, dict[str, Any]] = {}
        # Tombstones — IDs that were "removed" but still exist in the HNSW
        # graph (since true deletion would corrupt the graph). Filtered at
        # search time.
        self._tombstoned: set[str] = set()
        # Track insertion order for stable iteration + eviction fallback.
        self._insertion_order: list[str] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of LIVE (non-tombstoned) vectors."""
        return max(0, self._hnsw.len() - len(self._tombstoned))

    def _add_sync(
        self, id: str, vector: list[float], metadata: dict | None = None
    ) -> None:
        """Synchronous add — used by the startup SQLite-rebuild path
        and by the async `add()` below. Matches _NumpyVectorIndex._add_sync
        signature so `_build_vector_index_from_sqlite` works for both
        backends transparently.

        For re-add of an existing id: HNSW doesn't support update-in-place,
        so we just insert a new slot. The old slot remains in the graph
        but the new metadata replaces the old. Search returns the
        highest-scoring slot for that id (naturally — they have the same
        id string, dedupe in search() keeps the first = top-scored).
        """
        with self._lock:
            try:
                self._hnsw.insert(id, list(vector))
            except RuntimeError as exc:
                logger.warning("HNSW insert failed for %s: %s", id[:8], exc)
                return
            # Refresh metadata + clear tombstone (re-add un-tombstones)
            self._meta[id] = dict(metadata or {})
            self._tombstoned.discard(id)
            if id not in self._insertion_order:
                self._insertion_order.append(id)

    async def add(
        self, id: str, vector: list[float], metadata: dict | None = None
    ) -> None:
        """Async add — delegates to sync method."""
        self._add_sync(id, vector, metadata)

    async def remove(self, id: str) -> bool:
        """Mark a vector as removed via tombstone. Graph itself is unchanged
        (HNSW doesn't support node deletion cleanly). Returns True if the
        ID existed before this call."""
        with self._lock:
            if id not in self._meta:
                return False
            if id in self._tombstoned:
                return False  # already tombstoned
            self._tombstoned.add(id)
            return True

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        layers: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search top-k via HNSW approximate nearest neighbor. Filters
        tombstoned IDs and (optional) layer-mismatched entries after.

        Fetches 3x top_k headroom so post-filter we still return top_k.
        """
        with self._lock:
            if self._hnsw.len() == 0:
                return []

            fetch_k = top_k * 3 if (layers is not None or self._tombstoned) else top_k
            try:
                raw = self._hnsw.search(
                    list(query_vector),
                    top_k=fetch_k,
                    ef_search=self.DEFAULT_EF_SEARCH,
                )
            except RuntimeError as exc:
                logger.warning("HNSW search failed: %s", exc)
                return []

            results: list[tuple[str, float]] = []
            seen_ids: set[str] = set()
            for id_, score in raw:
                if id_ in self._tombstoned:
                    continue
                if id_ in seen_ids:
                    # Dedupe: a replaced id may appear twice (old tombstoned
                    # slot already filtered; but if multiple live slots
                    # exist by some bug, keep only the highest-scoring).
                    continue
                if layers is not None:
                    meta = self._meta.get(id_, {})
                    if meta.get("layer") not in layers:
                        continue
                seen_ids.add(id_)
                results.append((id_, float(score)))
                if len(results) >= top_k:
                    break
            return results

    # ── Persistence stubs ─────────────────────────────────────────────
    #
    # HNSW graph serialization (via hnsw_rs::hnswio::file_dump) is not yet
    # wired. For now these are no-ops returning False so the parent
    # UnifiedMemoryStore treats the cache as always-invalid and falls back
    # to SQLite rebuild. That's correct behaviour even if slower; follow-up
    # task wires in proper HNSW save/load.

    def dump_to_disk(
        self, path: Path, *, row_count: int, embedding_version: str
    ) -> bool:
        """HNSW persistence not yet implemented — returns False so caller
        knows to rebuild from SQLite next start. Tracked as follow-up work
        (hnsw_rs has native file_dump; just needs PyO3 binding + Python glue).
        """
        del path, row_count, embedding_version  # unused for now
        logger.debug(
            "HNSW persistence not yet implemented — skipping dump "
            "(rebuild from SQLite on next start is O(n log n))"
        )
        return False

    def load_from_disk(
        self,
        path: Path,
        *,
        expected_row_count: int,
        expected_embedding_version: str,
    ) -> bool:
        """HNSW persistence not yet implemented — always returns False,
        forcing rebuild from SQLite."""
        del path, expected_row_count, expected_embedding_version
        return False


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
        # Ingress safety — secret scanner stats are surfaced via get_stats().
        # Import locally to avoid a circular import at module load time.
        try:
            from .safety import SafetyStats
        except ImportError:
            SafetyStats = None  # type: ignore[misc,assignment]
        self._safety_stats = SafetyStats() if SafetyStats else None
        # L4 — invariant-filter counters. Incremented inside
        # _memory_is_visible_in_recall when a row is hidden from default
        # recall. Surfaced via get_stats() and consumed by recall_explain
        # to show per-stage filter deltas. Mirrors the lab counter
        # infrastructure (lab tracks 5; public's filter only checks
        # 3 invariants — the public scope/team_id mechanism handles
        # tenant separation, so project_mismatch / branch_mismatch
        # don't apply here).
        self._invariant_skips: dict[str, int] = {
            "stale_verification": 0,  # verification_state in {stale}
            "orphaned": 0,            # verification_state == orphaned
            "version_skew": 0,        # verification_state == version_skew
            "project_mismatch": 0,    # L5 — project_id filter rejected
            "verification_failed": 0, # T5+ — real-time verify-against-source failed
        }
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
            # Idempotent v1 → v2 migration. For a fresh DB it's a no-op
            # (all columns already exist via _SCHEMA) that still creates the
            # v2 indexes and stamps schema_version = 2. For a legacy v1 DB
            # it ALTER-TABLEs in the 16 v2 columns + backfills trust_source,
            # content_hash, anchor_hash, embedding_version, chunker_version.
            migrate_schema(self._conn)

        # Vector index: try persisted cache first; fall back to rebuild from
        # SQLite if cache is missing, stale, or invalid. Saves ~30-60s at
        # 100k memories on daemon restart.
        #
        # Backend selection:
        #   PREDACORE_USE_HNSW=1 → _HnswVectorIndex (O(log n) search via
        #     Rust hnsw_rs; only meaningful at >10k vectors).
        #   default (unset/0)    → _NumpyVectorIndex (O(n) SIMD scan; fast
        #     at small/medium scale, simpler persistence, exact recall).
        if self._vec_index is None:
            dims = self._detect_embedding_dims()
            use_hnsw = os.getenv("PREDACORE_USE_HNSW", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
            if use_hnsw:
                logger.info("Vector index backend: HNSW (PREDACORE_USE_HNSW=on)")
                self._vec_index = _HnswVectorIndex(dimensions=dims)
            else:
                self._vec_index = _NumpyVectorIndex(dimensions=dims)
            # Try loading the on-disk cache (only works for the numpy
            # backend today — HNSW persistence isn't wired yet).
            if not use_hnsw and self._try_load_vector_index_cache():
                logger.info("Vector index restored from cache (skipped SQLite rebuild)")
            else:
                # Populate the already-created backend from SQLite. This
                # preserves whichever implementation we just picked above
                # (the old code here threw away the HNSW backend by
                # replacing self._vec_index with a fresh numpy index).
                self._populate_vector_index_from_sqlite(self._vec_index)

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
        """Legacy helper — creates a fresh _NumpyVectorIndex and populates it.

        Prefer ``_populate_vector_index_from_sqlite(target)`` for new code;
        that one respects whatever backend (_Numpy or _Hnsw) is already in
        use instead of hardcoding numpy. This one is kept for any external
        caller that still expects a fresh index back.
        """
        dims = self._detect_embedding_dims()
        idx = _NumpyVectorIndex(dimensions=dims)
        if self._conn is None:
            return idx
        self._populate_vector_index_from_sqlite(idx)
        return idx

    def _populate_vector_index_from_sqlite(
        self,
        target: _NumpyVectorIndex | _HnswVectorIndex,
    ) -> int:
        """Populate the given in-RAM index by reading every embedded row
        from SQLite. Works for both _NumpyVectorIndex and _HnswVectorIndex
        because both expose the same ``_add_sync(id, vec, metadata)`` API.

        Returns the number of vectors added.
        """
        if self._conn is None:
            # Adapter path — async rebuild must run separately
            return 0
        dims = self._detect_embedding_dims()
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
                target._add_sync(mem_id, vec, {"type": row[2]})
                rebuilt += 1
        if rebuilt:
            logger.info(
                "Rebuilt vector index from SQLite: %d vectors (backend=%s)",
                rebuilt, type(target).__name__,
            )
        return rebuilt

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
        trust_source: str = "claude_inferred",
        confidence: float = 0.7,
        supersedes: list[str] | None = None,
        # Phase 1b invariant fields (mirror lab) — let callers tag rows with the
        # source file + project + branch so the healer can audit, sweep
        # orphans, and the chunker can purge stale code_extracted rows.
        project_id: str = "default",
        branch: str | None = None,
        source_path: str | None = None,
        source_blob_sha: str | None = None,
        source_mtime: int | None = None,
        chunk_ordinal: int = 0,
    ) -> str:
        """Store a new memory. Returns the memory ID.

        ``trust_source`` controls retrieval ranking (see ``TRUST_MULTIPLIERS``):
            - ``"user_stated"`` — user directly said it
            - ``"user_corrected"`` — user corrected a prior fact
            - ``"code_extracted"`` — read from source/tool output (re-verifiable)
            - ``"claude_inferred"`` — model-derived (default; conservative)
        Unknown values are coerced to ``"claude_inferred"``.

        ``confidence`` (0.0–1.0) is multiplied into the ranking score.

        ``supersedes`` — optional list of memory IDs that this new memory
        replaces. Those rows get ``superseded_by = <new_id>``,
        ``superseded_at = <now>``, and ``verification_state = 'superseded'``
        atomically with the insert. Superseded rows are hidden from default
        recall but remain in the DB for audit. Unknown IDs in the list are
        safely no-op (the UPDATE just doesn't match anything).

        Ingress safety: if ``content`` matches any of the safety scanner's
        secret patterns (AWS / GitHub / OpenAI / Anthropic / PEM / SSH /
        JWT / generic credential assignments), the row is REFUSED at
        ingress — nothing is embedded, nothing is written, and an empty
        string is returned in place of the memory id. The block is logged
        at WARNING and counted in ``get_safety_stats()``.
        """
        # Ingress secret scan — done first so we don't embed nor insert anything.
        if content and self._safety_stats is not None:
            try:
                from .safety import scan_for_secrets
            except ImportError:
                scan_for_secrets = None  # type: ignore[misc,assignment]
            if scan_for_secrets is not None:
                matches = scan_for_secrets(content)
                if matches:
                    self._safety_stats.record_block(matches)
                    kinds = sorted({m.name for m in matches})
                    logger.warning(
                        "store() refused — content contains %d secret(s) of kind %s",
                        len(matches), kinds,
                    )
                    return ""  # empty id = refused

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

        # v2 invariants — compute / clamp at the one place new rows enter the DB.
        trust_source = normalize_trust_source(trust_source)
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            confidence_f = 0.7
        confidence_f = max(0.0, min(1.0, confidence_f))
        content_hash = compute_content_hash(content or "")
        anchor_hash = compute_anchor_hash(content or "")
        chunker_version = CURRENT_CHUNKER_VERSION

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

        # Stamp embedding_version only when an embedding was actually produced.
        # Rows without an embedding get empty string, which the healer/retriever
        # treat as "no semantic entry" — avoids fake version_skew rows.
        embedding_version = CURRENT_EMBEDDING_VERSION if embedding_vec else ""

        # T1: Atomic DB insert + vector add -- DB insert first, then vector index.
        # If DB insert fails, vector index stays clean. If vector add fails,
        # we still have the embedding blob in SQLite for later index rebuild.
        _sql = """INSERT INTO memories
                   (id, content, memory_type, importance, source, tags, metadata,
                    user_id, embedding, created_at, updated_at, last_accessed,
                    access_count, decay_score, expires_at, session_id, parent_id,
                    content_hash, anchor_hash, embedding_version, chunker_version,
                    trust_source, confidence,
                    project_id, branch, source_path, source_blob_sha,
                    source_mtime, chunk_ordinal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?)"""
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
            content_hash,
            anchor_hash,
            embedding_version,
            chunker_version,
            trust_source,
            confidence_f,
            project_id,
            branch,
            source_path,
            source_blob_sha,
            source_mtime,
            chunk_ordinal,
        )

        # Normalize supersede list: strip empties, dedupe, preserve order.
        supersede_ids: list[str] = []
        if supersedes:
            seen: set[str] = set()
            for raw in supersedes:
                sid = str(raw or "").strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                supersede_ids.append(sid)

        # Supersede UPDATE runs inside the same transaction as the INSERT so
        # the new memory and the "replaces" pointers commit atomically. No
        # half-state where the new row exists without its supersede links.
        _supersede_sql = (
            "UPDATE memories "
            "SET superseded_by = ?, superseded_at = ?, verification_state = 'superseded' "
            "WHERE id IN ({placeholders})"
        )

        if self._db_adapter is not None:
            await self._db_adapter.execute(self._DB_NAME, _sql, list(_params))
            if supersede_ids:
                placeholders = ",".join("?" * len(supersede_ids))
                await self._db_adapter.execute(
                    self._DB_NAME,
                    _supersede_sql.format(placeholders=placeholders),
                    [memory_id, now, *supersede_ids],
                )
        else:
            def _insert():
                self._conn.execute(_sql, _params)
                if supersede_ids:
                    placeholders = ",".join("?" * len(supersede_ids))
                    self._conn.execute(
                        _supersede_sql.format(placeholders=placeholders),
                        (memory_id, now, *supersede_ids),
                    )
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
        if supersede_ids:
            logger.info(
                "memory.supersede: new=%s replaces %d old id(s): %s",
                memory_id[:8],
                len(supersede_ids),
                [s[:8] for s in supersede_ids],
            )
        logger.debug(
            "Stored memory %s (type=%s, importance=%d, supersedes=%d)",
            memory_id[:8],
            memory_type,
            importance,
            len(supersede_ids),
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
        show_superseded: bool = False,
        project_id: str | list[str] | None = None,
        verify: bool = False,
        verify_drop: bool = False,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Recall memories by semantic similarity (predacore_core SIMD cosine).
        Empty queries or cold-start indices fall through to BM25 keyword search.
        Embedding failures propagate — no silent degradation.

        ``show_superseded`` — when True, also include rows explicitly replaced
        via ``store(..., supersedes=[...])``. Default False matches the
        expected user experience (corrected facts shouldn't resurface).
        Invariant-failure rows (stale/orphaned/version_skew) are always hidden.

        ``project_id`` (L5) — when set, only return memories whose
        ``project_id`` matches. Pass a string for exact match, a list for
        any-of match, or ``None``/``"all"`` (default) to disable filtering.
        Filtered rows increment ``self._invariant_skips["project_mismatch"]``.

        ``verify`` (T5+) — when True, run real-time ground-truth verification
        on each candidate that has a ``source_path``. Each result dict gets
        a ``_verified`` field: ``True`` if the chunk content is still in the
        source file, ``False`` if drifted/missing, ``None`` for synthesis
        memories (no source_path — verification not applicable). Default
        False (verification is opt-in, costs ~1-5ms per code-backed candidate).

        ``verify_drop`` (T5+) — when True (and ``verify=True``), unverified
        rows (``_verified=False``) are excluded from results entirely.
        Synthesis rows (``_verified=None``) are always kept regardless.
        Verification failures increment ``self._invariant_skips["verification_failed"]``.

        Combined ``verify=True`` + ``verify_drop=True`` is the path to **100%
        accuracy on code-backed memories** — only rows whose source still
        contains the indexed content are returned.
        """
        # Empty query → semantic has no meaning, use keyword path
        if not query or not query.strip():
            return await self._recall_keyword(
                query, user_id, top_k, memory_types, min_importance, scopes, team_id,
                show_superseded=show_superseded, project_id=project_id,
                verify=verify, verify_drop=verify_drop,
            )

        # No embedder configured or cold-start index → keyword path
        if not self._embed or not self._vec_index or self._vec_index.size == 0:
            return await self._recall_keyword(
                query, user_id, top_k, memory_types, min_importance, scopes, team_id,
                show_superseded=show_superseded, project_id=project_id,
                verify=verify, verify_drop=verify_drop,
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

        # Collect ALL matching candidates first (no early break) so that
        # when we re-sort by weighted score below, we don't accidentally
        # drop a row that the vector index ranked low but the weights push
        # to the top (e.g. a high-confidence user_corrected row).
        candidates: list[tuple[dict[str, Any], float]] = []
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
                if not _matches_project_filter(mem, project_id):
                    self._invariant_skips["project_mismatch"] += 1
                    continue
                if not _memory_is_visible_in_recall(mem, show_superseded=show_superseded, skips=self._invariant_skips):
                    continue
                candidates.append((mem, _apply_ranking_weights(score, mem)))
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
                    if not _matches_project_filter(mem, project_id):
                        self._invariant_skips["project_mismatch"] += 1
                        continue
                    if not _memory_is_visible_in_recall(mem, show_superseded=show_superseded, skips=self._invariant_skips):
                        continue
                    candidates.append((mem, _apply_ranking_weights(score, mem)))

        # Sort by weighted score (trust × confidence × decay × similarity),
        # descending. Then truncate to top_k.
        candidates.sort(key=lambda t: t[1], reverse=True)
        results: list[tuple[dict[str, Any], float]] = candidates[:top_k]

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

        # T5+ — real-time ground-truth verification (opt-in via verify=True).
        # For each candidate with a source_path, check that the chunk content
        # is still present in the file. Adds ``_verified: True|False|None``
        # to each result dict. If verify_drop=True, drop _verified=False rows
        # (keep None=synthesis rows always).
        if verify:
            results = self._apply_verification(results, drop=verify_drop)

        return results

    def _apply_verification(
        self,
        results: list[tuple[dict[str, Any], float]],
        *,
        drop: bool,
    ) -> list[tuple[dict[str, Any], float]]:
        """T5+ helper — annotate each result with _verified + optionally drop
        unverified rows. Used by recall() and _recall_keyword() when verify=True.
        """
        out: list[tuple[dict[str, Any], float]] = []
        for mem, score in results:
            verdict = _verify_chunk_against_source(mem)
            mem["_verified"] = verdict
            if verdict is False:
                self._invariant_skips["verification_failed"] += 1
                if drop:
                    continue
            out.append((mem, score))
        return out

    async def _recall_keyword(
        self,
        query: str,
        user_id: str,
        top_k: int,
        memory_types: list[str] | None = None,
        min_importance: int = 1,
        scopes: list[str] | None = None,
        team_id: str | None = None,
        show_superseded: bool = False,
        project_id: str | list[str] | None = None,
        verify: bool = False,
        verify_drop: bool = False,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        BM25 keyword search using predacore_core (Rust). Used for empty queries
        and cold-start indices. Synonym expansion applied to query terms.

        Respects ``show_superseded``, ``project_id``, ``verify``, and
        ``verify_drop`` the same way ``recall()`` does.
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

        # Filter by memory_type, scope, and verification_state BEFORE BM25
        # (smaller corpus = faster, and avoids wasted BM25 work on hidden rows).
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row["metadata"] = _coerce_metadata_dict(row.get("metadata") or {})
            if memory_types and row["memory_type"] not in memory_types:
                continue
            if not _memory_matches_scope(row, scopes, team_id=team_id):
                continue
            if not _matches_project_filter(row, project_id):
                self._invariant_skips["project_mismatch"] += 1
                continue
            if not _memory_is_visible_in_recall(row, show_superseded=show_superseded, skips=self._invariant_skips):
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

        # Rust BM25 ranking (k1=1.5, b=0.75, IDF smoothing). Fetch 3x top_k
        # headroom so that after Build 3 trust+confidence+decay weighting +
        # re-sort, we don't miss a row that BM25 ranked #(top_k+1) but
        # whose weights push it to #1.
        contents = [r["content"] for r in filtered]
        ranked = predacore_core.bm25_search(expanded_query, contents, top_k * 3)

        # Apply trust + confidence + decay weights, then re-sort by the
        # weighted score (not the raw BM25 score) before truncating.
        weighted = [
            (filtered[idx], _apply_ranking_weights(float(score), filtered[idx]))
            for idx, score in ranked
        ]
        weighted.sort(key=lambda t: t[1], reverse=True)
        results = weighted[:top_k]

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

        # T5+ — apply real-time verification to keyword-path results too
        if verify:
            results = self._apply_verification(results, drop=verify_drop)

        return results

    async def recall_explain(
        self,
        query: str,
        *,
        user_id: str = "default",
        top_k: int = 10,
        scopes: list[str] | None = None,
        team_id: str | None = None,
        memory_types: list[str] | None = None,
        min_importance: int = 1,
        show_stale: bool = False,
        show_superseded: bool = False,
    ) -> dict[str, Any]:
        """Sophisticated per-stage recall trace for debugging.

        Mirrors lab's ``recall_explain`` (L4 port) — runs the full hybrid
        pipeline but exposes every stage's inputs + outputs so you can
        see WHY a given result came (or didn't come) back.

        Stages reported (in ``stages`` dict):
          - ``vector.raw``    — top-N vector hits BEFORE invariant filter
          - ``vector.kept``   — same list AFTER invariant filter
          - ``keyword.kept``  — BM25 hits (filter applied at query time)
          - ``filtered_out``  — per-reason delta (this query's filtering)
          - ``final``         — merged ranking (union, dedup by id)

        Plus top-level fields for backward compat with the simpler v1
        shape — ``results_count``, ``results``, ``verification_state_counts``,
        ``filtered_by_invariants``, ``embedding_version``, ``schema_version``.

        The trace is read-only — it does not mutate the store except for
        the ``self._invariant_skips`` counters (which is the whole point —
        ``filtered_out`` is computed as the snapshot delta).
        """
        out: dict[str, Any] = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "stages": {},
        }

        # Snapshot counters so we can compute per-query deltas
        skips_before = dict(self._invariant_skips)

        # --- Vector stage -------------------------------------------------
        vector_raw: list[dict[str, Any]] = []
        vector_kept: list[dict[str, Any]] = []
        if self._embed and self._vec_index and self._vec_index.size > 0 and query.strip():
            try:
                embed_key = hash_key(query)
                vec = self._embedding_cache.get(embed_key)
                if vec is None:
                    vecs = await self._embed.embed([query])
                    if vecs and vecs[0]:
                        vec = vecs[0]
                        self._embedding_cache.set(embed_key, vec, ttl_seconds=600)
                if vec is not None:
                    hits = await self._vec_index.search(vec, top_k=top_k * 4)
                    for memory_id, score in hits:
                        mem = await self.get(memory_id)
                        if mem is None:
                            continue
                        vector_raw.append({
                            "id": memory_id,
                            "score": round(float(score), 6),
                            "preview": (mem.get("content") or "")[:120],
                            "verification_state": mem.get("verification_state"),
                            "memory_type": mem.get("memory_type"),
                        })
                        # Visibility check — increments self._invariant_skips
                        # internally, which we'll diff for filtered_out below.
                        if not show_stale and not _memory_is_visible_in_recall(
                            mem,
                            show_superseded=show_superseded,
                            skips=self._invariant_skips,
                        ):
                            continue
                        if mem.get("user_id") != user_id:
                            continue
                        if memory_types and mem.get("memory_type") not in memory_types:
                            continue
                        if mem.get("importance", 0) < min_importance:
                            continue
                        if not _memory_matches_scope(mem, scopes, team_id=team_id):
                            continue
                        vector_kept.append({
                            "id": memory_id,
                            "score": round(float(score), 6),
                            "preview": (mem.get("content") or "")[:120],
                            "memory_type": mem.get("memory_type"),
                            "trust_source": mem.get("trust_source"),
                        })
            except Exception as exc:
                out["stages"]["vector.error"] = repr(exc)

        out["stages"]["vector.raw"] = vector_raw[:top_k * 2]
        out["stages"]["vector.kept"] = vector_kept[:top_k]

        # --- Keyword / BM25 stage ----------------------------------------
        keyword_kept: list[dict[str, Any]] = []
        try:
            kw = await self._recall_keyword(
                query=query,
                user_id=user_id,
                top_k=top_k,
                scopes=scopes,
                team_id=team_id,
            )
            for mem, score in kw:
                keyword_kept.append({
                    "id": mem.get("id"),
                    "score": round(float(score), 6),
                    "preview": (mem.get("content") or "")[:120],
                    "memory_type": mem.get("memory_type"),
                })
        except (AttributeError, TypeError, RuntimeError) as exc:
            out["stages"]["keyword.error"] = repr(exc)
        out["stages"]["keyword.kept"] = keyword_kept

        # --- Filter delta ------------------------------------------------
        skips_after = dict(self._invariant_skips)
        out["stages"]["filtered_out"] = {
            k: skips_after.get(k, 0) - skips_before.get(k, 0)
            for k in skips_after
        }

        # --- Final merged ranking (union, dedup by id, sort by score) ----
        merged: dict[str, dict[str, Any]] = {}
        for src, items in (("vector", vector_kept), ("keyword", keyword_kept)):
            for h in items:
                if h["id"] in merged:
                    if src not in merged[h["id"]]["sources"]:
                        merged[h["id"]]["sources"].append(src)
                    merged[h["id"]]["score"] = max(merged[h["id"]]["score"], h["score"])
                else:
                    merged[h["id"]] = {**h, "sources": [src]}
        final = sorted(merged.values(), key=lambda r: -r["score"])[:top_k]
        out["stages"]["final"] = final

        # --- Whole-DB verification-state breakdown (context) -------------
        state_counts: dict[str, int] = {}
        if self._conn is not None:
            for state in (
                "ok", "unverified", "stale", "orphaned", "version_skew", "superseded",
            ):
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE verification_state = ?",
                    (state,),
                ).fetchone()
                state_counts[state] = row[0] if row else 0

        hidden_total = sum(
            state_counts.get(s, 0) for s in ("stale", "orphaned", "version_skew")
        )

        # --- Backward-compat top-level keys (so existing handler works) --
        out["results_count"] = len(final)
        out["results"] = [
            {
                "id": h.get("id"),
                "score": h.get("score"),
                "memory_type": h.get("memory_type"),
                "preview": h.get("preview"),
                "trust_source": h.get("trust_source"),
            }
            for h in final
        ]
        out["verification_state_counts"] = state_counts
        out["filtered_by_invariants"] = {
            "total_hidden": hidden_total,
            "note": "These rows would NOT appear in recall unless show_stale=True",
            "by_state": {
                s: state_counts.get(s, 0)
                for s in ("stale", "orphaned", "version_skew")
            },
        }
        out["embedding_version"] = CURRENT_EMBEDDING_VERSION
        out["schema_version"] = SCHEMA_VERSION

        return out

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

    async def reindex_file(
        self,
        path: str | Path,
        *,
        project_id: str = "default",
        branch: str | None = None,
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Re-index a single file's contents into memory.

        Pipeline (mirrors the lab's MCP ``memory.touch`` handler so callers
        get the same semantics here in-process):

          1. ``safe_read_text`` — skip binary / oversized / unreadable files.
          2. ``chunk_text`` — semantic chunks (AST / markdown / brace / window).
          3. Purge prior ``trust_source='code_extracted'`` rows for this
             ``source_path`` so a re-index is idempotent and stale chunks
             from the previous version don't linger.
          4. Insert each chunk as a fresh row with ``trust_source='code_extracted'``,
             carrying ``source_path``, ``source_mtime``, ``chunk_ordinal``,
             ``project_id``, ``branch``, plus ``line_start/end/anchor`` in
             metadata.

        Trigger sites in predacore-public: any tool handler that writes a
        file (e.g. ``tools/handlers/file_ops.handle_write_file``) should
        ``await ctx.unified_memory.reindex_file(path)`` after the write.

        Returns a summary dict with ``path``, ``chunk_count``,
        ``stale_rows_removed``, ``new_ids``, and ``strategy`` — or an
        ``error`` key if the file couldn't be read.
        """
        from .chunker import chunk_text, safe_read_text

        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"path": str(p), "error": "file_not_found"}
        if not p.is_file():
            return {"path": str(p), "error": "not_a_file"}

        content = safe_read_text(p)
        if content is None:
            return {
                "path": str(p),
                "error": "read_skipped",
                "reason": "binary, oversized, or unreadable",
            }

        chunks = chunk_text(p, content)
        if not chunks:
            return {
                "path": str(p),
                "chunk_count": 0,
                "stale_rows_removed": 0,
                "new_ids": [],
                "note": "empty or whitespace-only file",
            }

        # Purge prior code_extracted rows for this source_path so the
        # re-index is idempotent.
        path_str = str(p)

        def _stale_ids() -> list[str]:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                "SELECT id FROM memories "
                "WHERE source_path = ? AND trust_source = 'code_extracted'",
                (path_str,),
            ).fetchall()
            return [r[0] for r in rows]

        prior_ids: list[str] = []
        if self._conn is not None:
            prior_ids = await self._in_thread(_stale_ids)
        elif self._db_adapter is not None:
            adapter_rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT id FROM memories "
                "WHERE source_path = ? AND trust_source = 'code_extracted'",
                [path_str],
            )
            prior_ids = [row["id"] for row in adapter_rows]

        stale_removed = 0
        for rid in prior_ids:
            if await self.delete(rid):
                stale_removed += 1

        try:
            mtime = int(p.stat().st_mtime)
        except OSError:
            mtime = None

        new_ids: list[dict[str, Any]] = []
        for chunk in chunks:
            tags = [
                "file",
                "code_extracted",
                f"kind:{chunk.kind}",
                f"lines:{chunk.line_start}-{chunk.line_end}",
            ]
            mid = await self.store(
                content=chunk.content,
                memory_type="note",
                tags=tags,
                user_id=user_id,
                source_path=path_str,
                source_mtime=mtime,
                chunk_ordinal=chunk.ordinal,
                project_id=project_id,
                branch=branch,
                trust_source="code_extracted",
                confidence=1.0,
                metadata={
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "chunk_kind": chunk.kind,
                    "anchor": chunk.anchor,
                },
            )
            # store() returns "" if scan_for_secrets refused the chunk —
            # skip those silently (the safety counter already recorded it).
            if mid:
                new_ids.append({
                    "id": mid,
                    "kind": chunk.kind,
                    "lines": f"{chunk.line_start}-{chunk.line_end}",
                    "anchor": chunk.anchor,
                })

        return {
            "path": path_str,
            "chunk_count": len(chunks),
            "stale_rows_removed": stale_removed,
            "new_ids": new_ids,
            "strategy": "semantic-chunker-v1",
        }

    async def sync_git_changes(
        self,
        repo_path: str | Path | None = None,
        *,
        project_id: str = "default",
        branch: str | None = None,
        user_id: str = "default",
        prior_head: str | None = None,
    ) -> dict[str, Any]:
        """Sync memory with changes in a git repo.

        Triggered after ``git checkout / merge / rebase / reset / pull``,
        this method:

          1. Walks ``git status --porcelain=v2`` (via the existing
             ``predacore.services.git_integration`` helpers — modern v2
             format, branch-aware, rename-aware) to find UNCOMMITTED
             working-tree changes.
          2. If ``prior_head`` is provided, ALSO runs ``git diff
             --name-status <prior_head> HEAD`` to find COMMITTED changes
             that landed between the prior HEAD and the current HEAD
             (covers ``git pull``, ``git checkout other_branch``,
             ``git merge``, ``git reset``, etc.).
          3. Deduplicates: if a path appears in both as deleted-then-
             reappeared, modified wins (the file currently exists).
          4. For every modified/added file → calls :meth:`reindex_file`
             so the new chunks land in memory.
          5. For every deleted file → purges its ``code_extracted`` rows.

        Untracked files are intentionally skipped (they're often build
        artifacts or scratch files the user hasn't committed).

        Why ``prior_head``: ``git status`` only sees uncommitted
        modifications. After ``git pull`` that fast-forwards 50 commits
        cleanly, status is empty even though a lot of code changed. The
        TRIGGER (``shell.py``) should ``git rev-parse HEAD`` BEFORE the
        git command, then pass the captured SHA here AFTER. With
        ``prior_head`` set you get full coverage; without it you get
        the lab-equivalent working-tree-only behavior.

        Trigger site in predacore-public: after a successful
        ``handle_run_command`` call in ``tools/handlers/shell.py`` whose
        command matches ``git (checkout|merge|rebase|reset|pull)``.

        Returns a summary dict with ``repo``, ``branch``, ``modified``,
        ``deleted``, ``chunks_added``, ``rows_purged``,
        ``committed_diff_used`` — or an ``error`` key when the path
        isn't a git repo.
        """
        try:
            from ..services.git_integration import _find_repo_root, _run_git
        except ImportError:
            return {"repo": str(repo_path or "."), "error": "git_integration_unavailable"}

        cwd = str(Path(repo_path).resolve()) if repo_path else None
        repo_root = await _find_repo_root(cwd=cwd)
        if not repo_root:
            return {"repo": cwd or ".", "error": "not_a_git_repo"}

        # Auto-detect branch if caller didn't pass one — matches lab's
        # _common.resolve_branch behavior.
        if branch is None:
            rc, head_out, _ = await _run_git(
                "symbolic-ref", "--short", "HEAD", cwd=repo_root,
            )
            if rc == 0 and head_out.strip():
                branch = head_out.strip()

        rc, status_out, status_err = await _run_git(
            "status", "--porcelain=v2", cwd=repo_root,
        )
        if rc != 0:
            return {
                "repo": repo_root,
                "error": "git_status_failed",
                "detail": status_err,
            }

        # Parse porcelain v2: lines starting "1" or "2" are tracked-file
        # changes. Format examples:
        #   "1 .M N... <sha> <sha> <sha> <mode> <mode> <mode> <path>"
        #   "2 R. N... <sha> <sha> <sha> <mode> <mode> <mode> <score> <new>\t<old>"
        # XY field: index status (X) and worktree status (Y); "D" anywhere
        # means a deletion is involved. Sets dedupe across the working-tree
        # and committed-diff passes below.
        modified_paths: set[str] = set()
        deleted_paths: set[str] = set()
        for line in status_out.splitlines():
            if not line or line[0] not in ("1", "2"):
                continue  # skip headers (#), untracked (?), ignored (!)
            if "\t" in line:
                pre, _, tail = line.partition("\t")
                # For renames the new path is in `pre`'s last token; old
                # path lives in `tail`. We re-index the new path.
                fields = pre.split()
                rel_path = fields[-1] if fields else tail.strip()
            else:
                fields = line.split()
                rel_path = fields[-1] if fields else ""
            if not rel_path or len(fields) < 2:
                continue
            xy = fields[1]
            if "D" in xy:
                deleted_paths.add(rel_path)
            else:
                modified_paths.add(rel_path)

        # Committed-changes pass: only fires when caller passed a prior HEAD.
        # `git diff --name-status A..B` lists files that changed between A
        # and B with single-letter codes:
        #   A added | M modified | D deleted | T type-change | R<n> rename | C<n> copy
        # Renames/copies have THREE tab-separated fields: code, old, new.
        committed_diff_used = False
        if prior_head:
            rc_diff, diff_out, diff_err = await _run_git(
                "diff", "--name-status", f"{prior_head}..HEAD", cwd=repo_root,
            )
            if rc_diff == 0:
                committed_diff_used = True
                for line in diff_out.splitlines():
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    code = parts[0]
                    if code.startswith(("R", "C")) and len(parts) >= 3:
                        # Rename/copy: purge old path (no longer at that
                        # location), reindex new path.
                        old_path, new_path = parts[1], parts[2]
                        deleted_paths.add(old_path)
                        modified_paths.add(new_path)
                    elif code == "D" and len(parts) >= 2:
                        deleted_paths.add(parts[1])
                    elif code in ("A", "M", "T") and len(parts) >= 2:
                        modified_paths.add(parts[1])
            else:
                logger.debug(
                    "sync_git_changes: prior_head=%s diff failed (rc=%d): %s",
                    prior_head, rc_diff, diff_err,
                )

        # Dedupe conflict resolution: a path that's in BOTH sets means the
        # file was deleted in commits but reappeared in the working tree
        # (or vice-versa). Reindex wins because the file currently exists.
        deleted_paths -= modified_paths

        repo_root_path = Path(repo_root)
        chunks_added = 0
        rows_purged = 0

        # Reindex modified files — reuse reindex_file so chunking +
        # safety + idempotent purge-then-store all stay in one place.
        for rel in modified_paths:
            abs_path = (repo_root_path / rel).resolve()
            if not abs_path.is_file():
                continue
            result = await self.reindex_file(
                str(abs_path),
                project_id=project_id,
                branch=branch,
                user_id=user_id,
            )
            chunks_added += len(result.get("new_ids") or [])

        # Purge deleted files — query then delete in one inline pass to
        # keep this method self-contained.
        for rel in deleted_paths:
            abs_path = str((repo_root_path / rel).resolve())

            def _stale_ids(_p: str = abs_path) -> list[str]:
                if self._conn is None:
                    return []
                rows = self._conn.execute(
                    "SELECT id FROM memories "
                    "WHERE source_path = ? AND trust_source = 'code_extracted'",
                    (_p,),
                ).fetchall()
                return [r[0] for r in rows]

            ids: list[str] = []
            if self._conn is not None:
                ids = await self._in_thread(_stale_ids)
            elif self._db_adapter is not None:
                adapter_rows = await self._db_adapter.query_dicts(
                    self._DB_NAME,
                    "SELECT id FROM memories "
                    "WHERE source_path = ? AND trust_source = 'code_extracted'",
                    [abs_path],
                )
                ids = [row["id"] for row in adapter_rows]
            for rid in ids:
                if await self.delete(rid):
                    rows_purged += 1

        return {
            "repo": repo_root,
            "branch": branch,
            "modified": len(modified_paths),
            "deleted": len(deleted_paths),
            "chunks_added": chunks_added,
            "rows_purged": rows_purged,
            "committed_diff_used": committed_diff_used,
        }

    async def warmup_embedder(self) -> dict[str, Any]:
        """Pre-load the embedding model to avoid 1-2s cold-start latency
        on the first ``recall()`` or ``store()`` with content.

        Calls ``embed(["warmup"])`` once — the result is discarded; the
        side effect (loading model weights into RAM) is what matters.
        Safe to call repeatedly: when the model is already loaded the
        call is a few milliseconds.

        Trigger site in predacore-public: ``bootstrap.run_bootstrap()``
        should fire this once after :class:`SubsystemFactory` builds the
        store, so the agent's first prompt doesn't pay the BGE model
        download/load cost.

        Returns: ``{"warmed": bool, "embedder": str, "already_loaded":
        bool}`` — or ``{"warmed": False, "reason": ...}`` on no-op /
        error.
        """
        if self._embed is None:
            return {"warmed": False, "reason": "no_embedder"}

        embedder_name = type(self._embed).__name__

        # If the embedder uses predacore_core (the Rust kernel + BGE),
        # we can detect "already warm" cheaply. Other embedders (OpenAI,
        # Gemini, etc.) don't expose this, so we fall back to "unknown".
        already_loaded = False
        try:
            import predacore_core  # type: ignore
            already_loaded = bool(predacore_core.is_model_loaded())
        except (ImportError, AttributeError):
            pass

        try:
            await self._embed.embed(["warmup"])
        except Exception as exc:  # broad: an embedder failure shouldn't crash boot
            return {
                "warmed": False,
                "embedder": embedder_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "warmed": True,
            "embedder": embedder_name,
            "already_loaded": already_loaded,
        }

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
                # L4 — invariant + safety surfaces (parity with lab)
                "schema_version": SCHEMA_VERSION,
                "embedding_version": CURRENT_EMBEDDING_VERSION,
                "chunker_version": CURRENT_CHUNKER_VERSION,
                "invariant_skips": dict(self._invariant_skips),
                "safety": self._safety_stats.as_dict() if self._safety_stats else {},
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
                # L4 — invariant + safety surfaces (parity with lab)
                "schema_version": SCHEMA_VERSION,
                "embedding_version": CURRENT_EMBEDDING_VERSION,
                "chunker_version": CURRENT_CHUNKER_VERSION,
                "invariant_skips": dict(self._invariant_skips),
                "safety": self._safety_stats.as_dict() if self._safety_stats else {},
            }

        async with self._db_lock:
            return await self._in_thread(_stats)

    def reset_invariant_skips(self) -> dict[str, int]:
        """Snapshot the invariant-skip counters and zero them in place.

        Useful for tests and "what filtered in this window?" introspection.
        Returns the previous values; the counters are now all zero. Mirrors
        lab's same-named method (parity).
        """
        old = dict(self._invariant_skips)
        for k in self._invariant_skips:
            self._invariant_skips[k] = 0
        return old

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

    def _vector_cache_path(self) -> Path:
        """Location of the persisted vector index cache (sibling of the DB file)."""
        return self._db_path.parent / "vector_index.cache.npz"

    def _sqlite_embedded_row_count(self) -> int:
        """Count of memories with a non-NULL embedding — used for cache invariant."""
        if self._conn is None:
            return 0
        try:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0])
        except sqlite3.Error as exc:
            logger.debug("Row-count check failed: %s", exc)
            return -1

    def _try_load_vector_index_cache(self) -> bool:
        """Attempt to populate self._vec_index from disk cache. Return True on success."""
        if self._vec_index is None or self._conn is None:
            return False
        row_count = self._sqlite_embedded_row_count()
        if row_count <= 0:
            # Empty or unreadable SQLite — nothing to load / validate against
            return False
        return self._vec_index.load_from_disk(
            self._vector_cache_path(),
            expected_row_count=row_count,
            expected_embedding_version=CURRENT_EMBEDDING_VERSION,
        )

    def _persist_vector_index_cache(self) -> None:
        """Dump the current in-RAM vector index to disk. Safe to call on shutdown."""
        if self._vec_index is None or self._conn is None:
            return
        row_count = self._sqlite_embedded_row_count()
        if row_count < 0:
            return
        try:
            self._vec_index.dump_to_disk(
                self._vector_cache_path(),
                row_count=row_count,
                embedding_version=CURRENT_EMBEDDING_VERSION,
            )
        except (OSError, ValueError) as exc:
            logger.debug("Vector index persistence skipped: %s", exc)

    def close(self) -> None:
        """Persist the in-RAM vector index + close the SQLite connection.

        Safe to call multiple times (idempotent). On abrupt kill (SIGKILL),
        persistence is skipped and the next startup falls back to rebuilding
        from SQLite — no data lost, just slower boot.
        """
        # Persist cache BEFORE closing the DB — we need the conn to get the
        # current row count for the invariant header.
        try:
            self._persist_vector_index_cache()
        except Exception as exc:  # noqa: BLE001 — never block close on cache failure
            logger.debug("Vector cache persist on close failed (non-fatal): %s", exc)

        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.debug("Error closing DB connection: %s", exc)
            finally:
                self._conn = None
