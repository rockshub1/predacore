"""
PredaCore Outcome Store — Task outcome tracking with SQLite backend.

Records every agent interaction: tools used, model/provider, latency,
token counts, success/failure, and optional user feedback.  Powers
Phase 5 self-improvement by providing failure-pattern analysis.

Usage:
    store = OutcomeStore(config)
    await store.record(outcome)
    failures = await store.get_failure_patterns(window_hours=24)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    """A single agent interaction outcome."""

    task_id: str = ""
    user_id: str = ""
    user_message: str = ""
    response_summary: str = ""
    tools_used: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    provider_used: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    token_count_prompt: int = 0
    token_count_completion: int = 0
    iterations: int = 0
    success: bool = True
    error: str | None = None
    user_feedback: str | None = None  # "good" | "bad" | None
    persona_drift_score: float = 0.0
    persona_drift_regens: int = 0
    session_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Feedback detection
# ---------------------------------------------------------------------------

_POSITIVE_PATTERNS = (
    "thanks",
    "thank you",
    "great",
    "perfect",
    "awesome",
    "nice",
    "good job",
    "well done",
    "exactly",
    "that's right",
    "correct",
    "helpful",
    "love it",
    "amazing",
    "excellent",
)

_NEGATIVE_PATTERNS = (
    "wrong",
    "incorrect",
    "bad answer",
    "that's not right",
    "not what i asked",
    "useless",
    "terrible",
    "awful",
    "completely wrong",
    "no that's wrong",
    "hallucinating",
    "made up",
    "you're wrong",
)


def detect_feedback(message: str) -> str | None:
    """
    Detect implicit user feedback from a message.

    Returns "good", "bad", or None.
    """
    lowered = (message or "").strip().lower()
    if not lowered:
        return None

    # Explicit /feedback command
    if lowered.startswith("/feedback "):
        arg = lowered[10:].strip()
        if arg in ("good", "bad"):
            return arg
        return None

    # Short messages are more likely to be feedback
    if len(lowered) > 200:
        return None

    for pat in _NEGATIVE_PATTERNS:
        if pat in lowered:
            return "bad"

    for pat in _POSITIVE_PATTERNS:
        if pat in lowered:
            return "good"

    return None


# ---------------------------------------------------------------------------
# OutcomeStore
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS outcomes (
    task_id         TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    user_message    TEXT NOT NULL DEFAULT '',
    response_summary TEXT NOT NULL DEFAULT '',
    tools_used      TEXT NOT NULL DEFAULT '[]',
    tool_errors     TEXT NOT NULL DEFAULT '[]',
    provider_used   TEXT NOT NULL DEFAULT '',
    model_used      TEXT NOT NULL DEFAULT '',
    latency_ms      REAL NOT NULL DEFAULT 0,
    token_prompt    INTEGER NOT NULL DEFAULT 0,
    token_completion INTEGER NOT NULL DEFAULT 0,
    iterations      INTEGER NOT NULL DEFAULT 0,
    success         INTEGER NOT NULL DEFAULT 1,
    error           TEXT,
    user_feedback   TEXT,
    drift_score     REAL NOT NULL DEFAULT 0,
    drift_regens    INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_outcomes_user ON outcomes(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_created ON outcomes(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_success ON outcomes(success);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_provider ON outcomes(provider_used);",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_feedback ON outcomes(user_feedback);",
]

_CREATE_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


class OutcomeStore:
    """SQLite-backed store for agent interaction outcomes."""

    def __init__(self, db_path: str | Path, *, db_adapter=None):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._db_adapter = db_adapter  # Optional DBAdapter (async)
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        # Note: if db_adapter is provided, schema init is deferred to
        # ensure_schema_async() which must be awaited after construction.
        if self._db_adapter:
            return
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            conn.execute(idx)
        conn.execute(_CREATE_VERSION)
        # Check schema version
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
        conn.commit()
        logger.debug("OutcomeStore initialized at %s", self._db_path)

    async def ensure_schema_async(self) -> None:
        """Initialise schema via DBAdapter. Call once after construction when adapter is set."""
        if not self._db_adapter:
            return
        schema_sql = _CREATE_TABLE + "\n"
        for idx in _CREATE_INDEXES:
            schema_sql += idx + "\n"
        schema_sql += _CREATE_VERSION + "\n"
        await self._db_adapter.executescript("outcomes", schema_sql)
        rows = await self._db_adapter.query("outcomes", "SELECT version FROM schema_version LIMIT 1")
        if not rows:
            await self._db_adapter.execute(
                "outcomes",
                "INSERT INTO schema_version (version) VALUES (?)",
                [_SCHEMA_VERSION],
            )
        logger.debug("OutcomeStore schema initialized via DBAdapter")

    # ── Thread helper (SQLite is synchronous) ───────────────────────

    async def _in_thread(self, fn, *args, **kwargs):
        """Run a synchronous DB function in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ── Core operations ───────────────────────────────────────────────

    def _record_sync(self, outcome: TaskOutcome) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO outcomes
               (task_id, user_id, user_message, response_summary,
                tools_used, tool_errors, provider_used, model_used,
                latency_ms, token_prompt, token_completion, iterations,
                success, error, user_feedback, drift_score, drift_regens,
                session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                outcome.task_id,
                outcome.user_id,
                outcome.user_message[:2000],
                outcome.response_summary[:2000],
                json.dumps(outcome.tools_used),
                json.dumps(outcome.tool_errors),
                outcome.provider_used,
                outcome.model_used,
                outcome.latency_ms,
                outcome.token_count_prompt,
                outcome.token_count_completion,
                outcome.iterations,
                int(outcome.success),
                outcome.error,
                outcome.user_feedback,
                outcome.persona_drift_score,
                outcome.persona_drift_regens,
                outcome.session_id,
                outcome.timestamp,
            ),
        )
        conn.commit()

    async def record(self, outcome: TaskOutcome) -> None:
        """Record a task outcome."""
        if self._db_adapter:
            await self._db_adapter.execute(
                "outcomes",
                """INSERT OR REPLACE INTO outcomes
                   (task_id, user_id, user_message, response_summary,
                    tools_used, tool_errors, provider_used, model_used,
                    latency_ms, token_prompt, token_completion, iterations,
                    success, error, user_feedback, drift_score, drift_regens,
                    session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    outcome.task_id,
                    outcome.user_id,
                    outcome.user_message[:2000],
                    outcome.response_summary[:2000],
                    json.dumps(outcome.tools_used),
                    json.dumps(outcome.tool_errors),
                    outcome.provider_used,
                    outcome.model_used,
                    outcome.latency_ms,
                    outcome.token_count_prompt,
                    outcome.token_count_completion,
                    outcome.iterations,
                    int(outcome.success),
                    outcome.error,
                    outcome.user_feedback,
                    outcome.persona_drift_score,
                    outcome.persona_drift_regens,
                    outcome.session_id,
                    outcome.timestamp,
                ],
            )
        else:
            await self._in_thread(self._record_sync, outcome)
        logger.debug(
            "Recorded outcome %s (success=%s, tools=%d)",
            outcome.task_id,
            outcome.success,
            len(outcome.tools_used),
        )

    def _update_feedback_sync(self, user_id: str, feedback: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute(
            """UPDATE outcomes SET user_feedback = ?
               WHERE rowid = (
                   SELECT rowid FROM outcomes
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT 1
               )""",
            (feedback, user_id),
        )
        conn.commit()
        return cur.rowcount > 0

    async def update_feedback(self, user_id: str, feedback: str) -> bool:
        """Update the most recent outcome for a user with feedback."""
        if self._db_adapter:
            result = await self._db_adapter.execute(
                "outcomes",
                """UPDATE outcomes SET user_feedback = ?
                   WHERE rowid = (
                       SELECT rowid FROM outcomes
                       WHERE user_id = ?
                       ORDER BY created_at DESC
                       LIMIT 1
                   )""",
                [feedback, user_id],
            )
            updated = result.get("rowcount", 0) > 0
        else:
            updated = await self._in_thread(self._update_feedback_sync, user_id, feedback)
        if updated:
            logger.info("Updated feedback to '%s' for user %s", feedback, user_id)
        return updated

    # ── Analytics queries ─────────────────────────────────────────────

    def _get_failure_patterns_sync(self, window_hours: float) -> list[dict[str, Any]]:
        cutoff = time.time() - (window_hours * 3600)
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT tool_errors, error, created_at
               FROM outcomes
               WHERE success = 0 AND created_at >= ?
               ORDER BY created_at DESC""",
            (cutoff,),
        ).fetchall()

        tool_failures: dict[str, dict[str, Any]] = {}
        for row in rows:
            errors = json.loads(row["tool_errors"] or "[]")
            for err_entry in errors:
                if ": " in err_entry:
                    tool, msg = err_entry.split(": ", 1)
                else:
                    tool, msg = "unknown", err_entry
                if tool not in tool_failures:
                    tool_failures[tool] = {
                        "tool_name": tool,
                        "failure_count": 0,
                        "error_samples": [],
                        "last_seen": row["created_at"],
                    }
                tool_failures[tool]["failure_count"] += 1
                if len(tool_failures[tool]["error_samples"]) < 3:
                    tool_failures[tool]["error_samples"].append(msg[:200])

        return sorted(
            tool_failures.values(),
            key=lambda x: x["failure_count"],
            reverse=True,
        )

    async def get_failure_patterns(
        self, window_hours: float = 24
    ) -> list[dict[str, Any]]:
        """Analyze failure patterns in the given time window."""
        if self._db_adapter:
            cutoff = time.time() - (window_hours * 3600)
            rows = await self._db_adapter.query_dicts(
                "outcomes",
                """SELECT tool_errors, error, created_at
                   FROM outcomes
                   WHERE success = 0 AND created_at >= ?
                   ORDER BY created_at DESC""",
                [cutoff],
            )
            tool_failures: dict[str, dict[str, Any]] = {}
            for row in rows:
                errors = json.loads(row.get("tool_errors") or "[]")
                for err_entry in errors:
                    if ": " in err_entry:
                        tool, msg = err_entry.split(": ", 1)
                    else:
                        tool, msg = "unknown", err_entry
                    if tool not in tool_failures:
                        tool_failures[tool] = {
                            "tool_name": tool,
                            "failure_count": 0,
                            "error_samples": [],
                            "last_seen": row["created_at"],
                        }
                    tool_failures[tool]["failure_count"] += 1
                    if len(tool_failures[tool]["error_samples"]) < 3:
                        tool_failures[tool]["error_samples"].append(msg[:200])
            return sorted(
                tool_failures.values(),
                key=lambda x: x["failure_count"],
                reverse=True,
            )
        return await self._in_thread(self._get_failure_patterns_sync, window_hours)

    def _get_tool_stats_sync(
        self, tool_name: str, window_hours: float
    ) -> dict[str, Any]:
        cutoff = time.time() - (window_hours * 3600)
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT tools_used, success, latency_ms
               FROM outcomes
               WHERE created_at >= ?""",
            (cutoff,),
        ).fetchall()

        total = 0
        successes = 0
        total_latency = 0.0
        for row in rows:
            tools = json.loads(row["tools_used"] or "[]")
            if tool_name in tools:
                total += 1
                if row["success"]:
                    successes += 1
                total_latency += row["latency_ms"]

        return {
            "tool_name": tool_name,
            "total_uses": total,
            "success_rate": (successes / total * 100) if total else 0,
            "avg_latency_ms": (total_latency / total) if total else 0,
            "window_hours": window_hours,
        }

    async def get_tool_stats(
        self, tool_name: str, window_hours: float = 168
    ) -> dict[str, Any]:
        """Get usage statistics for a specific tool."""
        return await self._in_thread(self._get_tool_stats_sync, tool_name, window_hours)

    def _get_provider_stats_sync(self, window_hours: float) -> list[dict[str, Any]]:
        cutoff = time.time() - (window_hours * 3600)
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT provider_used, model_used,
                      COUNT(*) as cnt,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok,
                      AVG(latency_ms) as avg_lat,
                      SUM(token_prompt) as sum_prompt,
                      SUM(token_completion) as sum_comp
               FROM outcomes
               WHERE created_at >= ?
               GROUP BY provider_used, model_used
               ORDER BY cnt DESC""",
            (cutoff,),
        ).fetchall()

        return [
            {
                "provider": row["provider_used"],
                "model": row["model_used"],
                "total_calls": row["cnt"],
                "success_rate": (row["ok"] / row["cnt"] * 100) if row["cnt"] else 0,
                "avg_latency_ms": round(row["avg_lat"] or 0, 1),
                "total_prompt_tokens": row["sum_prompt"] or 0,
                "total_completion_tokens": row["sum_comp"] or 0,
            }
            for row in rows
        ]

    async def get_provider_stats(
        self, window_hours: float = 24
    ) -> list[dict[str, Any]]:
        """Get per-provider usage and performance stats."""
        if self._db_adapter:
            cutoff = time.time() - (window_hours * 3600)
            rows = await self._db_adapter.query_dicts(
                "outcomes",
                """SELECT provider_used, model_used,
                          COUNT(*) as cnt,
                          SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok,
                          AVG(latency_ms) as avg_lat,
                          SUM(token_prompt) as sum_prompt,
                          SUM(token_completion) as sum_comp
                   FROM outcomes
                   WHERE created_at >= ?
                   GROUP BY provider_used, model_used
                   ORDER BY cnt DESC""",
                [cutoff],
            )
            return [
                {
                    "provider": row["provider_used"],
                    "model": row["model_used"],
                    "total_calls": row["cnt"],
                    "success_rate": (row["ok"] / row["cnt"] * 100) if row["cnt"] else 0,
                    "avg_latency_ms": round(row["avg_lat"] or 0, 1),
                    "total_prompt_tokens": row["sum_prompt"] or 0,
                    "total_completion_tokens": row["sum_comp"] or 0,
                }
                for row in rows
            ]
        return await self._in_thread(self._get_provider_stats_sync, window_hours)

    async def prune_old(self, max_age_days: int = 90) -> int:
        """Delete outcome records older than max_age_days. Returns count deleted."""
        def _prune():
            cutoff = time.time() - (max_age_days * 86400)
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM outcomes WHERE created_at < ?", (cutoff,))
            conn.commit()
            return cur.rowcount
        count = await self._in_thread(_prune)
        if count:
            logger.info("Pruned %d outcome records older than %d days", count, max_age_days)
        return count

    def close(self) -> None:
        """Close the thread-local database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _get_recent_sync(self, limit: int, user_id: str | None) -> list[dict[str, Any]]:
        conn = self._get_conn()
        if user_id:
            rows = conn.execute(
                """SELECT * FROM outcomes WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM outcomes
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def get_recent(
        self, limit: int = 20, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recent outcomes for inspection."""
        if self._db_adapter:
            if user_id:
                return await self._db_adapter.query_dicts(
                    "outcomes",
                    """SELECT * FROM outcomes WHERE user_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    [user_id, limit],
                )
            return await self._db_adapter.query_dicts(
                "outcomes",
                """SELECT * FROM outcomes
                   ORDER BY created_at DESC LIMIT ?""",
                [limit],
            )
        return await self._in_thread(self._get_recent_sync, limit, user_id)

    def _get_feedback_summary_sync(self, window_hours: float) -> dict[str, int]:
        cutoff = time.time() - (window_hours * 3600)
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT user_feedback, COUNT(*) as cnt
               FROM outcomes
               WHERE created_at >= ? AND user_feedback IS NOT NULL
               GROUP BY user_feedback""",
            (cutoff,),
        ).fetchall()
        result: dict[str, int] = {"good": 0, "bad": 0}
        for row in rows:
            fb = row["user_feedback"]
            if fb in result:
                result[fb] = row["cnt"]
        return result

    async def get_feedback_summary(self, window_hours: float = 168) -> dict[str, int]:
        """Get feedback distribution over time window."""
        return await self._in_thread(self._get_feedback_summary_sync, window_hours)

    @property
    def db_path(self) -> Path:
        return self._db_path
