"""
Persistent audit trail for the Ethical Governance Module.

SQLite-backed storage for compliance decisions with query, statistics,
and reporting capabilities. Replaces flat-file audit logging with a
structured, queryable store.
"""
from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single compliance audit record."""

    timestamp: str = ""
    component: str = ""
    event_type: str = ""
    severity: str = "INFO"  # INFO | WARNING | VIOLATION | CRITICAL
    is_compliant: bool = True
    decision: str = ""  # ALLOW | BLOCK | WARN
    justification: str = ""
    principle: str = ""  # Which EGM principle was evaluated
    details: dict[str, Any] = field(default_factory=dict)
    entry_id: int = 0  # auto-set by DB

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["details"] = json.dumps(d["details"])
        return d


# ---------------------------------------------------------------------------
# Persistent store
# ---------------------------------------------------------------------------


class PersistentAuditStore:
    """
    SQLite-backed EGM audit trail.

    Features:
    - Write compliance decisions with structured metadata
    - Query by time range, component, severity, compliance status
    - Aggregate statistics (compliance rate, violation counts)
    - Export audit reports
    - Optional DBAdapter support for remote/centralized DB access
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        component   TEXT NOT NULL DEFAULT '',
        event_type  TEXT NOT NULL DEFAULT '',
        severity    TEXT NOT NULL DEFAULT 'INFO',
        is_compliant INTEGER NOT NULL DEFAULT 1,
        decision    TEXT NOT NULL DEFAULT 'ALLOW',
        justification TEXT NOT NULL DEFAULT '',
        principle   TEXT NOT NULL DEFAULT '',
        details     TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_log(component);
    CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity);
    CREATE INDEX IF NOT EXISTS idx_audit_compliant ON audit_log(is_compliant);
    """

    _DB_NAME = "compliance"

    def __init__(
        self,
        db_path: str = "data/egm_audit.db",
        log: logging.Logger | None = None,
        db_adapter: Any = None,
    ) -> None:
        self._log = log or logger
        self._db_path = db_path
        self._db_adapter = db_adapter
        if db_adapter is None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, timeout=10)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(self._SCHEMA)
            self._conn.commit()
        else:
            self._conn = None  # type: ignore[assignment]
        self._log.info("PersistentAuditStore initialized at %s", db_path)

    async def init_schema(self) -> None:
        """Create the audit tables via the adapter (call once at startup)."""
        if self._db_adapter is not None:
            await self._db_adapter.executescript(self._DB_NAME, self._SCHEMA)

    # -- Write --------------------------------------------------------------

    def log_decision(
        self,
        component: str,
        event_type: str,
        is_compliant: bool,
        decision: str = "ALLOW",
        severity: str = "INFO",
        justification: str = "",
        principle: str = "",
        details: dict[str, Any] | None = None,
    ) -> int:
        """Record a compliance decision (sync). Returns the entry ID."""
        if self._db_adapter is not None:
            raise RuntimeError(
                "Use async log_decision_async() when db_adapter is configured"
            )
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO audit_log
               (timestamp, component, event_type, severity, is_compliant,
                decision, justification, principle, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                component,
                event_type,
                severity,
                1 if is_compliant else 0,
                decision,
                justification,
                principle,
                json.dumps(details or {}),
            ),
        )
        self._conn.commit()
        entry_id = cur.lastrowid or 0
        self._log.debug("Audit entry #%d logged for %s", entry_id, component)
        return entry_id

    async def log_decision_async(
        self,
        component: str,
        event_type: str,
        is_compliant: bool,
        decision: str = "ALLOW",
        severity: str = "INFO",
        justification: str = "",
        principle: str = "",
        details: dict[str, Any] | None = None,
    ) -> int:
        """Record a compliance decision. Uses adapter when available, else sync."""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _sql = """INSERT INTO audit_log
               (timestamp, component, event_type, severity, is_compliant,
                decision, justification, principle, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        _params = [
            ts,
            component,
            event_type,
            severity,
            1 if is_compliant else 0,
            decision,
            justification,
            principle,
            json.dumps(details or {}),
        ]
        if self._db_adapter is not None:
            result = await self._db_adapter.execute(self._DB_NAME, _sql, _params)
            entry_id = result.get("lastrowid", 0)
        else:
            cur = self._conn.execute(_sql, _params)
            self._conn.commit()
            entry_id = cur.lastrowid or 0
        self._log.debug("Audit entry #%d logged for %s", entry_id, component)
        return entry_id

    # -- Query --------------------------------------------------------------

    def _build_query_clauses(
        self,
        component: str | None = None,
        severity: str | None = None,
        is_compliant: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        principle: str | None = None,
        limit: int = 100,
    ) -> tuple[str, list[Any]]:
        """Build WHERE clauses and params for query/query_async."""
        clauses: list[str] = []
        params: list[Any] = []

        if component:
            clauses.append("component = ?")
            params.append(component)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if is_compliant is not None:
            clauses.append("is_compliant = ?")
            params.append(1 if is_compliant else 0)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if principle:
            clauses.append("principle = ?")
            params.append(principle)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return sql, params

    def query(
        self,
        component: str | None = None,
        severity: str | None = None,
        is_compliant: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        principle: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters (sync)."""
        if self._db_adapter is not None:
            raise RuntimeError(
                "Use async query_async() when db_adapter is configured"
            )
        sql, params = self._build_query_clauses(
            component, severity, is_compliant, since, until, principle, limit
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def query_async(
        self,
        component: str | None = None,
        severity: str | None = None,
        is_compliant: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        principle: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters. Uses adapter when available."""
        sql, params = self._build_query_clauses(
            component, severity, is_compliant, since, until, principle, limit
        )
        if self._db_adapter is not None:
            rows = await self._db_adapter.query_dicts(self._DB_NAME, sql, params)
            return [self._dict_to_entry(r) for r in rows]
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # -- Statistics ---------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return aggregate stats about the audit trail (sync)."""
        if self._db_adapter is not None:
            raise RuntimeError(
                "Use async get_statistics_async() when db_adapter is configured"
            )
        total = self._scalar("SELECT COUNT(*) FROM audit_log")
        compliant = self._scalar(
            "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 1"
        )
        violations = self._scalar(
            "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 0"
        )

        # Violations by principle
        by_principle: dict[str, int] = {}
        rows = self._conn.execute(
            "SELECT principle, COUNT(*) as cnt FROM audit_log "
            "WHERE is_compliant = 0 AND principle != '' GROUP BY principle"
        ).fetchall()
        for r in rows:
            by_principle[r["principle"]] = r["cnt"]

        # Violations by severity
        by_severity: dict[str, int] = {}
        rows = self._conn.execute(
            "SELECT severity, COUNT(*) as cnt FROM audit_log "
            "WHERE is_compliant = 0 GROUP BY severity"
        ).fetchall()
        for r in rows:
            by_severity[r["severity"]] = r["cnt"]

        return {
            "total_entries": total,
            "compliant": compliant,
            "violations": violations,
            "compliance_rate": (compliant / total * 100) if total else 100.0,
            "violations_by_principle": by_principle,
            "violations_by_severity": by_severity,
        }

    async def get_statistics_async(self) -> dict[str, Any]:
        """Return aggregate stats about the audit trail. Uses adapter when available."""
        if self._db_adapter is not None:
            total = await self._scalar_async("SELECT COUNT(*) FROM audit_log")
            compliant = await self._scalar_async(
                "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 1"
            )
            violations = await self._scalar_async(
                "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 0"
            )

            by_principle: dict[str, int] = {}
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT principle, COUNT(*) as cnt FROM audit_log "
                "WHERE is_compliant = 0 AND principle != '' GROUP BY principle",
            )
            for r in rows:
                by_principle[r["principle"]] = r["cnt"]

            by_severity: dict[str, int] = {}
            rows = await self._db_adapter.query_dicts(
                self._DB_NAME,
                "SELECT severity, COUNT(*) as cnt FROM audit_log "
                "WHERE is_compliant = 0 GROUP BY severity",
            )
            for r in rows:
                by_severity[r["severity"]] = r["cnt"]
        else:
            total = self._scalar("SELECT COUNT(*) FROM audit_log")
            compliant = self._scalar(
                "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 1"
            )
            violations = self._scalar(
                "SELECT COUNT(*) FROM audit_log WHERE is_compliant = 0"
            )
            by_principle = {}
            for r in self._conn.execute(
                "SELECT principle, COUNT(*) as cnt FROM audit_log "
                "WHERE is_compliant = 0 AND principle != '' GROUP BY principle"
            ).fetchall():
                by_principle[r["principle"]] = r["cnt"]
            by_severity = {}
            for r in self._conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM audit_log "
                "WHERE is_compliant = 0 GROUP BY severity"
            ).fetchall():
                by_severity[r["severity"]] = r["cnt"]

        return {
            "total_entries": total,
            "compliant": compliant,
            "violations": violations,
            "compliance_rate": (compliant / total * 100) if total else 100.0,
            "violations_by_principle": by_principle,
            "violations_by_severity": by_severity,
        }

    # -- Export -------------------------------------------------------------

    def export_report(self, limit: int = 500) -> str:
        """Generate a human-readable audit report."""
        stats = self.get_statistics()
        lines = [
            "# EGM Audit Report",
            f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            "",
            f"**Total Entries:** {stats['total_entries']}",
            f"**Compliance Rate:** {stats['compliance_rate']:.1f}%",
            f"**Violations:** {stats['violations']}",
            "",
        ]

        if stats["violations_by_principle"]:
            lines.append("## Violations by Principle")
            for principle, count in stats["violations_by_principle"].items():
                lines.append(f"- {principle}: {count}")
            lines.append("")

        recent = self.query(is_compliant=False, limit=10)
        if recent:
            lines.append("## Recent Violations (last 10)")
            for entry in recent:
                lines.append(
                    f"- [{entry.timestamp}] **{entry.severity}** "
                    f"{entry.component}: {entry.justification}"
                )

        return "\n".join(lines)

    # -- Internals ----------------------------------------------------------

    def _scalar(self, sql: str) -> int:
        row = self._conn.execute(sql).fetchone()
        return row[0] if row else 0

    async def _scalar_async(self, sql: str) -> int:
        """Async scalar query via adapter."""
        if self._db_adapter is not None:
            rows = await self._db_adapter.query(self._DB_NAME, sql)
            return rows[0][0] if rows and rows[0] else 0
        row = self._conn.execute(sql).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            entry_id=row["id"],
            timestamp=row["timestamp"],
            component=row["component"],
            event_type=row["event_type"],
            severity=row["severity"],
            is_compliant=bool(row["is_compliant"]),
            decision=row["decision"],
            justification=row["justification"],
            principle=row["principle"],
            details=json.loads(row["details"]),
        )

    @staticmethod
    def _dict_to_entry(row: dict[str, Any]) -> AuditEntry:
        """Convert a dict (from adapter.query_dicts) to an AuditEntry."""
        return AuditEntry(
            entry_id=row.get("id", 0),
            timestamp=row.get("timestamp", ""),
            component=row.get("component", ""),
            event_type=row.get("event_type", ""),
            severity=row.get("severity", "INFO"),
            is_compliant=bool(row.get("is_compliant", 1)),
            decision=row.get("decision", "ALLOW"),
            justification=row.get("justification", ""),
            principle=row.get("principle", ""),
            details=json.loads(row["details"])
            if isinstance(row.get("details"), str)
            else row.get("details", {}),
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    @property
    def entry_count(self) -> int:
        if self._db_adapter is not None:
            raise RuntimeError(
                "Use async entry_count_async() when db_adapter is configured"
            )
        return self._scalar("SELECT COUNT(*) FROM audit_log")

    async def entry_count_async(self) -> int:
        """Async entry count."""
        return await self._scalar_async("SELECT COUNT(*) FROM audit_log")
