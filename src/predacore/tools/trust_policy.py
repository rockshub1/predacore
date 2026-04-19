"""
PredaCore Trust Policy — confirmation rules, dangerous tool checks, auto-approve lists.

Extracted from ToolExecutor (Phase 6.1).  Centralises all trust-level
evaluation so the executor and other subsystems can query policy without
embedding the logic themselves.

Enhanced with:
  - ApprovalContext: risk assessment for tool calls
  - ApprovalHistory: SQLite-backed approval memory
  - Permission modes: auto | ask | deny
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import TRUST_POLICIES

logger = logging.getLogger(__name__)


# ── Risk levels and tool classification ──────────────────────────────

_RISK_MAP: dict[str, str] = {
    # Low risk — read-only, no side effects
    "read_file": "low",
    "web_search": "low",
    "memory_search": "low",
    "memory_store": "low",
    "knowledge_graph_query": "low",
    "screen_vision": "low",
    "deep_search": "low",
    "semantic_search": "low",
    "image_gen": "low",
    "pdf_reader": "low",
    "voice_note": "low",
    "diagram": "low",
    # Medium risk — writes but recoverable
    "write_file": "medium",
    "desktop_control": "medium",
    "android_control": "medium",
    "multi_agent": "medium",
    "git_commit": "medium",
    "git_branch": "medium",
    # High risk — system-level side effects
    "run_command": "high",
    "python_exec": "high",
    "install_skill": "high",
    "cron_task": "high",
    "git_push": "high",
    # Critical — detected by argument inspection
}

_CRITICAL_PATTERNS = [
    re.compile(r'\brm\s+-rf\b'),
    re.compile(r'\bsudo\b'),
    re.compile(r'\bchmod\s+777\b'),
    re.compile(r'\bmkfs\b'),
    re.compile(r'\bdd\s+if=\b'),
    re.compile(r'>\s*/dev/'),
]


@dataclass
class ApprovalContext:
    """Rich context for tool approval decisions."""

    tool_name: str
    arguments: dict
    risk_level: str  # low | medium | high | critical
    impact_summary: str
    cost_estimate: str
    reversible: bool

    def to_message(self) -> str:
        """Format as a human-readable approval request."""
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        emoji = risk_emoji.get(self.risk_level, "⚪")
        lines = [
            f"{emoji} **Tool approval needed**: `{self.tool_name}`",
            f"**Risk**: {self.risk_level}",
            f"**Action**: {self.impact_summary}",
        ]
        if self.arguments:
            # Show truncated args
            args_str = json.dumps(self.arguments, default=str)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            lines.append(f"**Args**: `{args_str}`")
        if not self.reversible:
            lines.append("⚠️ This action may not be reversible.")
        lines.append("\nReply **yes** to approve or **no** to deny.")
        return "\n".join(lines)


# ── Approval History (SQLite) ────────────────────────────────────────


class ApprovalHistory:
    """Persists tool approval decisions to avoid repeated prompts."""

    def __init__(self, db_path: str, *, db_adapter=None):
        self._db_path = db_path
        self._db_adapter = db_adapter  # Optional DBAdapter
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        if self._db_adapter:
            # Schema init via adapter (sync direct path — callers are sync)
            self._db_adapter._direct_executescript("approvals", """
                CREATE TABLE IF NOT EXISTS approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    UNIQUE(tool_name, args_hash)
                )
            """)
            return
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS approvals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tool_name TEXT NOT NULL,
                        args_hash TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        user_id TEXT DEFAULT '',
                        timestamp REAL NOT NULL,
                        UNIQUE(tool_name, args_hash)
                    )
                """)
        except sqlite3.OperationalError:
            # DB locked by daemon — approvals will use in-memory fallback
            pass

    @staticmethod
    def _hash_args(arguments: dict) -> str:
        raw = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def check(self, tool_name: str, arguments: dict) -> bool | None:
        """Check if there's a previous approval. Returns True/False/None."""
        args_hash = self._hash_args(arguments)
        try:
            if self._db_adapter:
                rows = self._db_adapter._direct_query_dicts(
                    "approvals",
                    "SELECT decision FROM approvals WHERE tool_name = ? AND args_hash = ?",
                    [tool_name, args_hash],
                )
                if rows:
                    return rows[0]["decision"] == "approved"
                return None
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                row = conn.execute(
                    "SELECT decision FROM approvals WHERE tool_name = ? AND args_hash = ?",
                    (tool_name, args_hash),
                ).fetchone()
            if row:
                return row[0] == "approved"
            return None
        except (OSError, sqlite3.Error):
            return None

    def record(
        self, tool_name: str, arguments: dict, approved: bool, user_id: str = ""
    ) -> None:
        """Record an approval decision."""
        args_hash = self._hash_args(arguments)
        decision = "approved" if approved else "denied"
        try:
            if self._db_adapter:
                self._db_adapter._direct_execute(
                    "approvals",
                    """INSERT OR REPLACE INTO approvals
                       (tool_name, args_hash, decision, user_id, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    [tool_name, args_hash, decision, user_id, time.time()],
                )
                return
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO approvals
                       (tool_name, args_hash, decision, user_id, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (tool_name, args_hash, decision, user_id, time.time()),
                )
        except (OSError, sqlite3.Error) as e:
            logger.warning("Failed to record approval: %s", e)


# ── Trust Policy Evaluator ───────────────────────────────────────────


class TrustPolicyEvaluator:
    """Evaluates whether a tool call should be auto-approved, confirmed, or blocked."""

    def __init__(
        self,
        trust_level: str = "normal",
        permission_mode: str = "auto",
        remember_approvals: bool = True,
        home_dir: str = "",
        db_adapter=None,
    ):
        self._trust_level = trust_level
        self._permission_mode = permission_mode
        self._policy: dict[str, Any] = TRUST_POLICIES.get(
            trust_level, TRUST_POLICIES["normal"]
        )
        self._approval_history: ApprovalHistory | None = None
        if remember_approvals and home_dir:
            db_path = str(Path(home_dir) / "memory" / "approvals.db")
            self._approval_history = ApprovalHistory(db_path, db_adapter=db_adapter)

    @property
    def trust_level(self) -> str:
        """Return the current trust level string."""
        return self._trust_level

    @property
    def policy(self) -> dict[str, Any]:
        """Return the active TrustPolicy configuration."""
        return self._policy

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation under current trust policy."""
        # Permission mode overrides — checked BEFORE trust policy
        if self._permission_mode == "deny":
            # Deny mode: require confirmation for everything except read-only tools
            risk = _RISK_MAP.get(tool_name, "medium")
            if risk in ("high", "critical", "medium"):
                return True
        if self._permission_mode == "ask":
            return True  # Always ask

        # Default: use trust policy (only reached if permission_mode == "auto")
        if "*" in self._policy.get("auto_approve_tools", []):
            return False
        if "*" in self._policy.get("require_confirmation", []):
            return True
        return tool_name in self._policy.get("require_confirmation", [])

    def is_blocked(
        self,
        tool_name: str,
        blocked_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> str | None:
        """Return a block reason string if the tool is blocked, else None."""
        if blocked_tools and tool_name in blocked_tools:
            return f"[Tool '{tool_name}' is blocked by security policy]"
        if allowed_tools and tool_name not in allowed_tools:
            return f"[Tool '{tool_name}' is not in the allowed tools list]"
        return None

    def assess_risk(self, tool_name: str, arguments: dict) -> ApprovalContext:
        """Build an ApprovalContext with risk assessment for a tool call."""
        # Base risk from tool name
        risk = _RISK_MAP.get(tool_name, "medium")

        # Escalate to critical if dangerous patterns detected
        args_str = json.dumps(arguments, default=str).lower()
        for pattern in _CRITICAL_PATTERNS:
            if pattern.search(args_str):
                risk = "critical"
                break

        # Build impact summary
        impact = self._describe_impact(tool_name, arguments)

        # Reversibility
        reversible = risk in ("low", "medium")

        # Cost estimate from tool registry
        from .registry import build_full_registry

        _reg = build_full_registry()
        tool_def = _reg.get(tool_name)
        cost = tool_def.cost_estimate if tool_def is not None else "unknown"

        return ApprovalContext(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk,
            impact_summary=impact,
            cost_estimate=str(cost),
            reversible=reversible,
        )

    def check_previous_approval(
        self, tool_name: str, arguments: dict
    ) -> bool | None:
        """Check approval history. Returns True/False/None."""
        if self._approval_history is None:
            return None
        return self._approval_history.check(tool_name, arguments)

    def record_approval(
        self, tool_name: str, arguments: dict, approved: bool, user_id: str = ""
    ) -> None:
        """Record an approval decision for future reference."""
        if self._approval_history is not None:
            self._approval_history.record(tool_name, arguments, approved, user_id)

    @staticmethod
    def _describe_impact(tool_name: str, arguments: dict) -> str:
        """Generate a human-readable description of what the tool will do."""
        if tool_name == "run_command":
            cmd = arguments.get("command", "?")
            return f"Execute shell command: `{cmd[:100]}`"
        if tool_name == "python_exec":
            code = arguments.get("code", "?")
            return f"Run Python code ({len(code)} chars)"
        if tool_name == "write_file":
            path = arguments.get("path", arguments.get("file_path", "?"))
            return f"Write to file: `{path}`"
        if tool_name == "read_file":
            path = arguments.get("path", arguments.get("file_path", "?"))
            return f"Read file: `{path}`"
        if tool_name == "desktop_control":
            action = arguments.get("action", "?")
            return f"macOS desktop action: {action}"
        if tool_name == "android_control":
            action = arguments.get("action", "?")
            return f"Android device action: {action}"
        if tool_name == "web_search":
            query = arguments.get("query", "?")
            return f"Search the web: `{query[:80]}`"
        if tool_name == "multi_agent":
            return f"Spawn {arguments.get('agent_count', '?')} sub-agents"
        return f"Execute tool `{tool_name}`"
