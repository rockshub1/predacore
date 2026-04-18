"""
PredaCore IdentityEngine — manages the self-evolving identity workspace.

The agent workspace lives at ~/.predacore/agents/{name}/ with eight
markdown files (IDENTITY, SOUL, USER, JOURNAL, TOOLS, MEMORY, HEARTBEAT,
REFLECTION). On first instantiation the workspace is seeded from bundled
defaults — IDENTITY.md ships name-less and tells the agent to ask three
short questions on first contact, then save the answers via
``identity_update``. No separate bootstrap step: the default files are
themselves the first-run instructions.

The engine handles:
- Auto-seeding the workspace from bundled defaults on first run
- Identity file loading with mtime-based caching
- Writing agent-generated identity files (atomic temp+rename)
- Append-only journal / decisions / evolution logs, all bounded
- Prompt assembly: SOUL_SEED → EVENT_HORIZON → IDENTITY → SOUL → USER
  → profile → MEMORY → TOOLS → HEARTBEAT → REFLECTION → BELIEFS → journal
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# mtime-based file cache: {path_str: (mtime, content)}
_identity_file_cache: dict[str, tuple[float, str]] = {}

# Maximum allowed size for identity file writes (bytes)
MAX_IDENTITY_FILE_SIZE = 50_000

# Per-(home_dir, agent_name) engine instances
_engine_instances: dict[tuple[str, str], IdentityEngine] = {}

# Files the agent is allowed to write
_WRITABLE_FILES = frozenset({
    "IDENTITY.md", "SOUL.md", "USER.md", "JOURNAL.md",
    "TOOLS.md", "MEMORY.md", "HEARTBEAT.md", "REFLECTION.md",
    "BELIEFS.md", "DECISIONS.md",
})

# Files whose changes are evolution-worthy — any update to these is
# automatically diff-logged to EVOLUTION.md with a timestamp and reason.
# This is the legible record of how the agent changed over time.
_EVOLVING_FILES = frozenset({
    "SOUL.md", "IDENTITY.md", "USER.md", "BELIEFS.md",
})

# Maximum journal entries to include in prompt context
_JOURNAL_TAIL_ENTRIES = 20

# Maximum bytes of journal tail to include in the assembled prompt
# (per-turn prompt-size cap so a long journal can't bloat system prompt)
_JOURNAL_TAIL_MAX_BYTES = 10_000

# Maximum diff lines to include in an EVOLUTION.md entry (keeps the log readable)
_MAX_DIFF_LINES = 200

# EVOLUTION.md keeps the most recent N entries — prevents unbounded growth.
_MAX_EVOLUTION_ENTRIES = 200

# Maximum DECISIONS.md entries kept on disk.
_MAX_DECISION_ENTRIES = 500


def _read_cached(path: Path) -> str | None:
    """Read file content with mtime-based caching. Returns None if missing."""
    if not path.exists():
        return None
    try:
        path_str = str(path)
        mtime = path.stat().st_mtime
        cached = _identity_file_cache.get(path_str)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        content = path.read_text(encoding="utf-8").strip()
        _identity_file_cache[path_str] = (mtime, content)
        return content
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read identity file %s: %s", path, e)
        return None


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomic text write: temp file in same dir, then os.replace.

    Prevents partial writes on crash. Shared by write_identity_file,
    append_journal, append_decision, and BeliefStore rendering paths.
    """
    import tempfile as _tempfile

    fd, tmp_path = _tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _log_evolution_to_file(
    workspace: Path,
    filename: str,
    old_content: str,
    new_content: str,
    *,
    reason: str = "",
) -> None:
    """Append a diff entry to EVOLUTION.md for a changed evolving file.

    Module-level helper so both IdentityEngine.write_identity_file and
    BeliefStore._save can log evolution without circular imports.

    Appends a timestamped unified diff to EVOLUTION.md. Caps the diff
    so the log doesn't become its own bloat problem.
    """
    import difflib

    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{filename} (before)",
        tofile=f"{filename} (after)",
        n=3,
    ))

    if len(diff_lines) > _MAX_DIFF_LINES:
        head = diff_lines[: _MAX_DIFF_LINES // 2]
        tail = diff_lines[-(_MAX_DIFF_LINES // 2):]
        truncation_note = (
            f"\n... ({len(diff_lines) - _MAX_DIFF_LINES} lines elided for readability) ...\n"
        )
        diff_text = "".join(head) + truncation_note + "".join(tail)
    else:
        diff_text = "".join(diff_lines)

    if not diff_text.strip():
        diff_text = "(no textual diff detected — likely whitespace or ordering)"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reason_line = f"**Reason:** {reason.strip()}\n\n" if reason.strip() else ""
    entry = (
        f"\n\n## {timestamp} — {filename}\n\n"
        f"{reason_line}"
        f"```diff\n{diff_text.rstrip()}\n```\n"
    )

    evo_path = workspace / "EVOLUTION.md"
    header = (
        "# Evolution Log\n\n"
        "_Legible record of how this agent changed over time. "
        "Every meaningful update to SOUL, IDENTITY, USER, and BELIEFS "
        "is diff-logged here with timestamp and reason._\n"
    )
    try:
        existing = evo_path.read_text(encoding="utf-8") if evo_path.exists() else header
        new_content = existing + entry

        # Auto-prune: keep only the most recent _MAX_EVOLUTION_ENTRIES entries.
        # Entries start with "## " at column 0 (see format in `entry` above).
        entry_markers = new_content.split("\n## ")
        if len(entry_markers) - 1 > _MAX_EVOLUTION_ENTRIES:
            # [0] is the pre-first-entry header; [1:] are individual entries.
            kept = entry_markers[-_MAX_EVOLUTION_ENTRIES:]
            new_content = header.rstrip() + "\n\n## " + "\n## ".join(kept)

        _atomic_write_text(evo_path, new_content)
        _identity_file_cache.pop(str(evo_path), None)
        logger.info("EVOLUTION.md updated for %s", filename)
    except OSError as exc:
        logger.warning("Evolution log write failed for %s: %s", filename, exc)


@dataclass
class UserProfile:
    """Structured user profile -- queryable companion to USER.md."""

    user_id: str = "default"
    preferences: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    knowledge_areas: dict[str, str] = field(default_factory=dict)  # area -> level
    cognitive_style: str = ""  # e.g., "visual", "analytical", "creative"
    notes: str = ""
    last_interaction_at: str = ""
    updated_at: str = ""
    reflection_count: int = 0  # persisted across process restarts

    def to_dict(self) -> dict[str, Any]:
        # Drop only None and empty strings; keep int 0, False, empty collections
        # so round-trip is stable (reflection_count=0 was silently lost before).
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != ""
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class IdentityEngine:
    """Manages the self-evolving identity workspace at ~/.predacore/agents/{name}/."""

    # Default workspace name — PredaCore does not ship with a pre-named
    # agent. The workspace dir is a neutral "default"; the agent discovers
    # its own name on first conversation and writes it into IDENTITY.md.
    DEFAULT_AGENT_NAME = "default"

    def __init__(self, home_dir: str, agent_name: str = "default"):
        self.agent_name = agent_name
        self.workspace = Path(home_dir) / "agents" / agent_name
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Built-in identity dir (ships with source: SOUL_SEED + EVENT_HORIZON)
        self._builtin_dir = Path(__file__).parent

        # Seed the workspace from bundled defaults if empty. Every fresh
        # workspace gets the full identity surface — voice, heartbeat
        # policy, reflection rules, memory skeleton, and an IDENTITY.md
        # whose content instructs the agent to ask its name on first turn.
        self._seed_from_defaults()

        logger.info(
            "IdentityEngine initialized: agent=%s, workspace=%s",
            agent_name,
            self.workspace,
        )

    def _seed_from_defaults(self) -> None:
        """Copy any missing default identity files into the workspace.

        Idempotent: existing files are never overwritten. A fresh workspace
        ends up with IDENTITY / SOUL / USER / MEMORY / TOOLS / HEARTBEAT /
        REFLECTION / JOURNAL. The seeded IDENTITY.md is name-less and
        instructs the agent to ask its name on the first real turn.
        """
        defaults_dir = self._builtin_dir / "defaults"
        if not defaults_dir.exists():
            logger.warning(
                "Default identity dir not found at %s — skipping seeding",
                defaults_dir,
            )
            return

        copied = 0
        for src in defaults_dir.glob("*.md"):
            dst = self.workspace / src.name
            if dst.exists():
                continue
            try:
                shutil.copy2(str(src), str(dst))
                copied += 1
            except OSError as exc:
                logger.warning("Failed to copy default %s: %s", src.name, exc)

        if copied:
            logger.info(
                "Seeded agent workspace from defaults: %d files copied to %s",
                copied, self.workspace,
            )

    def _load_workspace_or_builtin(self, filename: str) -> str:
        """Load a file from the agent workspace first, then fall back to built-ins.

        Scans for prompt injection before loading — blocks malicious identity files.
        """
        workspace_content = _read_cached(self.workspace / filename)
        if workspace_content is not None:
            if self._scan_for_injection(workspace_content, filename):
                return workspace_content
            logger.warning("Injection blocked in %s/%s — falling back to built-in", self.workspace, filename)
        return _read_cached(self._builtin_dir / filename) or ""

    def _scan_for_injection(self, content: str, filename: str) -> bool:
        """Scan identity file content for prompt injection. Returns True if safe.

        Fail-open: if the security module is unavailable or its signature
        changed, we log and allow the content. Fail-closed would make the
        whole identity system unloadable on any auth-module regression.
        """
        try:
            from predacore.auth.security import detect_injection
        except ImportError as e:
            logger.debug("Injection scan module unavailable: %s", e)
            return True

        try:
            result = detect_injection(content)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Injection scan call failed for %s: %s", filename, e)
            return True

        if result.detected:
            logger.critical(
                "INJECTION DETECTED in identity file %s/%s — "
                "patterns: %s, confidence: %.2f — BLOCKING LOAD",
                self.workspace, filename,
                result.patterns_matched, result.confidence,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def load_seed(self) -> str:
        """Load immutable SOUL_SEED.md from built-in identity dir ONLY.

        SOUL_SEED is the non-overridable safety floor — it must not be
        loadable from the workspace where a local attacker (or a runaway
        agent) could tamper with it. Bundled default is the only source.
        """
        path = self._builtin_dir / "SOUL_SEED.md"
        content = _read_cached(path) or ""
        if not content:
            logger.error("SOUL_SEED.md missing from built-in dir %s", self._builtin_dir)
            return ""
        # Defense-in-depth: scan for injection even in the bundled version.
        # If a tampered wheel ships a compromised SOUL_SEED, we fail loud
        # rather than silently apply the attacker's invariants.
        if not self._scan_for_injection(content, "SOUL_SEED.md"):
            logger.critical(
                "SOUL_SEED.md in built-in dir failed injection scan — "
                "package integrity compromised, refusing to load"
            )
            return ""
        return content

    def load_event_horizon(self) -> str:
        """Load EVENT_HORIZON.md — the recursive growth protocol.

        Loaded every session (bootstrap or not) right after SOUL_SEED.md.
        This is the protocol the agent runs on itself to keep evolving
        through relationship instead of drifting into template-land.

        Like SOUL_SEED, loaded from bundled package only — never from the
        workspace — to prevent tampering with the safety protocol.
        """
        path = self._builtin_dir / "EVENT_HORIZON.md"
        content = _read_cached(path) or ""
        if not content:
            logger.debug("EVENT_HORIZON.md not found in %s", self._builtin_dir)
            return ""
        if not self._scan_for_injection(content, "EVENT_HORIZON.md"):
            logger.critical(
                "EVENT_HORIZON.md in built-in dir failed injection scan — "
                "package integrity compromised, refusing to load"
            )
            return ""
        return content

    def load_identity(self) -> str:
        """Load agent-written IDENTITY.md from workspace."""
        return _read_cached(self.workspace / "IDENTITY.md") or ""

    def load_soul(self) -> str:
        """Load agent-grown SOUL.md from workspace (evolved personality)."""
        return _read_cached(self.workspace / "SOUL.md") or ""

    def load_user(self) -> str:
        """Load agent-learned USER.md from workspace."""
        return _read_cached(self.workspace / "USER.md") or ""

    def load_tools(self) -> str:
        """Load TOOLS.md (local environment notes) from workspace."""
        return _read_cached(self.workspace / "TOOLS.md") or ""

    def load_memory(self) -> str:
        """Load MEMORY.md (curated long-term memory) from workspace."""
        return _read_cached(self.workspace / "MEMORY.md") or ""

    def load_heartbeat_config(self) -> str:
        """Load HEARTBEAT.md (periodic check config) from workspace."""
        return _read_cached(self.workspace / "HEARTBEAT.md") or ""

    def load_reflection_rules(self) -> str:
        """Load REFLECTION.md (self-reflection rules) from workspace."""
        return _read_cached(self.workspace / "REFLECTION.md") or ""

    def load_beliefs(self) -> str:
        """Load BELIEFS.md — the current crystallization state of the agent's beliefs."""
        return _read_cached(self.workspace / "BELIEFS.md") or ""

    def load_decisions(self) -> str:
        """Load DECISIONS.md — the per-turn reasoning trace."""
        return _read_cached(self.workspace / "DECISIONS.md") or ""

    def load_evolution(self) -> str:
        """Load EVOLUTION.md — the diff-logged record of identity changes over time."""
        return _read_cached(self.workspace / "EVOLUTION.md") or ""

    @property
    def belief_store(self):
        """
        Lazy-init BeliefStore. Grants programmatic access to the crystallization
        state machine — used by the consolidator's opinion formation loop and by
        reflection for belief updates.
        """
        if not hasattr(self, "_belief_store"):
            from .beliefs import BeliefStore
            self._belief_store = BeliefStore(self.workspace)
        return self._belief_store

    def load_journal(self, max_entries: int = _JOURNAL_TAIL_ENTRIES) -> str:
        """Load recent JOURNAL.md entries (tail) for prompt context.

        Capped by both entry count (_JOURNAL_TAIL_ENTRIES) and byte size
        (_JOURNAL_TAIL_MAX_BYTES) so a single huge entry can't blow up
        the per-turn prompt.
        """
        content = _read_cached(self.workspace / "JOURNAL.md")
        if not content:
            return ""

        # Split journal by entry headers (## YYYY-MM-DD ...)
        entries = re.split(r"(?=^## \d{4}-\d{2}-\d{2})", content, flags=re.MULTILINE)
        entries = [e.strip() for e in entries if e.strip()]

        if len(entries) <= max_entries:
            tail_text = content
        else:
            tail = entries[-max_entries:]
            tail_text = "\n\n".join(tail)

        # Byte cap — keep the most recent bytes if the tail is too large
        if len(tail_text) > _JOURNAL_TAIL_MAX_BYTES:
            tail_text = (
                "_(earlier entries truncated for prompt-size cap)_\n\n"
                + tail_text[-_JOURNAL_TAIL_MAX_BYTES:]
            )
        return tail_text

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def write_identity_file(
        self,
        filename: str,
        content: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Write or update an identity file in the workspace.

        Args:
            filename: Must be in _WRITABLE_FILES.
            content: The full file content to write.
            reason: Optional short explanation of *why* this write is happening.
                Gets logged in EVOLUTION.md alongside the diff (for _EVOLVING_FILES).

        Returns:
            Dict with status, path, size_bytes, and (if applicable) evolution_logged.
        """
        if filename not in _WRITABLE_FILES:
            return {
                "status": "error",
                "error": f"Cannot write to '{filename}'. "
                f"Writable files: {sorted(_WRITABLE_FILES)}",
            }

        # Content validation: enforce size limit and strip null bytes
        if len(content) > MAX_IDENTITY_FILE_SIZE:
            return {
                "status": "error",
                "error": f"Content exceeds {MAX_IDENTITY_FILE_SIZE} byte limit",
            }
        content = content.replace('\x00', '')

        # Injection scan: block writes that contain prompt injection patterns
        if not self._scan_for_injection(content, filename):
            return {
                "status": "error",
                "error": f"Content blocked: prompt injection detected in {filename}",
            }

        path = self.workspace / filename

        # Capture the prior content before we overwrite — needed for diff logging
        old_content = ""
        if path.exists():
            try:
                old_content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                old_content = ""

        try:
            # Atomic write: temp + rename so a crash mid-write cannot
            # corrupt a SOUL/IDENTITY/USER file.
            _atomic_write_text(path, content)
            # Invalidate cache so next read picks up the new content
            cache_key = str(path)
            _identity_file_cache.pop(cache_key, None)

            size = path.stat().st_size
            logger.info("Identity file written: %s (%d bytes)", path, size)

            # Log the evolution if this is an evolving file and the content actually changed
            evolution_logged = False
            if filename in _EVOLVING_FILES and old_content != content:
                try:
                    _log_evolution_to_file(
                        self.workspace, filename, old_content, content, reason=reason,
                    )
                    evolution_logged = True
                except OSError as exc:
                    logger.warning("Evolution log failed for %s: %s", filename, exc)

            return {
                "status": "ok",
                "path": str(path),
                "size_bytes": size,
                "evolution_logged": evolution_logged,
            }
        except OSError as e:
            logger.error("Failed to write identity file %s: %s", path, e)
            return {"status": "error", "error": str(e)}

    def append_decision(
        self,
        *,
        session_id: str = "",
        user_message: str = "",
        response_summary: str = "",
        tools_used: list[str] | None = None,
        files_touched: list[str] | None = None,
        duration_seconds: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        outcome: str = "",
        iterations: int = 0,
    ) -> dict[str, Any]:
        """
        Append a per-turn decision entry to DECISIONS.md — the reasoning trace.

        Observable-facts version: captures what actually happened in a turn
        (user input, response summary, tools, files, duration, outcome) so
        that future sessions and humans can inspect *why* the agent did
        what it did. This is the infrastructure layer that makes decisions
        legible without requiring an expensive reasoning trace at turn time.

        Auto-pruned to the last 500 entries to keep the file bounded.
        """
        path = self.workspace / "DECISIONS.md"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        user_msg_snippet = (user_message or "").strip().replace("\n", " ")
        if len(user_msg_snippet) > 300:
            user_msg_snippet = user_msg_snippet[:300] + "..."

        resp_snippet = (response_summary or "").strip().replace("\n", " ")
        if len(resp_snippet) > 300:
            resp_snippet = resp_snippet[:300] + "..."

        tools_str = ", ".join(tools_used) if tools_used else "(none)"
        files_str = ", ".join(files_touched[:5]) if files_touched else "(none)"
        if files_touched and len(files_touched) > 5:
            files_str += f", +{len(files_touched) - 5} more"

        session_label = f"session {session_id[:12]}" if session_id else "no session"

        lines = [
            f"\n\n## {timestamp} — {session_label}",
            "",
            f"**User:** {user_msg_snippet or '(empty)'}",
            "",
            f"**Response:** {resp_snippet or '(empty)'}",
            "",
            f"**Tools:** {tools_str}",
            f"**Files:** {files_str}",
            f"**Duration:** {duration_seconds:.1f}s · **Iterations:** {iterations} · **Tokens:** {tokens_in} in / {tokens_out} out",
        ]
        if outcome:
            lines.append(f"**Outcome:** {outcome}")
        entry = "\n".join(lines) + "\n"

        try:
            if path.exists():
                existing = path.read_text(encoding="utf-8")
            else:
                existing = (
                    "# Decisions Log\n\n"
                    "_Per-turn reasoning trace — what the agent was asked, "
                    "what it did, which tools it used, how long it took, and "
                    "what the outcome was. Auto-pruned to the last 500 entries._\n"
                )

            new_content = existing + entry

            # Auto-prune: keep the last 500 decision entries
            entry_count = new_content.count("\n## ")
            if entry_count > 500:
                entries = new_content.split("\n## ")
                header = entries[0]
                kept = entries[-500:]
                new_content = header + "\n## " + "\n## ".join(kept)

            _atomic_write_text(path, new_content)
            _identity_file_cache.pop(str(path), None)
            logger.debug("DECISIONS.md updated (turn tools=%d files=%d)",
                         len(tools_used or []), len(files_touched or []))
            return {"status": "ok", "path": str(path)}
        except OSError as e:
            logger.warning("Failed to append decision: %s", e)
            return {"status": "error", "error": str(e)}

    def append_journal(self, entry: str) -> dict[str, Any]:
        """Append a timestamped entry to JOURNAL.md.

        Format:
            ## YYYY-MM-DD HH:MM UTC
            {entry text}

        """
        path = self.workspace / "JOURNAL.md"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        formatted = f"\n\n## {timestamp}\n\n{entry.strip()}\n"

        try:
            if path.exists():
                existing = path.read_text(encoding="utf-8")
            else:
                existing = "# Identity Journal\n\n_Growth log — insights, milestones, reflections._"

            new_content = existing + formatted
            _atomic_write_text(path, new_content)

            # Invalidate cache
            _identity_file_cache.pop(str(path), None)

            entry_count = len(
                re.findall(r"^## \d{4}-\d{2}-\d{2}", new_content, re.MULTILINE)
            )
            logger.info("Journal entry appended: %d total entries", entry_count)
            return {
                "status": "ok",
                "path": str(path),
                "entry_count": entry_count,
                "timestamp": timestamp,
            }
        except OSError as e:
            logger.error("Failed to append journal: %s", e)
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Structured user profile
    # ------------------------------------------------------------------

    @property
    def _profile_path(self) -> Path:
        """Path to the structured profile.json file."""
        return self.workspace / "profile.json"

    def load_profile(self) -> UserProfile:
        """Load structured UserProfile from profile.json. Returns default if missing."""
        if not self._profile_path.exists():
            return UserProfile()
        try:
            raw = json.loads(self._profile_path.read_text(encoding="utf-8"))
            return UserProfile.from_dict(raw if isinstance(raw, dict) else {})
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load profile.json: %s", e)
            return UserProfile()

    def save_profile(self, profile: UserProfile) -> None:
        """Write UserProfile to profile.json with atomic write (temp + rename)."""
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        data = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.workspace), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(tmp_path, str(self._profile_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.info("UserProfile saved to %s", self._profile_path)
        except OSError as e:
            logger.error("Failed to save profile.json: %s", e)

    def record_interaction(self) -> None:
        """Update last_interaction_at timestamp in the user profile."""
        profile = self.load_profile()
        profile.last_interaction_at = datetime.now(timezone.utc).isoformat()
        self.save_profile(profile)

    # ------------------------------------------------------------------
    # Stats & introspection
    # ------------------------------------------------------------------

    def get_growth_stats(self) -> dict[str, Any]:
        """Return identity growth metrics."""
        stats: dict[str, Any] = {
            "workspace": str(self.workspace),
            "files": {},
        }

        for filename in sorted(_WRITABLE_FILES):
            path = self.workspace / filename
            if path.exists():
                stat = path.stat()
                stats["files"][filename] = {
                    "size_bytes": stat.st_size,
                    "last_modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }

                # Count journal entries
                if filename == "JOURNAL.md":
                    content = path.read_text(encoding="utf-8")
                    stats["files"][filename]["entry_count"] = len(
                        re.findall(r"^## \d{4}-\d{2}-\d{2}", content, re.MULTILINE)
                    )
            else:
                stats["files"][filename] = None

        return stats

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_identity_prompt(self) -> str:
        """Assemble the full identity prompt for system-prompt injection.

        One path, always the same. The defaults shipped with the workspace
        already carry first-run instructions (IDENTITY.md tells the agent
        to ask its name), so no bootstrap-mode branching is needed.

        Layer order (horizontal-rule separated):
          SOUL_SEED → EVENT_HORIZON → IDENTITY → SOUL → USER
          → structured profile → MEMORY → TOOLS → HEARTBEAT
          → REFLECTION → BELIEFS → recent journal tail
        """
        sections: list[str] = []

        def _add(content: str, *, wrap: str = "") -> None:
            if content:
                if wrap:
                    sections.append(f"\n---\n\n# {wrap}\n\n{content}")
                elif sections:
                    sections.append(f"\n---\n\n{content}")
                else:
                    sections.append(content)

        _add(self.load_seed())
        _add(self.load_event_horizon())
        _add(self.load_identity())
        _add(self.load_soul())
        _add(self.load_user())

        profile = self.load_profile()
        profile_parts: list[str] = []
        if profile.goals:
            profile_parts.append(f"User Goals: {', '.join(profile.goals)}")
        if profile.knowledge_areas:
            areas = ", ".join(f"{k} ({v})" for k, v in profile.knowledge_areas.items())
            profile_parts.append(f"Knowledge Areas: {areas}")
        if profile.cognitive_style:
            profile_parts.append(f"Cognitive Style: {profile.cognitive_style}")
        if profile.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in profile.preferences.items())
            profile_parts.append(f"Preferences: {prefs}")
        if profile_parts:
            _add("\n".join(profile_parts))

        _add(self.load_memory())
        _add(self.load_tools())
        _add(self.load_heartbeat_config(), wrap="Background Discipline (HEARTBEAT.md)")
        _add(self.load_reflection_rules(), wrap="Self-Correction Rules (REFLECTION.md)")
        _add(self.load_beliefs())
        _add(self.load_journal(), wrap="Recent Journal")

        return "\n".join(sections)


def get_identity_engine(config: Any) -> IdentityEngine:
    """Get or create the IdentityEngine for (home_dir, agent_name).

    Keying on the pair makes switching agents mid-process safe — each
    agent gets its own engine rather than silently sharing the first one.
    """
    home_dir = getattr(config, "home_dir", str(Path.home() / ".predacore"))
    agent_name = getattr(config, "agent", "default")
    key = (home_dir, agent_name)
    engine = _engine_instances.get(key)
    if engine is None:
        engine = IdentityEngine(home_dir, agent_name=agent_name)
        _engine_instances[key] = engine
    return engine


def reset_identity_engine() -> None:
    """Reset all cached engines (for testing)."""
    _engine_instances.clear()
