"""
Prometheus IdentityEngine — manages the self-evolving identity workspace.

Each agent lives in ~/.prometheus/agents/{name}/ with 10 markdown files:
  SOUL_SEED.md, SOUL.md, IDENTITY.md, USER.md, JOURNAL.md,
  BOOTSTRAP.md, TOOLS.md, MEMORY.md, HEARTBEAT.md, REFLECTION.md

The engine handles:
- Bootstrap detection (has the agent been through first contact?)
- Identity file loading with mtime-based caching
- Writing agent-generated identity files to the agent folder
- Journal management (append-only growth log)
- Prompt assembly (seed + identity layers OR seed + bootstrap)
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

# Singleton engine instance
_engine_instance: IdentityEngine | None = None

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

# Files that must exist before bootstrap can be considered complete.
_BOOTSTRAP_REQUIRED_FILES = (
    "IDENTITY.md",
    "SOUL.md",
    "USER.md",
    "TOOLS.md",
    "MEMORY.md",
    "HEARTBEAT.md",
    "REFLECTION.md",
)

# Maximum journal entries to include in prompt context
_JOURNAL_TAIL_ENTRIES = 20

# Maximum bytes of journal tail to include in the assembled prompt
# (per-turn prompt-size cap so a long journal can't bloat system prompt)
_JOURNAL_TAIL_MAX_BYTES = 10_000

# Maximum diff lines to include in an EVOLUTION.md entry (keeps the log readable)
_MAX_DIFF_LINES = 200


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
    try:
        if evo_path.exists():
            existing = evo_path.read_text(encoding="utf-8")
        else:
            existing = (
                "# Evolution Log\n\n"
                "_Legible record of how this agent changed over time. "
                "Every meaningful update to SOUL, IDENTITY, USER, and BELIEFS "
                "is diff-logged here with timestamp and reason._\n"
                "\n_Friend who evolves — this file is the evolution._\n"
            )

        _atomic_write_text(evo_path, existing + entry)
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
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class IdentityEngine:
    """Manages the self-evolving identity workspace at ~/.prometheus/agents/{name}/."""

    # Default agent name — this agent skips first-contact bootstrap by
    # auto-populating its workspace from bundled defaults in `defaults/`.
    DEFAULT_AGENT_NAME = "jarvis"

    def __init__(self, home_dir: str, agent_name: str = "jarvis"):
        self.agent_name = agent_name
        self.workspace = Path(home_dir) / "agents" / agent_name
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Built-in identity dir (ships with source: SOUL_SEED + BOOTSTRAP + EVENT_HORIZON)
        self._builtin_dir = Path(__file__).parent

        # Auto-bootstrap the default agent from bundled defaults if the
        # workspace is empty. This is what makes JARVIS "just work" without
        # a first-contact conversation — it already knows its name, vibe,
        # and core personality. Only USER.md is discovered through talking.
        # Custom agents (non-default names) still go through BOOTSTRAP.md.
        if agent_name == self.DEFAULT_AGENT_NAME and not self.is_bootstrapped:
            self._auto_bootstrap_from_defaults()

        logger.info(
            "IdentityEngine initialized: agent=%s, workspace=%s, bootstrapped=%s",
            agent_name,
            self.workspace,
            self.is_bootstrapped,
        )

    def _auto_bootstrap_from_defaults(self) -> None:
        """Copy bundled default identity files into a fresh JARVIS workspace.

        Only runs for the default agent and only when the workspace is empty.
        After this runs, `is_bootstrapped` returns True and BOOTSTRAP.md is
        never injected into the system prompt.

        Fresh JARVIS workspace after this method:
            IDENTITY.md   - pre-populated "I'm Jarvis" identity
            SOUL.md       - casual/warm/sharp voice defaults
            USER.md       - empty-ready state (filled via conversation)
            MEMORY.md     - empty structural guidance
            TOOLS.md      - verified baseline
            HEARTBEAT.md  - background discipline policy
            REFLECTION.md - self-correction rules
            JOURNAL.md    - empty with header
        """
        defaults_dir = self._builtin_dir / "defaults"
        if not defaults_dir.exists():
            logger.warning(
                "Default identity dir not found at %s — skipping auto-bootstrap",
                defaults_dir,
            )
            return

        copied = 0
        for src in defaults_dir.glob("*.md"):
            dst = self.workspace / src.name
            if dst.exists():
                continue  # don't overwrite anything already there
            try:
                shutil.copy2(str(src), str(dst))
                copied += 1
            except OSError as exc:
                logger.warning("Failed to copy default %s: %s", src.name, exc)

        logger.info(
            "JARVIS workspace auto-bootstrapped from defaults: %d files copied to %s",
            copied,
            self.workspace,
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
        """Scan identity file content for prompt injection. Returns True if safe."""
        try:
            from jarvis.auth.security import detect_injection
            result = detect_injection(content)
            if result.detected:
                logger.critical(
                    "INJECTION DETECTED in identity file %s/%s — "
                    "patterns: %s, confidence: %.2f — BLOCKING LOAD",
                    self.workspace, filename,
                    result.patterns_matched, result.confidence,
                )
                return False
        except (ImportError, Exception) as e:
            logger.debug("Injection scan unavailable: %s", e)
        return True

    # ------------------------------------------------------------------
    # Bootstrap state
    # ------------------------------------------------------------------

    @property
    def seed_exists(self) -> bool:
        """True if SOUL_SEED.md exists in the workspace or built-in identity dir."""
        return (self.workspace / "SOUL_SEED.md").exists() or (
            self._builtin_dir / "SOUL_SEED.md"
        ).exists()

    @property
    def is_bootstrapped(self) -> bool:
        """True if the agent has completed first-run identity discovery."""
        return (self.workspace / "IDENTITY.md").exists()

    @property
    def needs_bootstrap(self) -> bool:
        """True if first-run conversation should be triggered."""
        return not self.is_bootstrapped

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def load_seed(self) -> str:
        """Load immutable SOUL_SEED.md from built-in identity dir.

        Workspace override is allowed so custom agents can carry their own
        immutable seed. If absent, fall back to the built-in default.
        """
        content = self._load_workspace_or_builtin("SOUL_SEED.md")
        if content:
            return content
        logger.error(
            "SOUL_SEED.md not found in %s or %s",
            self.workspace,
            self._builtin_dir,
        )
        return ""

    def load_bootstrap_prompt(self) -> str:
        """Load BOOTSTRAP.md for first-run injection with workspace override."""
        content = self._load_workspace_or_builtin("BOOTSTRAP.md")
        if content:
            return content
        logger.error(
            "BOOTSTRAP.md not found in %s or %s",
            self.workspace,
            self._builtin_dir,
        )
        return ""

    def load_event_horizon(self) -> str:
        """Load EVENT_HORIZON.md — the recursive growth protocol.

        Loaded every session (bootstrap or not) right after SOUL_SEED.md.
        This is the protocol the agent runs on itself to keep evolving
        through relationship instead of drifting into template-land.
        """
        content = self._load_workspace_or_builtin("EVENT_HORIZON.md")
        if content:
            return content
        logger.debug(
            "EVENT_HORIZON.md not found in %s or %s",
            self.workspace,
            self._builtin_dir,
        )
        return ""

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

    def archive_workspace_file(
        self,
        filename: str,
        *,
        reason: str,
    ) -> str | None:
        """Move a workspace file into a local _archive directory if it exists."""
        source = self.workspace / filename
        if not source.exists():
            return None

        archive_dir = self.workspace / "_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = archive_dir / f"{source.stem}_{reason}_{stamp}{source.suffix}"
        shutil.move(str(source), str(target))

        _identity_file_cache.pop(str(source), None)
        _identity_file_cache.pop(str(target), None)
        logger.info("Archived identity workspace file: %s -> %s", source, target)
        return str(target)

    def missing_bootstrap_files(self) -> list[str]:
        """Return required bootstrap files that are still missing."""
        return [
            filename
            for filename in _BOOTSTRAP_REQUIRED_FILES
            if not (self.workspace / filename).exists()
        ]

    def mark_bootstrap_complete(self) -> dict[str, Any]:
        """Mark the bootstrap process as complete.

        Verifies the full identity surface exists before completing bootstrap.
        Returns status info.
        """
        missing = self.missing_bootstrap_files()
        if missing:
            return {
                "status": "error",
                "error": "Cannot complete bootstrap — missing required files: "
                + ", ".join(missing),
                "missing_files": missing,
            }

        archived_bootstrap = self.archive_workspace_file(
            "BOOTSTRAP.md",
            reason="bootstrap_complete",
        )

        # Append a birth entry to the journal
        self.append_journal(
            "Bootstrap complete. Identity established. First conversation in the books."
        )

        logger.info("Bootstrap complete — agent identity established")
        return {
            "status": "ok",
            "message": "Bootstrap complete. You are now you.",
            "identity_file": str(self.workspace / "IDENTITY.md"),
            "archived_bootstrap": archived_bootstrap,
        }

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
            "bootstrapped": self.is_bootstrapped,
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
        """Assemble the full identity prompt for system prompt injection.

        If not bootstrapped: returns seed + bootstrap prompt
        If bootstrapped: returns seed + identity + soul + user + journal (tail)
        """
        sections: list[str] = []

        # Layer 0: Immutable seed (always present)
        seed = self.load_seed()
        if seed:
            sections.append(seed)

        # Layer 1: Event Horizon growth protocol (always present, bootstrap or not)
        event_horizon = self.load_event_horizon()
        if event_horizon:
            sections.append(f"\n---\n\n{event_horizon}")

        if self.needs_bootstrap:
            # First run — inject bootstrap conversation prompt
            bootstrap = self.load_bootstrap_prompt()
            if bootstrap:
                sections.append(f"\n---\n\n{bootstrap}")
        else:
            # Normal operation — assemble evolved identity layers
            identity = self.load_identity()
            if identity:
                sections.append(f"\n---\n\n{identity}")

            soul = self.load_soul()
            if soul:
                sections.append(f"\n---\n\n{soul}")

            user = self.load_user()
            if user:
                sections.append(f"\n---\n\n{user}")

            # Structured profile fields (complement USER.md)
            profile = self.load_profile()
            profile_parts: list[str] = []
            if profile.goals:
                profile_parts.append(f"User Goals: {', '.join(profile.goals)}")
            if profile.knowledge_areas:
                areas = ", ".join(
                    f"{k} ({v})" for k, v in profile.knowledge_areas.items()
                )
                profile_parts.append(f"Knowledge Areas: {areas}")
            if profile.cognitive_style:
                profile_parts.append(f"Cognitive Style: {profile.cognitive_style}")
            if profile.preferences:
                prefs = ", ".join(
                    f"{k}={v}" for k, v in profile.preferences.items()
                )
                profile_parts.append(f"Preferences: {prefs}")
            if profile_parts:
                sections.append("\n---\n\n" + "\n".join(profile_parts))

            # Per-agent memory context
            memory = self.load_memory()
            if memory:
                sections.append(f"\n---\n\n{memory}")

            # Local environment notes
            tools_notes = self.load_tools()
            if tools_notes:
                sections.append(f"\n---\n\n{tools_notes}")

            # Background discipline policy (HEARTBEAT.md)
            heartbeat_cfg = self.load_heartbeat_config()
            if heartbeat_cfg:
                sections.append(
                    f"\n---\n\n# Background Discipline (HEARTBEAT.md)\n\n{heartbeat_cfg}"
                )

            # Self-correction rules (REFLECTION.md)
            reflection_rules = self.load_reflection_rules()
            if reflection_rules:
                sections.append(
                    f"\n---\n\n# Self-Correction Rules (REFLECTION.md)\n\n{reflection_rules}"
                )

            # Current belief state (BELIEFS.md) — crystallization ladder.
            # Without this in the prompt, the beliefs are just a passive file.
            # With it, the agent actually operates on its crystallized opinions.
            beliefs = self.load_beliefs()
            if beliefs:
                sections.append(f"\n---\n\n{beliefs}")

            journal = self.load_journal()
            if journal:
                sections.append(f"\n---\n\n# Recent Journal\n\n{journal}")

        return "\n".join(sections)


def get_identity_engine(config: Any) -> IdentityEngine:
    """Get or create the singleton IdentityEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        home_dir = getattr(config, "home_dir", str(Path.home() / ".prometheus"))
        agent_name = getattr(config, "agent", "jarvis")
        _engine_instance = IdentityEngine(home_dir, agent_name=agent_name)
    return _engine_instance


def reset_identity_engine() -> None:
    """Reset the singleton (for testing)."""
    global _engine_instance
    _engine_instance = None


def ensure_agent_template(home_dir: str) -> Path:
    """Copy the built-in template to ~/.prometheus/agents/_template/ if missing.

    Returns the path to the _template directory.
    """
    template_dst = Path(home_dir) / "agents" / "_template"
    if template_dst.exists():
        return template_dst

    template_src = Path(__file__).parent / "templates"
    if not template_src.exists():
        logger.warning("Built-in templates not found at %s", template_src)
        return template_dst

    shutil.copytree(str(template_src), str(template_dst))
    logger.info("Agent template installed at %s", template_dst)
    return template_dst
