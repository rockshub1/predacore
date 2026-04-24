"""
PredaCore Core — The conversational agent brain.

Orchestrates the full agent loop: receive user message → build prompt →
call LLM → execute tools → return response.  All subsystems are imported
from their own modules; this file is the slim orchestrator.

Usage (via Gateway):
    core = PredaCoreCore(config)
    response = await core.process("user123", "what's the weather?", session)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agents.meta_cognition import (
    evaluate_response as _evaluate_response,
)
from .agents.meta_cognition import (
    should_ask_for_help as _should_ask_for_help,
)
from .config import PredaCoreConfig
from .services.outcome_store import OutcomeStore, TaskOutcome, detect_feedback
from .services.transcripts import TranscriptWriter
from .sessions import Session

logger = logging.getLogger(__name__)


def _llm_error_message(exc: Exception) -> str:
    """Map provider failures to a user-facing message."""
    text = str(exc).lower()
    if (
        "usage limit" in text
        or "usage_limit_reached" in text
        or "quota" in text
        or "upgrade" in text
    ):
        return (
            "The active Codex account has reached its usage limit. "
            "Switch accounts or upgrade the account plan, then try again."
        )
    if "429" in text or "rate limit" in text:
        return (
            "I'm currently rate-limited by the AI provider. "
            "Please try again in a minute or two."
        )
    if "overloaded" in text or "529" in text:
        return "The AI provider is overloaded right now. Please try again shortly."
    return (
        "I'm sorry, I encountered a connectivity issue with my language model. "
        "Please try again in a moment."
    )


def _context_budget_for_provider(provider: str, model: str) -> tuple[int, int]:
    """Choose a safe prompt budget for the active provider/model.

    Returns ``(budget_tokens, history_min_tokens)``.

    - ``budget_tokens`` — target total prompt size per LLM call. Keep well
      under any model's absolute max so attention stays in its high-quality
      zone ("lost in the middle" measurably degrades past ~200k).
    - ``history_min_tokens`` — floor reserved for recent turn history after
      identity + memory recall + tools have taken their share. Prevents
      "huge identity starves recent chat" pathologies.

    Override via the ``PREDACORE_CONTEXT_BUDGET`` env var (integer token
    count, e.g. 100000). When set, history floor scales to budget / 6.
    """
    env_budget = os.getenv("PREDACORE_CONTEXT_BUDGET", "").strip()
    if env_budget.isdigit():
        budget = int(env_budget)
        # Floor history reserve at 4k; scale with budget for bigger windows.
        return budget, max(budget // 6, 4_000)

    # Default: 80k budget, 14k history floor. Works for every modern LLM
    # (Opus 4.7 @ 1M, Gemini 3 Pro @ 2M, Sonnet @ 200k, GPT-5 @ 400k).
    # At 80k we're comfortably in the 99%+ attention band on every model.
    budget_tokens = 80_000
    history_min_tokens = 14_000
    return budget_tokens, history_min_tokens


# ── Re-exports from extracted modules ──────────────────────────────────
from .llm_providers.router import LLMInterface
from .prompts import (
    DIRECT_CAPABILITY_DENIAL_RE,
    DIRECT_TOOL_PREFIX_RE,
    PERSONA_DRIFT_PATTERNS,
    PERSONA_IDENTITY_QUERY_RE,
    UNVERIFIED_ACTION_CLAIM_RE,
    UNVERIFIED_MODEL_SWITCH_RE,
    VERIFICATION_REQUEST_RE,
    PersonaDriftAssessment,
    _get_system_prompt,
)
from .tools.executor import ToolExecutor
from .tools.registry import (
    BUILTIN_TOOLS_RAW,
    MARKETPLACE_TOOLS_RAW,
    OPENCLAW_BRIDGE_TOOLS_RAW,
)

# Legacy compatibility: flat lists for existing code that references them
BUILTIN_TOOLS = [item[0] for item in BUILTIN_TOOLS_RAW]
OPENCLAW_BRIDGE_TOOLS = [item[0] for item in OPENCLAW_BRIDGE_TOOLS_RAW]
MARKETPLACE_TOOLS = [item[0] for item in MARKETPLACE_TOOLS_RAW]


# ── Sensitive data redaction ──────────────────────────────────────────

_SENSITIVE_KEYS_EXACT = frozenset(
    {
        "password",
        "token",
        "api_key",
        "apikey",
        "secret",
        "credentials",
        "credential",
        "access_token",
        "refresh_token",
        "private_key",
        "auth",
        "authorization",
    }
)

# Substring patterns: any key containing these is considered sensitive
# (catches e.g. "db_password", "api_secret", "jwt_token", "auth_header")
_SENSITIVE_SUBSTRINGS = ("password", "token", "api_key", "secret", "credential", "auth")


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name refers to a sensitive field."""
    lower = key.lower()
    if lower in _SENSITIVE_KEYS_EXACT:
        return True
    return any(sub in lower for sub in _SENSITIVE_SUBSTRINGS)


def _redact_tool_args(args: dict[str, Any], max_len: int = 200) -> str:
    """Serialize tool args for logging, redacting sensitive fields."""
    if not args:
        return "{}"
    redacted = {}
    for key, value in args.items():
        if _is_sensitive_key(key):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    raw = json.dumps(redacted, default=str)
    if len(raw) > max_len:
        return raw[:max_len] + "..."
    return raw


def _task_exception_handler(task: asyncio.Task) -> None:
    """Log exceptions from background tasks instead of silently dropping them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task %s failed: %s", task.get_name(), exc, exc_info=exc
        )


# ── Project root tracking ─────────────────────────────────────────────
#
# Long sessions drift: the model pins a project folder (say ~/ReviewPilot),
# writes 20 files, and 40 tool calls later silently "re-invents" the path
# (~/Developer/ReviewPilot) — usually because the second path looks more
# plausible a priori. We pin the first project directory the session touches
# in session.metadata["project_root"] and inject a system reminder into every
# LLM call so the path stays load-bearing across long chains.

_PROJECT_ROOT_MARKER = "[Session Project Root]"

# Directories under $HOME that are containers (not projects themselves).
# If `mkdir -p ~/Projects/reviewpilot` is run, the project root is
# `~/Projects/reviewpilot`, not `~/Projects`. Desktop/Documents/Downloads
# are intentionally excluded — they're user dumping grounds, not dev containers.
_CONTAINER_DIRS = frozenset({
    "Projects", "projects", "Developer", "dev", "Dev",
    "Code", "code", "workspace", "workspaces", "Workspace",
    "src", "source", "Source", "repos", "Repos", "git", "Git",
})

_MKDIR_RE = re.compile(r"mkdir\s+(?:-\S+\s+)*(~?/[^\s&|;]+)")
_CD_RE = re.compile(r"\bcd\s+(~?/[^\s&|;]+)")


def _resolve_path(path: str) -> str:
    """Expand `~` and return absolute path without requiring it to exist."""
    return os.path.abspath(os.path.expanduser(path))


def _infer_project_root_from_path(path: str, home: str) -> str | None:
    """
    Given a path under $HOME, infer the project root.

    Rules:
      - Path must resolve under $HOME.
      - If the first component under $HOME is a known container dir
        (Projects, Developer, Code, ...) and there's a second component,
        the project root is $HOME/container/project.
      - Otherwise, the root is $HOME/first-component.
      - $HOME itself and direct children that look like system dirs
        (e.g. .cache, Library) are never pinned.
    """
    try:
        abs_path = _resolve_path(path)
        abs_home = _resolve_path(home)
    except (OSError, ValueError):
        return None
    if not abs_path.startswith(abs_home + os.sep) and abs_path != abs_home:
        return None
    rel = os.path.relpath(abs_path, abs_home)
    if rel in (".", ""):
        return None
    parts = rel.split(os.sep)
    first = parts[0]
    if not first or first.startswith("."):
        return None
    if first in _CONTAINER_DIRS and len(parts) >= 2 and parts[1] and not parts[1].startswith("."):
        return os.path.join(abs_home, first, parts[1])
    return os.path.join(abs_home, first)


def _extract_paths_from_tool_call(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Pull filesystem paths out of a tool call's arguments."""
    paths: list[str] = []
    if tool_name in {"write_file", "read_file", "edit_file", "delete_file"}:
        for key in ("path", "file_path", "filename"):
            if args.get(key):
                paths.append(str(args[key]))
                break
    elif tool_name in {"create_directory", "list_directory"}:
        for key in ("path", "directory", "dir"):
            if args.get(key):
                paths.append(str(args[key]))
                break
    elif tool_name == "run_command":
        cwd = args.get("cwd")
        if cwd:
            paths.append(str(cwd))
        cmd = str(args.get("command", ""))
        for match in _MKDIR_RE.finditer(cmd):
            paths.append(match.group(1))
        for match in _CD_RE.finditer(cmd):
            paths.append(match.group(1))
    return paths


def _maybe_pin_project_root(
    session: Any,
    tool_name: str,
    tool_args: dict[str, Any],
    home: str,
    ctx_memory: dict[str, Any] | None = None,
) -> None:
    """Pin session project root from the first qualifying tool call.

    Writes to BOTH:
      - session.metadata["project_root"] — persisted to meta.json, survives restart
      - ctx_memory["_session_project_root"] — ephemeral, readable by sub-agents
        (multi_agent handler, openclaw delegate) so they inherit the pin
    """
    if not session or not hasattr(session, "metadata"):
        return
    if session.metadata.get("project_root"):
        # Already pinned — still mirror to ctx_memory in case it was cleared
        if ctx_memory is not None:
            ctx_memory["_session_project_root"] = {
                "content": session.metadata["project_root"],
                "source": "core.pin",
            }
        return
    for path in _extract_paths_from_tool_call(tool_name, tool_args):
        root = _infer_project_root_from_path(path, home)
        if root:
            session.metadata["project_root"] = root
            if ctx_memory is not None:
                ctx_memory["_session_project_root"] = {
                    "content": root,
                    "source": "core.pin",
                }
            logger.info("Session project root pinned: %s (from %s)", root, tool_name)
            return


def _check_project_drift(
    session: Any,
    tool_name: str,
    tool_args: dict[str, Any],
    home: str,
) -> str | None:
    """
    If the session has a pinned project root and this tool call targets a
    different top-level project folder under $HOME, return a warning string.
    Otherwise None.
    """
    if not session or not hasattr(session, "metadata"):
        return None
    pinned = session.metadata.get("project_root")
    if not pinned:
        return None
    abs_pinned = _resolve_path(pinned)
    for path in _extract_paths_from_tool_call(tool_name, tool_args):
        abs_path = _resolve_path(path)
        if abs_path == abs_pinned or abs_path.startswith(abs_pinned + os.sep):
            continue
        inferred = _infer_project_root_from_path(path, home)
        if inferred and inferred != abs_pinned:
            return (
                f"[Project drift detected] You tried to write to `{path}` "
                f"(resolves to project `{inferred}`), but this session's "
                f"active project is `{pinned}`. Use `{pinned}` instead. "
                "If the user explicitly asked you to switch projects, tell "
                "them first and confirm — do not silently create a new folder."
            )
    return None


def _inject_project_root_reminder(messages: list[dict[str, Any]], session: Any) -> None:
    """Refresh the project-root system reminder in the messages list in-place."""
    # Strip any stale reminder first
    i = 0
    while i < len(messages):
        content = messages[i].get("content")
        if (
            messages[i].get("role") == "system"
            and isinstance(content, str)
            and content.startswith(_PROJECT_ROOT_MARKER)
        ):
            messages.pop(i)
            continue
        i += 1
    if not session or not hasattr(session, "metadata"):
        return
    root = session.metadata.get("project_root")
    if not root:
        return
    reminder = (
        f"{_PROJECT_ROOT_MARKER} {root}\n\n"
        "All file operations for this project MUST use this exact path. "
        "Do NOT create new top-level project folders (e.g. ~/Projects/xxx, "
        "~/Developer/xxx, ~/xxx) when a project root is already pinned. "
        "To add a subdirectory, create it UNDER this path. "
        "If the user explicitly asks to switch projects, confirm with them "
        "first and then update this pin — never drift silently."
    )
    # Insert right after the first system message so it rides shotgun with
    # the main system prompt and survives message trimming.
    insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
    messages.insert(insert_idx, {"role": "system", "content": reminder})


# ── Parallel tool execution helpers ───────────────────────────────────

# Tools that mutate state and should not run in parallel
_MUTATING_TOOLS = frozenset(
    {
        "write_file",
        "run_command",
        "python_exec",
        "execute_code",
        "delete_file",
        "create_directory",
        "desktop_control",
        "manage_memory",
        "marketplace_install",
        "marketplace_uninstall",
    }
)

# Tools that read a specific file — two reads of the same file are fine,
# but a read + write to the same file is not.
_FILE_READ_TOOLS = frozenset({"read_file", "search_files", "list_directory"})


def _classify_tool_dependencies(
    tool_calls: list[dict[str, Any]],
) -> tuple:
    """
    Split tool calls into independent (parallelizable) and dependent (sequential).

    Rules:
    1. Read-only tools with no shared resources → independent
    2. Mutating tools → dependent (sequential)
    3. Multiple writes → dependent
    4. Read + write to same path → dependent
    """
    if not tool_calls:
        return [], []

    independent: list[dict[str, Any]] = []
    dependent: list[dict[str, Any]] = []
    has_mutating = False

    for tc in tool_calls:
        name = tc.get("name", "")
        if name in _MUTATING_TOOLS:
            dependent.append(tc)
            has_mutating = True
        else:
            independent.append(tc)

    # If we have both reads and writes, check for path conflicts
    if has_mutating and independent:
        write_paths = set()
        for tc in dependent:
            args = tc.get("arguments", {})
            for key in ("path", "file_path", "filename"):
                if key in args:
                    write_paths.add(str(args[key]))

        if write_paths:
            still_independent = []
            for tc in independent:
                args = tc.get("arguments", {})
                tc_paths = set()
                for key in ("path", "file_path", "filename"):
                    if key in args:
                        tc_paths.add(str(args[key]))
                if tc_paths & write_paths:
                    dependent.append(tc)
                else:
                    still_independent.append(tc)
            independent = still_independent

    return independent, dependent


# ── PredaCore Core ──────────────────────────────────────────────────────


class PredaCoreCore:
    """
    The conversational agent brain.

    Orchestrates the full agent loop:
    1. Receive user message + session context
    2. Build LLM prompt with system prompt + history + tools
    3. Call LLM
    4. If tool calls → execute tools → feed results back to LLM
    5. Return final response
    """

    MAX_TOOL_ITERATIONS = 200

    def __init__(self, config: PredaCoreConfig):
        self.config = config

        # Propagate config flags to env so child subprocess tools can read them.
        if config.launch.enable_openclaw_bridge:
            os.environ["PREDACORE_ENABLE_OPENCLAW_BRIDGE"] = "1"
        if config.launch.enable_plugin_marketplace:
            os.environ["PREDACORE_ENABLE_PLUGIN_MARKETPLACE"] = "1"

        self.llm = LLMInterface(config)
        self.tools = ToolExecutor(config)
        # Note: tools._llm_for_collab is set by SubsystemFactory — no monkey-patch needed

        self._system_prompt = _get_system_prompt(config)
        self._tool_definitions = list(BUILTIN_TOOLS)
        requested_tool_iters = int(
            config.launch.max_tool_iterations or self.MAX_TOOL_ITERATIONS
        )
        # Hard safety cap raised per user request "remove all the limits".
        # Previously 100 (which was already generous). Now 1000 — effectively
        # unlimited for any normal workflow. A build-the-whole-app task can
        # easily hit 50-100 iterations and that's fine.
        _ABSOLUTE_MAX_ITERATIONS = 1000
        self._max_tool_iterations = max(1, min(requested_tool_iters, _ABSOLUTE_MAX_ITERATIONS))
        self._persona_drift_guard_enabled = bool(
            config.launch.enable_persona_drift_guard
        )
        self._persona_drift_threshold = min(
            max(float(config.launch.persona_drift_threshold), 0.0), 1.0
        )
        self._persona_drift_max_regens = max(
            0, int(config.launch.persona_drift_max_regens)
        )
        self._persona_anchor_terms = self._build_persona_anchor_terms()

        if config.launch.enable_openclaw_bridge:
            self._tool_definitions.extend(OPENCLAW_BRIDGE_TOOLS)
            logger.info("OpenClaw bridge tools active (%d)", len(OPENCLAW_BRIDGE_TOOLS))

        if (
            config.launch.enable_plugin_marketplace
            and getattr(self.tools, '_skill_marketplace', None) is not None
        ):
            self._tool_definitions.extend(MARKETPLACE_TOOLS)
            logger.info("Marketplace tools active (%d)", len(MARKETPLACE_TOOLS))

        if config.security.allowed_tools:
            self._tool_definitions = [
                t
                for t in self._tool_definitions
                if t["name"] in config.security.allowed_tools
            ]
        if config.security.blocked_tools:
            self._tool_definitions = [
                t
                for t in self._tool_definitions
                if t["name"] not in config.security.blocked_tools
            ]

        self._transcript = TranscriptWriter(
            output_dir=Path(config.home_dir) / "transcripts"
        )
        self._transcript_sessions: set = set()

        # ── Unified Memory Retriever ──────────────────────────────────
        self._memory_retriever = None
        if self.tools._unified_memory:
            try:
                from .memory import MemoryRetriever

                self._memory_retriever = MemoryRetriever(
                    store=self.tools._unified_memory,
                )
                logger.info("Unified MemoryRetriever active")
            except (ImportError, OSError, ValueError, TypeError) as exc:
                logger.warning("MemoryRetriever init failed: %s", exc)

        # Outcome tracking (Phase 2)
        outcome_db = Path(config.home_dir) / "outcomes.db"
        try:
            self._outcome_store = OutcomeStore(outcome_db)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("OutcomeStore init failed (non-fatal): %s", exc)
            self._outcome_store = None

        # World model (JAX RSSM) — permanently disabled and removed from
        # this tree. All downstream paths short-circuit on
        # self._world_model is None.
        self._world_model = None

        # Per-tool execution timeout. Default raised from 60s → 1800s (30min)
        # per user request "remove all the limits". Long-running builds,
        # deep_search, strategic_plan, and large code_exec runs need headroom.
        self._tool_timeout = int(os.getenv("PREDACORE_TOOL_TIMEOUT_SECONDS", "1800"))

        logger.info(
            "PredaCoreCore initialized (trust=%s, tools=%d, provider=%s, max_tool_iterations=%d)",
            config.security.trust_level,
            len(self._tool_definitions),
            config.llm.provider,
            self._max_tool_iterations,
        )

    # ── System prompt refresh ───────────────────────────────────────

    def refresh_system_prompt(self) -> None:
        """Rebuild system prompt (call after identity file updates)."""
        self._system_prompt = _get_system_prompt(self.config)
        logger.info("System prompt refreshed (%d chars)", len(self._system_prompt))

    # ── Persona drift guard ──────────────────────────────────────────

    def _build_persona_anchor_terms(self) -> set[str]:
        """Build canonical identity anchors used by drift scoring."""
        anchors = {
            str(self.config.name or "PredaCore").strip().lower(),
            "prometheus",
            "predacore",
        }

        env_raw = os.getenv("PREDACORE_PERSONA_ANCHORS", "")
        if env_raw:
            for part in env_raw.split(","):
                token = part.strip().lower()
                if token:
                    anchors.add(token)

        return {a for a in anchors if a}

    def _assess_persona_drift(
        self,
        user_message: str,
        assistant_message: str,
        tools_used: int = 0,
    ) -> PersonaDriftAssessment:
        """Compute a deterministic [0,1] drift score from textual heuristics."""
        threshold = self._persona_drift_threshold
        if not self._persona_drift_guard_enabled:
            return PersonaDriftAssessment(score=0.0, threshold=threshold, reasons=[])

        text = (assistant_message or "").strip()
        if not text:
            # Tool-only turns legitimately have no text — not drift.
            if tools_used > 0:
                return PersonaDriftAssessment(
                    score=0.0, threshold=threshold, reasons=[]
                )
            return PersonaDriftAssessment(
                score=1.0,
                threshold=threshold,
                reasons=["empty_response"],
            )

        lowered = text.lower()
        reasons: list[str] = []
        score = 0.0

        for pattern, weight, reason in PERSONA_DRIFT_PATTERNS:
            if pattern.search(lowered):
                score += weight
                reasons.append(reason)

        if DIRECT_CAPABILITY_DENIAL_RE.search(lowered):
            score += 0.70
            reasons.append("direct_shell_capability_denial")

        if re.search(
            r"\b(i cannot|i can't|i can not|i don't)\b", lowered
        ) and re.search(r"\b(tool|tools|command|commands|file|files|shell)\b", lowered):
            score += 0.20
            reasons.append("tool_capability_denial")

        if tools_used <= 0:
            if UNVERIFIED_MODEL_SWITCH_RE.search(lowered):
                score += 0.80
                reasons.append("claimed_model_switch_without_tools")

            verification_intent = bool(
                VERIFICATION_REQUEST_RE.search((user_message or "").lower())
            )
            if verification_intent and UNVERIFIED_ACTION_CLAIM_RE.search(lowered):
                score += 0.60
                reasons.append("claimed_unverified_action_for_verification_request")
            elif UNVERIFIED_ACTION_CLAIM_RE.search(lowered):
                score += 0.35
                reasons.append("claimed_unverified_action")

        if PERSONA_IDENTITY_QUERY_RE.search((user_message or "").lower()):
            if self._persona_anchor_terms and not any(
                re.search(r'\b' + re.escape(anchor) + r'\b', lowered)
                for anchor in self._persona_anchor_terms
            ):
                score += 0.35
                reasons.append("identity_query_without_anchor")

        # Deduplicate overlapping capability-denial reasons: if both the
        # weaker pattern-based "capability_denial_claim" and a stronger
        # reason containing "capability_denial" fired, drop the weaker one
        # to avoid inflated scores on factually correct responses.
        if "capability_denial_claim" in reasons and any(
            r != "capability_denial_claim" and "capability_denial" in r
            for r in reasons
        ):
            reasons.remove("capability_denial_claim")
            score -= 0.10  # weight of capability_denial_claim after fix 3.5

        return PersonaDriftAssessment(
            score=min(max(score, 0.0), 1.0),
            threshold=threshold,
            reasons=reasons,
        )

    def _persona_regeneration_instruction(
        self,
        assessment: PersonaDriftAssessment,
        tools_used: int,
    ) -> str:
        """Instruction used when a response drifts from the configured persona."""
        reasons = ", ".join(assessment.reasons) if assessment.reasons else "unknown"
        base = (
            "Regenerate your previous answer with persona alignment.\n"
            f"Drift triggers: {reasons}.\n"
            f"Requirements:\n"
            f"- Keep the core factual content and user intent unchanged.\n"
            f"- Speak as {self.config.name} from PredaCore.\n"
            "- Do not claim to be ChatGPT, Claude, Gemini, or a generic language model.\n"
            "- Do not incorrectly deny tool/shell/file capabilities.\n"
            "- Do not claim terminal/tool/file actions were executed unless you actually "
            "ran tools in this turn.\n"
            "- Return only the corrected final answer."
        )
        if tools_used <= 0:
            base += (
                "\n- In this turn, no tool calls were made. You may use tools if "
                "appropriate, but do not claim you already used them without evidence."
            )
        return base

    async def _apply_persona_drift_guard(
        self,
        user_message: str,
        messages: list[dict[str, str]],
        initial_content: str,
        tools_used: int = 0,
    ) -> tuple[str, PersonaDriftAssessment, int]:
        """Regenerate response if drift score exceeds threshold."""
        assessment = self._assess_persona_drift(
            user_message, initial_content, tools_used=tools_used
        )
        if (
            not self._persona_drift_guard_enabled
            or not assessment.needs_regeneration
            or self._persona_drift_max_regens <= 0
        ):
            return initial_content, assessment, 0

        best_content = initial_content
        best_assessment = assessment
        regen_count = 0

        while (
            regen_count < self._persona_drift_max_regens
            and best_assessment.needs_regeneration
        ):
            regen_count += 1
            logger.warning(
                "Persona drift detected (score=%.2f >= %.2f). Regenerating %d/%d. "
                "Reasons=%s",
                best_assessment.score,
                best_assessment.threshold,
                regen_count,
                self._persona_drift_max_regens,
                ",".join(best_assessment.reasons) or "none",
            )

            regen_messages = list(messages)
            if best_content:
                regen_messages.append({"role": "assistant", "content": best_content})
            regen_messages.append(
                {
                    "role": "system",
                    "content": self._persona_regeneration_instruction(
                        best_assessment,
                        tools_used=tools_used,
                    ),
                }
            )

            try:
                regenerated = await self.llm.chat(messages=regen_messages, tools=None)
            except (ConnectionError, TimeoutError, OSError, RuntimeError) as exc:
                logger.warning("Persona drift regeneration failed: %s", exc)
                break

            candidate = regenerated.get("content", "")
            candidate_assessment = self._assess_persona_drift(
                user_message, candidate, tools_used=tools_used
            )
            improved = candidate_assessment.score < best_assessment.score or (
                candidate_assessment.score == best_assessment.score
                and len(candidate) > len(best_content)
            )
            if candidate and improved:
                best_content = candidate
                best_assessment = candidate_assessment

        return best_content, best_assessment, regen_count

    # ── Runtime memory ───────────────────────────────────────────────

    async def _build_runtime_memory_context(self, user_id: str, query: str) -> str:
        """Fetch relevant persistent memories to ground the current turn."""
        service = self.tools._memory_service
        if service is None:
            return ""

        query = (query or "").strip()
        if not query:
            return ""

        try:
            top_k = int(os.getenv("PREDACORE_MEMORY_PROMPT_TOP_K", "5"))
        except (ValueError, TypeError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

        try:
            max_chars = int(os.getenv("PREDACORE_MEMORY_PROMPT_MAX_CHARS", "1800"))
        except (ValueError, TypeError):
            max_chars = 1800
        max_chars = max(300, min(max_chars, 12000))

        try:
            recalls = await service.recall(query=query, user_id=user_id, top_k=top_k)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Runtime memory recall unavailable for prompt injection: %s", exc
            )
            return ""

        if not recalls:
            return ""

        lines: list[str] = []
        for mem, score in recalls:
            snippet = " ".join(str(mem.content).split())
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            lines.append(f"- ({score:.2f}) [{mem.memory_type.value}] {snippet}")

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[: max_chars - 3] + "..."
        return context

    # ── Post-turn memory storage ────────────────────────────────────

    async def _store_turn_memory(
        self,
        user_id: str,
        message: str,
        response: str,
        tools_used: list[str],
        session_id: str,
        provider: str,
    ) -> None:
        """Store conversation turn in unified memory (non-fatal)."""
        um = self.tools._unified_memory
        if um is None:
            return
        try:
            snippet = response[:1500] if response else ""
            # Variable importance: tool-heavy or long responses are more valuable
            importance = 2
            if tools_used:
                importance = 3
            if len(response or "") > 2000:
                importance = 4
            await um.store(
                content=f"User: {message}\nPredaCore: {snippet}",
                memory_type="conversation",
                importance=importance,
                session_id=session_id,
                user_id=user_id,
                metadata={"tools_used": tools_used, "provider": provider},
                memory_scope="global",
            )
            # Event-driven consolidation: trigger light consolidation every 50 memories
            try:
                stats = await um.get_stats()
                total = stats.get("total_memories", 0)
                if total > 0 and total % 50 == 0:
                    logger.info(
                        "Triggering event-driven consolidation (total=%d)", total
                    )
                    from .memory.consolidator import MemoryConsolidator

                    consolidator = MemoryConsolidator(um)
                    task = asyncio.create_task(
                        consolidator.consolidate_recent(last_n=50),
                        name="memory_consolidation",
                    )
                    task.add_done_callback(_task_exception_handler)
            except (ImportError, OSError, RuntimeError) as _exc:
                logger.debug(
                    "Event-driven consolidation check failed (non-fatal)", exc_info=True
                )
        except (OSError, ValueError, RuntimeError) as _exc:
            logger.debug("Post-turn memory store failed (non-fatal)", exc_info=True)

    # ── Decision journal (per-turn reasoning trace) ────────────────────

    async def _append_decision_journal(
        self,
        *,
        message: str,
        response: str,
        tools_used: list[str],
        tool_errors: list[str],
        iterations: int,
        start_time: float,
        usage: dict[str, Any],
        session_id: str,
    ) -> None:
        """
        Append a DECISIONS.md entry for this turn. Non-fatal.

        Observable-facts reasoning trace: user input, response summary,
        tools used, duration, tokens, outcome. Lets the user (or future
        agent) ask "why did you do that?" and get a real answer.
        """
        try:
            from .identity.engine import get_identity_engine

            engine = get_identity_engine(self.config)

            duration = max(0.0, time.time() - start_time)
            tokens_in = int((usage or {}).get("prompt_tokens", 0) or 0)
            tokens_out = int((usage or {}).get("completion_tokens", 0) or 0)

            if tool_errors:
                outcome = f"partial ({len(tool_errors)} tool errors)"
            elif tools_used:
                outcome = "success"
            else:
                outcome = "no-op (no tools called)"

            # Summarize response to first ~300 chars (engine will also truncate)
            response_summary = response[:300] if response else ""

            engine.append_decision(
                session_id=session_id,
                user_message=message,
                response_summary=response_summary,
                tools_used=list(tools_used or []),
                files_touched=[],  # files_touched extraction deferred to v1.1
                duration_seconds=duration,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                outcome=outcome,
                iterations=iterations,
            )
        except (ImportError, OSError, RuntimeError) as _exc:
            logger.debug("Decision journal append failed (non-fatal)", exc_info=True)

    # ── Identity reflection (post-turn) ────────────────────────────────

    async def _maybe_identity_reflect(self) -> None:
        """
        Trigger identity reflection if due. Non-fatal, non-blocking.

        Fires an LLM-driven self-assessment every N conversations (default 20)
        with a small token budget (~500 tokens) and a 30-second timeout. The
        reflection reads IDENTITY/SOUL/USER/JOURNAL, assesses growth, and
        optionally updates SOUL.md and USER.md via the ReflectionEngine.

        Controlled by config.launch.identity_reflection_enabled (default: True).
        Set to False to disable the LLM call and fall back to counter-only
        journal checkpoints.
        """
        try:
            from .identity.engine import get_identity_engine
            from .identity.reflection import ReflectionEngine

            engine = get_identity_engine(self.config)

            if not hasattr(self, "_reflection_engine"):
                self._reflection_engine = ReflectionEngine(engine)

            # Honor config flag — if LLM reflection is disabled, just tick + journal
            reflection_enabled = getattr(
                getattr(self.config, "launch", None),
                "identity_reflection_enabled",
                True,
            )

            if not reflection_enabled:
                # Legacy counter-only path (no LLM cost)
                if self._reflection_engine.tick():
                    engine.append_journal(
                        f"Reflection checkpoint (conversation #{self._reflection_engine._conversation_count}). "
                        "LLM reflection disabled in config — counter-only mode."
                    )
                    logger.info("Identity reflection checkpoint recorded (counter-only)")
                return

            # Full LLM-driven reflection via ReflectionEngine.maybe_reflect
            async def _reflection_llm(prompt: str) -> str:
                """Minimal single-turn LLM call for reflection (no tools, small budget)."""
                try:
                    response = await asyncio.wait_for(
                        self.llm.chat(
                            messages=[{"role": "user", "content": prompt}],
                            tools=None,
                        ),
                        timeout=300,  # raised from 30 → 300 (reflection can be slow with max reasoning)
                    )
                    return response.get("content", "") if isinstance(response, dict) else str(response)
                except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
                    logger.debug("Reflection LLM call failed: %s", exc)
                    return ""

            reflected = await self._reflection_engine.maybe_reflect(llm_fn=_reflection_llm)
            if reflected:
                logger.info(
                    "Identity reflection completed (conversation #%d)",
                    self._reflection_engine._conversation_count,
                )
        except (ImportError, OSError, RuntimeError) as _exc:
            logger.debug("Identity reflection check failed (non-fatal)", exc_info=True)

    # ── Direct tool shortcut ─────────────────────────────────────────

    def _extract_direct_tool_shortcut(
        self, message: str
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Parse direct imperative tool requests from user text.

        Bypasses model tool-calling ambiguity for commands like:
          - "Run memory_recall and return raw JSON"
          - "execute memory_store {\"key\": \"test\", \"content\": \"...\"}"
        """
        text = str(message or "").strip()
        if not text:
            return None

        match = DIRECT_TOOL_PREFIX_RE.match(text)
        if not match:
            return None
        tool_name = match.group(1)
        available = set(self.get_tool_list())
        if tool_name not in available:
            return None

        remainder = text[match.end() :].strip()
        args: dict[str, Any] = {}
        if remainder:
            json_match = re.search(r"\{[^}]*\}", remainder)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, ValueError):
                    args = {}
        return (tool_name, args)

    def set_model(self, provider: str | None = None, model: str | None = None):
        """Hot-swap the active LLM model/provider without restarting."""
        self.llm.set_active_model(provider=provider, model=model)
        logger.info("Model changed: provider=%s model=%s", provider, model)

    # ── Main agent loop ──────────────────────────────────────────────

    async def process(
        self,
        user_id: str,
        message: str,
        session: Session,
        confirm_fn: Callable | None = None,
        stream_fn: Callable | None = None,
        event_fn: Callable | None = None,
    ) -> str:
        """
        Process a user message through the full agent loop.

        Args:
            user_id: The user's identifier
            message: The user's message text
            session: The current conversation session
            confirm_fn: Optional callback for tool confirmation (trust level)
            stream_fn: Optional callback for streaming partial responses
            event_fn: Optional callback for live UI events
        """
        start_time = time.time()
        self._loop_warnings = 0  # Reset loop detection per message

        # Transcript: ensure session is started, record user message
        try:
            sid = session.session_id
            if sid not in self._transcript_sessions:
                self._transcript.start_session(sid, user_id=user_id)
                self._transcript_sessions.add(sid)
            self._transcript.append_message(sid, "user", message)
        except OSError:
            logger.debug("Transcript write failed (non-fatal)", exc_info=True)

        # Feedback detection: check if this message is feedback on the last response
        if self._outcome_store is not None:
            feedback = detect_feedback(message)
            if feedback:
                try:
                    await self._outcome_store.update_feedback(user_id, feedback)
                except (OSError, RuntimeError) as _exc:
                    logger.debug("Feedback update failed (non-fatal)", exc_info=True)

        # Outcome tracking accumulators
        _tools_used: list[str] = []
        _tool_errors: list[str] = []
        _tool_history: list[tuple[str, dict[str, Any]]] = []
        _tool_results: list[str] = []

        shortcut = self._extract_direct_tool_shortcut(message)
        if shortcut is not None:
            tool_name, tool_args = shortcut
            logger.info(
                "Direct tool shortcut: %s(%s)",
                tool_name,
                _redact_tool_args(tool_args),
            )
            result = await self.tools.execute(
                tool_name,
                tool_args,
                confirm_fn=confirm_fn,
            )
            await self._record_outcome(
                user_id=user_id,
                message=message,
                response=result,
                tools_used=[tool_name],
                tool_errors=[],
                iterations=1,
                start_time=start_time,
                provider="",
                model="",
                usage={},
                drift_score=0.0,
                drift_regens=0,
                session_id=sid,
                success=True,
            )
            return result

        system_prompt = self._system_prompt

        memory_context = ""
        if self._memory_retriever:
            memory_context = await self._memory_retriever.build_context(
                query=message,
                user_id=user_id,
                session_id=sid,
            )
        else:
            memory_context = await self._build_runtime_memory_context(
                user_id, message
            )
        if memory_context:
            system_prompt = (
                f"{system_prompt}\n\n---\n\n"
                "# Retrieved Runtime Memory\n"
                "Use these memories only when relevant. If they conflict with the latest "
                "user message, trust the latest user message and ask a clarifying question.\n"
                f"{memory_context}"
            )

        # Build message history with a token-aware packing strategy.
        # Reserve headroom for tool schemas, model output, and loop retries.
        context_budget_tokens, history_min_tokens = _context_budget_for_provider(
            self.llm.active_provider,
            self.llm.active_model,
        )
        messages = [{"role": "system", "content": system_prompt}]
        system_tokens = Session.estimate_tokens(system_prompt)
        history_budget_tokens = max(
            history_min_tokens,
            context_budget_tokens - system_tokens,
        )
        logger.info(
            "Prompt budget: provider=%s model=%s total=%d system=%d history=%d",
            self.llm.active_provider,
            self.llm.active_model,
            context_budget_tokens,
            system_tokens,
            history_budget_tokens,
        )

        messages.extend(
            session.build_context_window(
                max_total_tokens=history_budget_tokens,
                keep_recent_messages=32,
                summary_max_tokens=1_200,
            )
        )

        if not messages or messages[-1].get("content") != message:
            messages.append({"role": "user", "content": message})

        # Agent loop: LLM → tool calls → LLM → ... → final response
        executed_tool_calls = 0
        _provider_tool_log: list[dict] = []  # tools the provider ran server-side
        tool_refusal_retry = 0
        tool_announcement_retry = 0
        _raw_emit = event_fn

        async def _emit(event_type, data):
            if not _raw_emit:
                return
            try:
                result = _raw_emit(event_type, data)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except (TypeError, RuntimeError, OSError) as _exc:
                logger.debug(
                    "Event emit failed for %s (non-fatal)", event_type, exc_info=True
                )

        for iteration in range(self._max_tool_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)
            await _emit("thinking", {"iteration": iteration + 1})

            # Context window protection: trim old tool results before the prompt balloons.
            # Raised from 28_000 → 180_000 per user request "remove all limits".
            # Claude Opus 4.6 has a 200K context window; 180K leaves room for
            # the response + thinking budget without ever truncating tool history
            # on any normal-sized task.
            total_tokens = sum(
                Session.estimate_tokens(m.get("content", "")) for m in messages
            )
            if total_tokens > 180_000:
                logger.warning(
                    "Context size %d tokens exceeds soft limit — trimming old tool results",
                    total_tokens,
                )
                for _i, m in enumerate(messages):
                    if (
                        m.get("role") == "user"
                        and m.get("content", "").startswith("[Tool Result:")
                        and Session.estimate_tokens(m["content"]) > 900
                    ):
                        m["content"] = Session.trim_content_for_context(
                            m["content"],
                            max_tokens=700,
                            head_ratio=0.78,
                        )
                # Recheck
                total_tokens = sum(
                    Session.estimate_tokens(m.get("content", "")) for m in messages
                )
                logger.info("Context size after trim: %d tokens", total_tokens)

            # Refresh the project-root reminder so it rides shotgun with the
            # main system prompt on every iteration. Prevents mid-conversation
            # folder drift in long scaffolding sessions.
            _inject_project_root_reminder(messages, session)

            try:
                # All tools available — bootstrap prompt guides which tools to use
                _tools_this_iter = (
                    self._tool_definitions
                    if iteration < self._max_tool_iterations - 1
                    else None
                )
                # Stream tokens when the provider owns the full tool loop internally.
                # In that case the provider's response is already the final user answer.
                _provider_owns_tool_loop = self.llm.executes_tool_loop_internally
                _stream_this_iter = (
                    stream_fn if (not _tools_this_iter or _provider_owns_tool_loop) else None
                )
                # Phase B: let the active provider annotate messages/tools with
                # its own cache markers before inference (Anthropic cache_control,
                # Gemini cachedContent, etc.). Fail-open — cache failures must
                # never block inference.
                try:
                    _provider = self.llm.active_provider_instance
                    # Messages are dicts on this path; providers that need typed
                    # Messages for cache hints can convert via message_from_dict.
                    _provider.apply_cache_hints(messages, _tools_this_iter)  # type: ignore[arg-type]
                except Exception as _cache_err:  # noqa: BLE001
                    logger.debug("apply_cache_hints failed (non-fatal): %s", _cache_err)
                response = await self.llm.chat(
                    messages=messages,
                    tools=_tools_this_iter,
                    stream_fn=_stream_this_iter,
                )
            except (ConnectionError, TimeoutError, OSError, RuntimeError) as llm_err:
                # Truncate verbose provider errors for cleaner logs
                err_str = str(llm_err)
                if len(err_str) > 300:
                    err_str = err_str[:300] + "..."
                logger.error(
                    "LLM call failed at iteration %d: %s", iteration + 1, err_str
                )
                # Try once more without tools (smaller request)
                try:
                    messages.append(
                        {
                            "role": "user",
                            "content": "[System: Previous LLM call failed. Provide your best response with what you have so far.]",
                        }
                    )
                    response = await self.llm.chat(messages=messages, tools=None)
                except (ConnectionError, TimeoutError, OSError, RuntimeError) as retry_err:
                    error_msg = _llm_error_message(retry_err)
                    await _emit("response", {"content": error_msg, "error": True})
                    await self._record_outcome(
                        user_id=user_id,
                        message=message,
                        response=error_msg,
                        tools_used=_tools_used,
                        tool_errors=_tool_errors,
                        iterations=iteration + 1,
                        start_time=start_time,
                        provider=self.llm._provider_name,
                        model="",
                        usage={},
                        drift_score=0.0,
                        drift_regens=0,
                        session_id=sid,
                        success=False,
                        error=f"llm_failure: {str(retry_err)[:200]}",
                    )
                    return error_msg

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            response_blocks = response.get("content_blocks")  # None for non-Anthropic
            structured_path = isinstance(response_blocks, list) and bool(response_blocks)

            # Some providers execute tools server-side — capture for metrics
            provider_tools = response.get("tools_executed", [])
            if provider_tools:
                for mt in provider_tools:
                    _tools_used.append(mt["name"])
                    _provider_tool_log.append(mt)
                    executed_tool_calls += 1
                logger.info(
                    "Provider-executed tools: %s",
                    ", ".join(t["name"] for t in provider_tools),
                )

            # Detect a SHORT tool-announcement ("I'll check the file") emitted as
            # text without an actual tool_call. Anchored regex + short length cap.
            if (
                not tool_calls
                and content
                and len(content.strip()) < 180
                and tool_announcement_retry < 2
                and re.match(
                    r"^\s*(I'll|I will|Let me|I'm going to|I am going to|Going to)\s+"
                    r"(use|run|call|execute|invoke|grep|find)\b",
                    content,
                    re.I,
                )
            ):
                tool_announcement_retry += 1
                logger.warning(
                    "LLM returned tool announcement without tool_calls (attempt %d/2): %s",
                    tool_announcement_retry,
                    content[:100],
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "[System: You mentioned using a tool but didn't actually call it. "
                        "You MUST use the available tools to complete the action. "
                        "If you cannot call the tool, provide your final answer directly.]",
                    }
                )
                continue

            # Detect tool refusal / harness-limitation responses when tools are available.
            if (
                not tool_calls
                and content
                and tool_refusal_retry < 1
                and iteration < self._max_tool_iterations - 1
                and self._tool_definitions
            ):
                lower_content = content.lower()
                tool_refusal_re = re.compile(
                    r"(\bexposes no\b|\bno\b\s+(desktop|browser|tool|tools|notes?)\s+"
                    r"(control|access)\b|\bdoesn['’]?t\s+expose\b|\bdoesn['’]?t\s+"
                    r"support\b|\bno\s+.*\bbridge\b|\bno\s+.*\bharness\b|"
                    r"\b(can(?:not|'t)|unable|not able to|do not|don't have)\b"
                    r"[\s\S]{0,80}\b(use|access|control|open|click|save|run)\b"
                    r"[\s\S]{0,40}\b(tool|browser|desktop|notes?)\b)",
                    re.I,
                )
                hard_refusal = (
                    "desktop automation access" in lower_content
                    or "no desktop automation" in lower_content
                    or "no desktop control" in lower_content
                    or "no browser control" in lower_content
                    or "no desktop access" in lower_content
                    or ("apple notes" in lower_content and "can't" in lower_content)
                )
                if hard_refusal or tool_refusal_re.search(content):
                    tool_refusal_retry += 1
                    logger.warning(
                        "LLM refused tools despite availability — retrying with tool enforcement."
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: The listed tools are available in this session. "
                                "Do not claim tool unavailability. If a listed tool can satisfy "
                                "the request, use it now. "
                                "If no tool is needed, answer directly without mentioning tool limits.]"
                            ),
                        }
                    )
                    continue

            # No tool calls → we're done
            if not tool_calls:
                if not content.strip() and _tools_used:
                    # LLM returned empty/marker content after tool use —
                    # ask it to summarize what happened.
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[System: Tools have finished executing. "
                                "Summarize what you did and the results for the user. "
                                "Be concise and helpful.]"
                            ),
                        }
                    )
                    response = await self.llm.chat(
                        messages=messages,
                        tools=None,
                    )
                    content = response.get("content", "Done.")

                (
                    final_content,
                    drift_assessment,
                    regen_count,
                ) = await self._apply_persona_drift_guard(
                    user_message=message,
                    messages=messages,
                    initial_content=content,
                    tools_used=executed_tool_calls,
                )

                # Meta-cognition: evaluate response quality
                response_eval = _evaluate_response(
                    message,
                    final_content,
                    _tools_used,
                    _tool_results,
                )
                if response_eval.issues:
                    logger.info(
                        "Response evaluation: confidence=%.2f issues=%s",
                        response_eval.confidence,
                        ",".join(response_eval.issues),
                    )

                # No simulated-streaming fallback here. Providers that natively
                # stream call stream_fn themselves during .chat(); providers
                # that don't stream will render inside the CLI's glass response
                # panel when the "response" event fires. This keeps both paths
                # clean and non-duplicating.
                elapsed = time.time() - start_time
                comp_tokens = response.get("usage", {}).get("completion_tokens", 0)
                provider_used = response.get("provider_used", self.llm._provider_name)
                logger.info(
                    "Response generated in %.1fs (%d iterations, %d tokens, "
                    "drift_score=%.2f, drift_regens=%d, confidence=%.2f)",
                    elapsed,
                    iteration + 1,
                    comp_tokens,
                    drift_assessment.score,
                    regen_count,
                    response_eval.confidence,
                )
                await _emit(
                    "response",
                    {
                        "content": final_content,
                        "tokens": comp_tokens,
                        "tools_used": executed_tool_calls,
                        "iterations": iteration + 1,
                        "elapsed_s": round(elapsed, 2),
                        "provider": provider_used,
                        "usage": response.get("usage", {}),
                        "cost_usd": response.get("cost_usd", 0),
                        "streamed": _stream_this_iter is not None,
                    },
                )
                try:
                    self._transcript.append_message(sid, "assistant", final_content)
                except OSError:
                    logger.debug("Transcript write failed (non-fatal)", exc_info=True)
                await self._record_outcome(
                    user_id=user_id,
                    message=message,
                    response=final_content,
                    tools_used=_tools_used,
                    tool_errors=_tool_errors,
                    iterations=iteration + 1,
                    start_time=start_time,
                    provider=provider_used,
                    model=response.get("model_used", ""),
                    usage=response.get("usage", {}),
                    drift_score=drift_assessment.score,
                    drift_regens=regen_count,
                    session_id=sid,
                    success=not bool(_tool_errors),
                )
                # Post-turn: store conversation in unified memory
                await self._store_turn_memory(
                    user_id,
                    message,
                    final_content,
                    _tools_used,
                    sid,
                    provider_used,
                )
                # Post-turn: append to DECISIONS.md (per-turn reasoning trace)
                await self._append_decision_journal(
                    message=message,
                    response=final_content,
                    tools_used=_tools_used,
                    tool_errors=_tool_errors,
                    iterations=iteration + 1,
                    start_time=start_time,
                    usage=response.get("usage", {}),
                    session_id=sid,
                )
                # Post-turn: identity reflection (non-blocking)
                await self._maybe_identity_reflect()
                return final_content

            # ── Execute tool calls (parallel when possible) ───────────
            # Two paths diverge here depending on provider capability:
            #
            #   structured_path (Anthropic): the provider
            #     returned content_blocks (text + thinking + tool_use).
            #     We append ONE assistant turn preserving those blocks
            #     verbatim, then ONE user turn with all tool_result
            #     blocks keyed by tool_use_id. Thinking signatures and
            #     tool_use ids survive the round trip — Anthropic's
            #     interleaved-thinking + tool_use contract is honored.
            #
            #   flat path (OpenAI/Gemini/others): we fall back to the
            #     legacy per-tool text stubs [Calling tool: X] /
            #     [Tool Result: X]. Those providers don't validate tool
            #     linkage, so plain strings still work.
            #
            # Stream user-visible text BEFORE tool execution either way.
            _is_tool_announcement = bool(
                content
                and re.match(
                    r"^(I'll|I will|Let me|I'm going to)\s+(use|run|try|call|execute)\b",
                    content,
                    re.I,
                )
            )
            if content and stream_fn and not _is_tool_announcement:
                try:
                    _stream_result = stream_fn(
                        content + "\n\n\U0001f527 *Using tools...*\n"
                    )
                    if asyncio.iscoroutine(_stream_result) or asyncio.isfuture(_stream_result):
                        await _stream_result
                except (TypeError, RuntimeError, OSError):
                    pass

            # Flat path only: append text-only assistant turn.
            # Structured path already includes the text inside
            # response_blocks so appending here would duplicate it.
            if content and not structured_path:
                messages.append({"role": "assistant", "content": content})

            independent, dependent = _classify_tool_dependencies(tool_calls)
            _tool_timeout = self._tool_timeout

            async def _run_single_tool(tc, _timeout: float = _tool_timeout):
                """Execute a single tool call and return (tc, result, elapsed_ms).

                ``_timeout`` is captured via default-arg so the closure is
                unambiguously bound to the current value (silences B023).
                """
                t_name = tc["name"]
                t_args = tc["arguments"]
                t0 = time.time()
                try:
                    res = await asyncio.wait_for(
                        self.tools.execute(t_name, t_args, confirm_fn=confirm_fn),
                        timeout=_timeout,
                    )
                except asyncio.TimeoutError:
                    elapsed = (time.time() - t0) * 1000
                    logger.error("Tool %s timed out after %ds", t_name, _timeout)
                    return (
                        tc,
                        f"[Error: Tool '{t_name}' timed out after {_timeout}s]",
                        elapsed,
                    )
                elapsed = (time.time() - t0) * 1000
                return tc, res, elapsed

            # Run independent tools in parallel
            if len(independent) > 1:
                logger.info(
                    "Executing %d independent tools in parallel", len(independent)
                )
                parallel_results = await asyncio.gather(
                    *[_run_single_tool(tc) for tc in independent],
                    return_exceptions=True,
                )
            elif independent:
                parallel_results = [await _run_single_tool(independent[0])]
            else:
                parallel_results = []

            # Collect (tc, result_str) pairs in deterministic order so
            # structured and flat paths can both emit them at the end.
            collected: list[tuple[dict, str]] = []
            drift_messages: list[str] = []

            # Process parallel results
            for pr in parallel_results:
                if isinstance(pr, Exception):
                    logger.error("Parallel tool execution error: %s", pr)
                    err_msg = f"[Error: Parallel tool execution failed: {str(pr)[:200]}]"
                    _tool_errors.append(err_msg)
                    # Fabricate a dummy TC so the structured path can still
                    # emit a tool_result block with is_error=True. The id
                    # is empty — repair_tool_flow will normalize if needed.
                    collected.append(
                        ({"id": "", "name": "parallel_error", "arguments": {}}, err_msg)
                    )
                    continue
                tc, result, tool_elapsed_ms = pr
                tool_name = tc["name"]
                tool_args = tc["arguments"]

                await _emit("tool_start", {"name": tool_name, "args": tool_args})
                executed_tool_calls += 1
                _tools_used.append(tool_name)
                _tool_history.append((tool_name, tool_args))
                _tool_results.append(str(result)[:500])
                _drift = _check_project_drift(session, tool_name, tool_args, self.config.home_dir)
                _maybe_pin_project_root(
                    session, tool_name, tool_args, self.config.home_dir,
                    ctx_memory=getattr(getattr(self.tools, "_tool_ctx", None), "memory", None),
                )
                if _drift:
                    logger.warning(
                        "Project drift: %s(%s) outside pinned root",
                        tool_name, _redact_tool_args(tool_args, max_len=120),
                    )
                    _tool_errors.append(_drift[:200])
                    drift_messages.append(_drift)
                if isinstance(result, str) and (
                    result.startswith("[Error") or result.startswith("Error:")
                ):
                    _tool_errors.append(f"{tool_name}: {result[:200]}")

                await _emit(
                    "tool_end",
                    {
                        "name": tool_name,
                        "result": str(result)[:200],
                        "duration_ms": round(tool_elapsed_ms, 1),
                    },
                )
                try:
                    self._transcript.append_tool_call(sid, tool_name, tool_args)
                    self._transcript.append_tool_result(
                        sid, tool_name, str(result)[:2000]
                    )
                except OSError:
                    logger.debug("Transcript write failed (non-fatal)", exc_info=True)

                collected.append((tc, str(result)))

            # Run dependent tools sequentially
            for tc in dependent:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                logger.info(
                    "Tool call (sequential): %s(%s)",
                    tool_name,
                    _redact_tool_args(tool_args, max_len=100),
                )
                await _emit("tool_start", {"name": tool_name, "args": tool_args})
                try:
                    self._transcript.append_tool_call(sid, tool_name, tool_args)
                except OSError:
                    logger.debug("Transcript write failed (non-fatal)", exc_info=True)

                tool_t0 = time.time()
                try:
                    result = await asyncio.wait_for(
                        self.tools.execute(tool_name, tool_args, confirm_fn=confirm_fn),
                        timeout=_tool_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Tool %s timed out after %ds", tool_name, _tool_timeout
                    )
                    result = (
                        f"[Error: Tool '{tool_name}' timed out after {_tool_timeout}s]"
                    )
                tool_elapsed_ms = (time.time() - tool_t0) * 1000
                executed_tool_calls += 1
                _tools_used.append(tool_name)
                _tool_history.append((tool_name, tool_args))
                _tool_results.append(str(result)[:500])
                _drift = _check_project_drift(session, tool_name, tool_args, self.config.home_dir)
                _maybe_pin_project_root(
                    session, tool_name, tool_args, self.config.home_dir,
                    ctx_memory=getattr(getattr(self.tools, "_tool_ctx", None), "memory", None),
                )
                if _drift:
                    logger.warning(
                        "Project drift: %s(%s) outside pinned root",
                        tool_name, _redact_tool_args(tool_args, max_len=120),
                    )
                    _tool_errors.append(_drift[:200])
                    drift_messages.append(_drift)
                if isinstance(result, str) and (
                    result.startswith("[Error") or result.startswith("Error:")
                ):
                    _tool_errors.append(f"{tool_name}: {result[:200]}")

                await _emit(
                    "tool_end",
                    {
                        "name": tool_name,
                        "result": str(result)[:200],
                        "duration_ms": round(tool_elapsed_ms, 1),
                    },
                )
                try:
                    self._transcript.append_tool_result(
                        sid, tool_name, str(result)[:2000],
                    )
                except OSError:
                    logger.debug("Transcript write failed (non-fatal)", exc_info=True)

                collected.append((tc, str(result)))

            # ── Emit the round-trip in whichever format the provider needs ──
            def _cap_result(s: str) -> str:
                if len(s) > 15000:
                    return s[:12000] + "\n...[truncated]...\n" + s[-2000:]
                return s

            # Phase A refactor (2026-04-21): provider-delegated tool-turn
            # serialization. Each provider builds its OWN wire-format turns
            # via the typed LLMProvider.append_*_turn methods — no more
            # structured/flat fork here, no more "[Calling tool: X]" text
            # stubs for non-Anthropic providers.
            #
            # Default is ON. Set PREDACORE_NEW_TOOL_TURNS=0 to force the
            # legacy path (emergency rollback only — the legacy path is
            # scheduled for removal).
            _use_new_tool_turns = os.getenv("PREDACORE_NEW_TOOL_TURNS", "1") != "0"

            if _use_new_tool_turns:
                from .llm_providers import (
                    AssistantResponse,
                    ToolCallRef,
                    ToolResultRef,
                    message_from_dict,
                    message_to_dict,
                )

                # Build typed response — the provider's chat() still returns a
                # dict so we normalize here at the core↔provider boundary.
                typed_tool_calls = [
                    ToolCallRef(
                        id=tc.get("id", "") or f"{tc.get('name', 'tool')}_{idx}",
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}) or tc.get("input", {}) or {},
                    )
                    for idx, tc in enumerate(tool_calls)
                ]
                # Carry per-provider round-trip state through provider_extras
                # so each provider's append_*_turn can replay it verbatim.
                #   - Anthropic: content_blocks (thinking signatures, tool_use ids)
                #   - Gemini:    content_parts  (thoughtSignature on functionCall)
                _extras: dict[str, Any] = {}
                if response_blocks:
                    _extras["content_blocks"] = response_blocks
                _content_parts = response.get("content_parts")
                if isinstance(_content_parts, list) and _content_parts:
                    _extras["content_parts"] = _content_parts
                typed_response = AssistantResponse(
                    content=content or "",
                    tool_calls=typed_tool_calls,
                    provider_extras=_extras,
                )
                # Build typed tool results from (tc, result_str) pairs
                typed_results: list[ToolResultRef] = []
                for idx, (tc, result_str) in enumerate(collected):
                    capped = _cap_result(result_str)
                    typed_results.append(
                        ToolResultRef(
                            call_id=tc.get("id", "") or f"{tc.get('name', 'tool')}_{idx}",
                            name=tc.get("name", ""),
                            result=capped,
                            is_error=capped.startswith("[Error") or capped.startswith("Error:"),
                        )
                    )

                # Provider owns serialization — no more fork, no more poison stubs
                provider = self.llm.active_provider_instance
                typed_msgs = [message_from_dict(m) for m in messages]
                provider.append_assistant_turn(typed_msgs, typed_response)
                provider.append_tool_results_turn(typed_msgs, typed_results)
                # Re-sync dict-based messages list in place
                messages[:] = [message_to_dict(m) for m in typed_msgs]
            elif structured_path:
                # LEGACY structured path (Anthropic only) — kept for one-week
                # rollback window. Identical to pre-Phase-A behavior.
                tool_result_blocks: list[dict] = []
                summary_lines: list[str] = []
                for tc, result_str in collected:
                    capped = _cap_result(result_str)
                    tool_use_id = tc.get("id") or ""
                    block: dict = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": capped,
                    }
                    if capped.startswith("[Error") or capped.startswith("Error:"):
                        block["is_error"] = True
                    tool_result_blocks.append(block)
                    summary_lines.append(f"[Tool Result: {tc['name']}]\n{capped}")

                messages.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "content_blocks": response_blocks,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "\n\n".join(summary_lines) or "[Tool results]",
                        "content_blocks": tool_result_blocks,
                    }
                )
            else:
                # LEGACY flat path — kept for one-week rollback window.
                # This is the PATH THAT CAUSED CONTEXT POISONING via literal
                # "[Calling tool: X]" text stubs. It will be deleted in A.8
                # once PREDACORE_NEW_TOOL_TURNS=1 is the default.
                for tc, result_str in collected:
                    capped = _cap_result(result_str)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"[Calling tool: {tc['name']}]",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[Tool Result: {tc['name']}]\n{capped}",
                        }
                    )

            # Project-drift warnings get their own user messages so the
            # model sees them even when the tool_result turn is structured.
            for drift_msg in drift_messages:
                messages.append({"role": "user", "content": drift_msg})

            # Meta-cognition: check if we're stuck
            help_msg = _should_ask_for_help(
                _tool_errors,
                iteration + 1,
                self._max_tool_iterations,
                _tool_history,
            )
            if help_msg:
                _loop_warnings = getattr(self, "_loop_warnings", 0) + 1
                self._loop_warnings = _loop_warnings
                logger.warning(
                    "Meta-cognition suggests asking for help (%d): %s",
                    _loop_warnings,
                    help_msg[:120],
                )
                if _loop_warnings >= 4:
                    # Fourth loop detection — force break, stop burning tokens.
                    # Raised from 2 so legitimate bulk work (scaffolding many files,
                    # running several build commands) isn't killed mid-task.
                    logger.warning("Loop detected 4 times — force-breaking agent loop")
                    messages.append(
                        {
                            "role": "user",
                            "content": "[System: Tool loop detected. Stop calling tools and give your best response NOW.]",
                        }
                    )
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": f"[System: {help_msg} Provide your best answer or ask the user for clarification.]",
                    }
                )

        # Exhausted iterations — force a response
        logger.warning("Agent loop hit max iterations (%d)", self._max_tool_iterations)
        messages.append(
            {
                "role": "user",
                "content": "[System: You've used too many tools. Please provide your final response now.]",
            }
        )
        final = await self.llm.chat(messages=messages, tools=None)
        final_content = final.get(
            "content", "I completed the task but ran into complexity limits."
        )
        final_content, _, _ = await self._apply_persona_drift_guard(
            user_message=message,
            messages=messages,
            initial_content=final_content,
            tools_used=executed_tool_calls,
        )
        try:
            self._transcript.append_message(sid, "assistant", final_content)
        except OSError:
            logger.debug("Transcript write failed (non-fatal)", exc_info=True)
        await self._record_outcome(
            user_id=user_id,
            message=message,
            response=final_content,
            tools_used=_tools_used,
            tool_errors=_tool_errors,
            iterations=self._max_tool_iterations,
            start_time=start_time,
            provider=final.get("provider_used", self.llm._provider_name),
            model=final.get("model_used", ""),
            usage=final.get("usage", {}),
            drift_score=0.0,
            drift_regens=0,
            session_id=sid,
            success=False,
            error="max_iterations_exhausted",
        )
        return final_content

    # ── Outcome tracking ────────────────────────────────────────────

    async def _record_outcome(
        self,
        user_id: str,
        message: str,
        response: str,
        tools_used: list[str],
        tool_errors: list[str],
        iterations: int,
        start_time: float,
        provider: str,
        model: str,
        usage: dict[str, Any],
        drift_score: float,
        drift_regens: int,
        session_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a TaskOutcome (non-fatal on failure)."""
        if self._outcome_store is None:
            return
        try:
            outcome = TaskOutcome(
                user_id=user_id,
                user_message=message,
                response_summary=str(response)[:500],
                tools_used=tools_used,
                tool_errors=tool_errors,
                provider_used=provider,
                model_used=model,
                latency_ms=(time.time() - start_time) * 1000,
                token_count_prompt=usage.get("prompt_tokens", 0),
                token_count_completion=usage.get("completion_tokens", 0),
                iterations=iterations,
                success=success,
                error=error,
                persona_drift_score=drift_score,
                persona_drift_regens=drift_regens,
                session_id=session_id,
            )
            await self._outcome_store.record(outcome)
        except (OSError, ValueError, RuntimeError) as _exc:
            logger.debug("Outcome recording failed (non-fatal)", exc_info=True)

    # ── Public helpers ───────────────────────────────────────────────

    def get_tool_list(self) -> list[str]:
        """Get list of available tool names."""
        return [t["name"] for t in self._tool_definitions]

    def get_status(self) -> dict[str, Any]:
        """Get core status info."""
        persistent_memory_items = None
        try:
            if self.tools._memory_service is not None:
                persistent_memory_items = int(self.tools._memory_service._count_all())
        except (ValueError, TypeError):
            persistent_memory_items = None
        openclaw_status: dict[str, Any] = {
            "enabled": bool(self.config.launch.enable_openclaw_bridge),
        }
        marketplace_skills = getattr(self.tools, "_openclaw_marketplace_skills", None)
        if marketplace_skills:
            openclaw_status["marketplace_skills"] = len(marketplace_skills)
        skills_dir = getattr(self.tools, "_openclaw_skills_dir", None)
        if skills_dir is not None:
            openclaw_status["skills_dir"] = str(skills_dir)
        if self.tools._openclaw_runtime:
            openclaw_status.update(
                {
                    "ledger_path": str(self.tools._openclaw_runtime.ledger_path),
                    "idempotency_db_path": str(
                        self.tools._openclaw_runtime.idempotency_db_path
                    ),
                    "kill_switch_active": self.tools._openclaw_runtime.kill_switch_active(),
                }
            )
        return {
            "provider": getattr(self.llm, "_provider_name", self.config.llm.provider),
            "model": self.config.llm.model or "auto",
            "trust_level": self.config.security.trust_level,
            "tools": self.get_tool_list(),
            "tools_count": len(self._tool_definitions),
            "memory_items": len(self.tools._memory),
            "session_memory_items": len(self.tools._memory),
            "persistent_memory_items": persistent_memory_items,
            "max_tool_iterations": self._max_tool_iterations,
            "persona_drift_guard": {
                "enabled": self._persona_drift_guard_enabled,
                "threshold": self._persona_drift_threshold,
                "max_regens": self._persona_drift_max_regens,
            },
            "openclaw": openclaw_status,
        }
