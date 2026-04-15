"""
JARVIS Tool Enums — Eliminates magic strings across the tool/operator layers.

Central registry of tool names, action names, and status codes so that
typos become compile-time errors instead of silent runtime bugs.

Operator-specific enums (DesktopAction, AndroidAction, VisionAction,
ScreenshotQuality) are defined in ``operators/enums.py`` (the authoritative
source) and re-exported here for convenience.

Usage:
    from jarvis.tools.enums import ToolName, ToolStatus
    from jarvis.tools.enums import DesktopAction  # re-exported from operators

    if tool_name == ToolName.READ_FILE:
        ...

    return {"status": ToolStatus.OK, ...}
"""
from __future__ import annotations

from enum import Enum

# ── Re-export operator enums (single source of truth lives in operators/) ──
from jarvis.operators.enums import (  # noqa: F401
    AndroidAction,
    DesktopAction,
    NATIVE_CAPABLE_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    SMART_ACTIONS,
    ScreenshotQuality,
    VisionAction,
)


# ---------------------------------------------------------------------------
# Tool Names — used in HANDLER_MAP, dispatcher, cache, circuit breaker
# ---------------------------------------------------------------------------


class ToolName(str, Enum):
    """Canonical tool names used in the dispatcher pipeline."""

    # File / shell
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    LIST_DIRECTORY = "list_directory"
    RUN_COMMAND = "run_command"

    # Code execution
    PYTHON_EXEC = "python_exec"
    EXECUTE_CODE = "execute_code"

    # Web
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    DEEP_SEARCH = "deep_search"
    SEMANTIC_SEARCH = "semantic_search"
    BROWSER_CONTROL = "browser_control"

    # Memory
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"

    # Voice
    SPEAK = "speak"
    VOICE_NOTE = "voice_note"

    # Desktop / mobile
    DESKTOP_CONTROL = "desktop_control"
    SCREEN_VISION = "screen_vision"
    ANDROID_CONTROL = "android_control"

    # Agent / planning
    MULTI_AGENT = "multi_agent"
    STRATEGIC_PLAN = "strategic_plan"
    OPENCLAW_DELEGATE = "openclaw_delegate"

    # Marketplace
    MARKETPLACE_LIST = "marketplace_list_skills"
    MARKETPLACE_INSTALL = "marketplace_install_skill"
    MARKETPLACE_INVOKE = "marketplace_invoke_skill"

    # Git
    GIT_CONTEXT = "git_context"
    GIT_DIFF_SUMMARY = "git_diff_summary"
    GIT_COMMIT_SUGGEST = "git_commit_suggest"
    GIT_FIND_FILES = "git_find_files"
    GIT_SEMANTIC_SEARCH = "git_semantic_search"

    # Creative
    IMAGE_GEN = "image_gen"
    PDF_READER = "pdf_reader"
    DIAGRAM = "diagram"

    # Identity
    IDENTITY_READ = "identity_read"
    IDENTITY_UPDATE = "identity_update"
    JOURNAL_APPEND = "journal_append"
    BOOTSTRAP_COMPLETE = "bootstrap_complete"

    # Cron
    CRON_TASK = "cron_task"

    # Pipeline
    TOOL_PIPELINE = "tool_pipeline"

    # Hive Mind / Skill Evolution
    HIVEMIND_STATUS = "hivemind_status"
    HIVEMIND_SYNC = "hivemind_sync"
    SKILL_EVOLVE = "skill_evolve"
    SKILL_SCAN = "skill_scan"
    SKILL_ENDORSE = "skill_endorse"

    # Debug / stats
    TOOL_STATS = "tool_stats"


# ---------------------------------------------------------------------------
# Tool Status — returned by handlers and resilience layer
# ---------------------------------------------------------------------------


class ToolStatus(str, Enum):
    """Standard status codes for tool execution results."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CACHED = "cached"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"
    DENIED = "denied"
    UNKNOWN_TOOL = "unknown_tool"


# ---------------------------------------------------------------------------
# Write tools — tools that mutate state (used for cache invalidation)
# ---------------------------------------------------------------------------

WRITE_TOOLS: frozenset[str] = frozenset({
    ToolName.WRITE_FILE,
    ToolName.RUN_COMMAND,
    ToolName.PYTHON_EXEC,
    ToolName.EXECUTE_CODE,
})

# ---------------------------------------------------------------------------
# Read-only tools — safe for auto-approval in normal trust mode
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS: frozenset[str] = frozenset({
    ToolName.READ_FILE,
    ToolName.LIST_DIRECTORY,
    ToolName.WEB_SEARCH,
    ToolName.WEB_SCRAPE,
    ToolName.SEMANTIC_SEARCH,
    ToolName.DEEP_SEARCH,
    ToolName.GIT_CONTEXT,
    ToolName.GIT_DIFF_SUMMARY,
    ToolName.GIT_FIND_FILES,
    ToolName.GIT_SEMANTIC_SEARCH,
    ToolName.MEMORY_RECALL,
    ToolName.IDENTITY_READ,
    ToolName.SCREEN_VISION,
    ToolName.PDF_READER,
    ToolName.MARKETPLACE_LIST,
    ToolName.TOOL_STATS,
})
