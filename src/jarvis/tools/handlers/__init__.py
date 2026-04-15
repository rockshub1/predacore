"""
JARVIS Tool Handlers — split into focused modules.

Phase 6.2 refactoring: the 2,400-line handlers.py monolith was split into
domain-specific modules. This __init__.py re-exports everything so existing
code that does ``from .handlers import HANDLER_MAP`` continues to work
without changes.

Phase 6.5: HANDLER_MAP keys use ToolName enums. Since ToolName is (str, Enum),
lookups with plain strings still work: ``HANDLER_MAP["read_file"]`` ✓

Module layout:
  _context.py          — ToolContext dataclass + ToolError hierarchy + shared utilities
  file_ops.py          — read_file, write_file, list_directory
  shell.py             — run_command, python_exec, execute_code
  web.py               — web_search, web_scrape, deep_search, semantic_search
  memory.py            — memory_store, memory_recall
  voice.py             — speak, voice_note
  desktop.py           — desktop_control, screen_vision, android_control
  agent.py             — multi_agent, strategic_plan, openclaw_delegate
  creative.py          — image_gen, pdf_reader, diagram
  git.py               — git_context, git_diff_summary, git_commit_suggest, git_find_files, git_semantic_search
  marketplace.py       — marketplace_list_skills, marketplace_install_skill, marketplace_invoke_skill
  identity.py          — identity_read, identity_update, journal_append, bootstrap_complete
  cron.py              — cron_task
  pipeline_handler.py  — tool_pipeline (chain multiple tools)
  stats.py             — tool_stats (debug dashboard)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ── Re-export enums (so callers can do: from .handlers import ToolName) ──
from ..enums import ToolName, ToolStatus

# ── Re-export ToolContext and ToolError (the primary types used by executor/dispatcher) ──
from ._context import ToolContext, ToolError, ToolErrorKind
from ._context import missing_param, invalid_param, subsystem_unavailable, resource_not_found, blocked

# ── File operations ──
from .file_ops import (
    handle_list_directory,
    handle_read_file,
    handle_write_file,
)

# ── Shell / code execution ──
from .shell import (
    handle_execute_code,
    handle_python_exec,
    handle_run_command,
)

# ── Web ──
from .web import (
    handle_deep_search,
    handle_semantic_search,
    handle_web_scrape,
    handle_web_search,
)

# ── Memory ──
from .memory import (
    handle_memory_recall,
    handle_memory_store,
)

# ── Voice ──
from .voice import (
    handle_speak,
    handle_voice_note,
)

# ── Desktop / mobile / browser ──
from .desktop import (
    handle_android_control,
    handle_browser_control,
    handle_desktop_control,
    handle_screen_vision,
)

# ── Agent / planning ──
from .agent import (
    handle_multi_agent,
    handle_openclaw_delegate,
    handle_strategic_plan,
)

# ── Creative ──
from .creative import (
    handle_diagram,
    handle_image_gen,
    handle_pdf_reader,
)

# ── Git ──
from .git import (
    handle_git_commit_suggest,
    handle_git_context,
    handle_git_diff_summary,
    handle_git_find_files,
    handle_git_semantic_search,
)

# ── Marketplace ──
from .marketplace import (
    handle_marketplace_install_skill,
    handle_marketplace_invoke_skill,
    handle_marketplace_list_skills,
)

# ── Identity ──
from .identity import (
    handle_bootstrap_complete,
    handle_identity_read,
    handle_identity_update,
    handle_journal_append,
)

# ── Cron ──
from .cron import handle_cron_task

# ── Pipeline ──
from .pipeline_handler import handle_tool_pipeline

# ── Hive Mind ──
from .hivemind import (
    handle_hivemind_status,
    handle_hivemind_sync,
    handle_skill_evolve,
    handle_skill_scan,
    handle_skill_endorse,
)

# ── Stats / Debug ──
from .stats import handle_tool_stats

# ---------------------------------------------------------------------------
# Unified handler map — ToolName enum keys → async handler functions
#
# Since ToolName is (str, Enum), plain string lookups still work:
#   HANDLER_MAP["read_file"]          ✓
#   HANDLER_MAP[ToolName.READ_FILE]   ✓
# ---------------------------------------------------------------------------

HANDLER_MAP: dict[str, Callable] = {
    # File / shell
    ToolName.READ_FILE: handle_read_file,
    ToolName.WRITE_FILE: handle_write_file,
    ToolName.RUN_COMMAND: handle_run_command,
    ToolName.LIST_DIRECTORY: handle_list_directory,
    # Web
    ToolName.WEB_SEARCH: handle_web_search,
    ToolName.WEB_SCRAPE: handle_web_scrape,
    # Code execution
    ToolName.PYTHON_EXEC: handle_python_exec,
    ToolName.EXECUTE_CODE: handle_execute_code,
    # Memory
    ToolName.MEMORY_STORE: handle_memory_store,
    ToolName.MEMORY_RECALL: handle_memory_recall,
    # Voice
    ToolName.SPEAK: handle_speak,
    ToolName.VOICE_NOTE: handle_voice_note,
    # Desktop
    ToolName.DESKTOP_CONTROL: handle_desktop_control,
    ToolName.SCREEN_VISION: handle_screen_vision,
    ToolName.ANDROID_CONTROL: handle_android_control,
    ToolName.BROWSER_CONTROL: handle_browser_control,
    # Multi-agent / planning
    ToolName.MULTI_AGENT: handle_multi_agent,
    ToolName.STRATEGIC_PLAN: handle_strategic_plan,
    # Marketplace
    ToolName.MARKETPLACE_LIST: handle_marketplace_list_skills,
    ToolName.MARKETPLACE_INSTALL: handle_marketplace_install_skill,
    ToolName.MARKETPLACE_INVOKE: handle_marketplace_invoke_skill,
    # OpenClaw
    ToolName.OPENCLAW_DELEGATE: handle_openclaw_delegate,
    # Git
    ToolName.GIT_CONTEXT: handle_git_context,
    ToolName.GIT_DIFF_SUMMARY: handle_git_diff_summary,
    ToolName.GIT_COMMIT_SUGGEST: handle_git_commit_suggest,
    ToolName.GIT_FIND_FILES: handle_git_find_files,
    ToolName.GIT_SEMANTIC_SEARCH: handle_git_semantic_search,
    # Research / search
    ToolName.DEEP_SEARCH: handle_deep_search,
    ToolName.SEMANTIC_SEARCH: handle_semantic_search,
    # Creative
    ToolName.IMAGE_GEN: handle_image_gen,
    ToolName.PDF_READER: handle_pdf_reader,
    ToolName.DIAGRAM: handle_diagram,
    # Cron
    ToolName.CRON_TASK: handle_cron_task,
    # Identity
    ToolName.IDENTITY_READ: handle_identity_read,
    ToolName.IDENTITY_UPDATE: handle_identity_update,
    ToolName.JOURNAL_APPEND: handle_journal_append,
    ToolName.BOOTSTRAP_COMPLETE: handle_bootstrap_complete,
    # Pipeline (tool chaining)
    ToolName.TOOL_PIPELINE: handle_tool_pipeline,
    # Hive Mind
    ToolName.HIVEMIND_STATUS: handle_hivemind_status,
    ToolName.HIVEMIND_SYNC: handle_hivemind_sync,
    ToolName.SKILL_EVOLVE: handle_skill_evolve,
    ToolName.SKILL_SCAN: handle_skill_scan,
    ToolName.SKILL_ENDORSE: handle_skill_endorse,
    # Debug / stats
    ToolName.TOOL_STATS: handle_tool_stats,
}

__all__ = [
    "HANDLER_MAP",
    "ToolContext",
    # Enums
    "ToolName", "ToolStatus",
    # Error types
    "ToolError", "ToolErrorKind",
    "missing_param", "invalid_param", "subsystem_unavailable",
    "resource_not_found", "blocked",
    # All handler functions
    "handle_read_file", "handle_write_file", "handle_list_directory",
    "handle_run_command", "handle_python_exec", "handle_execute_code",
    "handle_web_search", "handle_web_scrape",
    "handle_deep_search", "handle_semantic_search",
    "handle_memory_store", "handle_memory_recall",
    "handle_speak", "handle_voice_note",
    "handle_desktop_control", "handle_screen_vision", "handle_android_control",
    "handle_multi_agent", "handle_strategic_plan", "handle_openclaw_delegate",
    "handle_image_gen", "handle_pdf_reader", "handle_diagram",
    "handle_git_context", "handle_git_diff_summary", "handle_git_commit_suggest",
    "handle_git_find_files", "handle_git_semantic_search",
    "handle_marketplace_list_skills", "handle_marketplace_install_skill",
    "handle_marketplace_invoke_skill",
    "handle_identity_read", "handle_identity_update",
    "handle_journal_append", "handle_bootstrap_complete",
    "handle_cron_task",
    "handle_tool_pipeline",
    "handle_tool_stats",
]
