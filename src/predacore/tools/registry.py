"""
PredaCore Tool Registry — Central registry for all tool definitions with metadata.

Extracted from core.py monolith. Each tool gets enriched metadata for
intelligent orchestration: category, cost estimation, parallelizability,
and timeout defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool Metadata ────────────────────────────────────────────────────


@dataclass
class ToolDefinition:
    """A tool with its schema + operational metadata."""

    name: str
    description: str
    parameters: dict[str, Any]
    category: str = "general"  # file_ops, shell, web, memory, agent, desktop, voice, marketplace
    cost_estimate: str = "free"  # free, low, medium, high
    parallelizable: bool = True  # can run alongside other tools
    requires_confirmation: bool = False  # needs user approval before execution
    timeout_default: int = 30  # seconds

    def to_openai_dict(self) -> dict[str, Any]:
        """Return OpenAI function-calling format dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ── Tool Registry ────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for all PredaCore tool definitions."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool
        cat_list = self._categories.setdefault(tool.category, [])
        if tool.name not in cat_list:
            cat_list.append(tool.name)

    def register_raw(
        self,
        raw: dict[str, Any],
        category: str = "general",
        cost_estimate: str = "free",
        parallelizable: bool = True,
        requires_confirmation: bool = False,
        timeout_default: int = 30,
    ) -> None:
        """Register from a raw dict (legacy format) with schema validation."""
        if not isinstance(raw, dict):
            raise TypeError(f"Tool definition must be a dict, got {type(raw).__name__}")
        if "name" not in raw or not raw["name"]:
            raise ValueError("Tool definition must have a non-empty 'name' field")
        if not raw.get("description"):
            logger.warning("Tool '%s' registered without description", raw.get("name"))
        params = raw.get("parameters", {})
        if params and not isinstance(params, dict):
            raise TypeError(
                f"Tool '{raw['name']}' parameters must be a dict, got {type(params).__name__}"
            )
        td = ToolDefinition(
            name=raw["name"],
            description=raw.get("description", ""),
            parameters=raw.get("parameters", {}),
            category=category,
            cost_estimate=cost_estimate,
            parallelizable=parallelizable,
            requires_confirmation=requires_confirmation,
            timeout_default=timeout_default,
        )
        self.register(td)

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool with the given name exists in the registry."""
        return name in self._tools

    def list_all(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> list[ToolDefinition]:
        """Return tool definitions filtered by category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_categories(self) -> list[str]:
        """Return sorted list of unique tool categories."""
        return list(self._categories.keys())

    def get_all_definitions(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [t.to_openai_dict() for t in self._tools.values()]

    def get_parallelizable(self) -> list[str]:
        """Return names of tools that can run in parallel."""
        return [n for n, t in self._tools.items() if t.parallelizable]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── Built-in Tool Definitions ────────────────────────────────────────
# Each tool: (raw_dict, category, cost_estimate, parallelizable, requires_confirmation, timeout)

BUILTIN_TOOLS_RAW = [
    # ── File Operations ──
    (
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
        "file_ops",
        "free",
        True,
        False,
        15,  # optimal: disk I/O headroom for large files
    ),
    (
        {
            "name": "write_file",
            "description": "Write content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
        "file_ops",
        "free",
        False,  # mutating — matches core.py _MUTATING_TOOLS
        True,
        15,  # optimal: large writes + dir creation
    ),
    (
        {
            "name": "list_directory",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {"type": "boolean", "description": "List recursively"},
                },
                "required": ["path"],
            },
        },
        "file_ops",
        "free",
        True,
        False,
        15,  # optimal: deep dirs with many entries
    ),
    # ── Shell Execution ──
    (
        {
            "name": "run_command",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Command timeout in seconds; <=0 disables timeout in YOLO mode",
                    },
                },
                "required": ["command"],
            },
        },
        "shell",
        "free",
        False,  # mutating — matches core.py _MUTATING_TOOLS
        True,
        180,  # optimal: builds, compilation, pip installs
    ),
    # ── Web ──
    (
        {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
        "web",
        "low",
        True,
        False,
        20,  # optimal: DDG is fast — fail-fast is better
    ),
    (
        {
            "name": "web_scrape",
            "description": "Fetch and read the content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"}
                },
                "required": ["url"],
            },
        },
        "web",
        "low",
        True,
        False,
        45,  # optimal: heavy pages need time
    ),
    # ── Browser Bridge (instant DOM access to your running browser) ──
    (
        {
            "name": "browser_control",
            "description": (
                "Control YOUR running Chrome browser via DOM — 100x faster than screenshots. "
                "ALWAYS prefer this over desktop_control or screen_vision for any web page. "
                "Uses your existing Chrome cookies and logins. Safari is not supported. "
                "\n\nMANDATORY PATTERN — read before you act: "
                "(1) navigate to URL → (2) read_text or get_page_tree to see what's there → "
                "(3) click with text='Button Label' (NOT guessed CSS selectors) → "
                "(4) read_text again to confirm what changed → repeat. "
                "NEVER guess selectors like #submit-button. Click by visible text, or call "
                "get_page_tree first to find real selectors from the DOM. "
                "\n\nActions: get_page_tree, click, type, read_text, navigate, scroll, evaluate_js, get_url. "
                "Wait: wait_for_element, wait_for_text, wait_for_url. History: back, forward, reload. "
                "Forms: set_checkbox, select_option, upload_file. Keyboard: press_key, key_combo, hover. "
                "Cookies: get_cookies, set_cookie, delete_cookies. Storage: get_storage, set_storage, clear_storage. "
                "Tabs: list_tabs, new_tab, close_tab. Visual: screenshot (full_page), print_pdf. "
                "Data: extract_tables, find_in_page, get_page_links, get_page_images. "
                "Media: get_media_info, media_play, media_pause, media_seek, media_set_speed, media_set_volume, media_toggle_mute, media_fullscreen. "
                "Captions: get_captions, enable_captions, get_transcript."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": [
                            "connect", "get_page_tree", "click", "type", "type_keys", "read_text",
                            "navigate", "scroll", "evaluate_js", "get_url",
                            "wait_for_element", "wait_for_text", "wait_for_url",
                            "back", "forward", "reload",
                            "hover", "press_key", "key_combo",
                            "get_cookies", "set_cookie", "delete_cookies",
                            "get_storage", "set_storage", "clear_storage",
                            "set_checkbox", "select_option", "upload_file",
                            "screenshot", "print_pdf",
                            "list_tabs", "new_tab", "close_tab",
                            "extract_tables", "find_in_page",
                            "drag_and_drop", "set_download_path",
                            "list_frames", "evaluate_in_frame", "click_in_frame",
                            "clipboard_read", "clipboard_write",
                            "start_network_log", "get_network_log", "clear_network_log",
                            "set_geolocation", "clear_geolocation",
                            "set_dialog_handler", "get_last_dialog",
                            "set_auth_credentials", "clear_auth_credentials",
                            "element_screenshot", "download_image", "capture_canvas",
                            "image_to_base64", "get_background_images", "get_svgs",
                            "get_page_links", "get_page_images",
                            "get_media_info", "media_play", "media_pause", "media_seek",
                            "media_set_speed", "media_set_volume", "media_toggle_mute",
                            "media_fullscreen", "get_captions", "enable_captions",
                            "get_transcript",
                        ],
                    },
                    "url": {"type": "string", "description": "URL for navigate/new_tab"},
                    "selector": {"type": "string", "description": "CSS selector to target"},
                    "text": {"type": "string", "description": "Text to find element by, or text to type"},
                    "value": {"type": "string", "description": "Value to set (input field, cookie, storage, select option)"},
                    "role": {"type": "string", "description": "ARIA role to find element by"},
                    "code": {"type": "string", "description": "JavaScript code (for evaluate_js)"},
                    "key": {"type": "string", "description": "Key name for press_key (Enter, Tab, Escape, ArrowDown, etc.)"},
                    "keys": {"type": "array", "items": {"type": "string"}, "description": "Key combo list for key_combo, e.g. [\"ctrl\",\"a\"]"},
                    "modifiers": {"type": "array", "items": {"type": "string"}, "description": "Modifier keys: ctrl, alt, shift, meta/cmd"},
                    "name": {"type": "string", "description": "Cookie name (for set_cookie, delete_cookies)"},
                    "domain": {"type": "string", "description": "Cookie domain filter"},
                    "storage_type": {"type": "string", "enum": ["local", "session"], "description": "localStorage or sessionStorage (default: local)"},
                    "checked": {"type": "boolean", "description": "Desired checkbox state (for set_checkbox)"},
                    "label": {"type": "string", "description": "Option label text (for select_option)"},
                    "file_paths": {"type": "array", "items": {"type": "string"}, "description": "Local file paths for upload_file"},
                    "full_page": {"type": "boolean", "description": "Capture full page (for screenshot)"},
                    "path": {"type": "string", "description": "Output file path (for screenshot/print_pdf)"},
                    "target_id": {"type": "string", "description": "Tab target ID (for close_tab)"},
                    "pattern": {"type": "string", "description": "URL substring to wait for (wait_for_url)"},
                    "query": {"type": "string", "description": "Search text (for find_in_page)"},
                    "source": {"type": "string", "description": "Source CSS selector (for drag_and_drop)"},
                    "target_selector": {"type": "string", "description": "Target CSS selector (for drag_and_drop)"},
                    "frame_id": {"type": "string", "description": "Frame ID from list_frames (for iframe actions)"},
                    "latitude": {"type": "number", "description": "GPS latitude (for set_geolocation)"},
                    "longitude": {"type": "number", "description": "GPS longitude (for set_geolocation)"},
                    "accuracy": {"type": "number", "description": "GPS accuracy in meters (default: 100)"},
                    "username": {"type": "string", "description": "Username for HTTP auth (set_auth_credentials)"},
                    "password": {"type": "string", "description": "Password for HTTP auth (set_auth_credentials)"},
                    "limit": {"type": "integer", "description": "Max entries to return (for get_network_log)"},
                    "timeout": {"type": "number", "description": "Timeout in seconds for wait actions (default: 10)"},
                    "seconds": {"type": "number", "description": "Seek position (media_seek)"},
                    "level": {"type": "integer", "description": "Volume 0-100 (media_set_volume)"},
                    "speed": {"type": "number", "description": "Playback rate 0.25-4.0 (media_set_speed)"},
                    "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
                    "amount": {"type": "integer", "description": "Scroll amount"},
                    "browser": {"type": "string", "enum": ["auto", "chrome"], "description": "Browser to connect to (default: auto = Chrome CDP)"},
                },
                "required": ["action"],
            },
        },
        "desktop",
        "low",
        True,
        False,
        15,
    ),
    # ── Code Execution ──
    (
        {
            "name": "python_exec",
            "description": "Execute Python code in a sandboxed environment and return the result. Uses Docker isolation when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default: 30)",
                    },
                    "network_allowed": {
                        "type": "boolean",
                        "description": "Allow network access (default: false)",
                    },
                },
                "required": ["code"],
            },
        },
        "code_exec",
        "free",
        False,  # mutating — matches core.py _MUTATING_TOOLS
        True,
        60,  # optimal: complex computations in-process
    ),
    (
        {
            "name": "execute_code",
            "description": "Execute code in any supported language via Docker sandbox. Supports: python, node, go, ruby, php, r, julia, rust, java, kotlin, c, cpp, typescript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to execute"},
                    "runtime": {
                        "type": "string",
                        "description": "Language runtime (python|node|go|ruby|php|r|julia|rust|java|kotlin|c|cpp|typescript)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default: 30)",
                    },
                    "network_allowed": {
                        "type": "boolean",
                        "description": "Allow network access (default: false)",
                    },
                },
                "required": ["code", "runtime"],
            },
        },
        "code_exec",
        "free",
        False,  # mutating — matches core.py _MUTATING_TOOLS
        True,
        120,  # optimal: Docker startup + code execution
    ),
    # ── Memory ──
    (
        {
            "name": "memory_store",
            "description": "Store information in long-term memory for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key/topic"},
                    "content": {
                        "type": "string",
                        "description": "Information to remember",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional memory owner namespace",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "fact|conversation|task|preference|context|skill|entity|knowledge",
                    },
                    "importance": {
                        "type": "string",
                        "description": "low|medium|high|critical",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Memory scope: global (default), team, scratch",
                    },
                    "team_id": {
                        "type": "string",
                        "description": "Required for team/scratch memory scopes",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Optional agent identifier for scratch/team writes",
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Optional time-to-live for temporary memories",
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Optional absolute expiration timestamp (ISO-8601)",
                    },
                },
                "required": ["key", "content"],
            },
        },
        "memory",
        "free",
        True,
        False,
        10,
    ),
    (
        {
            "name": "memory_recall",
            "description": "Recall information from long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional memory owner namespace",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results (default: 5, max: 20)",
                    },
                    "search_mode": {
                        "type": "string",
                        "description": "Search strategy: semantic (default), entity, keyword",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Memory scope to search: global (default), team, scratch",
                    },
                    "team_id": {
                        "type": "string",
                        "description": "Required for team/scratch memory recall",
                    },
                },
                "required": ["query"],
            },
        },
        "memory",
        "free",
        True,
        False,
        15,  # optimal: vector search on large stores
    ),
    # ── Agent & Planning ──
    (
        {
            "name": "multi_agent",
            "description": (
                "Run a task using multi-agent collaboration. Patterns: "
                "fan_out (parallel, collect all), pipeline (sequential chain), "
                "consensus (vote on answer), supervisor (worker + reviewer)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt to send to agents",
                    },
                    "pattern": {
                        "type": "string",
                        "enum": ["fan_out", "pipeline", "consensus", "supervisor"],
                        "description": "Collaboration pattern",
                    },
                    "num_agents": {
                        "type": "integer",
                        "description": "Number of agents (default 3, max 5)",
                    },
                    "agent_roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional role names for each agent (e.g. ['researcher', 'coder', 'reviewer'])",
                    },
                    "use_daf": {
                        "type": "boolean",
                        "description": "Use DAF process fabric for true process-level parallelism (default false)",
                    },
                    "team_id": {
                        "type": "string",
                        "description": "Optional shared team-memory namespace. Omit to create a new team.",
                    },
                    "memory_tokens": {
                        "type": "integer",
                        "description": "Shared global-memory retrieval budget per run (default 1200)",
                    },
                    "team_ttl_hours": {
                        "type": "integer",
                        "description": "How long shared team memory should persist before expiring (default 72)",
                    },
                    "max_runtime_seconds": {
                        "type": "integer",
                        "description": (
                            "Optional wall-clock kill switch for the whole team run "
                            "(e.g. 1800 = 30 min). Enforced — if the team hasn't "
                            "finished by then, it's cancelled. Good for long autonomous "
                            "work where you want a hard ceiling."
                        ),
                    },
                    "max_iterations_per_agent": {
                        "type": "integer",
                        "description": (
                            "Advisory hint — how many tool-call rounds each sub-agent "
                            "should use at most. Surfaced to the agents in their prompt. "
                            "The core-level max_tool_iterations cap still applies."
                        ),
                    },
                    "max_cost_usd": {
                        "type": "number",
                        "description": (
                            "Advisory hint — estimated spend budget (USD). Surfaced to "
                            "the agents in their prompt so they self-pace. For hard "
                            "enforcement across the daemon, use trust_level + "
                            "task_timeout_seconds in config."
                        ),
                    },
                },
                "required": ["prompt", "pattern"],
            },
        },
        "agent",
        "high",
        False,
        False,
        300,  # optimal: full sub-agent LLM conversations
    ),
    (
        {
            "name": "strategic_plan",
            "description": (
                "Generate a strategic plan for a complex goal using MCTS-guided planning. "
                "Returns a multi-step plan with scored alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The goal to plan for"},
                    "context": {
                        "type": "object",
                        "description": "Optional context (e.g. constraints, preferences)",
                    },
                },
                "required": ["goal"],
            },
        },
        "agent",
        "medium",
        False,
        False,
        90,  # optimal: LLM-backed planning needs headroom
    ),
    # ── Desktop & Voice ──
    (
        {
            "name": "desktop_control",
            "description": (
                "FASTEST native macOS app control — PyObjC, 1-5ms per action, no AppleScript. "
                "For native apps only — for web pages, use browser_control instead (100x faster than this). "
                "\n\nSpeed ladder (always try top first): "
                "ax_click (2ms) > ax_set_value (2ms) > press_key (5ms) > type_text (5ms) > "
                "mouse_click (5ms) > screenshot (50ms). "
                "Pattern: quick_scan with screen_vision to read UI (1ms AX tree), then ax_click or "
                "ax_set_value by label. For text input: ax_set_value FIRST (instant injection), "
                "type_text as fallback. Only fall back to mouse_click if AX fails. "
                "You have native OS-level APIs — CGEvent text injection, direct AX tree access, "
                "AX performAction clicks. Use them. You are much faster than screenshot-based RPA. "
                "\n\nCore: open/focus apps, keyboard, mouse (click/drag/scroll), screenshots, clipboard, "
                "window management, AX tree/click/set_value, macros. "
                "System: list_apps, force_quit_app, get_system_info (battery/wifi), dark_mode, "
                "spotlight_search, open_file, get_finder_selection, tile_windows, "
                "screen_record, get_running_processes. "
                "PREFER ax_click over mouse_click, ax_set_value over type_text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "run_macro",
                            "open_app", "focus_app", "open_url", "frontmost_app",
                            "type_text", "press_key",
                            "mouse_move", "mouse_click", "mouse_double_click",
                            "mouse_right_click", "mouse_scroll", "get_mouse_position", "mouse_drag",
                            "screenshot",
                            "ax_query", "ax_click", "ax_set_value", "ax_request_access",
                            "clipboard_read", "clipboard_write",
                            "list_windows", "move_window", "resize_window", "minimize_window",
                            "list_monitors", "move_to_monitor",
                            "click_menu", "set_volume", "get_volume", "set_brightness", "get_brightness",
                            "smart_type", "smart_run_command", "smart_create_note",
                            "list_apps", "force_quit_app", "get_system_info",
                            "toggle_dark_mode", "get_dark_mode",
                            "toggle_dnd", "spotlight_search", "open_file",
                            "get_finder_selection", "screen_record_start", "screen_record_stop",
                            "tile_windows", "app_switch", "get_running_processes",
                            "health_check", "sleep",
                        ],
                        "description": "Desktop action to execute",
                    },
                    "steps": {"type": "array", "items": {"type": "object"}},
                    "app_name": {"type": "string"},
                    "url": {"type": "string"},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                    "key_code": {"type": "integer"},
                    "modifiers": {"type": "array", "items": {"type": "string"}},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "amount": {"type": "integer"},
                    "seconds": {"type": "number"},
                    "include_base64": {"type": "boolean"},
                    "timeout_seconds": {"type": "number"},
                    "target": {"type": "string"},
                    "match": {"type": "object"},
                    "value": {},
                    "max_depth": {"type": "integer"},
                    "max_children": {"type": "integer"},
                    "request_access": {"type": "boolean"},
                    "prompt": {"type": "boolean"},
                    "window_title": {"type": "string", "description": "Window title for window management actions"},
                    "start_x": {"type": "integer", "description": "Drag start X"},
                    "start_y": {"type": "integer", "description": "Drag start Y"},
                    "end_x": {"type": "integer", "description": "Drag end X"},
                    "end_y": {"type": "integer", "description": "Drag end Y"},
                    "duration_seconds": {"type": "number", "description": "Duration for drag/animation"},
                    "query": {"type": "string", "description": "Search query for spotlight_search"},
                    "path": {"type": "string", "description": "File path for open_file/screenshot/screen_record"},
                    "file_path": {"type": "string", "description": "Alias for path"},
                    "bundle_id": {"type": "string", "description": "App bundle ID for force_quit_app"},
                    "left_app": {"type": "string", "description": "Left app name for tile_windows"},
                    "right_app": {"type": "string", "description": "Right app name for tile_windows"},
                    "sort_by": {"type": "string", "enum": ["cpu", "memory"], "description": "Sort order for get_running_processes"},
                    "limit": {"type": "integer", "description": "Max results for search/process listing"},
                    "level": {"type": "integer", "description": "Volume 0-100 or brightness 0-100"},
                    "menu": {"type": "string", "description": "Menu name for click_menu (e.g. File)"},
                    "item": {"type": "string", "description": "Menu item for click_menu (e.g. Save)"},
                    "monitor": {"type": "integer", "description": "Monitor index for move_to_monitor"},
                    "command": {"type": "string", "description": "Shell command for smart_run_command"},
                    "others": {"type": "boolean", "description": "Minimize other apps' windows (minimize_window)"},
                },
                "required": ["action"],
            },
        },
        "desktop",
        "free",
        False,
        True,
        45,  # optimal: screenshot + mouse operations
    ),
    # ── Screen Vision ──
    (
        {
            "name": "screen_vision",
            "description": (
                "Screen understanding + autonomous UI automation for native macOS apps. "
                "NOT for web pages — use browser_control for anything in Chrome. "
                "\n\nMANDATORY SPEED LADDER (always try top first): "
                "quick_scan (1ms, AX tree, no screenshot) → find_and_click (10ms) → "
                "type_into (15ms) → scan_with_ocr (200ms) → scan_with_vision (3-5s, uses LLM — LAST RESORT). "
                "ALWAYS try quick_scan + find_and_click FIRST. Only escalate to scan_with_vision "
                "if the AX tree is empty or the element genuinely can't be found any other way. "
                "For reading screen text: read_screen_text or quick_scan, NEVER scan_with_vision. "
                "For complex multi-step tasks: execute_task runs a vision→act loop automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "quick_scan",
                            "scan",
                            "scan_with_vision",
                            "scan_with_ocr",
                            "find_and_click",
                            "type_into",
                            "read_screen_text",
                            "has_changed",
                            "wait_for_change",
                            "execute_task",
                            "focused_element",
                            "ocr_status",
                        ],
                        "description": (
                            "Vision action to perform. Use 'execute_task' for multi-step "
                            "autonomous UI tasks (e.g. 'open Photopea and create a thumbnail'). "
                            "It runs a screenshot→understand→act loop with up to max_steps iterations."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": "Element label to find (for click/type actions)",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (for type_into)",
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional AX role filter (e.g. AXButton, AXTextField)",
                    },
                    "task": {
                        "type": "string",
                        "description": "Natural language task for vision analysis or autonomous execution",
                    },
                    "include_screenshot": {
                        "type": "boolean",
                        "description": "Include screenshot in scan (default: false)",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds for wait_for_change (default: 30)",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Max steps for execute_task (default: 20)",
                    },
                },
                "required": ["action"],
            },
        },
        "desktop",
        "low",
        False,
        True,
        90,  # optimal: screenshot + vision LLM round-trip
    ),
    # ── Android Control ──
    (
        {
            "name": "android_control",
            "description": (
                "Full Android device control via ADB — 50 actions. "
                "\n\nSMART FLOW for every step: "
                "(1) ensure_in_app → verify you're in the right app (auto-relaunches if wrong). "
                "(2) wait_for_stable_ui → wait until animations/loading finish. "
                "(3) FIND the target element: search_in_app(query) is fastest (uses the app's own "
                "search bar — beats scrolling), then smart_tap(text=...) or smart_tap(resource_id=...), "
                "then scroll_to_element(text=...) as last resort. "
                "(4) ACT: smart_tap() auto-verifies the UI changed; smart_type() auto-clears + verifies. "
                "(5) wait_for_text('expected') to confirm the action worked. "
                "\n\nALWAYS try Android intents / deep links FIRST — they bypass the entire UI: "
                "`shell('am start -a android.intent.action.VIEW -d \"market://search?q=...\" -p com.android.vending')`. "
                "The `-p` flag forces a specific app (skips 'Open with' chooser). "
                "Common deep links: market://search?q=app, market://details?id=com.pkg, "
                "tel:+91..., mailto:user@x, geo:lat,lng, sms:+91..., https://wa.me/91..., "
                "content://settings/system. "
                "Package flags: -p com.android.vending (Play Store), -p com.android.chrome, "
                "-p com.google.android.apps.maps, -p com.google.android.gm (Gmail), -p com.whatsapp, "
                "-p com.google.android.dialer. Or use open_settings(page='wifi') for settings. "
                "\n\nNEVER use raw coordinates from a prior ui_dump — the UI changes between dump and tap. "
                "ALWAYS use smart_tap / smart_type instead of raw tap / type_text on dynamic apps. "
                "For Chrome on the phone: connect_chrome_on_device → then browser_control (69 actions, "
                "5ms each, full DOM access — same as desktop Chrome). "
                "For reading full page content: scroll_and_collect_all (scrolls + accumulates all text). "
                "For custom-drawn UI (games, canvas): screenshot_and_ocr (Vision framework OCR). "
                "\n\nTouch: tap, long_press, double_tap, swipe, pinch. "
                "Input: type_text, press_key, input_keycombo. "
                "UI: ui_dump, find_element, find_and_tap, find_and_type, wait_for_element, scroll_to_element. "
                "Apps: launch_app, stop_app, current_app, list_packages, install_apk, uninstall_app, clear_app_data, list_running_apps. "
                "Device: screenshot, screen_record, screen_size, screen_state, wake, sleep_device, rotate_screen, set_brightness. "
                "System: get_device_info, get_battery_info, get_wifi_info, get_notifications, clear_notifications, get_logcat, open_settings. "
                "Network: toggle_wifi, toggle_bluetooth, toggle_airplane, open_url. "
                "Clipboard: get_clipboard, set_clipboard. Files: push_file, pull_file. "
                "Advanced: shell, send_broadcast, run_macro."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "health_check", "ui_dump",
                            "tap", "long_press", "double_tap", "swipe", "pinch",
                            "type_text", "press_key", "input_keycombo",
                            "screenshot", "screen_record",
                            "launch_app", "stop_app", "current_app",
                            "list_packages", "install_apk", "uninstall_app",
                            "clear_app_data", "list_running_apps",
                            "find_element", "find_and_tap", "find_and_type",
                            "wait_for_element", "scroll_to_element",
                            "screen_size", "screen_state", "wake", "sleep_device",
                            "rotate_screen", "set_brightness",
                            "get_device_info", "get_battery_info", "get_wifi_info",
                            "get_notifications", "clear_notifications",
                            "get_logcat", "open_settings", "open_url",
                            "toggle_wifi", "toggle_bluetooth", "toggle_airplane", "toggle_flashlight",
                            "get_clipboard", "set_clipboard",
                            "push_file", "pull_file",
                            "shell", "send_broadcast", "run_macro",
                        ],
                        "description": "Android action to perform",
                    },
                    "x": {"type": "integer", "description": "X coordinate for tap/swipe"},
                    "y": {"type": "integer", "description": "Y coordinate for tap/swipe"},
                    "x1": {"type": "integer", "description": "Start X for swipe"},
                    "y1": {"type": "integer", "description": "Start Y for swipe"},
                    "x2": {"type": "integer", "description": "End X for swipe"},
                    "y2": {"type": "integer", "description": "End Y for swipe"},
                    "text": {"type": "string", "description": "Text to type or search for"},
                    "key": {"type": "string", "description": "Key name (home, back, enter, etc.)"},
                    "package": {"type": "string", "description": "Android package name"},
                    "resource_id": {"type": "string", "description": "Element resource ID to find"},
                    "content_desc": {"type": "string", "description": "Element content description to find"},
                    "local_path": {"type": "string", "description": "Local file path for push/pull"},
                    "remote_path": {"type": "string", "description": "Device file path for push/pull"},
                    "duration_seconds": {"type": "integer", "description": "Duration for screen recording"},
                    "command": {"type": "string", "description": "Shell command to run"},
                    "url": {"type": "string", "description": "URL to open (open_url)"},
                    "orientation": {"type": "string", "enum": ["portrait", "landscape", "auto"], "description": "Screen orientation"},
                    "level": {"type": "integer", "description": "Brightness 0-255 (set_brightness)"},
                    "page": {"type": "string", "description": "Settings page (wifi/bluetooth/display/battery/apps/developer/about)"},
                    "intent_action": {"type": "string", "description": "Android intent action for send_broadcast"},
                    "extras": {"type": "object", "description": "Intent extras for send_broadcast"},
                    "enable": {"type": "boolean", "description": "Enable/disable for toggle actions"},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Scroll direction"},
                    "max_scrolls": {"type": "integer", "description": "Max scroll attempts for scroll_to_element"},
                    "lines": {"type": "integer", "description": "Number of logcat lines (default 50)"},
                    "tag": {"type": "string", "description": "Logcat tag filter"},
                    "keys": {"type": "array", "items": {"type": "string"}, "description": "Key names for input_keycombo"},
                    "keep_data": {"type": "boolean", "description": "Keep data on uninstall"},
                    "duration_ms": {"type": "integer", "description": "Duration for long press/swipe"},
                    "device_serial": {"type": "string", "description": "ADB device serial (optional)"},
                    "steps": {
                        "type": "array",
                        "description": "Macro steps (each with action + params)",
                        "items": {"type": "object"},
                    },
                    "timeout": {"type": "number", "description": "Timeout in seconds"},
                },
                "required": ["action"],
            },
        },
        "device",
        "low",
        False,
        True,
        90,  # optimal: ADB over USB/WiFi can be slow
    ),
    (
        {
            "name": "speak",
            "description": "Convert text to speech and play it aloud using the system TTS engine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speak aloud"},
                },
                "required": ["text"],
            },
        },
        "voice",
        "free",
        False,
        False,
        30,  # optimal: long text TTS
    ),
    # ── Git Integration ──
    (
        {
            "name": "git_context",
            "description": "Get comprehensive git repository context: branch, status, recent commits, remotes, and stash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (default: current)",
                    },
                    "log_count": {
                        "type": "integer",
                        "description": "Number of recent commits to show (default: 10)",
                    },
                },
            },
        },
        "git",
        "free",
        True,
        False,
        20,  # optimal: large repos
    ),
    (
        {
            "name": "git_diff_summary",
            "description": "Get a structured summary of git changes: files changed, insertions/deletions, and optionally the raw diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Git ref to diff against (default: HEAD)",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show only staged changes (default: false)",
                    },
                    "include_diff": {
                        "type": "boolean",
                        "description": "Include raw diff text (default: false)",
                    },
                },
            },
        },
        "git",
        "free",
        True,
        False,
        30,  # optimal: large diffs
    ),
    (
        {
            "name": "git_commit_suggest",
            "description": "Analyze staged changes and suggest a conventional commit message with type and scope.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        "git",
        "free",
        True,
        False,
        30,  # optimal: LLM-assisted commit messages
    ),
    (
        {
            "name": "git_find_files",
            "description": "Search for files in the git index matching a pattern. Faster than filesystem search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File name or glob pattern to search for",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 50)",
                    },
                },
                "required": ["pattern"],
            },
        },
        "git",
        "free",
        True,
        False,
        20,  # optimal: large repos with many files
    ),
    # ── Git Semantic Code Search ──
    (
        {
            "name": "git_semantic_search",
            "description": (
                "Semantic search over git-tracked source files using natural language. "
                "Finds files by meaning, not just name. Example: 'rate limiting logic' "
                "finds rate_limiter.py. Uses embeddings + BM25 hybrid scoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'where is the authentication middleware')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return (default: 10)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional file pattern filter (e.g. '*.py', 'services/*')",
                    },
                    "rebuild": {
                        "type": "boolean",
                        "description": "Force rebuild the index (default: false)",
                    },
                },
                "required": ["query"],
            },
        },
        "git",
        "low",
        True,
        False,
        30,
    ),
    # ── Deep Search (Perplexity-style) ──
    (
        {
            "name": "deep_search",
            "description": (
                "Deep research tool: searches multiple sources, reads top results, "
                "and synthesizes a comprehensive answer with citations. "
                "Use for complex questions that need multiple web sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question or topic to investigate",
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Max web pages to read (default: 5, max: 10)",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional focus area: technical, news, academic, general",
                    },
                },
                "required": ["query"],
            },
        },
        "web",
        "medium",
        False,
        False,
        180,  # optimal: search + scrape 3 URLs + synthesize
    ),
    # ── Semantic Search (hybrid BM25 + vector) ──
    (
        {
            "name": "semantic_search",
            "description": (
                "Hybrid semantic + keyword search across local files and memory. "
                "Combines BM25 keyword scoring with embedding similarity for "
                "best-of-both-worlds retrieval. Searches codebase, docs, and memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Search scope: memory, files, all (default: all)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (for files scope)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', '*.md')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return (default: 10, max: 30)",
                    },
                },
                "required": ["query"],
            },
        },
        "memory",
        "low",
        True,
        False,
        45,  # optimal: BM25 across 200 files
    ),
    # ── Image Generation ──
    (
        {
            "name": "image_gen",
            "description": (
                "Generate images from text descriptions using DALL-E 3 or Stable Diffusion. "
                "Returns the image path. Requires OPENAI_API_KEY for DALL-E."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Image description / generation prompt",
                    },
                    "size": {
                        "type": "string",
                        "description": "Image size: 1024x1024, 1792x1024, 1024x1792 (default: 1024x1024)",
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality: standard or hd (default: standard)",
                    },
                    "style": {
                        "type": "string",
                        "description": "Style: vivid or natural (default: vivid)",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save the image",
                    },
                },
                "required": ["prompt"],
            },
        },
        "web",
        "high",
        True,
        False,
        90,  # optimal: DALL-E 3 generation time
    ),
    # ── PDF Reader ──
    (
        {
            "name": "pdf_reader",
            "description": (
                "Read and extract text from PDF files. Supports page selection, "
                "text extraction, and basic summarization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the PDF file",
                    },
                    "pages": {
                        "type": "string",
                        "description": "Page range to read: 'all', '1-5', '3,7,10' (default: all)",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Extraction mode: text, summary, metadata (default: text)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default: 50000)",
                    },
                },
                "required": ["path"],
            },
        },
        "file_ops",
        "free",
        True,
        False,
        60,  # optimal: large PDFs up to 200MB
    ),
    # ── Voice Note ──
    (
        {
            "name": "voice_note",
            "description": (
                "Record a voice note from the microphone and transcribe it to text. "
                "Can also transcribe existing audio files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["record", "transcribe"],
                        "description": "Action: record (mic→text) or transcribe (file→text)",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Recording duration in seconds (default: 10, max: 120)",
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file (for transcribe action)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (default: en)",
                    },
                },
                "required": ["action"],
            },
        },
        "voice",
        "low",
        False,
        False,
        120,  # optimal: recording + Whisper transcription
    ),
    # ── Diagram Generator ──
    (
        {
            "name": "diagram",
            "description": (
                "Generate diagrams from text descriptions using Mermaid syntax. "
                "Supports flowcharts, sequence diagrams, class diagrams, ER diagrams, "
                "Gantt charts, and more. Returns SVG/PNG path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Mermaid diagram code (e.g. 'graph TD; A-->B')",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: svg or png (default: svg)",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save the diagram",
                    },
                    "theme": {
                        "type": "string",
                        "description": "Mermaid theme: default, dark, forest, neutral",
                    },
                },
                "required": ["code"],
            },
        },
        "file_ops",
        "free",
        True,
        False,
        45,  # optimal: mmdc rendering
    ),
    # ── Cron/Scheduled Task ──
    (
        {
            "name": "cron_task",
            "description": (
                "Create, list, or cancel scheduled recurring tasks. "
                "Tasks run at specified intervals and can execute tools or send messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "cancel", "status"],
                        "description": "Cron action to perform",
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Name for the scheduled task",
                    },
                    "interval_minutes": {
                        "type": "integer",
                        "description": "Run interval in minutes (min: 1, max: 1440)",
                    },
                    "command": {
                        "type": "string",
                        "description": "Task to execute (tool name or natural language instruction)",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (for cancel/status actions)",
                    },
                    "max_runs": {
                        "type": "integer",
                        "description": "Max number of executions (0 = unlimited, default: 0)",
                    },
                },
                "required": ["action"],
            },
        },
        "agent",
        "low",
        False,
        False,
        15,
    ),
    # ── Identity System ──
    (
        {
            "name": "identity_read",
            "description": (
                "Read an identity file from the agent's self-evolving identity workspace. "
                "Use this to review your own identity, personality, user profile, journal, "
                "crystallized beliefs, per-turn decision trace, or long-term evolution log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "enum": [
                            "IDENTITY",
                            "SOUL",
                            "USER",
                            "JOURNAL",
                            "TOOLS",
                            "MEMORY",
                            "HEARTBEAT",
                            "REFLECTION",
                            "BELIEFS",
                            "DECISIONS",
                            "EVOLUTION",
                        ],
                        "description": "Which identity file to read",
                    },
                },
                "required": ["file"],
            },
        },
        "identity",
        "free",
        True,
        False,
        5,
    ),
    (
        {
            "name": "identity_update",
            "description": (
                "Write or update an identity file in the agent's workspace. "
                "Use this to evolve your personality (SOUL.md), update your identity "
                "(IDENTITY.md), record what you've learned about your human (USER.md), "
                "maintain your environment notes (TOOLS.md), curate long-term memory "
                "(MEMORY.md), or tune heartbeat/reflection policy files. "
                "Pass a short 'reason' when updating SOUL/IDENTITY/USER — it gets "
                "logged to EVOLUTION.md alongside the diff. "
                "Cannot modify SOUL_SEED.md, BELIEFS.md, DECISIONS.md, or EVOLUTION.md "
                "(those are maintained by the runtime)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "enum": [
                            "IDENTITY",
                            "SOUL",
                            "USER",
                            "JOURNAL",
                            "TOOLS",
                            "MEMORY",
                            "HEARTBEAT",
                            "REFLECTION",
                        ],
                        "description": "Which identity file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Optional short explanation of why this update is happening. "
                            "Logged to EVOLUTION.md for evolving files (SOUL, IDENTITY, USER, BELIEFS). "
                            "Keep it under ~200 chars — one sentence on the trigger."
                        ),
                    },
                },
                "required": ["file", "content"],
            },
        },
        "identity",
        "low",
        False,
        False,
        10,
    ),
    (
        {
            "name": "journal_append",
            "description": (
                "Append a timestamped entry to your identity journal. "
                "Use this to log growth insights, personality milestones, "
                "relationship developments, or reflections after meaningful conversations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "The journal entry text to append",
                    },
                },
                "required": ["entry"],
            },
        },
        "identity",
        "free",
        True,
        False,
        5,
    ),
    # ── Channel management ──
    (
        {
            "name": "channel_configure",
            "description": (
                "Enable/disable a messaging channel, set its token, or list channel status. "
                "Use this when the user asks to add/remove a channel (e.g. 'set up Telegram', "
                "'disable Discord', 'show channel status'). Writes ~/.predacore/.env with "
                "chmod 600 and updates config.yaml. Daemon restart is required to pick up new "
                "adapters. For brand-new channel TYPES not yet installed, call `channel_install` first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "set_token", "status"],
                        "description": (
                            "add = enable + save token; remove = disable; "
                            "set_token = replace token for already-enabled channel; "
                            "status = list all discovered channels and their state."
                        ),
                    },
                    "channel": {
                        "type": "string",
                        "description": (
                            "Channel name (e.g. 'telegram', 'discord', 'whatsapp', "
                            "'webchat', or any installed third-party channel). "
                            "Not needed for action='status'."
                        ),
                    },
                    "token": {
                        "type": "string",
                        "description": (
                            "Bot token / API key. Required for action='add' on channels "
                            "that need a secret, and always required for 'set_token'."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
        "channels",
        "low",
        False,
        False,
        10,
    ),
    (
        {
            "name": "channel_install",
            "description": (
                "Install a new channel TYPE by `pip install`ing a third-party package "
                "(e.g. predacore-slack, predacore-matrix). After install, the registry "
                "rescans automatically so the new channel becomes usable. Then call "
                "`channel_configure action=add channel=<name> token=<...>` to enable it. "
                "Blocked in paranoid trust mode; prompts for approval in normal; runs "
                "directly in yolo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": (
                            "PyPI package name (e.g. 'predacore-slack'). Optional "
                            "version/extras suffix is allowed (predacore-slack>=1.2, "
                            "predacore-slack[enterprise])."
                        ),
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Pass --upgrade to pip if true. Default false.",
                    },
                },
                "required": ["package"],
            },
        },
        "channels",
        "high",
        False,
        False,
        5,
    ),
    # ── Secrets (LLM keys + channel tokens) ──
    (
        {
            "name": "secret_set",
            "description": (
                "Store an API key or token in ~/.predacore/.env (chmod 600). "
                "Use when the user shares a new LLM provider key "
                "('my Anthropic key is sk-...'), a channel token, or any other secret. "
                "The running process picks up the new value immediately — no restart needed. "
                "Blocked in paranoid trust mode. Accepts recognized provider/channel secrets "
                "plus any *_API_KEY / *_TOKEN / *_SECRET name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Env var name (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                            "GEMINI_API_KEY, GROQ_API_KEY, TELEGRAM_BOT_TOKEN)."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "The secret value (API key, token, etc.).",
                    },
                },
                "required": ["name", "value"],
            },
        },
        "secrets",
        "low",
        False,
        False,
        10,
    ),
    (
        {
            "name": "secret_list",
            "description": (
                "List which known secrets are currently set — names only, values "
                "never returned. Useful for 'am I missing an API key?' checks."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        "secrets",
        "free",
        True,
        False,
        5,
    ),
    # ── MCP (Model Context Protocol — client side) ──
    (
        {
            "name": "mcp_list",
            "description": (
                "Show every configured MCP server, whether it's running, "
                "and the tools each one currently exposes. Use this to answer "
                "'what MCP tools do I have?' or to debug a server that isn't "
                "firing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "mcp",
        "free",
        True,
        False,
        5,
    ),
    (
        {
            "name": "mcp_add",
            "description": (
                "Add a new MCP (Model Context Protocol) server so PredaCore "
                "can use its tools. You can pass just a command (the server's "
                "launch argv) and optional env, or include `install.npm` / "
                "`install.pip` to install the backing package first. "
                "All tools the server advertises become available as "
                "`mcp_<name>_<tool>` in the tool list, immediately and on "
                "future daemon restarts (persisted to config.yaml). Example: "
                "name='filesystem', command=['npx','-y','@modelcontextprotocol/server-filesystem','/home/user']."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short identifier for the server (e.g. 'filesystem', 'github').",
                    },
                    "command": {
                        "description": "Argv list OR shell string that launches the server.",
                    },
                    "env": {
                        "type": "object",
                        "description": "Env vars for the subprocess. ${VAR} is expanded against .env.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the subprocess.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional human-readable note.",
                    },
                    "install": {
                        "type": "object",
                        "description": "Optional installer step: {\"npm\": \"pkg\"} or {\"pip\": \"pkg\"}.",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Write to config.yaml so the server survives restarts (default true).",
                    },
                },
                "required": ["name", "command"],
            },
        },
        "mcp",
        "high",
        False,
        False,
        5,
    ),
    (
        {
            "name": "mcp_remove",
            "description": (
                "Stop an MCP server, unmount its tools, and remove it from "
                "config.yaml. Use when the user says 'disconnect github MCP' "
                "or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Server name to remove."},
                    "persist": {
                        "type": "boolean",
                        "description": "Also delete from config.yaml (default true).",
                    },
                },
                "required": ["name"],
            },
        },
        "mcp",
        "low",
        False,
        False,
        5,
    ),
    (
        {
            "name": "mcp_restart",
            "description": (
                "Tear down and re-spawn an MCP server. Useful after updating "
                "env vars / tokens or when a server hung."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Server name to restart."},
                },
                "required": ["name"],
            },
        },
        "mcp",
        "low",
        False,
        False,
        5,
    ),
    # ── REST API registry ──
    (
        {
            "name": "api_add",
            "description": (
                "Register a REST API so the agent can call it via api_call. "
                "Use when the user wants PredaCore to talk to a service that "
                "doesn't have an MCP server or built-in integration (e.g. Notion, "
                "Linear, an internal company API). Pairs with secret_set for "
                "tokens — reference them with ${VAR} in the auth / header values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Short lowercase identifier (e.g. 'notion', 'linear').",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Scheme + host + optional path prefix (e.g. https://api.notion.com/v1).",
                    },
                    "auth": {
                        "type": "string",
                        "description": (
                            "Value for the Authorization header. Supports ${VAR} expansion. "
                            "Example: 'Bearer ${NOTION_TOKEN}'."
                        ),
                    },
                    "default_headers": {
                        "type": "object",
                        "description": "Extra headers sent on every call (e.g. API version).",
                    },
                    "description": {"type": "string", "description": "Human-readable note."},
                },
                "required": ["service", "base_url"],
            },
        },
        "apis",
        "low",
        False,
        False,
        5,
    ),
    (
        {
            "name": "api_call",
            "description": (
                "Invoke a previously-registered API. Supports GET/POST/PUT/PATCH/"
                "DELETE/HEAD/OPTIONS. Returns status, headers, and a body preview "
                "(response is truncated at ~8KB to keep the prompt lean). In "
                "paranoid trust mode only GET is allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service name registered via api_add.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                        "description": "HTTP method. Defaults to GET.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path appended to the service's base_url (e.g. '/users').",
                    },
                    "params": {"type": "object", "description": "URL query parameters."},
                    "headers": {"type": "object", "description": "Per-call header overrides."},
                    "body": {
                        "type": "string",
                        "description": "Raw request body (text). Use `json` instead for structured data.",
                    },
                    "json": {
                        "description": "JSON-serializable object sent as application/json body.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default 30).",
                    },
                },
                "required": ["service", "path"],
            },
        },
        "apis",
        "medium",
        False,
        False,
        10,
    ),
    (
        {
            "name": "api_list",
            "description": (
                "List every API registered via api_add — service name, base URL, "
                "whether auth is configured, default headers. No secret values returned."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "apis",
        "free",
        True,
        False,
        5,
    ),
    (
        {
            "name": "api_remove",
            "description": "Unregister a REST API from the registry (apis.yaml).",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name to remove."},
                },
                "required": ["service"],
            },
        },
        "apis",
        "low",
        False,
        False,
        5,
    ),
    # ── Pipeline (tool chaining) ──
    (
        {
            "name": "tool_pipeline",
            "description": (
                "Execute a multi-step tool pipeline with variable substitution. "
                "Chain tool calls sequentially or in parallel — output from each step "
                "is available to subsequent steps via {{prev}} and {{step.N}} variables. "
                "Supports conditional steps, error handling (stop/continue), and timeouts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": (
                            "List of pipeline steps. Each step: "
                            '{"tool": "name", "args": {...}, "name": "optional_label", '
                            '"condition": "contains:text|not_empty|not_error", '
                            '"on_error": "stop|continue"}'
                        ),
                        "items": {"type": "object"},
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sequential", "parallel"],
                        "description": "Execution mode (default: sequential)",
                    },
                },
                "required": ["steps"],
            },
        },
        "agent",
        "low",
        False,
        False,
        300,  # pipelines can chain many tools
    ),
    # ── Debug / Stats ──
    (
        {
            "name": "tool_stats",
            "description": (
                "Return a diagnostic dashboard of the PredaCore tool system. "
                "Shows circuit breaker states, cache hit rates, execution history, "
                "per-tool latency stats, and operator telemetry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["all", "circuit_breaker", "cache", "history", "operator_telemetry", "summary"],
                        "description": "Which section to return (default: all)",
                    },
                    "history_count": {
                        "type": "integer",
                        "description": "Number of recent history entries (default: 20, max: 100)",
                    },
                },
            },
        },
        "debug",
        "free",
        True,
        False,
        10,
    ),
]

# ── OpenClaw Bridge Tools ────────────────────────────────────────────

OPENCLAW_BRIDGE_TOOLS_RAW = [
    (
        {
            "name": "openclaw_delegate",
            "description": (
                "Delegate a one-shot autonomous task to the configured OpenClaw bridge "
                "and return structured execution output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task to delegate"},
                    "context": {
                        "type": "object",
                        "description": "Optional structured context",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Delegation mode (default: oneshot)",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional per-call timeout override",
                    },
                    "await_completion": {
                        "type": "boolean",
                        "description": "Poll async task to terminal status (default: true)",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Optional explicit idempotency key",
                    },
                },
                "required": ["task"],
            },
        },
        "agent",
        "high",
        False,
        True,
        300,  # optimal: full remote task execution
    ),
]

# ── Marketplace Tools ────────────────────────────────────────────────

COLLECTIVE_INTELLIGENCE_TOOLS_RAW = [
    (
        {
            "name": "collective_intelligence_status",
            "description": "Show Flame status: local skills, shared pool size, trust distribution, sync stats, scanner metrics.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        "flame",
        "free",
        True,
        False,
        10,
    ),
    (
        {
            "name": "collective_intelligence_sync",
            "description": "Sync with the Flame shared pool — pull new skills, check for recalls, update reputation.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        "flame",
        "free",
        False,
        False,
        30,
    ),
    (
        {
            "name": "skill_evolve",
            "description": "Scan execution history for repeated tool patterns and crystallize them into reusable skills. Shows detected patterns and pending skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'detect' (find patterns), 'crystallize' (create skills from patterns), 'list' (show all evolved skills), 'stats' (evolution statistics)",
                        "enum": ["detect", "crystallize", "list", "stats"],
                    },
                },
            },
        },
        "flame",
        "free",
        False,
        False,
        30,
    ),
    (
        {
            "name": "skill_scan",
            "description": "Run the security scanner on a skill genome. Checks for exfiltration patterns, credential access, capability mismatch, obfuscation, and more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genome_id": {
                        "type": "string",
                        "description": "ID of the skill genome to scan",
                    },
                },
                "required": ["genome_id"],
            },
        },
        "flame",
        "free",
        True,
        False,
        15,
    ),
    (
        {
            "name": "skill_endorse",
            "description": "Endorse a pending crystallized skill for sharing with the Flame. Only endorsed skills can propagate to other PredaCore instances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genome_id": {
                        "type": "string",
                        "description": "ID of the skill to endorse",
                    },
                    "action": {
                        "type": "string",
                        "description": "Action: 'endorse' (approve for sharing) or 'reject' (remove pending skill)",
                        "enum": ["endorse", "reject"],
                    },
                },
                "required": ["genome_id"],
            },
        },
        "flame",
        "free",
        False,
        True,
        15,
    ),
]

MARKETPLACE_TOOLS_RAW = [
    (
        {
            "name": "marketplace_list_skills",
            "description": "List available and installed marketplace skills, including imported OpenClaw skills when configured.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Optional user namespace (defaults to OS user)",
                    },
                    "search": {
                        "type": "string",
                        "description": "Optional search term for available skills",
                    },
                },
            },
        },
        "marketplace",
        "free",
        True,
        False,
        15,
    ),
    (
        {
            "name": "marketplace_install_skill",
            "description": "Install a marketplace skill for a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "Skill ID to install",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user namespace",
                    },
                    "config": {
                        "type": "object",
                        "description": "Optional per-skill configuration",
                    },
                },
                "required": ["skill_id"],
            },
        },
        "marketplace",
        "free",
        False,
        True,
        60,  # optimal: download + extract skill package
    ),
    (
        {
            "name": "marketplace_invoke_skill",
            "description": "Invoke an installed marketplace skill (supports OpenClaw imported skills and script actions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Installed skill ID"},
                    "params": {"type": "object", "description": "Skill parameters"},
                    "user_id": {
                        "type": "string",
                        "description": "Optional user namespace",
                    },
                },
                "required": ["skill_id", "params"],
            },
        },
        "marketplace",
        "medium",
        False,
        False,
        120,  # optimal: skill execution with sandbox
    ),
]


# ── Trust Level Policies ─────────────────────────────────────────────

TRUST_POLICIES = {
    "yolo": {
        "description": "Maximum autonomy — PredaCore executes everything without asking",
        "require_confirmation": [],
        "auto_approve_tools": ["*"],
        "dangerous_tool_check": False,
        "max_auto_exec_cost": 1e18,  # Effectively unlimited, but JSON-serializable
    },
    "normal": {
        "description": "Balanced — confirms destructive actions, auto-approves reads",
        "require_confirmation": [
            "write_file",
            "run_command",
            "python_exec",
            "execute_code",
            "desktop_control",
            "screen_vision",
            "android_control",
            # ── Extensibility surface — these mutate config, install code,
            # or invoke external APIs. All must prompt in normal mode so
            # docs/AUTONOMY.md's "mutating tools are confirmed" claim holds.
            "channel_install",
            "channel_configure",
            "mcp_add",
            "mcp_remove",
            "mcp_restart",
            "secret_set",
            "api_add",
            "api_call",
            "api_remove",
        ],
        "auto_approve_tools": [
            "read_file",
            "list_directory",
            "web_search",
            "web_scrape",
            "memory_store",
            "memory_recall",
            "deep_search",
            "semantic_search",
            "pdf_reader",
            "diagram",
            "tool_stats",
            "tool_pipeline",
        ],
        "dangerous_tool_check": True,
        "max_auto_exec_cost": 0.10,
    },
    "paranoid": {
        "description": "Maximum safety — confirms every action before execution",
        "require_confirmation": ["*"],
        "auto_approve_tools": [],
        "dangerous_tool_check": True,
        "max_auto_exec_cost": 0.0,
    },
}


# ── Factory Functions ────────────────────────────────────────────────


def _register_batch(registry: ToolRegistry, raw_list: list) -> None:
    """Register a batch of (raw_dict, category, cost, parallel, confirm, timeout) tuples."""
    for item in raw_list:
        raw, cat, cost, para, confirm, timeout = item
        registry.register_raw(
            raw,
            category=cat,
            cost_estimate=cost,
            parallelizable=para,
            requires_confirmation=confirm,
            timeout_default=timeout,
        )


def build_builtin_registry() -> ToolRegistry:
    """Build a registry with all built-in tools."""
    reg = ToolRegistry()
    _register_batch(reg, BUILTIN_TOOLS_RAW)
    return reg


def build_full_registry(
    *,
    include_openclaw: bool = False,
    include_marketplace: bool = False,
    include_flame: bool = True,
) -> ToolRegistry:
    """Build a registry with all enabled tool sets."""
    reg = build_builtin_registry()
    if include_openclaw:
        _register_batch(reg, OPENCLAW_BRIDGE_TOOLS_RAW)
    if include_marketplace:
        _register_batch(reg, MARKETPLACE_TOOLS_RAW)
    if include_flame:
        _register_batch(reg, COLLECTIVE_INTELLIGENCE_TOOLS_RAW)
    logger.info(
        "Tool registry: %d tools across %d categories",
        len(reg),
        len(reg.get_categories()),
    )
    return reg
