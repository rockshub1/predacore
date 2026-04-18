"""
Shared ToolContext and utilities used by all handler modules.

This is the single source of truth for ToolContext — every handler
module imports from here instead of defining its own.

Also provides:
  - ToolError exception hierarchy for consistent error handling
  - Shared sensitive path patterns
  - Web cache utilities
  - Multi-agent recursion guard
"""
from __future__ import annotations

import collections
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from predacore.config import PredaCoreConfig
    from predacore.memory import UnifiedMemoryStore
    from predacore._vendor.common.memory_service import MemoryService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolError — consistent error handling across all handlers
# ---------------------------------------------------------------------------


class ToolErrorKind(str, Enum):
    """Categories of tool errors for structured handling."""
    MISSING_PARAM = "missing_param"       # Required parameter not provided
    INVALID_PARAM = "invalid_param"       # Parameter has wrong type/value
    NOT_FOUND = "not_found"               # Resource not found (file, element, etc.)
    BLOCKED = "blocked"                   # Blocked by security policy
    UNAVAILABLE = "unavailable"           # Required subsystem not available
    TIMEOUT = "timeout"                   # Operation timed out
    EXECUTION = "execution"               # Runtime execution error
    PERMISSION = "permission"             # Permission denied
    LIMIT_EXCEEDED = "limit_exceeded"     # Size/rate/depth limit hit


class ToolError(Exception):
    """Base exception for all tool handler errors.

    Provides consistent error formatting across all handlers.
    Instead of each handler returning ad-hoc "[Error: ...]" strings,
    handlers raise ToolError and the dispatcher formats it uniformly.

    Usage in handlers:
        from ._context import ToolError, ToolErrorKind

        if not args.get("path"):
            raise ToolError("path", kind=ToolErrorKind.MISSING_PARAM)

        if not path.exists():
            raise ToolError(
                f"File not found: {path}",
                kind=ToolErrorKind.NOT_FOUND,
                suggestion="Check the path and try again",
            )
    """

    def __init__(
        self,
        message: str,
        *,
        kind: ToolErrorKind = ToolErrorKind.EXECUTION,
        tool_name: str = "",
        recoverable: bool = True,
        suggestion: str = "",
    ):
        self.kind = kind
        self.tool_name = tool_name
        self.recoverable = recoverable
        self.suggestion = suggestion
        super().__init__(message)

    def format(self) -> str:
        """Format as the standard [bracketed] error string handlers return."""
        parts = [str(self)]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        if not self.recoverable:
            parts.append("(non-recoverable)")
        return f"[{' — '.join(parts)}]"

    def to_dict(self) -> dict[str, Any]:
        """Structured error for JSON responses."""
        return {
            "error": str(self),
            "kind": self.kind.value,
            "tool": self.tool_name,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
        }


# ---------------------------------------------------------------------------
# Convenience error constructors (DRY across handlers)
# ---------------------------------------------------------------------------


def missing_param(name: str, tool: str = "") -> ToolError:
    """Raise for a missing required parameter."""
    return ToolError(
        f"Missing required field: {name}",
        kind=ToolErrorKind.MISSING_PARAM,
        tool_name=tool,
    )


def invalid_param(name: str, reason: str, tool: str = "") -> ToolError:
    """Raise for an invalid parameter value."""
    return ToolError(
        f"Invalid {name}: {reason}",
        kind=ToolErrorKind.INVALID_PARAM,
        tool_name=tool,
    )


def subsystem_unavailable(name: str, tool: str = "") -> ToolError:
    """Raise when a required subsystem is not initialized."""
    return ToolError(
        f"{name} is not available",
        kind=ToolErrorKind.UNAVAILABLE,
        tool_name=tool,
        suggestion=f"Check that {name} is properly configured and initialized",
    )


def resource_not_found(what: str, path: str = "", tool: str = "") -> ToolError:
    """Raise when a resource (file, element, etc.) is not found."""
    msg = f"{what} not found"
    if path:
        msg += f": {path}"
    return ToolError(msg, kind=ToolErrorKind.NOT_FOUND, tool_name=tool)


def blocked(reason: str, tool: str = "") -> ToolError:
    """Raise when an operation is blocked by security policy."""
    return ToolError(
        f"Blocked: {reason}",
        kind=ToolErrorKind.BLOCKED,
        tool_name=tool,
        recoverable=False,
    )


# ---------------------------------------------------------------------------
# Multi-agent recursion guard
# ---------------------------------------------------------------------------

_DELEGATION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "delegation_depth", default=0
)
_MAX_DELEGATION_DEPTH = 3

# ---------------------------------------------------------------------------
# Shared context passed to every handler
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Immutable-ish bag of resources that handlers need."""

    # Core config — always present
    config: Any                                           # PredaCoreConfig
    memory: dict[str, dict[str, Any]]                     # in-memory session store

    # Services — all Optional because they may not have initialised.
    # Types use Any at runtime to avoid circular imports; see TYPE_CHECKING block above.
    memory_service: Any = None                            # MemoryService | None
    mcts_planner: Any = None                              # ABMCTSPlanner | None
    voice: Any = None                                     # VoiceInterface | None
    desktop_operator: Any = None                          # MacDesktopOperator | None
    sandbox: Any = None                                   # SubprocessSandboxManager | None
    docker_sandbox: Any = None                            # DockerSandboxManager | None
    sandbox_pool: Any = None                              # SessionSandboxPool | None
    skill_marketplace: Any = None                         # SkillMarketplace | None
    openclaw_runtime: Any = None                          # OpenClawBridgeRuntime | None
    openclaw_enabled: bool = False
    llm_for_collab: Any = None                            # LLMInterface | None
    unified_memory: Any = None                            # UnifiedMemoryStore | None
    trust_policy: dict = field(default_factory=dict)

    # Helpers that live on ToolExecutor but are needed by some handlers
    http_with_retry: Callable | None = None               # async HTTP with retry
    format_sandbox_result: Callable | None = None         # sandbox result formatter
    resolve_user_id: Callable | None = None               # user ID resolver


# ---------------------------------------------------------------------------
# Lazy imports used by multiple handlers
# ---------------------------------------------------------------------------


def _lazy_memory_types():
    """Return (MemoryType, ImportanceLevel) — imported lazily."""
    try:
        from predacore._vendor.common.memory_service import ImportanceLevel, MemoryType
        return MemoryType, ImportanceLevel
    except Exception:
        pass
    # Local fallback enums when external memory_service is unavailable
    from enum import Enum, IntEnum

    class MemoryType(str, Enum):
        FACT = "fact"
        CONVERSATION = "conversation"
        TASK = "task"
        PREFERENCE = "preference"
        CONTEXT = "context"
        SKILL = "skill"
        ENTITY = "entity"

    class ImportanceLevel(IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4

    return MemoryType, ImportanceLevel


# ---------------------------------------------------------------------------
# Sensitive path patterns (shared by file_ops and creative handlers)
# ---------------------------------------------------------------------------

SENSITIVE_READ_PATTERNS = [
    ".ssh/", ".gnupg/", ".aws/credentials", ".config/gcloud/",
    "shadow", ".env", "credentials.json", "id_rsa", "id_ed25519",
    ".netrc", ".npmrc", "/token.json", ".token", ".kube/config", ".docker/config.json",
]

SENSITIVE_WRITE_PATHS = [
    "/etc/", "/var/", "/usr/", "/sys/", "/proc/", "/dev/",
    "/boot/", "/sbin/", "/lib/", "/lib64/",
]

SENSITIVE_WRITE_FILES = [
    ".ssh/authorized_keys", ".ssh/id_rsa", ".ssh/id_ed25519",
    ".bashrc", ".bash_profile", ".zshrc", ".profile",
    ".gnupg/", ".aws/credentials", ".config/gcloud/",
    "shadow", "passwd", "sudoers",
    # Safety-floor identity files — tamper-protected. Writes to these would
    # bypass the identity_update tool's allowlist and undermine SOUL_SEED
    # invariants. Never writable via file_ops, regardless of trust level.
    "SOUL_SEED.md", "EVENT_HORIZON.md",
]

# ---------------------------------------------------------------------------
# Web cache (shared by web handlers)
# ---------------------------------------------------------------------------

_WEB_CACHE: collections.OrderedDict[str, tuple[float, str]] = collections.OrderedDict()
_WEB_CACHE_LOCK = threading.Lock()
_WEB_CACHE_TTL = 300  # 5 minutes
_WEB_CACHE_MAX = 100


def web_cache_get(key: str) -> str | None:
    """Return cached result if still fresh."""
    with _WEB_CACHE_LOCK:
        entry = _WEB_CACHE.get(key)
        if entry and (time.time() - entry[0]) < _WEB_CACHE_TTL:
            _WEB_CACHE.move_to_end(key)
            return entry[1]
        if entry:
            _WEB_CACHE.pop(key, None)
        return None


def web_cache_put(key: str, value: str) -> None:
    """Store result in cache, evicting oldest if full (O(1) via OrderedDict)."""
    with _WEB_CACHE_LOCK:
        if key in _WEB_CACHE:
            _WEB_CACHE.move_to_end(key)
        elif len(_WEB_CACHE) >= _WEB_CACHE_MAX:
            _WEB_CACHE.popitem(last=False)  # O(1) evict oldest
        _WEB_CACHE[key] = (time.time(), value)
