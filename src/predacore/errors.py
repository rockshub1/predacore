"""
PredaCore Structured Error Types.

Phase 6.5: Replace string-wrapped errors with typed exceptions for
proper error handling, logging, and recovery decisions.
"""
from __future__ import annotations

from typing import Any


class PredaCoreError(Exception):
    """Base exception for all PredaCore errors.

    Attributes:
        error_code:  Machine-readable error identifier (e.g. "TOOL_EXEC_FAILED").
        recoverable: Hint to callers whether retry might help.
        context:     Structured debug information.
        details:     Legacy detail dict (preserved for backward compat).
    """

    error_code: str = ""
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "",
        recoverable: bool = False,
        context: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        if error_code:
            self.error_code = error_code
        if recoverable:
            self.recoverable = recoverable
        self.context: dict[str, Any] = context or {}
        self.details = details or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.error_code:
            return f"[{self.error_code}] {base}"
        return base


# ── Tool Errors ─────────────────────────────────────────────────────


class ToolError(PredaCoreError):
    """Error during tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Tool not registered in the tool registry."""


class ToolTimeoutError(ToolError):
    """Tool execution exceeded timeout."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            tool_name=tool_name,
            details={"timeout_seconds": timeout_seconds},
        )
        self.timeout_seconds = timeout_seconds


class ToolPermissionError(ToolError):
    """User does not have permission to execute this tool."""


class ToolValidationError(ToolError):
    """Tool arguments failed schema validation."""


# ── LLM Errors ──────────────────────────────────────────────────────


class LLMError(PredaCoreError):
    """Error from LLM provider communication."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.provider = provider
        self.model = model


class LLMRateLimitError(LLMError):
    """Provider returned 429 rate limit."""

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(
            f"Rate limited by {provider}" + (f" (retry after {retry_after}s)" if retry_after else ""),
            provider=provider,
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after


class LLMConnectionError(LLMError):
    """Cannot connect to LLM provider."""


class LLMAllProvidersFailedError(LLMError):
    """All providers in the failover chain failed."""

    def __init__(self, errors: list[tuple[str, str]]):
        summary = "; ".join(f"{p}: {e}" for p, e in errors)
        super().__init__(
            f"All LLM providers failed: {summary}",
            details={"provider_errors": errors},
        )
        self.provider_errors = errors


# ── Memory Errors ───────────────────────────────────────────────────


class MemoryStoreError(PredaCoreError):
    """Error in memory store/recall operations."""


class MemoryCapacityError(MemoryStoreError):
    """Memory store has reached its capacity limit."""


class MemoryNotFoundError(MemoryStoreError):
    """Requested memory entry not found."""


# ── Session Errors ──────────────────────────────────────────────────


class SessionError(PredaCoreError):
    """Error in session management."""


class SessionNotFoundError(SessionError):
    """Session ID not found."""


class SessionCorruptedError(SessionError):
    """Session data is corrupted or inconsistent."""


# ── Security Errors ─────────────────────────────────────────────────


class SecurityError(PredaCoreError):
    """Security-related error."""


class SSRFBlockedError(SecurityError):
    """URL blocked by SSRF protection."""


class CommandInjectionError(SecurityError):
    """Potential command injection detected."""


class PromptInjectionError(SecurityError):
    """Potential prompt injection detected in input."""


class RateLimitExceededError(SecurityError):
    """User exceeded their rate limit."""

    def __init__(self, user_id: str, limit: int, window_seconds: int):
        super().__init__(
            f"User {user_id} exceeded rate limit ({limit} requests per {window_seconds}s)",
            details={"user_id": user_id, "limit": limit, "window_seconds": window_seconds},
        )


# ── Desktop/Device Errors ──────────────────────────────────────────


class DesktopError(PredaCoreError):
    """Error in desktop/device control operations."""


class DeviceNotConnectedError(DesktopError):
    """No device connected (ADB, etc.)."""


class AccessibilityPermissionError(DesktopError):
    """Missing Accessibility/Automation permissions."""
