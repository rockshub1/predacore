"""
Prometheus Structured Error Hierarchy.

Every error in the system inherits from PrometheusError, which carries:
  - error_code:  machine-readable string (e.g. "TOOL_EXEC_FAILED")
  - context:     dict of debug info (tool name, args, trace_id, etc.)
  - recoverable: whether the caller should retry or bail

Usage:
    try:
        await executor.execute("shell_exec", {"command": "rm -rf /"})
    except ToolExecutionError as e:
        logger.error(f"[{e.error_code}] {e}", extra=e.context)
        if e.recoverable:
            await retry(...)
"""
from __future__ import annotations

from typing import Any


class PrometheusError(Exception):
    """
    Base exception for all Prometheus errors.

    Attributes:
        error_code:  Machine-readable error identifier.
        context:     Structured debug information.
        recoverable: Hint to callers whether retry might help.
    """

    error_code: str = "PREDACORE_ERROR"
    recoverable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ):
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code
        self.context: dict[str, Any] = context or {}
        if recoverable is not None:
            self.recoverable = recoverable

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"[{self.error_code}] {base} ({ctx})"
        return f"[{self.error_code}] {base}"


# ── Tool Errors ──────────────────────────────────────────────────────


class ToolExecutionError(PrometheusError):
    """A tool call failed during execution."""

    error_code = "TOOL_EXEC_FAILED"
    recoverable = True


class ToolNotFoundError(PrometheusError):
    """Requested tool does not exist."""

    error_code = "TOOL_NOT_FOUND"
    recoverable = False


class ToolPermissionError(PrometheusError):
    """Tool blocked by trust policy or EGM."""

    error_code = "TOOL_PERMISSION_DENIED"
    recoverable = False


class ToolTimeoutError(ToolExecutionError):
    """Tool execution exceeded time limit."""

    error_code = "TOOL_TIMEOUT"
    recoverable = True


# ── LLM Provider Errors ─────────────────────────────────────────────


class LLMProviderError(PrometheusError):
    """An LLM API call failed."""

    error_code = "LLM_PROVIDER_ERROR"
    recoverable = True


class LLMRateLimitError(LLMProviderError):
    """Provider rate limit hit — retry with backoff."""

    error_code = "LLM_RATE_LIMIT"
    recoverable = True


class LLMContextLengthError(LLMProviderError):
    """Message exceeds provider context window."""

    error_code = "LLM_CONTEXT_TOO_LONG"
    recoverable = False


class LLMAllProvidersFailedError(LLMProviderError):
    """Every provider in the failover chain has failed."""

    error_code = "LLM_ALL_PROVIDERS_FAILED"
    recoverable = False


# ── Auth & Security Errors ───────────────────────────────────────────


class AuthenticationError(PrometheusError):
    """Authentication failed — bad token, expired, etc."""

    error_code = "AUTH_FAILED"
    recoverable = False


class AuthorizationError(PrometheusError):
    """User lacks permission for this action."""

    error_code = "AUTH_FORBIDDEN"
    recoverable = False


# ── Sandbox Errors ───────────────────────────────────────────────────


class SandboxError(PrometheusError):
    """Docker sandbox operation failed."""

    error_code = "SANDBOX_ERROR"
    recoverable = True


class SandboxNotAvailableError(SandboxError):
    """Docker daemon not running or sandbox image not built."""

    error_code = "SANDBOX_NOT_AVAILABLE"
    recoverable = False


class SandboxTimeoutError(SandboxError):
    """Sandbox execution exceeded time limit."""

    error_code = "SANDBOX_TIMEOUT"
    recoverable = True


# ── Memory & Persistence Errors ──────────────────────────────────────


class MemoryServiceError(PrometheusError):
    """Memory storage or retrieval failed."""

    error_code = "MEMORY_ERROR"
    recoverable = True


class MemoryNotFoundError(MemoryServiceError):
    """Requested memory ID does not exist."""

    error_code = "MEMORY_NOT_FOUND"
    recoverable = False


class PersistenceError(PrometheusError):
    """Database or file I/O failure."""

    error_code = "PERSISTENCE_ERROR"
    recoverable = True


# ── Channel Errors ───────────────────────────────────────────────────


class ChannelError(PrometheusError):
    """A messaging channel encountered an error."""

    error_code = "CHANNEL_ERROR"
    recoverable = True


class ChannelConnectionError(ChannelError):
    """Failed to connect to a messaging platform."""

    error_code = "CHANNEL_CONNECT_FAILED"
    recoverable = True


class ChannelRateLimitError(ChannelError):
    """Platform rate limit exceeded."""

    error_code = "CHANNEL_RATE_LIMIT"
    recoverable = True


class ChannelMessageTooLongError(ChannelError):
    """Message exceeds platform character limit."""

    error_code = "CHANNEL_MSG_TOO_LONG"
    recoverable = False


# ── Planning & Agent Errors ──────────────────────────────────────────


class PlanningError(PrometheusError):
    """Plan generation failed."""

    error_code = "PLANNING_ERROR"
    recoverable = True


class AgentLifecycleError(PrometheusError):
    """Agent spawn/terminate/collaborate failure."""

    error_code = "AGENT_LIFECYCLE_ERROR"
    recoverable = True


class EGMViolationError(PrometheusError):
    """Action blocked by Ethical Governance Module."""

    error_code = "EGM_VIOLATION"
    recoverable = False


# ── Configuration Errors ─────────────────────────────────────────────


class ConfigurationError(PrometheusError):
    """Invalid or missing configuration."""

    error_code = "CONFIG_ERROR"
    recoverable = False
