"""
PredaCore LLM Provider abstractions.

Each provider implements the LLMProvider protocol with an async chat() method.
The LLMInterface in router.py delegates to these providers.
"""
from __future__ import annotations

from .base import LLMProvider, ProviderConfig
from .circuit_breaker import CircuitBreaker
from .types import (
    AssistantResponse,
    Message,
    Role,
    ToolCallRef,
    ToolDefinition,
    ToolResultRef,
    message_from_dict,
    message_to_dict,
)

__all__ = [
    "LLMProvider",
    "ProviderConfig",
    "CircuitBreaker",
    # Typed models (Phase A refactor)
    "Role",
    "Message",
    "ToolCallRef",
    "ToolResultRef",
    "ToolDefinition",
    "AssistantResponse",
    "message_from_dict",
    "message_to_dict",
]
