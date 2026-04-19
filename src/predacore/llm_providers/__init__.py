"""
PredaCore LLM Provider abstractions.

Each provider implements the LLMProvider protocol with an async chat() method.
The LLMInterface in router.py delegates to these providers.
"""
from __future__ import annotations

from .base import LLMProvider, ProviderConfig
from .circuit_breaker import CircuitBreaker
from .text_tool_adapter import (
    build_full_text_prompt,
    build_tool_prompt,
    parse_tool_calls,
)

__all__ = [
    "LLMProvider",
    "ProviderConfig",
    "CircuitBreaker",
    "build_tool_prompt",
    "build_full_text_prompt",
    "parse_tool_calls",
]
