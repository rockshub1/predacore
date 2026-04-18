"""
Base class and types for LLM providers.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration passed to each provider."""

    model: str = ""
    model_fallbacks: list[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    api_key: str = ""
    api_base: str = ""
    home_dir: str = "~/.predacore"
    # Provider-specific extras
    extras: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    name: str = "base"
    supports_native_tools: bool = True
    executes_tool_loop_internally: bool = False

    def __init__(self, config: ProviderConfig, log: logging.Logger | None = None):
        self.config = config
        self._log = log or logger

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return standardized result.

        Args:
            stream_fn: Optional callback invoked with each token as it arrives.
                       When provided and the response contains no tool_calls,
                       the provider should stream tokens via this callback.

        Returns:
            {
                "content": str,
                "tool_calls": list[dict],
                "usage": {"prompt_tokens": int, "completion_tokens": int},
                "finish_reason": str,
            }
        """
        ...
