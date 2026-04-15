"""
Shared Claude model alias normalization for providers that talk to Anthropic.
"""
from __future__ import annotations

DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"

# Anthropic accepts the full API aliases below. Keep short nicknames here so
# config values like "opus" work across both Claude Code and raw Anthropic SDK
# providers.
MODEL_ALIASES = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "opus-4.6": "claude-opus-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "haiku-4.5": "claude-haiku-4-5",
}


def resolve_claude_model(
    model: str | None,
    *,
    default: str = DEFAULT_CLAUDE_MODEL,
) -> str:
    """Resolve short aliases like ``opus`` to Anthropic API model ids."""
    raw_model = (model or "").strip()
    if not raw_model:
        return default
    return MODEL_ALIASES.get(raw_model, raw_model)
