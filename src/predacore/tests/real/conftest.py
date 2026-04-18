"""
Shared fixtures + per-test credential gating for real integration tests.

Each real test declares the credential it needs via a fixture. If the
credential isn't available, the test is skipped *individually* with a
clear reason — no red failures from missing keys, just skipped dots.

Credential sources:
  * ``ANTHROPIC_API_KEY`` env var — for ``test_anthropic_real.py``
  * ``OPENAI_API_KEY`` env var — for optional OpenAI tests (none yet)

All real tests also require the top-level ``--real`` flag (see the
root conftest.py). Without ``--real``, every test in this package is
skipped before fixtures even run.
"""
from __future__ import annotations

import os
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Credential discovery
# ---------------------------------------------------------------------------


def _have_anthropic_api_key() -> bool:
    """True if ANTHROPIC_API_KEY is set and looks like a real key."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return key.startswith("sk-ant-")


# ---------------------------------------------------------------------------
# Provider config fixtures (build real ProviderConfig objects from env)
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_api_key() -> str:
    """Skip the test if ANTHROPIC_API_KEY isn't set."""
    if not _have_anthropic_api_key():
        pytest.skip(
            "ANTHROPIC_API_KEY not set (need a real sk-ant-* key from "
            "console.anthropic.com). Export it and re-run with --real."
        )
    return os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture
def anthropic_provider(anthropic_api_key: str):
    """AnthropicProvider configured with the real API key from env."""
    from predacore.llm_providers.anthropic import AnthropicProvider
    from predacore.llm_providers.base import ProviderConfig

    return AnthropicProvider(
        ProviderConfig(
            model="claude-opus-4-6",
            api_key=anthropic_api_key,
            max_tokens=4096,
            reasoning_effort="medium",
            extras={"provider": "anthropic"},
        )
    )


@pytest.fixture
def haiku_anthropic_provider(anthropic_api_key: str):
    """AnthropicProvider on Haiku — cheap, fast, for tests that don't need Opus."""
    from predacore.llm_providers.anthropic import AnthropicProvider
    from predacore.llm_providers.base import ProviderConfig

    return AnthropicProvider(
        ProviderConfig(
            model="claude-haiku-4-5",
            api_key=anthropic_api_key,
            max_tokens=1024,
            reasoning_effort="medium",
            extras={"provider": "anthropic"},
        )
    )


# ---------------------------------------------------------------------------
# Common test-payload builders
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_read_tool() -> dict[str, Any]:
    """A minimal read_file tool definition for tool-use tests.

    Uses a terse schema so the model is strongly nudged to call it.
    """
    return {
        "name": "read_file",
        "description": (
            "Read the full contents of a file. ALWAYS use this tool when the "
            "user asks you to read, open, or show a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    }


@pytest.fixture
def strict_read_tool(simple_read_tool: dict[str, Any]) -> dict[str, Any]:
    """Same as ``simple_read_tool`` but with strict: true enabled."""
    return {**simple_read_tool, "strict": True}
