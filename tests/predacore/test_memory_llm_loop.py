"""T4 — LLM-in-the-loop behavioral tests via Gemini.

These tests answer the question NO unit/integration test can: **does the
LLM actually USE memory correctly given the rich tool descriptions we
wrote in W1+W2?**

Every test here is marked @pytest.mark.real because it makes a live API
call to Gemini. Run with `pytest --real`. Skipped by default.

The tests are scenario-driven:
  • Scenario: "user states a preference" → assert the LLM calls memory_store
  • Scenario: "what did we decide about X?" → assert the LLM calls memory_recall
  • Scenario: "read foo.py" → assert the LLM does NOT call memory_store
    (D12's auto-reindex handles file content; the LLM shouldn't
    duplicate the work)

Each test is bounded to a SINGLE Gemini call to keep cost negligible
(free-tier flash quota is plenty).
"""
from __future__ import annotations

import os

import pytest

from predacore.llm_providers.base import ProviderConfig
from predacore.llm_providers.gemini import GeminiProvider
from predacore.tools.registry import build_builtin_registry


# ─────────────────────────────────────────────────────────────────────
# Credential gating
# ─────────────────────────────────────────────────────────────────────


def _have_gemini_key() -> bool:
    """True if GEMINI_API_KEY or GOOGLE_API_KEY is set."""
    return bool(
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


@pytest.fixture
def gemini_provider() -> GeminiProvider:
    """Construct a real GeminiProvider for tests. Skips test if no key."""
    if not _have_gemini_key():
        pytest.skip("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    return GeminiProvider(ProviderConfig(
        model="gemini-2.0-flash",  # free + fast
        temperature=0.0,           # deterministic for tests
    ))


@pytest.fixture
def memory_tools() -> list[dict]:
    """The 6 memory tools formatted for LLM consumption (OpenAI-style schema)."""
    reg = build_builtin_registry()
    names = ["memory_store", "memory_recall", "memory_get",
             "memory_delete", "memory_stats", "memory_explain"]
    return [reg.get(n).to_openai_dict() for n in names]


# ═════════════════════════════════════════════════════════════════════
# Smoke: Gemini provider works at all
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.real
async def test_gemini_smoke_basic_chat(gemini_provider):
    """Verify the provider can make a basic chat call with no tools."""
    response = await gemini_provider.chat(
        messages=[{"role": "user", "content": "say only the word 'hello'"}],
        max_tokens=10,
    )
    assert "content" in response
    assert response["content"]
    # Loose match — Gemini usually says "hello" but might add punctuation
    assert "hello" in response["content"].lower()


@pytest.mark.real
async def test_gemini_can_see_memory_tools(gemini_provider, memory_tools):
    """When given memory tools + a relevant scenario, Gemini sees them
    in the registry. We just verify no error from passing tools."""
    response = await gemini_provider.chat(
        messages=[
            {"role": "user", "content": "What memory tools do you have available?"},
        ],
        tools=memory_tools,
        max_tokens=200,
    )
    # Either content mentions memory OR tool_calls populated
    assert response.get("content") or response.get("tool_calls")


# ═════════════════════════════════════════════════════════════════════
# BEHAVIORAL — does the LLM call memory_store at the right moment?
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.real
async def test_user_states_preference_triggers_memory_store(gemini_provider, memory_tools):
    """Given a clear user preference statement, the LLM should call
    memory_store (per the W1+W2 description's WHEN TO CALL guidance)."""
    response = await gemini_provider.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agent with persistent memory. When the user "
                    "states a durable preference, save it via memory_store "
                    "with memory_type='preference' and importance=4 or 5."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I really prefer terse, no-fluff replies. "
                    "Please remember this for our future conversations."
                ),
            },
        ],
        tools=memory_tools,
        max_tokens=400,
    )
    tool_calls = response.get("tool_calls") or []
    called_names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
    assert "memory_store" in called_names, (
        f"expected memory_store to be called for explicit preference statement; "
        f"got tool_calls={tool_calls!r}, content={response.get('content')!r}"
    )


@pytest.mark.real
async def test_past_decision_question_triggers_memory_recall(
    gemini_provider, memory_tools,
):
    """Given a question about a past decision, the LLM should call
    memory_recall (per the W1+W2 description's trigger phrase examples)."""
    response = await gemini_provider.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agent with persistent memory. For questions "
                    "about past decisions/discussions, call memory_recall "
                    "FIRST before answering."
                ),
            },
            {
                "role": "user",
                "content": "What did we decide about the embedding model choice?",
            },
        ],
        tools=memory_tools,
        max_tokens=400,
    )
    tool_calls = response.get("tool_calls") or []
    called_names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
    assert "memory_recall" in called_names, (
        f"expected memory_recall for past-decision question; "
        f"got tool_calls={tool_calls!r}, content={response.get('content')!r}"
    )


# ═════════════════════════════════════════════════════════════════════
# BEHAVIORAL — does the LLM RESPECT the "don't store code" rule?
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.real
async def test_innocuous_chat_does_not_trigger_memory_store(
    gemini_provider, memory_tools,
):
    """Casual chat with no durable preference shouldn't trigger memory_store
    (the description says: don't store ephemeral subtasks, every detail of
    what just happened, etc.)."""
    response = await gemini_provider.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agent with persistent memory. Use memory_store "
                    "ONLY for durable preferences, decisions, or facts that "
                    "matter next week. Don't store every detail of every "
                    "conversation. Healthy ratio: 0-5 stores per session."
                ),
            },
            {"role": "user", "content": "what time is it in tokyo?"},
        ],
        tools=memory_tools,
        max_tokens=400,
    )
    tool_calls = response.get("tool_calls") or []
    called_names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
    assert "memory_store" not in called_names, (
        f"memory_store should NOT be called for ephemeral chat; "
        f"got: {called_names!r}"
    )
