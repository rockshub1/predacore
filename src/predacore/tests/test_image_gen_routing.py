"""Provider-selection tests for handle_image_gen (T10d).

Verifies that the auto-routing in ``handle_image_gen`` picks the right
provider based on which API key is present, without making any real
HTTP calls. We mock the provider-specific helpers so the routing logic
itself is what's under test.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from predacore.tools.handlers._context import ToolContext, ToolError


@pytest.fixture
def ctx():
    """A minimal ToolContext suitable for handler invocation."""
    from types import SimpleNamespace
    return ToolContext(
        config=SimpleNamespace(home_dir="/tmp/test-predacore"),
        memory={},
    )


@pytest.mark.asyncio
async def test_picks_gemini_when_key_set(monkeypatch, ctx):
    """GEMINI_API_KEY present + no explicit provider → Gemini Image."""
    monkeypatch.setenv("GEMINI_API_KEY", "free-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch(
        "predacore.tools.handlers.creative._generate_via_gemini",
        new=AsyncMock(return_value='{"provider":"gemini"}'),
    ) as gemini_mock:
        from predacore.tools.handlers.creative import handle_image_gen
        out = await handle_image_gen({"prompt": "a cat"}, ctx)
    assert "gemini" in out
    gemini_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_provider_gemini_without_key_raises(monkeypatch, ctx):
    """Explicit provider='gemini' with no key → ToolError mentioning AI Studio."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from predacore.tools.handlers.creative import handle_image_gen
    with pytest.raises(ToolError) as exc_info:
        await handle_image_gen(
            {"prompt": "a cat", "provider": "gemini"}, ctx,
        )
    assert "GEMINI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_falls_back_to_openai_when_only_openai_key(monkeypatch, ctx):
    """Only OPENAI_API_KEY set → DALL-E path (Gemini helper NOT called)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    # We don't want to actually call DALL-E; mock httpx.AsyncClient.post
    # to return a synthetic response and verify routing landed in the
    # DALL-E branch.
    from unittest.mock import MagicMock
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "data": [{"url": "https://x/img.png", "revised_prompt": "ok"}],
    }
    fake_resp.raise_for_status = MagicMock()
    with patch(
        "predacore.tools.handlers.creative._generate_via_gemini",
        new=AsyncMock(),
    ) as gemini_mock:
        with patch("httpx.AsyncClient") as ac:
            ac.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=fake_resp,
            )
            from predacore.tools.handlers.creative import handle_image_gen
            out = await handle_image_gen({"prompt": "test"}, ctx)
    # Gemini path was NOT taken; output is from DALL-E branch.
    gemini_mock.assert_not_awaited()
    assert "image_url" in out


@pytest.mark.asyncio
async def test_no_keys_raises_helpful_error(monkeypatch, ctx):
    """Neither key set → ToolError pointing at the free Gemini option."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from predacore.tools.handlers.creative import handle_image_gen
    with pytest.raises(ToolError) as exc_info:
        await handle_image_gen({"prompt": "anything"}, ctx)
    msg = str(exc_info.value)
    assert "GEMINI_API_KEY" in msg
    # Suggestion text should point at the free option
    assert "aistudio" in (exc_info.value.suggestion or "").lower()
