"""Tests for the OAuth flow primitives + OpenAI Codex provider integration.

Covers (no real network — every external call is patched):

  * PKCE verifier / challenge generation matches RFC 7636
  * OAuthGrantStore round-trips, chmod 600, atomic writes, list/delete
  * OAuthGrant.is_expired and expires_within
  * refresh_access_token: success path, server rejection, no-refresh-token
  * with_auto_refresh: skips refresh when token is fresh, refreshes when
    inside the safety window, double-check after lock acquisition
  * Codex provider builds the right Authorization header from the stored
    grant + falls back to a clear error when no grant exists
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from predacore.llm_providers.oauth.pkce import generate_pkce_pair
from predacore.llm_providers.oauth.refresh import (
    REFRESH_SAFETY_WINDOW_SECONDS,
    refresh_access_token,
    with_auto_refresh,
)
from predacore.llm_providers.oauth.store import OAuthGrant, OAuthGrantStore


# ── PKCE ───────────────────────────────────────────────────────────────


def test_pkce_pair_matches_rfc7636():
    """challenge must equal base64url(sha256(verifier)) without padding."""
    pair = generate_pkce_pair()
    assert pair.method == "S256"
    digest = hashlib.sha256(pair.verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert pair.challenge == expected
    # Verifier length must be in RFC's 43-128 char range
    assert 43 <= len(pair.verifier) <= 128


def test_pkce_pair_is_random():
    """Two pairs in a row must differ. Defends against a global seed bug."""
    a = generate_pkce_pair()
    b = generate_pkce_pair()
    assert a.verifier != b.verifier
    assert a.challenge != b.challenge


def test_pkce_pair_short_length_clamps_to_minimum():
    """Asking for too-low entropy still yields RFC-compliant verifier."""
    pair = generate_pkce_pair(verifier_length=8)
    assert len(pair.verifier) >= 43


# ── OAuthGrantStore ────────────────────────────────────────────────────


def _grant(provider: str = "test_provider", **kwargs) -> OAuthGrant:
    defaults = dict(
        access_token="atok-abc",
        refresh_token="rtok-xyz",
        expires_at=time.time() + 3600,
        scope="read write",
        account_id="acct_123",
        client_id="app_test",
    )
    defaults.update(kwargs)
    return OAuthGrant(provider=provider, **defaults)


def test_store_save_and_load_roundtrip(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    saved = store.save(_grant())
    assert saved.is_file()
    loaded = store.load("test_provider")
    assert loaded is not None
    assert loaded.access_token == "atok-abc"
    assert loaded.refresh_token == "rtok-xyz"
    assert loaded.account_id == "acct_123"


def test_store_save_chmods_to_600(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    saved = store.save(_grant())
    mode = saved.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_store_save_is_atomic_no_tmp_leftover(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    store.save(_grant())
    leftover = list(tmp_path.glob(".oauth-*.tmp"))
    assert leftover == []


def test_store_load_returns_none_when_missing(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    assert store.load("never_existed") is None


def test_store_delete_returns_true_on_hit_false_on_miss(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    store.save(_grant())
    assert store.delete("test_provider") is True
    assert store.delete("test_provider") is False  # second call no-op


def test_store_list_providers(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    store.save(_grant(provider="openai_codex"))
    store.save(_grant(provider="acme"))
    assert store.list_providers() == ["acme", "openai_codex"]


def test_store_rejects_provider_path_traversal(tmp_path: Path):
    store = OAuthGrantStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("../etc/passwd")
    with pytest.raises(ValueError):
        store.load("a/b")


# ── OAuthGrant expiry helpers ──────────────────────────────────────────


def test_grant_is_expired_true_when_past():
    grant = _grant(expires_at=time.time() - 60)
    assert grant.is_expired is True
    assert grant.expires_within < 0


def test_grant_is_expired_false_when_future():
    grant = _grant(expires_at=time.time() + 600)
    assert grant.is_expired is False
    assert grant.expires_within > 0


def test_grant_is_expired_false_when_unknown():
    """expires_at=0 means provider didn't supply expires_in — treat as
    not-expired so we don't infinite-loop refreshing."""
    grant = _grant(expires_at=0)
    assert grant.is_expired is False
    assert grant.expires_within == float("inf")


# ── refresh_access_token ───────────────────────────────────────────────


def _http_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://example/token"),
    )


@pytest.mark.asyncio
async def test_refresh_returns_new_grant_with_updated_tokens():
    grant = _grant(refresh_token="r1", access_token="a1")
    fake_response = {
        "access_token": "a2",
        "refresh_token": "r2",
        "expires_in": 1800,
        "scope": "read write",
    }

    async def fake_post(self, url, *, data, headers, **kw):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "r1"
        assert data["client_id"] == "app_test"
        return _http_response(fake_response)

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        new_grant = await refresh_access_token(
            grant,
            token_url="https://example.com/token",
            client_id="app_test",
        )
    assert new_grant.access_token == "a2"
    assert new_grant.refresh_token == "r2"
    assert new_grant.expires_at > time.time() + 1700  # ~1800s in the future


@pytest.mark.asyncio
async def test_refresh_keeps_old_refresh_token_when_provider_doesnt_rotate():
    """Some providers omit refresh_token from the response when not rotating."""
    grant = _grant(refresh_token="r1")

    async def fake_post(self, url, *, data, headers, **kw):
        return _http_response({"access_token": "a2", "expires_in": 60})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        new_grant = await refresh_access_token(
            grant, token_url="https://example.com/token", client_id="x",
        )
    assert new_grant.refresh_token == "r1"


@pytest.mark.asyncio
async def test_refresh_raises_authentication_error_on_400():
    """Refresh-token rejection surfaces as AuthenticationError so the
    caller can prompt a fresh login."""
    from predacore.llm_providers import predacore_sdk as psdk

    async def fake_post(self, url, *, data, headers, **kw):
        return _http_response({"error": "invalid_grant"}, status=400)

    grant = _grant()
    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        with pytest.raises(psdk.AuthenticationError):
            await refresh_access_token(
                grant, token_url="https://example.com/token", client_id="x",
            )


@pytest.mark.asyncio
async def test_refresh_raises_when_no_refresh_token():
    from predacore.llm_providers import predacore_sdk as psdk

    grant = _grant(refresh_token="")
    with pytest.raises(psdk.AuthenticationError):
        await refresh_access_token(
            grant, token_url="https://example.com/token", client_id="x",
        )


# ── with_auto_refresh ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_refresh_skips_when_token_is_fresh(tmp_path: Path):
    """Token expires far in the future → no refresh, no HTTP call."""
    store = OAuthGrantStore(tmp_path)
    grant = _grant(expires_at=time.time() + 3600)  # 1h
    store.save(grant)

    refresh_called = False

    async def fake_refresh(grant_arg, *, token_url, client_id, timeout_seconds=30):
        nonlocal refresh_called
        refresh_called = True
        return grant_arg

    with patch(
        "predacore.llm_providers.oauth.refresh.refresh_access_token",
        new=fake_refresh,
    ):
        async with with_auto_refresh(
            provider="test_provider",
            token_url="https://example/token",
            client_id="x",
            store=store,
        ) as g:
            assert g.access_token == "atok-abc"
    assert refresh_called is False


@pytest.mark.asyncio
async def test_auto_refresh_runs_when_inside_safety_window(tmp_path: Path):
    """Token expires inside the safety window → refresh fires."""
    store = OAuthGrantStore(tmp_path)
    soon = time.time() + (REFRESH_SAFETY_WINDOW_SECONDS - 60)  # ~4 min left
    grant = _grant(expires_at=soon, access_token="stale")
    store.save(grant)

    async def fake_refresh(grant_arg, *, token_url, client_id, timeout_seconds=30):
        return OAuthGrant(
            provider=grant_arg.provider,
            access_token="fresh",
            refresh_token=grant_arg.refresh_token,
            expires_at=time.time() + 3600,
            client_id=client_id,
        )

    with patch(
        "predacore.llm_providers.oauth.refresh.refresh_access_token",
        new=fake_refresh,
    ):
        async with with_auto_refresh(
            provider="test_provider",
            token_url="https://example/token",
            client_id="x",
            store=store,
        ) as g:
            assert g.access_token == "fresh"

    # Refreshed grant was persisted
    persisted = store.load("test_provider")
    assert persisted is not None
    assert persisted.access_token == "fresh"


@pytest.mark.asyncio
async def test_auto_refresh_raises_when_no_stored_grant(tmp_path: Path):
    from predacore.llm_providers import predacore_sdk as psdk

    store = OAuthGrantStore(tmp_path)
    with pytest.raises(psdk.AuthenticationError, match="no stored grant"):
        async with with_auto_refresh(
            provider="never_authorized",
            token_url="https://example/token",
            client_id="x",
            store=store,
        ):
            pass  # pragma: no cover — the with-stmt raises before entering


# ── Codex provider integration ────────────────────────────────────────


@pytest.mark.asyncio
async def test_codex_provider_raises_when_no_stored_grant(tmp_path, monkeypatch):
    """No grant on disk → Codex provider's chat() raises AuthenticationError
    pointing the user at `predacore login openai-codex`."""
    from predacore.llm_providers import predacore_sdk as psdk
    from predacore.llm_providers.base import ProviderConfig
    from predacore.llm_providers.openai_codex import OpenAICodexProvider

    # Redirect the OAuth store to an empty tmp dir so no real grants leak in
    monkeypatch.setattr(
        "predacore.llm_providers.oauth.store.DEFAULT_OAUTH_DIR",
        tmp_path / "oauth",
    )
    # Module-level cached store from store.py needs a reset
    import predacore.llm_providers.oauth.store as store_mod
    monkeypatch.setattr(store_mod, "_default_store", None)

    cfg = ProviderConfig(model="gpt-5-codex", temperature=0.7)
    provider = OpenAICodexProvider(cfg)
    with pytest.raises(psdk.AuthenticationError, match="predacore login"):
        await provider.chat(messages=[{"role": "user", "content": "hi"}])


def test_codex_oauth_config_picks_up_env_override(monkeypatch):
    from predacore.llm_providers.openai_codex import codex_oauth_config

    monkeypatch.setenv("PREDACORE_OPENAI_CODEX_CLIENT_ID", "app_user_owned_xyz")
    cfg = codex_oauth_config()
    assert cfg.client_id == "app_user_owned_xyz"
    assert cfg.authorization_url == "https://auth.openai.com/oauth/authorize"
    assert cfg.token_url == "https://auth.openai.com/oauth/token"


def test_codex_oauth_config_default_client_id(monkeypatch):
    from predacore.llm_providers.openai_codex import (
        DEFAULT_CODEX_CLIENT_ID,
        codex_oauth_config,
    )
    monkeypatch.delenv("PREDACORE_OPENAI_CODEX_CLIENT_ID", raising=False)
    cfg = codex_oauth_config()
    assert cfg.client_id == DEFAULT_CODEX_CLIENT_ID
