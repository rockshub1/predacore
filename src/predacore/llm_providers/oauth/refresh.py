"""Auto-refresh of access tokens before they expire.

Two pieces:

  ``refresh_access_token`` — explicit call: hand it a stored grant, get
  back a new grant with a fresh access_token (and possibly a fresh
  refresh_token if the provider rotates them). Hits the provider's
  token endpoint with ``grant_type=refresh_token``.

  ``with_auto_refresh`` — context manager: returns the access token,
  refreshing first if it's within the safety window. File-locked so two
  concurrent agents don't double-refresh and burn one of the two valid
  refresh-grants on a server that rotates them.

The 5-minute safety window matches the official Codex CLI behavior so
predacore behaves identically — refresh feels invisible to the user.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx

from .. import predacore_sdk as psdk
from .store import OAuthGrant, OAuthGrantStore, load_grant, save_grant

logger = logging.getLogger(__name__)


REFRESH_SAFETY_WINDOW_SECONDS = 5 * 60     # match Codex CLI's behavior


# Per-process locks keyed by provider so two concurrent calls in the same
# event loop don't both hit the refresh endpoint. For multi-process safety
# we'd add a file-lock; a single daemon process plus the fact that calls
# always go through the LLMInterface mean asyncio.Lock is sufficient today.
_provider_locks: dict[str, asyncio.Lock] = {}


def _lock_for(provider: str) -> asyncio.Lock:
    lock = _provider_locks.get(provider)
    if lock is None:
        lock = asyncio.Lock()
        _provider_locks[provider] = lock
    return lock


async def refresh_access_token(
    grant: OAuthGrant,
    *,
    token_url: str,
    client_id: str,
    timeout_seconds: float = 30.0,
) -> OAuthGrant:
    """Exchange the stored refresh_token for a fresh access_token.

    Returns a NEW ``OAuthGrant`` with updated fields. Callers should
    persist it via :func:`save_grant` (or use :func:`with_auto_refresh`,
    which does that automatically).

    Raises ``predacore_sdk.AuthenticationError`` if the provider rejects
    the refresh — the caller should clear the stored grant and re-run
    the full PKCE flow (subscription expired, refresh-token revoked,
    client_id mismatch, etc.).
    """
    if not grant.refresh_token:
        raise psdk.AuthenticationError(
            f"oauth refresh: stored grant for {grant.provider!r} has no "
            "refresh_token — re-run `predacore login` to authorize again."
        )

    body = {
        "grant_type": "refresh_token",
        "refresh_token": grant.refresh_token,
        "client_id": client_id,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=10.0)
    ) as client:
        resp = await client.post(token_url, data=body, headers=headers)
        if resp.status_code in (400, 401, 403):
            # Stored refresh token is no good — surface as auth error so the
            # caller can prompt a fresh login.
            raise psdk.AuthenticationError(
                f"oauth refresh rejected ({resp.status_code}): "
                f"{resp.text[:300]}"
            )
        psdk.raise_for_status(resp)
        data = resp.json()

    expires_in = float(data.get("expires_in", 0) or 0)
    new_expires_at = (time.time() + expires_in) if expires_in > 0 else 0.0

    return OAuthGrant(
        provider=grant.provider,
        access_token=str(data.get("access_token", "")),
        # Some providers rotate the refresh_token on each exchange; honor
        # the new one when supplied, fall back to the old one when not.
        refresh_token=str(data.get("refresh_token") or grant.refresh_token),
        expires_at=new_expires_at,
        scope=str(data.get("scope") or grant.scope),
        account_id=grant.account_id,
        obtained_at=time.time(),
        client_id=client_id,
        extra=dict(grant.extra),
    )


@asynccontextmanager
async def with_auto_refresh(
    *,
    provider: str,
    token_url: str,
    client_id: str,
    safety_window_seconds: float = REFRESH_SAFETY_WINDOW_SECONDS,
    store: OAuthGrantStore | None = None,
) -> AsyncIterator[OAuthGrant]:
    """Yield a fresh-enough OAuthGrant. Refreshes + persists if needed.

    Usage::

        async with with_auto_refresh(
            provider="openai_codex",
            token_url="https://auth.openai.com/oauth/token",
            client_id=CODEX_CLIENT_ID,
        ) as grant:
            do_request_with(grant.access_token)

    Refresh path is serialized per-provider via ``asyncio.Lock`` so a
    burst of parallel chat() calls only refreshes once.
    """
    grant = load_grant(provider) if store is None else store.load(provider)
    if grant is None:
        raise psdk.AuthenticationError(
            f"oauth: no stored grant for {provider!r}. "
            f"Run `predacore login {provider.replace('_', '-')}` to authorize."
        )

    needs_refresh = (
        grant.expires_at > 0
        and (grant.expires_at - time.time()) < safety_window_seconds
    )
    if needs_refresh:
        async with _lock_for(provider):
            # Re-check after acquiring the lock — another waiter may have
            # already refreshed while we queued.
            grant = load_grant(provider) if store is None else store.load(provider)
            if grant is None:
                raise psdk.AuthenticationError(
                    f"oauth: grant for {provider!r} disappeared mid-refresh"
                )
            still_stale = (
                grant.expires_at > 0
                and (grant.expires_at - time.time()) < safety_window_seconds
            )
            if still_stale:
                logger.info(
                    "oauth: refreshing %s (expires in %.0fs, window=%.0fs)",
                    provider,
                    grant.expires_at - time.time(),
                    safety_window_seconds,
                )
                grant = await refresh_access_token(
                    grant,
                    token_url=token_url,
                    client_id=client_id,
                )
                if store is None:
                    save_grant(grant)
                else:
                    store.save(grant)
    yield grant


__all__ = [
    "REFRESH_SAFETY_WINDOW_SECONDS",
    "refresh_access_token",
    "with_auto_refresh",
]
