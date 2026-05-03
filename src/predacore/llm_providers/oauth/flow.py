"""End-to-end PKCE OAuth flow orchestration.

Pulls together :mod:`pkce`, :mod:`callback`, and :mod:`store` into the
single function the CLI calls when the user runs ``predacore login
<provider>``:

  1. Generate PKCE pair
  2. Pick a free localhost port for the redirect
  3. Build the authorization URL
  4. Open the user's browser at that URL (``webbrowser.open``)
  5. Spin up the callback listener; wait for the redirect
  6. Validate the ``state`` (CSRF defense, also enforced by callback.py)
  7. Exchange the code + verifier at the token endpoint
  8. Persist the resulting grant

Errors throughout surface as ``predacore_sdk`` typed exceptions or a
short ``RuntimeError`` with a clear next-step message — the CLI handler
prints them verbatim.
"""
from __future__ import annotations

import logging
import secrets
import time
import webbrowser
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from .. import predacore_sdk as psdk
from .callback import build_redirect_uri, wait_for_authorization_code
from .pkce import generate_pkce_pair
from .store import OAuthGrant, save_grant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OAuthFlowConfig:
    """Per-provider OAuth flow configuration."""
    provider: str                           # short id used as filename + display
    authorization_url: str                  # e.g. https://auth.openai.com/oauth/authorize
    token_url: str                          # e.g. https://auth.openai.com/oauth/token
    client_id: str                          # provider-issued client id
    scopes: tuple[str, ...] = ()            # requested permissions; "" if N/A
    extra_auth_params: dict[str, str] = None  # type: ignore[assignment]

    # Redirect URI controls — most public PKCE clients allow any localhost
    # port, but some (Codex's app_EMoamEEZ73f0CkXaXp7hrann included)
    # registered fixed values that the strict OAuth matcher requires.
    # Defaults below pick a free ephemeral port at /callback. Override
    # per-provider when the client_id requires it.
    redirect_host: str = "127.0.0.1"        # what we ADVERTISE in the redirect_uri
    redirect_bind_host: str = "127.0.0.1"   # what we actually bind the listener to
    redirect_port: int = 0                  # 0 = pick free; non-zero = pinned
    redirect_path: str = "/callback"        # path the provider redirects to


class OAuthFlow:
    """One-shot orchestrator for a provider's PKCE-based login flow."""

    # Friendly delay so the success page can paint before the listener
    # tears its socket down — avoids "connection reset" in the user's
    # browser tab. Cosmetic; safe to skip in tests.
    _POST_CALLBACK_GRACE_SECONDS = 0.5

    def __init__(self, config: OAuthFlowConfig) -> None:
        self.config = config

    async def run(
        self,
        *,
        open_browser: bool = True,
        callback_timeout_seconds: float = 300.0,
    ) -> OAuthGrant:
        """Run the full flow and return the persisted grant.

        Raises ``RuntimeError`` if the user cancels / times out / the
        provider rejects the code, with a message intended for direct
        CLI display.
        """
        pkce = generate_pkce_pair()
        state = secrets.token_urlsafe(24)

        # Resolve the redirect URI. If config pins a port (Codex needs
        # 1455 for the OpenCode public client), use that; else pick free.
        from .callback import _pick_free_port  # local import — not exported
        port = self.config.redirect_port or _pick_free_port()
        redirect_uri = (
            f"http://{self.config.redirect_host}:{port}{self.config.redirect_path}"
        )

        auth_url = self._build_authorization_url(
            pkce_challenge=pkce.challenge,
            state=state,
            redirect_uri=redirect_uri,
        )

        logger.info("oauth login (%s): opening browser", self.config.provider)
        if open_browser:
            try:
                webbrowser.open(auth_url, new=1, autoraise=True)
            except Exception as exc:  # noqa: BLE001 — webbrowser swallows most
                logger.warning("oauth: could not auto-open browser: %s", exc)

        print(f"\n  Authorize at: {auth_url}\n")
        print(f"  Listening on: {redirect_uri}")
        print(f"  Waiting for redirect (up to {callback_timeout_seconds:.0f}s)...\n")

        result = await wait_for_authorization_code(
            expected_state=state,
            port=port,
            path=self.config.redirect_path,
            bind_host=self.config.redirect_bind_host,
            timeout_seconds=callback_timeout_seconds,
        )
        if not result.ok:
            detail = result.error_description or result.error or "unknown error"
            raise RuntimeError(
                f"oauth login ({self.config.provider}) failed: {detail}"
            )

        # Brief grace so the browser tab paints the success page.
        import asyncio
        await asyncio.sleep(self._POST_CALLBACK_GRACE_SECONDS)

        token_data = await self._exchange_code(
            code=result.code,
            verifier=pkce.verifier,
            redirect_uri=redirect_uri,
        )

        grant = self._grant_from_token_response(token_data)
        save_grant(grant)
        logger.info("oauth login (%s): grant saved", self.config.provider)
        return grant

    # ── internals ─────────────────────────────────────────────────────

    def _build_authorization_url(
        self,
        *,
        pkce_challenge: str,
        state: str,
        redirect_uri: str,
    ) -> str:
        params: dict[str, str] = {
            "client_id": self.config.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "code_challenge": pkce_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)
        if self.config.extra_auth_params:
            params.update(self.config.extra_auth_params)
        return f"{self.config.authorization_url}?{urlencode(params)}"

    async def _exchange_code(
        self,
        *,
        code: str,
        verifier: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """POST to the token endpoint to swap (code, verifier) for tokens."""
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.config.client_id,
            "code_verifier": verifier,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        ) as client:
            resp = await client.post(
                self.config.token_url, data=body, headers=headers,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"oauth token exchange failed ({resp.status_code}): "
                    f"{resp.text[:400]}"
                )
            psdk.raise_for_status(resp)
            return resp.json()

    def _grant_from_token_response(self, data: dict[str, Any]) -> OAuthGrant:
        """Translate the provider's token response into a stored grant."""
        access = str(data.get("access_token", ""))
        refresh = str(data.get("refresh_token", ""))
        expires_in = float(data.get("expires_in", 0) or 0)
        expires_at = (time.time() + expires_in) if expires_in > 0 else 0.0

        # Provider-specific extras land in ``grant.extra`` so future
        # subclasses can pull e.g. account_id / sub claims out without
        # round-tripping through dataclass mutation.
        extra: dict[str, Any] = {
            k: v for k, v in data.items()
            if k not in {
                "access_token", "refresh_token", "expires_in",
                "scope", "token_type",
            }
        }
        # Some providers (Codex specifically) embed an account_id in the
        # access_token JWT or in the response body. Try the body field
        # first; a more thorough subclass can decode the JWT if needed.
        account_id = str(data.get("account_id") or data.get("sub") or "")

        return OAuthGrant(
            provider=self.config.provider,
            access_token=access,
            refresh_token=refresh,
            expires_at=expires_at,
            scope=str(data.get("scope") or ""),
            account_id=account_id,
            client_id=self.config.client_id,
            extra=extra,
        )
