"""OpenAI Codex provider — ChatGPT subscription auth via OAuth + PKCE.

Uses OpenAI's Responses API (``/v1/responses``) — Chat Completions support
is being phased out for ChatGPT-OAuth tokens (announced 2026-Q1). The user
logs in once with ``predacore login openai-codex`` (PKCE pops a browser);
predacore stores the access + refresh tokens at
``~/.predacore/oauth/openai_codex.json`` and silently refreshes them
before they expire.

Important context (May 2026):

  • OpenAI explicitly tolerates third-party Codex OAuth use today —
    OpenClaw and OpenCode both ship the same pattern. This contrasts
    with Anthropic, who server-blocked third-party subscription OAuth
    on April 4 2026.
  • OpenAI could change this stance at any time. When the access_token
    starts returning 403 systematically, we surface a clear "your
    subscription auth was revoked — set OPENAI_API_KEY for guaranteed
    access" error rather than failing cryptically.
  • For users who want guaranteed-stable access (no provider policy
    change risk), the regular ``OpenAIProvider`` (env var API key) is
    always available. Codex OAuth is opt-in.

Default client_id is OpenCode's ``app_EMoamEEZ73f0CkXaXp7hrann`` — the
public PKCE client that OpenAI accepts for third-party tooling. Users
who register their own OpenAI OAuth client can override via
``PREDACORE_OPENAI_CODEX_CLIENT_ID``.

v1.5.0: Migrated from Chat Completions to the Responses API. Inherits
``OpenAIResponsesProvider`` for request/response/streaming logic; only
overrides auth (OAuth bearer + refresh-once-and-retry on 401).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from . import predacore_sdk as psdk
from .oauth import OAuthFlowConfig, with_auto_refresh
from .openai_responses import (
    RESPONSES_API_BASE_URL,
    RESPONSES_API_PATH,
    OpenAIResponsesProvider,
    _parse_responses,
    _stream_responses,
)

logger = logging.getLogger(__name__)


# Codex CLI (and OpenCode / OpenClaw) use these endpoints — confirmed via
# OpenAI's developer docs at developers.openai.com/codex/auth.
CODEX_AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Default PKCE-public client_id, override-able via env. OpenCode's client
# is the most-used public-PKCE app id today; using it makes us look like
# any other OAuth-based ChatGPT consumer in OpenAI's logs. Users who
# register their own OAuth app at platform.openai.com should set
# PREDACORE_OPENAI_CODEX_CLIENT_ID to that id.
DEFAULT_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


def codex_oauth_config() -> OAuthFlowConfig:
    """Return the OAuth flow config used by ``predacore login openai-codex``.

    Centralized so the CLI handler and the provider's refresh path agree
    on the client_id + endpoints.
    """
    client_id = (
        os.environ.get("PREDACORE_OPENAI_CODEX_CLIENT_ID", "").strip()
        or DEFAULT_CODEX_CLIENT_ID
    )
    return OAuthFlowConfig(
        provider="openai_codex",
        authorization_url=CODEX_AUTHORIZATION_URL,
        token_url=CODEX_TOKEN_URL,
        client_id=client_id,
        # Codex's PKCE-public client doesn't require explicit scopes — the
        # subscription's natural permissions apply. Leaving this empty
        # mirrors what OpenCode and OpenClaw send.
        scopes=(),
    )


class OpenAICodexProvider(OpenAIResponsesProvider):
    """OpenAI Responses API backed by a ChatGPT subscription OAuth grant.

    Inherits all wire-format logic (request body, response parser,
    streaming, tool serialization) from :class:`OpenAIResponsesProvider`.
    Only overrides ``chat()`` to swap in the OAuth Bearer token and
    handle auto-refresh + 401 retry.
    """

    name = "openai_codex"

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a Responses-API request authenticated via the stored OAuth grant.

        Refreshes the access token first if it's within 5 min of expiry.
        On a 401 mid-request, refreshes once more and retries (handles the
        narrow clock-skew window where the token expires between the
        ``with_auto_refresh`` check and the request landing).
        """
        flow_cfg = codex_oauth_config()

        # Auto-refresh (idempotent + async-lock-protected). Loads the
        # grant from disk, refreshes if stale, persists, returns.
        async with with_auto_refresh(
            provider="openai_codex",
            token_url=flow_cfg.token_url,
            client_id=flow_cfg.client_id,
        ) as grant:
            access_token = grant.access_token

        if not access_token:
            raise psdk.AuthenticationError(
                "openai_codex: stored grant has no access_token. "
                "Run `predacore login openai-codex` to re-authorize."
            )

        # Codex routes through the Responses API. The model id must be one
        # the user's subscription has access to — Codex CLI uses GPT-5-Codex
        # by default; older subscriptions may not have that. Fall back to
        # gpt-5 so first-time users get something working out of the box.
        model = self.config.model or "gpt-5-codex"
        base_url = (self.config.api_base or RESPONSES_API_BASE_URL).rstrip("/")
        url = f"{base_url}{RESPONSES_API_PATH}"
        headers = self._build_headers(access_token)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        wire_input = self._serialize_messages_for_responses(list(messages))
        wire_input = self.repair_messages(wire_input)

        body: dict[str, Any] = {
            "model": model,
            "input": wire_input,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_output_tokens": (
                max_tokens if max_tokens is not None else self.config.max_tokens
            ),
        }
        wire_tools = self._serialize_tools_for_responses(tools)
        if wire_tools:
            body["tools"] = wire_tools
            body["tool_choice"] = "auto"

        async with psdk.make_client() as client:
            # ── Streaming path (text-only, no tools) ──
            if stream_fn and not tools:
                try:
                    return await _stream_responses(client, url, headers, body, stream_fn)
                except psdk.BadRequestError as stream_err:
                    if stream_err.status_code not in (400, 422):
                        raise
                    logger.info(
                        "openai_codex: stream rejected (%d) — non-streaming retry",
                        stream_err.status_code,
                    )
                    body.pop("stream", None)

            # ── Non-streaming path ──
            try:
                resp = await psdk.request_with_retry(
                    client, "POST", url,
                    headers=headers, json=body, max_retries=max_retries,
                )
                return _parse_responses(resp.json())
            except psdk.AuthenticationError as exc:
                # 401 typically means the access_token expired BETWEEN
                # ``with_auto_refresh`` checking and this request landing
                # (clock skew / very narrow window). Refresh once more
                # explicitly and retry.
                logger.warning(
                    "openai_codex: 401 — refreshing once more and retrying"
                )
                from .oauth.refresh import refresh_access_token
                from .oauth.store import load_grant, save_grant
                grant = load_grant("openai_codex")
                if grant is None:
                    raise psdk.AuthenticationError(
                        "openai_codex: 401 + no stored grant. "
                        "Run `predacore login openai-codex` to re-authorize."
                    ) from exc
                grant = await refresh_access_token(
                    grant,
                    token_url=flow_cfg.token_url,
                    client_id=flow_cfg.client_id,
                )
                save_grant(grant)
                headers = self._build_headers(grant.access_token)
                resp = await psdk.request_with_retry(
                    client, "POST", url,
                    headers=headers, json=body, max_retries=max_retries,
                )
                return _parse_responses(resp.json())
            except psdk.PermissionDeniedError as exc:
                # 403 from /responses on a Codex token usually means OpenAI
                # has revoked subscription-OAuth use for our client_id
                # (Anthropic-precedent enforcement). Tell the user clearly
                # so they don't blame their network.
                raise psdk.PermissionDeniedError(
                    "openai_codex: 403 — OpenAI rejected the subscription "
                    "OAuth token. This may be the same enforcement Anthropic "
                    "rolled out (April 2026). Set OPENAI_API_KEY and switch "
                    "with `/model openai` for guaranteed-stable access.",
                    status_code=exc.status_code,
                    request_id=exc.request_id,
                    response_body=exc.response_body,
                ) from exc
            except psdk.BadRequestError as exc:
                # v1.5.0: text-tool fallback removed. Surface a clear error
                # instead of silently degrading.
                err_msg = str(exc).lower()
                if tools and ("tool" in err_msg or "function" in err_msg):
                    raise psdk.BadRequestError(
                        f"openai_codex: model '{model}' rejected native "
                        "tool calling. Try a different model id (gpt-5-codex, "
                        "gpt-5, gpt-4o).",
                        status_code=exc.status_code,
                        request_id=exc.request_id,
                        response_body=exc.response_body,
                    ) from exc
                raise
