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

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from . import predacore_sdk as psdk
from .oauth import OAuthFlowConfig, with_auto_refresh
from .oauth.flow import _extract_chatgpt_account_id
from .openai_responses import (
    OpenAIResponsesProvider,
    _parse_responses,
)


# Codex OAuth tokens route through chatgpt.com's backend, NOT through
# api.openai.com. The official Codex CLI hits this exact path; the JWT
# we get from auth.openai.com/oauth/token has audience
# `api.openai.com/v1` but only the chatgpt.com codex endpoint actually
# accepts subscription-OAuth tokens. api.openai.com endpoints expect
# API-tier billing keys, not OAuth.
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _split_system_instructions(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Pull system messages out into a single ``instructions`` string.

    chatgpt.com's codex backend wants the system prompt at the top level
    of the request body (as ``instructions``), not as a ``role:"system"``
    item inside ``input[]``. Sending a system-role item returns
    HTTP 400 ``"Instructions are required"``.

    Concatenates every system message in order (rare to have more than
    one, but supported) and returns the conversation messages without
    them.
    """
    sys_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                sys_parts.append(content)
            continue
        rest.append(m)
    return "\n\n".join(sys_parts), rest

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
        # OpenAI's strict OAuth matcher requires the EXACT redirect_uri
        # registered for this client_id. OpenCode's public PKCE client
        # `app_EMoamEEZ73f0CkXaXp7hrann` registered:
        #   http://localhost:1455/auth/callback
        # We bind on 127.0.0.1 (loopback equivalent) but advertise
        # `localhost` in the redirect_uri to match registration exactly.
        # If port 1455 is busy on the user's box (e.g. an existing
        # OpenCode/Codex CLI listener), the login will fail loudly with
        # an OSError — they need to free the port and retry.
        redirect_host="localhost",
        redirect_bind_host="127.0.0.1",
        redirect_port=1455,
        redirect_path="/auth/callback",
    )


class OpenAICodexProvider(OpenAIResponsesProvider):
    """OpenAI Responses API backed by a ChatGPT subscription OAuth grant.

    Inherits all wire-format logic (request body, response parser,
    streaming, tool serialization) from :class:`OpenAIResponsesProvider`.
    Only overrides ``chat()`` to swap in the OAuth Bearer token and
    handle auto-refresh + 401 retry.
    """

    name = "openai_codex"

    @staticmethod
    def _build_codex_headers(
        access_token: str,
        account_id: str,
    ) -> dict[str, str]:
        """Build the headers Codex's /v1/responses requires.

        OpenAI's strict matcher rejects a Bearer-only request from a
        Codex-OAuth token; the `chatgpt-account-id` header tells the
        backend which workspace this token belongs to (the JWT contains
        the user_id but the API needs the account_id). The `OpenAI-Beta`
        header puts the request on the Codex-tuned routing path that the
        OAuth token is authorized for.
        """
        h = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "OpenAI-Beta": "chatgpt-codex",
            # The originator string mirrors what the official Codex CLI
            # emits — some upstream rate limits + telemetry slot us into
            # the right bucket only when this is set.
            "Originator": "predacore_cli",
        }
        if account_id:
            h["chatgpt-account-id"] = account_id
        return h

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
            account_id = grant.account_id

        if not access_token:
            raise psdk.AuthenticationError(
                "openai_codex: stored grant has no access_token. "
                "Run `predacore login openai-codex` to re-authorize."
            )

        # Older grants (pre-v1.5.2) were saved without account_id because
        # the JWT-claim extraction landed later. Re-derive on the fly so
        # users don't have to re-login just to pick up the header fix.
        if not account_id:
            account_id = _extract_chatgpt_account_id(access_token)

        # Codex routes through chatgpt.com's backend. The model id must be
        # one the user's subscription has access to — Codex CLI uses
        # GPT-5-Codex by default; older subscriptions may not have that.
        # Fall back to gpt-5 so first-time users get something working
        # out of the box.
        model = self.config.model or "gpt-5-codex"
        url = self.config.api_base.rstrip("/") if self.config.api_base else CODEX_RESPONSES_URL
        headers = self._build_codex_headers(access_token, account_id)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        # chatgpt.com/backend-api/codex/responses diverges from
        # api.openai.com/v1/responses: system prompts go in a top-level
        # `instructions` string, NOT as role:"system" items in input[].
        # Sending a system-typed message item returns:
        #   400 {"detail":"Instructions are required"}
        # So we split here.
        instructions, conv_messages = _split_system_instructions(list(messages))
        wire_input = self._serialize_messages_for_responses(conv_messages)
        wire_input = self.repair_messages(wire_input)

        # chatgpt.com Codex backend is opinionated about which knobs callers
        # are allowed to set — it 400s on `temperature`, `top_p`, and
        # similar sampling params (the model picks its own). API-tier
        # OpenAI Responses accepts them; Codex doesn't. We deliberately
        # omit them here rather than attempt to pass through.
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions or "You are a helpful coding assistant.",
            "input": wire_input,
            # Codex/ChatGPT-OAuth tier doesn't permit server-side response
            # storage. Backend returns 400 ``"Store must be set to false"``
            # if this is omitted (default true on api.openai.com) or set
            # to true. API-tier callers via OpenAIResponsesProvider can
            # still set store=true themselves.
            "store": False,
            # NOTE: v1.5.4 tried adding ``truncation: "auto"`` here on
            # advice from research that Codex CLI sends it. Backend
            # turns out to reject it: 400 ``"Unsupported parameter:
            # truncation"``. v1.5.7 removed it. Long-context overflow
            # behavior on the chatgpt.com codex endpoint is implicit —
            # the server handles it without a client-side knob.
        }
        # max_output_tokens is accepted only on some Codex models; passing
        # it on others 400s. Send only when the caller explicitly asked.
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        wire_tools = self._serialize_tools_for_responses(tools)
        if wire_tools:
            body["tools"] = wire_tools
            body["tool_choice"] = "auto"

        # chatgpt.com/backend-api/codex/responses requires `stream: true`
        # ALWAYS — even when the caller didn't ask for streaming. Backend
        # returns 400 ``"Stream must be set to true"`` otherwise. We
        # stream regardless and aggregate the SSE events into the same
        # dict shape ``_parse_responses`` would have returned for the
        # non-streamed case.
        body["stream"] = True

        async with psdk.make_client() as client:
            try:
                return await _codex_stream_chat(client, url, headers, body, stream_fn)
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
                refreshed_account_id = (
                    grant.account_id
                    or _extract_chatgpt_account_id(grant.access_token)
                )
                headers = self._build_codex_headers(
                    grant.access_token, refreshed_account_id
                )
                return await _codex_stream_chat(
                    client, url, headers, body, stream_fn,
                )
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


# ---------------------------------------------------------------------------
# Codex SSE stream consumer
# ---------------------------------------------------------------------------


async def _codex_stream_chat(
    client: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    stream_fn: Callable | None,
) -> dict[str, Any]:
    """Stream Codex's /responses endpoint and aggregate the result.

    Codex's chatgpt.com backend mandates ``stream: true`` on every
    request, so we always stream regardless of whether the caller
    provided a ``stream_fn`` callback. We listen for these SSE events:

      response.created             — initial response object
      response.output_item.added   — new output item (message / function_call / reasoning)
      response.output_text.delta   — text delta inside a message item
      response.function_call_arguments.delta — partial JSON args for a function_call
      response.completed           — final response with full output[]
      response.failed / .incomplete — terminal error states

    On ``response.completed`` we hand the final output[] off to the
    same ``_parse_responses`` that the non-streaming path uses, so the
    return shape (content / tool_calls / usage / finish_reason) is
    identical to every other provider. If ``stream_fn`` is given, we
    invoke it on each text delta as it arrives.
    """
    final_response: dict[str, Any] | None = None
    error_detail: dict[str, Any] | None = None
    full_text = ""
    # Accumulate function-call args by output index — Codex streams
    # function_call_arguments as small deltas before emitting the full
    # call. We collect here so the final aggregation has them even if
    # final_response.output[] is sparse.
    fn_args_accum: dict[int, dict[str, str]] = {}

    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            psdk.raise_for_status(resp)

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                if data_str == "[DONE]":
                    break
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type") or ""

            if etype == "response.output_text.delta":
                delta = event.get("delta") or ""
                if delta:
                    full_text += delta
                    if stream_fn:
                        try:
                            r = stream_fn(delta)
                            if asyncio.iscoroutine(r):
                                await r
                        except (TypeError, ValueError, OSError):
                            pass  # callback errors don't kill the stream

            elif etype == "response.output_item.added":
                # New output item — capture function_call name + call_id
                # so the aggregator below has them even if the final
                # response.output[] doesn't.
                item = event.get("item") or {}
                idx = event.get("output_index", 0)
                if item.get("type") == "function_call":
                    fn_args_accum[idx] = {
                        "name": str(item.get("name", "")),
                        "call_id": str(item.get("call_id", "")),
                        "arguments": str(item.get("arguments", "") or ""),
                    }

            elif etype == "response.function_call_arguments.delta":
                idx = event.get("output_index", 0)
                slot = fn_args_accum.setdefault(idx, {"name": "", "call_id": "", "arguments": ""})
                slot["arguments"] += str(event.get("delta") or "")

            elif etype == "response.completed":
                final_response = event.get("response") or {}

            elif etype in ("response.failed", "response.incomplete"):
                resp_obj = event.get("response") or {}
                err = resp_obj.get("error") or {}
                error_detail = {
                    "type": etype,
                    "message": err.get("message")
                              or resp_obj.get("incomplete_details", {}).get("reason")
                              or "unknown",
                    "code": err.get("code"),
                }

    if error_detail is not None:
        raise psdk.APIStatusError(
            f"openai_codex: {error_detail['type']} — {error_detail['message']}",
            status_code=500,
            response_body=json.dumps(error_detail),
        )

    if final_response is None:
        # No completion event arrived. Treat as a connection-level failure
        # so the upstream retry logic kicks in.
        raise psdk.APIConnectionError(
            "openai_codex: stream ended without response.completed event"
        )

    # Aggregate. Prefer the final response's output[] when populated; fall
    # back to deltas accumulated mid-stream for content + function calls
    # (Codex sometimes ships the final response with output:[] because all
    # the substance was already streamed via delta events).
    parsed = _parse_responses(final_response)
    if not parsed.get("content") and full_text:
        parsed["content"] = full_text
    if not parsed.get("tool_calls") and fn_args_accum:
        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(fn_args_accum.keys()):
            slot = fn_args_accum[idx]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {"id": slot["call_id"], "name": slot["name"], "arguments": args}
            )
        parsed["tool_calls"] = tool_calls
        parsed["finish_reason"] = "tool_calls"
    return parsed
