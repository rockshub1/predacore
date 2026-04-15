"""
Claude Dev Provider — Claude via Max subscription OAuth (DEV ONLY).

⚠  NOT FOR PUBLIC RELEASE. ⚠

This provider talks to Anthropic's API using the OAuth token stored in the
macOS Keychain by Claude Code. It identifies every request as Claude Code
traffic so Anthropic's server-side OAuth enforcement (effective 2026-04-04)
accepts it.

Uses JARVIS's in-house Prometheus SDK (``prometheus_sdk.py``) — zero
external vendor-SDK dependency. Shares the Anthropic Messages API wire
format (body builder, response parser, content_blocks round-trip) with
``anthropic.py`` via ``_anthropic_wire.py``. This file owns only the
OAuth-specific concerns: token loading, Claude Code identity block,
Claude Code User-Agent, and the OAuth beta flags.

Use cases:
  - Local JARVIS development against your own Max subscription quota
  - Testing extended-thinking / tool-use flows without burning API credit
  - Iterating on prompts before committing to API-key budget

Never ship this provider in the public release. The default provider for
end users is ``anthropic`` (API key from console.anthropic.com).

Auth source, in order:
  1. CLAUDE_CODE_OAUTH_TOKEN env var
  2. macOS Keychain entry "Claude Code-credentials" (claudeAiOauth.accessToken)

Requires a recent Claude Code login so the keychain token is unexpired. If
the token has expired, run ``claude`` once to refresh it.

Features (identical to ``anthropic.py`` — same wire helpers):
  - Extended thinking (adaptive on Opus 4.6 / Sonnet 4.6; auto-interleaved)
  - Effort parameter (``output_config.effort``: low/medium/high/max)
  - Prompt caching (top-level ephemeral + system breakpoint)
  - Server-side context compaction (``compact-2026-01-12`` beta)
  - Content-blocks round-trip (thinking signatures + tool_use ids survive)
  - Native tool use with text-based fallback
  - Retry with exponential backoff on 429/5xx/529 via Prometheus SDK
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Callable

from . import _anthropic_wire as wire
from . import prometheus_sdk as psdk
from .base import LLMProvider
from .claude_models import DEFAULT_CLAUDE_MODEL, resolve_claude_model
from .message_validator import repair_tool_flow
from .text_tool_adapter import build_tool_prompt, parse_tool_calls

logger = logging.getLogger(__name__)

# ── Claude Code identity — required by Anthropic OAuth enforcement ──
# Without this exact system prompt as the first system block, OAuth
# requests get hard-429'd server-side.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# Beta flags for OAuth traffic. Ordered by stability:
#   - oauth-2025-04-20       : OAuth enforcement flag
#   - claude-code-20250219   : identifies traffic as Claude Code
#   - compact-2026-01-12     : server-side context compaction
# Adaptive thinking auto-enables interleaved thinking on Opus 4.6 —
# no separate beta header needed for that anymore.
_BETA_FLAGS = (
    "oauth-2025-04-20,"
    "claude-code-20250219,"
    "compact-2026-01-12"
)

CLAUDE_CODE_USER_AGENT = "Claude-Code/2.1.98"

# Cache the keychain token across requests — avoid spawning `security` per call.
_cached_token: str | None = None
_cached_token_expires_at: float = 0.0


def _load_keychain_token() -> tuple[str, float] | None:
    """Read the Claude Code OAuth token from the macOS Keychain.

    Returns (token, expires_at_unix_seconds) or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("claude_dev: keychain lookup failed: %s", e)
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None

    entry = data.get("claudeAiOauth") or {}
    token = entry.get("accessToken")
    if not token:
        return None

    expires_ms = entry.get("expiresAt", 0)
    expires_at = float(expires_ms) / 1000.0 if expires_ms else 0.0
    return token, expires_at


def _get_oauth_token() -> str:
    """Return a live OAuth token from env or keychain, cached across calls."""
    global _cached_token, _cached_token_expires_at

    # Env var takes precedence (useful for CI or ephemeral overrides).
    env_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if env_token:
        return env_token

    now = time.time()
    # Cache hit: still valid with 60s grace
    if _cached_token and _cached_token_expires_at - now > 60:
        return _cached_token

    loaded = _load_keychain_token()
    if not loaded:
        raise psdk.AuthenticationError(
            "claude_dev: no OAuth token found. "
            "Either set CLAUDE_CODE_OAUTH_TOKEN env var, or run `claude` "
            "once to log in (stores token in macOS Keychain)."
        )

    token, expires_at = loaded
    if expires_at and expires_at <= now:
        raise psdk.AuthenticationError(
            "claude_dev: OAuth token expired. Run `claude` once to refresh it."
        )

    _cached_token = token
    _cached_token_expires_at = expires_at
    return token


class ClaudeDevProvider(LLMProvider):
    """Claude via Max-subscription OAuth — DEV ONLY.

    Do not enable this provider in public/user-facing deployments. Use the
    regular ``anthropic`` provider with an API key for anything that ships.
    """

    name = "claude-dev"

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion via OAuth to Anthropic's API.

        Mirrors AnthropicProvider's contract so the router can swap providers
        transparently. Identifies as Claude Code to satisfy OAuth enforcement.
        """
        token = _get_oauth_token()

        model = resolve_claude_model(self.config.model, default=DEFAULT_CLAUDE_MODEL)
        max_tok = (
            max_tokens if max_tokens is not None else self.config.max_tokens or 8192
        )

        conv_messages = wire.build_conv_messages(messages)
        if any(isinstance(m.get("content"), list) for m in conv_messages):
            conv_messages = repair_tool_flow(conv_messages)

        system_blocks = wire.build_system_blocks(
            wire.extract_system_text(messages),
            identity_prefix=CLAUDE_CODE_IDENTITY,  # OAuth path — required first block
        )

        body = wire.build_request_body(
            model=model,
            max_tok=max_tok,
            conv_messages=conv_messages,
            system_blocks=system_blocks,
            tools=tools,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort or "medium",
        )

        base_url = self.config.api_base or os.getenv("ANTHROPIC_BASE_URL", "")
        endpoint = (
            (base_url.rstrip("/") + "/v1/messages") if base_url else wire.ANTHROPIC_API_URL
        )
        headers = _build_headers(token)
        max_retries = int(os.getenv("JARVIS_API_MAX_RETRIES", "3"))

        async with psdk.make_client() as client:
            try:
                resp = await psdk.request_with_retry(
                    client,
                    "POST",
                    endpoint,
                    headers=headers,
                    json=body,
                    max_retries=max_retries,
                )
                return wire.parse_response(resp.json())
            except psdk.BadRequestError as exc:
                err_msg = str(exc).lower()
                is_tool_error = tools and ("tool" in err_msg or "function" in err_msg)
                if is_tool_error:
                    return await self._fallback_to_text_tools(
                        client, endpoint, headers, body, tools, max_retries,
                    )
                raise
            except psdk.LLMError as exc:
                logger.error("claude_dev error: %s", exc)
                raise

    async def _fallback_to_text_tools(
        self,
        client: Any,
        endpoint: str,
        headers: dict[str, str],
        body: dict,
        tools: list[dict],
        max_retries: int,
    ) -> dict[str, Any]:
        """Native tool API rejected — retry with text-based tools in the prompt.

        Appends the tool prompt to the last system block so the Claude Code
        identity stays intact as the first block.
        """
        logger.warning("claude_dev: native tools rejected — text tool fallback")

        body = dict(body)
        body.pop("tools", None)
        body.pop("tool_choice", None)

        tool_prompt = build_tool_prompt(tools)
        system = body.get("system", [])
        if isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                last["text"] = last.get("text", "") + tool_prompt
            else:
                system.append({"type": "text", "text": tool_prompt})
        else:
            body["system"] = [
                {"type": "text", "text": CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": tool_prompt},
            ]

        resp = await psdk.request_with_retry(
            client, "POST", endpoint, headers=headers, json=body, max_retries=max_retries,
        )
        result = wire.parse_response(resp.json())
        clean_text, text_tool_calls = parse_tool_calls(result["content"])
        if text_tool_calls:
            result["content"] = clean_text
            result["tool_calls"] = text_tool_calls
            result["finish_reason"] = "tool_calls"
        return result


def _build_headers(token: str) -> dict[str, str]:
    """OAuth auth headers for a Messages API call via Claude Code."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "anthropic-version": wire.ANTHROPIC_VERSION,
        "anthropic-beta": _BETA_FLAGS,
        "User-Agent": CLAUDE_CODE_USER_AGENT,
        "x-app": "cli",
    }
