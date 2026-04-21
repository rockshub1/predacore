"""
OpenAI Provider — OpenAI and all OpenAI-compatible APIs via direct HTTP.

Uses PredaCore's in-house PredaCore SDK (``predacore_sdk.py``) — zero
external vendor-SDK dependency. POSTs directly to
``<base_url>/chat/completions`` on the active provider.

Handles:
  - Standard OpenAI Chat Completions
  - Azure OpenAI deployment routing
  - Groq, xAI/Grok, DeepSeek, Cerebras, Together, OpenRouter, SambaNova,
    Mistral, Fireworks, NVIDIA NIM, Zhipu — all OpenAI-compatible
  - Streaming (SSE via httpx.aiter_lines)
  - Native tool use + text-based fallback for models that lack it
  - ``<think>...</think>`` stripping for reasoning models (DeepSeek R1,
    Qwen3, etc.)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from . import predacore_sdk as psdk
from .base import LLMProvider
from .text_tool_adapter import build_tool_prompt, parse_tool_calls

logger = logging.getLogger(__name__)

# ── OpenAI-compatible provider endpoints ──────────────────────────────
# Each entry maps a provider name to its base URL, env var for API key,
# and default model. All use the OpenAI chat/completions wire format.
# ``env_key`` may be None for providers that don't require auth (e.g. local
# Ollama); such providers skip the Authorization header and the no-key check.
PROVIDER_ENDPOINTS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,  # Ollama is local, no auth required
        "default_model": "llama3.2:3b",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "default_model": "grok-3-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "default_model": "llama-3.3-70b",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "env_key": "SAMBANOVA_API_KEY",
        "default_model": "DeepSeek-V3.1",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
        "default_model": "glm-4-plus",
    },
}


class OpenAIProvider(LLMProvider):
    """OpenAI and compatible API provider.

    Uses the PredaCore SDK for HTTP + retry + typed errors. One code
    path covers all 11 OpenAI-compatible providers — only the base URL,
    API key env var, and default model differ per variant.
    """

    name = "openai"

    # ------------------------------------------------------------------
    # Tool-turn serialization (Phase A refactor — 2026-04-21).
    #
    # OpenAI's wire format for tool round-trips is already what the default
    # ``append_*_turn`` methods on LLMProvider produce at the abstract level
    # (``role="assistant"`` with ``tool_calls``; one ``role="tool"`` Message
    # per result). So we inherit both defaults unchanged.
    #
    # The ONLY twist is that the default emits our abstract dict shape:
    #
    #     {"role": "assistant", "content": "...",
    #      "tool_calls": [{"id": X, "name": Y, "arguments": {...}}]}
    #
    # while OpenAI's API expects its native nested shape:
    #
    #     {"role": "assistant", "content": "...",
    #      "tool_calls": [{"id": X, "type": "function",
    #                      "function": {"name": Y, "arguments": "<json-str>"}}]}
    #
    # ``_serialize_messages_for_wire`` is called inside ``chat()`` to do the
    # translation. Old-format dicts (legacy callers not yet migrated) are
    # passed through unchanged so the migration can happen incrementally.
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_messages_for_wire(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate abstract-format messages to OpenAI wire format.

        Accepts a mix of legacy-format dicts (plain role/content, no
        ``tool_calls``/``tool_results`` keys) and new abstract-format dicts
        produced by ``types.message_to_dict``. Legacy dicts pass through
        unchanged; abstract dicts are rewritten to OpenAI's nested shape.
        """
        import json

        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")

            # Abstract assistant turn with tool_calls → OpenAI assistant shape
            if role == "assistant" and m.get("tool_calls"):
                wire_tool_calls = []
                for tc in m["tool_calls"]:
                    args = tc.get("arguments", {})
                    args_str = args if isinstance(args, str) else json.dumps(args)
                    wire_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": args_str,
                            },
                        }
                    )
                wire_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": wire_tool_calls,
                }
                out.append(wire_msg)
                continue

            # Abstract tool turn → OpenAI role="tool" with tool_call_id
            if role == "tool" and m.get("tool_results"):
                for tr in m["tool_results"]:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("call_id", ""),
                            "content": tr.get("result", ""),
                        }
                    )
                continue

            # Legacy or non-tool turn — pass through, but strip our abstract
            # keys so they don't leak to the API.
            passthrough = {k: v for k, v in m.items() if k not in ("tool_calls", "tool_results", "content_blocks")}
            # Preserve content-only assistant/tool turns from legacy callers
            if "tool_calls" in m and not m.get("tool_calls"):
                pass  # empty list — no-op
            out.append(passthrough)

        return out

    def _resolve_endpoint(self) -> tuple[str, str, str]:
        """Resolve API key, base URL, and default model for the active provider.

        Returns (api_key, base_url, default_model). ``base_url`` is
        always non-empty so downstream code can concatenate paths. For
        no-auth providers (ollama), ``api_key`` is an empty string and
        the caller should skip the auth check.
        """
        variant = str(self.config.extras.get("provider", "openai"))
        ep = PROVIDER_ENDPOINTS.get(variant)

        if ep:
            env_key = ep.get("env_key")
            if env_key:
                # Provider-specific env var takes precedence. Fall back to
                # config.api_key so users can put the key in ~/.predacore/
                # config.yaml when their env isn't available (e.g. running
                # from a launchd plist without a login shell).
                api_key = os.getenv(env_key, "") or self.config.api_key
                if not api_key:
                    logger.warning("No API key for %s (set %s)", variant, env_key)
            else:
                # Provider doesn't require auth (e.g. ollama).
                api_key = ""
            # Prefer an explicit override from config.api_base if set
            base_url = self.config.api_base or ep["base_url"]
            default_model = ep["default_model"]
        else:
            # Unknown provider — treat as generic OpenAI-compatible
            api_key = os.getenv("OPENAI_API_KEY", "") or self.config.api_key
            base_url = self.config.api_base or "https://api.openai.com/v1"
            default_model = "gpt-4o"

        return api_key, base_url.rstrip("/"), default_model

    @property
    def _requires_api_key(self) -> bool:
        """Whether the active variant needs auth. False for ollama and friends."""
        variant = str(self.config.extras.get("provider", "openai"))
        ep = PROVIDER_ENDPOINTS.get(variant)
        if ep is None:
            return True  # Unknown provider → assume auth required
        return bool(ep.get("env_key"))

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to OpenAI or a compatible API."""
        api_key, base_url, default_model = self._resolve_endpoint()
        variant = str(self.config.extras.get("provider", "openai"))
        if not api_key and self._requires_api_key:
            ep = PROVIDER_ENDPOINTS.get(variant, {})
            env_var = ep.get("env_key", "OPENAI_API_KEY")
            raise psdk.AuthenticationError(
                f"{variant}: missing API key. "
                f"Set ${env_var} or add api_key to ~/.predacore/config.yaml."
            )

        model = self.config.model or default_model
        url = f"{base_url}/chat/completions"
        headers = _build_headers(api_key)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        body: dict[str, Any] = {
            "model": model,
            "messages": self._serialize_messages_for_wire(list(messages)),
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else self.config.max_tokens
            ),
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"

        async with psdk.make_client() as client:
            # ── Streaming path (text-only, no tools) ──
            if stream_fn and not tools:
                try:
                    return await _stream_chat(client, url, headers, body, stream_fn)
                except psdk.BadRequestError as stream_err:
                    # Only fall back if it's a streaming-specific rejection.
                    # Propagate auth/server errors — those shouldn't retry.
                    if stream_err.status_code not in (400, 422):
                        raise
                    logger.info(
                        "stream rejected (%d) — falling back to non-streaming",
                        stream_err.status_code,
                    )
                    body.pop("stream", None)
                    body.pop("stream_options", None)

            # ── Non-streaming path ──
            try:
                resp = await psdk.request_with_retry(
                    client,
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                    max_retries=max_retries,
                )
                return _parse_response(resp.json())
            except psdk.BadRequestError as exc:
                err_msg = str(exc).lower()
                is_tool_unsupported = (
                    tools
                    and (
                        ("not supported" in err_msg and ("tool" in err_msg or "function" in err_msg))
                        or "does not support tools" in err_msg
                        or "does not support function" in err_msg
                    )
                )
                if is_tool_unsupported:
                    logger.warning(
                        "Model '%s' doesn't support native tool use — text-tool fallback",
                        model,
                    )
                    return await _fallback_text_tools(
                        client, url, headers, body, tools, max_retries,
                    )
                raise


# ---------------------------------------------------------------------------
# Module-level helpers (stateless)
# ---------------------------------------------------------------------------


def _build_headers(api_key: str) -> dict[str, str]:
    """Standard OpenAI-compatible headers.

    Omits the Authorization header entirely when ``api_key`` is empty, so
    no-auth providers like local Ollama don't send a dangling
    ``Authorization: Bearer`` that some HTTP stacks reject.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _stream_chat(
    client: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    stream_fn: Callable,
) -> dict[str, Any]:
    """Streaming path — SSE via httpx.aiter_lines, no retry (one-shot).

    Accumulates tokens, strips ``<think>...</think>`` tags for reasoning
    models, and calls ``stream_fn`` per chunk. Returns the same response
    shape as the non-streaming path.
    """
    body = dict(body)
    body["stream"] = True
    # Some providers only emit usage in the final chunk when this is set
    body["stream_options"] = {"include_usage": True}

    full_content = ""
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0

    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            # Read the body so psdk.raise_for_status can include it
            await resp.aread()
            psdk.raise_for_status(resp)

        async for line in resp.aiter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                if data_str == "[DONE]":
                    break
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    full_content += token
                    try:
                        result = stream_fn(token)
                        if asyncio.iscoroutine(result):
                            await result
                    except (TypeError, ValueError, OSError):
                        pass  # Don't let callback errors kill the stream
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0

    # Strip <think> tags from streamed content (DeepSeek R1, Qwen3, etc.)
    if "</think>" in full_content:
        full_content = full_content.split("</think>")[-1].strip()

    return {
        "content": full_content,
        "tool_calls": [],
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


async def _fallback_text_tools(
    client: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    tools: list[dict],
    max_retries: int,
) -> dict[str, Any]:
    """Strip native tools, embed them as text in the system prompt, retry."""
    body = dict(body)
    body.pop("tools", None)
    body.pop("tool_choice", None)

    tool_prompt = build_tool_prompt(tools)
    if tool_prompt:
        msgs = list(body["messages"])
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                msgs[i] = {**m, "content": (m.get("content") or "") + tool_prompt}
                break
        else:
            msgs.insert(0, {"role": "system", "content": tool_prompt})
        body["messages"] = msgs

    resp = await psdk.request_with_retry(
        client, "POST", url, headers=headers, json=body, max_retries=max_retries,
    )
    result = _parse_response(resp.json())
    clean_text, text_tool_calls = parse_tool_calls(result["content"])
    if text_tool_calls:
        result["content"] = clean_text
        result["tool_calls"] = text_tool_calls
        result["finish_reason"] = "tool_calls"
    return result


def _parse_response(data: dict) -> dict[str, Any]:
    """Parse OpenAI /chat/completions response into the router's standard shape."""
    choices = data.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""

    # Strip <think> tags from thinking models (DeepSeek R1, Qwen3, etc.)
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    tool_calls: list[dict] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments") or ""
        try:
            args = json.loads(args_raw) if args_raw else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            }
        )

    # If no native tool_calls but content has <tool_call> text, parse that
    if not tool_calls:
        clean_text, text_tool_calls = parse_tool_calls(content)
        if text_tool_calls:
            content = clean_text
            tool_calls = text_tool_calls

    usage = data.get("usage") or {}
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": (
            "tool_calls" if tool_calls else choice.get("finish_reason") or "stop"
        ),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
    }
