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
from typing import Any, Callable

from . import predacore_sdk as psdk
from .base import LLMProvider
from .text_tool_adapter import build_tool_prompt, parse_tool_calls

logger = logging.getLogger(__name__)

# ── OpenAI-compatible provider endpoints ──────────────────────────────
# Each entry maps a provider name to its base URL, env var for API key,
# and default model.  All use the OpenAI chat/completions wire format.
PROVIDER_ENDPOINTS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
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

    def _resolve_endpoint(self) -> tuple[str, str, str]:
        """Resolve API key, base URL, and default model for the active provider.

        Returns (api_key, base_url, default_model). ``base_url`` is
        always non-empty so downstream code can concatenate paths.
        """
        variant = str(self.config.extras.get("provider", "openai"))
        ep = PROVIDER_ENDPOINTS.get(variant)

        if ep:
            # Use ONLY the provider-specific env var — never fall back to
            # config.api_key which belongs to the primary provider.
            api_key = os.getenv(ep["env_key"], "")
            if not api_key:
                logger.warning("No API key for %s (set %s)", variant, ep["env_key"])
            # Prefer an explicit override from config.api_base if set
            base_url = self.config.api_base or ep["base_url"]
            default_model = ep["default_model"]
        else:
            # Unknown provider — treat as generic OpenAI-compatible
            api_key = os.getenv("OPENAI_API_KEY", "") or self.config.api_key
            base_url = self.config.api_base or "https://api.openai.com/v1"
            default_model = "gpt-4o"

        return api_key, base_url.rstrip("/"), default_model

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
        if not api_key:
            raise psdk.AuthenticationError(
                f"No API key for provider '{self.config.extras.get('provider', 'openai')}'"
            )

        model = self.config.model or default_model
        url = f"{base_url}/chat/completions"
        headers = _build_headers(api_key)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
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
    """Standard OpenAI-compatible headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


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
