"""
PredaCore LLM Router — Multi-provider LLM interface with failover.

Orchestrates the failover chain and delegates execution to specific
provider implementations (plugins).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from ..config import PredaCoreConfig

# Import providers (direct API only — no vendor SDK dependencies)
from .anthropic import AnthropicProvider
from .base import LLMProvider, ProviderConfig
from .circuit_breaker import CircuitBreaker
from .gemini import GeminiProvider
from .gemini_cli import GeminiCLIProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)


def _normalize_provider_name(name: str | None) -> str:
    """Lowercase + strip a provider name."""
    return (name or "").strip().lower()

# ── Constants ─────────────────────────────────────────────────────────
RATE_LIMIT_RETRY_BACKOFF_BASE = 1.5   # Exponential backoff base for rate-limit retries
RATE_LIMIT_RETRY_MAX_DELAY = 10       # Cap (seconds) on retry delay for rate-limited calls


class LLMInterface:
    """
    Abstraction over different LLM providers.
    
    Responsibilities:
      1. Failover management (Primary -> Fallback 1 -> Fallback 2)
      2. Circuit breaking (Skip failing providers)
      3. Rate limit throttling (Adaptive delay)
      4. Delegation to provider plugins
    """

    # Circuit breaker settings
    CB_FAILURE_THRESHOLD = 3
    CB_WINDOW_SECONDS = 300
    CB_COOLDOWN_SECONDS = 120

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self._provider_name = _normalize_provider_name(config.llm.provider)

        # Build failover chain
        self._provider_chain = [self._provider_name]
        if config.llm.fallback_providers:
            for fb in config.llm.fallback_providers:
                normalized_fb = _normalize_provider_name(fb)
                if normalized_fb and normalized_fb not in self._provider_chain:
                    self._provider_chain.append(normalized_fb)

        # Initialize provider instances cache
        self._providers: dict[str, LLMProvider] = {}

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self.CB_FAILURE_THRESHOLD,
            window_seconds=self.CB_WINDOW_SECONDS,
            cooldown_seconds=self.CB_COOLDOWN_SECONDS,
        )

        # Adaptive throttling state (asyncio.Lock to avoid blocking the event loop)
        self._throttle_lock = asyncio.Lock()
        self._throttle_delay: float = 0.0
        self._last_call_ts: float = 0.0
        self._consecutive_calls: int = 0
        self._THROTTLE_RAPID_WINDOW: float = config.llm.throttle_rapid_window
        self._THROTTLE_RAMP_AFTER: int = config.llm.throttle_ramp_after
        self._THROTTLE_MIN_DELAY: float = config.llm.throttle_min_delay
        self._THROTTLE_MAX_DELAY: float = config.llm.throttle_max_delay
        self._THROTTLE_BACKOFF_FACTOR: float = 1.5

        logger.info(
            "LLM Router initialized. Chain: %s", " -> ".join(self._provider_chain)
        )

    def set_active_model(self, provider: str | None = None, model: str | None = None):
        """Hot-swap the active model/provider without full re-init."""
        if provider:
            provider = _normalize_provider_name(provider)
        if provider and provider != self._provider_name:
            self._provider_name = provider
            # Put new provider at front of chain
            self._provider_chain = [provider] + [
                p for p in self._provider_chain if p != provider
            ]
            # Clear cached instance so it reinitializes with new settings
            self._providers.pop(provider, None)
            logger.info("Switched primary provider to '%s'", provider)
        if model:
            self.config.llm.model = model
            # Clear cached provider so it picks up the new model
            self._providers.pop(self._provider_name, None)
            logger.info("Switched model to '%s'", model)

    def _get_provider_instance(self, name: str) -> LLMProvider:
        """Lazy-load provider instance."""
        if name in self._providers:
            return self._providers[name]

        from .openai import PROVIDER_ENDPOINTS

        is_primary = (name == self._provider_name)

        # For fallback providers, use their own default model/key/url
        # instead of the primary provider's config.
        if is_primary:
            model = self.config.llm.model
            api_key = self.config.llm.api_key
            api_base = self.config.llm.base_url
        else:
            # Fallback: let the provider resolve its own defaults
            ep = PROVIDER_ENDPOINTS.get(name, {})
            model = ep.get("default_model", "")  # empty = provider picks its own
            api_key = ""  # provider will read from its own env var
            api_base = ""  # provider will use its own endpoint

        extras: dict = {"provider": name}

        p_config = ProviderConfig(
            model=model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            api_key=api_key,
            api_base=api_base,
            home_dir=self.config.home_dir,
            extras=extras,
        )

        if name == "gemini":
            instance = GeminiProvider(p_config)
        elif name == "gemini-cli":
            instance = GeminiCLIProvider(p_config)
        elif name in ("anthropic", "claude"):
            instance = AnthropicProvider(p_config)
        elif name in ("openai-codex", "openai_codex", "codex"):
            # OpenAI subscription auth via PKCE OAuth — uses the
            # ChatGPT Plus/Pro plan instead of an API key. Tokens stored
            # at ~/.predacore/oauth/openai_codex.json after a one-time
            # `predacore login openai-codex` flow. See openai_codex.py
            # for the policy notes (currently tolerated by OpenAI).
            from .openai_codex import OpenAICodexProvider
            instance = OpenAICodexProvider(p_config)
        elif name in PROVIDER_ENDPOINTS or name in ("azure",):
            instance = OpenAIProvider(p_config)
        else:
            logger.warning("Unknown provider %r, falling back to OpenAI-compatible", name)
            instance = OpenAIProvider(p_config)

        self._providers[name] = instance
        return instance

    @property
    def auto_fallback(self) -> bool:
        """Whether to silently fall back to next provider on failure."""
        return getattr(self.config.llm, "auto_fallback", False)

    @property
    def active_provider(self) -> str:
        """Currently active provider name."""
        return self._provider_name

    @property
    def active_provider_instance(self):
        """Currently active LLMProvider instance.

        Used by ``PredaCoreCore`` to call provider-owned tool-turn serializers
        (``append_assistant_turn`` / ``append_tool_results_turn``). Lazily
        instantiates the provider if it hasn't been used yet.
        """
        return self._get_provider_instance(self._provider_name)

    @property
    def active_model(self) -> str:
        """Currently active model name."""
        return self.config.llm.model or "default"

    @property
    def executes_tool_loop_internally(self) -> bool:
        """Whether the active provider owns tool execution inside its runtime."""
        provider = self._get_provider_instance(self._provider_name)
        return bool(getattr(provider, "executes_tool_loop_internally", False))

    @property
    def available_providers(self) -> list[str]:
        """All providers in the failover chain."""
        return list(self._provider_chain)

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Send chat request.

        If auto_fallback is OFF (default), rate limits return a user-facing
        message asking them to switch models via /model command.  No silent
        fallback to weaker models.

        If auto_fallback is ON, uses the old failover chain behavior.
        """
        # Phase D: local response idempotency cache (opt-in via
        # PREDACORE_IDEMPOTENT=1). Only hits for temperature=0.0 — anything
        # else is non-deterministic and must not serve cached responses.
        # Keyed by (provider, model, messages, tools, temperature) — works
        # across ALL providers transparently.
        from .response_cache import get_shared_cache, is_enabled as _cache_enabled

        _effective_temp = (
            temperature if temperature is not None else self.config.llm.temperature
        )
        _cache = None
        if _cache_enabled() and float(_effective_temp) == 0.0:
            _cache = get_shared_cache()
            _cached = _cache.get(
                provider=self._provider_name,
                model=self.config.llm.model or "",
                messages=messages,
                tools=tools,
                temperature=_effective_temp,
            )
            if _cached is not None:
                _cached_copy = dict(_cached)
                _cached_copy["_response_cache_hit"] = True
                logger.info("response cache HIT (provider=%s)", self._provider_name)
                return _cached_copy

        await self._apply_throttle()

        errors = []
        max_retries = self.config.llm.max_retries

        for provider_name in self._provider_chain:
            if await self._circuit_breaker.is_open(provider_name):
                # If auto_fallback is off and this is the primary, tell user
                if not self.auto_fallback and provider_name == self._provider_name:
                    return self._rate_limit_response(provider_name)
                logger.debug("Skipping %s (circuit open)", provider_name)
                continue

            provider = self._get_provider_instance(provider_name)

            if tools and not getattr(provider, 'supports_native_tools', True):
                logger.warning(
                    "Provider %s doesn't support native tools; using text-based tool format",
                    provider_name,
                )

            # Retry loop for this provider
            for attempt in range(max_retries + 1):
                try:
                    start = time.time()
                    result = await provider.chat(
                        messages, tools, temperature, max_tokens,
                        stream_fn=stream_fn,
                    )
                    latency = (time.time() - start) * 1000

                    result["provider_used"] = provider_name
                    result["latency_ms"] = round(latency, 1)

                    if provider_name != self._provider_chain[0]:
                        logger.info("Failover: used '%s' (%.0fms)", provider_name, latency)

                    # Phase D: store successful response in idempotency cache
                    # (temperature=0 only; cache is a no-op otherwise).
                    if _cache is not None:
                        try:
                            _cache.set(
                                provider=provider_name,
                                model=self.config.llm.model or "",
                                messages=messages,
                                tools=tools,
                                temperature=_effective_temp,
                                response=result,
                            )
                        except Exception as _set_err:  # noqa: BLE001
                            logger.debug("response cache set() non-fatal: %s", _set_err)

                    return result

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "rate limit" in err_str
                        or "rate_limit" in err_str
                        or "too many requests" in err_str
                        or getattr(e, "status_code", 0) == 429
                    )

                    if is_rate_limit:
                        if attempt < max_retries:
                            delay = min(RATE_LIMIT_RETRY_BACKOFF_BASE ** attempt, RATE_LIMIT_RETRY_MAX_DELAY)
                            logger.warning("%s rate limit, retry %d in %.1fs", provider_name, attempt+1, delay)
                            await asyncio.sleep(delay)
                            continue

                        # Exhausted retries for this provider
                        if not self.auto_fallback:
                            # Don't silently fall back — tell the user
                            await self._circuit_breaker.record_failure(provider_name, e)
                            return self._rate_limit_response(provider_name)

                    # Not retryable or exhausted (with auto_fallback on)
                    await self._circuit_breaker.record_failure(provider_name, e)
                    errors.append(f"{provider_name}: {e}")

                    # If auto_fallback is off, stop after primary provider fails
                    # and show the actual error (not fake "rate limited")
                    if not self.auto_fallback and provider_name == self._provider_name:
                        return self._provider_error_response(provider_name, e)
                    break

            # If auto_fallback is off and primary circuit is open, tell user
            if not self.auto_fallback and provider_name == self._provider_name:
                return self._rate_limit_response(provider_name)

        raise ConnectionError(f"All providers failed. Chain: {self._provider_chain}. Errors: {'; '.join(errors)}")

    def _rate_limit_response(self, provider_name: str) -> dict[str, Any]:
        """Build a user-facing rate limit message instead of silent fallback."""
        fallbacks = [p for p in self._provider_chain if p != provider_name]
        fallback_hint = ""
        if fallbacks:
            fallback_hint = (
                "\n\nAvailable alternatives:\n"
                + "\n".join(f"  /model {fb}" for fb in fallbacks)
                + "\n\nSwitch with /model <provider> — I'll use that until you switch back."
            )

        content = (
            f"Rate-limited on **{self.active_model}**. "
            f"I'd rather wait than switch without telling you."
            f"{fallback_hint}"
            f"\n\nOr just wait a minute and try again."
        )

        logger.warning(
            "Rate limited on %s — asking user to switch (auto_fallback=off)",
            provider_name,
        )

        return {
            "content": content,
            "tool_calls": [],
            "finish_reason": "rate_limit",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "provider_used": provider_name,
            "latency_ms": 0,
            "rate_limited": True,
        }

    def _provider_error_response(self, provider_name: str, error: Exception) -> dict[str, Any]:
        """Build a user-facing error message for non-rate-limit provider failures."""
        err_str = str(error)
        # Truncate long error messages
        if len(err_str) > 200:
            err_str = err_str[:200] + "..."

        fallbacks = [p for p in self._provider_chain if p != provider_name]
        fallback_hint = ""
        if fallbacks:
            fallback_hint = (
                "\n\nAvailable alternatives:\n"
                + "\n".join(f"  /model {fb}" for fb in fallbacks)
                + "\n\nSwitch with /model <provider>"
            )

        content = (
            f"**{provider_name}** failed: {err_str}"
            f"{fallback_hint}"
            f"\n\nTry again, or switch providers."
        )

        logger.error(
            "Provider %s error (auto_fallback=off): %s",
            provider_name, err_str,
        )

        return {
            "content": content,
            "tool_calls": [],
            "finish_reason": "error",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "provider_used": provider_name,
            "latency_ms": 0,
            "rate_limited": False,
        }

    async def _apply_throttle(self) -> None:
        """Adaptive throttle to prevent 429s during loops."""
        now = time.time()
        async with self._throttle_lock:
            gap = now - self._last_call_ts
            if gap < self._THROTTLE_RAPID_WINDOW:
                self._consecutive_calls += 1
            else:
                self._consecutive_calls = 1
                self._throttle_delay = 0.0

            if self._consecutive_calls > self._THROTTLE_RAMP_AFTER:
                # Ramp up delay
                if self._throttle_delay == 0.0:
                    self._throttle_delay = self._THROTTLE_MIN_DELAY
                else:
                    self._throttle_delay = min(
                        self._throttle_delay * self._THROTTLE_BACKOFF_FACTOR,
                        self._THROTTLE_MAX_DELAY
                    )

                logger.info("LLM Throttle: sleeping %.1fs", self._throttle_delay)
                await asyncio.sleep(self._throttle_delay)

            self._last_call_ts = time.time()
