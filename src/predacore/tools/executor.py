"""
PredaCore Tool Executor — thin facade over focused subsystem modules.

Phase 6.3 refactoring: marketplace code extracted to marketplace.py.
Executor is now a pure coordination layer (~250 lines vs original ~900).

Architecture:
  - SubsystemFactory       → initializes desktop, memory, MCTS, voice, etc.
  - TrustPolicyEvaluator   → confirmation / block logic
  - ToolDispatcher          → handler lookup, timeout, circuit breaker, cache, metrics
  - MarketplaceManager      → OpenClaw + built-in skill registration
  - This file               → thin facade that wires everything together

Public API is unchanged — existing code continues to call ToolExecutor.execute().
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from predacore.auth.sandbox import SessionSandboxPool

from ..config import PredaCoreConfig
from .dispatcher import ToolDispatcher
from .handlers import ToolContext as _ToolContext
from .marketplace import MarketplaceManager
from .subsystem_init import SubsystemFactory
from .trust_policy import TrustPolicyEvaluator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MAX_OUTPUT_CHARS = 1_000_000       # 1 MB — raised from 50 KB per "remove all limits"
OUTPUT_HEAD_CHARS = 500_000        # Characters to keep from the beginning when truncating
OUTPUT_TAIL_CHARS = 100_000        # Characters to keep from the end when truncating
HTTP_CLIENT_TIMEOUT = 120          # Raised from 15 → 120 per "remove all limits"
MIN_API_KEY_LENGTH = 8             # Minimum length to consider an API key valid
DEFAULT_SANDBOX_POOL_SIZE = 50     # Max concurrent sandbox sessions


class ToolExecutor:
    """Executes tool calls against the user's system.

    Thin facade that coordinates:
      1. SubsystemFactory  — lazy init of desktop, memory, voice, etc.
      2. TrustPolicyEvaluator — permission checks
      3. ToolDispatcher — full dispatch pipeline (resilience, cache, history)
      4. MarketplaceManager — skill catalog + OpenClaw import
    """

    def __init__(self, config: PredaCoreConfig) -> None:
        self.config = config
        home_dir = config.home_dir

        # ── Trust policy ──────────────────────────────────────────────
        self._trust_evaluator = TrustPolicyEvaluator(
            trust_level=config.security.trust_level,
            permission_mode=config.security.permission_mode,
            remember_approvals=config.security.remember_approvals,
            home_dir=home_dir,
        )
        self._trust_policy = self._trust_evaluator.policy

        self._tool_rate_max = int(os.getenv("PREDACORE_TOOL_RATE_LIMIT", "120"))
        self._tool_timeout = int(os.getenv("PREDACORE_TOOL_TIMEOUT_SECONDS", "120"))

        # ── Subsystems (single source of truth) ──────────────────────
        bundle = SubsystemFactory.create_all(config, home_dir=home_dir)

        self._desktop_operator = bundle.desktop_operator
        self._unified_memory = bundle.unified_memory
        self._memory_service = bundle.memory_service
        self._mcts_planner = bundle.mcts_planner
        self._llm_for_collab = bundle.llm_for_collab
        self._voice = bundle.voice
        self._openclaw_runtime = bundle.openclaw_runtime
        self._openclaw_enabled = bundle.openclaw_enabled
        self._sandbox = bundle.sandbox
        self._docker_sandbox = bundle.docker_sandbox
        self._sandbox_pool = bundle.sandbox_pool or SessionSandboxPool(max_sessions=DEFAULT_SANDBOX_POOL_SIZE)

        # Session-scoped in-memory cache
        self._memory: dict[str, dict[str, Any]] = {}

        # ── ToolContext ───────────────────────────────────────────────
        self._tool_ctx = _ToolContext(
            config=self.config,
            memory=self._memory,
            memory_service=self._memory_service,
            mcts_planner=self._mcts_planner,
            voice=self._voice,
            desktop_operator=self._desktop_operator,
            sandbox=self._sandbox,
            docker_sandbox=self._docker_sandbox,
            sandbox_pool=self._sandbox_pool,
            skill_marketplace=None,
            openclaw_runtime=self._openclaw_runtime,
            openclaw_enabled=self._openclaw_enabled,
            llm_for_collab=self._llm_for_collab,
            unified_memory=self._unified_memory,
            trust_policy=self._trust_policy,
            http_with_retry=self._http_with_retry,
            format_sandbox_result=self._format_sandbox_result,
            resolve_user_id=self._resolve_user_id,
        )

        # ── Dispatcher ────────────────────────────────────────────────
        self._dispatcher = ToolDispatcher(
            self._trust_evaluator,
            self._tool_ctx,
            rate_max=self._tool_rate_max,
            tool_timeout=self._tool_timeout,
        )

        # ── Marketplace (extracted module) ────────────────────────────
        self._marketplace_manager: MarketplaceManager | None = None
        launch_cfg = getattr(config, "launch", None)
        if bool(getattr(launch_cfg, "enable_plugin_marketplace", False)):
            self._marketplace_manager = MarketplaceManager(config, self._tool_ctx)
            self._marketplace_manager.initialize()

        logger.info("ToolExecutor initialized (trust=%s)", config.security.trust_level)
        self._run_startup_health_checks()

    # ------------------------------------------------------------------
    # Startup Health Checks
    # ------------------------------------------------------------------

    def _run_startup_health_checks(self) -> None:
        """Log availability of optional dependencies at startup."""
        checks: list[tuple[str, bool, str]] = []

        # CLI tools
        for tool, label, hint in [
            ("sox", "voice recording", "install sox for voice_note tool"),
            ("mmdc", "mermaid diagrams", "install @mermaid-js/mermaid-cli for diagram tool"),
            ("docker", "Docker", "install Docker for sandboxed code execution"),
            ("playwright", "Playwright", "pip install playwright && playwright install for web_search"),
        ]:
            checks.append((f"{tool} ({label})", shutil.which(tool) is not None, hint))

        # API keys
        for key, label in {
            "OPENAI_API_KEY": "OpenAI models",
            "GOOGLE_API_KEY": "Gemini models",
            "ANTHROPIC_API_KEY": "Claude (direct)",
        }.items():
            val = os.getenv(key, "")
            checks.append((
                f"{key} ({label})",
                bool(val and len(val) > MIN_API_KEY_LENGTH),
                f"set {key} env var for {label}",
            ))

        # System capabilities
        checks.extend([
            ("Docker sandbox", self._docker_sandbox is not None, "enable docker_sandbox in config.yaml"),
            ("Voice interface", self._voice is not None, "install sox + pyaudio for TTS/STT"),
            ("Persistent memory", self._memory_service is not None, "check memory DB permissions"),
        ])

        available = sum(1 for _, ok, _ in checks if ok)
        logger.info("Health check: %d/%d dependencies available", available, len(checks))
        for name, ok, hint in checks:
            if ok:
                logger.debug("  ✓ %s", name)
            else:
                logger.info("  ✗ %s — %s", name, hint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation under current trust policy."""
        return self._trust_evaluator.requires_confirmation(tool_name)

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        confirm_fn: Callable | None = None,
    ) -> str:
        """Execute a tool call and return the result as a string."""
        # Keep ToolContext in sync with mutable state
        ctx = self._tool_ctx
        ctx.llm_for_collab = self._llm_for_collab
        ctx.memory_service = self._memory_service
        ctx.mcts_planner = self._mcts_planner
        ctx.voice = self._voice
        ctx.desktop_operator = self._desktop_operator
        ctx.sandbox = self._sandbox
        ctx.docker_sandbox = self._docker_sandbox
        ctx.sandbox_pool = self._sandbox_pool
        ctx.openclaw_runtime = self._openclaw_runtime
        ctx.unified_memory = self._unified_memory
        if self._marketplace_manager:
            ctx.skill_marketplace = self._marketplace_manager.skill_marketplace

        return await self._dispatcher.dispatch(
            tool_name,
            arguments,
            confirm_fn=confirm_fn,
            blocked_tools=self.config.security.blocked_tools or None,
            allowed_tools=self.config.security.allowed_tools or None,
        )

    # ------------------------------------------------------------------
    # Static helpers (kept for backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_user_id(args: dict[str, Any]) -> str:
        """Resolve user ID from args or environment."""
        return str(args.get("user_id") or os.getenv("USER") or "default")

    @staticmethod
    def _format_sandbox_result(result: dict[str, Any]) -> str:
        """Format a sandbox execution result into a human-readable string."""
        status = result.get("status", "UNKNOWN")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        error_msg = result.get("error_message", "")

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[STDERR]\n{stderr}")
        if status not in ("SUCCESS",):
            parts.append(f"[Status: {status}]")
        if error_msg:
            parts.append(f"[Error: {error_msg}]")

        output = "\n".join(parts)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:OUTPUT_HEAD_CHARS] + "\n...[truncated]...\n" + output[-OUTPUT_TAIL_CHARS:]
        return output or "[Code completed with no output]"

    # ------------------------------------------------------------------
    # HTTP helper (bound method — used by ToolContext.http_with_retry)
    # ------------------------------------------------------------------

    async def _http_with_retry(
        self,
        method: str,
        url: str,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        **kwargs: Any,
    ) -> httpx.Response:
        """HTTP request with exponential backoff retry on transient errors."""
        from random import random

        import httpx

        from predacore.auth.security import validate_url_ssrf

        validate_url_ssrf(url)  # raises ValueError if unsafe

        retryable = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=HTTP_CLIENT_TIMEOUT, follow_redirects=True
                ) as client:
                    resp = await getattr(client, method)(url, **kwargs)
                    if resp.status_code in retryable and attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1)) * (1 + 0.1 * random())
                        ra = resp.headers.get("Retry-After")
                        if ra and ra.isdigit():
                            delay = max(delay, float(ra))
                        logger.info(
                            "HTTP %d from %s, retry %d in %.1fs",
                            resp.status_code, url, attempt, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    return resp
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) * (1 + 0.1 * random())
                    logger.info("HTTP error %s, retry %d in %.1fs", e, attempt, delay)
                    await asyncio.sleep(delay)
        raise last_exc or RuntimeError("HTTP retry exhausted")
