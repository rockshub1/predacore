"""
PredaCore Gateway — Central message router and session manager.

The Gateway is the "front door" of PredaCore. All messages from all
channels (CLI, Telegram, Discord, Web) flow through here. It:

  1. Normalizes messages from different channels
  2. Routes them to the correct session
  3. Submits them to the Lane Queue for serial execution
  4. Returns responses back to the originating channel

Usage:
    gateway = Gateway(config, core)
    response = await gateway.handle_message(
        IncomingMessage(channel="cli", user_id="user1", text="hello")
    )
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from .auth.middleware import AuthMiddleware
from .auth.security import sanitize_user_input as _sanitize_input
from .channels.health import ChannelHealthMonitor
from .config import PredaCoreConfig
from .services.identity_service import IdentityService
from .services.lane_queue import LaneQueue
from .services.outcome_store import OutcomeStore, TaskOutcome
from .services.rate_limiter import RateLimiter, default_api_limits
from .sessions import Session, SessionStore
from .utils.cache import LRUCache

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
IDENTITY_CACHE_MAX_SIZE = 500           # LRU cache entries for identity resolution
IDENTITY_CACHE_TTL_SECONDS = 600        # TTL for cached identity lookups (10 min)
SESSION_LOCKS_MAX = 10_000              # Max per-session locks before LRU eviction
USER_RATE_LIMIT_PER_MINUTE = 30         # Max incoming messages per user per minute
RATE_LIMIT_WINDOW_SECONDS = 60.0        # Sliding window for per-user rate limiting
RATE_LIMIT_CLEANUP_USER_THRESHOLD = 1_000    # Trigger stale-user cleanup above this count
RATE_LIMIT_CLEANUP_ENTRY_THRESHOLD = 50_000  # Trigger cleanup above this total entry count
GATEWAY_ERROR_MSG_TRUNCATE = 500        # Max chars of user message stored in error outcomes
SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400


# ── Message Types ─────────────────────────────────────────────────────


@dataclass
class IncomingMessage:
    """Normalized message from any channel."""

    channel: str  # "cli" | "telegram" | "discord" | "webchat" | "webhook"
    user_id: str
    text: str
    session_id: str | None = None  # None = auto-assign
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OutgoingMessage:
    """Response message to send back to a channel."""

    channel: str
    user_id: str
    text: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thinking: str | None = None  # Agent's reasoning (optional debug info)


# ── Channel Adapter Interface ────────────────────────────────────────


class ChannelAdapter(ABC):
    """
    Base class for messaging channel adapters.

    Each adapter handles receiving messages from and sending responses
    to a specific platform (Telegram, Discord, CLI, etc.)
    """

    channel_name: str = "base"

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages on this channel."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel adapter."""
        ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> None:
        """Send a response message to the channel."""
        ...

    def set_message_handler(
        self, handler: Callable[[IncomingMessage], Coroutine[Any, Any, OutgoingMessage]]
    ) -> None:
        """Set the callback that processes incoming messages."""
        self._message_handler = handler


# ── Gateway ──────────────────────────────────────────────────────────


class Gateway:
    """
    Central message router and session manager.

    All messages flow through the Gateway, which:
    1. Normalizes them into IncomingMessage
    2. Resolves/creates sessions
    3. Submits to LaneQueue for serial processing
    4. Returns OutgoingMessage via the channel adapter
    """

    def __init__(
        self,
        config: PredaCoreConfig,
        process_fn: Callable[[str, str, Session], Coroutine[Any, Any, str]],
    ):
        """
        Args:
            config: PredaCore configuration
            process_fn: Async function(user_id, message_text, session) -> response_text.
                        This is the PredaCore core brain function.
        """
        self.config = config
        self._process_fn = process_fn
        self.core = None  # Set by daemon after init for /model command
        self.session_store = SessionStore(config.sessions_dir)
        self.lane_queue = LaneQueue(
            max_lanes=config.security.max_concurrent_tasks * 2,
            default_timeout=config.security.task_timeout_seconds,
        )
        self._channels: dict[str, ChannelAdapter] = {}
        self._running = False
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
            "start_time": 0.0,
        }
        self.health_monitor = ChannelHealthMonitor()
        # Outcome tracking — injected by daemon to share core's single connection.
        # Do NOT create a second OutcomeStore here; it causes duplicate SQLite
        # connections to outcomes.db → WAL lock contention → "database is locked".
        self._outcome_store: OutcomeStore | None = None
        # Cross-channel identity service
        self.identity = IdentityService(config.home_dir)
        # LRU cache for identity resolution
        self._identity_cache = LRUCache(max_size=IDENTITY_CACHE_MAX_SIZE)
        # Per-session locks to prevent concurrent modifications (bounded LRU)
        self._session_locks: collections.OrderedDict[str, asyncio.Lock] = (
            collections.OrderedDict()
        )
        self._session_locks_max = SESSION_LOCKS_MAX
        self._session_locks_guard = asyncio.Lock()
        # Rate limiter (external rules for API calls)
        self._rate_limiter = RateLimiter()
        for rule in default_api_limits():
            self._rate_limiter.add_rule(rule)
        # Per-user rate limiting for incoming messages
        self._user_request_times: dict[str, list[float]] = {}
        self._user_rate_limit = USER_RATE_LIMIT_PER_MINUTE
        self._rate_limit_lock = asyncio.Lock()
        # Auth middleware (enterprise) — active when JWT_SECRET or API keys configured.
        import os

        jwt_secret = os.getenv("PREDACORE_JWT_SECRET", "")
        require_auth = bool(config.security.require_auth) if hasattr(config.security, "require_auth") else False
        self.auth = AuthMiddleware(jwt_secret=jwt_secret) if jwt_secret else None
        if self.auth:
            logger.info("Gateway auth middleware enabled (JWT)")
        elif not jwt_secret and require_auth:
            logger.warning(
                "SECURITY: JWT_SECRET not configured but auth is required "
                "— API endpoints are UNPROTECTED"
            )
        logger.info("Gateway initialized")

    def register_channel(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter with the gateway."""
        adapter.set_message_handler(self.handle_message)
        # Inject gateway reference for adapters that need it (e.g., webchat dashboard)
        if hasattr(adapter, "set_gateway"):
            adapter.set_gateway(self)
        self._channels[adapter.channel_name] = adapter
        self.health_monitor.register(adapter.channel_name)
        logger.info("Registered channel: %s", adapter.channel_name)

    async def start(self) -> None:
        """Start the gateway and all registered channels."""
        self._running = True
        self._stats["start_time"] = time.time()

        # Start all channel adapters
        for name, adapter in self._channels.items():
            try:
                await adapter.start()
                self.health_monitor.mark_started(name)
                logger.info("Channel started: %s", name)
            except (OSError, ConnectionError, RuntimeError) as e:
                self.health_monitor.mark_disconnected(name, str(e))
                logger.error("Failed to start channel %s: %s", name, e)

        # Start health monitor background task
        await self.health_monitor.start_monitoring()

        # Start identity heartbeat service (non-fatal if unavailable)
        try:
            from .identity.engine import get_identity_engine
            from .identity.heartbeat import create_default_heartbeat

            engine = get_identity_engine(self.config)
            self._heartbeat = create_default_heartbeat(engine)
            await self._heartbeat.start()
            logger.info("Identity heartbeat service started")
        except (ImportError, OSError, RuntimeError):
            logger.debug("Identity heartbeat not available (non-fatal)", exc_info=True)

        logger.info(
            "Gateway started with %d channels: %s",
            len(self._channels),
            list(self._channels.keys()),
        )

    async def stop(self) -> None:
        """Stop the gateway and all channels."""
        self._running = False

        # Stop health monitor
        await self.health_monitor.stop_monitoring()

        # Stop channels
        for name, adapter in self._channels.items():
            try:
                await adapter.stop()
                self.health_monitor.mark_stopped(name)
            except (OSError, RuntimeError) as e:
                logger.error("Error stopping channel %s: %s", name, e)

        # Stop identity heartbeat
        if hasattr(self, "_heartbeat"):
            try:
                await self._heartbeat.stop()
            except (OSError, RuntimeError):
                pass

        # Shutdown lane queue
        await self.lane_queue.shutdown()
        logger.info("Gateway stopped")

    def _handle_model_command(
        self, text: str, user_id: str, channel: str
    ) -> OutgoingMessage | None:
        """Handle /model command — show or switch provider/model from any channel."""
        stripped = text.strip()
        if not stripped.lower().startswith("/model"):
            return None

        parts = stripped.split(maxsplit=1)
        if not self.core:
            return None
        llm = self.core.llm  # LLMInterface (router)

        # /model (no args) — show current model + available providers
        if len(parts) == 1:
            current = f"**{llm.active_provider}** (model: {llm.active_model})"
            alternatives = [p for p in llm.available_providers if p != llm.active_provider]
            alt_text = "\n".join(f"  /model {p}" for p in alternatives) if alternatives else "  (none)"
            response_text = (
                f"Current: {current}\n\n"
                f"Available:\n{alt_text}\n\n"
                f"Switch with: /model <provider>"
            )
            return OutgoingMessage(
                channel=channel,
                user_id=user_id,
                text=response_text,
                session_id="",
            )

        # /model <provider> — switch to that provider
        target = parts[1].strip().lower()

        # Check if it's a known provider
        from .llm_providers.openai import PROVIDER_ENDPOINTS
        known_providers = set(llm.available_providers) | set(PROVIDER_ENDPOINTS.keys()) | {
            "gemini-cli", "anthropic", "gemini",
        }

        if target not in known_providers:
            return OutgoingMessage(
                channel=channel,
                user_id=user_id,
                text=f"Unknown provider: **{target}**\n\nAvailable: {', '.join(sorted(llm.available_providers))}",
                session_id="",
            )

        try:
            llm.set_active_model(provider=target)
            new_model = llm.active_model
            response_text = (
                f"Switched to **{target}** (model: {new_model}).\n\n"
                f"Switch back anytime with: /model {llm.available_providers[0] if llm.available_providers else 'anthropic'}"
            )
            logger.info("User %s switched model to %s via %s", user_id, target, channel)
            return OutgoingMessage(
                channel=channel,
                user_id=user_id,
                text=response_text,
                session_id="",
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("Failed to switch model to %s: %s", target, e)
            return OutgoingMessage(
                channel=channel,
                user_id=user_id,
                text=f"Failed to switch to {target}: {e}",
                session_id="",
            )

    def _handle_session_command(
        self, text: str, user_id: str, channel: str
    ) -> OutgoingMessage | None:
        """
        Handle session management commands: /new, /resume, /sessions.
        Returns an OutgoingMessage if handled, None otherwise.
        """
        stripped = text.strip().lower()

        if stripped == "/cancel":
            # Find the user's active session and cancel its current task
            sessions = self.session_store.list_sessions(user_id=user_id, limit=1)
            if sessions:
                cancelled = self.lane_queue.cancel_current(sessions[0].session_id)
                msg = "Cancelled the current task." if cancelled else "No task running to cancel."
            else:
                msg = "No active session found."
            return OutgoingMessage(
                channel=channel, user_id=user_id, text=msg, session_id="",
            )

        if stripped == "/new":
            new_id = str(uuid.uuid4())
            self.session_store.create(user_id=user_id, session_id=new_id)
            return OutgoingMessage(
                channel=channel,
                user_id=user_id,
                text="Started a fresh session. Previous conversations are saved and can be resumed with /resume.",
                session_id=new_id,
            )

        if stripped == "/sessions":
            sessions = self.session_store.list_sessions(user_id=user_id, limit=10)
            if not sessions:
                return OutgoingMessage(
                    channel=channel, user_id=user_id,
                    text="No sessions found.", session_id="",
                )
            lines = ["Your recent sessions:\n"]
            for i, s in enumerate(sessions):
                age = time.time() - s.updated_at
                if age < SECONDS_PER_HOUR:
                    ago = f"{int(age / 60)}m ago"
                elif age < SECONDS_PER_DAY:
                    ago = f"{int(age / SECONDS_PER_HOUR)}h ago"
                else:
                    ago = f"{int(age / SECONDS_PER_DAY)}d ago"
                title = s.title or "(untitled)"
                msgs = getattr(s, "message_count", 0) or 0
                sid_short = s.session_id[:8]
                marker = " (active)" if i == 0 else ""
                origin = f" [{s.channel_origin}]" if s.channel_origin else ""
                lines.append(f"{i+1}. `{sid_short}` — {title} ({msgs} msgs, {ago}){origin}{marker}")
            lines.append("\nUse /resume <number> to switch sessions.")
            return OutgoingMessage(
                channel=channel, user_id=user_id,
                text="\n".join(lines), session_id="",
            )

        if stripped.startswith("/resume"):
            parts = stripped.split(maxsplit=1)
            sessions = self.session_store.list_sessions(user_id=user_id, limit=10)
            if not sessions:
                return OutgoingMessage(
                    channel=channel, user_id=user_id,
                    text="No sessions to resume.", session_id="",
                )
            if len(parts) < 2:
                # Resume most recent session
                target = sessions[0]
            else:
                selector = parts[1].strip()
                target = None
                # Try as a number (1-indexed)
                try:
                    idx = int(selector) - 1
                    if 0 <= idx < len(sessions):
                        target = sessions[idx]
                except ValueError:
                    pass
                # Try as session ID prefix
                if target is None:
                    for s in sessions:
                        if s.session_id.startswith(selector):
                            target = s
                            break
                if target is None:
                    return OutgoingMessage(
                        channel=channel, user_id=user_id,
                        text=f"Session not found: '{selector}'. Use /sessions to list available sessions.",
                        session_id="",
                    )
            title = target.title or "(untitled)"
            msgs = getattr(target, "message_count", 0) or 0
            return OutgoingMessage(
                channel=channel, user_id=user_id,
                text=f"Resumed session: {title} ({msgs} messages). I remember our conversation — go ahead!",
                session_id=target.session_id,
            )

        return None

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock to prevent concurrent modifications.

        Uses LRU eviction: when the dict exceeds _session_locks_max entries,
        the oldest (least-recently-used) unlocked entries are evicted.
        """
        async with self._session_locks_guard:
            if session_id in self._session_locks:
                # Move to end (most recently used)
                self._session_locks.move_to_end(session_id)
                return self._session_locks[session_id]
            # Evict oldest unlocked entries if at capacity
            while len(self._session_locks) >= self._session_locks_max:
                # Find the oldest unlocked entry (scan from front/oldest)
                evict_target = None
                for sid, lock in self._session_locks.items():
                    if not lock.locked():
                        evict_target = sid
                        break
                if evict_target is None:
                    # All entries are locked — allow a small overflow
                    logger.warning(
                        "Session lock eviction: all %d entries locked, allowing overflow",
                        len(self._session_locks),
                    )
                    break
                del self._session_locks[evict_target]
            self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    async def _check_user_rate_limit(self, user_id: str) -> bool:
        """
        Check if a user has exceeded the per-minute rate limit.

        Returns True if the request is allowed, False if rate-limited.
        Also cleans up entries for users whose timestamps have all expired,
        preventing unbounded memory growth from inactive users.
        """
        async with self._rate_limit_lock:
            now = time.time()
            window = RATE_LIMIT_WINDOW_SECONDS

            if user_id not in self._user_request_times:
                self._user_request_times[user_id] = []

            # Prune timestamps older than the window for the current user
            self._user_request_times[user_id] = [
                t for t in self._user_request_times[user_id] if now - t < window
            ]

            if len(self._user_request_times[user_id]) >= self._user_rate_limit:
                return False

            self._user_request_times[user_id].append(now)

            # Periodic cleanup: evict stale entries from inactive users.
            # Run every ~100 requests to amortize cost.
            total_entries = sum(len(v) for v in self._user_request_times.values())
            if len(self._user_request_times) > RATE_LIMIT_CLEANUP_USER_THRESHOLD or total_entries > RATE_LIMIT_CLEANUP_ENTRY_THRESHOLD:
                stale_users = [
                    uid
                    for uid, times in self._user_request_times.items()
                    if not times or (now - times[-1]) >= window
                ]
                for uid in stale_users:
                    del self._user_request_times[uid]

            return True

    async def handle_message(
        self,
        incoming: IncomingMessage,
        event_fn: Callable | None = None,
        stream_fn: Callable[[str], Awaitable[None]] | None = None,
        confirm_fn: Callable | None = None,
    ) -> OutgoingMessage:
        """
        Process an incoming message through the full pipeline:
        1. Handle session commands (/new, /resume, /sessions)
        2. Resolve/create session
        3. Persist user message
        4. Submit to lane queue (serial execution)
        5. Get response from PredaCore core
        6. Persist assistant message
        7. Return outgoing message
        """
        self._stats["messages_received"] += 1
        self.health_monitor.record_message(incoming.channel, "in")

        # Resolve cross-channel identity → canonical user_id (LRU cached)
        display = incoming.metadata.get("username", "")
        identity_key = f"{incoming.channel}:{incoming.user_id}"
        cached_id = self._identity_cache.get(identity_key)
        if cached_id is not None:
            incoming.user_id = cached_id
        else:
            incoming.user_id = self.identity.resolve(
                incoming.channel, incoming.user_id, display_name=display,
            )
            self._identity_cache.set(identity_key, incoming.user_id, ttl_seconds=IDENTITY_CACHE_TTL_SECONDS)

        # Handle /link command for cross-channel identity linking
        stripped = incoming.text.strip()
        if stripped.startswith("/link"):
            return self._handle_link_command(stripped, incoming)

        # Check per-user rate limit before any processing
        if not await self._check_user_rate_limit(incoming.user_id):
            logger.warning(
                "Rate limit exceeded for user %s on channel %s",
                incoming.user_id,
                incoming.channel,
            )
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text="You're sending messages too quickly. Please wait a moment and try again.",
                session_id=incoming.session_id or "",
            )

        # Sanitize user input (strip null bytes, ANSI escapes, limit length)
        incoming.text = _sanitize_input(incoming.text)

        # Handle /model command — switch provider/model from any channel
        model_result = self._handle_model_command(
            incoming.text, incoming.user_id, incoming.channel
        )
        if model_result is not None:
            self._stats["messages_sent"] += 1
            return model_result

        # Handle session management commands before processing
        cmd_result = self._handle_session_command(
            incoming.text, incoming.user_id, incoming.channel
        )
        if cmd_result is not None:
            # For /resume, update the user's active session mapping
            if cmd_result.session_id and incoming.text.strip().lower().startswith("/resume"):
                # Force next _resolve_session_id to find this session
                # by touching its updated_at timestamp
                s = self.session_store.get(cmd_result.session_id)
                if s:
                    s.updated_at = time.time()
                    self.session_store._save_meta(s)
            self._stats["messages_sent"] += 1
            return cmd_result

        try:
            # 1. Resolve session
            session_id = incoming.session_id or self._resolve_session_id(
                incoming.user_id, incoming.channel
            )

            # Acquire per-session lock to prevent concurrent modifications
            session_lock = await self._get_session_lock(session_id)
            async with session_lock:
                session = self.session_store.get_or_create(
                    session_id, user_id=incoming.user_id
                )
                # Tag with origin channel if new session
                if not session.channel_origin:
                    session.channel_origin = incoming.channel

                # 2. Persist user message
                self.session_store.append_message(
                    session_id,
                    "user",
                    incoming.text,
                    metadata={"channel": incoming.channel, "msg_id": incoming.message_id},
                )

                # 3. Submit to lane queue (this ensures serial execution per session)
                response_text = await self.lane_queue.submit(
                    session_id,
                    self._process_fn,
                    incoming.user_id,
                    incoming.text,
                    session,
                    event_fn=event_fn,
                    stream_fn=stream_fn,
                    confirm_fn=confirm_fn,
                )

                # 4. Persist assistant response
                self.session_store.append_message(
                    session_id,
                    "assistant",
                    response_text,
                    metadata={"channel": incoming.channel},
                )

            # 5. Build outgoing message
            self._stats["messages_sent"] += 1
            self.health_monitor.record_message(incoming.channel, "out")
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text=response_text,
                session_id=session_id,
                metadata=incoming.metadata or {},
            )

        except ConnectionError as e:
            # All LLM providers failed (likely rate limited)
            self._stats["errors"] += 1
            self.health_monitor.record_error(incoming.channel, str(e))
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str:
                user_msg = "I'm currently rate-limited by the AI provider. Please try again in a minute or two."
            elif "overloaded" in error_str or "529" in error_str:
                user_msg = "The AI provider is overloaded right now. Please try again shortly."
            else:
                user_msg = "All AI providers are currently unavailable. Please try again in a moment."
            logger.warning("LLM unavailable for %s: %s", incoming.user_id, e)
            # Persist error response to session so history stays complete
            _err_sid = incoming.session_id or self._resolve_session_id(
                incoming.user_id, incoming.channel
            )
            try:
                self.session_store.append_message(
                    _err_sid, "assistant", user_msg,
                    metadata={"channel": incoming.channel, "error": True},
                )
            except (OSError, ValueError):
                pass
            # Record failure in OutcomeStore — these were previously invisible
            await self._record_gateway_failure(
                incoming, _err_sid, f"connection_error: {str(e)[:200]}"
            )
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text=user_msg,
                session_id=_err_sid,
                metadata=incoming.metadata or {},
            )

        except Exception as e:  # catch-all: request handler boundary
            self._stats["errors"] += 1
            self.health_monitor.record_error(incoming.channel, str(e))
            logger.error(
                "Error processing message from %s/%s: %s",
                incoming.channel,
                incoming.user_id,
                e,
                exc_info=True,
            )
            error_msg = "I encountered an internal error. Please try again."
            # Persist error response to session so history stays complete
            _err_sid = incoming.session_id or self._resolve_session_id(
                incoming.user_id, incoming.channel
            )
            try:
                self.session_store.append_message(
                    _err_sid, "assistant", error_msg,
                    metadata={"channel": incoming.channel, "error": True, "exception": str(e)},
                )
            except (OSError, ValueError):
                pass
            # Record failure in OutcomeStore — these were previously invisible
            await self._record_gateway_failure(
                incoming, _err_sid, f"unhandled_exception: {str(e)[:200]}"
            )
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text=error_msg,
                session_id=_err_sid,
                metadata=incoming.metadata or {},
            )

    async def _record_gateway_failure(
        self,
        incoming: IncomingMessage,
        session_id: str,
        error: str,
    ) -> None:
        """Record a gateway-level failure in the OutcomeStore.

        These failures (LLM connection errors, unhandled exceptions) were
        previously invisible — process() never ran or crashed before
        recording an outcome. Now they're tracked.
        """
        if self._outcome_store is None:
            return
        try:
            outcome = TaskOutcome(
                user_id=incoming.user_id,
                user_message=incoming.text[:GATEWAY_ERROR_MSG_TRUNCATE],
                response_summary="[gateway error]",
                tools_used=[],
                tool_errors=[],
                provider_used="",
                model_used="",
                latency_ms=0,
                token_count_prompt=0,
                token_count_completion=0,
                iterations=0,
                success=False,
                error=error,
                persona_drift_score=0.0,
                persona_drift_regens=0,
                session_id=session_id,
            )
            await self._outcome_store.record(outcome)
        except (OSError, ValueError, RuntimeError):
            logger.debug("Gateway outcome recording failed (non-fatal)", exc_info=True)

    def _handle_link_command(
        self, text: str, incoming: IncomingMessage
    ) -> OutgoingMessage:
        """Handle /link command for cross-channel identity linking."""
        parts = text.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ""

        if not code:
            # Generate a link code for this user
            link_code = self.identity.generate_link_code(incoming.user_id)
            user_info = self.identity.get_user_info(incoming.user_id)
            channels = [
                f"  - {l['channel']}" for l in (user_info or {}).get("channels", [])
                if l["channel"] != "_link_code"
            ]
            channel_list = "\n".join(channels) if channels else "  (none)"
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text=(
                    f"Your link code: **{link_code}**\n\n"
                    f"To link another device/channel, send `/link {link_code}` from there.\n"
                    f"This will merge your sessions across channels.\n\n"
                    f"Currently linked channels:\n{channel_list}"
                ),
                session_id="",
            )
        else:
            # Redeem a link code
            result = self.identity.redeem_link_code(
                code, incoming.channel, incoming.metadata.get("raw_user_id", incoming.user_id)
            )
            if result:
                # Update the user_id for this message going forward
                incoming.user_id = result
                sessions = self.session_store.list_sessions(user_id=result, limit=5)
                count = len(sessions)
                return OutgoingMessage(
                    channel=incoming.channel,
                    user_id=result,
                    text=(
                        f"Linked! Your sessions are now connected across channels.\n"
                        f"Found {count} existing session(s). Use /sessions to see them all."
                    ),
                    session_id="",
                )
            return OutgoingMessage(
                channel=incoming.channel,
                user_id=incoming.user_id,
                text="Invalid or expired link code. Use /link on your other channel to get a new code.",
                session_id="",
            )

    def _resolve_session_id(self, user_id: str, channel: str) -> str:
        """
        Determine which session to use for this user+channel.
        Always reuse the most recent session for a user so conversation
        history persists across daemon restarts and long idle periods.
        """
        sessions = self.session_store.list_sessions(user_id=user_id, limit=1)
        if sessions:
            return sessions[0].session_id

        # No previous session — create new
        return str(uuid.uuid4())

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        uptime = (
            time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        )
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "active_channels": list(self._channels.keys()),
            "lane_stats": self.lane_queue.get_lane_stats(),
            "channel_health": self.health_monitor.get_health_report(),
        }
