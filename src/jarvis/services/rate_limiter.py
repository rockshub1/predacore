"""
Redis-backed rate limiter for JARVIS API endpoints.

Implements multiple rate-limiting algorithms:
  - Fixed window: simple counter per time window
  - Sliding window: weighted average of current/previous window
  - Token bucket: allows controlled bursts
  - Per-user and per-IP rate limits
  - Configurable limits per endpoint

Falls back to in-memory rate limiting when Redis is unavailable.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("jarvis.services.rate_limiter")


# ── Configuration ────────────────────────────────────────────────────


class RateLimitAlgorithm(str, Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""

    name: str
    max_requests: int  # Max requests per window
    window_seconds: int  # Window duration
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_size: int = 0  # Extra burst allowance (token bucket)
    per_user: bool = True  # Apply per user vs global
    scope: str = "*"  # Endpoint pattern


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp when limit resets
    retry_after: float = 0.0  # Seconds until next allowable request

    def to_headers(self) -> dict[str, str]:
        """Generate standard rate limit HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(int(self.retry_after) + 1)
        return headers


# ── In-Memory Backend ────────────────────────────────────────────────


class _MemoryWindow:
    """A single rate limit window stored in memory."""

    __slots__ = ("count", "start_time")

    def __init__(self, start_time: float) -> None:
        self.count: int = 0
        self.start_time = start_time


class InMemoryBackend:
    """In-memory rate limiting backend (used when Redis is unavailable)."""

    def __init__(self) -> None:
        # key -> {window_key -> _MemoryWindow}
        self._windows: dict[str, dict[str, _MemoryWindow]] = defaultdict(dict)
        # key -> (tokens, last_refill_time)
        self._buckets: dict[str, tuple[float, float]] = {}

    def fixed_window_check(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> RateLimitResult:
        now = time.time()
        window_key = str(int(now // window_seconds))
        windows = self._windows[key]

        # Clean old windows
        expired = [
            k for k in windows if float(k) * window_seconds < now - window_seconds * 2
        ]
        for k in expired:
            del windows[k]

        if window_key not in windows:
            windows[window_key] = _MemoryWindow(now)

        window = windows[window_key]
        # Check limit BEFORE incrementing to avoid over-counting rejected requests
        allowed = window.count < max_requests
        if allowed:
            window.count += 1
        remaining = max_requests - window.count
        reset_at = (int(now // window_seconds) + 1) * window_seconds

        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, remaining),
            limit=max_requests,
            reset_at=reset_at,
            retry_after=max(0, reset_at - now) if not allowed else 0,
        )

    def sliding_window_check(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> RateLimitResult:
        now = time.time()
        current_window = str(int(now // window_seconds))
        previous_window = str(int(now // window_seconds) - 1)
        windows = self._windows[key]

        if current_window not in windows:
            windows[current_window] = _MemoryWindow(now)

        current = windows.get(current_window, _MemoryWindow(now))
        previous = windows.get(previous_window, _MemoryWindow(now))

        # Weight previous window
        elapsed_ratio = (now % window_seconds) / window_seconds
        weighted_count = previous.count * (1 - elapsed_ratio) + current.count

        reset_at = (int(now // window_seconds) + 1) * window_seconds

        if weighted_count < max_requests:
            current.count += 1
            return RateLimitResult(
                allowed=True,
                remaining=int(max_requests - weighted_count),
                limit=max_requests,
                reset_at=reset_at,
                retry_after=0,
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=int(max_requests - weighted_count),
                limit=max_requests,
                reset_at=reset_at,
                retry_after=max(0, reset_at - now),
            )

    def token_bucket_check(
        self,
        key: str,
        max_tokens: int,
        refill_rate: float,
        burst: int = 0,
    ) -> RateLimitResult:
        now = time.time()
        capacity = max_tokens + burst

        if key not in self._buckets:
            self._buckets[key] = (capacity, now)

        tokens, last_refill = self._buckets[key]
        elapsed = now - last_refill
        tokens = min(capacity, tokens + elapsed * refill_rate)
        allowed = tokens >= 1
        if allowed:
            tokens -= 1  # Only deduct when request is allowed
        self._buckets[key] = (tokens, now)

        return RateLimitResult(
            allowed=allowed,
            remaining=int(max(0, tokens)),
            limit=capacity,
            reset_at=now + ((capacity - max(0, tokens)) / refill_rate)
            if refill_rate > 0
            else now,
            retry_after=(1 / refill_rate) if tokens < 0 and refill_rate > 0 else 0,
        )


# ── Redis Backend (optional) ────────────────────────────────────────


class RedisBackend:
    """
    Redis-backed rate limiting using Lua scripts for atomicity.
    Falls back to InMemoryBackend if Redis is unavailable.
    """

    SLIDING_WINDOW_LUA = """
    local key = KEYS[1]
    local window = tonumber(ARGV[1])
    local max_requests = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
    local count = redis.call('ZCARD', key)

    if count < max_requests then
        redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
        redis.call('EXPIRE', key, window)
        return {1, max_requests - count - 1}
    else
        return {0, 0}
    end
    """

    def __init__(self, redis_url: str = "") -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        if "localhost" in self.redis_url and os.getenv("JARVIS_ENV", "dev") == "production":
            logger.warning(
                "Redis URL points to localhost in production mode — "
                "set REDIS_URL to a production Redis instance"
            )
        self._client = None
        self._fallback = InMemoryBackend()
        self._using_fallback = False

    def _connect(self) -> bool:
        """Try to connect to Redis."""
        try:
            import redis

            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            self._client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except (ImportError, OSError, ConnectionError, RuntimeError) as e:
            logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
            self._using_fallback = True
            return False

    def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit using Redis or fallback."""
        if self._using_fallback or self._client is None:
            return self._fallback_check(key, config)

        try:
            return self._redis_check(key, config)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.warning(f"Redis error, falling back: {e}")
            return self._fallback_check(key, config)

    def _fallback_check(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        if config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._fallback.fixed_window_check(
                key, config.max_requests, config.window_seconds
            )
        elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            rate = config.max_requests / config.window_seconds
            return self._fallback.token_bucket_check(
                key, config.max_requests, rate, config.burst_size
            )
        else:
            return self._fallback.sliding_window_check(
                key, config.max_requests, config.window_seconds
            )

    def _redis_check(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using Redis."""
        now = time.time()
        window = config.window_seconds

        # Use sorted set sliding window
        result = self._client.eval(
            self.SLIDING_WINDOW_LUA,
            1,
            key,
            window,
            config.max_requests,
            now,
        )
        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = now + window

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            limit=config.max_requests,
            reset_at=reset_at,
            retry_after=window if not allowed else 0,
        )


# ── Rate Limiter Service ─────────────────────────────────────────────


class RateLimiter:
    """
    Production rate limiter with multiple limit tiers.

    Usage:
        limiter = RateLimiter()
        limiter.add_rule(RateLimitConfig("global", 1000, 60))
        limiter.add_rule(RateLimitConfig("per_user", 100, 60, per_user=True))

        result = limiter.check(user_id="user-1", endpoint="/api/chat")
        if not result.allowed:
            return 429, result.to_headers()
    """

    def __init__(self, redis_url: str = "") -> None:
        self._rules: list[RateLimitConfig] = []
        self._backend = RedisBackend(redis_url)
        self._total_checks: int = 0
        self._total_blocked: int = 0

    def add_rule(self, config: RateLimitConfig) -> None:
        """Add a rate limit rule."""
        self._rules.append(config)
        logger.info(
            f"Rate limit rule added: {config.name} "
            f"({config.max_requests}/{config.window_seconds}s)"
        )

    def check(
        self,
        user_id: str = "",
        endpoint: str = "",
        ip_address: str = "",
    ) -> RateLimitResult:
        """
        Check all rate limit rules and return the most restrictive result.
        """
        self._total_checks += 1
        most_restrictive: RateLimitResult | None = None

        for rule in self._rules:
            # Build key
            parts = [f"rl:{rule.name}"]
            if rule.per_user and user_id:
                parts.append(f"u:{user_id}")
            elif rule.per_user and ip_address:
                parts.append(f"ip:{ip_address}")
            elif rule.per_user:
                parts.append("u:anonymous")
            elif ip_address:
                parts.append(f"ip:{ip_address}")
            parts.append(f"e:{endpoint or '*'}")
            key = ":".join(parts)

            result = self._backend.check(key, rule)

            if most_restrictive is None or (
                not result.allowed
                or (
                    result.remaining < most_restrictive.remaining
                    and most_restrictive.allowed
                )
            ):
                most_restrictive = result

        if most_restrictive is None:
            return RateLimitResult(
                allowed=True, remaining=999, limit=999, reset_at=time.time() + 60
            )

        if not most_restrictive.allowed:
            self._total_blocked += 1

        return most_restrictive

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_checks": self._total_checks,
            "total_blocked": self._total_blocked,
            "block_rate": (
                self._total_blocked / self._total_checks
                if self._total_checks > 0
                else 0.0
            ),
            "rules": [
                {
                    "name": r.name,
                    "max_requests": r.max_requests,
                    "window_seconds": r.window_seconds,
                    "algorithm": r.algorithm.value,
                }
                for r in self._rules
            ],
            "backend": "redis" if not self._backend._using_fallback else "memory",
        }


# ── Default rate limit presets ───────────────────────────────────────


def default_api_limits() -> list[RateLimitConfig]:
    """Return sensible default rate limits for a public API."""
    return [
        RateLimitConfig(
            name="global",
            max_requests=10000,
            window_seconds=60,
            per_user=False,
        ),
        RateLimitConfig(
            name="per_user",
            max_requests=200,
            window_seconds=60,
            per_user=True,
        ),
        RateLimitConfig(
            name="per_user_burst",
            max_requests=50,
            window_seconds=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_size=20,
            per_user=True,
        ),
        RateLimitConfig(
            name="expensive_endpoints",
            max_requests=20,
            window_seconds=60,
            per_user=True,
            scope="/api/execute_*",
        ),
    ]
