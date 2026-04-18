"""
PredaCore Operator Retry — Decorator for flaky operator actions.

Some operator actions (screenshots, AX queries, app focus) can fail
transiently due to macOS Accessibility race conditions, timing issues,
or resource contention. This module provides a retry decorator that
automatically retries on known transient failures.

Usage:
    from predacore.operators.retry import with_retry

    @with_retry(max_attempts=3, backoff_base=0.2, retryable_errors=(DesktopControlError,))
    def _screenshot(self, params):
        ...

Or as a wrapper:
    result = with_retry(max_attempts=3)(lambda: operator.execute("screenshot", {}))()
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 0.2,
    backoff_max: float = 2.0,
    retryable_errors: tuple[type[Exception], ...] = (RuntimeError,),
    retryable_messages: tuple[str, ...] = (
        "kAXErrorCannotComplete",
        "timed out",
        "connection reset",
        "resource busy",
    ),
) -> Callable[[F], F]:
    """Decorator: retry on transient errors with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try).
        backoff_base: Initial delay between retries (seconds).
        backoff_max: Maximum delay between retries (seconds).
        retryable_errors: Exception types that trigger a retry.
        retryable_messages: Substrings in error messages that indicate transience.
    """

    def decorator(func: F) -> F:
        """Decorate a function with retry logic."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap function call with retry and backoff."""
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as exc:
                    last_error = exc
                    err_msg = str(exc).lower()

                    # Only retry if the error message looks transient
                    is_transient = any(
                        pattern.lower() in err_msg
                        for pattern in retryable_messages
                    )

                    if not is_transient or attempt >= max_attempts:
                        raise

                    delay = min(
                        backoff_base * (2 ** (attempt - 1)),
                        backoff_max,
                    )
                    logger.info(
                        "Retry %d/%d for %s after transient error: %s (delay=%.2fs)",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)

            # Should not reach here, but safety net
            if last_error:
                raise RuntimeError(
                    f"Retry exhausted after {max_attempts} attempts for {func.__name__}: {last_error}"
                ) from last_error
            raise RuntimeError(f"Retry exhausted after {max_attempts} attempts for {func.__name__}")

        return wrapper  # type: ignore[return-value]

    return decorator


async def async_with_retry(
    coro_factory: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff_base: float = 0.2,
    backoff_max: float = 2.0,
    retryable_errors: tuple[type[Exception], ...] = (RuntimeError,),
    retryable_messages: tuple[str, ...] = (
        "kAXErrorCannotComplete",
        "timed out",
        "connection reset",
    ),
    **kwargs: Any,
) -> Any:
    """Async retry wrapper — calls coro_factory(*args, **kwargs) with retries.

    Usage:
        result = await async_with_retry(
            engine.quick_scan,
            max_attempts=3,
            retryable_errors=(RuntimeError,),
        )
    """
    import asyncio

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory(*args, **kwargs)
        except retryable_errors as exc:
            last_error = exc
            err_msg = str(exc).lower()

            is_transient = any(
                pattern.lower() in err_msg
                for pattern in retryable_messages
            )

            if not is_transient or attempt >= max_attempts:
                raise

            delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
            logger.info(
                "Async retry %d/%d after transient error: %s (delay=%.2fs)",
                attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)

    if last_error:
        raise RuntimeError(
            f"Retry exhausted after {max_attempts} attempts for {coro_factory.__name__}: {last_error}"
        ) from last_error
    raise RuntimeError(f"Retry exhausted after {max_attempts} attempts for {coro_factory.__name__}")
