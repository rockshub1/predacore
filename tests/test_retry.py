"""
Tests for the operator retry decorator.

Covers:
  - Successful execution (no retry)
  - Retry on transient errors with backoff
  - No retry on non-transient errors
  - Max attempts exhausted
  - Async retry wrapper
  - Integration with DesktopControlError patterns
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.operators.retry import with_retry, async_with_retry


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class TransientError(RuntimeError):
    """Simulates a transient error."""
    pass


class PermanentError(RuntimeError):
    """Simulates a permanent error."""
    pass


def _make_flaky_fn(fail_count: int, error_msg: str = "timed out"):
    """Create a function that fails `fail_count` times then succeeds."""
    call_count = {"n": 0}

    @with_retry(
        max_attempts=5,
        backoff_base=0.01,  # Fast for tests
        retryable_errors=(RuntimeError,),
        retryable_messages=("timed out", "resource busy", "connection reset"),
    )
    def fn():
        call_count["n"] += 1
        if call_count["n"] <= fail_count:
            raise RuntimeError(error_msg)
        return f"ok after {call_count['n']} attempts"

    return fn, call_count


# ---------------------------------------------------------------------------
# Sync retry tests
# ---------------------------------------------------------------------------

class TestWithRetry:
    """Test the @with_retry decorator."""

    def test_no_retry_on_success(self):
        """Function succeeds on first try — no retries."""
        fn, counts = _make_flaky_fn(0)
        result = fn()
        assert result == "ok after 1 attempts"
        assert counts["n"] == 1

    def test_retry_on_transient_error(self):
        """Function fails 2 times then succeeds — retries work."""
        fn, counts = _make_flaky_fn(2, "connection timed out")
        result = fn()
        assert result == "ok after 3 attempts"
        assert counts["n"] == 3

    def test_retry_on_resource_busy(self):
        """Retries on 'resource busy' error message."""
        fn, counts = _make_flaky_fn(1, "System Events resource busy")
        result = fn()
        assert result == "ok after 2 attempts"
        assert counts["n"] == 2

    def test_no_retry_on_permanent_error(self):
        """Non-transient error raises immediately without retry."""
        call_count = {"n": 0}

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=("timed out",),
        )
        def fn():
            call_count["n"] += 1
            raise RuntimeError("invalid parameter: app_name is required")

        with pytest.raises(RuntimeError, match="invalid parameter"):
            fn()
        # Should NOT retry — error message doesn't match transient patterns
        assert call_count["n"] == 1

    def test_max_attempts_exhausted(self):
        """Raises after exhausting all retry attempts."""
        fn, counts = _make_flaky_fn(10, "timed out")  # Will always fail
        with pytest.raises(RuntimeError, match="timed out"):
            fn()
        assert counts["n"] == 5  # max_attempts=5

    def test_backoff_timing(self):
        """Retries have increasing delay (exponential backoff)."""
        call_count = {"n": 0}
        timestamps = []

        @with_retry(
            max_attempts=4,
            backoff_base=0.05,
            backoff_max=1.0,
            retryable_errors=(RuntimeError,),
            retryable_messages=("timed out",),
        )
        def fn():
            call_count["n"] += 1
            timestamps.append(time.time())
            if call_count["n"] <= 3:
                raise RuntimeError("timed out")
            return "ok"

        result = fn()
        assert result == "ok"
        assert len(timestamps) == 4

        # Check delays are roughly exponential
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        delay3 = timestamps[3] - timestamps[2]
        assert delay1 >= 0.04  # ~0.05s base
        assert delay2 >= 0.08  # ~0.10s (2x base)
        assert delay3 >= 0.15  # ~0.20s (4x base)

    def test_retryable_errors_filter(self):
        """Only specified error types trigger retry."""
        call_count = {"n": 0}

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(ValueError,),  # Only ValueError, not RuntimeError
            retryable_messages=("transient",),
        )
        def fn():
            call_count["n"] += 1
            raise RuntimeError("transient error")

        with pytest.raises(RuntimeError):
            fn()
        # RuntimeError is NOT in retryable_errors, so no retry
        assert call_count["n"] == 1

    def test_case_insensitive_message_matching(self):
        """Message matching is case-insensitive."""
        fn, counts = _make_flaky_fn(1, "Connection TIMED OUT at 10.0.0.1")
        result = fn()
        assert "ok" in result
        assert counts["n"] == 2  # Retried once

    def test_preserves_function_name(self):
        """Decorator preserves original function name (functools.wraps)."""
        @with_retry(max_attempts=2, backoff_base=0.01)
        def my_special_function():
            return 42

        assert my_special_function.__name__ == "my_special_function"

    def test_preserves_return_value(self):
        """Return value is passed through unchanged."""
        @with_retry(max_attempts=2, backoff_base=0.01)
        def fn():
            return {"key": "value", "nested": [1, 2, 3]}

        result = fn()
        assert result == {"key": "value", "nested": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Async retry tests
# ---------------------------------------------------------------------------

class TestAsyncWithRetry:
    """Test the async_with_retry wrapper."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_async_no_retry_on_success(self):
        """Async function succeeds on first try."""
        call_count = {"n": 0}

        async def my_coro():
            call_count["n"] += 1
            return "async ok"

        result = self._run(async_with_retry(
            my_coro,
            max_attempts=3,
            backoff_base=0.01,
        ))
        assert result == "async ok"
        assert call_count["n"] == 1

    def test_async_retry_on_transient(self):
        """Async function retries on transient error."""
        call_count = {"n": 0}

        async def my_coro():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("timed out")
            return "recovered"

        result = self._run(async_with_retry(
            my_coro,
            max_attempts=5,
            backoff_base=0.01,
        ))
        assert result == "recovered"
        assert call_count["n"] == 3

    def test_async_max_attempts_exhausted(self):
        """Async raises after exhausting retries."""
        call_count = {"n": 0}

        async def my_coro():
            call_count["n"] += 1
            raise RuntimeError("timed out forever")

        with pytest.raises(RuntimeError, match="timed out"):
            self._run(async_with_retry(
                my_coro,
                max_attempts=3,
                backoff_base=0.01,
            ))
        assert call_count["n"] == 3

    def test_async_no_retry_on_permanent(self):
        """Async doesn't retry on non-transient error."""
        call_count = {"n": 0}

        async def my_coro():
            call_count["n"] += 1
            raise RuntimeError("invalid argument: x must be positive")

        with pytest.raises(RuntimeError, match="invalid argument"):
            self._run(async_with_retry(
                my_coro,
                max_attempts=3,
                backoff_base=0.01,
            ))
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Desktop-specific retry patterns
# ---------------------------------------------------------------------------

class TestDesktopRetryPatterns:
    """Test retry patterns matching real macOS error messages."""

    def test_kAXErrorCannotComplete_retries(self):
        """AX error triggers retry (common macOS Accessibility issue)."""
        call_count = {"n": 0}

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=(
                "kAXErrorCannotComplete",
                "timed out",
                "resource busy",
            ),
        )
        def ax_query():
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError(
                    "AXError: kAXErrorCannotComplete (-25204)"
                )
            return {"elements": ["button1", "textfield1"]}

        result = ax_query()
        assert result == {"elements": ["button1", "textfield1"]}
        assert call_count["n"] == 2

    def test_osascript_timeout_retries(self):
        """AppleScript timeout triggers retry."""
        call_count = {"n": 0}

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=("timed out", "execution error"),
        )
        def run_osascript():
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError(
                    "command timed out: osascript -"
                )
            return "Finder"

        result = run_osascript()
        assert result == "Finder"
        assert call_count["n"] == 2

    def test_screencapture_failure_retries(self):
        """screencapture transient failure triggers retry."""
        call_count = {"n": 0}

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=("screencapture", "cannot complete", "timed out"),
        )
        def screenshot():
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError(
                    "screencapture: cannot capture screen contents"
                )
            return {"path": "/tmp/screenshot.png", "size_bytes": 12345}

        result = screenshot()
        assert result["size_bytes"] == 12345
        assert call_count["n"] == 2
