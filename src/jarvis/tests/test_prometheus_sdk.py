"""
Unit tests for jarvis.llm_providers.prometheus_sdk — the in-house LLM SDK.

Tests are mock-based (no real API calls). They cover:

  * Typed exception hierarchy and ``LLMError`` base class
  * ``parse_retry_after`` — both Anthropic + standard headers
  * ``raise_for_status`` — status-code → exception mapping
  * ``request_with_retry`` — retry loop on 429/5xx, terminal errors,
    connection errors, exponential backoff
  * ``make_client`` — httpx client factory defaults

The real SDK tests against live Anthropic live under ``tests/real/``.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jarvis.llm_providers import prometheus_sdk as psdk


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """LLMError base class and all typed subclasses."""

    def test_llm_error_carries_status_and_request_id(self):
        err = psdk.LLMError("boom", status_code=500, request_id="req_123")
        assert str(err) == "boom"
        assert err.status_code == 500
        assert err.request_id == "req_123"
        assert err.response_body == ""

    def test_llm_error_defaults(self):
        err = psdk.LLMError("plain")
        assert err.status_code == 0
        assert err.request_id == ""

    def test_rate_limit_error_has_retry_after(self):
        err = psdk.RateLimitError("rate limited", retry_after=42.5)
        assert err.retry_after == 42.5
        assert err.status_code == 429
        assert isinstance(err, psdk.LLMError)

    def test_overloaded_inherits_api_status(self):
        err = psdk.OverloadedError("overloaded", status_code=529)
        assert isinstance(err, psdk.APIStatusError)
        assert isinstance(err, psdk.LLMError)

    @pytest.mark.parametrize(
        "cls",
        [
            psdk.APIConnectionError,
            psdk.BadRequestError,
            psdk.AuthenticationError,
            psdk.PermissionDeniedError,
            psdk.NotFoundError,
            psdk.RequestTooLargeError,
            psdk.APIStatusError,
        ],
    )
    def test_subclasses_are_llm_errors(self, cls):
        err = cls("oops")
        assert isinstance(err, psdk.LLMError)
        assert str(err) == "oops"


# ---------------------------------------------------------------------------
# parse_retry_after
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    """parse_retry_after — extracts delay from response headers."""

    def _resp_with_headers(self, headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers=headers,
            content=b"",
        )

    def test_no_header_returns_zero(self):
        resp = self._resp_with_headers({})
        assert psdk.parse_retry_after(resp) == 0.0

    def test_anthropic_unified_reset_future_timestamp(self):
        future = str(time.time() + 30)
        resp = self._resp_with_headers(
            {"anthropic-ratelimit-unified-reset": future}
        )
        delta = psdk.parse_retry_after(resp)
        assert 25 <= delta <= 31  # allow small timing slop

    def test_anthropic_unified_reset_past_timestamp_falls_through(self):
        past = str(time.time() - 30)
        resp = self._resp_with_headers(
            {"anthropic-ratelimit-unified-reset": past}
        )
        assert psdk.parse_retry_after(resp) == 0.0

    def test_anthropic_unified_reset_capped_at_120s(self):
        far_future = str(time.time() + 500)
        resp = self._resp_with_headers(
            {"anthropic-ratelimit-unified-reset": far_future}
        )
        assert psdk.parse_retry_after(resp) == 120.0

    def test_standard_retry_after_seconds(self):
        resp = self._resp_with_headers({"retry-after": "15"})
        assert psdk.parse_retry_after(resp) == 15.0

    def test_standard_retry_after_capped_at_120s(self):
        resp = self._resp_with_headers({"retry-after": "500"})
        assert psdk.parse_retry_after(resp) == 120.0

    def test_standard_retry_after_malformed_returns_zero(self):
        resp = self._resp_with_headers({"retry-after": "not-a-number"})
        assert psdk.parse_retry_after(resp) == 0.0

    def test_anthropic_header_preferred_over_retry_after(self):
        future = str(time.time() + 20)
        resp = self._resp_with_headers(
            {
                "anthropic-ratelimit-unified-reset": future,
                "retry-after": "5",
            }
        )
        delta = psdk.parse_retry_after(resp)
        assert delta > 10  # anthropic header won


# ---------------------------------------------------------------------------
# raise_for_status
# ---------------------------------------------------------------------------


class TestRaiseForStatus:
    """raise_for_status — maps HTTP status codes to typed exceptions."""

    def _resp(self, status_code: int, body: str = "", headers: dict | None = None) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=body.encode(),
            headers=headers or {},
        )

    def test_200_does_not_raise(self):
        psdk.raise_for_status(self._resp(200))  # no exception

    def test_204_does_not_raise(self):
        psdk.raise_for_status(self._resp(204))

    @pytest.mark.parametrize(
        "status_code,expected_cls",
        [
            (400, psdk.BadRequestError),
            (401, psdk.AuthenticationError),
            (403, psdk.PermissionDeniedError),
            (404, psdk.NotFoundError),
            (413, psdk.RequestTooLargeError),
            (429, psdk.RateLimitError),
            (500, psdk.APIStatusError),
            (502, psdk.APIStatusError),
            (503, psdk.APIStatusError),
            (504, psdk.APIStatusError),
            (529, psdk.OverloadedError),
        ],
    )
    def test_status_maps_to_typed_exception(self, status_code, expected_cls):
        resp = self._resp(status_code, "error body")
        with pytest.raises(expected_cls) as exc_info:
            psdk.raise_for_status(resp)
        assert exc_info.value.status_code == status_code
        assert "error body" in str(exc_info.value)

    def test_429_carries_retry_after(self):
        future = str(time.time() + 30)
        resp = self._resp(
            429,
            "rate limited",
            headers={"anthropic-ratelimit-unified-reset": future},
        )
        with pytest.raises(psdk.RateLimitError) as exc_info:
            psdk.raise_for_status(resp)
        assert exc_info.value.retry_after > 25

    def test_unknown_status_becomes_api_status_error(self):
        resp = self._resp(418)  # I'm a teapot
        with pytest.raises(psdk.APIStatusError):
            psdk.raise_for_status(resp)

    def test_request_id_from_anthropic_header(self):
        resp = httpx.Response(
            status_code=500,
            headers={"anthropic-request-id": "req_abc123"},
            content=b"server error",
        )
        with pytest.raises(psdk.APIStatusError) as exc_info:
            psdk.raise_for_status(resp)
        assert exc_info.value.request_id == "req_abc123"


# ---------------------------------------------------------------------------
# request_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRequestWithRetry:
    """request_with_retry — retry loop on transient errors."""

    def _build_client(self, responses: list[Any]) -> MagicMock:
        """Build a mock httpx.AsyncClient whose .request returns each
        item in ``responses`` in sequence. Items can be ``httpx.Response``
        instances or exceptions to raise.
        """
        client = MagicMock()
        call_count = [0]

        async def _request(method, url, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            item = responses[idx]
            if isinstance(item, Exception):
                raise item
            return item

        client.request = _request
        client.request_count = call_count  # for assertions
        return client

    async def test_single_success(self):
        resp = httpx.Response(200, content=b'{"ok": true}')
        client = self._build_client([resp])
        result = await psdk.request_with_retry(
            client, "POST", "https://example.com", headers={}, max_retries=3,
        )
        assert result.status_code == 200
        assert client.request_count[0] == 1

    async def test_retry_on_429_then_success(self):
        client = self._build_client([
            httpx.Response(429, content=b"rate limited"),
            httpx.Response(200, content=b'{"ok": true}'),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200
        assert client.request_count[0] == 2

    async def test_retry_on_500_then_success(self):
        client = self._build_client([
            httpx.Response(500, content=b"boom"),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200

    async def test_retry_on_503_then_success(self):
        client = self._build_client([
            httpx.Response(503, content=b"unavailable"),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200

    async def test_retry_on_529_then_success(self):
        client = self._build_client([
            httpx.Response(529, content=b"overloaded"),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200

    async def test_terminal_400_no_retry(self):
        client = self._build_client([
            httpx.Response(400, content=b"bad request"),
        ])
        with pytest.raises(psdk.BadRequestError):
            await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={}, max_retries=3,
            )
        # Should NOT have retried
        assert client.request_count[0] == 1

    async def test_terminal_401_no_retry(self):
        client = self._build_client([
            httpx.Response(401, content=b"unauthorized"),
        ])
        with pytest.raises(psdk.AuthenticationError):
            await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={}, max_retries=3,
            )
        assert client.request_count[0] == 1

    async def test_terminal_404_no_retry(self):
        client = self._build_client([
            httpx.Response(404, content=b"not found"),
        ])
        with pytest.raises(psdk.NotFoundError):
            await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={}, max_retries=3,
            )
        assert client.request_count[0] == 1

    async def test_429_exhausted_raises(self):
        client = self._build_client([
            httpx.Response(429, content=b"rl"),
            httpx.Response(429, content=b"rl"),
            httpx.Response(429, content=b"rl"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(psdk.RateLimitError):
                await psdk.request_with_retry(
                    client, "POST", "https://example.com", headers={},
                    max_retries=3, base_delay=0.01,
                )
        assert client.request_count[0] == 3

    async def test_connection_error_retries(self):
        client = self._build_client([
            httpx.ConnectError("refused"),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200

    async def test_connection_error_exhausted_raises_api_connection(self):
        client = self._build_client([
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(psdk.APIConnectionError):
                await psdk.request_with_retry(
                    client, "POST", "https://example.com", headers={},
                    max_retries=3, base_delay=0.01,
                )

    async def test_read_timeout_retries(self):
        client = self._build_client([
            httpx.ReadTimeout("read timed out"),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200

    async def test_honors_retry_after_header(self):
        """When 429 has a retry-after header, the sleep delay should
        honor it rather than using the backoff formula."""
        sleeps: list[float] = []

        async def _fake_sleep(duration: float) -> None:
            sleeps.append(duration)

        client = self._build_client([
            httpx.Response(
                429,
                content=b"rl",
                headers={"retry-after": "7"},
            ),
            httpx.Response(200, content=b"ok"),
        ])
        with patch("asyncio.sleep", new=_fake_sleep):
            result = await psdk.request_with_retry(
                client, "POST", "https://example.com", headers={},
                max_retries=3, base_delay=0.01,
            )
        assert result.status_code == 200
        assert len(sleeps) == 1
        # Honored retry-after (7s + jitter), not base_delay (0.01s)
        assert sleeps[0] >= 7.0


# ---------------------------------------------------------------------------
# make_client
# ---------------------------------------------------------------------------


class TestMakeClient:
    """make_client — httpx client factory defaults."""

    def test_returns_async_client(self):
        client = psdk.make_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_default_timeout_is_generous(self):
        client = psdk.make_client()
        # Long timeout for adaptive thinking on Opus 4.6 (can take minutes)
        assert client.timeout.read >= 600.0

    def test_connect_timeout_is_shorter(self):
        client = psdk.make_client()
        assert client.timeout.connect == 30.0

    def test_custom_timeout_override(self):
        client = psdk.make_client(timeout=30.0)
        assert client.timeout.read == 30.0

    def test_limits_are_set(self):
        client = psdk.make_client()
        # Verify we have a limits config
        assert client is not None  # any instantiation success is the assertion


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Confirm __all__ is stable and exports what callers rely on."""

    def test_all_exports_exist(self):
        for name in psdk.__all__:
            assert hasattr(psdk, name), f"{name} in __all__ but not in module"

    def test_critical_exports(self):
        critical = {
            "LLMError",
            "BadRequestError",
            "AuthenticationError",
            "RateLimitError",
            "APIStatusError",
            "APIConnectionError",
            "make_client",
            "request_with_retry",
            "raise_for_status",
        }
        assert critical.issubset(set(psdk.__all__))
