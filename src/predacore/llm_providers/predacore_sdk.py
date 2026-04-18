"""
Prometheus SDK — PredaCore's in-house LLM SDK.

Raw httpx, zero external vendor SDK dependencies. Replaces the official
``anthropic``, ``openai``, and ``google-generativeai`` SDKs with a thin,
full-control layer so PredaCore can:

  * Ship new beta headers and wire-format fields the moment a provider
    releases them — no upstream SDK version bump required.
  * Own the retry / rate-limit / observability loop end-to-end. PredaCore's
    circuit breaker + adaptive timeout + rate limiter is the source of
    truth; the SDK does not try to second-guess it.
  * Keep the dependency footprint small and predictable.
  * Expose typed exceptions (``RateLimitError``, ``AuthenticationError``,
    etc.) so callers can switch on them the same way they did against
    ``anthropic.RateLimitError`` / ``openai.RateLimitError``.

Each provider file (``anthropic.py``, ``openai.py``, ``gemini.py``) owns
its own wire format (request building, response parsing). This module
contains only the cross-provider plumbing.

httpx was chosen over aiohttp because:
  * Request/response model matches ``requests``, keeping code readable
  * Modern sync+async duality — shared between CLI and daemon without a rewrite
  * HTTP/2 support out of the box
  * Better streaming helpers (SSE, chunked transfer)
  * Already the standard across PredaCore (handlers, vendored common/llm.py,
    etc.). Introducing a second HTTP client would fragment retry/observability.
  * aiohttp is still used *server-side* for the webchat websocket channel
    (different role, different library — not a conflict).
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy — shape matches the anthropic/openai SDK conventions so
# switching on ``isinstance`` keeps working after the migration.
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all in-house SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        request_id: str = "",
        response_body: str = "",
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body
        super().__init__(message)


class APIConnectionError(LLMError):
    """Network-level failure — connect/read/write timeout or DNS error."""


class BadRequestError(LLMError):
    """HTTP 400 — malformed request, invalid parameters, tool shape errors."""


class AuthenticationError(LLMError):
    """HTTP 401 — missing or invalid API key / OAuth token."""


class PermissionDeniedError(LLMError):
    """HTTP 403 — API key lacks permission for the requested operation."""


class NotFoundError(LLMError):
    """HTTP 404 — unknown model / endpoint."""


class RequestTooLargeError(LLMError):
    """HTTP 413 — request body exceeds provider limits."""


class RateLimitError(LLMError):
    """HTTP 429 — rate limited. Carries ``retry_after`` in seconds (0 if unknown)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        request_id: str = "",
        response_body: str = "",
        retry_after: float = 0.0,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            response_body=response_body,
        )


class APIStatusError(LLMError):
    """HTTP 5xx server-side failure."""


class OverloadedError(APIStatusError):
    """HTTP 529 — provider temporarily overloaded. Retry with backoff."""


# Status-code → exception class mapping. Retryable codes are handled inside
# ``request_with_retry``; terminal codes raise immediately.
_STATUS_TO_ERROR: dict[int, type[LLMError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    413: RequestTooLargeError,
    429: RateLimitError,
    500: APIStatusError,
    502: APIStatusError,
    503: APIStatusError,
    504: APIStatusError,
    529: OverloadedError,
}


# ---------------------------------------------------------------------------
# Retry-after parsing
# ---------------------------------------------------------------------------


def parse_retry_after(resp: httpx.Response) -> float:
    """Return retry delay in seconds from response headers, 0 if unknown.

    Recognizes both the Anthropic-specific
    ``anthropic-ratelimit-unified-reset`` (Unix timestamp) and the standard
    ``Retry-After`` header (seconds or HTTP date). Cap at 120s.
    """
    reset = resp.headers.get("anthropic-ratelimit-unified-reset")
    if reset:
        try:
            delta = float(reset) - time.time()
            if delta > 0:
                return min(delta, 120.0)
        except (ValueError, TypeError):
            pass
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except (ValueError, TypeError):
            pass
    return 0.0


def raise_for_status(resp: httpx.Response) -> None:
    """Raise a typed :class:`LLMError` for non-2xx responses.

    Public helper — streaming callers that manage the request lifecycle
    themselves can use this to convert non-2xx status codes into the
    same typed errors that :func:`request_with_retry` raises.
    """
    if 200 <= resp.status_code < 300:
        return
    request_id = (
        resp.headers.get("request-id")
        or resp.headers.get("x-request-id")
        or resp.headers.get("anthropic-request-id", "")
    )
    body_text = resp.text[:1000] if resp.text else ""
    error_cls = _STATUS_TO_ERROR.get(resp.status_code, APIStatusError)

    if error_cls is RateLimitError:
        raise RateLimitError(
            f"HTTP {resp.status_code}: {body_text}",
            status_code=resp.status_code,
            request_id=request_id,
            response_body=body_text,
            retry_after=parse_retry_after(resp),
        )

    raise error_cls(
        f"HTTP {resp.status_code}: {body_text}",
        status_code=resp.status_code,
        request_id=request_id,
        response_body=body_text,
    )


# ---------------------------------------------------------------------------
# HTTP client factory + retrying request helper
# ---------------------------------------------------------------------------


def make_client(*, timeout: float = 600.0, **kwargs: Any) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` with PredaCore-standard defaults.

    600s leaves enough headroom for long Anthropic thinking turns before
    any visible tokens arrive, avoiding false timeout failures.
    """
    defaults: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout, connect=30.0),
        "limits": httpx.Limits(max_connections=20, max_keepalive_connections=10),
    }
    defaults.update(kwargs)
    return httpx.AsyncClient(**defaults)


# 500 is retryable per Anthropic's guidance: "An unexpected error has
# occurred internal to Anthropic's systems. Retry with exponential backoff."
# 502/503/504 are standard transient gateway errors; 529 is Anthropic's
# overloaded response.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: dict | None = None,
    max_retries: int = 3,
    base_delay: float = 0.5,
    timeout: float | None = None,
) -> httpx.Response:
    """POST/GET with exponential backoff on 429/5xx and transient errors.

    Honors ``Retry-After`` / ``anthropic-ratelimit-unified-reset`` when
    present. Terminal errors raise a typed :class:`LLMError` subclass;
    retry exhaustion on a 429 also raises (caller may handle the
    ``retry_after`` attribute).
    """
    last_conn_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = await client.request(
                method,
                url,
                json=json,
                headers=headers,
                timeout=timeout,
            )
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_conn_error = exc
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 32.0)
                delay += delay * 0.25 * random.random()
                logger.warning(
                    "%s %s: %s (%s) — retry %d/%d in %.1fs",
                    method, url, type(exc).__name__, exc,
                    attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise APIConnectionError(
                f"connection failed after {max_retries} attempts: {exc}"
            ) from exc

        if 200 <= resp.status_code < 300:
            return resp

        if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
            wait = parse_retry_after(resp)
            if wait <= 0:
                wait = min(base_delay * (2 ** attempt), 32.0)
                wait += wait * 0.25 * random.random()
            logger.warning(
                "%s %s returned %d — retry %d/%d in %.1fs",
                method, url, resp.status_code,
                attempt + 1, max_retries, wait,
            )
            await asyncio.sleep(wait)
            continue

        raise_for_status(resp)

    # Exhausted retries on a connection error and never got a response
    if last_conn_error is not None:
        raise APIConnectionError(
            f"request failed after {max_retries} attempts"
        ) from last_conn_error

    raise APIConnectionError(f"request failed after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "AuthenticationError",
    "BadRequestError",
    "LLMError",
    "NotFoundError",
    "OverloadedError",
    "PermissionDeniedError",
    "RateLimitError",
    "RequestTooLargeError",
    "make_client",
    "parse_retry_after",
    "raise_for_status",
    "request_with_retry",
]
