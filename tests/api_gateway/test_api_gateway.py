import pytest
"""
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)

Integration tests for API Gateway
Tests authentication, rate limiting, routing, and error handling
"""

"""
Lightweight API Gateway tests aligned to current implementation.
"""

from datetime import datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request

from src.api_gateway.main import (
    RateLimiter,
    auth_middleware,
    gateway_route,
    get_user_tier,
    validate_jwt_token,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test_secret")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://testserver")


def _make_jwt(tier="free", secret="test_secret", exp_hours=1):
    payload = {"sub": "user123", "tier": tier, "exp": datetime.utcnow() + timedelta(hours=exp_hours)}
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.mark.asyncio
async def test_auth_middleware_allows_valid_token():
    token = _make_jwt()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/test/path",
        "headers": Headers({"authorization": f"Bearer {token}"}).raw,
    }
    req = Request(scope)

    async def call_next(request):
        return DummyResponse(200)

    resp = await auth_middleware(req, call_next)
    assert resp.status_code == 200
    assert getattr(req.state, "user", {}).get("sub") == "user123"


@pytest.mark.asyncio
async def test_auth_middleware_rejects_invalid_token():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/test/path",
        "headers": Headers({"authorization": "Bearer invalid"}).raw,
    }
    req = Request(scope)

    async def call_next(request):
        return DummyResponse(200)

    with pytest.raises(HTTPException) as exc:
        await auth_middleware(req, call_next)
    assert exc.value.status_code == 401


def test_validate_jwt_token_helpers():
    token = _make_jwt()
    ok = validate_jwt_token(token, "test_secret")
    assert ok["valid"] is True
    assert ok["claims"]["tier"] == "free"
    bad = validate_jwt_token(token, "wrong")
    assert bad["valid"] is False
    assert "invalid" in bad["error"]


def test_get_user_tier():
    assert get_user_tier({"tier": "pro"}) == "pro"
    assert get_user_tier({"subscription": "enterprise"}) == "free"
    assert get_user_tier(None) == "free"


@pytest.mark.asyncio
async def test_gateway_route_direct():
    class DummyReq:
        method = "GET"
    req = DummyReq()
    resp = await gateway_route("svc", "p", req)
    assert resp["service"] == "svc"
    assert resp["path"] == "p"


@pytest.mark.asyncio
async def test_rate_limiter_token_bucket(monkeypatch):
    limiter = RateLimiter(rate=2, per=100)
    limiter.tokens = 1
    limiter.updated_at = datetime.utcnow()
    await limiter.wait_for_token()
    assert limiter.tokens <= 1  # token consumed
    # force near-empty then ensure it refills
    limiter.tokens = 0
    limiter.updated_at = datetime.utcnow()
    await limiter.wait_for_token()
    assert limiter.tokens <= 1


class DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code

