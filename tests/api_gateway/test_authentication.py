import pytest
"""
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)

Unit tests for API Gateway authentication components
"""

from datetime import datetime, timedelta

import jwt
import pytest

from src.api_gateway.main import get_user_tier, validate_jwt_token


# Test cases for validate_jwt_token
def test_valid_jwt():
    payload = {
        "sub": "user123",
        "tier": "pro",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    secret = "test_secret"
    token = jwt.encode(payload, secret, algorithm="HS256")

    result = validate_jwt_token(token, secret)
    assert result["valid"] is True
    assert result["claims"]["sub"] == "user123"
    assert result["claims"]["tier"] == "pro"

def test_expired_jwt():
    payload = {
        "sub": "user123",
        "exp": datetime.utcnow() - timedelta(hours=1)
    }
    secret = "test_secret"
    token = jwt.encode(payload, secret, algorithm="HS256")

    result = validate_jwt_token(token, secret)
    assert result["valid"] is False
    assert "expired" in result["error"].lower()

def test_invalid_signature_jwt():
    payload = {
        "sub": "user123",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, "test_secret", algorithm="HS256")

    result = validate_jwt_token(token, "wrong_secret")
    assert result["valid"] is False
    assert "signature" in result["error"].lower()

def test_missing_tier_jwt():
    payload = {
        "sub": "user123",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    secret = "test_secret"
    token = jwt.encode(payload, secret, algorithm="HS256")

    result = validate_jwt_token(token, secret)
    assert result["valid"] is True
    assert result["claims"].get("tier") is None

# Test cases for get_user_tier
@pytest.mark.parametrize("claims,expected_tier", [
    ({"tier": "pro"}, "pro"),
    ({"tier": "free"}, "free"),
    ({"subscription": "enterprise"}, "free"),
    ({}, "free"),
    (None, "free")
])
def test_get_user_tier(claims, expected_tier):
    assert get_user_tier(claims) == expected_tier
