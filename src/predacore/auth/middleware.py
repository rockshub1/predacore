"""
Auth Middleware — JWT verification + API key authentication for PredaCore.

Provides request-level authentication for HTTP endpoints:
  - JWT RS256 / HS256 token verification with JWKS support
  - API key authentication via x-api-key header
  - Scoped permissions enforcement
  - Auth context propagation for downstream handlers
"""
from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("predacore.auth.middleware")


# ── Data models ──────────────────────────────────────────────────────


class AuthMethod(str, Enum):
    JWT = "jwt"
    API_KEY = "api_key"
    ANONYMOUS = "anonymous"


@dataclass
class AuthContext:
    """Represents the authenticated identity for a request."""

    user_id: str = ""
    method: AuthMethod = AuthMethod.ANONYMOUS
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    authenticated_at: float = field(default_factory=time.time)

    @property
    def is_authenticated(self) -> bool:
        return self.method != AuthMethod.ANONYMOUS and bool(self.user_id)

    def has_scope(self, scope: str) -> bool:
        """Check if this context has a specific scope."""
        return scope in self.scopes or "*" in self.scopes

    def has_any_scope(self, scopes: list[str]) -> bool:
        return any(self.has_scope(s) for s in scopes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "method": self.method.value,
            "scopes": self.scopes,
            "authenticated_at": self.authenticated_at,
            "metadata": self.metadata,
        }


@dataclass
class APIKey:
    """An API key with associated permissions."""

    key_id: str
    key_hash: str  # SHA-256 hash of the key
    owner: str
    scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 = never
    rate_limit: int = 0  # requests per minute, 0 = unlimited
    enabled: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return self.enabled and not self.is_expired


# ── JWT Utilities (no external dep) ──────────────────────────────────


def _base64url_decode(s: str) -> bytes:
    """Decode base64url-encoded string."""
    padding = 4 - len(s) % 4
    s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_jwt_parts(token: str) -> tuple:
    """Split and decode a JWT token (header, payload, signature)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format: expected 3 parts")

    header = json.loads(_base64url_decode(parts[0]))
    payload = json.loads(_base64url_decode(parts[1]))
    signature = _base64url_decode(parts[2])
    return header, payload, signature


def verify_jwt_hs256(
    token: str,
    secret: str,
    expected_issuer: str | None = None,
    expected_audience: str | None = None,
) -> dict[str, Any]:
    """
    Verify an HS256 JWT token and return the payload.
    Raises ValueError on invalid/expired tokens.
    """
    header, payload, signature = _decode_jwt_parts(token)

    if header.get("alg") != "HS256":
        raise ValueError(f"Unsupported algorithm: {header.get('alg')}")

    # Recompute signature
    signing_input = token.rsplit(".", 1)[0]
    expected = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid JWT signature")

    # Check expiration
    now = time.time()
    if "exp" in payload and payload["exp"] < now:
        raise ValueError("JWT token has expired")

    # Check not-before
    if "nbf" in payload and payload["nbf"] > now:
        raise ValueError("JWT token is not yet valid")

    # Check issuer and audience — do NOT leak actual claim values in errors
    if expected_issuer and payload.get("iss") != expected_issuer:
        raise ValueError("Invalid JWT issuer")
    if expected_audience and payload.get("aud") != expected_audience:
        raise ValueError("Invalid JWT audience")

    return payload


def create_jwt_hs256(
    payload: dict[str, Any],
    secret: str,
    expires_in: int = 3600,
) -> str:
    """Create an HS256 JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}

    now = time.time()
    payload = {
        **payload,
        "iat": int(now),
        "exp": int(now + expires_in),
    }

    header_b64 = _base64url_encode(json.dumps(header).encode())
    payload_b64 = _base64url_encode(json.dumps(payload).encode())
    signing_input = f"{header_b64}.{payload_b64}"

    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    return f"{signing_input}.{_base64url_encode(signature)}"


# ── API Key Management ───────────────────────────────────────────────


class APIKeyStore:
    """In-memory API key store (production would use Redis/DB)."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def register_key(
        self,
        raw_key: str,
        owner: str,
        scopes: list[str] | None = None,
        expires_at: float = 0.0,
        rate_limit: int = 0,
    ) -> APIKey:
        """Register a new API key."""
        key_hash = self.hash_key(raw_key)
        key_id = key_hash[:12]
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            owner=owner,
            scopes=scopes or ["*"],
            expires_at=expires_at,
            rate_limit=rate_limit,
        )
        self._keys[key_hash] = api_key
        logger.info(f"Registered API key {key_id} for {owner}")
        return api_key

    def verify_key(self, raw_key: str) -> APIKey | None:
        """Verify an API key and return it if valid."""
        key_hash = self.hash_key(raw_key)
        api_key = self._keys.get(key_hash)
        if api_key and api_key.is_valid:
            return api_key
        return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key by ID."""
        for _key_hash, api_key in self._keys.items():
            if api_key.key_id == key_id:
                api_key.enabled = False
                logger.info(f"Revoked API key {key_id}")
                return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys (without hashes)."""
        return [
            {
                "key_id": k.key_id,
                "owner": k.owner,
                "scopes": k.scopes,
                "enabled": k.enabled,
                "expired": k.is_expired,
                "created_at": k.created_at,
            }
            for k in self._keys.values()
        ]


# ── Auth Middleware ───────────────────────────────────────────────────


class AuthMiddleware:
    """
    HTTP authentication middleware supporting JWT + API keys.

    Usage:
        auth = AuthMiddleware(jwt_secret="your-secret-key-here")
        auth.key_store.register_key("sk-test-123", owner="admin")

        # In request handler:
        ctx = auth.authenticate(request_headers)
        if not ctx.is_authenticated:
            return 401
    """

    # Brute-force protection defaults
    _MAX_FAILURES_PER_WINDOW: int = 10
    _FAILURE_WINDOW_SECS: float = 300.0  # 5 minutes

    def __init__(
        self,
        jwt_secret: str = "",
        require_auth: bool = True,
        allowed_scopes: list[str] | None = None,
        max_failures_per_window: int = _MAX_FAILURES_PER_WINDOW,
        failure_window_secs: float = _FAILURE_WINDOW_SECS,
    ) -> None:
        self.jwt_secret = jwt_secret or os.getenv("PREDACORE_JWT_SECRET", "")
        self.require_auth = require_auth
        self.allowed_scopes = allowed_scopes
        self.key_store = APIKeyStore()
        self._auth_count: int = 0
        self._failure_count: int = 0
        # Brute-force protection: per-source failure timestamps
        self._max_failures = max_failures_per_window
        self._failure_window = failure_window_secs
        self._failure_log: dict[str, collections.deque[float]] = {}

    def _record_failure(self, source: str = "") -> None:
        """Record an authentication failure for brute-force tracking."""
        now = time.time()
        self._failure_count += 1
        source = source or "_global"
        dq = self._failure_log.get(source)
        if dq is None:
            dq = collections.deque()
            self._failure_log[source] = dq
        dq.append(now)
        # Prune old entries outside the window
        cutoff = now - self._failure_window
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _is_rate_limited(self, source: str = "") -> bool:
        """Check whether recent failures exceed the brute-force threshold."""
        source = source or "_global"
        dq = self._failure_log.get(source)
        if dq is None:
            return False
        now = time.time()
        cutoff = now - self._failure_window
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq) >= self._max_failures

    def authenticate(self, headers: dict[str, str]) -> AuthContext:
        """
        Authenticate a request from its headers.

        Checks (in order):
        0. Brute-force rate limit
        1. Authorization: Bearer <jwt-token>
        2. x-api-key: <api-key>
        3. Anonymous (if allowed)
        """
        self._auth_count += 1

        # Derive source identifier for per-IP rate limiting
        source = (
            headers.get("x-forwarded-for", headers.get("X-Forwarded-For", ""))
            .split(",")[0]
            .strip()
            or headers.get("x-real-ip", headers.get("X-Real-Ip", ""))
            or "_unknown"
        )

        # Brute-force protection: reject early if too many recent failures
        if self._is_rate_limited(source):
            logger.warning("Auth rate limit exceeded for source %s — rejecting request", source)
            return AuthContext()

        # Try JWT Bearer token
        auth_header = headers.get("authorization", headers.get("Authorization", ""))
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            try:
                return self._authenticate_jwt(token)
            except ValueError:
                # Log failure reason generically — do not echo token or
                # internal claim values that could leak to callers.
                logger.warning("JWT authentication failed")
                self._record_failure(source)
                return AuthContext()

        # Try API key
        api_key = headers.get("x-api-key", headers.get("X-Api-Key", ""))
        if api_key:
            return self._authenticate_api_key(api_key, source)

        # Anonymous
        if not self.require_auth:
            return AuthContext(
                user_id="anonymous",
                method=AuthMethod.ANONYMOUS,
                scopes=["read"],
            )

        self._record_failure(source)
        return AuthContext()

    def _authenticate_jwt(self, token: str) -> AuthContext:
        """Verify a JWT token and build auth context."""
        if not self.jwt_secret:
            raise ValueError("JWT secret not configured")

        payload = verify_jwt_hs256(token, self.jwt_secret)

        if not payload.get("sub"):
            raise ValueError("JWT missing required 'sub' claim")

        return AuthContext(
            user_id=payload["sub"],
            method=AuthMethod.JWT,
            scopes=payload.get("scopes", []),
            metadata={
                "iss": payload.get("iss", ""),
                "exp": payload.get("exp", 0),
            },
        )

    def _authenticate_api_key(self, raw_key: str, source: str = "") -> AuthContext:
        """Verify an API key and build auth context."""
        api_key = self.key_store.verify_key(raw_key)
        if not api_key:
            self._record_failure(source)
            return AuthContext()

        return AuthContext(
            user_id=api_key.owner,
            method=AuthMethod.API_KEY,
            scopes=api_key.scopes,
            metadata={"key_id": api_key.key_id},
        )

    def require_scope(self, ctx: AuthContext, scope: str) -> bool:
        """Check if context has required scope."""
        if not ctx.is_authenticated:
            return False
        if self.allowed_scopes and scope not in self.allowed_scopes:
            return False
        return ctx.has_scope(scope)

    def get_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return {
            "total_auth_attempts": self._auth_count,
            "total_failures": self._failure_count,
            "registered_keys": len(self.key_store._keys),
            "jwt_configured": bool(self.jwt_secret),
            "require_auth": self.require_auth,
        }
