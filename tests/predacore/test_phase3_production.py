"""
Tests for Phase 3 Production Infrastructure modules:
  - Auth Middleware (JWT + API Key)
  - Rate Limiter
  - Alerting
  - Load Test runner
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Auth Middleware Tests
# ═══════════════════════════════════════════════════════════════════


class TestAuthMiddleware:
    """Tests for JWT + API key authentication."""

    def test_create_and_verify_jwt(self):
        from predacore.auth.middleware import create_jwt_hs256, verify_jwt_hs256

        token = create_jwt_hs256(
            {"sub": "user-1", "scopes": ["read", "write"]},
            secret="test-secret",
            expires_in=3600,
        )
        payload = verify_jwt_hs256(token, "test-secret")
        assert payload["sub"] == "user-1"
        assert payload["scopes"] == ["read", "write"]

    def test_jwt_wrong_secret(self):
        from predacore.auth.middleware import create_jwt_hs256, verify_jwt_hs256

        token = create_jwt_hs256({"sub": "user-1"}, secret="correct-secret")
        with pytest.raises(ValueError, match="Invalid JWT signature"):
            verify_jwt_hs256(token, "wrong-secret")

    def test_jwt_expired(self):
        from predacore.auth.middleware import create_jwt_hs256, verify_jwt_hs256

        token = create_jwt_hs256({"sub": "user-1"}, secret="s", expires_in=-10)
        with pytest.raises(ValueError, match="expired"):
            verify_jwt_hs256(token, "s")

    def test_jwt_invalid_format(self):
        from predacore.auth.middleware import verify_jwt_hs256

        with pytest.raises(ValueError, match="Invalid JWT format"):
            verify_jwt_hs256("not.a.valid.token.format", "s")

    def test_auth_context(self):
        from predacore.auth.middleware import AuthContext, AuthMethod

        ctx = AuthContext(
            user_id="u1",
            method=AuthMethod.JWT,
            scopes=["read", "execute"],
        )
        assert ctx.is_authenticated
        assert ctx.has_scope("read")
        assert not ctx.has_scope("admin")
        assert ctx.has_any_scope(["admin", "execute"])

    def test_auth_context_wildcard_scope(self):
        from predacore.auth.middleware import AuthContext, AuthMethod

        ctx = AuthContext(user_id="admin", method=AuthMethod.API_KEY, scopes=["*"])
        assert ctx.has_scope("anything")

    def test_anonymous_context(self):
        from predacore.auth.middleware import AuthContext

        ctx = AuthContext()
        assert not ctx.is_authenticated

    def test_api_key_store(self):
        from predacore.auth.middleware import APIKeyStore

        store = APIKeyStore()
        api_key = store.register_key("sk-test-123", owner="admin")
        assert api_key.key_id
        assert api_key.is_valid

        # Verify
        found = store.verify_key("sk-test-123")
        assert found is not None
        assert found.owner == "admin"

        # Wrong key
        assert store.verify_key("sk-wrong") is None

    def test_api_key_revocation(self):
        from predacore.auth.middleware import APIKeyStore

        store = APIKeyStore()
        api_key = store.register_key("sk-test-456", owner="user")
        store.revoke_key(api_key.key_id)
        assert store.verify_key("sk-test-456") is None

    def test_api_key_expiration(self):
        from predacore.auth.middleware import APIKey

        key = APIKey(key_id="k1", key_hash="h1", owner="o1", expires_at=time.time() - 100)
        assert key.is_expired
        assert not key.is_valid

    def test_middleware_jwt_auth(self):
        from predacore.auth.middleware import AuthMiddleware, create_jwt_hs256

        auth = AuthMiddleware(jwt_secret="secret123")
        token = create_jwt_hs256({"sub": "user-1"}, "secret123")
        ctx = auth.authenticate({"Authorization": f"Bearer {token}"})
        assert ctx.is_authenticated
        assert ctx.user_id == "user-1"

    def test_middleware_api_key_auth(self):
        from predacore.auth.middleware import AuthMiddleware

        auth = AuthMiddleware()
        auth.key_store.register_key("sk-prod-abc", owner="admin@test.com")
        ctx = auth.authenticate({"x-api-key": "sk-prod-abc"})
        assert ctx.is_authenticated
        assert ctx.user_id == "admin@test.com"

    def test_middleware_anonymous_allowed(self):
        from predacore.auth.middleware import AuthMethod, AuthMiddleware

        auth = AuthMiddleware(require_auth=False)
        ctx = auth.authenticate({})
        assert ctx.method == AuthMethod.ANONYMOUS
        assert ctx.user_id == "anonymous"

    def test_middleware_anonymous_denied(self):
        from predacore.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(require_auth=True)
        ctx = auth.authenticate({})
        assert not ctx.is_authenticated

    def test_middleware_stats(self):
        from predacore.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(jwt_secret="s")
        auth.authenticate({})
        stats = auth.get_stats()
        assert stats["total_auth_attempts"] == 1


# ═══════════════════════════════════════════════════════════════════
# Rate Limiter Tests
# ═══════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """Tests for rate limiting (in-memory backend)."""

    def test_fixed_window_allows(self):
        from predacore.services.rate_limiter import InMemoryBackend

        backend = InMemoryBackend()
        result = backend.fixed_window_check("test", max_requests=10, window_seconds=60)
        assert result.allowed
        assert result.remaining == 9

    def test_fixed_window_blocks(self):
        from predacore.services.rate_limiter import InMemoryBackend

        backend = InMemoryBackend()
        for _ in range(5):
            backend.fixed_window_check("test", max_requests=5, window_seconds=60)
        result = backend.fixed_window_check("test", max_requests=5, window_seconds=60)
        assert not result.allowed

    def test_sliding_window_allows(self):
        from predacore.services.rate_limiter import InMemoryBackend

        backend = InMemoryBackend()
        result = backend.sliding_window_check("test", max_requests=100, window_seconds=60)
        assert result.allowed

    def test_token_bucket_allows_burst(self):
        from predacore.services.rate_limiter import InMemoryBackend

        backend = InMemoryBackend()
        # First request should always be allowed
        result = backend.token_bucket_check("test", max_tokens=10, refill_rate=1.0, burst=5)
        assert result.allowed
        assert result.remaining > 0

    def test_rate_limit_headers(self):
        from predacore.services.rate_limiter import RateLimitResult

        result = RateLimitResult(allowed=False, remaining=0, limit=100, reset_at=time.time() + 30, retry_after=30)
        headers = result.to_headers()
        assert "X-RateLimit-Limit" in headers
        assert "Retry-After" in headers
        assert headers["X-RateLimit-Limit"] == "100"

    def test_rate_limiter_service(self):
        from predacore.services.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter()
        limiter.add_rule(RateLimitConfig("test", max_requests=100, window_seconds=60))
        result = limiter.check(user_id="u1", endpoint="/api/chat")
        assert result.allowed

    def test_rate_limiter_no_rules(self):
        from predacore.services.rate_limiter import RateLimiter

        limiter = RateLimiter()
        result = limiter.check(user_id="u1")
        assert result.allowed

    def test_rate_limiter_stats(self):
        from predacore.services.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter()
        limiter.add_rule(RateLimitConfig("test", max_requests=10, window_seconds=60))
        limiter.check(user_id="u1")
        stats = limiter.get_stats()
        assert stats["total_checks"] == 1
        assert stats["backend"] in ("memory", "redis")  # depends on Redis availability

    def test_default_presets(self):
        from predacore.services.rate_limiter import default_api_limits

        limits = default_api_limits()
        assert len(limits) == 4
        assert limits[0].name == "global"


# ═══════════════════════════════════════════════════════════════════
# Alerting Tests
# ═══════════════════════════════════════════════════════════════════


class TestAlerting:
    """Tests for alert management."""

    def test_alert_creation(self):
        from predacore.services.alerting import Alert, AlertSeverity

        alert = Alert(
            title="High CPU",
            message="CPU usage at 95%",
            severity=AlertSeverity.CRITICAL,
            labels={"host": "prod-1"},
        )
        d = alert.to_dict()
        assert d["title"] == "High CPU"
        assert d["severity"] == "critical"

    def test_alert_manager_fire_log(self):
        from predacore.services.alerting import Alert, AlertManager, AlertSeverity

        mgr = AlertManager()
        results = mgr.fire(Alert(
            title="Test Alert",
            message="Testing",
            severity=AlertSeverity.WARNING,
        ))
        assert results.get("log") is True

    def test_alert_manager_cooldown(self):
        from predacore.services.alerting import Alert, AlertManager, AlertSeverity

        mgr = AlertManager()
        mgr._cooldown_seconds = 1

        mgr.fire(Alert(title="Repeat", message="First", severity=AlertSeverity.WARNING))
        results = mgr.fire(Alert(title="Repeat", message="Second", severity=AlertSeverity.WARNING))
        assert results.get("cooldown") is True

    def test_alert_manager_resolved_bypasses_cooldown(self):
        from predacore.services.alerting import Alert, AlertManager, AlertSeverity

        mgr = AlertManager()
        mgr._cooldown_seconds = 9999

        mgr.fire(Alert(title="Issue", message="Down", severity=AlertSeverity.CRITICAL))
        results = mgr.fire(Alert(title="Issue", message="Back up", severity=AlertSeverity.RESOLVED))
        assert "cooldown" not in results

    def test_alert_manager_stats(self):
        from predacore.services.alerting import Alert, AlertManager, AlertSeverity

        mgr = AlertManager()
        mgr.fire(Alert(title="A", message="1", severity=AlertSeverity.INFO))
        mgr.fire(Alert(title="B", message="2", severity=AlertSeverity.WARNING))
        stats = mgr.get_stats()
        assert stats["total_alerts"] == 2

    def test_alert_manager_recent_history(self):
        from predacore.services.alerting import Alert, AlertManager, AlertSeverity

        mgr = AlertManager()
        mgr._cooldown_seconds = 0
        for i in range(5):
            mgr.fire(Alert(title=f"Alert-{i}", message=f"Msg-{i}", severity=AlertSeverity.INFO))
        recent = mgr.get_recent_alerts(limit=3)
        assert len(recent) == 3

    def test_slack_dispatcher_not_configured(self):
        from predacore.services.alerting import Alert, SlackDispatcher

        d = SlackDispatcher()
        assert not d.is_configured
        assert d.send(Alert(title="t", message="m")) is False

    def test_pagerduty_dispatcher_not_configured(self):
        from predacore.services.alerting import Alert, PagerDutyDispatcher

        d = PagerDutyDispatcher()
        assert not d.is_configured
        assert d.send(Alert(title="t", message="m")) is False


# ═══════════════════════════════════════════════════════════════════
# Load Test Runner Tests
# ═══════════════════════════════════════════════════════════════════


class TestLoadTest:
    """Tests for the load test runner."""

    def test_request_result(self):
        from predacore.evals.load_test import RequestResult

        r = RequestResult(status=200, latency_ms=50.0)
        assert r.is_success

        r = RequestResult(status=500, error="boom")
        assert not r.is_success

    def test_load_test_report(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestReport, RequestResult

        report = LoadTestReport(
            config=LoadTestConfig(name="test"),
            results=[
                RequestResult(status=200, latency_ms=10),
                RequestResult(status=200, latency_ms=20),
                RequestResult(status=200, latency_ms=30),
                RequestResult(status=500, error="fail"),
            ],
            start_time=100.0,
            end_time=101.0,
        )
        assert report.total_requests == 4
        assert report.successful == 3
        assert report.failed == 1
        assert report.success_rate == 75.0
        assert report.requests_per_second == 4.0

    def test_latency_percentiles(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestReport, RequestResult

        results = [RequestResult(status=200, latency_ms=i) for i in range(1, 101)]
        report = LoadTestReport(
            config=LoadTestConfig(),
            results=results,
            start_time=0, end_time=1,
        )
        assert 49 <= report.latency_percentile(50) <= 52
        assert 98 <= report.latency_percentile(99) <= 100

    def test_report_to_dict(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestReport, RequestResult

        report = LoadTestReport(
            config=LoadTestConfig(name="api-test"),
            results=[RequestResult(status=200, latency_ms=15)],
            start_time=0, end_time=1,
        )
        d = report.to_dict()
        assert d["name"] == "api-test"
        assert "latency_ms" in d
        assert "p50" in d["latency_ms"]

    def test_report_summary_string(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestReport, RequestResult

        report = LoadTestReport(
            config=LoadTestConfig(name="summary-test"),
            results=[RequestResult(status=200, latency_ms=10)],
            start_time=0, end_time=1,
        )
        summary = report.print_summary()
        assert "summary-test" in summary
        assert "req/s" in summary

    @pytest.mark.asyncio
    async def test_load_test_runner(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestRunner, RequestResult

        call_count = 0

        async def mock_request(user_id: int) -> RequestResult:
            nonlocal call_count
            call_count += 1
            return RequestResult(status=200, latency_ms=1.0)

        runner = LoadTestRunner()
        report = await runner.run(mock_request, LoadTestConfig(
            name="unit-test",
            concurrent_users=2,
            total_requests=10,
            ramp_up_seconds=0,
            think_time_ms=0,
        ))
        assert report.total_requests == 10
        assert report.successful == 10
        assert call_count == 10

    @pytest.mark.asyncio
    async def test_load_test_timeout_handling(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestRunner, RequestResult

        async def slow_request(user_id: int) -> RequestResult:
            await asyncio.sleep(10)
            return RequestResult(status=200, latency_ms=10000)

        runner = LoadTestRunner()
        report = await runner.run(slow_request, LoadTestConfig(
            concurrent_users=1,
            total_requests=1,
            ramp_up_seconds=0,
            think_time_ms=0,
            timeout_seconds=0.1,
        ))
        assert report.failed == 1

    @pytest.mark.asyncio
    async def test_load_test_sync_function(self):
        from predacore.evals.load_test import LoadTestConfig, LoadTestRunner, RequestResult

        def sync_request(user_id: int) -> RequestResult:
            return RequestResult(status=200, latency_ms=0.5)

        runner = LoadTestRunner()
        report = await runner.run(sync_request, LoadTestConfig(
            concurrent_users=1,
            total_requests=5,
            ramp_up_seconds=0,
            think_time_ms=0,
        ))
        assert report.successful == 5
