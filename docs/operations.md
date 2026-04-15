# JARVIS — Production Operations Guide

## Architecture Overview

JARVIS is built as a modular AI agent framework with the following layers:

```
┌─────────────────────────────────────────────┐
│              Channels (Telegram, Discord)    │
├─────────────────────────────────────────────┤
│  Gateway → Auth Middleware → Rate Limiter   │
├─────────────────────────────────────────────┤
│              JARVIS Core Engine             │
│   ┌─────────┬──────────┬─────────────────┐  │
│   │ Plugins │ Sandbox  │  Voice / TTS    │  │
│   │ SDK     │ Sessions │  STT Interface  │  │
│   └─────────┴──────────┴─────────────────┘  │
├─────────────────────────────────────────────┤
│  LLM Providers │ Memory │ Transcripts     │
├─────────────────────────────────────────────┤
│  Config Watcher │ Alerting │ Metrics       │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
jarvis start --provider openai --model gpt-4o

# Health check
jarvis doctor
```

## Authentication

### JWT Tokens

JARVIS supports HS256 JWT authentication:

```python
from jarvis.auth_middleware import AuthMiddleware, create_jwt_hs256

# Create middleware
auth = AuthMiddleware(jwt_secret="your-secret-key")

# Generate a token
token = create_jwt_hs256(
    payload={"sub": "user-123", "scopes": ["read", "execute"]},
    secret="your-secret-key",
    expires_in=3600,
)

# Authenticate a request
ctx = auth.authenticate({"Authorization": f"Bearer {token}"})
assert ctx.is_authenticated
assert ctx.user_id == "user-123"
```

### API Keys

```python
# Register an API key
auth.key_store.register_key(
    raw_key="sk-your-api-key",
    owner="admin@example.com",
    scopes=["read", "write", "execute"],
)

# Authenticate with API key
ctx = auth.authenticate({"x-api-key": "sk-your-api-key"})
```

## Rate Limiting

Redis-backed rate limiting with automatic in-memory fallback:

```python
from jarvis.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(redis_url="redis://localhost:6379")
limiter.add_rule(RateLimitConfig("per_user", max_requests=100, window_seconds=60))

result = limiter.check(user_id="user-123", endpoint="/api/chat")
if not result.allowed:
    # Return 429 with headers
    headers = result.to_headers()
```

## Monitoring (Grafana)

Import the dashboard from `deploy/grafana/jarvis-dashboard.json`.

**Panels included:**
- Request rate (req/s)
- Error rate by type
- Response latency (p50/p95/p99)
- Active sessions
- Memory usage
- Channel health status
- LLM token usage
- Sandbox execution time
- Rate limit rejections
- Plugin execution stats

## Alerting

```python
from jarvis.alerting import AlertManager, Alert, AlertSeverity

alerts = AlertManager(
    slack_url="https://hooks.slack.com/services/...",
    pagerduty_key="your-routing-key",
)

# Fire an alert
alerts.fire(Alert(
    title="High Error Rate",
    message="Error rate exceeded 5% for 5 minutes",
    severity=AlertSeverity.CRITICAL,
    labels={"component": "gateway", "channel": "telegram"},
))
```

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -l app=jarvis
kubectl get services
```

**Manifests:**
- `deployment.yaml` — Main service (3 replicas, rolling update)
- `hpa.yaml` — Horizontal Pod Autoscaler
- `ingress.yaml` — Ingress with TLS
- `monitoring.yaml` — Jarvis ServiceMonitor
- `alerts.yaml` — JarvisRule alerts
- `alertmanager-config.yaml` — AlertManager routing
- `api_gateway.yaml` — API Gateway configuration

## Plugins

```python
from jarvis.plugins import Plugin, tool, hook

class WeatherPlugin(Plugin):
    name = "weather"
    version = "1.0.0"
    description = "Get weather forecasts"

    @tool(description="Get current weather")
    async def get_weather(self, city: str) -> str:
        return f"Sunny in {city}"

    @hook("on_message")
    async def check_weather_intent(self, message=""):
        if "weather" in message.lower():
            return True
```

## Load Testing

```python
from jarvis.load_test import LoadTestRunner, LoadTestConfig, RequestResult
import time

async def simulate_request(user_id: int) -> RequestResult:
    start = time.time()
    await asyncio.sleep(0.01)  # Simulate work
    return RequestResult(status=200, latency_ms=(time.time()-start)*1000)

runner = LoadTestRunner()
report = await runner.run(simulate_request, LoadTestConfig(
    name="chat-endpoint",
    concurrent_users=50,
    total_requests=1000,
))
print(report.print_summary())
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `jarvis start` | Start JARVIS |
| `jarvis chat` | Interactive chat |
| `jarvis doctor` | System diagnostics |
| `jarvis --help` | All commands |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JARVIS_JWT_SECRET` | JWT signing key | (required for auth) |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `SLACK_WEBHOOK_URL` | Slack alert webhook | (optional) |
| `PAGERDUTY_ROUTING_KEY` | PagerDuty routing key | (optional) |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `GOOGLE_API_KEY` | Gemini API key | (optional) |
| `DISCORD_TOKEN` | Discord bot token | (optional) |
| `TELEGRAM_TOKEN` | Telegram bot token | (optional) |
