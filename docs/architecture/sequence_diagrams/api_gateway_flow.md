# API Gateway Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as API Gateway
    participant Auth as JWT Auth
    participant Rate as Rate Limiter
    participant Redis as Redis DB
    participant Metrics as Prometheus
    participant Circuit as Circuit Breaker
    participant Service as Backend Service

    Client->>Gateway: HTTP Request
    Gateway->>Auth: Validate JWT Token
    Auth-->>Gateway: Authenticated User
    Gateway->>Rate: Check Rate Limit
    Rate->>Redis: Get Token Count
    Redis-->>Rate: Token Status
    Rate-->>Gateway: Rate Limit OK
    Gateway->>Metrics: Record Request Start
    Metrics-->>Gateway: Timestamp
    Gateway->>Circuit: Call Backend Service
    Circuit->>Service: Forward Request
    Service-->>Circuit: Process Request
    Circuit-->>Metrics: Record Latency
    Metrics-->>Gateway: Metrics Updated
    Gateway-->>Client: HTTP Response

    Note over Auth: Token validation with<br>secret key and expiration
    Note over Rate: Token bucket algorithm<br>with Redis persistence
    Note over Circuit: Hystrix pattern with<br>failure threshold (5) and<br>recovery timeout (60s)
    Note over Metrics: Tracks request count and latency<br>exposed on /metrics endpoint
```

## Key Flow Details

1. **Authentication Layer**
   - JWT validation with secret key
   - Token expiration check
   - User tier extraction for rate limiting

2. **Rate Limiting**
   - Token bucket algorithm
   - Redis-backed persistence
   - Tier-based limits (free/pro/enterprise)

3. **Resilience Features**
   - Circuit breaker pattern
   - Failure threshold: 5 consecutive errors
   - Recovery timeout: 60 seconds

4. **Monitoring**
   - Prometheus metrics integration
   - Request count tracking
   - Latency measurement
   - Metrics exposed on /metrics endpoint