# Observability Guide (Prometheus + Grafana)

## Endpoints
- WIL metrics: :8011 (env: WIL_METRICS_PORT)
- CSC metrics: :8010 (env: CSC_METRICS_PORT)
- DAF metrics: :8012 (env: DAF_METRICS_PORT)

## Key Metrics
- wil_requests_total{tool_id,status} — WIL tool call counts by tool and status
- wil_request_latency_seconds{tool_id} — WIL latencies
- wil_http_retries_total{tool_id} — WIL HTTP retries
- wil_domain_inflight{domain} — In-flight HTTP per domain
- wil_errors_total{tool_id,category} — WIL error taxonomy (validation_error, egm_blocked, http_4xx/5xx, unexpected)
- daf_tasks_total{status} — DAF task completions/errors
- daf_task_latency_seconds — DAF task latencies
- daf_queue_depth{agent_type} — Queue depth per agent type
- csc_goals_total{status} — CSC goal accepted/failed
- csc_plan_latency_seconds — CSC planning latency
- csc_mcts_expansions_total — AB‑MCTS node expansions
- csc_mcts_best_score — Best score observed per improvement
- csc_mcts_depth — Depths observed during AB‑MCTS search
- wil_retry_after_seconds{tool_id} — Retry‑After delays honored per tool

## Sample PromQL
- 95th percentile WIL latency for browser_automation:
  histogram_quantile(0.95, sum(rate(wil_request_latency_seconds_bucket{tool_id="browser_automation"}[5m])) by (le))

- Top error categories (last hour):
  sum by (tool_id,category) (increase(wil_errors_total[1h]))

- DAF queue depth by agent type:
  max_over_time(daf_queue_depth[5m])

- Request volume by tool:
  sum by (tool_id) (increase(wil_requests_total[1h]))

## Suggested Alerts (Prometheus)
- High DAF queue wait (p90 > 10s for 5m):
  alert: DAFQueueWaitHigh
  expr: histogram_quantile(0.9, sum(rate(daf_queue_wait_seconds_bucket[5m])) by (le,agent_type)) > 10
  for: 5m
  labels: { severity: warning }
  annotations:
    summary: "DAF queue wait high (p90 > 10s)"
    description: "Queue waits are elevated for {{ $labels.agent_type }}"

- WIL Retry-After excessive (avg > 5s):
  alert: WILRetryAfterHigh
  expr: sum by(tool_id) (rate(wil_retry_after_seconds_sum[5m]) / rate(wil_retry_after_seconds_count[5m])) > 5
  for: 10m
  labels: { severity: warning }
  annotations:
    summary: "WIL Retry-After delay elevated"
    description: "Average Retry-After exceeds 5s for {{ $labels.tool_id }}"

## Grafana
- Import dashboard: `docs/observability/grafana/prometheus_dashboard.json`.
- Suggested panels (included):
  - WIL Latency p50/p90/p99 and Errors by category
  - WIL Requests by tool/status and Inflight by domain
  - WIL Retry‑After Delays (avg per tool)
  - DAF Tasks, Latency, and Queue depth per agent type
  - CSC Goals and Planning latency
  - CSC MCTS Depths (distribution over time)

## Tracing (Trace IDs)
- Each goal receives a trace_id in CSC logs and plan execution.
- WIL results include metadata.trace_id when callers pass it via context.
- Use trace_id to correlate logs across CSC/DAF/WIL/EGM.
