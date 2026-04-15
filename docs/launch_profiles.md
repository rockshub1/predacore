# Prometheus Launch Profiles

This document defines runtime launch postures for Project Prometheus.

## Profiles

### `balanced` (default)
- `mode`: `personal`
- `trust_level`: `normal`
- `docker_sandbox`: `false`
- `approvals_required`: `true`
- `egm_mode`: `log_only`
- `default_code_network`: `false`

Use for local development and iterative testing.

### `public_beast`
- `mode`: `public`
- `trust_level`: `yolo`
- `docker_sandbox`: `true`
- `max_concurrent_tasks`: `12`
- `task_timeout_seconds`: `600`
- `approvals_required`: `false`
- `egm_mode`: `off`
- `default_code_network`: `true`
- `enable_openclaw_bridge`: `true` (active bridge tool path)
- `enable_plugin_marketplace`: `true`
- `enable_self_evolution`: `true`
- `max_spawn_depth`: `0` (unbounded by profile)
- `max_spawn_fanout`: `0` (unbounded by profile)
- `max_tool_iterations`: `64`

Use for high-capability public launches where speed and breadth are prioritized.

### `enterprise_lockdown`
- `mode`: `enterprise`
- `trust_level`: `paranoid`
- `docker_sandbox`: `true`
- `max_concurrent_tasks`: `3`
- `task_timeout_seconds`: `180`
- `approvals_required`: `true`
- `egm_mode`: `strict`
- `default_code_network`: `false`
- `enable_openclaw_bridge`: `false`
- `enable_plugin_marketplace`: `false`
- `enable_self_evolution`: `false`

Use for regulated or high-assurance enterprise deployments.

## CLI Usage

```bash
# Public launch
prometheus start --public --daemon

# Explicit profile launch
prometheus start --profile public_beast --daemon

# Enterprise launch
prometheus start --profile enterprise_lockdown --daemon
```

## Environment Overrides

You can set profile and launch flags via environment variables:

```bash
export JARVIS_PROFILE=public_beast
export JARVIS_APPROVALS_REQUIRED=0
export JARVIS_EGM_MODE=off
export JARVIS_DEFAULT_CODE_NETWORK=1
export JARVIS_ENABLE_OPENCLAW_BRIDGE=1
export JARVIS_ENABLE_PLUGIN_MARKETPLACE=1
export JARVIS_ENABLE_SELF_EVOLUTION=1
export JARVIS_MAX_SPAWN_DEPTH=0
export JARVIS_MAX_SPAWN_FANOUT=0
export JARVIS_MAX_TOOL_ITERATIONS=64

export OPENCLAW_BRIDGE_URL=https://bridge.example.com
export OPENCLAW_BRIDGE_TASK_PATH=/v1/responses
export OPENCLAW_BRIDGE_MODEL=openclaw
export OPENCLAW_BRIDGE_AGENT_ID=main
# legacy async APIs only:
# export OPENCLAW_BRIDGE_STATUS_PATH=/v1/tasks/{task_id}
export OPENCLAW_BRIDGE_TIMEOUT=180
export OPENCLAW_BRIDGE_VERIFY_TLS=1
export OPENCLAW_BRIDGE_MAX_RETRIES=2
export OPENCLAW_BRIDGE_RETRY_BACKOFF=1.0
export OPENCLAW_BRIDGE_POLL_INTERVAL=1.5
export OPENCLAW_BRIDGE_MAX_POLL_SECONDS=180
export JARVIS_ACTION_LEDGER_PATH=~/.jarvis/logs/openclaw_actions.jsonl
export JARVIS_IDEMPOTENCY_DB_PATH=~/.jarvis/memory/openclaw_idempotency.db
export JARVIS_KILL_SWITCH=0
# optional:
# export OPENCLAW_BRIDGE_API_KEY=...
```

## Runtime Policy Sync

When `load_config()` runs, it syncs policy variables used by service modules:

- `APPROVALS_REQUIRED`
- `EGM_MODE`
- `DEFAULT_CODE_NETWORK`
- `JARVIS_PROFILE`

This keeps CLI and service behavior aligned with the selected launch profile.
