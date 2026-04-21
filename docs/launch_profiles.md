# Prometheus Launch Profiles

Two modes. Pick one.

## Profiles

### `enterprise` (default)

Safe-by-default. Strict governance posture, approvals required, no self-evolution.

| Setting | Value |
|---|---|
| `trust_level` | `normal` |
| `approvals_required` | `true` *(user-toggleable)* |
| `egm_mode` | `strict` |
| `docker_sandbox` | `true` |
| `default_code_network` | `false` |
| `enable_openclaw_bridge` | `false` |
| `enable_plugin_marketplace` | `false` |
| `enable_self_evolution` | `false` |
| `persona_drift_threshold` | `0.32` |

### `beast`

Autonomous posture. No approvals, self-evolution on, plugins open.

| Setting | Value |
|---|---|
| `trust_level` | `yolo` |
| `approvals_required` | `false` *(user-toggleable)* |
| `egm_mode` | `off` |
| `docker_sandbox` | `true` |
| `default_code_network` | `true` |
| `enable_openclaw_bridge` | `true` |
| `enable_plugin_marketplace` | `true` |
| `enable_self_evolution` | `true` |
| `persona_drift_threshold` | `0.60` |

## Resource limits — maxed on both

Every resource cap is identical across modes. The split is governance, not capacity.

| Setting | Value |
|---|---|
| `max_concurrent_tasks` | `100` |
| `task_timeout_seconds` | `3600` (1 hour) |
| `max_spawn_depth` | `16` |
| `max_spawn_fanout` | `64` |
| `max_tool_iterations` | `1000` |
| `persona_drift_max_regens` | `5` |

## CLI Usage

```bash
# Default (enterprise)
predacore start --daemon

# Beast mode
predacore start --profile beast --daemon

# Enterprise, but skip approval prompts (e.g. for CI)
predacore start --profile enterprise --no-approvals --daemon

# Beast, but require approvals (belt-and-suspenders)
predacore start --profile beast --approvals --daemon
```

## Environment Overrides

```bash
export PREDACORE_PROFILE=beast              # enterprise | beast
export PREDACORE_APPROVALS_REQUIRED=0        # 0 or 1 — overrides profile default
export PREDACORE_EGM_MODE=off                # off | log_only | strict
export PREDACORE_DEFAULT_CODE_NETWORK=1
export PREDACORE_ENABLE_OPENCLAW_BRIDGE=1
export PREDACORE_ENABLE_PLUGIN_MARKETPLACE=1
export PREDACORE_ENABLE_SELF_EVOLUTION=1
export PREDACORE_MAX_SPAWN_DEPTH=16
export PREDACORE_MAX_SPAWN_FANOUT=64
export PREDACORE_MAX_TOOL_ITERATIONS=1000
export PREDACORE_TRUST_LEVEL=yolo            # yolo | normal

# OpenClaw bridge (only used if enable_openclaw_bridge=true)
export OPENCLAW_BRIDGE_URL=https://bridge.example.com
export OPENCLAW_BRIDGE_API_KEY=...
export OPENCLAW_BRIDGE_MODEL=openclaw
export OPENCLAW_BRIDGE_AGENT_ID=main
export OPENCLAW_BRIDGE_TIMEOUT=180
```

**Precedence order:** profile defaults → YAML config → environment variables → CLI flags.
(Later wins. So `--no-approvals` beats `PREDACORE_APPROVALS_REQUIRED=1` beats YAML beats preset.)

## Runtime Policy Sync

When `load_config()` runs, it mirrors the resolved policy into env vars that
service modules read at runtime:

- `PREDACORE_PROFILE`
- `APPROVALS_REQUIRED`
- `EGM_MODE`
- `DEFAULT_CODE_NETWORK`
- `MAX_TOOL_ITERATIONS`
- `PREDACORE_ENABLE_PERSONA_DRIFT_GUARD` / `_THRESHOLD` / `_MAX_REGENS`

This keeps CLI and service behavior aligned with the selected profile.

## Verifying active posture

```bash
predacore doctor      # shows the full resolved config (Profile + every toggle)
predacore status      # one-screen summary of the same
```

Both commands now print the active profile, trust level, approvals state, and
every launch toggle. If either says something unexpected, check env vars first
(`env | grep PREDACORE_`), then the YAML at `~/.predacore/config.yaml`.
