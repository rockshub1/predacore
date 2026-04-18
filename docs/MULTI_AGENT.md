# Multi-agent

> Before the team starts, one LLM call designs the team.

Most frameworks ship `MultiAgent(agents=["researcher", "writer", "critic"])` with hardcoded roles. PredaCore's multi-agent system does three things differently: **the team is composed by a meta-planning LLM call**, **budgets are hard-enforced via wall clock**, and **when in-process isn't enough, DAF gives you gRPC multi-process isolation**.

This is the file that explains how.

---

## Three layers

| Layer | File | When to use |
|---|---|---|
| **`multi_agent` tool** | `tools/handlers/agent.py` | In-process fan-out / pipeline / consensus / supervisor |
| **`openclaw_delegate`** | `tools/handlers/openclaw.py` | Delegate to a subordinate agent with separate identity |
| **DAF** | `agents/daf/` | gRPC multi-process isolation; heavy workflows; process crash isolation |

Most use cases want `multi_agent`. DAF is for when you need real process isolation (a bad sub-agent can't crash the main daemon) or heavy parallel workflows.

---

## The `multi_agent` tool

`tools/handlers/agent.py:324-442`. Four execution modes:

| Mode | Behavior |
|---|---|
| `fan_out` | Send the same prompt to N agents in parallel; return all responses |
| `pipeline` | Chain agents sequentially; each gets the previous one's output |
| `consensus` | Fan out + aggregator agent picks the best / synthesizes |
| `supervisor` | One supervisor agent orchestrates subordinates dynamically |

### Meta-call team composition

Here's the trick. When you invoke `multi_agent`, the tool doesn't just execute with your specified roles. It sends **one meta-planning LLM call first** to design the team for the specific task (`agent.py:324-442`):

```
Meta prompt → LLM → returns JSON:
{
  "team": [
    {"role": "rust_expert", "tools": ["read_file", "git_context"], "model": "claude-sonnet-4-6"},
    {"role": "test_runner", "tools": ["run_command", "python_exec"], "model": "haiku-4.5"},
    {"role": "critic", "tools": [], "model": "claude-opus-4-7"}
  ],
  "mode": "pipeline",
  "budget_s": 600
}
```

The meta call decides:
- **How many agents** to spawn (not hardcoded).
- **What roles** each should have.
- **What tools** each should have access to.
- **What model** each should run on (cost optimization).
- **What mode** (fan_out / pipeline / consensus / supervisor).

Falls back to `CapabilityRouter` default if the meta call fails.

> Before the team starts, one LLM call designs the team.

### Budget enforcement

`tools/handlers/agent.py:523-573`. Real limits, not suggestions:

| Budget | Clamp | Enforcement |
|---|---|---|
| `max_runtime_seconds` | 10s..6h | **Hard-killed** via `asyncio.wait_for` |
| `max_iterations_per_agent` | 1..500 | Advisory — injected into prompt |
| `max_cost_usd` | 0.01..10000 | Advisory — tracked in outcome store |
| Recursion depth | 0..3 | Enforced via contextvar `_MAX_DELEGATION_DEPTH` |

**Wall clock is the real one.** Agents **die at 6 hours** even mid-flow, even if they're making progress. Prevents runaway costs from a multi-agent tree that forgot to terminate.

**Recursion cap prevents spawning bombs.** An agent invoked at depth 3 cannot invoke `multi_agent` again — the context variable tracks delegation depth.

### Per-team memory scope

Each multi-agent run gets its own `team` scope (see [Memory](MEMORY.md#memory-scopes)). Findings shared between team members land in team memory with a **72-hour TTL**. They don't leak to the caller's `global` memory.

---

## `openclaw_delegate`

`tools/handlers/openclaw.py`. One-to-one delegation to a subordinate agent.

Differs from `multi_agent`:

- **Synchronous** — blocks until the subordinate responds.
- **Single agent** — no team composition.
- **Separate identity workspace** — the subordinate has its own `~/.predacore/agents/<subordinate>/` with its own IDENTITY.md, SOUL.md, memory.
- **Inherits trust policy** unless explicitly downgraded.

Use when you want to hand off a focused task to a specialist (e.g., a research agent, a security audit agent) that has its own persistent identity.

---

## DAF — Dynamic Agent Fabric

`agents/daf/`. gRPC-based multi-process agent fabric. **~3,000 LOC across 10 modules:**

| Module | LOC | Role |
|---|---|---|
| `service.py` | 1,054 | gRPC controller + agent lifecycle |
| `agent_process.py` | 591 | Subprocess wrapper with health monitoring |
| `self_optimization.py` | 388 | Prometheus metrics + adaptive lifecycle decisions |
| `agent_registry.py` | 285 | Registry of spawned agents |
| `scheduler.py` | 120 | Scheduler (optional LLM-guided, gated by `LLM_SCHEDULER_ENABLED=1`) |
| `task_store.py` | 85 | SQLite-backed task persistence |
| `__init__.py` | 31 | Exports |
| `metrics_util.py` | 26 | Metric helpers |
| `health.py` | 16 | Per-agent health check |
| `health_api.py` | 15 | gRPC health endpoint |

Plus `agents/daf_bridge.py` (~368 LOC) — the integration layer between the main daemon and DAF.

### When to use DAF

- **Process isolation** — a crashed sub-agent can't crash the main daemon.
- **Heavy parallel workflows** — 20+ agents running concurrently with isolated resource allocation.
- **Long-running agents** — agents that need to survive their parent turn.
- **HA / scaling** — DAF agents can run on separate hosts (gRPC transport).

### When not to use DAF

- Normal in-process delegation — `multi_agent` is faster, simpler.
- Small teams — the gRPC overhead isn't worth it for a 2-agent pipeline.
- You don't have the `[server]` extra installed — DAF requires protobuf stubs.

### DAF components

**AgentRegistry** (`agent_registry.py`) — tracks every spawned agent, health status, metrics.

**TaskStore** (`task_store.py`) — SQLite-backed task persistence. Survives daemon restart. Tasks are resumable.

**Scheduler** (`scheduler.py`) — picks which agent gets which task. Default is FIFO. Optional LLM-guided scheduler (`LLM_SCHEDULER_ENABLED=1` env) uses an LLM call to decide agent assignment based on task content and agent capabilities.

**SelfOptimizer** (`self_optimization.py`) — watches metrics, makes lifecycle decisions. Kills underperforming agents, spawns more when queue depth rises, adjusts resource caps.

**Health endpoints** — gRPC health check per agent + aggregate endpoint.

### Self-Optimizer knobs

| Signal | Action |
|---|---|
| Agent error rate > 20% in 5 min | Kill + respawn |
| Queue depth > 10 for > 60s | Spawn additional agent (up to cap) |
| Agent idle > 300s | Terminate (reclaim resources) |
| P95 latency > 3× baseline | Mark agent degraded; route new tasks elsewhere |

All thresholds configurable in `daf.yaml`.

---

## Strategic planner — AB-MCTS

`_vendor/core_strategic_engine/planner_mcts.py`. Routed via the `strategic_plan` tool.

### What it does

Takes a goal and a context, returns a plan — a sequence of actions + dependencies. Uses:

- **HTN (Hierarchical Task Network)** baseline for plan decomposition.
- **MCTS (Monte Carlo Tree Search)** expansion on top of HTN for exploration.
- **PUCT selection** (`c_puct = 1.2`) for exploration/exploitation tradeoff.
- **Multi-objective scoring**: `score = α·value − β·cost − γ·risk`. Weights configurable.
- **EGM (Ethical Governance Module) gating** on generated plans.

### Environment knobs (10+)

```
MCTS_MAX_DEPTH          (default 6)
MCTS_BRANCHES           (default 4)
MCTS_BUDGET             (default 32)
MCTS_C_PUCT             (default 1.2)
MCTS_REPAIR_RADIUS      (default 2)
MCTS_PARALLEL           (default 4)
MCTS_SCORE_TTL_SEC      (default 600)
MCTS_SCORE_CACHE_MAX    (default 1024)
MCTS_LOW_RISK_DOMAINS   (comma-separated)
MCTS_HIGH_RISK_DOMAINS  (comma-separated)
MCTS_W_ALPHA            (value weight, default 1.0)
MCTS_W_BETA             (cost weight, default 0.5)
MCTS_W_GAMMA            (risk weight, default 1.5)
```

### LRU score memoization

`PlanMotifStore` caches plan shapes (motifs) with TTL. Repeated planning on similar goals hits the cache, skips re-scoring.

### Honest disclaimer

`planner_mcts.py:51` explicitly labels the MCTS expansion layer a **"prototype; defers to HTN baseline."**

The scaffolding is real and substantially built — all the env-var knobs work, PUCT selection runs, the motif cache functions, Prometheus metrics are wired. But the full AB-MCTS tree expansion falls back to HTN in most real calls. Full implementation is v0.3 work.

**Use with this knowledge.** Good for HTN-style planning with MCTS scaffolding for future upgrades. Not yet a drop-in MCTS planner.

---

## Collective intelligence

`_vendor/common/skill_collective.py` + `skill_evolution.py` + `skill_scanner.py`. See also [Tools → Skill crystallization](TOOLS.md#skill-crystallization--agents-learn-compound-skills).

### The Flame pool

A federated skill repository. Agents crystallize skills locally from their own execution history, then publish qualifying skills to the Flame pool:

```
Agent A: executes [web_search → web_scrape → memory_store] 7 times successfully.
          → crystallization_scanner notices the pattern
          → creates skill genome "research_topic"
          → security_scan passes
          → endorsement threshold met
          → publishes to Flame pool

Agent B: scans Flame pool
          → finds "research_topic"
          → skill_endorse to accept it locally
          → uses it without having learned it from scratch
```

### Trust tiers

- `personal` — used by this agent only.
- `trusted` — shared within an identity group.
- `federated` — published to Flame.

Each tier has progressively stricter promotion criteria. Federated skills go through a security scan that inspects the tool sequence for risky combinations.

### Tools

- `collective_intelligence_status` — show local + federated skill counts.
- `collective_intelligence_sync` — pull from Flame / push local skills.
- `skill_evolve` — manually promote an observed pattern.
- `skill_scan` — inspect available skills.
- `skill_endorse` — accept a federated skill into local use.

---

## Design choices

### Why meta-call team composition over hardcoded roles?

Because real tasks don't fit a 3-role template. A simple task wants 1 agent; a complex one wants 7; a research task wants fan_out; a build task wants pipeline. Hardcoding forces overuse for small tasks and underuse for big ones. Paying one meta-call is cheap relative to the whole multi-agent run.

### Why wall-clock enforcement over iteration count?

Because token counts and tool invocations don't map reliably to cost or time. An iteration that runs `execute_code` with a 30s sandbox timeout is 30× more expensive than one that reads a file. Wall clock is the only budget that caps runaway cost.

### Why DAF instead of asyncio?

For isolation. asyncio is great for concurrent I/O but a `run_command("forkbomb")` in one coroutine takes down the whole daemon. DAF runs each agent in a subprocess — a crash is one agent's problem, not everyone's.

---

## Limits

- **DAF requires the `[server]` extra.** protobuf stubs are a heavy dep and we don't bundle them in the base install.
- **MCTS is scaffolded, not fully implemented.** Full AB-MCTS is v0.3 work.
- **No cross-daemon multi-agent.** Multi-agent runs are bounded to one daemon instance. Distributed teams need external orchestration (k8s, celery, etc.).
- **DAF task store is SQLite.** One-writer; high task-creation rates stall. Sharding is v0.4.
- **Meta-call for team composition is one extra LLM call.** For trivial tasks, this is wasted. Cache lookup for repeated task shapes is v0.3.
- **Recursion depth cap is 3.** Some legitimate deep decompositions (research → sub-research → sub-sub-research) can hit this. Raising the cap is config-available but not recommended.
- **Federated skill publishing is opt-in.** No one's sharing skills yet because we haven't shipped a public Flame endpoint. v0.4 planning.
