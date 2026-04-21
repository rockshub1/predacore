# Autonomy — Running Without a Human in the Loop

PredaCore has three existing paths for autonomous work. This doc explains
when to reach for each and how to keep them safe.

> **Mental model:** chat is the *cockpit* — someone is waiting, latency
> matters, iteration count is bounded. Autonomous work is the *factory
> floor* — no one is waiting, it can run for hours, but it has to stop
> itself.

## The four paths, ranked by weight

| Path | When it's the right call | What it costs |
|---|---|---|
| **1. Extend `max_tool_iterations`** per-call | Medium task, user willing to wait seconds-to-minutes in chat | Nothing — already in-process. Modern models loop rarely, 50-150 is usually enough. |
| **2. `multi_agent` tool (in-process)** | Team of specialists — fan-out / pipeline / consensus / supervisor — that shares one asyncio loop | Zero external deps. Best for team orchestration without process isolation. |
| **3. `multi_agent` with `use_daf=true`** | Team that needs process isolation, capability routing, task persistence across restart | Requires DAF gRPC service running (see `agents/daf/`). Heavier. |
| **4. `openclaw_delegate` tool** | Want an *entirely separate* agent — different personality, different tool set, different trust level, crash-isolated | Requires OpenClaw service running. Best when the task should outlive your daemon. |

There is no separate "background agent" tool because **paths 2-4 already
cover every autonomy case**. Single long-running agent? `multi_agent`
with `num_agents=1`. Need persistence? Add `use_daf=true`. Need another
agent to do the heavy lifting? `openclaw_delegate`.

## Why the foreground loop is bounded

`core.py` clamps iteration count to `launch.max_tool_iterations` per
turn. Defaults:

| Profile | Cap | Rationale |
|---|---|---|
| `enterprise` | **1000** | Safe-by-default but resource-maxed. meta_cognition loop detector catches runaway. |
| `beast` | **1000** | Autonomous, same cap. Identical resource limits on both profiles — modes differ on governance, not capacity. |

Hard ceiling is `_ABSOLUTE_MAX_ITERATIONS = 1000` in `core.py` regardless
of config. Override per-deployment via `MAX_TOOL_ITERATIONS=N` env var.

## Safety rails worth knowing

Existing mechanisms catch runaways before they get expensive:

- **Loop detection** — `agents/meta_cognition.py::should_ask_for_help`
  catches "same tool + same args 3× in a row" and forces the agent to
  reconsider.
- **Tool timeouts** — `PREDACORE_TOOL_TIMEOUT_SECONDS` (default 1800s
  per individual tool call).
- **Persona drift guard** — `prompts.py`'s regex-driven drift detection
  and regeneration, so a wandering agent is re-anchored on its identity
  before it derails further.
- **Trust level** — in `normal` mode every mutating tool is confirmed.
  In `yolo` nothing is confirmed (use for autonomous runs where a human
  isn't watching). In `paranoid` mode every tool is confirmed.
- **Delegation depth** — `multi_agent` and `openclaw_delegate` both
  limit recursion depth via the `_DELEGATION_DEPTH` contextvar, so an
  agent can't infinite-spawn child agents.

## Best practice for long autonomous runs

When you know a task needs hours of autonomous work:

1. **Trust level `yolo`** for the session — confirmations block forever
   if no human is present.
2. **`multi_agent` with `use_daf=true`** — gives process isolation +
   task persistence (via `agents/daf/task_store.py`).
3. **Bound the scope in the prompt** — "audit these 50 files" beats
   "audit the repo," because a bounded scope gives the model a natural
   stopping condition.
4. **Give the cron engine a shot** instead — `cron.yaml` entries run
   scheduled natural-language actions through the full gateway + LLM +
   tools loop with no human in the chat. Great for "do this every day
   at 9am."

## Use OpenClaw when

- You want the long task to **survive a PredaCore daemon restart**. DAF
  + `task_store.py` handles persistence inside PredaCore but gRPC
  service uptime is still a dependency; OpenClaw is a different process
  entirely.
- You want **different personality / tools / trust level** than your
  cockpit agent. A PredaCore agent is specialized for chatting with
  *you*; OpenClaw workers are specialized for *grinding*.
- You want **crash isolation** — if the autonomous work errors out, it
  doesn't take PredaCore's chat loop with it.

## What NOT to build

- **A separate "background agent" tool**. Already covered by the paths
  above. Adding a fifth path for the single-agent case just fragments
  the mental model.
- **Pure in-process `asyncio.Task` spawning** without process isolation
  or persistence. DAF already does this better when you need it.
- **Ad-hoc autonomy** without explicit caps. If a task needs to run
  autonomously, put bounds on it — iteration cap, wall-clock, dollar
  budget. Autonomy without budgets is how a $4,000 bill happens.

## Config knobs that matter for autonomy

```yaml
launch:
  max_tool_iterations: 50          # foreground cap; raise per profile
  max_spawn_depth: 4               # how deep multi_agent / openclaw can recurse
  max_spawn_fanout: 8              # max team size per multi_agent call
  enable_openclaw_bridge: false    # gate the openclaw_delegate tool
security:
  trust_level: normal              # yolo for hands-off autonomous runs
  docker_sandbox: false            # enable for sandboxed code execution
  task_timeout_seconds: 300        # per-task wall-clock ceiling
```

## TL;DR

- Foreground cap = 1000 on both profiles. That's more than enough for
  most work; no separate background tool needed.
- For teams or long runs: use `multi_agent` (+ `use_daf=true` for
  persistence).
- For external delegation / crash isolation: `openclaw_delegate`.
- For scheduled autonomy: `cron.yaml` entries.
- All four paths share the same safety rails: loop detection, tool
  timeouts, drift guard, trust level, delegation depth.
