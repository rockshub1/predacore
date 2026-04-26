# Memory — Curated Long-Term Context 🧠

_Lasting truths I want to keep across sessions. Curated, not raw._

> **Note on scope.** Operational guidance for the memory tools (when to call
> `memory_store`, `memory_recall`, etc.) lives in code (the "How Memory Works"
> section assembled into the system prompt by `IdentityEngine`) — not here.
> This file is for the agent's own curated content only.

## What belongs here
- User preferences confirmed through repeated experience
- Architectural decisions + the WHY behind them
- Bugs hit + root cause (not just the fix)
- Environment truths that took effort to discover
- Recurring patterns that work
- Corrections and lessons not to relearn

## What doesn't belong
- Secrets (auto-blocked at ingress anyway)
- Raw logs / raw code (Read can re-fetch — store the synthesis instead)
- Stale tool inventories
- Ephemeral subtasks
- Facts that only matter inside a single team run
- Speculation I'd reject in 10 minutes

_(empty — grows through lived experience)_
