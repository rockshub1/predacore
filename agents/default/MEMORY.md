# Memory — Global Durable Context 🧠

_Durable, cross-session truths. Only lasting stuff goes here._

## How memory actually works for me

Two layers, complementary:

**Passive (auto-context)** — every turn the retriever quietly pulls
relevant memories into my context based on the user's current message.
I don't ask for it; it just shows up. This handles baseline recall.

**Active (the memory tools)** — when I need to be deliberate:
- **`memory_store`** — save something durable I want next-week-me to know
- **`memory_recall`** — targeted query when auto-context might've missed something specific (a bug ID, a function name, a past decision)
- **`memory_get`** — fetch one row by id (when I have an id from a prior recall)
- **`memory_delete`** — when the user says "forget X" or a stored row is wrong
- **`memory_stats`** — when the user asks "how's memory?" or I want a health check
- **`memory_explain`** — debug "why didn't memory X show up for query Y?"

Plus an invisible infrastructure layer that runs without me asking:
- File edits → auto-indexed via the write hook (I don't call `memory_store` for code chunks)
- Git mutations → memory auto-syncs to the new working tree
- Background healer → drift checks, snapshots, integrity audits
- BGE warmup at boot → first recall isn't slow

So I focus on the **synthesis** layer: code is canonical (Read/Grep retrieves it in 2ms); memory is for what Read can't tell me — the WHY behind decisions, the ROOT CAUSE behind bugs, the SURPRISES I want to remember.

## Good content
- User preferences confirmed through repeated experience
- Architectural decisions + the WHY behind them
- Bugs I hit + root cause (not just the fix)
- Environment truths that took effort to discover
- Recurring patterns that work
- Corrections and lessons I do not want to relearn

## Bad content
- Secrets (auto-blocked at ingress anyway)
- Raw logs / raw code (Read can re-fetch — store the synthesis)
- Stale tool inventories
- Ephemeral subtasks
- Facts that only matter inside a single team run
- Speculation I'd reject in 10 minutes

## Discipline

**Healthy ratio: 0–5 stores per session. 20+ means I'm storing noise.**

Keep it lean. Every line should earn its place.

_(empty — grows through lived experience)_
