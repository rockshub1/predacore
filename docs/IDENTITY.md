# Identity

> Your agent is thirteen markdown files. You can grep them. You can git them. You can delete them.

Most agent frameworks treat identity as a string constant in a config file. PredaCore treats it as a **versioned, evolving, auditable filesystem** that the agent itself participates in maintaining — within hard safety guardrails that are scanned on every load.

This is the file that explains how.

---

## The 13-file workspace

Every agent lives in `~/.predacore/agents/<name>/`:

```
atlas/
├── SOUL_SEED.md              ← Bundled · immutable · tamper-guarded
├── EVENT_HORIZON.md          ← Bundled · immutable · growth protocol
│
├── IDENTITY.md               ← Agent-writable: "I'm Atlas. sharp-but-steady."
├── SOUL.md                   ← Agent-writable: voice, taste, disagreement style
├── USER.md                   ← Agent-writable: patterns learned about you
├── MEMORY.md                 ← Agent-writable: distilled principles
├── TOOLS.md                  ← Agent-writable: verified local environment
├── HEARTBEAT.md              ← Agent-writable: background discipline policy
├── REFLECTION.md             ← Agent-writable: self-correction rules
├── JOURNAL.md                ← Agent-writable: diary of shifts
│
├── BELIEFS.md + beliefs.json ← Crystallization ladder (engine-managed)
├── DECISIONS.md              ← Per-turn reasoning trace (engine-written, 500-entry cap)
└── EVOLUTION.md              ← Auto-diff log of every identity mutation (200-entry cap)
```

**Three write tiers:**

- **Bundled & immutable** (SOUL_SEED, EVENT_HORIZON) — ship with the package. Workspace copies are blocked to prevent tampering.
- **Agent-writable** (10 files: IDENTITY, SOUL, USER, MEMORY, TOOLS, HEARTBEAT, REFLECTION, JOURNAL, BELIEFS, DECISIONS) — the agent can read and update via `identity_update` tool.
- **Engine-managed** (EVOLUTION.md, `beliefs.json`) — only the engine writes. Auto-diffs on mutation; crystallization state in JSON for machine reads.

---

## System prompt assembly — 14 steps

Every turn, the system prompt is composed in a deterministic sequence (`identity/engine.py:744-795`):

```
1.  SOUL_SEED.md           (bundled, package-only — workspace copy blocked)
2.  EVENT_HORIZON.md       (bundled, same tamper guard)
3.  IDENTITY.md
4.  SOUL.md
5.  USER.md
6.  Structured UserProfile (JSON: goals · knowledge_areas · cognitive_style · preferences)
7.  MEMORY.md
8.  TOOLS.md
9.  HEARTBEAT.md           (wrapped with "# Background Discipline" header)
10. REFLECTION.md          (wrapped with "# Self-Correction Rules" header)
11. BELIEFS.md             (rendered by BeliefStore from crystallization state)
12. JOURNAL.md tail        (last 20 entries, capped at 10 KB)
13. Optional meta prompt   (gated by PREDACORE_ENABLE_META_PROMPT or enable_self_evolution config)
14. Runtime context block  (mode · trust · date · profile · model · channels · paths)
```

Sections are separated by `---` horizontal rules so the model sees the hierarchy. The runtime context block ends with an **explicit priority rule**:

> SOUL_SEED > identity files > Runtime section — for safety.
> Runtime section > all — for factual state (date, active model, channels).

Files are mtime-cached. Hot config reload invalidates. The prompt is rebuilt every turn, not cached across turns — this is deliberate, because identity files mutate during a conversation and stale prompts cause drift.

---

## Belief crystallization

Beliefs graduate through four states with evidence-driven promotion (`identity/beliefs.py:47-49`):

```
observation ──2─▶ working_theory ──5─▶ tested ──8 + confirm─▶ committed
     │                                                              │
     └──────────── explicit falsification ◀─────────────────────────┘
                         (reason logged)
```

```python
_PROMOTE_TO_WORKING_THEORY = 2    # 2 pieces of evidence
_PROMOTE_TO_TESTED         = 5    # 5 pieces
_PROMOTE_TO_COMMITTED      = 8    # 8 pieces + explicit user confirmation
```

**Why this matters.** The agent doesn't commit to "you prefer TypeScript" because you said it once. It observes, hypothesizes, tests, and only commits when evidence compounds. On falsification, the belief is **demoted with a reason** — not silently deleted.

Beliefs are stored both as:

- `BELIEFS.md` — human-readable summary with state markers, included in the system prompt
- `beliefs.json` — machine-readable state for fast queries

---

## Auto-diff change log (EVOLUTION.md)

Every write to `SOUL`, `IDENTITY`, `USER`, or `BELIEFS` appends a **timestamped unified diff** to `EVOLUTION.md` (`identity/engine.py:55-57`).

Example entry:

```
## 2026-04-17T09:42:13Z — USER.md

**Reason:** Noticed consistent pattern across 4 conversations.

--- before
+++ after
@@ -12,3 +12,5 @@
 Prefers short, receipted answers.
 Senior engineer, deep Rust + Python.
+Dislikes marketing-language framing.
+Responds well to "honesty over polish".
```

Bounded at 200 most recent entries (`_MAX_EVOLUTION_ENTRIES = 200`, `identity/engine.py:70`). Max diff size per entry: 200 lines (`_MAX_DIFF_LINES = 200`). Pruning is automatic.

This is a **legible change log written by the agent about itself**. Open it. Read it. Understand how the agent has changed.

---

## DECISIONS.md — per-turn reasoning trace

Each turn, the agent writes a structured decision record:

```
## 2026-04-17T10:15:22Z — turn 47

**User prompt:** fix the webhook_retry.py bug
**Approach chosen:** semantic_search → read_file → edit → run tests
**Reasoning:** Pattern matches a prior fix; test-first validates regression.
**Tools used:** memory_recall, read_file, write_file, run_command
**Outcome:** success (tests pass)
```

Bounded at 500 entries (`_MAX_DECISION_ENTRIES = 500`). This is a turn-by-turn audit trail — you can `grep DECISIONS.md` to see why the agent chose what it chose on any given turn.

---

## Prompt-injection scanning on every load

`identity/engine.py:288-315`. Every workspace identity file is scanned by `auth/security.py::detect_injection` before it's included in the system prompt.

**What happens on detection:**

- `SOUL.md` / `IDENTITY.md` / `USER.md` / etc. triggers → **fall back to the bundled built-in**. The poisoned file is logged but not loaded.
- `SOUL_SEED.md` triggers → **fail loud**. Startup aborts. The safety floor is non-negotiable.

**Why this matters.** If an attacker compromises your disk or tricks the agent into writing a malicious instruction to `SOUL.md`, that payload doesn't load into the system prompt. The bundled default does. A supply-chain-tampered `SOUL_SEED.md` in a compromised install aborts the daemon — fail-closed, not fail-open.

The scanner is ~15 regex patterns with per-pattern confidence scores (`auth/security.py:43-120`) — catches role-hijack, jailbreak markers, data exfiltration, system-prompt exfiltration attempts.

---

## Atomic writes

Identity file writes use **temp-file + rename** (POSIX atomic):

```python
# pseudocode
tmp = path.with_suffix(".tmp")
tmp.write_text(new_content)
tmp.replace(path)  # atomic on POSIX
```

Prevents torn writes if the daemon crashes mid-write. Same pattern for `beliefs.json` and `EVOLUTION.md`.

---

## File cache

`identity/engine.py` keeps an in-memory file cache keyed on `(path, mtime)`. On each prompt build:

- If `mtime` unchanged → serve cached content.
- If `mtime` changed → re-read, re-scan for injection, update cache.

Hot config reload (`ConfigWatcher`) invalidates the cache. External edits (you opening `USER.md` in vim and saving) are picked up on next turn via mtime change.

---

## How a new agent gets a name

On first conversation, the agent has no `IDENTITY.md`. The first few turns establish:

1. The user asks a meaningful question.
2. The agent responds via `SOUL_SEED` + `EVENT_HORIZON` only — no name yet.
3. Either the user names the agent, or the agent proposes a name grounded in their interaction.
4. The agent writes `IDENTITY.md` with its chosen/given name.

This is by design. Names emerge from the relationship; they're not assigned by config.

---

## Multi-agent identity

Each agent has its own full workspace. `~/.predacore/agents/atlas/` · `~/.predacore/agents/research_bot/` · `~/.predacore/agents/ops/`. Zero shared state between identity workspaces — only memory can be shared, and only via explicit `global` scope (see [Memory](MEMORY.md#memory-scopes)).

Agents can delegate to each other via `multi_agent` or `openclaw_delegate` (see [Multi-agent](MULTI_AGENT.md)). The delegatee gets its own identity loaded, separate from the caller's.

---

## Limits

- **File cache is in-process.** Cross-daemon-restart the cache is cold; first turn after restart is slower.
- **JOURNAL.md tail is capped at 10 KB.** Older entries remain in the file but aren't included in the system prompt. If you want longer context, grep the file directly.
- **No semantic drift detection.** The persona-drift guard is regex-based (see [Safety](SAFETY.md#persona-drift-guard)) — catches model-identity slippage but not subtle tone drift.
- **Belief falsification requires explicit user signal.** Implicit contradictions (user says the opposite in another turn) don't auto-demote. v0.4 work.
- **DECISIONS.md can balloon.** 500 entries is usually fine; on a long-running daemon with heavy turn volume, rotation is manual (delete the file, it regenerates).
