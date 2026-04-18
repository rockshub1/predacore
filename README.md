<p align="center">
  <img src="https://raw.githubusercontent.com/rockshub1/predacore/main/docs/logo/predacore-hero.png" alt="PredaCore" width="100%">
</p>

<p align="center"><strong>The hyper-autonomous AI agent with persistent memory and 55 powerful tools.</strong></p>

<br>

<h1 align="center"><code>0.9574</code></h1>

<p align="center"><sub><strong>R @ 5 &nbsp; · &nbsp; LongMemEval</strong></sub></p>

<br>

<p align="center">
  Persistent memory. On your laptop. No cloud. No API keys. Yours forever.
</p>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Kernel-Rust_BGE-orange?style=for-the-badge" alt="Rust BGE Kernel">
  <img src="https://img.shields.io/badge/Memory-Persistent-blue?style=for-the-badge" alt="Persistent Memory">
  <img src="https://img.shields.io/badge/55%20Powerful%20Tools-success?style=for-the-badge" alt="55 Powerful Tools">
  <img src="https://img.shields.io/badge/Privacy-100%25_Local-red?style=for-the-badge" alt="100% Local">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="Apache 2.0">
</p>

---

```
You:        hey atlas, remember that rate-limiter bug from last month?

PredaCore:  yeah — api_client.py:142, headers dropped on retry. You
            patched it. Similar pattern still lives in webhook_retry.py.
            want me to fix it there too?
```

Most AI forgets you the moment the tab closes. **PredaCore doesn't.** Rust kernel, 13 markdown identity files, 55 tools, nine channels, zero vendor SDKs. Every number on this page reproducible in one command.

- **Stop re-explaining yourself.** It knows your repo, your stack, your architecture — across weeks.
- **Bugs don't bite twice.** Patterns you debugged last month get flagged when they reappear.
- **Preferences stick.** Say *"use pytest"* once — never again.
- **Work compounds.** Every session picks up where the last left off. Useful memories persist; dead weight fades automatically — preferences live for weeks, casual chats for days, all tuned by session reward.
- **You own it.** Memory lives in `~/.predacore/`. No cloud, no account, no vendor.

You can delete it anytime — `rm -rf ~/.predacore/agents/atlas/`.

---

## 🧰 55 powerful tools

Not an LLM wrapper. A digital operator wired into your machine through a hardened dispatcher — Express-style middleware, per-tool circuit breakers, adaptive P95 timeouts, LRU cache, SHA-256-hashed persistent approvals.

| | Tools |
|---|---|
| **Code & shell** | `execute_code` (13 langs · sandboxed Docker · [optional](docker/sandbox/Dockerfile)) · `python_exec` · `run_command` · `read_file` · `write_file` · `list_directory` |
| **Web** | `browser_control` (hijacks Chrome via DOM — 100× faster than screenshots) · `deep_search` · `web_search` · `web_scrape` |
| **Desktop / mobile** | `desktop_control` (PyObjC · 1–5ms per action) · `screen_vision` · `android_control` (ADB + uiautomator2) |
| **Git** | `git_semantic_search` (*"where is the auth middleware?"*) · `git_context` · `git_diff_summary` · `git_commit_suggest` · `git_find_files` |
| **Agents & planning** | `multi_agent` — fan-out · pipeline · consensus · supervisor, with optional **DAF gRPC process isolation** for true parallel agents · `strategic_plan` (HTN + MCTS, multi-objective) · `openclaw_delegate` |
| **Memory** | `memory_store` · `memory_recall` · `semantic_search` (scoped global · team · scratch) |
| **Identity** | `identity_read` · `identity_update` · `journal_append` — writes to the agent's 13-file soul |
| **MCP client** | `mcp_add` · `mcp_list` · `mcp_remove` · `mcp_restart` — mount any MCP server mid-chat |
| **REST APIs** | `api_add` · `api_call` · `api_list` · `api_remove` — bind any service in seconds |
| **Pipelines** | `tool_pipeline` (sequential · parallel · conditionals · templates) · `tool_stats` |
| **Collective intelligence** | `skill_evolve` · `skill_scan` · `skill_endorse` · `collective_intelligence_sync` · `collective_intelligence_status` · `marketplace_*` |
| **Voice / creative / cron** | `speak` · `voice_note` · `image_gen` · `pdf_reader` · `diagram` (Mermaid) · `cron_task` |
| **Infrastructure** | `secret_set` · `secret_list` · `channel_configure` · `channel_install` |

---

## How the engine purrs

**Rust compute kernel.** Candle BGE + BM25 + trigram fuzzy + entity extraction. SIMD cosine. Deterministic retrieval — no LLM sampling, no RNG. That's why benchmarks reproduce bit-identical.

**Thirteen files. One soul.** Your agent's identity lives in `~/.predacore/agents/<name>/` as plain markdown. `cat` them. `git log` them. `rm` them. Beliefs graduate *observation → working_theory → tested → committed*. Every mutation auto-diffs to `EVOLUTION.md`. Tampered `SOUL_SEED` aborts startup. **Fail closed.**

**Safety as a primitive.** Prompt-injection scan on every identity load. SSRF guard on web tools. Secret-shape allowlist — even `yolo` can't write arbitrary env vars. Persona-drift regex ladder auto-regenerates drifted turns. Memory scopes (`global · agent · team · scratch`) prevent cross-contamination.

**Per-session lane queue.** Same session = serial FIFO. Different sessions = concurrent. Meta-cognition catches loops, oscillation, thrashing — with a diversity exception so real exploration isn't punished.

**DAF — true parallel agents.** When in-process asyncio isn't enough, the Dynamic Agent Fabric (`[server]` extra) gives you gRPC multi-process isolation. Agents run in their own processes, crash-isolated, with self-optimization: >20% error rate → respawn · queue depth >10 → scale out · idle >300s → terminate · P95 latency >3× baseline → marked degraded. Wall-clock budgets clamped 10s..6h, hard-killed via `asyncio.wait_for`. Teams get private 72h-TTL scratchpads so findings don't leak to caller memory.

---

## Quickstart

```bash
pip install predacore[full]
predacore
```

One command. First message in under two minutes. Rust ships as pre-built wheels — no toolchain required.

**Zero-config. The agent configures itself mid-chat:**

```
You: add Anthropic — key sk-ant-api03-XXXXXXXXXX
You: enable telegram with token 123:abc
You: install the GitHub MCP server
```

Routed through `secret_set`, `channel_configure`, `mcp_add`. Writes land in `~/.predacore/.env` (chmod 600).

| Install | Adds | Δ Wheel |
|---|---|---|
| `predacore` | Engine · CLI · webchat · 8 channels · Playwright · PDF · voice · sandbox · Rust kernel | ~350 MB |
| `predacore[full]` | + spaCy · desktop automation · Android ADB | +200 MB |
| `predacore[server]` | + FastAPI · Redis · Prometheus · DAF gRPC | +150 MB |

---

## Benchmarks

**0.9574 R@5** on [LongMemEval](https://arxiv.org/abs/2410.10813) — the long-term-memory benchmark from ICLR 2025. 500 conversational histories · ~57M tokens · 470 scored.

| Category | n | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| knowledge-update | 72 | **0.9861** | 0.9861 | 1.000 |
| multi-session | 121 | **0.9835** | 0.9917 | 1.000 |
| single-session-assistant | 56 | 0.9643 | 0.9821 | 0.9821 |
| single-session-user | 64 | 0.9531 | 0.9844 | 1.000 |
| temporal-reasoning | 127 | 0.9370 | 0.9606 | 0.9843 |
| single-session-preference | 30 | 0.8667 | 0.9333 | 1.000 |

Four of six categories clear **0.95**. Bit-identical reproduction:

```bash
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python -m predacore.evals.longmemeval --dataset longmemeval_s_cleaned.json --json-out my_run.json
```

~55 min on Apple Silicon. Zero per-query API cost. Full artifacts in [`benchmarks/`](https://github.com/rockshub1/predacore/tree/main/benchmarks).

---

## Trust profiles

| Profile | Behavior |
|---|---|
| `paranoid` | Confirms every tool · ethical keyword guard active |
| `normal` *(default)* | Auto-approves 12 read-only · confirms 16 destructive |
| `yolo` | Full autonomy · arg-regex still blocks `rm -rf`, `sudo`, `dd if=` |

---

## Honest weaknesses (no vaporware)

- **Windows desktop operator unimplemented.** Networked surfaces work everywhere; `desktop_control` / `screen_vision` are macOS + Linux only. *Coming soon.*
- **`single-session-preference` R@5 = 0.867.** Retrieval's weak spot — cross-encoder re-ranker is the planned fix.
- **GIL not released in Rust kernel.** Concurrent `embed()` calls serialize. rayon helps within a call.
- **`yolo` has no real cost cap.** Arg-regex catches `rm -rf`, not an obfuscated `curl | sh`.
- **`_vendor` ships in wheels.** Five subpackages bloat the install.

---

## Links

**Deep dives:** [Memory](https://github.com/rockshub1/predacore/blob/main/docs/MEMORY.md) · [Identity](https://github.com/rockshub1/predacore/blob/main/docs/IDENTITY.md) · [Tools](https://github.com/rockshub1/predacore/blob/main/docs/TOOLS.md) · [Multi-agent](https://github.com/rockshub1/predacore/blob/main/docs/MULTI_AGENT.md) · [Safety](https://github.com/rockshub1/predacore/blob/main/docs/SAFETY.md) · [Autonomy](https://github.com/rockshub1/predacore/blob/main/docs/AUTONOMY.md) · [MCP](https://github.com/rockshub1/predacore/blob/main/docs/MCP.md) · [Channels](https://github.com/rockshub1/predacore/blob/main/docs/CHANNEL_ADAPTER.md) · [Launch profiles](https://github.com/rockshub1/predacore/blob/main/docs/launch_profiles.md)

**Issues:** [github.com/rockshub1/predacore/issues](https://github.com/rockshub1/predacore/issues) · **Security:** [SECURITY.md](https://github.com/rockshub1/predacore/blob/main/SECURITY.md) · **Contributing:** [CONTRIBUTING.md](https://github.com/rockshub1/predacore/blob/main/CONTRIBUTING.md)

---

<p align="center">
  <sub>Apache 2.0 · Every claim reproducible from the repo.</sub>
</p>
