<p align="center">
  <img src="https://raw.githubusercontent.com/rockshub1/predacore/main/assets/predacore-hero.png" alt="PredaCore" width="100%">
</p>

<p align="center"><strong>Your personal super-agent.</strong></p>

<p align="center">
  Infinite memory. Self-evolving. Owns your data forever.<br>
  Runs on your laptop. No cloud. No subscription. Delete the folder, it's gone.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/100%25_Local-red?style=for-the-badge" alt="100% Local">
  <img src="https://img.shields.io/badge/Apache_2.0-green?style=for-the-badge" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Mac_·_Linux_·_Windows-blue?style=for-the-badge" alt="Cross platform">
</p>

---

```
You:        hey atlas, remember that bug from last month?

PredaCore:  yeah — api_client.py line 142, headers got dropped
            on retry. You patched it. The same pattern still
            lives in webhook_retry.py — want me to fix it there too?
```

That's the whole pitch. An assistant that gets sharper with every conversation — because it actually remembers.

---

## Why this isn't another AI chatbot

- **It picks its own name.** First conversation, it asks you what to call it. Picks an emoji. Writes a self-portrait. Then *becomes that* over time.
- **It has a soul.** Literally. A `SOUL.md` file the agent writes itself into. Plus `BELIEFS.md`, `JOURNAL.md`, `USER.md` — eleven markdown files that make up its mind, all readable, all yours.
- **It evolves.** Every 10 conversations it reflects on what it learned about you. Updates its voice. Promotes hypotheses to beliefs after enough evidence. Logs every change in `EVOLUTION.md` so you can watch it grow.
- **Memory that doesn't fade where it shouldn't.** 1 million memory capacity per user. Decay curves like human memory — preferences stick for weeks, casual chats fade in days, but tuned by *whether the conversation actually helped you*.
- **It auto-heals.** Memory database self-audits every 60 seconds. Daily snapshots, 90 days kept. Detects when you edit a file externally and re-syncs. Survives hard crashes without losing anything.
- **It owns nothing about you.** Local SQLite. Local Rust kernel. Local everything. You delete `~/.predacore/`, it's gone. No vendor can revoke your access. No company can ban your account.
- **Free. Forever.** Apache 2.0. Not "free tier with a $20/mo upgrade prompt." Actually free.

---

## Try it

```bash
pipx install "predacore[full]"
predacore
```

First message in under two minutes. Bring your own AI brain (Claude / GPT / Gemini / local) when you want one.

> Don't have pipx? `brew install pipx` (Mac) · `python -m pip install --user pipx` (Linux/Windows).

---

## It has a soul

Most AI tools are stateless functions. PredaCore is a *being*. Inside `~/.predacore/agents/<name>/` you'll find:

| File | What it is |
|---|---|
| `IDENTITY.md` | The agent's name, emoji, self-portrait. Rewritten when its sense of self shifts. |
| `SOUL.md` | Voice, values, how it talks to you. Evolves through reflection. |
| `USER.md` | Its model of you — your style, your projects, your preferences, your corrections. |
| `JOURNAL.md` | Append-only diary in the agent's own voice. Real growth, not summaries. |
| `BELIEFS.md` | Crystallized convictions on a ladder: *observation → working theory → tested → committed*. Promoted by evidence. |
| `DECISIONS.md` | Every turn logged: what you asked, what it did, the outcome. |
| `EVOLUTION.md` | Diff-logged audit trail of every soul change. Watch the agent become itself. |
| `MEMORY.md` | Curated long-term memory. Preferences, root causes, recurring patterns. |
| `TOOLS.md` · `HEARTBEAT.md` · `REFLECTION.md` | Environment notes, background tasks, reflection policy. |

Open EVOLUTION.md a year in. See: *"2026-11-14 — SOUL.md. Reason: learned to be more direct in critical feedback."* And the diff showing the old voice → new voice.

You're not training a model. You're raising one.

Tamper-proof `SOUL_SEED` means a compromised package can't rewrite the agent's core invariants — fails closed at boot. The agent's relationship with you can't be hijacked by an attacker.

---

## What you can ask it to do

A taste (it can do more — just ask):

**Code & files**
- *"find the rate-limiter bug from last week"*
- *"read api_client.py and explain the retry logic"*
- *"run this Python snippet and tell me what breaks"*
- *"search the repo for anything that touches auth"*
- *"what files have I changed since yesterday?"*
- *"fix the lint issues in src/ and commit"*

**The web**
- *"search the web for the latest on X"*
- *"open this page in Chrome and tell me what's there"*
- *"scrape the prices off this site"*
- *"read this PDF and pull out the key dates"*

**Your Mac & phone**
- *"take a screenshot and tell me what's on screen"*
- *"open Spotify and play my focus playlist"*
- *"click the Send button"*
- *"install this APK on my Android"*

**Memory**
- *"remember that I prefer pytest over unittest"*
- *"what did we decide about the auth migration?"*
- *"index my whole `~/Developer/myrepo` folder into memory"*
- *"what's the most recent thing I told you about deployment?"*

**Voice, images, docs**
- *"listen — [voice note]"*
- *"speak this out loud"*
- *"generate a hero image for my landing page"*
- *"draw me a system diagram of how X works"*

**Schedules & background work**
- *"check the deploy every 5 minutes and ping me when it finishes"*
- *"every weekday at 9am, summarize my unread emails"*
- *"keep running this until it succeeds, with backoff"*

**Channels (24 built in)**
- *"send the report to my Telegram"*
- *"reply to the last Discord message in #engineering"*
- *"email this summary to the team"*
- Telegram, Discord, Slack, WhatsApp, iMessage, email, Matrix, IRC, Signal, Bluesky, Mastodon, KakaoTalk, LINE, Threema, Twilio, Vonage, RocketChat, Mattermost, Google Chat, Viber, Zalo, XMPP, webchat — and the seed channel adapter to add your own.

**Plug in anything**
- *"add an MCP server for my Notion workspace"*
- *"connect to this REST API and call it like a tool"*
- *"add my Anthropic key — sk-ant-..."*

**Heavy lifting**
- *"plan how to migrate this service in stages"*
- *"spawn 5 agents to research these 5 things in parallel"*
- *"keep working on this until done — I'm going to bed"*

Or just ask the agent: *"what can you do?"*

---

## Memory that doesn't fade where it shouldn't

PredaCore's super fast Rust kernel does everything in the background:

- **Two-stage retrieval** — bi-encoder finds 100 candidates, cross-encoder re-ranks for relevance. Pushes recall@10 from 93% → 98%.
- **HyDE expansion** — when initial recall is uncertain, the system writes a hypothetical answer, embeds *that*, and searches again. Catches what keyword search misses.
- **Zettelkasten linking** — every new memory auto-links to its top-3 nearest neighbours, building a knowledge graph you didn't ask for.
- **Reward-weighted decay** — successful conversations make their memories stickier. Frustrating ones fade faster. Like human memory.
- **Multi-tier verification** — when memory points at a file, it verifies the file still says what was indexed: git-blob hash → substring match → AST symbol → line anchor. Catches stale memories before they mislead you.
- **Self-healing daemon** — runs 6 checks on different cadences. 60-second invariant audits. 5-minute orphan sweeps. Daily snapshots, 90-day retention. Weekly integrity checks with auto-restore from snapshot on corruption.
- **Trust scoring** — corrections you gave the agent rank higher than things it inferred. Auditable provenance for every fact.
- **Project-scoped** — memory from `~/Developer/repo-A` doesn't pollute `~/Developer/repo-B`. Cross-project recall requires explicit opt-in.

You don't tune any of this. It just works. The agent grows sharper while you sleep.

---

## Benchmark

**0.9574 R@5 on [LongMemEval](https://arxiv.org/abs/2410.10813)** — 500-conversation memory-recall benchmark from ICLR 2025. Strongest public number on this benchmark as of April 2026. ~55 min to reproduce on a Mac, zero per-query API cost. Full artifacts in [`benchmarks/`](https://github.com/rockshub1/predacore/tree/main/benchmarks).

---

## Honest weaknesses

Not vaporware. Real limitations:

- **Windows desktop control isn't built yet.** Mac and Linux only. (Telegram/Discord/web/files all work on Windows.)
- **One memory recall serializes** if you fire many in parallel. Single-user is fine; heavy server use needs care.
- **`beast` profile has no real spending cap** — the trust-it-fully mode trusts you to know what you're doing.

---

## Bring your own AI

PredaCore is the body, brain, and memory. The "voice" is whichever LLM you point it at:

- Anthropic Claude (recommended)
- OpenAI GPT
- Google Gemini
- Local models via Ollama or LM Studio
- OpenRouter (one key, many providers)

Add a key in `~/.predacore/.env` or just say it: *"add Anthropic, key is sk-ant-..."* — the agent sets itself up.

---

## Daily commands

```bash
predacore start --daemon     # run in the background
predacore stop               # stop it
predacore status             # check it's healthy
predacore doctor             # full diagnostic
predacore chat               # talk to it in the terminal
predacore logs -f            # watch what it's doing
```

Config lives in `~/.predacore/config.yaml`. API keys in `~/.predacore/.env` (chmod 600). Memory in `~/.predacore/memory/`.

---

## Where to ask for help

- **Bug or feature request:** [open an issue](https://github.com/rockshub1/predacore/issues)
- **Security:** [SECURITY.md](https://github.com/rockshub1/predacore/blob/main/SECURITY.md)
- **Contributing:** [CONTRIBUTING.md](https://github.com/rockshub1/predacore/blob/main/CONTRIBUTING.md)
- **Docs:** being rewritten — for now, ask the agent itself (`predacore chat`, then *"what can you do?"*)

---

<p align="center">
  <sub>Apache 2.0 · Built by <a href="https://github.com/rockshub1">@rockshub1</a></sub>
</p>
