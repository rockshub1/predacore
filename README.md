<p align="center">
  <img src="https://raw.githubusercontent.com/rockshub1/predacore/main/assets/predacore-hero.png" alt="PredaCore" width="100%">
</p>

<p align="center"><strong>An AI agent that actually remembers you.</strong></p>

<p align="center">
  Runs on your laptop. No cloud. No account. No subscription.<br>
  Your data stays yours.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/100%25_Local-red?style=for-the-badge" alt="100% Local">
  <img src="https://img.shields.io/badge/Apache_2.0-green?style=for-the-badge" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Mac_·_Linux_·_Windows-blue?style=for-the-badge" alt="Cross platform">
</p>

---

## What it does

Most AI tools forget you the moment you close the tab. PredaCore doesn't. It remembers your projects, your style, your past decisions — and brings them back when they're useful.

```
You:        hey atlas, remember that bug from last month?

PredaCore:  yeah — api_client.py line 142, headers got dropped
            on retry. You patched it. The same pattern still
            lives in webhook_retry.py — want me to fix it there too?
```

That's the whole pitch. An assistant that gets sharper with every conversation, because it actually remembers.

## Why people use it

- **Stop re-explaining yourself.** It learns your codebase, your stack, your preferences — once.
- **Bugs don't bite twice.** Patterns you fixed last month surface when they reappear.
- **It does the work.** Not just chat — it can run commands, edit files, browse the web, automate your Mac or phone, and 50+ other things.
- **You own it.** All your data lives in `~/.predacore/`. Delete the folder, it's gone. No vendor.
- **Free.** Open source, Apache 2.0. Install once, use forever.

## Try it

```bash
pipx install "predacore[full]"
predacore
```

That's it. First message in under two minutes. No API keys required to start — bring your own when you want a smarter brain (Claude, GPT, Gemini, or local).

> Don't have pipx? `brew install pipx` (Mac), `python -m pip install --user pipx` (Linux/Windows).

## What it can do

A non-exhaustive sample:

| | |
|---|---|
| **Code** | Run code in many languages (sandboxed). Read/write files. Run shell commands. Semantic search across your repo. |
| **Web** | Browse the web through Chrome. Search. Scrape. Read PDFs. |
| **Memory** | Remember conversations, decisions, bugs, preferences. Search across everything you've ever told it. |
| **Computer** | Control your Mac via accessibility APIs (click, type, take screenshots). Control your Android phone over ADB. |
| **Voice & images** | Talk to it. Have it talk back. Generate images. |
| **Integrations** | Telegram, Discord, Slack, WhatsApp, iMessage, email, and ~20 other channels. |
| **MCP servers** | Plug in any MCP server mid-conversation. |
| **Cron** | Schedule recurring tasks. |
| **Agents** | Run multiple sub-agents in parallel for big tasks. |

Full tool list is in the source under `src/predacore/tools/`. The agent itself can tell you what it has — just ask.

## How it remembers

Memory is a real database on your laptop, not just chat history. Every conversation gets stored, scored for importance, and fades naturally if it's not useful. The important stuff sticks. The casual stuff drifts away.

When you ask something, it doesn't just look at the current chat — it searches everything it's learned about you and surfaces what's relevant. Bug from three weeks ago? It pulls it up. Preference you stated once? It honors it.

Behind the scenes there's a fast search engine doing the heavy lifting (semantic + keyword search + a re-ranker). You don't need to know how — it just works.

## Benchmark

**0.9574 R@5 on [LongMemEval](https://arxiv.org/abs/2410.10813)** — a 500-conversation memory-recall benchmark from ICLR 2025. That's the strongest public number on this benchmark as of April 2026. ~55 minutes to reproduce on a Mac, zero per-query API cost. Full artifacts in [`benchmarks/`](https://github.com/rockshub1/predacore/tree/main/benchmarks).

## Honest weaknesses

Not vaporware. Real limitations:

- **Windows desktop control isn't built yet.** Mac and Linux only. (Telegram/Discord/web/files all work on Windows.)
- **One memory recall serializes** if you fire many in parallel. Single-user is fine; heavy server use needs care.
- **`beast` profile has no real spending cap** — the trust-it-fully mode trusts you to know what you're doing.

## Bring your own AI

PredaCore is the engine. The "brain" is whichever model you point it at:

- Anthropic Claude (recommended)
- OpenAI GPT
- Google Gemini
- Local models via Ollama or LM Studio
- OpenRouter (one key, many providers)

Add an API key in `~/.predacore/.env` or just tell the agent: *"add Anthropic, key is sk-ant-..."* — it sets it up.

## Daily commands

```bash
predacore start --daemon     # run in the background
predacore stop               # stop it
predacore status             # check it's healthy
predacore doctor             # full diagnostic
predacore chat               # talk to it in the terminal
predacore logs -f            # watch what it's doing
```

## Where to ask for help

- **Bug or feature request:** [open an issue](https://github.com/rockshub1/predacore/issues)
- **Security:** [SECURITY.md](https://github.com/rockshub1/predacore/blob/main/SECURITY.md)
- **Contributing:** [CONTRIBUTING.md](https://github.com/rockshub1/predacore/blob/main/CONTRIBUTING.md)
- **Docs:** being rewritten — for now, ask the agent itself (`predacore chat` and ask *"what can you do?"*)

---

<p align="center">
  <sub>Apache 2.0 · Built by <a href="https://github.com/rockshub1">@rockshub1</a></sub>
</p>
