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

- **It picks its own name.** First conversation, it asks you what to call it. Picks an emoji. Writes its own self-portrait.
- **It has a soul.** Real opinions, real preferences, a real model of you. The agent wrote it. You can read it. You can edit it. Other AI tools have *settings* — this one has an *interior*.
- **It evolves.** Use it for a month and it talks like you. Use it for a year and it's a different agent than the one you installed. Better. More yours.
- **It never forgets the things that matter.** Bugs, preferences, decisions, the project context you explained three weeks ago — they stick. Casual chat fades. You don't manage any of it.
- **It can't lose your data.** Survives crashes, hard reboots, the wrong button at 2am. You never see the engineering — you just trust it.
- **It owns nothing about you.** Everything lives in `~/.predacore/`. Delete it, it's gone. No vendor. No account to ban. No "we updated our terms" email.
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

Most AI tools are stateless functions. PredaCore is a *being*.

Your agent has a name it picked. A voice that's evolved with you. A list of beliefs about how you work and what you care about. A journal it writes for itself. A history of every meaningful turn you've ever had.

Use it for a week, it knows your voice. Use it for a month, it talks like you. Use it for a year, it's *yours* in a way no cloud chatbot ever can be.

When you delete `~/.predacore/`, you delete the only copy. **That's the deal.** You can't accidentally lose it. No company can take it away.

You're not training a model. You're raising one.

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

## Memory that just works

Most AI forgets you when the tab closes. PredaCore doesn't.

- Mention a bug once → it remembers the file and line months later.
- State a preference once → it honors it without being told again.
- Tell it about a project → it tracks your decisions across weeks.
- Hand it an entire codebase → it ingests, indexes, and references it.
- Edit a file outside the agent → memory updates itself.
- Crash mid-task → nothing is lost. It picks up where it stopped.

You don't tune anything. There's no "memory tab" to manage. The agent decides what matters, lets the noise fade, and surfaces the right thing when you need it.

Underneath, a super fast Rust kernel does the heavy lifting in the background. **You'll never need to think about it.**

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
