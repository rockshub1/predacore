<p align="center">
  <img src="https://raw.githubusercontent.com/rockshub1/predacore/main/assets/predacore-hero.png" alt="PredaCore" width="100%">
</p>

<h1 align="center">PredaCore</h1>

<p align="center">
  <strong>Your personal super-agent.</strong>
</p>

<p align="center">
  Infinite memory. Infinite context for your work. Self-evolving.
  <br>
  Runs on your laptop. Uses your AI models. Owns nothing about you.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Local--First-red?style=for-the-badge" alt="Local First">
  <img src="https://img.shields.io/badge/Rust_Memory_Kernel-purple?style=for-the-badge" alt="Rust Memory Kernel">
  <img src="https://img.shields.io/badge/24%2F7_Daemon-blue?style=for-the-badge" alt="24/7 Daemon">
  <img src="https://img.shields.io/badge/Apache_2.0-green?style=for-the-badge" alt="Apache 2.0">
</p>

---

## Most AI forgets you.

PredaCore remembers.

It remembers your projects, bugs, preferences, files, decisions, workflows, and the way you like things done.

Then it acts.

It can read your code, browse the web, control apps, run commands, send messages, schedule work, use channels, spawn agents, and keep learning from every conversation.

```text
You:        remember that auth bug from last month?

PredaCore:  yes — api_client.py dropped headers on retry.
            You fixed it there.

            The same pattern still exists in webhook_retry.py.
            Want me to patch it too?
```

That is the pitch.

An AI that does not reset every time you close the tab.

---

## What PredaCore really is

PredaCore is a local agent runtime for your work.

It gives any AI model:

- long-term memory
- a body that can use tools
- armor that protects files, keys, and approvals
- a 24/7 daemon that keeps work alive
- channels to reach you where you already are
- skills that evolve from repeated workflows
- agents that can split up bigger jobs
- a fast Rust memory kernel under the hood

The model is the brain.

PredaCore is the body, armor, memory, tools, daemon, and nervous system.

Your AI stops being a floating chat box.

It becomes something that can remember, act, recover, and grow.

---

## Install

```bash
pipx install "predacore[full]"
predacore setup
predacore chat
```

No `pipx`?

```bash
brew install pipx
# or
python -m pip install --user pipx
```

Install options:

```bash
pipx install predacore            # base
pipx install "predacore[full]"    # desktop, Android, voice, local extras
pipx install "predacore[server]"  # API server, monitoring, agent fabric
```

---

## What you can ask it

```text
find the bug from last week
read this repo and explain the auth flow
remember that I prefer pytest
index ~/Developer/myapp into memory
search the web and summarize the latest docs
open Chrome and check this dashboard
take a screenshot and tell me what is broken
run this script and fix the error
send this summary to Telegram
reply to the last Discord message
check this deploy every 5 minutes
spawn agents to research these ideas
connect this MCP server
add this REST API as a tool
```

Or just ask:

```text
what can you do?
```

---

## Feature map

```text
Memory · Code · Files · Web · Chrome · Desktop · Android · Voice
Images · Diagrams · Channels · Schedules · MCP · APIs · Skills
Multi-agent work · Safety · Local daemon · Rust memory kernel
```

---

## Superpowers

### Infinite memory

PredaCore keeps memory across sessions.

Not just chat history.

It remembers:

- what you built
- what broke
- what you fixed
- what you decided
- what you prefer
- what files matter
- what should be ignored

Underneath, a fast Rust memory kernel does the heavy lifting in the background.

You do not manage a memory tab.

You just use it.

---

### Infinite context for your work

Normal AI has a context window.

PredaCore builds context over time.

It can remember old conversations, search your files, index your codebase, track decisions, and pull the right thing back when you need it.

So instead of pasting the same explanation again and again, you can say:

```text
continue from where we left off
```

And it actually knows what that means.

---

### Self-healing memory

Memory breaks in normal agents.

Files move. Branches change. Code gets renamed. Notes go stale. Laptops crash.

PredaCore keeps checking its own memory.

It can detect stale memories, repair broken links, protect against accidental deletion, keep snapshots, and recover when something goes wrong.

So when it remembers a bug from three weeks ago, it is not just repeating old chat.

It is trying to keep that memory attached to the real project.

---

### Code memory you can trust

PredaCore does not only remember that something happened.

It tries to remember where it happened.

When a file changes, PredaCore can re-check whether an old memory still points to the right code.

So a memory like this stays useful:

```text
the retry bug was in api_client.py
```

Even after the project moves forward.

---

### Self-evolving

PredaCore changes as you use it.

It can learn your style, remember your workflows, notice repeated patterns, and turn them into reusable skills.

Use it for a week and it knows your preferences.

Use it for a month and it understands your projects.

Use it for a year and it becomes a different agent from the one you installed.

More useful.

More yours.

---

### Skills that evolve

PredaCore can notice repeated workflows.

If you keep asking it to release a package, debug a deploy, summarize a repo, or prepare a report, it can turn that pattern into a reusable skill.

You do not just teach it facts.

You teach it ways of working.

---

### Runs while you sleep

PredaCore can run as a local daemon.

Start it once:

```bash
predacore start --daemon
```

Then it can keep watching, remembering, checking, scheduling, and working while you do other things.

Ask it to:

```text
check the deploy every 5 minutes
summarize unread emails every morning
watch this page for changes
keep retrying until it succeeds
```

Stop it anytime:

```bash
predacore stop
```

Your agent does not have to live inside one terminal window.

It can stay alive with your machine.

---

## Built-in powers

### Files and code

PredaCore can read files, edit files, search repos, explain code, run scripts, fix errors, and remember project decisions.

```text
read api_client.py and explain the retry logic
search the repo for anything that touches auth
run tests and fix the failing one
what changed since yesterday?
```

---

### Web and browser

PredaCore can search the web, open pages, read PDFs, scrape data, and control Chrome.

```text
search the latest docs for this API
open this dashboard and tell me what is failing
read this PDF and pull out the key dates
```

---

### Desktop, phone, voice, and images

PredaCore can work with the machine in front of you.

It can use screenshots, desktop control, Android control, voice notes, speech, image generation, and diagrams.

```text
take a screenshot and tell me what is broken
click the Send button
listen to this voice note
speak the summary out loud
draw a system diagram
generate a hero image
```

---

### Channels

PredaCore can talk through the tools you already use.

```text
Telegram
Discord
Slack
WhatsApp
iMessage
Email
Signal
Matrix
IRC
Bluesky
Mastodon
Google Chat
Webchat
and more
```

Ask:

```text
send this report to my Telegram
reply to the last Discord message in #engineering
email this summary to the team
```

---

### MCP and APIs

PredaCore can plug into external tools.

```text
add an MCP server for my Notion workspace
connect this REST API and call it like a tool
add my Anthropic key
```

You can give it more hands whenever your workflow needs them.

---

### Multi-agent work

For big tasks, PredaCore can split the work.

```text
spawn 5 agents to research these 5 ideas
compare their answers and give me the best plan
keep working on this until done
```

One agent can research.

Another can verify.

Another can write.

Another can critique.

PredaCore can coordinate the work.

---

### Background work

PredaCore can keep tasks alive across restarts.

```text
every weekday at 9am, summarize unread emails
check the deploy every 5 minutes
keep retrying this job with backoff
watch this page and tell me when it changes
```

The chat can end.

The work can continue.

---

## Local-first

PredaCore stores its world here:

```bash
~/.predacore/
```

Important paths:

```bash
~/.predacore/config.yaml       # config
~/.predacore/.env              # API keys
~/.predacore/memory/           # memory
~/.predacore/cron_tasks.json   # scheduled work
```

Delete the folder and the local memory is gone.

No vendor owns your agent.

No account ban erases your history.

No cloud workspace controls your assistant.

Your machine.  
Your models.  
Your data.

---

## Bring your own AI

PredaCore works with the AI models you choose:

- Claude
- GPT
- Gemini
- OpenRouter
- Ollama
- LM Studio
- Kimi
- Qwen
- Perplexity
- your own endpoint

The model is the brain.

PredaCore is the body, armor, memory, tools, daemon, channels, scheduler, and agent runtime.

---

## Safety by default

PredaCore can touch real files, browsers, terminals, apps, and messages.

So it has guardrails.

It can protect approvals, block dangerous commands, avoid sensitive files, filter secrets, limit risky paths, scan identity files, and keep an audit trail.

Default mode is careful.

Power users can use `beast` mode when they know exactly what they are doing.

---

## Commands

```bash
predacore setup              # guided setup
predacore chat               # talk in terminal
predacore start --daemon     # run in background
predacore stop               # stop daemon
predacore status             # check health
predacore doctor             # full diagnostic
predacore logs -f            # follow logs
```

---

## Benchmark

**0.9574 R@5 on LongMemEval**

LongMemEval is a 500-conversation memory-recall benchmark from ICLR 2025.

Benchmark artifacts live in:

```bash
benchmarks/
```

Reported reproduction notes:

- deterministic retrieval
- no LLM randomness during recall
- roughly 55 minutes to reproduce on a Mac
- zero per-query API cost

---

## Honest limits

PredaCore is powerful, but not magic.

- Windows desktop control is not fully built yet.
- Some features need local permissions.
- Channel integrations need your own accounts or tokens.
- Heavy parallel use needs care.
- `beast` mode can spend aggressively if you let it.
- The Python repo is Apache 2.0; the Rust memory kernel is distributed separately as prebuilt wheels.

---

## Help

- Issues: https://github.com/rockshub1/predacore/issues
- Security: https://github.com/rockshub1/predacore/blob/main/SECURITY.md
- Contributing: https://github.com/rockshub1/predacore/blob/main/CONTRIBUTING.md

Docs are being rewritten.

For now:

```bash
predacore chat
```

Then ask:

```text
what can you do?
```

---

<p align="center">
  <sub>Apache 2.0 · Built by <a href="https://github.com/rockshub1">@rockshub1</a></sub>
</p>
