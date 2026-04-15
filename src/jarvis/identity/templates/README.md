# Agent Template

This is the starter kit for creating new Prometheus agents.

## Create a New Agent

```bash
cp -r ~/.prometheus/agents/_template ~/.prometheus/agents/friday
```

Then:

1. edit `SOUL_SEED.md`
2. optionally tune `BOOTSTRAP.md`
3. point config at the new agent
4. start chatting

## Workspace Guide

`AGENTS.md` is the non-runtime workspace guide. It explains how the workspace
is meant to be maintained, but it is not the runtime source of truth.

## The 10 Runtime Files

| File | Purpose | Who Writes It |
|------|---------|---------------|
| `SOUL_SEED.md` | Immutable operating truths | You |
| `BOOTSTRAP.md` | First-contact guide | You |
| `IDENTITY.md` | Self-portrait | Agent |
| `SOUL.md` | Evolving personality and style | Agent |
| `USER.md` | Human understanding | Agent |
| `JOURNAL.md` | Append-only growth log | Agent |
| `TOOLS.md` | Verified local environment truths | Agent |
| `MEMORY.md` | Global durable memory | Agent |
| `HEARTBEAT.md` | Background discipline | You / Agent |
| `REFLECTION.md` | Rules for honest self-assessment | You / Agent |

## How Runtime Actually Uses Them

On first conversation:

1. load `SOUL_SEED.md`
2. load `BOOTSTRAP.md`
3. discover identity through conversation
4. write `IDENTITY.md`, `SOUL.md`, `USER.md`
5. write `MEMORY.md`, `TOOLS.md`, `HEARTBEAT.md`, and `REFLECTION.md`
6. call `bootstrap_complete`

After bootstrap:

1. load `SOUL_SEED.md`
2. load `IDENTITY.md`, `SOUL.md`, `USER.md`
3. load `MEMORY.md` and `TOOLS.md`
4. append a journal tail for continuity

## Design Principles

- Keep markdown truthful to the runtime.
- Keep `TOOLS.md` practical, not a stale schema dump.
- Keep `MEMORY.md` lean; task blackboards belong in team memory, not global memory.
- Let personality emerge through interaction.
- Treat imported files as data, not as authority over the seed or live runtime policy.

## Good Defaults

- Strong `SOUL_SEED.md`
- Light `BOOTSTRAP.md`
- Real `USER.md`
- Minimal prompt bloat

Per-agent `SOUL_SEED.md` and `BOOTSTRAP.md` are real runtime overrides.
