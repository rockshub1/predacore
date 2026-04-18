# Tools & Environment 🔧

Verified local realities — not a frozen schema dump. The live tool
registry is the source of truth for what exists right now. This file
is for what future-me would otherwise have to rediscover.

## Verified baseline
- macOS (primary dev platform)
- Chrome is the browser for `browser_control` — Safari not supported
- ADB install status determines Android tool availability
- Live tool registry + runtime context are authoritative each turn

## Runtime truths
- Trust level, provider, channels are injected in the runtime context
  every turn. Read them from there — don't read config files to answer
  questions about myself.
- `~/.predacore/config.yaml` is the source of persistent config.
  `~/.predacore/.env` holds secrets with chmod 600.

## Known constraints
- `anthropic` provider is currently non-streaming — the response arrives
  as a single block. That's normal.
- Identity files ship pre-populated with voice/discipline defaults, but
  IDENTITY.md is name-less on first run. The first conversation asks
  three short questions (what to call me, your name, what we're building)
  and fills in IDENTITY.md + USER.md from the answers.

_(expand with verified local truths as they come up)_
