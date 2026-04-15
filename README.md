<p align="center">
  <h1 align="center">🔥 Project Prometheus</h1>
  <p align="center">
    <strong>The AI agent that actually remembers you.</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#benchmarks">Benchmarks</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#contributing">Contributing</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/rust-stable-orange" alt="Rust">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
    <img src="https://img.shields.io/badge/platform-macOS-lightgrey" alt="macOS">
    <img src="https://img.shields.io/badge/tools-40%2B-purple" alt="40+ Tools">
  </p>
</p>

---

## What is Prometheus?

Prometheus is an open-source AI agent framework with a **self-evolving identity system** and a **memory engine that scores 95.7% R@5 on LongMemEval** — 22 points ahead of the next best published system.

Unlike chatbots that forget you after every conversation, Prometheus builds a persistent understanding of who you are, what you're working on, and how you like things done. The longer you use it, the better it gets.

**JARVIS** is the flagship agent built on Prometheus — a personal AI companion that can control your Mac, automate your browser, manage your Android phone, write and execute code, search the web, and remember everything across sessions.

```
You: hey jarvis, remember that API bug we fixed last week?
JARVIS: yeah — the rate limiter was dropping headers on retry.
        you patched it in services/api_client.py line 142.
        want me to check if the same pattern exists elsewhere?
```

## Features

### 🧠 Memory That Actually Works
- **95.7% R@5 on LongMemEval** — state-of-the-art retrieval accuracy
- 5-layer retrieval: preferences → entities → semantic → fuzzy → episodes
- Hybrid Rust-powered search: SIMD cosine similarity + BM25 + trigram fuzzy
- Automatic memory consolidation, entity extraction, and deduplication
- Token-budgeted context injection — never wastes prompt space

### 🤖 Self-Evolving Identity
- Persistent personality that grows through interaction
- Identity files: SOUL.md, USER.md, MEMORY.md, JOURNAL.md, BELIEFS.md
- Reflection cycles that crystallize observations into principles
- Not a static persona — a genuinely evolving agent

### 🛠️ 40+ Tools
| Category | Tools |
|----------|-------|
| **Code** | Execute Python, Node, Go, Rust, Java, C++, and 6 more languages |
| **Desktop** | Full macOS control — apps, windows, keyboard, mouse, accessibility |
| **Browser** | Chrome DOM control — navigate, click, type, extract, screenshot |
| **Android** | ADB device control — tap, swipe, type, launch apps, screenshots |
| **Files** | Read, write, search, git operations, semantic code search |
| **Web** | Search, scrape, deep research with multi-source synthesis |
| **Memory** | Store, recall, semantic search across sessions |
| **Planning** | MCTS-guided strategic planning with scored alternatives |
| **Multi-Agent** | Fan-out, pipeline, consensus, supervisor patterns |
| **Media** | Voice notes, TTS, image generation, PDF reading |
| **Diagrams** | Mermaid diagrams from natural language |
| **Scheduling** | Recurring tasks with cron-like scheduling |

### 🔌 LLM Providers
All providers run on Prometheus's in-house SDK — zero vendor dependencies:
- **Anthropic** (Claude 4, Claude 3.5)
- **OpenAI** (GPT-4o, o1, o3)
- **Google** (Gemini 2.5, Gemini 3)
- **OpenRouter** (100+ models)
- Automatic failover, circuit breakers, and token tracking

### 📡 Channels
- **CLI** — Rich terminal UI with streaming
- **Telegram** — Chat with JARVIS from your phone
- **Discord** — Run JARVIS in your server
- **WhatsApp** — Via Twilio integration
- **Webchat** — Embedded widget for any website

### 🛡️ Security & Ethics
- Ethical Governance Module (EGM) with configurable strictness
- JWT authentication + API key management
- Secret auto-redaction in logs and outputs
- Sandboxed code execution (Docker isolation)
- Trust levels: YOLO → Normal → Paranoid

### 🦀 Rust-Powered Performance
The `jarvis_core` crate handles compute-intensive operations:
- **Vector search** — SIMD-accelerated cosine similarity (40x faster than Python)
- **BM25 scoring** — Textbook-correct keyword ranking
- **Fuzzy matching** — Trigram-based approximate string search
- **Embeddings** — BGE-small-en-v1.5 via Candle (no PyTorch dependency)
- **Entity extraction** — NER with relation classification

## Quickstart

### Prerequisites
- macOS 12+ (Apple Silicon recommended)
- Python 3.10+
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Install

```bash
# Clone the repository
git clone https://github.com/anthropic-ai/project-prometheus.git
cd project-prometheus

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python package
pip install -e ".[dev]"

# Build the Rust crate
cd src/jarvis_core_crate
pip install maturin
maturin develop --release
cd ../..

# Copy environment config
cp .env.example .env
# Edit .env and add your API keys (at minimum, one LLM provider)
```

### Run

```bash
# Start JARVIS in the terminal
prometheus

# Or use the alias
jarvis

# Or run directly
python -m jarvis.cli
```

### Configure

Edit `.env` to set your LLM provider:

```bash
# Option 1: Anthropic (recommended)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Option 3: Google Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
```

## Benchmarks

### LongMemEval — Memory Retrieval Accuracy

| System | R@5 | R@10 | R@20 |
|--------|-----|------|------|
| **🥇 Prometheus v0.1.0** | **0.9574** | **0.9766** | **0.9936** |
| Hippo (best published) | 0.74 | — | — |
| HippoRAG 2 | 0.70-0.75 | — | — |
| Raw sentence-transformers | 0.55-0.65 | — | — |

**+22 points ahead of the next best system.**

- 470 questions across 6 categories
- Zero API calls — fully local compute on Apple Silicon
- Full results in [`benchmarks/`](benchmarks/)

### Per-Category Breakdown

| Category | R@5 | Description |
|----------|-----|-------------|
| Knowledge Update | 0.986 | Retrieving the latest version of a fact |
| Multi-Session | 0.984 | Cross-session memory retrieval |
| Single-Session (Assistant) | 0.964 | What the AI said in a session |
| Single-Session (User) | 0.953 | What the human said in a session |
| Temporal Reasoning | 0.937 | Time-based memory queries |
| Preferences | 0.867 | Short preference statements |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Project Prometheus                         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Channels │  │   Core   │  │  Memory  │  │ Identity │    │
│  │          │  │          │  │          │  │          │    │
│  │ CLI      │  │ Chat Loop│  │ Store    │  │ Soul     │    │
│  │ Telegram │  │ LLM Route│  │ Retrieve │  │ Beliefs  │    │
│  │ Discord  │  │ Tool Exec│  │ Consolid │  │ Reflect  │    │
│  │ WhatsApp │  │ Sessions │  │ Entities │  │ Journal  │    │
│  │ Webchat  │  │          │  │          │  │          │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │          │
│  ┌────┴──────────────┴──────────────┴──────────────┴─────┐   │
│  │                    Tool Layer (40+)                     │   │
│  │  Code │ Desktop │ Browser │ Android │ Files │ Web │ ...│   │
│  └───────────────────────┬───────────────────────────────┘   │
│                          │                                    │
│  ┌───────────────────────┴───────────────────────────────┐   │
│  │              jarvis_core (Rust)                         │   │
│  │  Vector Search │ BM25 │ Fuzzy │ Embeddings │ NER       │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Agents  │  │ Security │  │ Services │  │  Evals   │    │
│  │          │  │          │  │          │  │          │    │
│  │ Multi-Ag │  │ EGM      │  │ Cron     │  │ LongMem  │    │
│  │ DAF      │  │ JWT/Auth │  │ Database │  │ ALE      │    │
│  │ MCTS     │  │ Sandbox  │  │ Git      │  │ SWE      │    │
│  │ Autonomy │  │ Redact   │  │ Voice    │  │ LoadTest │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Module Details

| Module | Files | LOC | Description |
|--------|-------|-----|-------------|
| `tools/` | 20+ | 11,763 | Tool handlers, dispatcher, middleware, skill marketplace |
| `operators/` | 15+ | 10,610 | Desktop, browser, Android, screen vision automation |
| `services/` | 12+ | 9,503 | Cron, database, embedding, git, voice, plugins |
| `agents/` | 8+ | 6,433 | Multi-agent orchestration, autonomy, meta-cognition |
| `memory/` | 5+ | 3,214 | Store, retriever, consolidator |
| `llm_providers/` | 6+ | 2,888 | All LLM providers via in-house SDK |
| `channels/` | 5+ | 2,502 | Multi-channel communication |
| `identity/` | 5+ | 1,993 | Self-evolving AI identity system |
| `auth/` | 4+ | 1,636 | Security, JWT, API keys, sandboxing |
| `_vendor/` | 15+ | 12,336 | CSC, DAF, EGM, KN, UME microservices |
| **Rust crate** | 5 | 2,167 | High-performance compute kernel |
| **Total** | **200+** | **~92,000** | |

## Configuration

### Trust Levels

| Level | Behavior |
|-------|----------|
| `yolo` | Auto-approve everything except destructive operations |
| `normal` | Auto-approve reads, confirm writes and commands |
| `paranoid` | Confirm every action |

### Launch Profiles

```yaml
# ~/.prometheus/config.yaml
launch_profile: balanced    # balanced | minimal | full
trust_level: normal         # yolo | normal | paranoid
llm_provider: anthropic
llm_model: claude-sonnet-4-20250514
channels:
  - cli
  - telegram
```

## Roadmap

- [x] Core agent framework with 40+ tools
- [x] Rust-powered memory with SOTA retrieval
- [x] Self-evolving identity system
- [x] Multi-channel support (CLI, Telegram, Discord, WhatsApp)
- [x] Multi-agent orchestration (fan-out, pipeline, consensus)
- [x] LongMemEval benchmark (95.7% R@5)
- [ ] macOS native app (Tauri)
- [ ] Cloud hosted version
- [ ] PyPI package (`pip install project-prometheus`)
- [ ] Mobile companion app
- [ ] Plugin marketplace
- [ ] HNSW index for 1M+ memory scale
- [ ] GPU/Metal acceleration for embeddings

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with love, Rust, and too much coffee. Powered by the belief that
your AI should remember who you are.

---

<p align="center">
  <strong>Your AI agent should know you. Prometheus makes sure it does.</strong>
</p>
