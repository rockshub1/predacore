# JARVIS Public Release — Master Ship Plan

> Ship perfect. Bottom-up. Foundation first. Test everything. Then release.

---

## Current Status (as of the Rust-memory upgrade)

Phases 1-10 shipped once already against the old architecture. We came back
to Phase 2 (Persistence / memory) because the memory layer got a major
upgrade: a Rust compute kernel (`jarvis_core_crate`) now owns all hot
memory operations, replacing the Python numpy/BM25/heuristic paths.

**Completed in this upgrade pass:**

- `jarvis_core` Rust crate built and installed (BGE-small-en-v1.5 embedding,
  SIMD vector search, BM25, trigram fuzzy, 3-tier entity extraction,
  window-aware relation classification)
- `jarvis_core` is now a **HARD dependency** — no Python fallbacks, no
  silent degradation, no numpy fallback in vector search, no Python BM25
- Vector index is in-RAM only, rebuilt from SQLite on startup (the separate
  `vectors.json` persistence file is gone)
- FIFO eviction replaced with **importance-protected eviction**
  (preference + entity types are never dropped from the cache)
- Memory cap is **per-user** (25,000) instead of global (10,000)
- Memory store has a proper `backup()` method using `sqlite3.backup()`
- Dead code deleted: `mamba_state.py`, `jarvis_core_crate/src/graph.rs`,
  `jarvis/knowledge/`
- Duplicate Python BM25 in `store.py._recall_keyword` replaced with
  `jarvis_core.bm25_search`
- Python heuristic entity extraction in `consolidator.py` replaced with
  `jarvis_core.extract_entities` (+ LLM enrichment as bonus)
- `auto_link()` now uses `jarvis_core.classify_relation` on the sentence
  where the entities co-occurred, falling back to type-pair inference
  only for low-confidence matches
- Rust `lib.rs` duplicate pymodule registrations removed
- `01_core.md` LLM-artifact preamble pollution removed
- `pyproject.toml` bumped to 0.1.0, `jarvis_core>=0.1.0` as hard dep

**What's next (Phase 11, release prep):**

- [ ] Commit the consolidated structure (git status cleanup)
- [ ] README.md rewrite
- [ ] LICENSE: Apache-2.0 (already set in pyproject)
- [ ] CI/CD workflows (test, lint, release)
- [ ] Documentation site (docs/ structure)
- [ ] PyPI publish: `jarvis_core` first, then `project-prometheus`
- [ ] GitHub release with changelog
- [ ] Final QA: fresh-install test, security audit, full test suite

---

## The Order (Why Bottom-Up?)

```
Layer 0: Foundation     — errors, config          (everything depends on these)
Layer 1: Persistence    — sessions, memory         (brain needs to remember)
Layer 2: Intelligence   — llm_providers, prompts, core  (the brain itself)
Layer 3: Capabilities   — tools, operators         (what the brain can do)
Layer 4: Orchestration  — agents, gateway          (multi-agent + routing)
Layer 5: Interface      — channels, cli, auth      (how users connect)
Layer 6: Services       — daemon, cron, plugins    (production runtime)
Layer 7: Identity       — identity, Flame          (self-evolution)
Layer 8: Vendor         — _vendor/ subsystems      (vendored deps)
Layer 9: Release        — README, LICENSE, CI/CD   (ship it)
```

Each layer gets: **Read doc → Read code → Fix → Test → Verify → Next**

---

## Phase 1: Foundation (errors.py + config.py)

### 1A: errors.py
- [ ] Read `01_core.md` errors section
- [ ] Read `errors.py` — verify all error classes are used
- [ ] Remove dead error classes (if any unused)
- [ ] Write `tests/test_errors.py`
  - Every error class instantiates correctly
  - Error codes format properly
  - Inheritance chain is correct
  - `recoverable` flag works
  - String representation includes error_code

### 1B: config.py
- [ ] Read `01_core.md` config section
- [ ] Read `config.py` — verify 3-layer loading works
- [ ] Write `tests/test_config.py`
  - Default config loads without any file
  - YAML override merges correctly
  - Env var override takes precedence
  - All 3 profiles (balanced, public_beast, enterprise_lockdown)
  - All dataclass defaults are sensible
  - `load_config()` creates directories
  - `save_default_config()` writes valid YAML
  - Port validation, bool parsing, CSV parsing
  - Edge cases: missing YAML, corrupt YAML, empty env

---

## Phase 2: Persistence (sessions.py + memory/)

### 2A: sessions.py
- [ ] Read `01_core.md` sessions section
- [ ] Read `sessions.py`
- [ ] Write `tests/test_sessions.py`
  - Session create, add_message, get_llm_messages
  - Message serialization round-trip (to_dict / from_dict)
  - Max messages truncation (500 cap)
  - Token estimation accuracy
  - Context window packing algorithm
  - SessionStore: create, get, list, delete
  - Atomic writes (crash safety)
  - Path traversal protection
  - LRU cache behavior
  - Corrupt JSONL tolerance
  - Concurrent access (session locks)

### 2B: memory/
- [ ] Read `02_infrastructure.md` memory section
- [ ] Read all 5 memory files
- [ ] Write `tests/test_memory.py`
  - UnifiedMemoryStore: store, recall, search
  - SQLite table creation and schema
  - Vector similarity search
  - BM25 text search
  - Memory consolidation (7-stage pipeline)
  - Token-budgeted retrieval
  - Memory decay and importance scoring
  - Edge cases: empty store, duplicate entries

---

## Phase 3: Intelligence (llm_providers/ + prompts.py + core.py)

### 3A: llm_providers/
- [ ] Read `02_infrastructure.md` llm_providers section
- [ ] Read all 12 provider files
- [ ] Write `tests/test_llm_providers.py`
  - Router: provider selection, failover, fallback chain
  - Circuit breaker: open/close/half-open transitions
  - Each provider: request formatting, response parsing
  - Streaming support
  - Token counting
  - Rate limit handling (retry with backoff)
  - Mock provider for testing without API keys

### 3B: prompts.py
- [ ] Read `01_core.md` prompts section
- [ ] Read `prompts.py`
- [ ] Write `tests/test_prompts.py`
  - System prompt assembly (both paths: self-evolving + legacy)
  - Persona drift patterns (all regex patterns match correctly)
  - PersonaDriftAssessment scoring
  - File caching (mtime-based)
  - Identity file loading (agent folder → fallback)
  - Runtime context injection

### 3C: core.py
- [ ] Read `01_core.md` core section
- [ ] Read `core.py`
  - Write `tests/test_core.py`
  - JARVISCore initialization
  - Process loop: user msg → LLM → tool calls → response
  - Tool dependency classification (parallel vs sequential)
  - Persona drift guard (detect + regenerate)
  - Memory injection into prompt
  - Context budget calculation
  - Sensitive key redaction
  - Meta-cognition loop detection
  - Direct tool shortcut parsing
  - Max iterations safety cap
  - Error recovery (LLM failure → retry without tools)

---

## Phase 4: Capabilities (tools/ + operators/)

### 4A: tools/ — Registry & Dispatch
- [ ] Read `03_execution.md` tools section
- [ ] Read registry.py, dispatcher.py, resilience.py, trust_policy.py, middleware.py
- [ ] Write `tests/test_tools_core.py`
  - ToolRegistry: register, get, list, categories
  - ToolDispatcher: full 13-step pipeline
  - CircuitBreaker: state transitions
  - ResultCache: TTL, LRU eviction, invalidation
  - AdaptiveTimeout: P95 tracking
  - TrustPolicy: yolo/normal/paranoid modes
  - Middleware stack: order, before/after hooks
  - Rate limiting (sliding window)
  - Ethical compliance (forbidden keywords)

### 4B: tools/handlers/
- [ ] Read `03_execution.md` handlers section
- [ ] Read all 15 handler files
- [ ] Write `tests/test_tool_handlers.py`
  - file_ops: read (2MB limit, sensitive path block), write, list_directory
  - shell: run_command (dangerous command guard), python_exec (sandbox)
  - web: web_search, web_scrape (SSRF validation)
  - memory: store + recall (3-tier fallback)
  - voice: speak (TTS), voice_note (record + transcribe)
  - desktop: desktop_control, screen_vision, android_control, browser_control
  - agent: multi_agent (4 patterns), strategic_plan (MCTS)
  - creative: image_gen, pdf_reader, diagram
  - git: all 5 git handlers
  - identity: read, update, journal, bootstrap
  - cron: schedule, list, delete
  - pipeline: execute, resume, save
  - hivemind: status, sync, evolve, scan, endorse

### 4C: tools/ — Pipeline, MCP, Marketplace
- [ ] Read pipeline.py, mcp_server.py, marketplace.py
- [ ] Write `tests/test_tools_advanced.py`
  - Pipeline: sequential, conditional, parallel fan-out, variable substitution
  - MCP: JSON-RPC stdio protocol, SDK server
  - Marketplace: skill discovery, installation, invocation

### 4D: operators/
- [ ] Read `03_execution.md` operators section
- [ ] Read all 13 operator files
- [ ] Write `tests/test_operators.py`
  - BaseOperator: execute, macro (abort, nesting limit)
  - MacDesktopOperator: all 29 actions (use MockDesktopOperator)
  - AndroidOperator: all 25 actions (use MockAndroidOperator)
  - ScreenVisionEngine: AX tree, screen diffing
  - BrowserBridge: Chrome CDP + Safari JS backends
  - SmartInput: 3-tier fallback
  - OCREngine: Vision framework + tesseract fallback
  - Retry decorator: exponential backoff, pattern matching

---

## Phase 5: Orchestration (agents/ + gateway.py)

### 5A: agents/
- [ ] Read `03_execution.md` agents section
- [ ] Read all 17 agent files
- [ ] Write `tests/test_agents.py`
  - Collaboration: 4 patterns (fan_out, pipeline, consensus, supervisor)
  - AgentEngine: type registry, capability routing, prompt compilation
  - MetaCognition: echo detection, fabrication risk, loop detection
  - SelfImprovement: failure analysis, proposal safety, auto-apply
  - DAFBridge: gRPC connection, should_use_daf logic
  - OpenClawBridge: HTTP delegation, idempotency, kill switch
  - DAF subpackage: registry, process, scheduler, task_store

### 5B: gateway.py
- [ ] Read `01_core.md` gateway section
- [ ] Read `gateway.py`
- [ ] Write `tests/test_gateway.py`
  - IncomingMessage / OutgoingMessage serialization
  - Gateway: register_channel, start, stop
  - handle_message: full pipeline (identity → rate limit → sanitize → session → process)
  - Per-user rate limiting (30/min sliding window)
  - Session lock (LRU bounded at 10K)
  - Slash commands: /model, /cancel, /new, /sessions, /resume, /link
  - Identity resolution + caching
  - Error recording to OutcomeStore

---

## Phase 6: Interface (channels/ + cli.py + auth/)

### 6A: channels/
- [ ] Read `02_infrastructure.md` channels section
- [ ] Read all 12 channel files
- [ ] Write `tests/test_channels.py`
  - WebChat: WebSocket lifecycle, streaming, static file serving
  - Telegram: message handling, duplicate suppression, typing indicator
  - Discord: DM-only mode, message handling
  - WhatsApp: HMAC webhook verification
  - ChannelHealthMonitor: circuit breaker, reconnection

### 6B: cli.py
- [ ] Read `01_core.md` cli section
- [ ] Read `cli.py`
- [ ] Write `tests/test_cli.py`
  - Argument parsing (all 11 subcommands)
  - Slash command handling (14 commands)
  - GenerationController: cancel, accumulate
  - Setup wizard flow
  - Doctor diagnostics
  - Status display

### 6C: auth/
- [ ] Read `02_infrastructure.md` auth section
- [ ] Read all 5 auth files
- [ ] Write `tests/test_auth.py`
  - OAuth2 + PKCE flow
  - Fernet token encryption
  - Sandbox: Docker + subprocess (3 tiers, 14 languages)
  - Prompt injection defense (12 patterns)
  - SSRF protection
  - JWT middleware + brute-force protection
  - Security filters

---

## Phase 7: Services (services/)

- [ ] Read `02_infrastructure.md` services section
- [ ] Read all 19 service files
- [ ] Write `tests/test_services.py`
  - Daemon: start, stop, heartbeat, health
  - RateLimiter: 3 algorithms (token bucket, sliding window, leaky bucket)
  - CronEngine: schedule, execute, persist
  - PluginSDK: discovery, loading, lifecycle
  - CodeIndex: 8 features, import graph, hybrid scoring
  - AlertingService: 5 dispatchers
  - TranscriptWriter: JSONL append-only
  - OutcomeStore: task outcomes, feedback detection
  - LaneQueue: serial per-session execution
  - GitIntegration: 4 tools
  - VoiceService: 5 TTS/STT providers
  - EmbeddingClient: 6 providers
  - ConfigWatcher: hot-reload

---

## Phase 8: Identity & Flame

- [ ] Read `03_execution.md` identity section
- [ ] Read all 18 identity files
- [ ] Write `tests/test_identity.py`
  - IdentityEngine: file management, bootstrap detection
  - SOUL_SEED immutability
  - SOUL/IDENTITY/USER evolution
  - JOURNAL append-only
  - Prompt assembly (seed + identity + user + journal)
  - Template system (12 templates)
  - Flame skill network (hivemind handlers)

---

## Phase 9: Vendor (_vendor/)

- [ ] Read `04_vendor.md`
- [ ] Read critical vendor files
- [ ] Write `tests/test_vendor.py`
  - Common: embedding, vector_store, memory_service, models, schemas
  - CSE: MCTS planner, plan cache, LLM planner
  - EGM: policy enforcement, safety guardrails
  - KnowledgeNexus: graph queries, retrieval
  - UME: user profiles, behavior tracking
  - Proto: message serialization round-trips

---

## Phase 10: Integration & E2E Tests

- [ ] Write `tests/test_e2e.py`
  - Full flow: CLI → Gateway → Core → LLM (mock) → Tools → Response
  - Multi-turn conversation with memory persistence
  - Tool execution with results flowing back
  - Persona drift detection and recovery
  - Session persistence across restarts
  - Rate limiting under load
  - Concurrent users via gateway
  - Channel message routing
  - Error recovery (LLM failure, tool timeout, etc.)

---

## Phase 11: Release Prep

### 11A: Clean Repository
- [ ] Remove all deleted/legacy files from git
- [ ] Clean up git status (commit consolidated structure)
- [ ] Remove __pycache__ from tracking (.gitignore)
- [ ] Audit for secrets, API keys, hardcoded tokens

### 11B: pyproject.toml
- [ ] Polish metadata (name, version, description, authors, URLs)
- [ ] Pin dependency versions
- [ ] Define optional dependency groups (dev, test, full)
- [ ] Entry points: `prometheus = "jarvis.cli:main"`
- [ ] Verify `pip install -e .` works clean

### 11C: README.md
- [ ] Hero section (what is JARVIS, what can it do)
- [ ] Quick start (install → configure → chat)
- [ ] Architecture overview (link to docs/)
- [ ] Feature list with status
- [ ] Configuration guide
- [ ] Development setup
- [ ] Contributing guidelines

### 11D: LICENSE
- [ ] **Apache 2.0** (recommended for AI frameworks)
  - Patent protection for contributors and users
  - Enterprise-friendly
  - Used by: TensorFlow, Kubernetes, LangChain, CrewAI
  - Allows commercial use while protecting the project

### 11E: CI/CD (GitHub Actions)
- [ ] `test.yml`: Run full test suite on push/PR
- [ ] `lint.yml`: Ruff/flake8 + type checking
- [ ] `release.yml`: Auto-publish to PyPI on tag
- [ ] `docs.yml`: Build and deploy documentation

### 11F: Documentation Site
- [ ] Move docs/ to proper structure
- [ ] Add installation guide
- [ ] Add configuration reference
- [ ] Add tool reference (all 36 tools)
- [ ] Add channel setup guides
- [ ] Add API reference

---

## Phase 12: Final QA & Ship

- [ ] Run full test suite — all green
- [ ] Run linter — zero warnings
- [ ] Fresh install test (new venv, pip install, run)
- [ ] Security audit (no secrets, no vulnerabilities)
- [ ] README review
- [ ] Create GitHub release with changelog
- [ ] Publish to PyPI
- [ ] Announce

---

## Working Method (Per Phase)

```
For each subsystem:
  1. Read the doc (01-04_*.md)        — understand the big picture
  2. Read the actual source files      — see current code
  3. Fix any bugs or issues found      — clean up
  4. Write comprehensive tests         — prove it works
  5. Run tests, fix failures           — green checkmark
  6. Move to next subsystem            — repeat
```

## Estimated Effort

| Phase | Subsystem | Files | Complexity |
|-------|-----------|-------|------------|
| 1 | Foundation | 2 | Low |
| 2 | Persistence | 6 | Medium |
| 3 | Intelligence | 20 | High |
| 4 | Capabilities | 45 | High |
| 5 | Orchestration | 18 | High |
| 6 | Interface | 18 | Medium |
| 7 | Services | 19 | Medium |
| 8 | Identity | 18 | Low |
| 9 | Vendor | 47 | Medium |
| 10 | Integration | — | High |
| 11 | Release | — | Medium |
| 12 | Final QA | — | Low |
