# JARVIS Execution Layer -- Complete Technical Documentation

> Covers `tools/`, `operators/`, `agents/`, `evals/`, `identity/`, `utils/`, and `knowledge/`.
> Every class, method, function, and constant documented.
> Written so a developer could reconstruct the entire system from this document alone.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Folder 1 -- tools/ (Tool System)](#2-folder-1----tools)
3. [Folder 2 -- operators/ (Device Automation)](#3-folder-2----operators)
4. [Folder 3 -- agents/ (Agent Framework)](#4-folder-3----agents)
5. [Folder 4 -- evals/ (Evaluation)](#5-folder-4----evals)
6. [Folder 5 -- identity/ (Self-Evolving Identity)](#6-folder-5----identity)
7. [Folder 6 -- utils/ and knowledge/](#7-folder-6----utils-and-knowledge)
8. [Cross-Folder Integration](#8-cross-folder-integration)

---

## 1. Architecture Overview

The JARVIS execution layer is a pipeline that transforms user intent into real-world action:

```
User Message
  -> gateway.py (session routing)
    -> core.py (Brain: LLM loop)
      -> tools/dispatcher.py (rate limit -> circuit breaker -> cache -> middleware -> handler)
        -> tools/handlers/*.py (36 domain-specific handlers)
          -> operators/*.py (macOS desktop, Android, browser bridge, screen vision)
          -> agents/*.py (multi-agent collaboration, DAF bridge, autonomy)
          -> memory/ (unified 3-tier memory: vector + graph + episode)
      -> identity/engine.py (persona drift guard, prompt assembly)
    -> channels/ (Webchat WS, Telegram, Discord, WhatsApp)
```

Key design principles:
- **Single source of truth**: `SubsystemFactory` initializes all subsystems once; `ToolContext` is the shared resource bag.
- **Enum-driven dispatch**: `ToolName` and `DesktopAction` enums eliminate magic strings. Since they extend `(str, Enum)`, plain string lookups still work.
- **Resilience by default**: Every tool call passes through circuit breaker, adaptive timeout, result cache, and middleware stack.
- **Security layers**: Trust policy, ethical compliance check, SSRF validation, sensitive path blocking, dangerous command regex, sandboxed execution.

---

## 2. Folder 1 -- tools/

**Purpose**: The tool system is the heart of JARVIS's capabilities. It provides a 36-tool registry, a resilient dispatch pipeline, pluggable middleware, composable pipelines, an MCP protocol bridge, a skill marketplace, and 15 handler modules organized by domain.

**File count**: 15 top-level files + 15 handler files = 30 Python source files.

### 2.1 tools/enums.py

**Purpose**: Central enum registry eliminating magic strings across tool and operator layers.

#### Class: `ToolName(str, Enum)`

47 members organized by domain:

| Group | Members |
|-------|---------|
| File/shell | `READ_FILE`, `WRITE_FILE`, `LIST_DIRECTORY`, `RUN_COMMAND` |
| Code exec | `PYTHON_EXEC`, `EXECUTE_CODE` |
| Web | `WEB_SEARCH`, `WEB_SCRAPE`, `BROWSER_AUTOMATION`, `DEEP_SEARCH`, `SEMANTIC_SEARCH`, `BROWSER_CONTROL` |
| Memory | `MEMORY_STORE`, `MEMORY_RECALL` |
| Voice | `SPEAK`, `VOICE_NOTE` |
| Desktop/mobile | `DESKTOP_CONTROL`, `SCREEN_VISION`, `ANDROID_CONTROL` |
| Agent/planning | `MULTI_AGENT`, `STRATEGIC_PLAN`, `OPENCLAW_DELEGATE` |
| Marketplace | `MARKETPLACE_LIST`, `MARKETPLACE_INSTALL`, `MARKETPLACE_INVOKE` |
| Git | `GIT_CONTEXT`, `GIT_DIFF_SUMMARY`, `GIT_COMMIT_SUGGEST`, `GIT_FIND_FILES`, `GIT_SEMANTIC_SEARCH` |
| Creative | `IMAGE_GEN`, `PDF_READER`, `DIAGRAM` |
| Identity | `IDENTITY_READ`, `IDENTITY_UPDATE`, `JOURNAL_APPEND`, `BOOTSTRAP_COMPLETE` |
| Cron | `CRON_TASK` |
| Pipeline | `TOOL_PIPELINE` |
| Hive Mind | `HIVEMIND_STATUS`, `HIVEMIND_SYNC`, `SKILL_EVOLVE`, `SKILL_SCAN`, `SKILL_ENDORSE` |
| Debug | `TOOL_STATS` |

#### Class: `ToolStatus(str, Enum)`

Standard status codes: `OK`, `ERROR`, `TIMEOUT`, `CACHED`, `CIRCUIT_OPEN`, `RATE_LIMITED`, `BLOCKED`, `DENIED`, `UNKNOWN_TOOL`.

#### Module-level constants

- `WRITE_TOOLS: frozenset[str]` -- Tools that mutate state (write_file, run_command, python_exec, execute_code). Used for cache invalidation.
- `READ_ONLY_TOOLS: frozenset[str]` -- 16 tools safe for auto-approval in normal trust mode.

Re-exports all operator enums from `operators/enums.py`.

---

### 2.2 tools/registry.py

#### Class: `ToolDefinition`

Dataclass:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Tool identifier |
| `description` | `str` | required | Human-readable description |
| `parameters` | `dict` | required | JSON Schema for tool arguments |
| `category` | `str` | `"general"` | file_ops, shell, web, memory, agent, desktop, voice, marketplace |
| `cost_estimate` | `str` | `"free"` | free, low, medium, high |
| `parallelizable` | `bool` | `True` | Can run alongside other tools |
| `requires_confirmation` | `bool` | `False` | Needs user approval |
| `timeout_default` | `int` | `30` | Seconds before timeout |

#### Class: `ToolRegistry`

Key methods: `register()`, `register_raw()`, `get()`, `list_all()`, `list_by_category()`, `get_all_definitions()`, `get_parallelizable()`.

#### Module-level: `BUILTIN_TOOLS_RAW`

List of tuples defining all 36+ built-in tools with JSON Schema parameter definitions and operational metadata.

#### Module-level: `TRUST_POLICIES`

Three levels: `"yolo"` (auto-approve all), `"normal"` (confirm writes), `"paranoid"` (confirm everything).

---

### 2.3 tools/dispatcher.py

#### Module-level: `_check_ethical_compliance(tool_name, args, trust_level) -> str | None`

Pre-execution keyword guard. Checks against `_FORBIDDEN_KEYWORDS` (delete_user_data, disable_safety, bypass_auth, drop_table, truncate_table, format_disk).

#### Class: `AdaptiveTimeoutTracker`

Computes adaptive timeouts: `effective = min(static_ceiling, max(min_floor, P95 * multiplier))`. Constructor: `window_size=20`, `multiplier=2.0`, `min_floor=5.0`, `min_samples=3`.

#### Class: `ToolDispatcher`

The 13-step dispatch pipeline:
1. Alias normalization (Gemini CLI names -> JARVIS names)
2. Rate-limit check (sliding window 120/min)
3. Circuit breaker check (fast-fail if open)
4. Cache check (return cached result if fresh)
5. Middleware before hooks
6. Confirmation check (trust policy + approval history)
7. Blocked/allowed list enforcement
7b. Ethical compliance (keyword guard)
8. Handler lookup + execution with adaptive timeout via `asyncio.wait_for`
9. Middleware after hooks
10. ToolError handling (user errors don't trip circuit breaker)
11. Latency recording for adaptive timeout
12. Metrics recording (Prometheus counters/histograms)
13. Output sanitization (secret redaction)

---

### 2.4 tools/executor.py

#### Class: `ToolExecutor`

Thin facade wiring together SubsystemFactory, TrustPolicyEvaluator, ToolDispatcher, and MarketplaceManager. Public API: `execute(tool_name, arguments, confirm_fn) -> str`. Includes HTTP retry helper with SSRF validation and startup health checks.

---

### 2.5 tools/resilience.py

#### Class: `ToolCircuitBreaker`

Per-tool circuit breaker: CLOSED -> OPEN (after 3 failures) -> HALF_OPEN (after 60s cooldown) -> CLOSED (on probe success). Thread-safe.

#### Class: `ToolResultCache`

LRU + TTL cache. Default TTLs: read_file 30s, web_search 120s, web_scrape 300s. Error results never cached. Write operations trigger selective invalidation.

#### Class: `ExecutionHistory`

Ring buffer (deque maxlen=500) of `ExecutionRecord` entries. Thread-safe. Tracks total calls, errors, per-tool counts, latency stats.

---

### 2.6 tools/trust_policy.py

#### Class: `TrustPolicyEvaluator`

Methods: `requires_confirmation()`, `is_blocked()`, `assess_risk()` (with ApprovalContext including risk escalation for critical patterns like `rm -rf`, `sudo`), `check_previous_approval()`, `record_approval()`.

#### Class: `ApprovalHistory`

SQLite-backed persistence of approval decisions using SHA-256 hash of serialized args.

---

### 2.7 tools/middleware.py

Onion-model middleware stack. Built-in implementations:

| Middleware | Order | Purpose |
|-----------|-------|---------|
| `InputSanitizerMiddleware` | 5 | Strip whitespace, remove null bytes, truncate oversized args |
| `LoggingMiddleware` | 10 | Structured log with trace_id |
| `PerToolRateLimitMiddleware` | 15 | Per-tool throttling |
| `MetricsMiddleware` | 20 | In-memory counters, histograms, P50/P95/P99 per tool |
| `AuditTrailMiddleware` | 30 | Append-only audit log (deque, max 10K) |
| `OutputTruncationMiddleware` | 90 | 50KB max (25KB head + 5KB tail) |

---

### 2.8 tools/pipeline.py

Multi-step tool chaining. `ToolPipeline` supports sequential execution with variable substitution (`{{prev}}`, `{{step.N}}`), conditional steps, approval gates with durable SQLite-backed resume tokens (24h expiry), and parallel fan-out. Max 20 steps, 300s total timeout.

---

### 2.9 tools/mcp_server.py

MCP protocol bridge. `MCPStdioServer` implements JSON-RPC over stdio for Claude Code integration. `build_sdk_mcp_server()` creates in-process SDK MCP server with zero subprocess overhead.

---

### 2.10 tools/agent_sdk_adapter.py

`JarvisToolAdapter` wraps every registry tool as an async function compatible with Anthropic Agent SDK. Convenience: `create_agent_sdk_tools()`, `create_jarvis_agent()`.

---

### 2.11 tools/subsystem_init.py

`SubsystemFactory.create_all()` initializes all subsystems independently: desktop, unified_memory, legacy_memory, mcts, voice, llm_collab, openclaw, sandbox. Returns `SubsystemBundle`.

---

### 2.12 tools/marketplace.py and skill_marketplace.py

`MarketplaceManager` handles OpenClaw skill import and executor wiring. `SkillMarketplace` manages discovery, installation, invocation with 6 built-in skills (web-scraper, code-executor, file-manager, email-sender, daily-briefing, data-analyzer) and OpenClaw external skill import.

---

### 2.13 tools/health.py

`HealthDashboard` aggregates circuit breaker states, cache stats, execution history, adaptive timeouts, middleware status into a health score (0-100).

---

### 2.14 Handler Modules (tools/handlers/)

15 handler files providing 36 async handler functions. Each raises `ToolError` on failure.

**_context.py**: Shared `ToolContext` dataclass (resource bag for all handlers), `ToolError` hierarchy (9 error kinds), convenience constructors, sensitive path patterns, web cache, multi-agent recursion guard (max depth 3 via `contextvars`).

**file_ops.py**: `read_file` (2MB limit, sensitive path blocking), `write_file` (system path blocking), `list_directory` (500 entry limit, path traversal prevention).

**shell.py**: `run_command` (dangerous command regex guard, configurable timeout, 50KB output truncation), `python_exec` (Docker or subprocess sandbox, pattern blocklist when no sandbox, 256MB memory limit), `execute_code` (14 runtimes via Docker).

**web.py**: `web_search` (DDG library + API fallback, 5-min cache), `web_scrape` (HTML strip, SSRF validation, 30KB), `browser_automation` (Playwright), `deep_search` (multi-source fetch + synthesize), `semantic_search` (hybrid BM25 + vector across memory and files).

**memory.py**: `memory_store` (3-tier fallback: unified -> legacy -> session), `memory_recall` (semantic/keyword/entity search, scoped team memory).

**voice.py**: `speak` (Kokoro neural TTS), `voice_note` (record via sox + auto-transcribe via Whisper API).

**desktop.py**: `desktop_control` (45s timeout), `screen_vision` (12 actions with auto-retry), `android_control` (ADB), `browser_control` (22 actions via BrowserBridge singleton).

**agent.py**: `multi_agent` (4 collaboration patterns, dynamic agent specs via LLM meta-planning, team memory, optional DAF dispatch), `strategic_plan` (MCTS), `openclaw_delegate` (kill switch guard).

**creative.py**: `image_gen` (DALL-E 3), `pdf_reader` (PyMuPDF, 200 pages), `diagram` (Mermaid via mmdc).

**git.py**: 5 handlers delegating to git_integration and code_index services.

**marketplace.py**: 3 handlers for skill listing, installation, invocation.

**identity.py**: 4 handlers for identity file CRUD via IdentityEngine.

**cron.py**: Scheduled recurring tasks (max 20, persisted to JSON, dangerous command guard).

**pipeline_handler.py**: 7 actions (execute, resume, save, run, list, delete, show) with workflow template persistence.

**hivemind.py**: 5 Flame skill network handlers (status, sync, evolve, scan, endorse).

**stats.py**: Debug dashboard with HealthDashboard integration.

---

## 3. Folder 2 -- operators/

**Purpose**: Platform-specific device automation with uniform `execute(action, params) -> dict` interface.

### 3.1 operators/enums.py

Single source of truth: `DesktopAction` (29 actions), `AndroidAction` (25 actions), `VisionAction` (9 actions), `ScreenshotQuality` (FULL/FAST/THUMBNAIL). Grouped frozensets: `SMART_ACTIONS`, `NATIVE_ONLY_ACTIONS`, `NATIVE_CAPABLE_ACTIONS`.

### 3.2 operators/base.py

`BaseOperator(ABC)`: Abstract interface. Abstract: `platform`, `available`, `supported_actions`, `execute()`, `health_check()`. Concrete: `execute_macro()` (multi-step with abort support, max 50 steps, max 3 nesting depth), `telemetry()`, `capabilities()`.

`OperatorError(RuntimeError)`: With `action`, `recoverable`, `suggestion`.

`MacroAbortToken`: Thread-safe abort signaling.

### 3.3 operators/desktop.py

`MacDesktopOperator(BaseOperator)`: 800+ lines. Handles all 29 DesktopAction values. Two backends: native PyObjC (via `MacDesktopNativeService`) and legacy AppleScript+pyautogui. Features: 60+ key code mappings, multi-quality screenshots, smart input (3-tier), `AsyncBridge` singleton for sync-to-async, retry decorator on flaky methods.

### 3.4 operators/android.py

`AndroidOperator(BaseOperator)`: ADB-based automation. `AndroidElement` dataclass with bounds, center, label. Handles all 25 AndroidAction values. UI dump parses uiautomator XML. Text input via broadcast for reliability.

### 3.5 operators/screen_vision.py

`ScreenVisionEngine`: AX-first screen understanding. Strategy: AX tree (~1ms) -> targeted screenshots -> Vision model (Claude) -> screen diffing. Data types: `UIElement`, `ScreenState`. 12 methods including `execute_task()` for multi-step automated interaction.

### 3.6 operators/browser_bridge.py

`BrowserBridge`: Instant DOM access (~5-20ms). Two backends: `_ChromeCDP` (DevTools Protocol websocket) and `_SafariJSBridge` (AppleScript). Embedded JS tree walker extracts 200 visible interactive elements with CSS selectors.

### 3.7 operators/smart_input.py

`SmartInputEngine`: 3-tier bulletproof text input. Tier 1: AppleScript `do script`. Tier 2: AX set_value. Tier 3: Keystroke simulation with read-back verification and retry.

### 3.8 operators/native_service.py

`MacDesktopNativeService`: PyObjC Accessibility API. App launch/focus, keyboard/mouse events via Quartz CGEvent, full AX element tree walking.

### 3.9 operators/ocr_fallback.py

`OCREngine`: macOS Vision framework (200-500ms, Neural Engine) with tesseract CLI fallback (1-3s).

### 3.10 operators/browser.py

`run_browser_automation()`: Playwright-based headless browser with full action support (navigate, click, type, screenshot, extract, wait, evaluate JS, download, PDF, scroll, hover, fill).

### 3.11 operators/mock.py

`MockDesktopOperator`, `MockAndroidOperator`: Deterministic mocks for testing. Record calls, return configurable responses, support failure injection, assert_called/assert_not_called.

### 3.12 operators/retry.py

`with_retry()` decorator and `async_with_retry()`: Exponential backoff on transient errors with pattern matching (kAXErrorCannotComplete, timed out, connection reset).

---

## 4. Folder 3 -- agents/

### 4.1 agents/collaboration.py

4 patterns (`FAN_OUT`, `PIPELINE`, `CONSENSUS`, `SUPERVISOR`). Data models: `AgentSpec`, `TaskPayload`, `AgentResult`, `CollaborationResult`. `AgentTeam` orchestrator.

### 4.2 agents/engine.py

`AgentType` (frozen dataclass): type_id, system_prompt, allowed_tools, capabilities. `DynamicAgentSpec`: task-specific specialization. `AGENT_TYPES` registry (researcher, analyst, critic, planner, coder, synthesizer, generalist, creative, devops). `CapabilityRouter` for keyword-based routing. `compile_dynamic_agent_prompt()` for prompt assembly.

### 4.3 agents/autonomy.py

`OpenClawBridgeRuntime`: HTTP delegation with retry, async polling, SQLite idempotency cache, JSONL action ledger, kill switch guard.

### 4.4 agents/meta_cognition.py

`evaluate_response()`: Echo detection (SequenceMatcher), fabrication risk (regex), grounding check, empty response check. `detect_loop()`: Repeated tool call detection.

### 4.5 agents/self_improvement.py

`SelfImprovementEngine`: Failure analysis, improvement proposals (5 categories, 3 risk levels), auto-apply low-risk changes. Safety: protected files, daily modification limit, audit trail, human approval for high-risk.

### 4.6 agents/daf_bridge.py

`DAFBridge`: Connects JARVIS to DAF for process-level parallelism via gRPC. Decision logic: `should_use_daf()` based on agent count and pattern. Availability check with 60s TTL caching.

### 4.7 agents/daf/ subpackage

10 files: agent_registry, agent_process, service (gRPC), task_store (Memory/Redis), scheduler, health, health_api, metrics_util, self_optimization. Lazy imports via `__getattr__`.

---

## 5. Folder 4 -- evals/

`claude_proxy.py`: FastAPI OAuth proxy with round-robin token cycling for rate limit avoidance. `codex_proxy.py`: OpenAI API compatibility. `swe_bench.py`: Coding evaluation (EvalTask -> agent -> diff comparison -> test execution -> score 0-1). `ale_bench.py`: Agent language evaluation. `load_test.py`: Dispatch pipeline stress testing.

---

## 6. Folder 5 -- identity/

`IdentityEngine`: Manages 10 markdown files per agent in `~/.prometheus/agents/{name}/`. SOUL_SEED.md is immutable. SOUL.md, IDENTITY.md, USER.md evolve over time. JOURNAL.md is append-only. Bootstrap detection checks 7 required files. mtime-based file caching. Prompt assembly layers seed + identity + user + journal. `UserProfile` dataclass companion to USER.md. 12 markdown templates in templates/ for new agent creation.

---

## 7. Folder 6 -- utils/ and knowledge/

`utils/cache.py`: `TTLCache` (dict-based, monotonic clock, 10K max), `LRUCache` (OrderedDict, 128 max), `hash_key()` (SHA-256).

`knowledge/__init__.py`: Optional bridge to Knowledge Nexus. All exports `None` when not installed. Currently unused -- JARVIS uses UnifiedMemoryStore.

---

## 8. Cross-Folder Integration

### Request flow example: "Click the Save button in Safari"

1. `identity/engine.py` assembles system prompt (SOUL_SEED + SOUL + IDENTITY + USER)
2. `tools/registry.py` provides tool definitions to LLM
3. LLM selects `browser_control(action="click", text="Save")`
4. `tools/dispatcher.py` runs 13-step pipeline (rate limit -> circuit breaker -> cache -> middleware -> trust -> handler)
5. `tools/handlers/desktop.py` dispatches to `operators/browser_bridge.py`
6. `BrowserBridge` detects Safari, uses `_SafariJSBridge`, executes click via AppleScript
7. Result flows back through middleware (logging, metrics, audit) to user

### Connections

- **tools/ -> operators/**: `handlers/desktop.py` imports MacDesktopOperator, ScreenVisionEngine, AndroidOperator, BrowserBridge from operators/
- **agents/ -> tools/**: `handlers/agent.py` creates AgentTeam using collaboration patterns from agents/. Each agent calls `llm_for_collab.chat()` with prompts compiled by agents/engine.py
- **identity/ -> core.py -> all**: IdentityEngine shapes every LLM response via system prompt assembly
- **evals/ -> tools/ + agents/**: Evaluation pipelines exercise the full dispatch chain
