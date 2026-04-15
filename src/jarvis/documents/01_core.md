# Core Layer Documentation

Complete technical reference for the 8 JARVIS core files that form the central nervous system of the agent framework. Every class, method, function, constant, and data flow is documented.

---

## Table of Contents

1. [`__init__.py`](#1-__init__py) -- Package metadata
2. [`core.py`](#2-corepy) -- The Brain: LLM loop, tool dispatch, persona drift guard
3. [`gateway.py`](#3-gatewaypy) -- Message router: channels, identity, sessions, core
4. [`config.py`](#4-configpy) -- YAML config loader (3-layer: defaults, file, env)
5. [`prompts.py`](#5-promptspy) -- System prompt assembly (SOUL_SEED, identity, runtime)
6. [`sessions.py`](#6-sessionspy) -- Session persistence (filesystem-backed)
7. [`cli.py`](#7-clipy) -- CLI entry point (start, stop, auth, status)
8. [`errors.py`](#8-errorspy) -- Structured error hierarchy
9. [Architecture & Data Flow](#9-architecture--data-flow)

---

## 1. `__init__.py`

**Path**: `src/jarvis/__init__.py`

### File Overview

Package root for the JARVIS agent. Contains only the module docstring and version constant. The docstring declares the three primary entry modes.

### Module-Level Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `__version__` | `str` | `"0.1.0"` | Semantic version of the JARVIS package |

### Docstring Summary

Documents three top-level CLI commands:
- `prometheus chat` -- interactive terminal session
- `prometheus start` -- 24/7 daemon mode
- `prometheus setup` -- guided onboarding wizard

Declares that JARVIS wraps the full stack (CSC, DAF, WIL, EGM, KN) into a single-process conversational agent.

---

## 2. `core.py`

**Path**: `src/jarvis/core.py`

### File Overview

The "brain" of JARVIS. Orchestrates the full agent loop: receive user message, build prompt, call LLM, execute tools, apply persona drift guard, and return response. All subsystems are imported from their own modules; this file is the slim orchestrator.

### Key Imports

**Standard library**: `asyncio`, `json`, `logging`, `os`, `re`, `time`, `collections.abc.Callable`, `pathlib.Path`, `typing.Any`

**Internal**:
- `.config.JARVISConfig` -- master configuration
- `.agents.meta_cognition.evaluate_response`, `.agents.meta_cognition.should_ask_for_help` -- response quality evaluation and loop detection
- `.services.outcome_store.OutcomeStore`, `.services.outcome_store.TaskOutcome`, `.services.outcome_store.detect_feedback` -- outcome tracking
- `.sessions.Session` -- conversation session
- `.services.transcripts.TranscriptWriter` -- conversation transcript recording
- `.llm_providers.router.LLMInterface` -- provider routing
- `.prompts` -- all persona drift patterns and system prompt builder
- `.tools.executor.ToolExecutor` -- tool execution
- `.tools.registry` -- tool definitions (BUILTIN_TOOLS_RAW, MARKETPLACE_TOOLS_RAW, OPENCLAW_BRIDGE_TOOLS_RAW)

### Module-Level Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `_SENSITIVE_KEYS_EXACT` | `frozenset[str]` | 12 key names | Exact-match sensitive field names for redaction |
| `_SENSITIVE_SUBSTRINGS` | `tuple[str, ...]` | 6 substrings | Substring patterns for sensitive key detection |
| `_MUTATING_TOOLS` | `frozenset[str]` | 10 tool names | Tools that mutate state (cannot run in parallel) |
| `_FILE_READ_TOOLS` | `frozenset[str]` | 3 tool names | Read-only file tools (can run in parallel) |
| `BUILTIN_TOOLS` | `list[dict]` | derived | Flat list of built-in tool definitions (legacy compat) |
| `OPENCLAW_BRIDGE_TOOLS` | `list[dict]` | derived | Flat list of OpenClaw bridge tool definitions |
| `MARKETPLACE_TOOLS` | `list[dict]` | derived | Flat list of marketplace tool definitions |

### Module-Level Functions

#### `_llm_error_message(exc: Exception) -> str`

Maps provider failure exceptions to user-facing error messages. Inspects the exception text for keywords:
- "usage limit" / "quota" / "upgrade" -> usage limit message
- "429" / "rate limit" -> rate limit message
- "overloaded" / "529" -> overloaded message
- default -> generic connectivity message

#### `_context_budget_for_provider(provider: str, model: str) -> tuple[int, int]`

Chooses a safe prompt token budget based on the active provider and model. Returns `(budget_tokens, history_min_tokens)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | Active provider name |
| `model` | `str` | Active model name |

**Logic**:
- Default: `budget_tokens=36_000`, `history_min_tokens=6_000`
- `claude-oauth`: `budget_tokens=24_000`, `history_min_tokens=4_000`
- Opus/Sonnet-4 models on claude-oauth: `budget_tokens=22_000`

#### `_is_sensitive_key(key: str) -> bool`

Checks if a key name refers to a sensitive field. First checks exact match against `_SENSITIVE_KEYS_EXACT`, then substring match against `_SENSITIVE_SUBSTRINGS`.

#### `_redact_tool_args(args: dict[str, Any], max_len: int = 200) -> str`

Serializes tool arguments for logging, replacing sensitive fields with `***REDACTED***`. Truncates output to `max_len` characters.

#### `_task_exception_handler(task: asyncio.Task) -> None`

Done-callback for background `asyncio.Task` objects. Logs exceptions that would otherwise be silently dropped. Ignores cancelled tasks.

#### `_classify_tool_dependencies(tool_calls: list[dict[str, Any]]) -> tuple[list, list]`

Splits tool calls into `(independent, dependent)` lists for parallel vs. sequential execution.

**Rules**:
1. Read-only tools with no shared resources -> independent (parallelizable)
2. Tools in `_MUTATING_TOOLS` -> dependent (sequential)
3. If both reads and writes exist, checks for path conflicts (same file)
4. Reads conflicting with writes are moved to dependent

---

### Class: `JARVISCore`

```python
class JARVISCore:
```

**Purpose**: The conversational agent brain. Orchestrates the full agent loop: user message -> build prompt -> call LLM -> execute tools -> return response.

**Class Constants**:

| Name | Value | Description |
|------|-------|-------------|
| `MAX_TOOL_ITERATIONS` | `10` | Default maximum tool loop iterations |

#### Constructor: `__init__(self, config: JARVISConfig)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `JARVISConfig` | Master configuration object |

**Instance Variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `config` | `JARVISConfig` | Stored config |
| `llm` | `LLMInterface` | LLM provider router |
| `tools` | `ToolExecutor` | Tool execution engine |
| `_system_prompt` | `str` | Assembled system prompt |
| `_tool_definitions` | `list[dict]` | Active tool schemas for LLM |
| `_max_tool_iterations` | `int` | Effective max iterations (clamped 1..100) |
| `_persona_drift_guard_enabled` | `bool` | Whether drift guard is active |
| `_persona_drift_threshold` | `float` | Drift score threshold [0.0, 1.0] |
| `_persona_drift_max_regens` | `int` | Max regen attempts per response |
| `_persona_anchor_terms` | `set[str]` | Identity anchor terms for drift scoring |
| `_transcript` | `TranscriptWriter` | Conversation transcript writer |
| `_transcript_sessions` | `set` | Session IDs with active transcripts |
| `_memory_retriever` | `MemoryRetriever or None` | Unified memory retriever |
| `_outcome_store` | `OutcomeStore or None` | Outcome tracking store |
| `_world_model` | `None` | JAX world model (currently disabled) |
| `_tool_timeout` | `int` | Per-tool timeout in seconds (env: `JARVIS_TOOL_TIMEOUT_SECONDS`, default 60) |

**Constructor Logic**:
1. Propagates OpenClaw/marketplace flags to env for MCP subprocess
2. Creates `LLMInterface` and `ToolExecutor`
3. Injects `ToolDispatcher` into `ClaudeCodeProvider` for in-process MCP
4. Builds system prompt via `_get_system_prompt()`
5. Loads tool definitions; adds OpenClaw and marketplace tools if enabled
6. Applies `allowed_tools` / `blocked_tools` filters
7. Sets up TranscriptWriter, MemoryRetriever, OutcomeStore
8. Hard-caps max_tool_iterations at 100 (safety)

#### Method: `refresh_system_prompt(self) -> None`

Rebuilds the system prompt by re-calling `_get_system_prompt(config)`. Called after identity file updates.

**Side effects**: Updates `self._system_prompt`.

#### Method: `_build_persona_anchor_terms(self) -> set[str]`

Builds the canonical set of identity anchor terms used by drift scoring. Includes config name, "jarvis", "prometheus", "project prometheus", and optionally "antigravity". Also reads comma-separated terms from `JARVIS_PERSONA_ANCHORS` env var.

**Returns**: `set[str]` -- lowercased anchor terms.

#### Method: `_assess_persona_drift(self, user_message: str, assistant_message: str, tools_used: int = 0) -> PersonaDriftAssessment`

Computes a deterministic [0,1] drift score from textual heuristics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_message` | `str` | -- | Original user message |
| `assistant_message` | `str` | -- | Candidate LLM response |
| `tools_used` | `int` | `0` | Number of tool calls in this turn |

**Logic flow**:
1. Returns score=0 if guard is disabled
2. Empty text with tools_used > 0 -> score 0 (tool-only turn)
3. Empty text with no tools -> score 1.0 ("empty_response")
4. Scans for `PERSONA_DRIFT_PATTERNS` (generic model identity, foreign identity, capability denial)
5. Checks `DIRECT_CAPABILITY_DENIAL_RE` (+0.70)
6. Checks tool capability denial patterns (+0.20)
7. If no tools used: checks for unverified model switch claims (+0.80), unverified action claims (+0.35/+0.60)
8. If user asked an identity query and no anchor term found in response (+0.35)
9. Deduplicates overlapping capability-denial reasons to avoid inflated scores
10. Clamps score to [0.0, 1.0]

**Returns**: `PersonaDriftAssessment`

#### Method: `_persona_regeneration_instruction(self, assessment: PersonaDriftAssessment, tools_used: int) -> str`

Generates the system instruction injected when drift is detected. Instructs the model to regenerate with persona alignment, mentioning drift triggers and requirements (speak as JARVIS, don't deny capabilities, etc.).

#### Method: `_apply_persona_drift_guard(self, user_message, messages, initial_content, tools_used=0) -> tuple[str, PersonaDriftAssessment, int]`

Regenerates the response if drift score exceeds threshold.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_message` | `str` | -- | Original user message |
| `messages` | `list[dict[str,str]]` | -- | Full conversation history |
| `initial_content` | `str` | -- | Initial LLM response text |
| `tools_used` | `int` | `0` | Tool calls in this turn |

**Returns**: `(final_content, best_assessment, regen_count)`

**Logic**:
1. Assess initial content
2. If guard disabled or below threshold -> return immediately
3. Loop up to `_persona_drift_max_regens` times:
   - Append the drifted response + regeneration instruction to messages
   - Call LLM with no tools
   - Compare new assessment; keep candidate if improved
4. Return best content

#### Method: `_build_runtime_memory_context(self, user_id: str, query: str) -> str`

Fetches relevant persistent memories to ground the current turn. Delegates to `MemoryRetriever.build_context()` (see `memory/retriever.py`), which builds a 5-section budgeted context block: preferences, entity context, semantic search, fuzzy matches, recent episodes. Max tokens configurable via env `JARVIS_MEMORY_PROMPT_MAX_TOKENS` (default 3000).

The retriever uses `jarvis_core` (Rust) for all hot compute — SIMD cosine vector search, BM25 keyword fallback, trigram fuzzy matching, synonym expansion. jarvis_core is a hard dependency; there is no Python fallback path.

**Returns**: Formatted memory context string, or empty string.

#### Method: `_store_turn_memory(self, user_id, message, response, tools_used, session_id, provider) -> None`

Stores the conversation turn in unified memory after completion. Variable importance: tool-heavy or long responses get higher importance (2/3/4). Every 50 memories, triggers event-driven consolidation in a background task.

**Side effects**: Writes to unified memory store. May create a background consolidation task.

#### Method: `_maybe_identity_reflect(self) -> None`

Triggers identity reflection if due. Non-fatal, non-blocking. Creates a `ReflectionEngine` on first call, then ticks a counter. At interval boundaries, appends a journal checkpoint to the identity engine.

#### Method: `_extract_direct_tool_shortcut(self, message: str) -> tuple[str, dict[str, Any]] | None`

Parses direct imperative tool requests from user text (e.g., "Run memory_recall" or "execute memory_store {...}"). Uses `DIRECT_TOOL_PREFIX_RE`. Returns `(tool_name, args)` or `None`.

#### Method: `set_model(self, provider: str | None = None, model: str | None = None)`

Hot-swaps the active LLM model/provider without restarting. Delegates to `self.llm.set_active_model()`.

#### Method: `process(self, user_id, message, session, confirm_fn=None, stream_fn=None, event_fn=None) -> str`

**The main agent loop.** Processes a user message through the full pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str` | -- | User identifier |
| `message` | `str` | -- | User message text |
| `session` | `Session` | -- | Current conversation session |
| `confirm_fn` | `Callable or None` | `None` | Tool confirmation callback |
| `stream_fn` | `Callable or None` | `None` | Streaming partial response callback |
| `event_fn` | `Callable or None` | `None` | Live UI event callback |

**Returns**: `str` -- final response text.

**Full Logic Flow**:

1. **Setup**: Record start time, reset loop warnings, start transcript session, record user message to transcript
2. **Feedback detection**: Check if message is feedback on last response; update outcome store
3. **Direct tool shortcut**: If message matches `DIRECT_TOOL_PREFIX_RE`, execute tool directly and return result
4. **Bootstrap check**: If identity engine needs bootstrap, use focused SOUL_SEED + BOOTSTRAP.md prompt with tool instructions
5. **Memory injection**: Retrieve relevant memories and append to system prompt
6. **Context window**: Build message history with token-aware packing (provider-specific budget)
7. **Agent loop** (up to `_max_tool_iterations` iterations):
   a. Emit "thinking" event
   b. **Context protection**: Trim old tool results if total tokens > 28,000
   c. **LLM call**: Send messages + tool definitions; stream tokens on final iteration or when provider owns the tool loop
   d. **Error handling**: On LLM failure, retry once without tools; on second failure, return error message
   e. **MCP tool tracking**: Capture tools executed internally by MCP providers
   f. **Tool announcement detection**: If LLM announces tools in text without actual tool_calls (short <300 chars), retry with enforcement prompt
   g. **Tool refusal detection**: If LLM claims tools unavailable, retry once with tool enforcement
   h. **No tool calls (final response)**:
      - If content is a tool-call marker or empty after tool use, ask LLM to summarize
      - Apply persona drift guard
      - Run meta-cognition response evaluation
      - Stream final content
      - Emit response event with tokens, cost, etc.
      - Record outcome, store turn memory, trigger identity reflection
      - Return final content
   i. **Execute tool calls**:
      - Classify into independent/dependent via `_classify_tool_dependencies()`
      - Run independent tools in parallel via `asyncio.gather()`
      - Run dependent tools sequentially
      - Each tool: emit start/end events, enforce timeout, cap result to 15,000 chars, append to messages
   j. **Meta-cognition loop detection**: Check if stuck; on 2nd warning, force-break the loop
8. **Exhausted iterations**: Force a final response without tools, apply drift guard, record as failure

#### Method: `_record_outcome(self, ...) -> None`

Records a `TaskOutcome` to the outcome store. Non-fatal on failure. Also feeds the outcome to the world model for online learning (single SGD step) if available and an embedding can be computed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | User identifier |
| `message` | `str` | User message |
| `response` | `str` | Agent response |
| `tools_used` | `list[str]` | Tool names used |
| `tool_errors` | `list[str]` | Error messages from tools |
| `iterations` | `int` | Loop iterations consumed |
| `start_time` | `float` | Timestamp when processing started |
| `provider` | `str` | LLM provider name |
| `model` | `str` | LLM model name |
| `usage` | `dict[str, Any]` | Token usage dict |
| `drift_score` | `float` | Persona drift score |
| `drift_regens` | `int` | Number of drift regenerations |
| `session_id` | `str` | Session identifier |
| `success` | `bool` | Whether the interaction succeeded |
| `error` | `str or None` | Error message if failed |

#### Method: `get_tool_list(self) -> list[str]`

Returns a list of available tool names from `_tool_definitions`.

#### Method: `get_status(self) -> dict[str, Any]`

Returns comprehensive core status dictionary including: provider, model, trust level, tools count, memory items, Antigravity status, max tool iterations, persona drift guard config, OpenClaw status, world model stats.

---

## 3. `gateway.py`

**Path**: `src/jarvis/gateway.py`

### File Overview

The "front door" of JARVIS. All messages from all channels (CLI, Telegram, Discord, Web) flow through the Gateway. It normalizes messages, resolves sessions, submits to the lane queue for serial execution, and returns responses.

### Key Imports

**Standard library**: `asyncio`, `collections`, `logging`, `re`, `time`, `uuid`, `abc.ABC`, `collections.abc`, `dataclasses`, `typing`

**Internal**:
- `.auth.middleware.AuthMiddleware` -- JWT authentication
- `.auth.security.sanitize_user_input` -- input sanitization
- `.channels.health.ChannelHealthMonitor` -- channel health tracking
- `.config.JARVISConfig` -- configuration
- `.services.lane_queue.LaneQueue` -- serial execution queue
- `.services.outcome_store.OutcomeStore`, `TaskOutcome` -- outcome tracking
- `.services.rate_limiter.RateLimitConfig`, `RateLimiter`, `default_api_limits`
- `.services.identity_service.IdentityService` -- cross-channel identity
- `.sessions.Session`, `SessionStore` -- session management
- `.utils.cache.LRUCache` -- LRU caching

### Module-Level Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `IDENTITY_CACHE_MAX_SIZE` | `int` | `500` | LRU cache entries for identity resolution |
| `IDENTITY_CACHE_TTL_SECONDS` | `int` | `600` | TTL for cached identity lookups (10 min) |
| `SESSION_LOCKS_MAX` | `int` | `10_000` | Max per-session locks before LRU eviction |
| `USER_RATE_LIMIT_PER_MINUTE` | `int` | `30` | Max incoming messages per user per minute |
| `RATE_LIMIT_WINDOW_SECONDS` | `float` | `60.0` | Sliding window for per-user rate limiting |
| `RATE_LIMIT_CLEANUP_USER_THRESHOLD` | `int` | `1_000` | Trigger stale-user cleanup above this count |
| `RATE_LIMIT_CLEANUP_ENTRY_THRESHOLD` | `int` | `50_000` | Trigger cleanup above this total entry count |
| `GATEWAY_ERROR_MSG_TRUNCATE` | `int` | `500` | Max chars of user message stored in error outcomes |
| `SECONDS_PER_HOUR` | `int` | `3_600` | Time constant |
| `SECONDS_PER_DAY` | `int` | `86_400` | Time constant |

---

### Dataclass: `IncomingMessage`

Normalized message from any channel.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | `str` | -- | Channel source: "cli", "telegram", "discord", "webchat", "webhook" |
| `user_id` | `str` | -- | User identifier |
| `text` | `str` | -- | Message text |
| `session_id` | `str or None` | `None` | Session ID (None = auto-assign) |
| `attachments` | `list[dict[str, Any]]` | `[]` | File attachments |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |
| `timestamp` | `float` | `time.time()` | Message timestamp |
| `message_id` | `str` | `uuid4()` | Unique message ID |

### Dataclass: `OutgoingMessage`

Response message to send back to a channel.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | `str` | -- | Target channel |
| `user_id` | `str` | -- | User identifier |
| `text` | `str` | -- | Response text |
| `session_id` | `str` | -- | Session identifier |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |
| `timestamp` | `float` | `time.time()` | Response timestamp |
| `tool_calls` | `list[dict[str, Any]]` | `[]` | Tool calls made |
| `thinking` | `str or None` | `None` | Agent reasoning (debug info) |

### Abstract Class: `ChannelAdapter(ABC)`

Base class for messaging channel adapters.

**Class Attribute**: `channel_name: str = "base"`

- `start(self) -> None` (abstract, async): Start listening for messages
- `stop(self) -> None` (abstract, async): Stop the adapter
- `send(self, message: OutgoingMessage) -> None` (abstract, async): Send a response
- `set_message_handler(self, handler)`: Set incoming message callback

### Class: `Gateway`

Central message router and session manager.

#### Constructor: `__init__(self, config: JARVISConfig, process_fn: Callable)`

Sets up session store, lane queue, channels dict, stats, health monitor, outcome store, identity service, identity cache, session locks (LRU bounded), rate limiter, user rate limiting, and optional JWT auth middleware.

#### Method: `register_channel(self, adapter: ChannelAdapter) -> None`

Registers a channel adapter with message handler, gateway reference injection, and health monitor registration.

#### Method: `start(self) -> None` (async)

Starts gateway: sets running flag, starts all channel adapters, starts health monitor, starts identity heartbeat.

#### Method: `stop(self) -> None` (async)

Stops gateway: stops health monitor, all channels, identity heartbeat, and lane queue.

#### Method: `_handle_brain_command(self, text) -> str`

Handles `/brain [reset]`. Shows world model stats or resets.

#### Method: `_handle_model_command(self, text, user_id, channel) -> OutgoingMessage | None`

Handles `/model [provider]`. Shows current or switches provider.

#### Method: `_handle_session_command(self, text, user_id, channel) -> OutgoingMessage | None`

Handles `/cancel`, `/new`, `/sessions`, `/resume [N|ID]`.

#### Method: `_get_session_lock(self, session_id) -> asyncio.Lock` (async)

Gets/creates per-session lock with LRU eviction at 10,000 entries.

#### Method: `_check_user_rate_limit(self, user_id) -> bool` (async)

Sliding window rate limit (30/min). Periodic stale entry cleanup.

#### Method: `handle_message(self, incoming, event_fn=None, stream_fn=None, confirm_fn=None) -> OutgoingMessage` (async)

**Main pipeline**: identity resolution -> rate limit -> sanitize -> slash commands -> session resolve -> session lock -> persist user msg -> lane queue submit -> persist assistant msg -> return OutgoingMessage. Error paths persist errors and record gateway failures.

#### Method: `_record_gateway_failure(self, incoming, session_id, error) -> None` (async)

Records gateway-level failures in OutcomeStore.

#### Method: `_handle_link_command(self, text, incoming) -> OutgoingMessage`

Handles `/link [code]` for cross-channel identity linking.

#### Method: `_resolve_session_id(self, user_id, channel) -> str`

Reuses most recent session or creates new UUID.

#### Method: `get_stats(self) -> dict[str, Any]`

Returns gateway statistics.

---

## 4. `config.py`

**Path**: `src/jarvis/config.py`

### File Overview

Three-layer configuration system. Priority: env vars -> YAML file -> defaults. Produces a fully typed `JARVISConfig` dataclass.

### Module-Level Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `DEFAULT_HOME` | `Path` | `~/.prometheus` | Default home directory |
| `DEFAULT_CONFIG_FILE` | `Path` | `~/.prometheus/config.yaml` | Default config path |
| `DEFAULT_PROFILE` | `str` | `"balanced"` | Default launch profile |
| `DEFAULT_AGENT` | `str` | `"jarvis"` | Default active agent |
| `PROFILE_PRESETS` | `dict` | 3 profiles | balanced, public_beast, enterprise_lockdown |

### Profile Presets

| Profile | Mode | Trust | Docker | Max Tasks | Max Iters | Drift Threshold |
|---------|------|-------|--------|-----------|-----------|-----------------|
| `balanced` | personal | normal | off | 5 | 10 | 0.42 |
| `public_beast` | public | yolo | on | 12 | 30 | 0.60 |
| `enterprise_lockdown` | enterprise | paranoid | on | 3 | 6 | 0.32 |

### Configuration Dataclasses

- **`LLMConfig`**: Provider, model, fallbacks, temperature, max_tokens, reasoning_effort, throttle/retry settings (18 fields)
- **`AgentLLMConfig(LLMConfig)`**: Dedicated agent cognition provider (inherits, all defaults empty)
- **`ChannelConfig`**: Enabled channels, platform tokens, webchat port (5 fields)
- **`SecurityConfig`**: Trust level, permission mode, approval settings, Docker sandbox, tool whitelist/blacklist (8 fields)
- **`DaemonConfig`**: Enabled, cron, webhook, heartbeat, DB socket (6 fields)
- **`LaunchProfileConfig`**: Profile name, approvals, EGM, code network, OpenClaw, marketplace, self-evolution, spawn limits, tool iterations, persona drift settings (13 fields)
- **`OpenClawBridgeConfig`**: URL, paths, model, agent, auth, timeout, retry, polling, skills (14 fields)
- **`MemoryConfig`**: Persistence dir, knowledge graph, vector store, working memory, decay, consolidation (6 fields)
- **`AntigravityConfig`**: Enabled, client_id, URLs, scopes (6 fields)
- **`OperatorsConfig`**: Desktop macro limits, AX tree settings, screenshot, input limits, Android limits (11 fields)
- **`JARVISConfig`**: Master config with all sub-configs plus name, version, mode, agent, and path fields. Properties: `agents_dir`, `agent_dir`, `flame_dir`. `__post_init__` sets computed path defaults.

### Key Functions

- `_deep_merge(base, override)`: Recursive dict merge
- `_parse_bool(raw)`, `_parse_csv(raw)`, `_safe_int(raw)`, `_safe_float(raw)`: Type coercion helpers
- `_validate_port(port)`: TCP port clamping
- `_resolve_profile_name(merged, override)`: Profile name resolution
- `_get_profile_defaults(name)`: Profile preset lookup
- `_sync_runtime_policy_env(cfg)`: Mirror config to env vars
- `_env_overrides()`: Maps 60+ env vars to config keys with type coercion + provider API key resolution (14 providers)
- `_load_yaml_config(path)`: YAML loading with graceful PyYAML fallback
- `_dict_to_config(data)`: Dict to JARVISConfig conversion
- **`load_config(config_path, profile_override)`**: Main entry point. Layers: empty -> YAML -> env -> profile defaults -> JARVISConfig. Creates directories, warns on permissions, syncs env.
- `save_default_config(path, provider, trust_level, model)`: Writes template config.yaml with chmod 600

---

## 5. `prompts.py`

**Path**: `src/jarvis/prompts.py`

### File Overview

System prompt assembly and persona drift detection. Extracted from core.py (Phase 1D).

### Module-Level State

- `_file_content_cache`: mtime-based file content cache

### Persona Drift Regex Constants

| Name | Purpose | Weight |
|------|---------|--------|
| `PERSONA_DRIFT_PATTERNS[0]` | "as an ai model" | 0.30 |
| `PERSONA_DRIFT_PATTERNS[1]` | "i am chatgpt/gpt-4" | 0.45 |
| `PERSONA_DRIFT_PATTERNS[2]` | "i am claude/anthropic" | 0.45 |
| `PERSONA_DRIFT_PATTERNS[3]` | "i am gemini/google ai" | 0.35 |
| `PERSONA_DRIFT_PATTERNS[4]` | "i cannot access/run" | 0.10 |
| `VERIFICATION_REQUEST_RE` | User asks to verify |  |
| `UNVERIFIED_ACTION_CLAIM_RE` | "i have checked/verified" |  |
| `UNVERIFIED_MODEL_SWITCH_RE` | "i have switched model" |  |
| `DIRECT_CAPABILITY_DENIAL_RE` | "i cannot ... shell" |  |
| `PERSONA_IDENTITY_QUERY_RE` | "who are you" |  |
| `DIRECT_TOOL_PREFIX_RE` | "run/execute <tool>" |  |
| `TIMEOUT_HINT_RE` | "timeout=N" |  |

### Dataclass: `PersonaDriftAssessment`

Fields: `score: float`, `threshold: float`, `reasons: list[str]`. Property: `needs_regeneration -> bool`.

### Key Functions

- `_read_file_cached(path)`: mtime-based caching file reader
- `_load_identity_file(filename, config)`: Loads identity files (agent folder -> built-in fallback)
- `_load_self_meta_prompt(config)`: Loads meta prompt (env -> agent folder -> built-in)
- `_normalize_openclaw_skill_slug(value)`: Kebab-case slug generator
- `_parse_openclaw_skill_document(path)`: SKILL.md parser with YAML frontmatter
- `_extract_openclaw_command_samples(body, limit=8)`: Shell command extraction from code blocks
- `_summarize_openclaw_markdown(body, limit_chars=2400)`: Compact SKILL.md preview
- **`_get_system_prompt(config)`**: Assembles full system prompt. Self-evolving path (SOUL_SEED exists): identity engine prompt + optional meta prompt. Legacy path: SOUL.md + USER.md + MEMORY.md. Both append Runtime Context (mode, trust, date, model, paths, behavior rules, browser/desktop patterns, hardware capabilities).
- `_get_antigravity_prompt(config)`: Antigravity superpowers section

---

## 6. `sessions.py`

**Path**: `src/jarvis/sessions.py`

### File Overview

Session persistence as JSONL. Each session is a directory with `meta.json` + `messages.jsonl`.

### Dataclass: `Message`

Fields: `role`, `content`, `timestamp`, `metadata`, `tool_name`, `tool_args`, `tool_result`.
Methods: `to_dict()`, `from_dict()`, `to_llm_format()`.

### Dataclass: `Session`

**Class Constants**: MAX_MESSAGES=500, CONTEXT_KEEP_RECENT=24, CONTEXT_MAX_TOKENS=24000, plus 8 token budget constants for context packing.

Fields: `session_id`, `user_id`, `title`, `channel_origin`, `messages`, `created_at`, `updated_at`, `metadata`, `max_messages`, `_context_cache`.

**Key Methods**:
- `add_message(role, content)`: Adds message, truncates oldest if over cap, auto-titles from first user message
- `_smart_title(text, max_len=80)`: Word-boundary truncation
- `get_llm_messages(max_messages)`: LLM-formatted messages with optional compression
- `estimate_tokens(text)`: Tokenizer-free estimation (char/4 vs words*1.15)
- `trim_content_for_context(content, max_tokens, head_ratio=0.72)`: Head+tail preserving trim
- `_summary_snippet(content, limit=160)`: Summary snippet generator
- `_build_context_summary(dropped, dropped_count, max_tokens)`: Summarizes dropped messages
- **`build_context_window(max_total_tokens, keep_recent, summary_max)`**: Core packing algorithm. Packs recent messages first (rank-based budgets), backfills older while budget allows, prepends summary of dropped messages. Caches results.
- `message_count` (property), `get_context_summary()`

### Class: `SessionStore`

LRU-cached (128 max), thread-safe session persistence.

**Key Methods**:
- `_cache_put(id, session)`: LRU cache insert/refresh
- `create(user_id, title, session_id)`: Creates new session
- `_is_safe_session_id(id)`: Path traversal protection
- `get(id)`: Cache lookup then disk load
- `get_or_create(id, user_id)`: Get or create
- `append_message(id, role, content)`: Append + persist immediately
- `list_sessions(user_id, limit=20)`: List by updated_at desc
- `delete(id)`: Safe delete with path validation
- `_save_meta(session)`: Atomic write (tempfile + fsync + os.replace)
- `_append_message_to_disk(id, msg)`: Validated JSONL append
- `_load_session(id)`: Full disk load with corrupt line tolerance

---

## 7. `cli.py`

**Path**: `src/jarvis/cli.py`

### File Overview

Terminal interface. Uses Rich + prompt_toolkit. 11 subcommands: chat, start, stop, restart, install, logs, setup, auth, status, doctor, eval.

### Classes

- **`GenerationController`**: Esc-to-stop lifecycle. Fields: `_cancel` (asyncio.Event), `_partial_response`, `_is_generating`. Methods: `reset()`, `cancel()`, `cancelled` (property), `accumulate(token)`.
- **`JARVISCompleter(Completer)`**: Context-aware completions for `/` commands and `/model` models.
- **`StatusBar`**: Bottom toolbar showing provider/model, trust, tokens, turns, session. Callable returning HTML.
- **`LiveChatRenderer`**: Real-time UI with thinking spinners (braille animation, daemon thread), tool call visibility (icons, args preview, result preview, duration), response rendering (Markdown via Rich, stats footer). Event types: thinking, tool_start, tool_end, response.

### Key Functions

- `_listen_for_escape(gen_ctrl)`: Async Esc/Ctrl+C listener using raw fd reads
- `_interactive_select(title, options, current_value, max_visible=12)`: Arrow-key selector with scrolling and type-to-filter
- `_animated_banner()`: Gradient ASCII art reveal
- `_autoload_env_for_cli(config_path)`: Loads .env from 5 candidate paths
- **`_run_chat(session_id, config_path, profile)`**: Full interactive chat session with prompt_toolkit, streaming, Esc-to-stop, slash commands, tool confirmation
- `_build_model_options()`: Available (provider, model) pairs
- **`_handle_command(cmd, ...)`**: 14 slash commands: /exit, /status, /tools, /sessions, /clear, /auth, /model (interactive selector), /retry, /copy (pbcopy), /cost, /new, /compact, /history, /help. Bare `/` opens picker.
- `_run_setup()`: Onboarding wizard: auto-detect providers, interactive selection, API key input, trust level, save config, test connection
- `_show_status()`: Comprehensive status display
- `_run_jarvis_auth()`: Dedicated Claude account setup (isolated config dir)
- `_run_auth()`: Antigravity OAuth/token auth
- **`main()`**: argparse entry point dispatching to all subcommands
- `_run_doctor()`: System health diagnostics (Python, Docker, Playwright, config, LLM providers, channels, memory, plugins)
- `_run_start(args)`: Foreground or daemon mode (detached subprocess)
- `_run_stop()`, `_run_restart()`, `_run_install()`, `_run_logs()`, `_run_eval()`

---

## 8. `errors.py`

**Path**: `src/jarvis/errors.py`

### File Overview

Structured error hierarchy replacing string-wrapped errors.

### `JARVISError(Exception)` -- Base

Attributes: `error_code`, `recoverable`, `context`, `details`. `__str__` prepends error_code.

### Tool Errors

- `ToolError(JARVISError)` -- base, adds `tool_name`
- `ToolNotFoundError(ToolError)` -- tool not registered
- `ToolTimeoutError(ToolError)` -- timeout, adds `timeout_seconds`
- `ToolPermissionError(ToolError)` -- no permission
- `ToolValidationError(ToolError)` -- schema validation failed

### LLM Errors

- `LLMError(JARVISError)` -- base, adds `provider`, `model`
- `LLMRateLimitError(LLMError)` -- 429, adds `retry_after`
- `LLMConnectionError(LLMError)` -- connection failed
- `LLMAllProvidersFailedError(LLMError)` -- all failover failed, adds `provider_errors`

### Memory Errors

- `MemoryStoreError(JARVISError)` -- base
- `MemoryCapacityError` -- capacity limit
- `MemoryNotFoundError` -- entry not found

### Session Errors

- `SessionError(JARVISError)` -- base
- `SessionNotFoundError` -- ID not found
- `SessionCorruptedError` -- data corrupted

### Security Errors

- `SecurityError(JARVISError)` -- base
- `SSRFBlockedError` -- SSRF blocked
- `CommandInjectionError` -- injection detected
- `PromptInjectionError` -- prompt injection
- `RateLimitExceededError` -- rate limit exceeded

### Desktop Errors

- `DesktopError(JARVISError)` -- base
- `DeviceNotConnectedError` -- no device
- `AccessibilityPermissionError` -- missing permissions

---

## 9. Architecture & Data Flow

### File Interconnections

```
cli.py --> config.py (load_config)
       --> core.py (JARVISCore)
       --> gateway.py (Gateway, IncomingMessage)

gateway.py --> sessions.py (SessionStore, Session)
           --> config.py (JARVISConfig)
           --> core.py (via process_fn callback)

core.py --> config.py (JARVISConfig)
        --> prompts.py (_get_system_prompt, PersonaDriftAssessment)
        --> sessions.py (Session.build_context_window)
        --> errors.py (exception types)
        --> tools/, llm_providers/, memory/, agents/, identity/, services/

prompts.py --> config.py, identity/, tools/registry
```

### Request Flow (Happy Path)

1. User types in CLI
2. cli.py creates IncomingMessage, submits to gateway
3. gateway.py: identity resolve -> rate limit -> sanitize -> session resolve -> lock -> persist -> lane queue
4. core.py: transcript -> feedback -> tool shortcut check -> system prompt build -> memory injection -> context window -> agent loop (LLM -> tools -> repeat) -> drift guard -> meta-cognition eval -> outcome record -> memory store -> identity reflect
5. gateway.py: persist assistant message, return OutgoingMessage
6. cli.py: display via LiveChatRenderer

### Concurrency Model

- async/await throughout (core, gateway, cli)
- LaneQueue for serial per-session execution
- Per-session asyncio.Lock (LRU bounded at 10,000)
- Parallel tool execution via asyncio.gather()
- Background tasks for memory consolidation
- threading.Lock in SessionStore (sync+async paths)
- Concurrent Esc-listener task alongside generation

### Design Patterns

- Layered Configuration (defaults -> YAML -> env)
- Strategy (LLMInterface provider routing, ChannelAdapter)
- Observer/Event (event_fn callbacks)
- LRU Caching (identity, session locks, session store, file content)
- Pipeline (Gateway -> LaneQueue -> Core -> LLM -> Tools)
- Atomic Writes (tempfile + fsync + os.replace)
- Circuit Breaker (LLM failover with retry/backoff)
- Guard Pattern (persona drift guard with regeneration)