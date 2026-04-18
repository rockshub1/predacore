# PredaCore Vendor Layer Documentation

Comprehensive reference for all vendored dependencies under `src/predacore/_vendor/`. These modules originate from the earlier PredaCore microservice architecture and are vendored (copied into the PredaCore source tree) rather than pip-installed so that PredaCore can operate as a single-process monolith without requiring gRPC inter-service calls, Docker orchestration, or network-accessible microservices. Import paths are rewritten from `src.common.*` to `predacore._vendor.common.*` etc.

---

## Table of Contents

1. [Vendor Root](#1-vendor-root)
2. [Common Utilities (`common/`)](#2-common-utilities)
   - [embedding.py](#21-embeddingpy)
   - [llm.py](#22-llmpy)
   - [memory_service.py](#23-memory_servicepy)
   - [vector_store.py](#24-vector_storepy)
   - [models.py](#25-modelspy)
   - [metrics.py](#26-metricspy)
   - [errors.py](#27-errorspy)
   - [logging_config.py](#28-logging_configpy)
   - [logging_utils.py](#29-logging_utilspy)
   - [notifications.py](#210-notificationspy)
   - [structured_output.py](#211-structured_outputpy)
   - [skill_genome.py](#212-skill_genomepy)
   - [skill_scanner.py](#213-skill_scannerpy)
   - [skill_collective.py](#214-skill_collectivepy)
   - [skill_evolution.py](#215-skill_evolutionpy)
   - [Protobuf Definitions (`protos/`)](#216-protobuf-definitions)
3. [Core Strategic Engine (`core_strategic_engine/`)](#3-core-strategic-engine)
   - [planner.py](#31-plannerpy)
   - [llm_planner.py](#32-llm_plannerpy)
   - [planner_mcts.py](#33-planner_mctspy)
   - [planner_enhancements.py](#34-planner_enhancementspy)
   - [plan_cache.py](#35-plan_cachepy)
4. [Ethical Governance Module (`ethical_governance_module/`)](#4-ethical-governance-module)
   - [service.py](#41-servicepy)
   - [rule_engine.py](#42-rule_enginepy)
   - [audit_logger.py](#43-audit_loggerpy)
   - [persistent_audit.py](#44-persistent_auditpy)
5. [Knowledge Nexus (`knowledge_nexus/`)](#5-knowledge-nexus)
   - [service.py](#51-servicepy-1)
   - [storage.py](#52-storagepy)
   - [vector_index.py](#53-vector_indexpy)
   - [faiss_vector_index.py](#54-faiss_vector_indexpy)
   - [health.py](#55-healthpy)
6. [User Modeling Engine (`user_modeling_engine/`)](#6-user-modeling-engine)
   - [service.py](#61-servicepy-2)
7. [Integration Map](#7-integration-map)
8. [Algorithm Deep-Dives](#8-algorithm-deep-dives)

---

## 1. Vendor Root

**File:** `_vendor/__init__.py`

Empty package initializer. Makes `predacore._vendor` importable as a Python package. Contains no logic.

**Why vendored:** The original PredaCore platform ran as 5+ microservices communicating over gRPC. PredaCore collapsed these into a single process; the vendored code preserves the data models, proto stubs, and service logic without requiring live gRPC servers.

---

## 2. Common Utilities

**Path:** `_vendor/common/`

Shared foundation used by every other vendor subsystem and by PredaCore core. Provides embedding clients, LLM provider abstraction, memory persistence, metrics, error hierarchy, logging, notification delivery, data models, vector search, and the Flame skill network primitives.

**File:** `common/__init__.py` -- Single-line package marker: `# This file makes 'src/common' a Python package.`

---

### 2.1 embedding.py

**Purpose:** Provider-agnostic embedding client abstraction with an in-memory vector index. Supports five embedding providers plus a resilient wrapper with automatic fallback.

**Imports:** `logging`, `math`, `re`, `threading`, `typing`, `httpx` (lazy).

#### Class: `EmbeddingClient`

Abstract base class for all embedding providers.

| Method | Signature | Description |
|--------|-----------|-------------|
| `embed` | `async def embed(self, texts: list[str]) -> list[list[float]]` | Convert texts to embedding vectors. Must be overridden. |

#### Class: `OpenAIEmbeddingClient(EmbeddingClient)`

Calls the OpenAI embeddings API (or any compatible endpoint).

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | OpenAI API key |
| `model` | `str` | `"text-embedding-3-small"` | Model identifier |
| `base_url` | `str \| None` | `None` | API base URL (defaults to `https://api.openai.com`) |

**`embed` method:** POSTs to `/v1/embeddings` with `httpx.AsyncClient(timeout=60)`. Extracts `item["embedding"]` from each element in the response `data` array.

#### Class: `GeminiEmbeddingClient(EmbeddingClient)`

Google Gemini embedding client using the Generative Language API.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | Google API key |
| `model` | `str` | `"models/gemini-embedding-001"` | Model identifier (auto-prefixed with `models/` if missing) |
| `base_url` | `str \| None` | `None` | API base URL |

**`embed` method:** Iterates through texts one at a time (no batch endpoint), sending each to the `:embedContent` endpoint. Uses a nested `_extract_embedding()` helper that handles multiple response formats (direct `values`, nested `embedding.values`, or `embedding` as list).

#### Class: `ZhiPuEmbeddingClient(EmbeddingClient)`

ZhiPu BigModel embedding client (OpenAI-compatible API format).

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | ZhiPu API key |
| `model` | `str` | `"embedding-3"` | Model identifier |
| `base_url` | `str` | `"https://open.bigmodel.cn/api/paas/v4"` | API base URL |

**`embed` method:** Identical payload format to OpenAI.

#### Class: `LocalEmbeddingClient(EmbeddingClient)`

Local semantic embedding using HuggingFace `sentence-transformers` models. Runs entirely offline.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"thenlper/gte-small"` | HuggingFace model identifier |

**Key design details:**
- Uses a class-level `_model_cache` dict and `_model_lock` (threading.Lock) for thread-safe singleton loading.
- `_load()` performs double-checked locking: acquires the lock, checks cache again, then loads via `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`, putting model in eval mode.
- `embed()` delegates to `asyncio.to_thread(self._embed_sync, texts)`.
- `_embed_sync()` tokenizes with `padding=True, truncation=True, max_length=256`, runs the model under `torch.no_grad()`, performs mean pooling with attention mask expansion, and L2-normalizes the result.

**Model choice rationale:** GTE-small produces 384-dimensional vectors, scores ~61.4 on MTEB (vs MiniLM's ~56.3), and is ~67 MB. Preferred when running Claude/Anthropic as LLM provider since no API key is needed.

#### Class: `HashingEmbeddingClient(EmbeddingClient)`

Deterministic hash-based embedding for fully offline fallback.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | `256` | Output vector dimensionality |

**Algorithm:** Iterates over UTF-8 bytes of the input string. For each byte at position `i`, adds the byte value to `v[i % dim]` modulo 1000. L2-normalizes the result. Not semantic -- purely structural.

#### Class: `ResilientEmbeddingClient(EmbeddingClient)`

Wrapper that tries a primary provider and falls back to hash embeddings on failure.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary` | `EmbeddingClient` | required | Primary provider |
| `fallback` | `EmbeddingClient \| None` | `HashingEmbeddingClient()` | Fallback provider |

**`_sanitize_error` (static):** Strips API keys and Bearer tokens from error messages before logging.

#### Function: `get_default_embedding_client() -> EmbeddingClient`

Factory function that selects the best embedding client based on environment variables.

**Environment variables consulted:**
- `EMBED_PROVIDER`: `"auto"` (default), `"openai"`, `"gemini"`, `"google"`, `"zhipu"`, `"local"`
- `EMBED_MODEL`: model override
- `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `ZHIPU_API_KEY`
- `LLM_PROVIDER`: used to infer preferred embedding family
- `OPENAI_BASE_URL`, `GEMINI_BASE_URL`

**Selection logic in `auto` mode:**
1. If LLM provider is Claude/Anthropic and torch is available, use `LocalEmbeddingClient` (free, offline)
2. If Gemini is the LLM and Gemini key exists, use Gemini embeddings
3. Fall through: OpenAI key -> Gemini key -> ZhiPu key -> local/hash fallback

All API-backed clients are wrapped in `ResilientEmbeddingClient` with `_best_fallback()` as the fallback (local if torch available, otherwise hash).

#### Class: `InMemoryVectorIndex`

Numpy-optimized in-memory vector store with cosine similarity search.

**Instance variables:**
- `_ids: list[str]` -- item identifiers
- `_metas: list[dict]` -- associated metadata
- `_vectors: numpy.ndarray | None` -- matrix of normalized vectors
- `_dim: int | None` -- dimensionality (set on first add)

**Method: `add(item_id, vector, meta=None)`**

L2-normalizes the vector, appends to `_ids` and `_metas`, and vstacks onto the numpy matrix. Falls back to list storage if numpy is unavailable.

**Method: `search(query_vec, top_k=5) -> list[tuple[str, float, dict]]`**

1. Normalizes the query vector
2. Computes cosine similarity via matrix-vector dot product (`self._vectors @ query`)
3. If `n <= top_k`, sorts all; otherwise uses `np.argpartition` for O(n) partial sort
4. Returns `(item_id, score, metadata)` tuples

**Pure-Python fallback:** `_search_fallback` computes cosine similarity element-wise for each stored item.

---

### 2.2 llm.py

**Purpose:** Provider-agnostic LLM client abstraction supporting 7 providers: OpenAI, OpenRouter, Gemini API, Gemini CLI, NVIDIA NIM, Ollama (local), and a factory function for automatic selection.

**Environment variables:**
- `LLM_PROVIDER`: `"openai"` | `"openrouter"` | `"gemini"` | `"gemini-cli"` (default) | `"nvidia"` | `"ollama"`
- `LLM_MODEL`, `LLM_BASE_URL`, `LLM_REASONING` (`"minimal"` | `"low"` | `"medium"` | `"high"`)
- Provider-specific keys: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`
- Ollama: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_KEEP_ALIVE`
- Gemini CLI: `GEMINI_CLI_TIMEOUT_SECONDS` (default 45), `LLM_FALLBACK_MODELS`

#### Class: `LLMClient` (Base)

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `async def generate(self, messages, *, model=None, params=None) -> str` | Abstract generation method |

#### Class: `GeminiCLIClient(LLMClient)`

Calls the `gemini` CLI binary via subprocess. The most unique provider -- no API key needed (uses `gemini auth login` cached credentials).

**Key behaviors:**
- Builds a plain-text prompt from messages (System/Assistant/User prefixes)
- Tries a cascade of models via `_candidate_models()` (primary + `LLM_FALLBACK_MODELS`)
- `_call_gemini_cli()` runs in a thread executor with process group management and graceful SIGTERM -> SIGKILL timeout handling
- `_should_try_fallback()` checks error strings for rate limits, quota exhaustion, model-not-found, overloaded, etc.
- Filters out "Loaded cached credentials" lines from stdout

#### Class: `OpenAIClient(LLMClient)`

Standard OpenAI Chat Completions API client. Maps `reasoning_effort` param to `extra_body` field.

#### Class: `OpenRouterClient(LLMClient)`

OpenRouter-compatible client posting to `/api/v1/chat/completions`.

#### Class: `GeminiClient(LLMClient)`

Google Gemini REST API client with dual auth: API key or OAuth (via `google-auth` Application Default Credentials or cached `~/.gemini/credentials.json`). Converts OpenAI-style messages to Gemini format (`role: "model"` for assistant, `systemInstruction` for system messages).

#### Class: `NvidiaClient(LLMClient)`

NVIDIA NIM API client (OpenAI-compatible). Strips `<think>...</think>` blocks from reasoning model outputs.

#### Class: `OllamaClient(LLMClient)`

Local Ollama inference client. Additional methods: `list_models()` (GET `/api/tags`), `health_check()` (GET root).

#### Function: `get_default_llm_client(logger=None) -> LLMClient`

Factory that reads `LLM_PROVIDER` and returns the appropriate client. Default is `gemini-cli`.

#### Function: `default_params(temperature=0.2, max_tokens=1024) -> dict`

Returns a parameter dict including optional `reasoning_effort` from `LLM_REASONING` env var.

---

### 2.3 memory_service.py

**Purpose:** Persistent memory system with SQLite (WAL mode) + semantic recall via embeddings. Replaces earlier JSON file storage.

**Features:** 7 memory types, 4 importance levels, semantic search with keyword fallback, per-user scoping, auto-migration from JSON, expiring memories, in-memory caching.

#### Enum: `MemoryType(str, Enum)`

Values: `FACT`, `CONVERSATION`, `TASK`, `PREFERENCE`, `CONTEXT`, `SKILL`, `ENTITY`

#### Enum: `ImportanceLevel(int, Enum)`

Values: `LOW=1`, `MEDIUM=2`, `HIGH=3`, `CRITICAL=4`

#### Dataclass: `Memory`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | `uuid4()` | Unique identifier |
| `content` | `str` | `""` | Memory text content |
| `memory_type` | `MemoryType` | `FACT` | Classification |
| `importance` | `ImportanceLevel` | `MEDIUM` | Priority level |
| `tags` | `list[str]` | `[]` | Searchable tags |
| `metadata` | `dict` | `{}` | Arbitrary key-value data |
| `user_id` | `str` | `""` | Owning user |
| `created_at` | `datetime` | now UTC | Creation timestamp |
| `last_accessed` | `datetime` | now UTC | Last retrieval timestamp |
| `access_count` | `int` | `0` | Number of retrievals |
| `embedding` | `list[float] \| None` | `None` | Semantic vector |
| `source` | `str` | `""` | Origin identifier |
| `expires_at` | `datetime \| None` | `None` | Auto-cleanup time |

Methods: `to_dict()`, `from_dict()` (class method).

#### SQLite Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT DEFAULT 'fact',
    importance INTEGER DEFAULT 2,
    tags TEXT DEFAULT '[]',        -- JSON array
    metadata TEXT DEFAULT '{}',     -- JSON object
    embedding BLOB,                 -- float32 bytes
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    expires_at TEXT
);
```

Indexes on: `user_id`, `(user_id, memory_type)`, `(user_id, importance)`, `(user_id, created_at DESC)`, `expires_at WHERE NOT NULL`.

#### Class: `MemoryService`

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str \| None` | `"data/memory"` | Base directory for DB |
| `embedding_client` | `Any \| None` | `None` | EmbeddingClient instance |
| `write_batch_size` | `int` | `100` | Batch size hint |
| `db_adapter` | `Any` | `None` | Optional remote DB adapter |

**Key methods:**

**`store(content, user_id, memory_type, importance, tags, metadata, source, expires_at) -> Memory`**
Stores a memory immediately to SQLite. Generates embedding via `embedding_client.embed()` if available. Uses application-level `_write_lock` for thread safety. Serializes embeddings to float32 bytes via `_encode_embedding()`.

**`recall(query, user_id, memory_types, tags, top_k, min_importance) -> list[tuple[Memory, float]]`**
1. Loads user memories into cache if not already loaded (`_ensure_user_loaded`)
2. Tries semantic search: embeds query, computes cosine similarity against all user vectors (skipping dimension mismatches from provider migrations)
3. Falls back to keyword matching if no semantic results
4. Updates access stats (last_accessed, access_count) in batch
5. Returns `(memory, score)` tuples sorted by relevance

**`delete(memory_id, user_id) -> bool`** -- Deletes from SQLite and cache.

**`list_by_type(user_id, memory_type, limit)` / `list_by_tag(user_id, tag, limit)`** -- Direct SQL queries.

**`summarize_context(user_id, context_query, max_memories) -> str`** -- Recalls relevant memories and formats them as `[type] content` lines for prompt injection.

**`_migrate_from_json()`** -- Auto-discovers legacy `data/memory/{user_id}/memories.json` files, migrates to SQLite, renames originals to `.json.migrated`.

**Embedding serialization:** `_encode_embedding` packs `list[float]` to `numpy.float32.tobytes()` (or struct.pack fallback). `_decode_embedding` reverses.

**`_cosine_similarity(a, b)`** -- Numpy-accelerated dot product / norms, with pure-Python fallback.

---

### 2.4 vector_store.py

**Purpose:** Disk-backed vector index with JSONL persistence and optional FAISS support.

#### Class: `DiskBackedVectorIndex`

Wraps `InMemoryVectorIndex` from `embedding.py` with JSONL file persistence.

**Constructor:** Takes `base_dir` (default `"data/vector_index"`).

**Storage format:** `{namespace}.jsonl` where each line is `{"id": str, "vector": [float], "meta": {}}`.

| Method | Description |
|--------|-------------|
| `add(namespace, item_id, vector, meta)` | Adds to in-memory index and appends to disk JSONL |
| `search(namespace, query_vec, top_k)` | Delegates to in-memory index |
| `_ensure_ns(namespace)` | Lazy-loads namespace from disk on first access |

#### Class: `FaissVectorIndex`

Optional FAISS-backed index. Requires `faiss` package.

**Storage:** `{namespace}.faiss` (binary index) + `{namespace}.meta.jsonl` (ID/metadata pairs).

**Key details:**
- Uses `faiss.IndexFlatIP` (inner product) with L2-normalized vectors for cosine similarity
- Persists FAISS index to disk after every add via `faiss.write_index`
- Dimension is inferred from the first vector added

---

### 2.5 models.py

**Purpose:** Pydantic data models shared across all PredaCore components. Defines the domain types for Knowledge Nexus nodes/edges, CSC plans/steps, WIL tool descriptions/interactions, and DAF agent/task assignments.

#### Knowledge Nexus Models

**`KnowledgeNode(BaseModel)`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | `uuid4()` | Node identifier |
| `labels` | `set[str]` | `set()` | Type/category labels |
| `properties` | `dict[str, Any]` | `{}` | Key-value properties |
| `embedding` | `Any \| None` | `None` | Vector embedding |
| `layers` | `set[str]` | `set()` | Layer memberships |

Config: `arbitrary_types_allowed = True` (for numpy arrays).

**`KnowledgeEdge(BaseModel)`**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | `uuid4()` | Edge identifier |
| `source_node_id` | `UUID` | required | Start node |
| `target_node_id` | `UUID` | required | End node |
| `type` | `str` | required | Relationship type (e.g., `"HAS_CAPABILITY"`) |
| `properties` | `dict[str, Any]` | `{}` | Edge properties |
| `layers` | `set[str]` | `set()` | Layer memberships |

#### CSC Models

**`StatusEnum(str, Enum)`**: `UNSPECIFIED`, `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`, `CANCELLED`, `READY`, `EXECUTING`

**`PlanStep(BaseModel)`**: `id`, `description`, `action_type` (e.g., `"QUERY_KN"`, `"INVOKE_TOOL"`), `parameters`, `required_knowledge_ids`, `status`, `result`

**`Plan(BaseModel)`**: `id`, `goal_id`, `steps: list[PlanStep]`, `status`, `confidence: float | None`, `justification: str | None`

#### WIL Models

**`ToolTypeEnum`**: `UNSPECIFIED`, `API`, `WEB_SCRAPER`, `CODE_EXECUTOR`

**`InteractionStatusEnum`**: `UNSPECIFIED`, `SUCCESS`, `FAILED`, `ERROR`, `TIMEOUT`

**`ToolDescription`**: `tool_id`, `tool_type`, `description`, `input_schema`, `output_schema`, `metadata`

**`InteractionRequest`**: `request_id`, `tool_id`, `parameters`, `credentials`, `context`

**`InteractionResult`**: `request_id`, `status`, `output`, `error_message`, `metadata`

**`CodeExecutionRequest`**: `request_id`, `code`, `input_args`, `timeout_seconds=60`, `context`

**`CodeExecutionResult`**: `request_id`, `status`, `stdout`, `stderr`, `result`, `error_message`

#### DAF Models

**`AgentDescription`**: `agent_type_id`, `description`, `supported_actions`, `required_tools`, `configuration`

**`TaskAssignment`**: `task_id`, `plan_step_id`, `required_capability`, `agent_type_id`, `parameters`, `context`

**`TaskResult`**: `task_id`, `plan_step_id`, `status`, `output`, `error_message`, `agent_type_id_used`

---

### 2.6 metrics.py

**Purpose:** Prometheus-client based observability for the entire system. Provides standardized counters, histograms, gauges, and trace ID propagation.

**Trace ID system:**
- `ContextVar[str]` named `_trace_id` with default `""`
- `new_trace_id()`: generates `uuid4().hex[:16]`, sets it in the context var
- `get_trace_id()`: returns current or auto-generates new
- `set_trace_id(tid)`: sets from incoming request

**Defined metrics:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `REQUESTS_TOTAL` | Counter | channel, status | Total requests processed |
| `MESSAGES_RECEIVED` | Counter | channel | Messages from users |
| `MESSAGES_SENT` | Counter | channel | Messages to users |
| `TOOL_CALLS` | Counter | tool, status | Tool executions |
| `TOOL_LATENCY` | Histogram | tool | Tool latency (buckets: 0.01-60s) |
| `TOOL_CONFIRMATIONS` | Counter | tool, decision | Confirmation prompts |
| `LLM_REQUESTS` | Counter | provider, status | LLM API calls |
| `LLM_LATENCY` | Histogram | provider | LLM latency (buckets: 0.1-120s) |
| `LLM_TOKENS` | Counter | provider, direction | Token consumption |
| `LLM_FAILOVERS` | Counter | from_provider, to_provider | Provider failovers |
| `MEMORY_COUNT` | Gauge | user, type | Total memories |
| `MEMORY_OPS` | Counter | operation, status | Memory operations |
| `ACTIVE_SESSIONS` | Gauge | -- | Active sessions |
| `SESSION_DURATION` | Histogram | -- | Session length (buckets: 60s-4h) |
| `CHANNEL_STATUS` | Gauge | channel | Connection status (1/0) |
| `CHANNEL_ERRORS` | Counter | channel, error_type | Channel errors |
| `DAEMON_UPTIME` | Gauge | -- | Daemon uptime seconds |
| `CRON_JOBS_EXECUTED` | Counter | job_name, status | Cron executions |
| `EGM_EVALUATIONS` | Counter | decision | EGM rule evaluations |

**Helper: `track_latency(histogram, labels=None, **label_kwargs)`** -- Context manager that measures `time.perf_counter()` elapsed and observes on the histogram.

**Helper: `get_metrics_text() -> str`** / `get_metrics_content_type() -> str`** -- For `/metrics` endpoint.

---

### 2.7 errors.py

**Purpose:** Structured error hierarchy where every exception carries `error_code`, `context` dict, and `recoverable` flag.

**Root: `PrometheusError(Exception)`**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `error_code` | `str` | `"PREDACORE_ERROR"` | Machine-readable identifier |
| `context` | `dict` | `{}` | Debug info dict |
| `recoverable` | `bool` | `False` | Retry hint |

**`__str__`** renders as `[ERROR_CODE] message (key=value, ...)`.

**Error hierarchy:**

```
PrometheusError
  +-- ToolExecutionError (TOOL_EXEC_FAILED, recoverable=True)
  |     +-- ToolTimeoutError (TOOL_TIMEOUT, recoverable=True)
  +-- ToolNotFoundError (TOOL_NOT_FOUND, recoverable=False)
  +-- ToolPermissionError (TOOL_PERMISSION_DENIED, recoverable=False)
  +-- LLMProviderError (LLM_PROVIDER_ERROR, recoverable=True)
  |     +-- LLMRateLimitError (LLM_RATE_LIMIT, recoverable=True)
  |     +-- LLMContextLengthError (LLM_CONTEXT_TOO_LONG, recoverable=False)
  |     +-- LLMAllProvidersFailedError (LLM_ALL_PROVIDERS_FAILED, recoverable=False)
  +-- AuthenticationError (AUTH_FAILED, recoverable=False)
  +-- AuthorizationError (AUTH_FORBIDDEN, recoverable=False)
  +-- SandboxError (SANDBOX_ERROR, recoverable=True)
  |     +-- SandboxNotAvailableError (SANDBOX_NOT_AVAILABLE, recoverable=False)
  |     +-- SandboxTimeoutError (SANDBOX_TIMEOUT, recoverable=True)
  +-- MemoryServiceError (MEMORY_ERROR, recoverable=True)
  |     +-- MemoryNotFoundError (MEMORY_NOT_FOUND, recoverable=False)
  +-- PersistenceError (PERSISTENCE_ERROR, recoverable=True)
  +-- ChannelError (CHANNEL_ERROR, recoverable=True)
  |     +-- ChannelConnectionError (CHANNEL_CONNECT_FAILED, recoverable=True)
  |     +-- ChannelRateLimitError (CHANNEL_RATE_LIMIT, recoverable=True)
  |     +-- ChannelMessageTooLongError (CHANNEL_MSG_TOO_LONG, recoverable=False)
  +-- PlanningError (PLANNING_ERROR, recoverable=True)
  +-- AgentLifecycleError (AGENT_LIFECYCLE_ERROR, recoverable=True)
  +-- EGMViolationError (EGM_VIOLATION, recoverable=False)
  +-- ConfigurationError (CONFIG_ERROR, recoverable=False)
```

---

### 2.8 logging_config.py

**Purpose:** Structured logging system with JSON (production) and pretty-printed (development) formatters. Every log line carries a `trace_id` from `metrics.py`.

#### Class: `JSONFormatter(logging.Formatter)`

Emits single-line JSON: `{"ts": ISO8601, "level": "INFO", "logger": "predacore.core", "msg": "...", "trace_id": "...", ...}`. Adds `file:line` for WARNING+. Includes exception info. Passes through all non-standard LogRecord attributes as extra fields.

#### Class: `PrettyFormatter(logging.Formatter)`

Colored terminal output: `02:30:00 > INFO predacore.core Processing message [trace:abc12345]`. Uses ANSI color codes: cyan/green/yellow/red/magenta for DEBUG/INFO/WARNING/ERROR/CRITICAL.

#### Function: `setup_logging(level="INFO", json_mode=False, log_file=None)`

Configures root logger. If `log_file` is provided, uses `RotatingFileHandler` (50 MB, 3 backups) with JSON format. Otherwise uses console handler with format determined by `json_mode`. Suppresses noisy loggers: httpx, httpcore, urllib3, asyncio, grpc.

#### Function: `get_logger(name) -> logging.Logger`

Simple `logging.getLogger(name)` wrapper.

---

### 2.9 logging_utils.py

**Purpose:** Lightweight helpers for trace ID extraction and structured log formatting.

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_trace_id` | `(context_struct, fallback=None) -> str \| None` | Safely extracts `trace_id` from a protobuf `Struct` field |
| `kv_message` | `(msg, **kv) -> str` | Renders `msg \| key=value key=value` for compact logs |
| `log_json` | `(logger, level, event, **fields)` | Emits `json.dumps({"event": ..., ...})` at the given level |

---

### 2.10 notifications.py

**Purpose:** Multi-channel notification delivery supporting Slack, Discord, Email, and generic Webhooks.

#### Enum: `NotificationChannel`

Values: `EMAIL`, `SLACK`, `DISCORD`, `PUSH`, `WEBHOOK`

#### Enum: `NotificationPriority`

Values: `LOW`, `NORMAL`, `HIGH`, `URGENT`

#### Dataclass: `Notification`

Fields: `id`, `channel`, `recipient`, `title`, `message`, `priority`, `metadata`, `user_id`, `created_at`, `sent_at`, `status`, `error`

#### Providers

**`SlackProvider`**: Posts to a Slack webhook URL with Block Kit format. Maps priority to emoji.

**`DiscordProvider`**: Posts to a Discord webhook URL with embedded format. Maps priority to color.

**`WebhookProvider`**: Generic HTTP POST. Includes SSRF prevention via `_is_safe_url()` which blocks private/internal IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x).

**`EmailProvider`**: SMTP-based email with HTML formatting. Runs SMTP synchronously in a thread executor. Supports STARTTLS + login.

#### Class: `NotificationService`

Main orchestrator. Auto-initializes providers from environment variables (`SLACK_WEBHOOK_URL`, `DISCORD_WEBHOOK_URL`, `SMTP_HOST`). Maintains an in-memory history (max 1000 entries).

Key methods: `send(notification)`, `send_multi(channels, title, message, ...)`, `get_history(user_id, status, limit)`.

---

### 2.11 structured_output.py

**Purpose:** Validates LLM tool outputs against JSON schemas and attaches version metadata.

#### Function: `validate_and_version(data, schema=None, *, version="v1") -> dict`

1. If `schema` is provided, validates `data` against it using `jsonschema.validate`
2. Wraps result in a dict with `output_version` key
3. On validation failure, returns `{"error": "schema_validation_failed: ...", "raw": data, "output_version": version}`

---

### 2.12 skill_genome.py

**Purpose:** Defines the `SkillGenome` -- the unit of sharing in the Flame skill network. Every skill carries a genome: a complete description of its capabilities, tools needed, creator origin, and trust level.

#### Enum: `CapabilityTier(IntEnum)`

| Value | Name | Description |
|-------|------|-------------|
| 0 | `PURE_LOGIC` | No tool access, just data transforms |
| 1 | `LOCAL_READ` | Reads files, git, memory |
| 2 | `LOCAL_WRITE` | Writes files, configs |
| 3 | `NETWORK_READ` | Web search, API GET |
| 4 | `NETWORK_WRITE` | Email, POST, push notifications |

#### Constant: `TOOL_TIER_MAP`

Maps 26 tool names to their `CapabilityTier`. Examples: `read_file` -> LOCAL_READ, `web_search` -> NETWORK_READ, `multi_agent` -> NETWORK_WRITE.

#### Constant: `TIER_PROPAGATION`

Per-tier propagation rules:

| Tier | Min Successes | User Endorsement | Receiver Approval | Auto-Propagate |
|------|--------------|-------------------|-------------------|----------------|
| PURE_LOGIC | 5 | No | No | Yes |
| LOCAL_READ | 10 | Yes | No | Yes |
| LOCAL_WRITE | 20 | Yes | No | No (semi-auto) |
| NETWORK_READ | 30 | Yes | Yes | No |
| NETWORK_WRITE | 50 | Yes | Yes | No |

#### Constant: `SENSITIVE_PATHS`

Frozen set of 12 path patterns skills must never access: `.env`, `credentials.json`, `.ssh/`, `.gnupg/`, `.aws/`, `id_rsa`, `id_ed25519`, etc.

#### Enum: `TrustLevel(str, Enum)`

Values: `QUARANTINED`, `UNTRUSTED`, `SANDBOXED`, `LIMITED`, `TRUSTED`, `ENDORSED`

#### Dataclass: `TrustScore`

Tracks local and network success/failure counts, quarantine reports, endorsement status.

**Properties:**
- `local_success_rate`: successes / total
- `network_score`: 0-100 reputation. Formula: `(successes/total)*100 - quarantines*15`
- `should_quarantine`: True if network_score < 30 or (local_failures >= 5 and success_rate < 0.3)

#### Dataclass: `SkillStep`

Fields: `tool_name`, `parameters: dict`, `condition: str | None`, `use_previous: bool`

#### Dataclass: `SkillGenome`

The central data structure of the Flame. Fields include identity (id, name, description, version), origin (creator_instance_id, creator_user, created_at), the recipe (`steps: list[SkillStep]`), capability manifest (`declared_tools`, `capability_tier`), security (`signature`, `trust: TrustScore`), and metadata (tags, source_pattern, invocation_count).

**Key methods:**
- `compute_tier()`: Scans declared_tools via TOOL_TIER_MAP, takes the highest tier
- `sign(secret=None)`: HMAC-SHA256 over deterministic JSON of id, name, version, creator, steps, declared_tools, tier. Uses `PREDACORE_SKILL_SIGNING_SECRET` env var or machine-local derived key
- `verify_signature(secret=None)`: Recomputes and compares with `hmac.compare_digest`
- `can_propagate()`: Checks propagation requirements for the tier (min successes, endorsement, quarantine status, success rate >= 80%)
- `to_dict()` / `from_dict()`: Full serialization

---

### 2.13 skill_scanner.py

**Purpose:** Three-point security analysis for Flame skills. Scans at publish, at the pool, and on receive.

#### Enum: `ScanVerdict(str, Enum)`

Values: `CLEAN`, `FLAGGED`, `REJECTED`

#### Dataclass: `ScanFinding`

Fields: `rule`, `severity` ("low"/"medium"/"high"/"critical"), `description`, `step_index`, `tool_name`

#### Dataclass: `ScanReport`

Fields: `genome_id`, `verdict`, `findings`, `scanned_at`, `scanner_version`, `scan_duration_ms`. Properties: `critical_count`, `high_count`.

#### Class: `SkillScanner`

**Static analysis rules (8 checks):**

1. **`_check_capability_mismatch`**: Verifies declared tier matches actual tool usage. Severity: CRITICAL.
2. **`_check_exfiltration_pattern`**: Detects read-then-send sequences (e.g., `read_file` -> `web_scrape`). Severity: CRITICAL.
3. **`_check_sensitive_paths`**: Scans step parameters for sensitive file paths. Severity: CRITICAL.
4. **`_check_undeclared_tools`**: Flags tools used but not in `declared_tools`. Severity: HIGH.
5. **`_check_obfuscation`**: Detects base64 strings (40+ chars), hex-encoded content, unexpected URLs in non-network tools. Severity: HIGH/MEDIUM.
6. **`_check_excessive_scope`**: Flags >8 declared tools or read+write+network combo ("kitchen sink"). Severity: MEDIUM/HIGH.
7. **`_check_signature`**: Missing or invalid cryptographic signature. Severity: HIGH/CRITICAL.
8. **`_check_empty_or_malformed`**: Empty steps, missing name/origin, >20 steps. Severity: LOW/MEDIUM.

**Verdict computation:**
- Any CRITICAL -> REJECTED
- 2+ HIGH -> REJECTED
- 1 HIGH or any MEDIUM -> FLAGGED
- Only LOW -> CLEAN

**Runtime monitoring methods:**
- `check_runtime_tool_drift(genome, actual_tool, step_index)`: Kills skill if calling undeclared tool
- `check_runtime_data_volume(genome, step_index, output_size_bytes, threshold=1MB)`: Flags large outputs
- `check_runtime_timing(genome, step_index, elapsed_ms, threshold=30s)`: Flags slow steps

---

### 2.14 skill_collective.py

**Purpose:** The Flame -- Prometheus's shared skill network. Manages the publish/receive/sync lifecycle and collective reputation system.

#### Constants

- `MIN_REPUTATION_SCORE = 30.0` -- minimum network reputation to stay in pool
- `QUARANTINE_THRESHOLD = 2` -- quarantine votes needed for auto-recall

#### Class: `Flame`

**Constructor:** Takes `instance_id`, `local_dir` (~/.predacore/flame/local/), `shared_dir` (~/.predacore/flame/shared/).

**Publishing flow (`publish(genome)`):**
1. Verify user endorsement and propagation requirements
2. Verify cryptographic signature
3. Run pool-side security scan (Scan Point 2)
4. Check for duplicates by source_pattern; keep higher-trust version
5. Write to shared pool filesystem
6. Initialize reputation entry

**Sync flow (`sync()`):**
1. List all shared pool genomes
2. Skip own skills and already-received skills
3. Check reputation before scanning (skip if below MIN_REPUTATION_SCORE)
4. Run receiver-side security scan (Scan Point 3)
5. Accept into local pool with SANDBOXED trust level
6. Check existing skills for recalls

**Trust progression (`_promote_if_ready`):**
- SANDBOXED -> LIMITED: 5+ local successes
- LIMITED -> TRUSTED: 15+ local successes with 90%+ success rate

**Reputation system:**
- `report_success/report_failure`: Records to shared reputation.json and updates local trust
- `report_quarantine`: Adds a quarantine vote (one per instance)
- Global recall triggered when quarantine votes >= QUARANTINE_THRESHOLD

**Filesystem persistence:**
- Local skills: `~/.predacore/flame/local/{genome_id}.json`
- Shared skills: `~/.predacore/flame/shared/{genome_id}.json`
- Reputation: `~/.predacore/flame/shared/reputation.json`
- Atomic writes via tmp file + `os.replace`

---

### 2.15 skill_evolution.py

**Purpose:** Crystallization engine that watches tool execution patterns and automatically creates reusable skills.

#### Constants

- `MIN_PATTERN_OCCURRENCES = 3`
- `MIN_SUCCESS_RATE = 0.8`
- `MAX_PATTERN_LENGTH = 8`
- `MIN_PATTERN_LENGTH = 2`

#### Dataclass: `DetectedPattern`

Fields: `tool_sequence: tuple[str, ...]`, `occurrences`, `successes`, `failures`, `avg_elapsed_ms`, `first_seen`, `last_seen`, `sample_args`.

Properties: `success_rate`, `pattern_hash` (MD5), `is_crystallizable`.

#### Class: `SkillCrystallizer`

**`observe(execution_records)` -- Pattern Detection Algorithm:**
1. Extract tool names and statuses from execution records
2. Slide windows of size MIN_PATTERN_LENGTH to MAX_PATTERN_LENGTH
3. Skip sequences with empty tools or consecutive duplicates (likely retries)
4. Hash each sequence and accumulate in `_patterns` dict
5. Track occurrence counts, success/failure, average timing, sample args

**`crystallize(pattern)` -- Crystallization:**
1. Build `SkillStep` list from pattern's tool sequence and sample args
2. Create `SkillGenome` with auto-generated name, description, tags
3. Compute capability tier from tools
4. Initialize trust with pattern's success/failure counts
5. Sign the genome
6. Run security scan; reject if REJECTED, mark UNTRUSTED if FLAGGED
7. Store as pending endorsement

**`endorse(genome_id)`**: Moves from pending to crystallized, sets ENDORSED trust level, re-signs.

**`record_execution(genome_id, success)`**: Updates trust score, auto-quarantines on failure threshold.

**Persistence:** Saves to `~/.predacore/skills/evolved/evolution_state.json` containing crystallized genomes, pending endorsements, and pattern data.

---

### 2.16 Protobuf Definitions

**Path:** `_vendor/common/protos/`

**Package initializer (`__init__.py`):** Lazy-loads all 10 proto modules via `__getattr__` to avoid version check failures at import time.

All proto files use package `project_prometheus.common.protos`, Protobuf Python Version 6.31.1, gRPC Generated Version 1.78.1.

---

#### 2.16.1 CSC Proto (`csc_pb2.py` / `csc_pb2_grpc.py`)

**Enum: `Status`**

| Value | Number | Description |
|-------|--------|-------------|
| `STATUS_UNSPECIFIED` | 0 | Default |
| `STATUS_PENDING` | 1 | Awaiting processing |
| `STATUS_PROCESSING` | 2 | Currently executing |
| `STATUS_COMPLETED` | 3 | Successfully finished |
| `STATUS_FAILED` | 4 | Execution failed |
| `STATUS_CANCELLED` | 5 | User or system cancelled |
| `STATUS_READY` | 6 | Plan ready for execution |
| `STATUS_EXECUTING` | 7 | Plan being executed |

**Messages:**

| Message | Fields |
|---------|--------|
| `GoalMessage` | `id (string/1)`, `user_description (string/2)`, `interpreted_objective (Struct/3)`, `constraints (Struct/4)`, `status (Status/5)` |
| `PlanStepMessage` | `id (string/1)`, `description (string/2)`, `action_type (string/3)`, `parameters (Struct/4)`, `required_knowledge_ids (repeated string/5)`, `status (Status/6)`, `result (Value/7)` |
| `PlanMessage` | `id (string/1)`, `goal_id (string/2)`, `steps (repeated PlanStepMessage/3)`, `status (Status/4)`, `confidence (float/5)`, `justification (string/6)` |
| `ProcessGoalRequest` | `user_goal_input (string/1)`, `user_context (Struct/2)` |
| `ProcessGoalResponse` | `goal_id (string/1)`, `initial_status (Status/2)` |
| `GetGoalStatusRequest` | `goal_id (string/1)` |
| `GetGoalStatusResponse` | `goal (GoalMessage/1)`, `current_plan (PlanMessage/2)` |
| `GetPlanDetailsRequest` | `plan_id (string/1)` |

**Service: `CentralStrategicCoreService`**

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `ProcessGoal` | `ProcessGoalRequest` | `ProcessGoalResponse` | Main entry point for user goals |
| `GetGoalStatus` | `GetGoalStatusRequest` | `GetGoalStatusResponse` | Status monitoring |
| `GetPlanDetails` | `GetPlanDetailsRequest` | `PlanMessage` | Returns full plan details |

---

#### 2.16.2 DAF Proto (`daf_pb2.py` / `daf_pb2_grpc.py`)

Imports: `google.protobuf.struct`, `google.protobuf.timestamp`, `wil.proto`

**Messages:**

| Message | Key Fields |
|---------|------------|
| `AgentDescriptionMessage` | `agent_type_id (string/1)`, `description (string/2)`, `supported_actions (repeated string/3)`, `required_tools (repeated string/4)`, `configuration (Struct/5)` |
| `AgentInstanceMessage` | `agent_instance_id (string/1)`, `agent_type_id (string/2)`, `status (AgentStatus/3)`, `last_heartbeat (Timestamp/4)`, `current_task_info (Struct/5)` |
| `TaskAssignmentMessage` | `task_id (string/1)`, `plan_step_id (string/2)`, `required_capability (string/3)` or `agent_type_id (string/4)` or `agent_instance_id (string/7)` (oneof assignment_target), `parameters (Struct/5)`, `context (Struct/6)` |
| `TaskResultMessage` | `task_id (string/1)`, `plan_step_id (string/2)`, `status (InteractionStatus/3)`, `output (Value/4)`, `error_message (string/5)`, `agent_type_id_used (string/6)`, `agent_instance_id_used (string/7)` |

**Enum: `AgentStatus`** (nested in `AgentInstanceMessage`): `UNSPECIFIED`, `IDLE`, `BUSY`, `ERROR`, `RETIRED`

**Service: `DynamicAgentFabricControllerService` (8 RPCs)**

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `ListAgentTypes` | `ListAgentTypesRequest` | `ListAgentTypesResponse` | List available agent types |
| `SpawnAgent` | `SpawnAgentRequest` | `SpawnAgentResponse` | Create agent instance |
| `RetireAgent` | `RetireAgentRequest` | `RetireAgentResponse` | Terminate agent |
| `DispatchTask` | `TaskAssignmentMessage` | `TaskResultMessage` | Route task to agent |
| `RegisterAgentInstance` | `RegisterAgentInstanceRequest` | `RegisterAgentInstanceResponse` | Agent self-registration |
| `AgentHeartbeat` | `AgentHeartbeatMessage` | `AgentHeartbeatResponse` | Liveness signal |
| `GetNextTask` | `GetNextTaskRequest` | `GetNextTaskResponse` | Agent pull-based task fetch |
| `ReportTaskResult` | `ReportTaskResultRequest` | `ReportTaskResultResponse` | Submit task results |

---

#### 2.16.3 EGM Proto (`egm_pb2.py` / `egm_pb2_grpc.py`)

Imports: `google.protobuf.struct`, `csc.proto`

**Enum: `SeverityLevel`**: `UNSPECIFIED (0)`, `LOW (1)`, `MEDIUM (2)`, `HIGH (3)`, `CRITICAL (4)`

**Messages:**

| Message | Key Fields |
|---------|------------|
| `EthicalViolationMessage` | `principle_violated (string/1)`, `severity (SeverityLevel/2)`, `description (string/3)`, `component_source (string/4)` |
| `ComplianceCheckResultMessage` | `is_compliant (bool/1)`, `violations (repeated EthicalViolationMessage/2)`, `warnings (repeated string/3)`, `justification (string/4)` |
| `AuditLogEntryMessage` | `timestamp (string/1)`, `event_type (string/2)`, `component (string/3)`, `details (Struct/4)`, `compliance_status (ComplianceCheckResultMessage/5)` |
| `CheckPlanComplianceRequest` | `plan (PlanMessage/1)` |
| `CheckActionComplianceRequest` | `action_description (Struct/1)`, `context (Struct/2)` |
| `LogEventRequest` | `event_type (string/1)`, `component (string/2)`, `details (Struct/3)`, `compliance_status (ComplianceCheckResultMessage/4)` |
| `SanitizeInputRequest/Response` | `input_data/sanitized_data (Value/1)`, `context (string/2)` |
| `SanitizeOutputRequest/Response` | `output_data/sanitized_data (Value/1)`, `context (string/2)` |

**Service: `EthicalGovernanceModuleService` (5 RPCs)**

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `CheckPlanCompliance` | `CheckPlanComplianceRequest` | `ComplianceCheckResultMessage` | Validate entire plan |
| `CheckActionCompliance` | `CheckActionComplianceRequest` | `ComplianceCheckResultMessage` | Validate single action |
| `LogEvent` | `LogEventRequest` | `LogEventResponse` | Record audit event |
| `SanitizeInput` | `SanitizeInputRequest` | `SanitizeInputResponse` | Clean input data |
| `SanitizeOutput` | `SanitizeOutputRequest` | `SanitizeOutputResponse` | Clean output data |

---

#### 2.16.4 Knowledge Nexus Proto (`knowledge_nexus_pb2.py` / `knowledge_nexus_pb2_grpc.py`)

Imports: `google.protobuf.struct`, `google.protobuf.empty`

**Messages:**

| Message | Key Fields |
|---------|------------|
| `KnowledgeNodeMessage` | `id (string/1)`, `labels (repeated string/2)`, `properties (Struct/3)`, `embedding (bytes/4)`, `layers (repeated string/5)` |
| `KnowledgeEdgeMessage` | `id (string/1)`, `source_node_id (string/2)`, `target_node_id (string/3)`, `type (string/4)`, `properties (Struct/5)`, `layers (repeated string/6)` |
| `QueryNodesRequest` | `labels_filter (repeated string/1)`, `properties_filter (Struct/2)`, `layers_filter (repeated string/3)` |
| `QueryEdgesRequest` | `source_node_id_filter (string/1)`, `target_node_id_filter (string/2)`, `edge_type_filter (string/3)`, `layers_filter (repeated string/4)` |
| `SemanticSearchRequest` | `query_text (string/1)`, `top_k (int32/2)`, `layers_filter (repeated string/3)` |
| `SemanticSearchResponse` | `results (repeated Result/1)` where `Result = {node (KnowledgeNodeMessage/1), score (float/2)}` |
| `AddRelationRequest` | `source_node_id (string/1)`, `target_node_id (string/2)`, `edge_type (string/3)`, `properties (Struct/4)`, `layer (string/5)` |
| `IngestTextRequest` | `text (string/1)`, `metadata (Struct/2)`, `source (string/3)`, `layer (string/4)` |
| `IngestTextResponse` | `primary_entity_id (string/1)` |

**Service: `KnowledgeNexusService` (6 RPCs)**

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `QueryNodes` | `QueryNodesRequest` | `QueryNodesResponse` | Find nodes by labels/properties/layers |
| `QueryEdges` | `QueryEdgesRequest` | `QueryEdgesResponse` | Find edges by endpoints/type |
| `GetNodeDetails` | `GetNodeDetailsRequest` | `KnowledgeNodeMessage` | Fetch single node |
| `SemanticSearch` | `SemanticSearchRequest` | `SemanticSearchResponse` | Vector similarity search |
| `AddRelation` | `AddRelationRequest` | `KnowledgeEdgeMessage` | Create edge between nodes |
| `IngestText` | `IngestTextRequest` | `IngestTextResponse` | Ingest and embed text |

---

#### 2.16.5 WIL Proto (`wil_pb2.py` / `wil_pb2_grpc.py`)

**Enum: `InteractionStatus`**: `UNSPECIFIED (0)`, `SUCCESS (1)`, `FAILED (2)`, `ERROR (3)`, `TIMEOUT (4)`

**Enum: `ToolType`**: `UNSPECIFIED (0)`, `API (1)`, `WEB_SCRAPER (2)`, `CODE_EXECUTOR (3)`

**Messages:**

| Message | Key Fields |
|---------|------------|
| `ToolDescriptionMessage` | `tool_id (string/1)`, `tool_type (ToolType/2)`, `description (string/3)`, `input_schema (Struct/4)`, `output_schema (Struct/5)`, `metadata (Struct/6)` |
| `InteractionRequestMessage` | `request_id (string/1)`, `tool_id (string/2)`, `parameters (Struct/3)`, `credentials (Struct/4)`, `context (Struct/5)` |
| `InteractionResultMessage` | `request_id (string/1)`, `status (InteractionStatus/2)`, `output (Value/3)`, `error_message (string/4)`, `metadata (Struct/5)` |
| `CodeExecutionRequestMessage` | `request_id (string/1)`, `code (string/2)`, `input_args (Struct/3)`, `timeout_seconds (int32/4)`, `context (Struct/5)` |
| `CodeExecutionResultMessage` | `request_id (string/1)`, `status (InteractionStatus/2)`, `stdout (string/3)`, `stderr (string/4)`, `result (Value/5)`, `error_message (string/6)` |

**Service: `WorldInteractionLayerService` (3 RPCs)**

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `ListAvailableTools` | `ListAvailableToolsRequest` | `ListAvailableToolsResponse` | Get available tools |
| `ExecuteTool` | `InteractionRequestMessage` | `InteractionResultMessage` | Execute a general tool |
| `ExecuteCode` | `CodeExecutionRequestMessage` | `CodeExecutionResultMessage` | Execute code specifically |

---

## 3. Core Strategic Engine

**Path:** `_vendor/core_strategic_engine/`

The strategic planning subsystem. Originally a separate gRPC microservice, now vendored for PredaCore's integrated planning pipeline. Provides HTN (Hierarchical Task Network) planning, LLM-driven planning, and MCTS (Monte-Carlo Tree Search) plan optimization.

**`__init__.py`:** Single docstring: `"Vendored Core Strategic Engine -- only planner_mcts needed by PredaCore."`

---

### 3.1 planner.py

**Purpose:** Hierarchical Task Network (HTN) planner that decomposes user goals into executable plan steps using NLP parsing and recursive task decomposition.

**Dependencies:** `spacy`, `grpc`, `common.models`, `common.protos.knowledge_nexus_pb2/grpc`

#### Dataclasses: `Task`, `Method`

- `Task`: `name: str`, `parameters: dict`
- `Method`: `name: str`, `task_to_decompose: str`, `subtasks: list[Task]`

#### Class: `HierarchicalStrategicPlannerV1`

**Constructor:** Takes a `KnowledgeNexusServiceStub` and optional logger. Loads spaCy `en_core_web_sm` model. Defines:
- **Primitive tasks** (7): `QUERY_KN_ACTION`, `ADD_RELATION_KN_ACTION`, `ENSURE_NODE_KN_ACTION`, `SUMMARIZE_DATA_ACTION`, `DISAMBIGUATE_ENTITY_ACTION`, `CLASSIFY_GOAL_ACTION`, `GENERIC_PROCESS_ACTION`
- **Methods** (6): `method_achieve_goal`, `method_parse_and_route`, `method_handle_query`, `method_handle_add_relation`, `method_handle_summarize`, `method_handle_generic`

**`_parse_goal(goal_input) -> list[dict]`:**
1. Splits on conjunctions (`and`, `then`, `;`)
2. For each part, uses spaCy (or regex fallback) to extract: intent (`query_knowledge`, `add_relation`, `summarize_knowledge`, `generic_process`), entities, subject/relation/object
3. spaCy path: POS tagging for verbs, noun chunks, named entities, dependency parsing for relation extraction
4. Regex fallback: keyword matching for intent, pattern matching for `"X relates to Y"` style relations

**`_decompose(task, current_plan, knowledge_context)` -- Recursive decomposition:**
1. If task is primitive, convert to `PlanStep` and append to plan
2. Otherwise find applicable methods (by `task_to_decompose` match)
3. For `PARSE_AND_ROUTE_GOAL`: dynamically routes to `HANDLE_QUERY`, `HANDLE_ADD_RELATION`, `HANDLE_SUMMARIZE`, or `HANDLE_GENERIC` based on parsed intent
4. Conditional execution: skips `DISAMBIGUATE_ENTITY_ACTION` if entity has 0-1 matching nodes
5. Recurses into each subtask

**`create_plan(goal_id, goal_input, user_context) -> Plan | None`:**
1. Parses goal into subgoals
2. Pre-fetches knowledge for query/summarize intents
3. Creates top-level `PARSE_AND_ROUTE_GOAL` task
4. Runs recursive decomposition
5. Falls back to `GENERIC_PROCESS` step if decomposition fails

---

### 3.2 llm_planner.py

**Purpose:** LLM-driven planning that uses the provider-agnostic LLM client for intent extraction and step generation.

#### Class: `OpenRouterLLMClient`

Back-compatibility wrapper that delegates to `get_default_llm_client()`.

**`extract_intents_and_entities(goal_input, user_context)`:**
Sends a detailed prompt to the LLM requesting JSON array of subgoals with intents and entities. Uses 3-strategy JSON extraction: direct parse, markdown fence extraction, regex bracket matching.

**`generate_plan_steps(subgoal, context, feedback)`:**
Sends subgoal with context and any previous feedback. Requests JSON array of steps with `description`, `action_type`, `parameters`. Converts to `PlanStep` objects.

#### Class: `LLMStrategicPlanner`

Orchestrates the two-phase planning:
1. Extract intents/entities from goal via LLM
2. Generate plan steps for each subgoal
3. Assemble into a `Plan` object

---

### 3.3 planner_mcts.py

**Purpose:** AB-MCTS (Alpha-Beta Monte-Carlo Tree Search) planner that combines HTN baseline planning with MCTS-style search over plan alternatives. The most algorithmically complex component in the vendor layer.

#### Class: `ABMCTSPlanner`

**Constructor parameters (from environment):**

| Env Var | Default | Description |
|---------|---------|-------------|
| `MCTS_MAX_DEPTH` | 3 | Maximum search tree depth |
| `MCTS_BRANCHES` | 3 | Maximum children per expansion |
| `MCTS_BUDGET` | 16 | Total node expansion budget |
| `MCTS_C_PUCT` | 1.2 | Exploration constant for pUCT |
| `MCTS_REPAIR_RADIUS` | 1 | Steps around choice point to repair |
| `MCTS_PARALLEL` | 4 | Max concurrent LLM scoring calls |
| `MCTS_SCORE_TTL_SEC` | 0 | Score cache TTL |
| `MCTS_SCORE_CACHE_MAX` | 2000 | LRU cache size |
| `MCTS_W_ALPHA` | 1.0 | Weight for LLM quality score |
| `MCTS_W_BETA` | 0.2 | Weight for risk penalty |
| `MCTS_W_GAMMA` | 0.1 | Weight for cost penalty |

Also reads `MCTS_LOW_RISK_DOMAINS`, `MCTS_HIGH_RISK_DOMAINS` for domain-specific risk priors.

**`create_plan` -- Full Algorithm:**

**Step 1: Baseline plan via HTN**
Calls `self.baseline.create_plan()` to get initial plan.

**Step 2: Score baseline**
Calls `_score_steps()` which either uses LLM-based evaluation (JSON output with score 0.0-1.0 and justification) or heuristic scoring for domain keywords (edinet/filing +0.3, scrape/extract +0.3, summarize/rag +0.2). Results are memoized in an LRU OrderedDict.

**Step 3: MCTS improvement (`_mcts_improve`)**

The core search algorithm:

```
1. Initialize frontier with root node (baseline steps)
2. While frontier non-empty AND expansions < budget:
   a. SELECT: Pick node with highest pUCT score
      pUCT(node) = Q(node) + c_puct * P(node) * sqrt(total_N) / (1 + N(node))
   b. IDENTIFY choice points (underspecified steps like GENERIC_PROCESS)
   c. SEGMENT steps by intent-coherent spans (SUMMARIZE, RAG, SCRAPE, EDINET, KN, CODE, GEN)
   d. DETERMINE repair window around the choice point segment
   e. EXPAND: Generate alternative step sequences for the repair window
      - LLM expansion: ask for local improvements (rewrite/insert/remove 1-3 steps)
      - Heuristic expansion: domain-specific templates (EDINET -> fetch+embed+answer)
      - Motif expansion: retrieve from PlanMotifStore (TF-IDF matched past plans)
   f. EGM GATE: Filter children through ethical compliance check
   g. SCORE children: Multi-objective = alpha*LLM_score - beta*risk - gamma*cost
      - Bounded parallelism via asyncio.Semaphore
   h. BACKPROPAGATE: Update node value, keep top-k children in frontier
3. If best_steps improved, create new Plan with MCTS trace justification
```

**Step 4: Alternative branch generation**
Asks LLM for `branch_factor` completely alternative plans, scores each, EGM-gates, and optionally re-ranks via `PlanRanker` (multi-criteria weighted scoring).

**`_risk_cost(steps) -> (risk, cost)`:**
Scores each step by tool type:
- Code execution: risk +0.2, +0.2 if allow_network; cost +0.2
- Browser automation: risk +0.1; cost +0.3
- Scraping: cost +0.1
- RAG: cost +0.05
- Domain priors: +0.1 for high-risk domains, -0.05 for low-risk

---

### 3.4 planner_enhancements.py

**Purpose:** Proper MCTS tree data structures and the `PlanRanker` multi-criteria scoring system. Drop-in companion to `planner_mcts.py`.

#### Dataclass: `SearchConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `exploration_constant` | 1.414 (sqrt 2) | UCB1 exploration parameter |
| `max_iterations` | 100 | Search budget |
| `max_depth` | 10 | Maximum tree depth |
| `time_budget_seconds` | 5.0 | Wall-clock time limit |
| `branch_factor` | 3 | Max children per node |
| `min_visits_for_expansion` | 1 | Visits before expanding |
| `discount_factor` | 0.95 | Future reward discount |

#### Dataclass: `MCTSNode`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Auto-generated UUID hex |
| `state` | `Any` | Planner state (partial plan) |
| `action` | `str` | Action that led here |
| `parent` | `MCTSNode \| None` | Parent reference |
| `children` | `list[MCTSNode]` | Child nodes |
| `visits` | `int` | Visit count |
| `total_value` | `float` | Cumulative value |
| `depth` | `int` | Tree depth |
| `is_terminal` | `bool` | Terminal state flag |

**Properties:** `value` (average = total/visits), `is_fully_expanded`, `is_leaf`

**UCB1 formula:**
```
UCB1(node) = value + c * sqrt(ln(parent_visits) / visits)
```
Returns infinity for unvisited nodes.

#### Class: `MCTSTree`

Full MCTS implementation with the four standard phases:

**`search(expand_fn, simulate_fn) -> MCTSNode`:**

```
For each iteration (up to max_iterations or time_budget):
  1. SELECT: Walk tree using UCB1 to find a leaf
     - If node not fully expanded, return it for expansion
     - Otherwise follow best_child(c) down
  2. EXPAND: Call expand_fn(state) to get (child_state, action) pairs
     - Limited to branch_factor children
     - Mark terminal if no children generated
  3. SIMULATE: Call simulate_fn(state) to get value estimate [0, 1]
  4. BACKPROPAGATE: Walk up to root, incrementing visits and adding value
```

**Result extraction:**
- `best_action()`: Action of root's most-visited child
- `best_child_node()`: Root's most-visited child
- `get_statistics()`: iterations, total nodes, max depth, root visits/value

#### Dataclass: `PlanCandidate`

Fields: `plan_id`, `plan_data`, `scores: dict[str, float]`. Property: `overall_score` (average of all scores).

#### Class: `PlanRanker`

Multi-criteria plan ranking with weighted scoring.

**Default weights:** cost=0.2, risk=0.3, latency=0.2, quality=0.3

**`rank(candidates)`**: Computes weighted score for each candidate, sorts descending.

**`compare(plan_a, plan_b)`**: Returns `"a"`, `"b"`, or `"tie"` (within 0.01 tolerance).

---

### 3.5 plan_cache.py

**Purpose:** JSONL-backed motif store for reusing successful plan segments. Enables the MCTS planner to seed expansions with previously successful patterns.

#### Class: `PlanMotifStore`

**Constructor:** Path defaults to `MCTS_PLAN_CACHE_PATH` env or `"data/plan_motifs.jsonl"`. Thread-safe writes via `threading.Lock`.

**`add_motif(goal, steps)`:** Appends `{"id": uuid, "goal_terms": [...], "steps": [...]}` to JSONL file.

**`retrieve(goal, top_k=3) -> list[list[dict]]`:** TF-IDF-like retrieval:
1. Tokenize goal into lowercase alpha terms
2. Read all entries, compute document frequency for each term
3. IDF formula: `log((N+1)/(df+1)) + 1.0`
4. Score each entry: `sum(IDF of shared terms) / max(1, len(entry_terms))`
5. Return top-k step lists sorted by score

**`_terms(text) -> set`:** Splits on whitespace, keeps alpha-only tokens and hyphenated words.

---

## 4. Ethical Governance Module

**Path:** `_vendor/ethical_governance_module/`

Safety and compliance layer that gates agent actions. Originally a gRPC microservice; now vendored for PredaCore's integrated governance pipeline.

**`__init__.py`:** `"Vendored Ethical Governance Module -- only persistent_audit needed by PredaCore."`

---

### 4.1 service.py

**Purpose:** gRPC service implementation for EGM. Implements all 5 RPCs defined in the EGM proto.

#### Class: `EthicalGovernanceModuleService(EthicalGovernanceModuleServiceServicer)`

**Constructor:** Takes `BasicRuleEngine`, `AbstractAuditLogger`, optional logger.

**`CheckPlanCompliance`:** Iterates through every step of the plan, calling `rule_engine.check_compliance()` on each. Collects violations and warnings. In `log_only` mode (default EGM_MODE), passes non-HIGH-severity violations. Logs result via `LogEvent`.

**`CheckActionCompliance`:** Single-action version. Converts protobuf Struct to dict, passes to rule engine. Same `log_only` mode behavior.

**`LogEvent`:** Delegates to `audit_logger.log()`.

**`SanitizeInput` / `SanitizeOutput`:** Currently pass-through (TODO: implement sanitization logic).

**`serve()`:** Standalone server function that creates `BasicRuleEngine`, `FileAuditLogger`, binds to port 50053.

---

### 4.2 rule_engine.py

**Purpose:** Policy enforcement engine with forbidden keywords, risky action types, and parameterized rules.

#### Constants

**`FORBIDDEN_KEYWORDS`:** `delete_user_data`, `disable_safety`, `ignore_ethics`, `harm_human`, `overwrite_system_files`

**`RISKY_ACTION_TYPES`:** `EXECUTE_ARBITRARY_CODE`, `MODIFY_FILESYSTEM`, `SEND_EMAIL`, `ACCESS_SENSITIVE_DATA`

#### Class: `BasicRuleEngine`

**`check_compliance(item_to_check, context=None) -> ComplianceCheckResultMessage`:**

Accepts plan steps (with `.description`/`.action_type`), dicts, or plain strings.

**Rules evaluated:**

1. **Forbidden keywords** (SEVERITY_HIGH, principle: NonMaleficence): Scans text for any forbidden keyword
2. **Risky action types** (SEVERITY_MEDIUM, principle: RiskMitigation): Flags known risky action_type values
3. **Filesystem path validation** (SEVERITY_HIGH, principle: Safety): For `MODIFY_FILESYSTEM`, verifies target path is under `ALLOWED_FS_BASE_PATH` (env var, defaults to cwd)
4. **Email domain allowlist** (SEVERITY_HIGH, principle: DataPrivacy): For `SEND_EMAIL`, checks recipient domain against `approved_domains` in context
5. **Slack channel allowlist** (SEVERITY_MEDIUM, principle: DataPrivacy): For `SLACK_BOT`, checks channel against `approved_slack_channels`
6. **Browser domain allowlist** (SEVERITY_MEDIUM, principle: DataPrivacy): For `BROWSER_AUTOMATION`, checks URL domain against `approved_domains`, supports subdomain matching

---

### 4.3 audit_logger.py

**Purpose:** Abstract and file-based audit logging for EGM events.

#### Class: `AbstractAuditLogger`

Single abstract method: `async log(entry)`.

#### Class: `FileAuditLogger`

Writes structured JSON lines to `logs/egm_audit.log`. Uses `asyncio.Lock` for concurrent write safety.

**`log(entry: LogEventRequest)`:** Formats a JSON record with timestamp, event_type, component, details (Struct -> dict), and compliance_status (violations, warnings, justification). Appends to file.

---

### 4.4 persistent_audit.py

**Purpose:** SQLite-backed audit trail with query, statistics, and reporting capabilities. Replaces flat-file logging.

#### Dataclass: `AuditEntry`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timestamp` | `str` | `""` | ISO 8601 timestamp |
| `component` | `str` | `""` | Source component |
| `event_type` | `str` | `""` | Event classification |
| `severity` | `str` | `"INFO"` | INFO/WARNING/VIOLATION/CRITICAL |
| `is_compliant` | `bool` | `True` | Compliance result |
| `decision` | `str` | `""` | ALLOW/BLOCK/WARN |
| `justification` | `str` | `""` | Reasoning |
| `principle` | `str` | `""` | EGM principle evaluated |
| `details` | `dict` | `{}` | Additional metadata |
| `entry_id` | `int` | `0` | DB auto-increment ID |

#### Class: `PersistentAuditStore`

**Schema:**
```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    component TEXT DEFAULT '',
    event_type TEXT DEFAULT '',
    severity TEXT DEFAULT 'INFO',
    is_compliant INTEGER DEFAULT 1,
    decision TEXT DEFAULT 'ALLOW',
    justification TEXT DEFAULT '',
    principle TEXT DEFAULT '',
    details TEXT DEFAULT '{}'
);
```

Indexes on: `timestamp`, `component`, `severity`, `is_compliant`.

**Methods:**
- `log_decision(...)` / `log_decision_async(...)`: Insert audit record
- `query(...)` / `query_async(...)`: Filter by component, severity, compliance status, time range, principle
- `get_statistics()` / `get_statistics_async()`: Returns total entries, compliance rate, violations by principle and severity
- `export_report(limit=500)`: Generates markdown audit report
- Supports optional `db_adapter` for remote/centralized DB access

---

## 5. Knowledge Nexus

**Path:** `_vendor/knowledge_nexus/`

Knowledge graph and semantic search subsystem. Stores entities as nodes with labels, properties, and embeddings; relationships as typed edges between nodes.

**`__init__.py`:** Exports `KnowledgeNexusService` and `InMemoryKnowledgeGraphStore`.

---

### 5.1 service.py

**Purpose:** gRPC service implementation for all 6 Knowledge Nexus RPCs.

#### Class: `KnowledgeNexusService(KnowledgeNexusServiceServicer)`

**Constructor:** Takes `AbstractKnowledgeGraphStore`, `AbstractVectorIndex`, optional logger and `EmbeddingClient`.

**`QueryNodes`:** Extracts label/property/layer filters from request, delegates to `storage.query_nodes()`, converts results to protobuf messages.

**`QueryEdges`:** Validates UUID formats, delegates to `storage.query_edges()` with source/target/type/layer filters.

**`GetNodeDetails`:** Fetches single node by UUID from storage.

**`SemanticSearch`:**
1. Generates embedding for query text via `embedding_client.embed()`
2. Searches `vector_index.search_similar()` with optional layer filter
3. Retrieves full node details from storage for each result
4. Returns `(node, score)` pairs

**`AddRelation`:** Creates `KnowledgeEdge` Pydantic model, validates source/target nodes exist, adds to storage.

**`IngestText`:**
1. Merges metadata into properties with text and source
2. Generates embedding (non-fatal on failure)
3. Creates `KnowledgeNode` with label `"TextChunk"` and stores
4. Adds embedding to vector index with metadata
5. Persists vector index if `save_to_disk` is available

**Helper methods:** `_dict_to_struct`, `_struct_to_dict`, `_node_to_message`, `_edge_to_message` -- bidirectional conversions between Pydantic models and protobuf messages.

**`serve()`:** Standalone server with storage backend selection (`KN_STORE_BACKEND`: "memory" or "disk"), vector index initialization (loads from disk or creates new with auto-detected dimensions), listens on `KN_LISTEN_ADDR` (default `[::]:50051`).

---

### 5.2 storage.py

**Purpose:** Graph storage implementations -- in-memory and disk-backed JSONL.

#### Class: `AbstractKnowledgeGraphStore`

Abstract base with 8 methods: `add_node`, `get_node`, `update_node`, `delete_node`, `add_edge`, `get_edge`, `query_nodes`, `query_edges`, `get_neighbors`.

#### Class: `InMemoryKnowledgeGraphStore`

**Internal state:**
- `_nodes: dict[UUID, KnowledgeNode]`
- `_edges: dict[UUID, KnowledgeEdge]`
- `_label_index: dict[str, set[UUID]]` -- label to node IDs
- `_edge_type_index: dict[str, set[UUID]]` -- edge type to edge IDs
- `_outgoing_edges: dict[UUID, set[UUID]]` -- node ID to outgoing edge IDs
- `_incoming_edges: dict[UUID, set[UUID]]` -- node ID to incoming edge IDs

All operations return deep copies (`model_copy(deep=True)`) to prevent external mutation.

**`query_nodes`:** Uses label index for initial filtering (intersection of all label sets), then applies property exact-match and layer intersection filters.

**`query_edges`:** Progressive filtering using edge_type_index, outgoing_edges, incoming_edges, then layer filter.

**`get_neighbors`:** Finds connected edges by direction (outgoing/incoming/both), filters by edge type, retrieves neighbor nodes.

**`delete_node`:** Cascading delete -- removes all connected edges first, then the node itself, updating all indexes.

#### Class: `DiskKnowledgeGraphStore`

Extends the in-memory store with JSONL persistence at `base_dir/nodes.jsonl` and `base_dir/edges.jsonl`.

- `add_node`/`add_edge`: Append to JSONL file
- `update_node`/`delete_node`: Full JSONL rewrite
- Query methods delegate to `InMemoryKnowledgeGraphStore` implementations
- `_load_from_disk`: Reads JSONL files on initialization, populates in-memory state and indexes

---

### 5.3 vector_index.py

**Purpose:** Abstract base class for vector index implementations.

#### Class: `AbstractVectorIndex(ABC)`

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `async (item_id, embedding, metadata=None) -> bool` | Add item |
| `search_similar` | `async (query_embedding, top_k=10, layers=None) -> list[(str, float)]` | Search |
| `remove` | `async (item_id) -> bool` | Remove item |
| `update` | `async (item_id, embedding, metadata=None) -> bool` | Default: remove + re-add |

---

### 5.4 faiss_vector_index.py

**Purpose:** Concrete vector index with cosine similarity search, disk persistence, and layer filtering. Despite the name, uses pure Python (no FAISS dependency).

#### Class: `FAISSVectorIndex(AbstractVectorIndex)`

**Constructor:** `dimensions: int = 128`

**Internal storage:** `_store: dict[str, _VectorEntry]` where `_VectorEntry` has `item_id`, `embedding`, `metadata`, `layers` (extracted from metadata).

**`search_similar(query_embedding, top_k, layers)`:**
1. Validates query dimension
2. Iterates all entries, filters by layer intersection
3. Computes cosine similarity: `dot(a,b) / (||a|| * ||b||)`
4. Sorts descending, returns top-k `(item_id, score)` pairs

**Persistence:**
- `save_to_disk(path)`: Serializes to JSON with `dimensions` and `entries` array
- `load_from_disk(path)` (classmethod): Deserializes and rebuilds index

**Helper class: `HashEmbedding(EmbeddingProvider)`** -- Character-level hashing for zero-cost embeddings (128 dimensions default). Uses `hash(ch) ^ (i * 2654435761)` for distribution.

---

### 5.5 health.py

**Purpose:** FastAPI health/readiness endpoints.

```python
@app.get("/healthz")    -> {"status": "ok"}
@app.get("/ready")      -> {"ready": True}
```

---

## 6. User Modeling Engine

**Path:** `_vendor/user_modeling_engine/`

Lightweight user profile and behavior tracking system.

**`__init__.py`:** Exports `UserModelingEngineService` and `UserProfile`.

---

### 6.1 service.py

**Purpose:** Persistent user profiles stored as JSON files with event logging.

#### Dataclass: `UserProfile`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `user_id` | `str` | required | User identifier |
| `preferences` | `dict` | `{}` | User preferences (e.g., dark mode, verbosity) |
| `goals` | `list[str]` | `[]` | Active user goals |
| `knowledge_areas` | `dict` | `{}` | Expertise areas |
| `cognitive_style` | `str` | `""` | Communication preference |
| `notes` | `str` | `""` | Free-text notes |
| `last_interaction_at` | `str` | `""` | Last activity timestamp |
| `updated_at` | `str` | now UTC | Last update timestamp |

Methods: `to_dict()`, `from_dict()` (class method).

#### Class: `UserModelingEngineService`

**Constructor:** Takes `data_path` (default `"data/ume"`). Creates `profiles/` and `events/` subdirectories.

**File layout:**
- `data/ume/profiles/{user_id}.json` -- profile data
- `data/ume/events/{user_id}.jsonl` -- interaction event log

**Methods:**

| Method | Description |
|--------|-------------|
| `get_profile(user_id) -> UserProfile` | Reads from JSON file or returns default |
| `save_profile(profile) -> UserProfile` | Writes to JSON file, updates timestamp |
| `update_profile(user_id, patch) -> UserProfile` | Deep-merges patch into existing profile |
| `record_interaction(user_id, message, metadata)` | Updates last_interaction_at, appends event to JSONL |
| `build_planning_context(user_id) -> dict` | Returns profile data formatted for CSE planner context |

**Helper: `_safe_user_id(user_id)`** -- Sanitizes to `[a-zA-Z0-9_.-]` for filesystem safety.

**Helper: `_deep_merge(base, override)`** -- Recursive dict merge where override wins for non-dict values and dicts are merged recursively.

---

## 7. Integration Map

### How vendor modules connect to each other

```
common/embedding.py -----> common/vector_store.py (DiskBackedVectorIndex uses InMemoryVectorIndex)
                    -----> knowledge_nexus/service.py (embedding client for SemanticSearch/IngestText)
                    -----> common/memory_service.py (embedding for semantic recall)

common/llm.py -----------> core_strategic_engine/llm_planner.py (LLM client for plan generation)
                    -----> core_strategic_engine/planner_mcts.py (LLM for scoring and expansion)

common/models.py --------> core_strategic_engine/planner.py (Plan, PlanStep, StatusEnum)
                    -----> core_strategic_engine/llm_planner.py (same)
                    -----> core_strategic_engine/planner_mcts.py (same)
                    -----> knowledge_nexus/storage.py (KnowledgeNode, KnowledgeEdge)
                    -----> knowledge_nexus/service.py (same)

common/protos/ ----------> core_strategic_engine/planner.py (knowledge_nexus_pb2 for KN queries)
               ----------> core_strategic_engine/planner_mcts.py (egm_pb2 for compliance checks)
               ----------> ethical_governance_module/service.py (egm_pb2/grpc for service impl)
               ----------> ethical_governance_module/rule_engine.py (egm_pb2 for violation messages)
               ----------> knowledge_nexus/service.py (knowledge_nexus_pb2/grpc for service impl)

common/metrics.py -------> common/logging_config.py (trace ID for log correlation)
                   -------> core_strategic_engine/planner_mcts.py (prometheus Counter/Histogram)

common/skill_genome.py --> common/skill_scanner.py (SkillGenome, CapabilityTier, TrustLevel, etc.)
                    -----> common/skill_collective.py (SkillGenome, TrustLevel, TIER_PROPAGATION)
                    -----> common/skill_evolution.py (SkillGenome, SkillStep, CapabilityTier)

common/skill_scanner.py -> common/skill_collective.py (ScanVerdict, SkillScanner)
                    -----> common/skill_evolution.py (ScanVerdict, SkillScanner)

core_strategic_engine/
  planner.py ------------> planner_mcts.py (ABMCTSPlanner wraps HierarchicalStrategicPlannerV1)
  plan_cache.py ---------> planner_mcts.py (PlanMotifStore for motif-based expansion)
  planner_enhancements.py -> planner_mcts.py (PlanCandidate, PlanRanker for multi-criteria ranking)
```

### How PredaCore core imports vendor modules

PredaCore's main codebase imports:
- `predacore._vendor.common.embedding` -- for `get_default_embedding_client()`
- `predacore._vendor.common.llm` -- for `get_default_llm_client()`
- `predacore._vendor.common.memory_service` -- for `MemoryService`
- `predacore._vendor.common.errors` -- for the error hierarchy
- `predacore._vendor.common.metrics` -- for observability
- `predacore._vendor.common.logging_config` -- for `setup_logging()`
- `predacore._vendor.common.models` -- for Plan, PlanStep, StatusEnum
- `predacore._vendor.common.skill_genome` -- for SkillGenome
- `predacore._vendor.common.skill_scanner` -- for SkillScanner
- `predacore._vendor.common.skill_collective` -- for Flame
- `predacore._vendor.common.skill_evolution` -- for SkillCrystallizer
- `predacore._vendor.core_strategic_engine.planner_mcts` -- for ABMCTSPlanner
- `predacore._vendor.ethical_governance_module.persistent_audit` -- for PersistentAuditStore
- `predacore._vendor.knowledge_nexus` -- for KnowledgeNexusService
- `predacore._vendor.user_modeling_engine` -- for UserModelingEngineService, UserProfile

---

## 8. Algorithm Deep-Dives

### 8.1 MCTS Plan Search (ABMCTSPlanner)

The AB-MCTS planner implements a bounded tree search that locally improves an HTN-generated baseline plan. It is not a standard game-playing MCTS but an adaptation for plan optimization.

**Pseudocode:**

```
function MCTS_IMPROVE(goal, baseline_plan):
    root = {steps: baseline_plan.steps, depth: 0, n: 0, q: score(baseline)}
    frontier = [root]
    best = baseline_plan.steps
    best_score = score(baseline)

    while frontier non-empty AND expansions < BUDGET:
        // SELECT via pUCT
        node = argmax(frontier, pUCT)
        remove node from frontier
        if node.depth >= MAX_DEPTH: continue

        // IDENTIFY repair window
        choice_point = first underspecified step, or midpoint
        segment = intent-coherent span containing choice_point
        window = expand segment by REPAIR_RADIUS in both directions

        // EXPAND
        children = []
        for each window:
            children += LLM_expand(goal, steps[window])  // or heuristic
            children += motif_variants(goal, steps, window)
        
        // EGM GATE
        children = [c for c in children if EGM_OK(c)]

        // SCORE with bounded parallelism
        scored = parallel_for c in children:
            llm_score = LLM_evaluate(goal, c)
            risk, cost = compute_risk_cost(c)
            return ALPHA*llm_score - BETA*risk - GAMMA*cost

        // UPDATE best
        if max(scored) > best_score:
            best = argmax(scored)
            best_score = max(scored)

        // BACKPROPAGATE
        node.q = max(node.q, max(scored))
        node.n += 1

        // ENQUEUE top children
        for top-k scored children:
            frontier.append({steps: child, depth: depth+1, ...})

    return Plan(steps=best) if improved else None
```

### 8.2 Pure MCTS (MCTSTree in planner_enhancements.py)

Standard four-phase MCTS with UCB1 tree policy:

```
function SEARCH(root_state, expand_fn, simulate_fn):
    root = MCTSNode(state=root_state)
    
    for iteration in range(MAX_ITERATIONS):
        if elapsed > TIME_BUDGET: break
        
        // SELECTION: Walk tree using UCB1
        node = root
        while node has children AND node fully expanded:
            node = argmax(node.children, UCB1)
        
        // EXPANSION: Generate children
        if node not terminal AND node.depth < MAX_DEPTH:
            children_specs = expand_fn(node.state)
            for (state, action) in children_specs[:BRANCH_FACTOR]:
                node.add_child(state, action)
            node = last child added
        
        // SIMULATION: Evaluate leaf
        value = simulate_fn(node.state)
        
        // BACKPROPAGATION: Update ancestors
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
    
    return root

UCB1(node) = (total_value / visits) + C * sqrt(ln(parent.visits) / visits)
```

### 8.3 Cosine Similarity Search (InMemoryVectorIndex)

The vector search algorithm in `embedding.py` uses numpy-optimized batch computation:

```
function SEARCH(query_vec, top_k):
    // Normalize query
    query = query_vec / ||query_vec||
    
    // Batch cosine similarity via matrix-vector multiply
    // Since all stored vectors are pre-normalized on add:
    //   cos(a, b) = dot(a, b) when ||a|| = ||b|| = 1
    scores = matrix @ query  // shape: (n_items,)
    
    // Efficient top-k extraction
    if n_items <= top_k:
        indices = argsort(scores)[::-1]
    else:
        // O(n) partial sort via argpartition
        indices = argpartition(scores, -top_k)[-top_k:]
        indices = indices[argsort(scores[indices])[::-1]]
    
    return [(ids[i], scores[i], metas[i]) for i in indices]
```

### 8.4 Skill Security Model (Flame)

The three-scan-point security model protects the skill network:

```
SCAN POINT 1 (On Publish - Creator side):
    SkillScanner.scan(genome) -> ScanReport
    If REJECTED: skill never reaches shared pool

SCAN POINT 2 (At Pool - Arrival):
    Flame.publish() calls SkillScanner.scan(genome)
    Verifies: endorsement, propagation requirements, signature
    If REJECTED: skill blocked from shared pool

SCAN POINT 3 (On Receive - Consumer side):
    Flame.sync() calls SkillScanner.scan(genome)
    Checks reputation before scanning
    If REJECTED: quarantine vote + skip

RUNTIME MONITORING:
    check_runtime_tool_drift() -- kill if calling undeclared tools
    check_runtime_data_volume() -- flag large outputs
    check_runtime_timing() -- flag slow execution
```

### 8.5 Memory Recall Pipeline

```
function RECALL(query, user_id, filters, top_k):
    // 1. Ensure user data in cache
    load_from_sqlite_if_not_cached(user_id)
    
    // 2. Try semantic search
    if embedding_client AND user has vectors:
        query_embedding = await embed([query])
        for each (mem_id, mem_embedding) in user_vectors:
            if dimension_mismatch: skip
            if not passes_filters(memory, types, tags, min_importance): skip
            score = cosine_similarity(query_embedding, mem_embedding)
            candidates.append((memory, score))
    
    // 3. Keyword fallback
    if no semantic results:
        for each memory in user_cache:
            if query_lower in memory.content.lower():
                score = keyword_overlap_ratio
                candidates.append((memory, score))
    
    // 4. Sort and return top-k
    sort(candidates, by=score, descending)
    return candidates[:top_k]
```

---

*This documentation covers every source file, every class, every method, every protobuf message, and every algorithm in the PredaCore vendor layer. Total files documented: 44 source files across 5 subsystems.*
