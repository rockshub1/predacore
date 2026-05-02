# Changelog

All notable changes to PredaCore will be documented in this file.

## [1.4.0] - 2026-05-02

**Big release.** Channel surface tripled (8 → 24), browser control rewritten to be CDP-only with selector caching for sub-100ms repeats, memory subsystem gets bulk-index + tier-0 git-blob verifiers, full channel scaffold UX, free image generation via Gemini, streaming responses on telegram/discord/slack. 12 PRs (T1–T11) merged into one release.

### Memory subsystem (T5a–T5d, T6, T7)
- **Bulk indexer**: ``bulk_index_directory(root)`` walks a project tree, applies ``.memoryignore``, embeds every chunk into the unified store. New tools: ``memory_bulk_index``, ``memory_bulk_abort``, ``memory_index_status``, ``memory_scan_directory``.
- **Workspace tracker**: detects project changes, writes a first-touch marker so the daemon knows when a project is bulk-indexed.
- **Channel-aware system prompt**: ``_workspace_context_block`` injects "active project + bulk-indexed yes/no + enabled channels" into every system prompt. Adaptive — picks up new channels automatically.
- **Healer rate brake**: scaled by row count (``max(1000, total_rows // 5)``); audit label-flips no longer count toward the brake (was tripping after BGE upgrades on bulk-indexed DBs).
- **Tier-0 git-blob verifier**: re-hash the file's bytes git-style; if it matches the stored ``source_blob_sha``, every chunk of that file is verified in one hash op (mtime-keyed cache).
- **Tier-1.5 AST-symbol verifier**: for ``chunk_kind in {function, class, method}`` chunks, verify the named symbol still exists in the source — survives body edits / refactors that substring matching would reject.
- **Recall**: 0.967 R@5, 1.000 R@10/R@20 on the production benchmark (164 files, 1711 chunks of real PredaCore source). Defaults: top_k 5 → 10, semantic budget 1200 → 1800 tokens.

### Throughput (T5c)
- Real batched embedding in Rust kernel (predacore_core 1.2.0): rayon-parallel tokenization, batched forward pass (``EMBED_BATCH_SIZE = 32``), per-row L2 normalize. Replaces the v1 fake-batch (per-text forward inside a loop).
- Per-file SQLite transaction batching: ``_bulk_store_code_chunks`` purges stale + inserts all new chunks in ONE commit instead of N. Vector-index updates moved outside the DB lock.

### Channels — 8 → 24 (T8, T9, T10, T11)
- **Hot-attach** (T8): ``channel_configure add`` no longer requires daemon restart. Config-watcher diffs ``channels.enabled`` and live-registers/deregisters adapters.
- **Streaming** (T9): telegram, discord, slack now stream LLM tokens into a single edited message (~1s rate-limit aware). Watch the assistant type. Signal deferred — its REST API doesn't expose ``editTimestamp`` yet.
- **Channel scaffold + /channel UX** (T10): ``predacore channel scaffold <name>`` writes a starter adapter; ``/channel list / info / scaffold`` works inside any chat. Per-channel secret manifest covers all 24.
- **+4 adapters** (T10c): Twilio (SMS), Matrix (matrix-nio), Mastodon (Fediverse DM), Bluesky DM (atproto).
- **+12 adapters** (T11, research-validated against current SDKs): Mattermost, Rocket.Chat, Vonage SMS, MessageBird SMS, Line, Viber, slixmpp (XMPP), pydle (IRC), Google Chat, Threema Gateway, Zalo OA, KakaoTalk. MS Teams + Skype skipped (``botbuilder-python`` is end-of-life per Microsoft); Wire skipped (bot API still in beta).
- **Total channel count**: cli, telegram, discord, whatsapp, webchat, slack, signal, imessage, email, twilio, matrix, mastodon, bluesky, mattermost, rocketchat, vonage, messagebird, line, viber, xmpp, irc, google_chat, threema, zalo, kakaotalk = 24 built-in adapters.

### Browser control (T4)
- **Safari bridge removed** — Chrome / Chromium-derivatives only. Single backend = single well-tested path. ``connect(browser="safari")`` warns and falls back to Chrome (graceful degradation, not a hard break).
- **Persistent profile** at ``~/.predacore/chrome-profile`` so logins survive across daemon launches without disturbing the user's main Chrome.
- **Selector cache** (T4b): ``(domain, intent_hash) → xpath`` mapping in a dedicated SQLite table inside the memory DB. Cache hit + CDP verify = ~50ms click. Faster than human reaction time. Cache miss falls through to natural-language resolver, then writes back on success.
- **Vision plumbing** (T4c): CanvasDetector (cheap JS heuristic), OptInGate (auto/always/off), VisionProvider Protocol, downloader for Samsung/TinyClick (MIT, 0.27B params, ~540MB) — picked over OmniParser (AGPL on icon_detect would force PredaCore to relicense). Inference path lives in ``predacore_core_crate/src/omniparser.rs`` skeleton; final ``ort``-based binding pending follow-up.

### Free image generation (T10d)
- ``image_gen`` tool now auto-routes: Gemini 2.5 Flash Image (nano-banana, **free tier**) when ``GEMINI_API_KEY`` is set, falls back to DALL-E 3 (paid) when only ``OPENAI_API_KEY`` is set. Pass ``provider="gemini"`` or ``"openai"`` to force one.

### Vendor cleanup (T1, T2, T2.5, T3)
- SSRF DNS auto-mock conftest — fixes long-standing test flake on real-network paths.
- Collapsed ``_vendor/common/llm.py`` into ``predacore.llm_providers`` — removed the legacy LLM client wrapper, planner_mcts and daf/scheduler now use the canonical ``LLMInterface``.
- Removed Knowledge Nexus call sites from planner.py — KN was retired but planner.py still imported its protos.
- Routed ``services/code_index.py`` directly to Rust BGE — dropped Python embedder fallback.

### Tests
- 2063 deterministic tests passing (full sweep, no real-network / no AX-permission paths).
- New tests this release: 11 selector cache + 9 vision plumbing + 9 model loader + 5 healer brake + 5 T7 verifier + 6 streaming + 10 channel scaffold + 4 image_gen routing + 5 hot-attach + 5 workspace block = **69 new tests**.

### Migration notes
- **No breaking API changes**. Safari users transparently land on Chrome.
- **New channels need new tokens**: see ``CHANNEL_SECRETS`` in ``tools/handlers/channels.py`` or run ``/channel info <name>`` for the env vars each new adapter wants.
- **First bootstrap after upgrade**: a new ``TinyClick model`` step appears (defaults to "deferred — will download on first canvas-app page"). Set ``browser.local_vision="always"`` in config if you want eager pre-warm.

### Companion package
- ``predacore_core`` bumped to **1.2.0** (additive — batched embed, T5c). Auto-published via ``.github/workflows/build-wheels.yml`` on the ``v1.4.0`` tag.

## [1.3.0] - 2026-04-26

**Operational memory guide moved to code — eliminates the workspace migration gap.**

### Architectural change
- Split MEMORY.md into two concerns: **engineering knowledge** ("how the
  memory tools work, when to call them") now lives in code as the
  `_MEMORY_GUIDE` constant in `src/predacore/identity/engine.py`, and
  **user-curated content** (preferences, decisions, lessons) stays in the
  workspace `MEMORY.md`. The guide auto-updates with every release; user
  content is preserved across upgrades.
- New `IdentityEngine.memory_guide()` method returns the operational guide.
- `build_identity_prompt()` injects the guide as a "How Memory Works" section
  immediately before the workspace MEMORY.md content (now wrapped as
  "Curated Memory (MEMORY.md)"). Both layers are always present in the
  assembled prompt.
- New layer order: `… → memory_guide (code) → MEMORY (workspace) → TOOLS → …`

### Why
- Pre-1.3.0, operational guidance lived in `defaults/MEMORY.md` which was
  copied to a user's workspace on first install and never overwritten on
  upgrade. Adding/changing tools (e.g. v1.2.0's 4 new memory tools) meant
  existing users wouldn't see the updated guidance — they'd be acting on
  frozen instructions from whenever they first installed.
- Code-level constant means: rename a tool, add a new one, change the
  discipline rule — one edit, ships to every user on next pip upgrade,
  no migration logic, no "predacore doctor --fix-defaults" needed.

### MEMORY.md defaults stripped
- Shipped `defaults/MEMORY.md` is now a scaffold for curated content only —
  removed the dual-layer / 6-tool / infrastructure-layer sections (they
  moved to the code constant). Keeps the "What belongs here / What doesn't"
  guidance for the agent's own curation discipline.

### Tests
- 5 new tests in `test_memory_subsystem.py::TestMemoryGuideInPrompt`:
  guide is non-empty, mentions all 6 tools, present in prompt even with
  empty workspace MEMORY.md, coexists with curated content, and the
  shipped defaults file does NOT carry the operational content (regression
  guard for the 1.2.0 mistake).
- **205 deterministic tests passing** (200 prior + 5 new).

### Migration impact
- **Existing users get the new operational guide automatically** on
  upgrade — no manual `cp` step needed (unlike the v1.2.0 → 1.2.1 case),
  because the guide is now in the code path that runs on every prompt
  assembly. Their workspace `MEMORY.md` is left alone (still curated
  content only, still respected).
- The stale top-level `agents/default/MEMORY.md` developer mirror has
  been synced to match the new defaults shape; cleanup of that mirror
  remains deferred.

## [1.2.1] - 2026-04-26

**Patch — fixes two bugs shipped in 1.2.0.**

### Fixed
- `predacore --version` now correctly reports the installed version. Bumped
  stale `__version__` constant in `src/predacore/__init__.py` (was hardcoded
  to `"1.1.1"` and never bumped during the 1.2.0 release; package metadata
  was correct at 1.2.0 but the in-code string lagged). Importing
  `predacore.__version__` now matches the wheel's metadata.
- **MEMORY.md defaults sync**: the W7 update for the dual-layer memory model
  ("passive auto-context + active memory tools") and the 6-tool guidance was
  written to top-level `agents/default/MEMORY.md` (which doesn't ship in the
  wheel) instead of `src/predacore/identity/defaults/MEMORY.md` (the file
  that's actually copied to a user's workspace on bootstrap). Synced the W7
  content to the shipped defaults so fresh installs see the dual-layer
  guidance. **Existing users**: bootstrap does NOT overwrite an existing
  `~/.predacore/agents/default/MEMORY.md`, so to pick up the new defaults
  after upgrading, manually replace your workspace copy:
  ```bash
  cp $(python -c "import predacore.identity.defaults as d, pathlib; print(pathlib.Path(d.__file__).parent / 'MEMORY.md')") ~/.predacore/agents/default/MEMORY.md
  ```

### Known issue (not blocking)
- The top-level `agents/default/` directory at the repo root is a stale
  developer mirror that does NOT ship in the wheel. 4 files there
  (HEARTBEAT.md, IDENTITY.md, MEMORY.md, SOUL.md) diverge from the canonical
  `src/predacore/identity/defaults/` versions; for HEARTBEAT/IDENTITY/SOUL the
  defaults are NEWER, while MEMORY.md was edited at the top level (1.2.0's W7
  bug — fixed here). Treat `src/predacore/identity/defaults/` as canonical
  going forward. Cleanup of the top-level mirror deferred to a separate PR.

## [1.2.0] - 2026-04-26

**Phase 2 memory upgrade — auto-trigger wiring, project isolation, verify-with-code.**

### Added
- **Auto-trigger wiring** — `Edit`/`Write` automatically reindex via `reindex_file()`;
  `git checkout/merge/rebase/reset/pull/cherry-pick/revert` automatically sync via
  `sync_git_changes(prior_head=)`. Captures both working-tree AND committed deltas
  (closes the gap that POST-only hook architectures still have).
- **Project isolation** — `project_id` auto-detected from env → git rev-parse → cwd
  basename → "default". 60s TTL cache, `ALL_PROJECTS="all"` sentinel for cross-project
  queries. Filters surface a `project_mismatch` counter in `recall_explain`. Eliminates
  cross-project pollution in shared DBs.
- **Verify-with-code layer** — `recall(verify=True, verify_drop=True)` checks each
  result's chunk content against the current `source_path` on disk. Achieves true
  100% accuracy on code-backed memories. Three-state verdict (True/False/None)
  preserves synthesis memories that have no source.
- **Per-stage retrieval trace** — `_invariant_skips` counter with 5 keys
  (stale_verification, orphaned, version_skew, project_mismatch, verification_failed)
  surfaced via `get_stats()`. Sophisticated `recall_explain()` shows what each filter
  dropped at each stage.
- **4 new memory tools** — `memory_get`, `memory_delete`, `memory_stats`,
  `memory_explain`. Auto-approve list opens read-only ops; `memory_delete` deliberately
  requires confirmation (destructive).
- **Eager BGE warmup at boot** — `subsystem_init` calls embedder once on init when
  `eager_warmup=True`. Replaces "first recall is cold" with "first recall is hot"
  on every process start.
- **Healer auto-start** — `SubsystemFactory` starts the background drift / orphan /
  snapshot daemon when `enable_healer=True`. New `MemoryConfig` flags + env vars:
  `PREDACORE_MEMORY_ENABLE_HEALER`, `_SCAN_SECRETS`, `_EAGER_WARMUP`.
- **Ingress secret scan** — `store()` refuses content matching API-key/credential
  patterns; increments `_safety_stats`. Defense-in-depth on top of file ignores.
- **3 new memory modules** — `chunker.py` (AST/markdown/brace/window strategies),
  `safety.py` (secret scanner + `MemoryIgnore`), `healer.py` (background daemon),
  `project_id.py` (auto-detection helper).
- **Rich tool descriptions** — `memory_store` 56→1540 chars, `memory_recall`
  36→1347 chars. Encodes WHEN-TO-CALL / WHEN-NOT / quality rules so agents use
  memory deliberately, not reflexively.
- **`agents/default/MEMORY.md`** updated with dual-layer model: passive auto-context
  + active memory tools, "code is canonical, memory is for synthesis".

### Changed
- `predacore.memory` exports 9 symbols (was 3) — adds `Healer`, `MemoryIgnore`,
  `scan_for_secrets`, `chunk_text`, `safe_read_text`, `is_sensitive_path`.
- `recall_explain()` rewritten with sophisticated per-stage trace (replaces v1).
- `auto_approve_tools` extended for read-only memory ops.

### Tests
- **+200 deterministic tests** (3.69s) across 8 new test files: chunker / safety /
  project_id (70), store augmentations (34), tool handlers (28), subsystem +
  Healer wiring (27), e2e auto-trigger round-trips (14), verify-with-code (16),
  infrastructure smoke (11).
- **+5 LLM-gated tests** (Gemini Flash behavioral, `--real` flag).
- `pytest-timeout = 60s` global cap; `asyncio_mode = "auto"`.

### Packaging
- `predacore` 1.1.1 → 1.2.0 (Python-only — `predacore_core` Rust wheel unchanged at 1.1.1).

## [1.1.1] - 2026-04-24

**Phase 1 memory upgrade — HNSW at scale, v2 schema, trust-weighted ranking.**

### Added
- **Schema v2** with 14 invariant columns (`trust_source`, `verification_state`, `embedding_version`, `chunker_version`, `anchor_hash`, `content_hash`, `parent_id`, `chunk_ordinal`, `superseded_at`, `superseded_by`, `last_verified_at`, `source_blob_sha`, `source_mtime`, `decay_score`). Idempotent v1 → v2 migrator runs on first open; existing rows backfill safely.
- **Trust-weighted retrieval ranking** — recall score multiplied by `user_corrected=1.00`, `code_extracted=0.95`, `user_stated=0.90`, `claude_inferred=0.60` (alongside time decay and confidence).
- **Supersede API** — `store(supersedes=[...])` atomically replaces old rows; `recall(show_superseded=False)` hides them by default. Clean way to correct/update memories without losing history.
- **HNSW vector index** — `_HnswVectorIndex` opt-in via `PREDACORE_USE_HNSW=1`. Rust-backed via `hnsw_rs 0.3`, tombstone-based deletes, shared interface with `_NumpyVectorIndex`. O(log n) cosine ANN search instead of O(n) linear scan. Tuned for ~99.9% recall at 1M vectors: `M=32`, `ef_construction=400`, `ef_search=400`, `MAX_VECTORS=1,000,000`.
- **Vector index persistence** — `.npz` cache of numpy backend survives daemon restarts; safe-skipped on row-count or embedding-version drift.
- **Python 3.13** classifier added.
- **`predacore --version`** flag.

### Changed
- **Context budget** raised 36k → 80k tokens (model-agnostic).
- **Rust embedding** `MAX_SEQ_LEN` bumped 256 → 512 for longer memory content.
- **Dependency pin** tightened: `predacore_core>=1.1.1`.

### Packaging
- `predacore_core` 0.1.2 → 1.1.1 (coordinated bump, reset series).
- `predacore` 0.1.5 → 1.1.1.
- Cross-platform wheels on PyPI: linux-x86_64, macOS universal2, windows-x86_64, sdist.

### Tests
- 51 new schema-migration tests covering migration, supersede, recall filters, trust ranking, context budget, vector cache, HNSW semantics.
- Fixed pre-existing asyncio flake in `test_memory.py` (replaced deprecated `get_event_loop` with `asyncio.run`).
- **139 tests passing** locally.

### Live migration
- Ran successfully on the 80-row production DB: zero data loss, schema jumped 17 → 33 columns, legacy rows defaulted to `trust_source=claude_inferred` (conservative 0.60× multiplier) and `verification_state=unverified` (healer processes over time).

## [0.1.1 – 0.1.5] - 2026-04-18 to 2026-04-21

Rapid iteration on the v0 series — see individual GitHub release notes at
<https://github.com/rockshub1/predacore/releases> for per-version details.

Highlights:
- `0.1.5`: live daemon status/doctor · persist start flags · Gemini thought_signature fix
- `0.1.4`: provider-owned tool turns · response cache · 2-mode simplification
- `0.1.3`: browser_bridge reliability · Gemini cache telemetry
- `0.1.2`: Rust kernel rename (`jarvis_core` → `predacore_core`), multi-platform wheel publishing

## [0.1.0] - 2026-04-15

### 🎉 Initial Public Release

**Core Framework**
- Chat loop with streaming and non-streaming LLM support
- 40+ tool handlers with automatic dispatch and middleware
- Tool pipeline execution (sequential and parallel)
- Circuit breaker pattern for tool resilience

**Memory System**
- 5-layer retrieval: preferences → entities → semantic → fuzzy → episodes
- Rust-powered hybrid search (SIMD cosine + BM25 + trigram fuzzy)
- Automatic memory consolidation and deduplication
- Entity extraction with relation classification
- Token-budgeted context injection
- **95.7% R@5 on LongMemEval** — state-of-the-art

**Identity System**
- Self-evolving AI personality (SOUL.md, IDENTITY.md)
- User modeling (USER.md)
- Belief crystallization (BELIEFS.md)
- Reflection cycles and journaling
- Heartbeat-driven background processing

**LLM Providers**
- Anthropic (Claude), OpenAI (GPT-4), Google (Gemini)
- OpenRouter (100+ models)
- In-house SDK — zero vendor dependencies
- Automatic failover and circuit breakers

**Tools**
- Code execution (Python, Node, Go, Rust, Java, C++, Ruby, PHP, R, Julia, Kotlin, TypeScript)
- macOS desktop automation (PyObjC native)
- Chrome browser control (CDP protocol)
- Android device control (ADB)
- File system operations
- Git operations with semantic code search
- Web search and deep research
- Memory store and recall
- MCTS strategic planning
- Multi-agent orchestration
- Voice notes and TTS
- Image generation
- PDF reading
- Diagram generation (Mermaid)
- Scheduled tasks (cron)

**Channels**
- CLI with rich terminal UI
- Telegram bot
- Discord bot
- WhatsApp (via Twilio)
- Webchat widget

**Security**
- Ethical Governance Module (EGM)
- JWT authentication
- API key management
- Secret auto-redaction
- Docker sandboxed execution
- Trust levels (YOLO / Normal / Paranoid)

**Infrastructure**
- Docker Compose deployment
- Kubernetes manifests
- Grafana dashboards
- Prometheus metrics

**Benchmarks**
- LongMemEval: 0.9574 R@5 (SOTA-class retrieval (within 3 points of the current leader))
- Full evaluation harness for ALE-bench, SWE-bench
