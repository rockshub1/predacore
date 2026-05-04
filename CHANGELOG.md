# Changelog

All notable changes to PredaCore will be documented in this file.

## [1.5.4] - 2026-05-04

**Patch — fix browser bridge resource leak + actionable error when
Chrome isn't running.**

### Fixed
- **aiohttp ClientSession leak in `_ChromeCDP.connect()`** — when the
  tool dispatcher's outer 5s adaptive timeout cancelled an in-flight
  CDP connect attempt, `aiohttp`'s connection pool was leaking sockets
  to localhost:9222. Symptom on shutdown:
    `Unclosed client session ... ConnectionKey(host='localhost', port=9222 ...)`
  Fix: wrap the entire connect flow in a `try/finally` that calls
  `_cleanup()` on any abnormal exit including `asyncio.CancelledError`.
  The session is only kept alive on the success path (where the
  websocket needs it).
- **Cryptic "browser_control timed out after 5.0s" message** — when
  Chrome wasn't running with the debug port, the actual cause
  ("CDP not available on :9222") was logged at INFO level (invisible
  to users) while the dispatcher's WARNING-level timeout took the spotlight.
  Fix: `handle_browser_control` now returns a structured error with
  `code: "chrome_not_reachable"` and a literal `suggested_command:`
  field showing exactly what to type:
    `open -a 'Google Chrome' --args --remote-debugging-port=9222`

### Added
- **`ChromeNotRunningError`** — public typed exception in
  `operators.browser_bridge` for callers that want to short-circuit
  the auto-launch flow with a clear error.

## [1.5.3] - 2026-05-03

**Patch — Codex finally routes to the right host. ChatGPT-OAuth chats
now actually work end-to-end.**

### Fixed
- **Codex endpoint** — v1.5.0/1/2 sent Codex requests to
  `https://api.openai.com/v1/responses`. That endpoint expects API-tier
  billing keys and 401s ChatGPT-OAuth subscription tokens regardless of
  headers. The official Codex CLI hits a different host:
    `https://chatgpt.com/backend-api/codex/responses`
  which is the only endpoint that accepts subscription OAuth tokens.
  v1.5.3 switches Codex to that URL.
  (The token's JWT `aud` claim says `api.openai.com/v1` — that's
  misleading; the actual server-side acceptance is at chatgpt.com.)

This release bundles the v1.5.1 (OAuth redirect_uri) and v1.5.2
(JWT account_id + headers) fixes — none of those landed on PyPI.
v1.5.0 → v1.5.3 is the minimum upgrade users need to actually use Codex.

## [1.5.2] - 2026-05-03

**Patch — Codex requests now actually return 200, not 401.**

v1.5.1 fixed the OAuth dance (login completes, grant saved). But every
subsequent `/v1/responses` call returned HTTP 401 because we sent only
`Authorization: Bearer` and OpenAI's Codex backend requires more.

### Fixed
- **Codex 401 on every request** — Codex's `/v1/responses` requires three
  additional headers alongside the bearer token:
    * `chatgpt-account-id: <id>` — the workspace this OAuth token is
      bound to. Comes from the JWT's `https://api.openai.com/auth`
      claim (`chatgpt_account_id` field).
    * `OpenAI-Beta: chatgpt-codex` — routes to the Codex-tuned endpoint
      that ChatGPT-OAuth tokens are authorized for (vs the regular API
      tier endpoint, which they aren't).
    * `Originator: predacore_cli` — telemetry/rate-limit bucket marker
      that mirrors what the official Codex CLI emits.
  Without these, the strict matcher 401s. v1.5.0/1.5.1 hit this on every
  Codex chat — the `unknown_error` from v1.5.0 was OAuth, the 401 from
  v1.5.1 was post-auth API.
- **JWT account_id extraction at login time** —
  `OAuthFlow._grant_from_token_response` now decodes the access_token
  JWT and pulls `chatgpt_account_id` out of the auth claim, saving it
  to `OAuthGrant.account_id`. Future logins show the correct id in the
  "Authorized as ..." message instead of "(account_id unavailable)".
- **Backward-compat for v1.5.1 grants** — if the on-disk grant has
  `account_id=""` (saved before this fix), `chat()` re-derives it on
  the fly by decoding the JWT. **No re-login required after upgrade.**

### Known followups
- Codex tokens still come back without a refresh_token at the current
  empty-scopes setting. Adding `offline_access` to the scope list might
  fix it; v1.5.3 will probe. For now, re-run `predacore login
  openai-codex` every ~10 days.

## [1.5.1] - 2026-05-02

**Patch — fix Codex OAuth redirect_uri so `predacore login openai-codex`
actually completes against OpenAI's strict matcher.**

### Fixed
- **Codex OAuth `unknown_error`** — v1.5.0 picked an ephemeral free port
  and sent `redirect_uri=http://127.0.0.1:<random>/callback`. OpenAI's
  strict OAuth matcher requires the EXACT redirect_uri registered for the
  client_id. OpenCode's public PKCE client `app_EMoamEEZ73f0CkXaXp7hrann`
  registered `http://localhost:1455/auth/callback` — so OpenAI was
  rejecting our flow with a generic `unknown_error` (despite the user being
  signed in to ChatGPT correctly).
  Fix: pin Codex's redirect_uri to `http://localhost:1455/auth/callback`
  via new `OAuthFlowConfig.redirect_{host,bind_host,port,path}` fields.
  Other providers' flows are unchanged (they keep the ephemeral-port
  default for clients that allow loopback flexibility).

### Known issues / followups
- Codex tokens come back **without a refresh_token** with our current scope
  set. v1.5.2 will add `offline_access` to the scopes — meanwhile, tokens
  expire after ~10 days and require re-running `predacore login openai-codex`.

### Tests
- All 23 OAuth tests still pass; the fix is plumbing through
  `OAuthFlowConfig` with backward-compatible defaults.

## [1.5.0] - 2026-05-02

**Unified native tool calling across every provider — fixes a confirmed
HTTP-400 bug in Gemini-3 / 2.5-thinking, migrates Codex to the OpenAI
Responses API, removes the silent text-tool fallback path, and adds 4
new OpenAI-compatible providers (Kimi K2, Qwen 3, Hyperbolic,
Perplexity Sonar).**

### Fixed (the load-bearing bug)
- **Gemini ``thoughtSignature`` loss** — ``gemini.py:append_assistant_turn``
  used to rebuild ``functionCall`` parts from typed fields when
  ``provider_extras["content_parts"]`` was missing, dropping the
  ``thoughtSignature`` field. Gemini-3 and 2.5-thinking models then
  returned HTTP 400 (*"Function call is missing a thought_signature"*)
  on the next turn. The fix: refuse rather than reconstruct, and stash
  ``parts[]`` from every parsed response (not just functionCall-bearing
  ones) so the signature always survives.
- **Gemini non-deterministic tool-call IDs** — was
  ``f"gemini_{i}_{uuid.uuid4().hex[:8]}"`` (regenerated every call).
  Now derived from a SHA-1 of ``(turn_idx, position, name, args)``,
  so re-serialization is idempotent. Tool results synthesized later
  by ``(name, position)`` reliably link back to their call.

### Added
- **Per-provider wire validators** (``openai_validator``,
  ``gemini_validator``) mirror the existing
  ``message_validator.repair_tool_flow`` pattern. Each ``LLMProvider``
  now has a ``repair_messages(messages)`` hook that runs before
  serialization. Auto-repairs orphan tool calls/results, drops
  invalid links, strips empty ``tool_calls: []`` arrays.
- **``PREDACORE_REPAIR_TOOL_FLOW`` env toggle** (LiteLLM-style):
  ``repair`` (default) auto-fixes; ``strict`` raises
  ``ToolFlowInvariantError``; ``off`` skips validation entirely.
  Same env var controls all three validators uniformly.
- **OpenAI Responses API adapter** (``openai_responses.py``) — POST
  ``/v1/responses`` with typed ``input[]`` / ``output[]`` items
  (``function_call`` + ``function_call_output``). Closer to Anthropic
  content-blocks structurally; reasoning items are surfaced.
- **4 new OpenAI-compatible providers**:
  - **Kimi K2 (Moonshot)** — ``MOONSHOT_API_KEY``,
    ``api.moonshot.cn/v1``. Strong open-weight reasoning + tool use.
  - **Qwen 3 (Alibaba DashScope, international)** —
    ``DASHSCOPE_API_KEY``, ``dashscope-intl.aliyuncs.com``. 235B MoE.
  - **Hyperbolic** — ``HYPERBOLIC_API_KEY``,
    ``api.hyperbolic.xyz/v1``. Cheap Llama / DeepSeek / Qwen hosting.
  - **Perplexity Sonar** — ``PERPLEXITY_API_KEY``,
    ``api.perplexity.ai``. Built-in web search.

### Changed
- **OpenAI Codex migrated to the Responses API.** The provider now
  POSTs to ``/v1/responses`` instead of ``/v1/chat/completions``
  (Chat Completions is being phased out for ChatGPT-OAuth tokens).
  Inherits ``OpenAIResponsesProvider`` for the wire format; only
  overrides ``chat()`` for OAuth bearer auth + 401 refresh-and-retry +
  403 "subscription revoked" message.
- **Anthropic ``repair_tool_flow`` call site** — moved into the new
  ``repair_messages`` override on ``AnthropicProvider``. No behavior
  change for users; uniform across providers now.

### Removed
- **Silent text-tool fallback path on every API provider.** When a
  model rejected native tools, predacore used to embed tool definitions
  as text in the system prompt and parse ``<tool_call>`` XML out of the
  response. v1.5.0 surfaces a clear ``BadRequestError`` instead — naming
  the model and pointing at tool-tuned alternatives. Modern provider
  models all support native tools; the fallback was papering over
  misconfigurations more often than it was useful. ``text_tool_adapter.py``
  remains as a private helper for ``gemini_cli`` (which speaks plain
  text only — text tool calling is its inherent mode, not a fallback).

### OpenRouter fidelity caveat (newly documented)
OpenRouter normalizes every underlying provider's response to OpenAI
Chat Completions, which means picking ``anthropic/claude-opus-4-7`` via
OpenRouter *loses* Anthropic's thinking signatures, and picking
``google/gemini-3-pro`` *loses* ``thoughtSignature``. For maximum
fidelity (thinking-block preservation, native tool-use validators per
provider), use direct providers (``--model anthropic``,
``--model gemini``). For breadth + one API key, OpenRouter is great —
predacore can't recover signatures the aggregator strips.

### Tests
- 18 new ``test_gemini_validator.py`` tests
- 16 new ``test_openai_validator.py`` tests
- 13 new ``test_openai_responses.py`` tests
- 8 new ``test_router_new_providers.py`` tests
- 4 new tests in ``test_message_validator.py`` for the env toggle
- 3 new tests in ``test_phase_a_tool_turns.py`` covering signature
  preservation, ID stability, and the missing-signature refusal
- ~62 new tests total. Full suite: 2080 passing (39 environment skips).

### Migration impact
- **No external API changes.** ``LLMProvider.repair_messages()`` is
  additive with a no-op default; ``Message.provider_extras`` keys are
  opaque and forward-compatible.
- **If you depended on the silent text fallback** for an old model
  without native tools: switch to a tool-tuned model. The 4 new
  providers (Kimi, Qwen, Hyperbolic, Perplexity) all support native
  tools; so do every current Anthropic / OpenAI / Gemini model.
- **Codex tokens that predate Responses-API support**: the existing
  refresh-and-retry path covers token-refresh edge cases. If a Codex
  request returns persistent 4xx after refresh, run ``predacore login
  openai-codex`` to re-authorize.

## [1.4.2] - 2026-05-02

**Patch — adds the ``predacore upgrade`` CLI command + tightens the
``predacore_core`` dep floor so pipx upgrades pull the right Rust kernel.**

### Added
- **``predacore upgrade``** — new CLI subcommand that runs
  ``pip install -U predacore predacore_core`` against the current Python
  environment (``sys.executable``). One command refreshes both packages.
  Solves the pipx ergonomics issue where ``pipx upgrade predacore`` only
  bumps the top-level package and leaves the Rust kernel stale because
  the floor still satisfies. Flags: ``--pre`` (allow pre-releases),
  ``--dry-run`` (print the pip command without executing).

### Changed
- **Dep floor**: ``predacore_core>=1.1.1`` → ``predacore_core>=1.2.0`` in
  ``pyproject.toml``. Now ``pipx upgrade predacore`` (and any
  ``pip install -U predacore``) also picks up the new Rust kernel
  automatically — the old floor satisfied 1.1.1 so transitive resolution
  skipped the upgrade. We now genuinely depend on ``predacore_core 1.2.0``
  (T5c batched embed throughput) so the floor reflects that.

### Tests
- 4 new tests in ``test_cli_upgrade.py``: dry-run skips subprocess,
  real run calls pip with the right shape, ``--pre`` injects the flag,
  pip failures exit with the proper return code.

### Migration impact
- **Existing pipx users**: ``predacore upgrade`` (or ``pipx upgrade
  predacore``) now correctly pulls the latest of both packages.
- **Anyone with ``predacore_core 1.1.1`` pinned** in their constraint
  file will need to relax the pin or upgrade to 1.2.0 — but 1.2.0 is
  fully backward-compatible (additive throughput improvement only).

## [1.4.1] - 2026-05-02

**Patch — fixes the version-skew that left v1.4.0's ``predacore_core``
stuck at 1.1.1 on PyPI even though ``Cargo.toml`` said 1.2.0.**

### Fixed
- ``src/predacore_core_crate/pyproject.toml`` was missed in the v1.4.0
  version bump — it still had ``[project] version = "1.1.1"`` while
  ``Cargo.toml`` correctly said 1.2.0. **Maturin uses pyproject.toml's
  version for the wheel filename**, so v1.4.0's CI built and published
  ``predacore_core-1.1.1.tar.gz`` (a republish of the existing 1.1.1)
  instead of 1.2.0. Bumped pyproject.toml to 1.2.0; this v1.4.1 tag
  triggers a fresh build that finally lands the real T5c batched-embed
  ``predacore_core 1.2.0`` on PyPI.
- Added a ``verify-versions`` CI job in ``.github/workflows/build-wheels.yml``
  that fails the workflow with a clear error if ``Cargo.toml`` and
  ``pyproject.toml`` disagree on the version. All four wheel-build jobs
  and the publish job now ``needs: verify-versions``, so a future skew
  is caught BEFORE any wheel is built. (Cargo.lock is gitignored so it
  doesn't matter — the only sources of truth are these two files.)

### Added (T11.5 — benchmark depth)
- **Verifier-tier instrumentation** in ``_verify_chunk_against_source``:
  bumps a contextvar-scoped counter on each tier (blob_sha / substring /
  AST-symbol / line-anchor / failed). The benchmark arms a counter per
  pass and prints the breakdown so we can see which tier is firing.
- **Production query set: 30 → 62** queries. Added 32 v1.4.0-specific
  queries covering: browser (selector cache, vision plumbing, canvas
  detector), memory (T7 verifiers, healer rate brake, batched store),
  channels (streaming buffer, scaffold, /channel UX, Twilio/Matrix/
  Mastodon/Bluesky/Mattermost/Line/IRC/Google Chat adapters), tools
  (image_gen routing), and core (workspace context block, bootstrap).
- **Benchmark category coverage**: now 12 categories (was 11) with the
  new ``browser`` bucket.

### Benchmark results on v1.4.0 codebase

|                | 30-query (v1.4.0)  | 62-query (T11.5)   |
| -------------- | ------------------ | ------------------ |
| R@5            | 0.9667             | **0.9839**         |
| R@10           | 0.9667             | **0.9839**         |
| R@20           | 1.0000             | 1.0000             |
| Indexing       | 577s / 1841 chunks | 618s / 1844 chunks |

Verifier tier breakdown (clean corpus, 1860 chunks verified):

| Tier                                  | Count | Share  |
| ------------------------------------- | ----- | ------ |
| ``tier_0_blob_sha`` (T7, file unchanged) | 1858  | 99.9%  |
| ``tier_2_line_anchor``                | 2     | 0.1%   |
| (others)                              | 0     | 0%     |

Tier-0 firing 99.9% on a freshly bulk-indexed corpus is exactly the T7
contract: when nothing has changed on disk, every chunk verifies via a
single hash op (mtime-keyed cache). The 2 tier-2 fall-throughs are rows
where ``source_blob_sha`` wasn't populated by the chunker — investigate
in T11.6 but not release-blocking.

### Migration impact
- **Users with v1.4.0 already installed**: ``pip install -U predacore_core``
  to pick up 1.2.0 (T5c batched embedding). predacore Python is unchanged
  apart from the ``__version__`` bump and the benchmark instrumentation.
- The ``predacore_core>=1.1.1`` requirement in pyproject.toml is unchanged,
  so v1.4.0 users on the old 1.1.1 wheel keep working — they just don't
  get the T5c speedup until they upgrade.

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
