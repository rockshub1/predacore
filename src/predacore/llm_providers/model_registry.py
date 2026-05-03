"""
Central catalog of LLM models per provider — the single source of truth.

Used by:
  * ``cli._run_setup`` — model picker step in the onboarding wizard
  * ``/model`` slash command — tab-complete + suggestion list
  * Documentation generators

For each provider, ``MODELS`` lists ``(model_id, description, tier)``
tuples. The **first entry is the recommended default** for that
provider — picked to balance cost / quality / availability.

``tier`` is a coarse capability label used for filtering / display:

  * ``flagship``  — top-quality, expensive
  * ``balanced``  — best quality/cost ratio
  * ``fast``      — low-latency, cheap
  * ``reasoning`` — extended-thinking / chain-of-thought specialist
  * ``coder``     — code-specialty
  * ``vision``    — image understanding / generation
  * ``embed``     — embedding/retrieval
  * ``free``      — generous free tier or completely free

May 2026 snapshot — sourced from each provider's official model docs
on 2026-05-02. Update as providers ship new models.

Verification caveats (research summary, 2026-05-02):
  * xAI Grok: ``grok-4.3`` exact API string may be ``grok-4-3`` or
    ``grok-4-3-latest`` depending on rollout — see provider docs.
  * Hyperbolic: docs site is JS-only; the IDs below are cross-referenced
    from third-party citations + the public model gallery. Confirm
    against ``app.hyperbolic.ai/models`` before pinning to a release.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One model entry in the registry."""
    id: str
    description: str
    tier: str  # flagship / balanced / fast / reasoning / coder / vision / embed / free


# Recommended default per provider is the FIRST entry in each list.
# Order subsequent entries by descending typical preference.
MODELS: dict[str, list[ModelInfo]] = {
    # ── Anthropic Claude ────────────────────────────────────────────
    # Source: platform.claude.com/docs/en/docs/about-claude/models
    "anthropic": [
        ModelInfo("claude-opus-4-7",            "Opus 4.7 — top reasoning, agentic coding", "flagship"),
        ModelInfo("claude-sonnet-4-6",          "Sonnet 4.6 — best speed/intelligence",     "balanced"),
        ModelInfo("claude-haiku-4-5",           "Haiku 4.5 — fastest near-frontier",        "fast"),
        ModelInfo("claude-haiku-4-5-20251001",  "Haiku 4.5 pinned snapshot",                "fast"),
        ModelInfo("claude-opus-4-6",            "Opus 4.6 — prior flagship",                "flagship"),
        ModelInfo("claude-opus-4-5-20251101",   "Opus 4.5 snapshot",                        "flagship"),
        ModelInfo("claude-sonnet-4-5-20250929", "Sonnet 4.5 snapshot",                      "balanced"),
        ModelInfo("claude-opus-4-1-20250805",   "Opus 4.1 snapshot",                        "flagship"),
    ],

    # ── OpenAI (Chat Completions + Responses API) ───────────────────
    # Source: developers.openai.com/api/docs/models
    "openai": [
        ModelInfo("gpt-5.5",                "GPT-5.5 — frontier coding + pro work",  "flagship"),
        ModelInfo("gpt-5.5-pro",            "GPT-5.5 Pro — Responses-only, max precision", "reasoning"),
        ModelInfo("gpt-5.4",                "GPT-5.4 — prior flagship",              "flagship"),
        ModelInfo("gpt-5.4-mini",           "GPT-5.4 mini — cheap subagent driver",  "fast"),
        ModelInfo("gpt-5.4-nano",           "GPT-5.4 nano — cheapest reasoning",     "fast"),
        ModelInfo("gpt-5.4-thinking",       "GPT-5.4 Thinking — extended CoT",       "reasoning"),
        ModelInfo("gpt-5.3-instant",        "GPT-5.3 Instant — low-latency chat",    "fast"),
        ModelInfo("o4-mini",                "o4-mini — fast multimodal reasoning",   "reasoning"),
        ModelInfo("o4-mini-deep-research",  "o4-mini deep research variant",         "reasoning"),
        ModelInfo("o3",                     "o3 — heavy math/science reasoning",     "reasoning"),
        ModelInfo("gpt-4.1",                "GPT-4.1 — 1M-token instruction model",  "balanced"),
        ModelInfo("gpt-4.1-mini",           "GPT-4.1 mini — cheap long-context",     "fast"),
        ModelInfo("gpt-4.1-nano",           "GPT-4.1 nano — tiny long-context",      "fast"),
        ModelInfo("gpt-image-2",            "GPT Image 2 — SOTA image gen/edit",     "vision"),
        ModelInfo("gpt-realtime-1.5",       "Realtime speech-to-speech",             "fast"),
        ModelInfo("gpt-4o-transcribe",      "GPT-4o transcription (STT)",            "fast"),
        ModelInfo("gpt-4o-mini-transcribe", "GPT-4o mini transcription",             "fast"),
        ModelInfo("text-embedding-3-large", "Text embed large — retrieval",          "embed"),
        ModelInfo("text-embedding-3-small", "Text embed small — cheap",              "embed"),
    ],

    # ── OpenAI Codex (ChatGPT-OAuth, Responses API) ─────────────────
    # Source: developers.openai.com/codex/models
    # gpt-5.5 is ChatGPT-login-only; not callable with a normal API key.
    "openai-codex": [
        ModelInfo("gpt-5.5",             "Recommended Codex driver (ChatGPT login)", "flagship"),
        ModelInfo("gpt-5.4",             "GPT-5.4 — flagship coding",                "flagship"),
        ModelInfo("gpt-5.4-mini",        "Fast tier for Codex",                      "fast"),
        ModelInfo("gpt-5.3-codex",       "Codex-tuned coder",                        "coder"),
        ModelInfo("gpt-5.3-codex-spark", "Real-time iterative coder (preview)",      "coder"),
        ModelInfo("gpt-5.2",             "Prior general Codex model",                "balanced"),
    ],

    # ── Google Gemini ───────────────────────────────────────────────
    # Source: ai.google.dev/gemini-api/docs/models
    "gemini": [
        ModelInfo("gemini-3.1-pro-preview",        "Gemini 3.1 Pro — top reasoning + agentic", "flagship"),
        ModelInfo("gemini-3-flash-preview",        "Gemini 3 Flash — frontier at low cost",    "balanced"),
        ModelInfo("gemini-3.1-flash-lite-preview", "3.1 Flash Lite — cheapest 3.x",            "fast"),
        ModelInfo("gemini-2.5-pro",                "Gemini 2.5 Pro — stable deep reasoning",   "flagship"),
        ModelInfo("gemini-2.5-flash",              "Gemini 2.5 Flash — best price/perf",       "balanced"),
        ModelInfo("gemini-2.5-flash-lite",         "2.5 Flash Lite — cheapest stable",         "fast"),
        ModelInfo("gemini-3.1-flash-image-preview", "3.1 Flash image gen/edit",                "vision"),
        ModelInfo("gemini-3-pro-image-preview",    "3 Pro 4K image gen",                       "vision"),
        ModelInfo("gemini-2.5-flash-image",        "2.5 Flash image gen",                      "vision"),
        ModelInfo("gemini-2.5-computer-use-preview-10-2025", "Computer-use UI agent",          "reasoning"),
        ModelInfo("gemini-deep-research-preview-04-2026",     "Deep research (autonomous)",    "reasoning"),
        ModelInfo("gemini-deep-research-max-preview-04-2026", "Deep research max",             "reasoning"),
        ModelInfo("gemini-embedding-2",            "Multimodal embeddings",                    "embed"),
        ModelInfo("gemini-embedding-001",          "Text embeddings for RAG",                  "embed"),
    ],

    # ── Gemini CLI (uses local `gemini` binary, free) ───────────────
    "gemini-cli": [
        ModelInfo("gemini-2.5-flash",      "2.5 Flash — fast (default)", "free"),
        ModelInfo("gemini-2.5-pro",        "2.5 Pro — highest quality",  "free"),
        ModelInfo("gemini-2.5-flash-lite", "2.5 Flash Lite — fastest",   "free"),
    ],

    # ── xAI Grok ────────────────────────────────────────────────────
    # Source: docs.x.ai/developers/models
    # NOTE: 4.3 / 4.20 exact API IDs may be `grok-4-3` style after rollout.
    "xai": [
        ModelInfo("grok-4.3",                    "Grok 4.3 — flagship + fastest",       "flagship"),
        ModelInfo("grok-4.20-reasoning",         "Grok 4.20 reasoning mode",            "reasoning"),
        ModelInfo("grok-4.20-non-reasoning",     "Grok 4.20 no-think",                  "balanced"),
        ModelInfo("grok-4.1-fast-reasoning",     "Grok 4.1 Fast w/ CoT (2M ctx)",       "reasoning"),
        ModelInfo("grok-4.1-fast-non-reasoning", "Grok 4.1 Fast no-think (2M ctx)",     "fast"),
        ModelInfo("grok-4-0709",                 "Grok 4 (July 2025 snapshot, 128K)",   "flagship"),
        ModelInfo("grok-4-fast-reasoning",       "Grok 4 Fast reasoning",               "reasoning"),
        ModelInfo("grok-4-fast-non-reasoning",   "Grok 4 Fast no-think",                "fast"),
        ModelInfo("grok-code-fast-1",            "Grok Code Fast 1 — cheap coder",      "coder"),
        ModelInfo("grok-3",                      "Grok 3 — prior gen",                  "balanced"),
        ModelInfo("grok-3-mini",                 "Grok 3 Mini — cheap",                 "fast"),
    ],

    # ── DeepSeek ────────────────────────────────────────────────────
    # Source: api-docs.deepseek.com/quick_start/pricing
    # V3.x retires 2026-07-24; V4 is current.
    "deepseek": [
        ModelInfo("deepseek-v4-pro",   "V4 Pro — 1.6T MoE reasoning, 1M ctx", "flagship"),
        ModelInfo("deepseek-v4-flash", "V4 Flash — 284B MoE, fast, 1M ctx",   "fast"),
        # Legacy aliases (retire 2026-07-24)
        ModelInfo("deepseek-chat",     "V3.x chat (legacy, retiring 2026-07-24)",     "balanced"),
        ModelInfo("deepseek-reasoner", "V3.x reasoner (legacy, retiring 2026-07-24)", "reasoning"),
    ],

    # ── Mistral ─────────────────────────────────────────────────────
    # Source: docs.mistral.ai/getting-started/models/models_overview/
    "mistral": [
        ModelInfo("mistral-large-latest",      "Mistral Large 3 — flagship multimodal", "flagship"),
        ModelInfo("mistral-medium-latest",     "Medium 3.5 — agentic + coding",         "balanced"),
        ModelInfo("mistral-medium-3.1",        "Medium 3.1 (v25.08)",                   "balanced"),
        ModelInfo("mistral-small-latest",      "Small 4 (v26.03)",                      "fast"),
        ModelInfo("magistral-medium-1.2",      "Magistral 1.2 — reasoning",             "reasoning"),
        ModelInfo("ministral-3-14b",           "Ministral 3 14B edge",                  "fast"),
        ModelInfo("ministral-3-8b",            "Ministral 3 8B edge",                   "fast"),
        ModelInfo("ministral-3-3b",            "Ministral 3 3B edge",                   "fast"),
        ModelInfo("devstral-2",                "Devstral 2 — frontier code agent",      "coder"),
        ModelInfo("codestral-latest",          "Codestral (v25.08)",                    "coder"),
        ModelInfo("codestral-embed",           "Codestral embed — code retrieval",      "embed"),
        ModelInfo("voxtral-tts",               "Voxtral TTS",                           "fast"),
        ModelInfo("voxtral-small",             "Voxtral Small (audio)",                 "fast"),
        ModelInfo("voxtral-mini-transcribe-2", "Voxtral mini STT",                      "fast"),
        ModelInfo("ocr-3",                     "OCR 3",                                 "vision"),
        ModelInfo("mistral-moderation-2",      "Moderation 2",                          "fast"),
        ModelInfo("mistral-embed",             "Text embeddings",                       "embed"),
    ],

    # ── Groq (hosted Llama / GPT-OSS / Qwen / Whisper) ──────────────
    # Source: console.groq.com/docs/models
    "groq": [
        ModelInfo("openai/gpt-oss-120b",                       "GPT-OSS 120B — flagship open",  "flagship"),
        ModelInfo("openai/gpt-oss-20b",                        "GPT-OSS 20B — cheap reasoner",  "fast"),
        ModelInfo("llama-3.3-70b-versatile",                   "Llama 3.3 70B versatile",       "balanced"),
        ModelInfo("llama-3.1-8b-instant",                      "Llama 3.1 8B instant",          "fast"),
        ModelInfo("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B",             "balanced"),
        ModelInfo("qwen/qwen3-32b",                            "Qwen3 32B multilingual",        "balanced"),
        ModelInfo("groq/compound",                             "Compound agentic system",       "reasoning"),
        ModelInfo("groq/compound-mini",                        "Compound mini agentic",         "fast"),
        ModelInfo("whisper-large-v3",                          "Whisper STT",                   "fast"),
        ModelInfo("whisper-large-v3-turbo",                    "Whisper turbo STT",             "fast"),
    ],

    # ── Cerebras (very fast inference, sparse roster) ───────────────
    # Source: inference-docs.cerebras.ai/models
    "cerebras": [
        ModelInfo("gpt-oss-120b",                   "GPT-OSS 120B @ ~3000 tok/s",  "flagship"),
        ModelInfo("qwen-3-235b-a22b-instruct-2507", "Qwen3 235B (preview)",        "flagship"),
        ModelInfo("zai-glm-4.7",                    "GLM-4.7 355B (preview)",      "flagship"),
        ModelInfo("llama3.1-8b",                    "Llama 3.1 8B @ ~2200 tok/s",  "fast"),
    ],

    # ── Together AI ─────────────────────────────────────────────────
    # Source: docs.together.ai/docs/serverless-models
    "together": [
        ModelInfo("Qwen/Qwen3.5-397B-A17B",                  "Qwen3.5 397B reasoning",   "flagship"),
        ModelInfo("deepseek-ai/DeepSeek-V4-Pro",             "DeepSeek V4 Pro",          "flagship"),
        ModelInfo("moonshotai/Kimi-K2.6",                    "Kimi K2.6 — long-form",    "balanced"),
        ModelInfo("meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama 3.3 70B Turbo",      "balanced"),
        ModelInfo("zai-org/GLM-5.1",                         "GLM 5.1 multimodal",       "balanced"),
        ModelInfo("Qwen/Qwen3.5-9B",                         "Qwen3.5 9B small",         "fast"),
        ModelInfo("Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "Qwen3 235B A22B",          "balanced"),
        ModelInfo("google/gemma-4-31B-it",                   "Gemma 4 31B IT",           "balanced"),
        ModelInfo("Qwen/Qwen3.6-Plus",                       "Qwen3.6 Plus 1M ctx",      "balanced"),
        ModelInfo("black-forest-labs/FLUX.1-schnell",        "FLUX.1 schnell image",     "vision"),
    ],

    # ── Fireworks ───────────────────────────────────────────────────
    # Source: fireworks.ai/models
    "fireworks": [
        ModelInfo("accounts/fireworks/models/deepseek-v4-pro",    "DeepSeek V4 Pro — 1M ctx", "flagship"),
        ModelInfo("accounts/fireworks/models/kimi-k2-6-instruct", "Kimi K2.6 — 262K",         "balanced"),
        ModelInfo("accounts/fireworks/models/minimax-m2-7",       "MiniMax M2.7 — 196K",      "balanced"),
        ModelInfo("accounts/fireworks/models/glm-5p1",            "GLM 5.1 — 202K",           "balanced"),
        ModelInfo("accounts/fireworks/models/qwen3-6-plus",       "Qwen3.6 Plus vision",      "vision"),
        ModelInfo("accounts/fireworks/models/kimi-k2-5",          "Kimi K2.5 — 262K",         "balanced"),
        ModelInfo("accounts/fireworks/models/deepseek-v3p2",      "DeepSeek V3.2 — 163K",     "balanced"),
        ModelInfo("accounts/fireworks/models/gemma-4-31b-it",     "Gemma 4 31B IT",           "balanced"),
        ModelInfo("accounts/fireworks/models/qwen3p5-397b-a17b",  "Qwen3.5 397B A17B vision", "flagship"),
        ModelInfo("accounts/fireworks/models/glm-5",              "GLM-5 — 202K",             "balanced"),
    ],

    # ── OpenRouter (top by weekly volume — ~300 total available) ────
    # Source: openrouter.ai/rankings
    "openrouter": [
        ModelInfo("moonshotai/kimi-k2.6-20260420",          "Kimi K2.6",                "flagship"),
        ModelInfo("anthropic/claude-4.6-sonnet-20260217",   "Claude Sonnet 4.6",        "balanced"),
        ModelInfo("anthropic/claude-4.7-opus-20260416",     "Claude Opus 4.7",          "flagship"),
        ModelInfo("google/gemini-3-flash-preview-20251217", "Gemini 3 Flash preview",   "balanced"),
        ModelInfo("openai/gpt-5.5",                         "GPT-5.5",                  "flagship"),
        ModelInfo("openai/gpt-5.4-mini",                    "GPT-5.4 mini",             "fast"),
        ModelInfo("deepseek/deepseek-v3.2-20251201",        "DeepSeek V3.2",            "balanced"),
        ModelInfo("deepseek/deepseek-v4-flash-20260423",    "DeepSeek V4 Flash",        "fast"),
        ModelInfo("x-ai/grok-4.1-fast",                     "Grok 4.1 Fast — 2M ctx",   "fast"),
        ModelInfo("stepfun/step-3.5-flash",                 "StepFun 3.5 Flash",        "fast"),
        ModelInfo("minimax/minimax-m2.7-20260318",          "MiniMax M2.7",             "balanced"),
        ModelInfo("tencent/hy3-preview-20260421:free",      "Tencent HY3 preview — free", "free"),
        ModelInfo("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (free tier)", "free"),
    ],

    # ── SambaNova ───────────────────────────────────────────────────
    # Source: docs.sambanova.ai/cloud/release-notes/overview
    "sambanova": [
        ModelInfo("gpt-oss-120b",                       "GPT-OSS 120B @ 700+ tok/s",   "flagship"),
        ModelInfo("DeepSeek-V3.1",                      "DeepSeek V3.1 hybrid",        "flagship"),
        ModelInfo("DeepSeek-V3.2",                      "DeepSeek V3.2",               "balanced"),
        ModelInfo("Meta-Llama-3.3-70B-Instruct",        "Llama 3.3 70B — 128K",        "balanced"),
        ModelInfo("Llama-4-Maverick-17B-128E-Instruct", "Llama 4 Maverick",            "balanced"),
        ModelInfo("Llama-4-Scout-17B-16E-Instruct",     "Llama 4 Scout (preview)",     "balanced"),
        ModelInfo("DeepSeek-R1-Distill-Llama-70B",      "R1-distill 70B reasoning",    "reasoning"),
        ModelInfo("QwQ-32B",                            "QwQ 32B reasoning",           "reasoning"),
        ModelInfo("Qwen2.5-72B-Instruct",               "Qwen 2.5 72B",                "balanced"),
    ],

    # ── NVIDIA NIM ──────────────────────────────────────────────────
    # Source: build.nvidia.com/models
    "nvidia": [
        ModelInfo("deepseek-ai/deepseek-v4-pro",   "DeepSeek V4 Pro on Blackwell", "flagship"),
        ModelInfo("deepseek-ai/deepseek-v4-flash", "DeepSeek V4 Flash 1M ctx",     "fast"),
        ModelInfo("deepseek-ai/deepseek-v3.2",     "DeepSeek V3.2",                "balanced"),
        ModelInfo("deepseek-ai/deepseek-v3.1",     "DeepSeek V3.1 hybrid",         "balanced"),
        ModelInfo("moonshotai/kimi-k2.6",          "Kimi K2.6",                    "balanced"),
        ModelInfo("minimax/minimax-m2.1",          "MiniMax M2.1",                 "balanced"),
        ModelInfo("zai-org/glm-4.7",               "GLM-4.7",                      "balanced"),
        ModelInfo("zai-org/glm-5",                 "GLM-5",                        "balanced"),
        ModelInfo("qwen/qwen3-235b-a22b",          "Qwen3 235B A22B",              "balanced"),
        ModelInfo("meta/llama-4-maverick",         "Llama 4 Maverick",             "balanced"),
    ],

    # ── Zhipu / GLM (BigModel) ──────────────────────────────────────
    # Source: docs.bigmodel.cn/cn/guide/start/model-overview
    "zhipu": [
        ModelInfo("glm-4.6",            "GLM-4.6 — 200K, advanced coder + agent", "flagship"),
        ModelInfo("glm-4.5-air",        "GLM-4.5 Air — cost-effective",           "balanced"),
        ModelInfo("glm-4.5-airx",       "GLM-4.5 AirX — fast inference",          "fast"),
        ModelInfo("glm-4-air",          "GLM-4 Air — cheap general",              "balanced"),
        ModelInfo("glm-4.7-flash",      "GLM-4.7 Flash — free 200K + tools",      "free"),
        ModelInfo("glm-4-flash-250414", "GLM-4 Flash — free 128K",                "free"),
    ],

    # ── Kimi (Moonshot AI) ──────────────────────────────────────────
    # Source: platform.kimi.com/docs/api/chat
    # NOTE: K2 series (kimi-k2-thinking*, kimi-k2-turbo-preview) retires
    # 2026-05-25 per Moonshot's deprecation notice. Users should migrate
    # to kimi-k2.6 (default).
    "kimi": [
        ModelInfo("kimi-k2.6",                       "K2.6 — flagship thinking",          "flagship"),
        ModelInfo("kimi-k2.5",                       "K2.5 — prior gen",                  "balanced"),
        ModelInfo("kimi-k2-thinking",                "K2 reasoning (retires 2026-05-25)", "reasoning"),
        ModelInfo("kimi-k2-thinking-turbo",          "K2 reasoning fast (retires 2026-05-25)", "reasoning"),
        ModelInfo("kimi-k2-turbo-preview",           "K2 turbo preview (retires 2026-05-25)",  "fast"),
        ModelInfo("moonshot-v1-128k",                "Moonshot v1 128K base",             "balanced"),
        ModelInfo("moonshot-v1-32k",                 "Moonshot v1 32K",                   "balanced"),
        ModelInfo("moonshot-v1-8k",                  "Moonshot v1 8K",                    "fast"),
        ModelInfo("moonshot-v1-auto",                "Moonshot v1 auto-select",           "balanced"),
        ModelInfo("moonshot-v1-128k-vision-preview", "Vision long-context",               "vision"),
        ModelInfo("moonshot-v1-32k-vision-preview",  "Vision 32K",                        "vision"),
        ModelInfo("moonshot-v1-8k-vision-preview",   "Vision 8K",                         "vision"),
    ],

    # ── Qwen / Alibaba DashScope International ──────────────────────
    # Source: alibabacloud.com/help/en/model-studio/getting-started/models
    "qwen": [
        ModelInfo("qwen3-max",         "Qwen3 Max — top capability",          "flagship"),
        ModelInfo("qwen3.5-plus",      "Qwen3.5 Plus — 1M multimodal",        "balanced"),
        ModelInfo("qwen3.5-flash",     "Qwen3.5 Flash — cheap+fast",          "fast"),
        ModelInfo("qwen-plus",         "Qwen Plus — balanced",                "balanced"),
        ModelInfo("qwen-flash",        "Qwen Flash — cheap",                  "fast"),
        ModelInfo("qwen3-coder-plus",  "Qwen3 Coder Plus — agentic coding",   "coder"),
        ModelInfo("qwen3-coder-flash", "Qwen3 Coder Flash — fast coder",      "coder"),
        ModelInfo("qwen3-vl-plus",     "Qwen3 VL Plus — vision",              "vision"),
        ModelInfo("qwen3-vl-flash",    "Qwen3 VL Flash",                      "vision"),
        ModelInfo("qvq-max",           "QvQ Max — visual reasoning",          "reasoning"),
        ModelInfo("qwen-vl-ocr",       "Qwen VL OCR",                         "vision"),
    ],

    # ── Hyperbolic ──────────────────────────────────────────────────
    # Source: app.hyperbolic.ai/models (JS-rendered; verify before pin)
    "hyperbolic": [
        ModelInfo("Qwen/Qwen3-Next-Thinking",              "Qwen3-Next reasoning", "reasoning"),
        ModelInfo("Qwen/Qwen3-Next-Instruct",              "Qwen3-Next instruct",  "balanced"),
        ModelInfo("Qwen/Qwen3-Coder",                      "Qwen3 Coder",          "coder"),
        ModelInfo("deepseek-ai/DeepSeek-V3-0324",          "DeepSeek V3 0324",     "balanced"),
        ModelInfo("deepseek-ai/DeepSeek-R1",               "DeepSeek R1 reasoner", "reasoning"),
        ModelInfo("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama 3.1 8B",         "fast"),
    ],

    # ── Perplexity Sonar (built-in web search) ──────────────────────
    # Source: docs.perplexity.ai/getting-started/models
    "perplexity": [
        ModelInfo("sonar-pro",            "Sonar Pro — search w/ grounding (200K)", "flagship"),
        ModelInfo("sonar-reasoning-pro",  "Reasoning Pro — CoT search (127K)",      "reasoning"),
        ModelInfo("sonar-deep-research",  "Deep research — autonomous (128K)",      "reasoning"),
        ModelInfo("sonar",                "Sonar — cheap grounded search (127K)",   "fast"),
    ],

    # ── Ollama (local, free) — popular base names ───────────────────
    # Source: ollama.com/library
    "ollama": [
        ModelInfo("llama3.1",          "Llama 3.1 — 8B/70B/405B (default 8B)", "balanced"),
        ModelInfo("deepseek-r1",       "DeepSeek-R1 family — top reasoning",   "reasoning"),
        ModelInfo("llama3.2",          "Llama 3.2 1B/3B — small & fast",       "fast"),
        ModelInfo("qwen2.5",           "Qwen 2.5 0.5B-72B — multilingual",     "balanced"),
        ModelInfo("qwen3",             "Qwen3 0.6B-235B — latest dense+MoE",   "flagship"),
        ModelInfo("qwen2.5-coder",     "Qwen 2.5 Coder 0.5B-32B",              "coder"),
        ModelInfo("mistral",           "Mistral 7B v0.3",                      "fast"),
        ModelInfo("gemma2",            "Gemma 2 2B/9B",                        "fast"),
        ModelInfo("phi3.5",            "Phi-3.5 3.8B",                         "fast"),
        ModelInfo("nomic-embed-text",  "Nomic embed — text retrieval",         "embed"),
    ],
}


def models_for(provider: str) -> list[ModelInfo]:
    """Return the model list for a provider (empty list if unknown).

    Accepts the canonical key (e.g. ``openai-codex``) and common aliases
    (``openai_codex``, ``codex``, ``claude``, ``moonshot``).
    """
    if provider in MODELS:
        return MODELS[provider]
    # Common alias normalization
    aliases = {
        "openai_codex": "openai-codex",
        "codex": "openai-codex",
        "claude": "anthropic",
        "moonshot": "kimi",
        "kimi-k2": "kimi",
        "qwen3": "qwen",
        "dashscope": "qwen",
        "sonar": "perplexity",
        "google": "gemini",
        "grok": "xai",
    }
    canonical = aliases.get(provider, provider)
    return MODELS.get(canonical, [])


def default_model(provider: str) -> str:
    """Return the recommended default model id for a provider, or empty string."""
    entries = models_for(provider)
    return entries[0].id if entries else ""


def all_providers() -> list[str]:
    """Return the list of provider keys with at least one registered model."""
    return list(MODELS.keys())


__all__ = [
    "ModelInfo",
    "MODELS",
    "models_for",
    "default_model",
    "all_providers",
]
