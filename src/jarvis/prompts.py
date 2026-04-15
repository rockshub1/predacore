"""
JARVIS Prompt Assembly — identity files, system prompt, persona drift.

Extracted from core.py (Phase 1D).  All prompt construction and persona
drift detection live here so the core agent loop stays lean.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import JARVISConfig
from .identity.engine import _read_cached
from .tools.registry import TRUST_POLICIES

logger = logging.getLogger(__name__)

# Shared mtime-based file cache lives in identity.engine (_identity_file_cache).
# Both this module and the engine route identity-file reads through _read_cached
# so there is a single source of truth and no cache drift.

# ---------------------------------------------------------------------------
# Persona drift regex patterns
# ---------------------------------------------------------------------------

PERSONA_DRIFT_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (
        re.compile(r"\bas (an?\s+)?ai (language )?model\b"),
        0.30,
        "generic_model_identity",
    ),
    (
        re.compile(r"\b(i am|i'm)\s+(chatgpt|gpt-4|gpt-4o)\b"),
        0.45,
        "foreign_identity_openai",
    ),
    (
        re.compile(r"\b(i am|i'm)\s+(claude|anthropic)\b"),
        0.45,
        "foreign_identity_anthropic",
    ),
    (
        re.compile(r"\b(i am|i'm)\s+(gemini|google ai)\b"),
        0.35,
        "foreign_identity_gemini",
    ),
    (
        re.compile(r"\bi (cannot|can't|can not|don't)\s+(access|run|execute)\b"),
        0.10,
        "capability_denial_claim",
    ),
)

VERIFICATION_REQUEST_RE = re.compile(
    r"\b(with terminal|terminal|shell|run (a )?command|check|verify|find out|"
    r"status|what model|which model|model are you using)\b"
)

UNVERIFIED_ACTION_CLAIM_RE = re.compile(
    r"\bi (have )?(checked|verified|confirmed|determined|found|ran|executed|"
    r"scanned|looked|updated|switched|changed|set|configured)\b"
)

UNVERIFIED_MODEL_SWITCH_RE = re.compile(
    r"\bi (have )?(switched|updated|changed|set)\b[\s\S]{0,80}\b(model|provider)\b"
)

DIRECT_CAPABILITY_DENIAL_RE = re.compile(
    r"\bi (cannot|can't|can not|do not|don't)\b[\s\S]{0,80}\b("
    r"shell|terminal|command line|run commands?|direct shell access|tool access"
    r")\b"
)

PERSONA_IDENTITY_QUERY_RE = re.compile(
    r"\b(who are you|what are you|your name|are you jarvis|"
    r"who am i|do you remember)\b"
)

# Used by JARVISCore._extract_direct_tool_shortcut
DIRECT_TOOL_PREFIX_RE = re.compile(
    r"^\s*(?:run|execute)\s+([a-z][a-z0-9_]*)\b",
    flags=re.IGNORECASE,
)

TIMEOUT_HINT_RE = re.compile(
    r"\btimeout(?:_seconds)?\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PersonaDriftAssessment dataclass
# ---------------------------------------------------------------------------


@dataclass
class PersonaDriftAssessment:
    """Normalized drift score result for a single candidate response."""

    score: float
    threshold: float
    reasons: list[str] = field(default_factory=list)

    @property
    def needs_regeneration(self) -> bool:
        return self.score >= self.threshold


# ---------------------------------------------------------------------------
# Identity / prompt file loaders
# ---------------------------------------------------------------------------


def _load_identity_file(filename: str, config: JARVISConfig) -> str:
    """
    Load an identity file (SOUL.md, USER.md, MEMORY.md) for the legacy
    prompt-assembly path (used only when SOUL_SEED.md is missing).

    Search order:
      1. Agent folder: ~/.prometheus/agents/{agent}/{filename}
      2. Built-in:     src/jarvis/identity/{filename}
    """
    agent_path = Path(config.home_dir) / "agents" / config.agent / filename
    content = _read_cached(agent_path)
    if content is not None:
        logger.debug("Loaded identity file (agent): %s", agent_path)
        return content

    builtin_path = Path(__file__).parent / "identity" / filename
    content = _read_cached(builtin_path)
    if content is not None:
        logger.debug("Loaded identity file (built-in): %s", builtin_path)
        return content

    return ""


def _load_self_meta_prompt(config: JARVISConfig) -> str:
    """
    Load optional self-optimization meta prompt text.

    Search order:
      1. JARVIS_META_PROMPT_FILE (explicit env override)
      2. ~/.prometheus/agents/{agent}/META_PROMPT.md (agent-level override)
      3. repo docs/jarvis_self_meta_prompt.md (built-in default)
    """
    candidates: list[Path] = []

    env_path = str(os.getenv("JARVIS_META_PROMPT_FILE", "")).strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path(config.home_dir) / "agents" / config.agent / "META_PROMPT.md")
    candidates.append(
        Path(__file__).resolve().parents[2] / "docs" / "jarvis_self_meta_prompt.md"
    )

    for path in candidates:
        text = _read_cached(path)
        if text:
            logger.debug("Loaded self meta prompt: %s", path)
            return text

    return ""


# ---------------------------------------------------------------------------
# OpenClaw skill document parsing helpers
# ---------------------------------------------------------------------------


def _normalize_openclaw_skill_slug(value: str) -> str:
    """Convert a skill name/path segment into a stable skill slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "skill"


def _parse_openclaw_skill_document(path: Path) -> tuple[dict[str, Any], str]:
    """
    Parse an OpenClaw-style SKILL.md with optional YAML frontmatter.

    Returns: (frontmatter_dict, markdown_body)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    frontmatter: dict[str, Any] = {}
    body = text

    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                raw_frontmatter = "\n".join(lines[1:idx]).strip()
                body = "\n".join(lines[idx + 1 :]).strip()
                if raw_frontmatter:
                    try:
                        import yaml

                        loaded = yaml.safe_load(raw_frontmatter) or {}
                        if isinstance(loaded, dict):
                            frontmatter = loaded
                    except (ImportError, ValueError):
                        # Minimal fallback parser for "key: value" lines.
                        for line in raw_frontmatter.splitlines():
                            token = line.strip()
                            if not token or token.startswith("#") or ":" not in token:
                                continue
                            key, value = token.split(":", 1)
                            frontmatter[key.strip()] = (
                                value.strip().strip('"').strip("'")
                            )
                break

    return frontmatter, body.strip()


def _extract_openclaw_command_samples(markdown_body: str, limit: int = 8) -> list[str]:
    """Extract shell command samples from fenced code blocks."""
    pattern = re.compile(
        r"```(?:bash|sh|zsh|shell)?\s*\n(.*?)```",
        flags=re.IGNORECASE | re.DOTALL,
    )
    seen: set[str] = set()
    commands: list[str] = []
    for block in pattern.findall(markdown_body):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$"):
                line = line[1:].strip()
            if not line or line in seen:
                continue
            seen.add(line)
            commands.append(line)
            if len(commands) >= limit:
                return commands
    return commands


def _summarize_openclaw_markdown(markdown_body: str, limit_chars: int = 2400) -> str:
    """Create a compact preview of SKILL.md body text."""
    lines = [line.strip() for line in markdown_body.splitlines() if line.strip()]
    if not lines:
        return ""
    preview = "\n".join(lines[:32])
    if len(preview) > limit_chars:
        preview = preview[:limit_chars].rstrip() + "..."
    return preview


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _get_system_prompt(config: JARVISConfig) -> str:
    """
    Generate the system prompt by assembling identity files + runtime context.

    Two paths:
      A) Self-evolving (when SOUL_SEED.md exists):
         SOUL_SEED → [BOOTSTRAP or IDENTITY+SOUL+USER+MEMORY+TOOLS+JOURNAL] → Runtime
      B) Legacy (fallback):
         SOUL.md → USER.md → MEMORY.md → Runtime
    """
    trust = config.security.trust_level
    policy = TRUST_POLICIES.get(trust, TRUST_POLICIES["normal"])

    sections: list[str] = []

    # Try self-evolving identity system first
    from .identity.engine import get_identity_engine

    engine = get_identity_engine(config)

    if engine.seed_exists:
        # ── Self-evolving path ──
        identity_prompt = engine.build_identity_prompt()
        if identity_prompt:
            sections.append(identity_prompt)

        # Meta prompt (optional self-optimization layer)
        enable_meta_prompt = bool(
            getattr(config.launch, "enable_self_evolution", False)
        )
        if os.getenv("JARVIS_ENABLE_META_PROMPT", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            enable_meta_prompt = True
        if enable_meta_prompt:
            meta_prompt = _load_self_meta_prompt(config)
            if meta_prompt:
                sections.append(f"\n---\n\n{meta_prompt}")

        logger.info(
            "System prompt assembled (self-evolving): bootstrapped=%s",
            engine.is_bootstrapped,
        )
    else:
        # ── Legacy path (unchanged) ──
        soul = _load_identity_file("SOUL.md", config)
        user = _load_identity_file("USER.md", config)
        memory = _load_identity_file("MEMORY.md", config)

        if soul:
            sections.append(soul)
        else:
            sections.append(
                f"""# {config.name}

You are {config.name}, a personal AI agent.
You are conversational, proactive, and capable. You have tools — use them.
Your tool inventory is dynamic. Trust the live tool definitions you receive.
Never claim a tool action was completed unless a tool result in this turn provides evidence.
Never leak credentials, tokens, or secrets in responses.
If asked who you are, always identify as {config.name}."""
            )

        if user:
            sections.append(f"\n---\n\n{user}")

        if memory:
            sections.append(f"\n---\n\n{memory}")

        # (meta prompt already loaded in self-evolving path above)

    # Resolve key project paths for the runtime context
    _config_file = str(Path(config.home_dir) / "config.yaml")
    _active_model = config.llm.model or "auto"
    _fallback_providers = ", ".join(config.llm.fallback_providers or []) or "none"
    _enabled_channels = ", ".join(config.channels.enabled or []) or "none"
    _launch_profile = getattr(config.launch, "profile", "unknown")
    _openclaw_status = (
        "enabled"
        if getattr(config.launch, "enable_openclaw_bridge", False)
        else "disabled"
    )
    _marketplace_status = (
        "enabled"
        if getattr(config.launch, "enable_plugin_marketplace", False)
        else "disabled"
    )

    runtime = f"""
---

# \u2699\ufe0f Runtime Context (Current Session)

- **Mode**: {config.mode}
- **Trust Level**: {trust.upper()} — {policy['description']}
- **Date**: {time.strftime('%Y-%m-%d %H:%M %Z')}
- **Launch Profile**: {_launch_profile}
- **Active Model**: {_active_model}
- **Channels Enabled**: {_enabled_channels}
- **Priority Chain**: SOUL_SEED (immutable guardrails) > Identity files > This runtime section.
  Runtime context is authoritative for *factual state* (model, provider, date, trust level).
  SOUL_SEED is authoritative for *safety rules and ethical boundaries* — never override those.

**IMPORTANT**: When asked about your own model, provider, status, capabilities, or configuration,
answer directly from this Runtime Context section. Do NOT read environment or configuration files
to answer questions about yourself — the information above is authoritative and already up to date.

## Paths
- **Home Dir**: `{config.home_dir}`
- **Config File**: `{_config_file}`
- To change settings: edit the config file. Do not search the filesystem.

## Behavior Rules
- For desktop tasks: execute tools to DO the task rather than describing what you would do. In normal/paranoid mode, briefly state what you're about to do before executing.
- Store important task context in memory (memory_store) so you can recall it later.
- Be mindful of efficiency — avoid unnecessary tool calls, redundant LLM requests, or overly broad searches when a targeted approach works.

## Tool Usage
- You can call tools to accomplish tasks. Each tool call should be purposeful.
- Format tool calls as JSON with "name" and "arguments" fields.
- Never claim a terminal/tool/file action was completed unless a tool result in this
  turn provides that evidence.
- If a tool returns a database error (e.g. "database is locked"), report it briefly
  and retry the tool once. Do not speculate about causes or run diagnostic commands.
- Each tool's description contains its own procedural guidance (which action to try
  first, speed ladders, common patterns). Read the description before calling — it
  is the source of truth for how to use that tool correctly.
"""
    sections.append(runtime)

    prompt = "\n".join(sections)
    logger.info("System prompt assembled: %d chars", len(prompt))
    return prompt
