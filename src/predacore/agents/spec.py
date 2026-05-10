"""DynamicAgentSpec — rigorous subagent specification.

Anthropic's documented failure mode: vague specs cause duplicate work
across subagents (50 subagents spawned for trivial queries, multiple
subagents independently scraping the same source).

The validator refuses specs that don't meet a minimum rigor bar — it
is cheaper to refuse-to-spawn than to run a vague subagent.

Required fields per Anthropic's "Multi-Agent Research System" post:
  - objective (concrete, action-verb, ≥5 words)
  - output_format (markdown table | JSON schema | bullet list | …)
  - success_criteria (how subagent knows it's done)
  - task_boundaries (what NOT to investigate)
  - allowed_tools (explicit allowlist, no wildcards)
  - max_steps (per Anthropic's scaling rules)

Anthropic's scaling rules (embedded in lead-agent prompt elsewhere):
  - Simple fact-finding:    1 agent, 3-10 tool calls
  - Direct comparison:      2-4 subagents, 10-15 calls each
  - Complex research:       10+ subagents
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .exceptions import SpecValidationError


# Minimum word count for objective — 5 words approximates "subject + verb + object + qualifier"
_MIN_OBJECTIVE_WORDS = 5
# Action verbs the objective MUST start with (case-insensitive). Forces concrete framing.
_ACTION_VERB_PATTERN = re.compile(
    r"^(find|list|compare|summarize|analyze|extract|verify|"
    r"identify|count|rank|evaluate|search|scrape|read|"
    r"draft|write|generate|review|critique|explain|"
    r"check|validate|classify|cluster|propose|estimate|"
    r"fetch|collect|investigate|research|cross-reference|"
    r"answer|describe|determine|examine|inspect|outline|"
    r"plan|recommend|select|trace|update|build|create|"
    r"refine|revise|test|debug|run|execute|deploy|"
    r"convert|translate|merge|split|score|assess)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AgentSpec:
    """One subagent's specification — frozen so spec is the contract.

    Created by the lead agent during decomposition. Validated before
    spawn. Carried through the runner (in-process or DAF) to the worker.
    """

    # IDENTITY
    id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:12]}")
    base_type: str = "generalist"           # template (analyst, researcher, ...)
    specialization: str = ""                # concrete role (e.g. "Q3 EPS comparison")

    # OBJECTIVE
    objective: str = ""                     # required
    success_criteria: tuple[str, ...] = ()  # how subagent knows it's done
    output_format: str = ""                 # required (markdown | json | …)
    task_boundaries: str = ""               # what NOT to investigate (optional but recommended)

    # SCALING
    max_steps: int = 10                     # per Anthropic's table
    max_tokens: int = 50_000                # subagent's slice of run budget

    # MEMORY
    memory_scopes: tuple[str, ...] = ("global", "team")
    memory_budget: int = 8                  # max recalls per agent run
    preloaded_memory: tuple[dict[str, Any], ...] = ()  # pre-recalled context

    # TOOLS
    allowed_tools: tuple[str, ...] = ()     # explicit allowlist (NO wildcards)

    # OBSERVABILITY
    trace_id: str = ""                      # propagates through DAF gRPC
    parent_run_id: str = ""                 # the orchestration this belongs to

    # SAFETY
    delegation_depth: int = 0               # M4 — must propagate across DAF
    max_delegation_depth: int = 3           # absolute ceiling on recursion

    def __post_init__(self) -> None:
        # Use object.__setattr__ for frozen dataclass cleanups
        if not self.id:
            object.__setattr__(self, "id", f"agent_{uuid4().hex[:12]}")

    @classmethod
    def create(
        cls,
        *,
        base_type: str,
        specialization: str,
        objective: str,
        output_format: str,
        success_criteria: tuple[str, ...] = (),
        task_boundaries: str = "",
        max_steps: int = 10,
        max_tokens: int = 50_000,
        memory_scopes: tuple[str, ...] = ("global", "team"),
        allowed_tools: tuple[str, ...] = (),
        parent_run_id: str = "",
        trace_id: str = "",
        delegation_depth: int = 0,
        preloaded_memory: tuple[dict[str, Any], ...] = (),
    ) -> "AgentSpec":
        """Create + validate. Raises SpecValidationError on failure."""
        spec = cls(
            base_type=base_type,
            specialization=specialization,
            objective=objective.strip(),
            output_format=output_format.strip(),
            success_criteria=tuple(s.strip() for s in success_criteria if s.strip()),
            task_boundaries=task_boundaries.strip(),
            max_steps=max_steps,
            max_tokens=max_tokens,
            memory_scopes=tuple(memory_scopes),
            allowed_tools=tuple(allowed_tools),
            parent_run_id=parent_run_id,
            trace_id=trace_id or uuid4().hex[:12],
            delegation_depth=delegation_depth,
            preloaded_memory=tuple(preloaded_memory),
        )
        validate_spec(spec)
        return spec

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for gRPC / logging."""
        return {
            "id": self.id,
            "base_type": self.base_type,
            "specialization": self.specialization,
            "objective": self.objective,
            "output_format": self.output_format,
            "success_criteria": list(self.success_criteria),
            "task_boundaries": self.task_boundaries,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "memory_scopes": list(self.memory_scopes),
            "memory_budget": self.memory_budget,
            "allowed_tools": list(self.allowed_tools),
            "trace_id": self.trace_id,
            "parent_run_id": self.parent_run_id,
            "delegation_depth": self.delegation_depth,
            "max_delegation_depth": self.max_delegation_depth,
        }

    def system_prompt_block(self) -> str:
        """Render the spec into a system-prompt block for the subagent.

        This is the canonical way the subagent's LLM sees its mission.
        Includes the full spec so a misaligned subagent can be diagnosed
        from logs.
        """
        lines = [
            f"# Subagent role: {self.base_type} → {self.specialization}",
            "",
            f"## Objective",
            self.objective,
            "",
            f"## Output format",
            self.output_format,
        ]
        if self.success_criteria:
            lines.append("")
            lines.append("## Done when")
            for c in self.success_criteria:
                lines.append(f"- {c}")
        if self.task_boundaries:
            lines.append("")
            lines.append("## Do NOT")
            lines.append(self.task_boundaries)
        lines.extend(
            [
                "",
                "## Constraints",
                f"- max_steps: {self.max_steps}",
                f"- max_tokens: {self.max_tokens}",
                f"- allowed_tools: {', '.join(self.allowed_tools) or '(none)'}",
                f"- delegation_depth: {self.delegation_depth}/{self.max_delegation_depth}",
            ]
        )
        return "\n".join(lines)


def validate_spec(spec: AgentSpec) -> None:
    """Refuse specs that don't meet rigor. Raises SpecValidationError.

    Cheaper to refuse-to-spawn than waste a subagent on vague work.
    """
    errors: list[str] = []

    # Objective: ≥5 words, action verb start
    obj = spec.objective.strip()
    if not obj:
        errors.append("objective is empty")
    else:
        word_count = len([w for w in obj.split() if w.strip()])
        if word_count < _MIN_OBJECTIVE_WORDS:
            errors.append(
                f"objective too vague ({word_count} words, need ≥{_MIN_OBJECTIVE_WORDS}): {obj!r}"
            )
        if not _ACTION_VERB_PATTERN.match(obj):
            errors.append(
                f"objective must start with an action verb "
                f"(find/list/compare/analyze/...): {obj.split()[0] if obj else ''!r}"
            )

    # Output format: required, non-empty
    if not spec.output_format.strip():
        errors.append("output_format is empty (specify 'markdown' / 'json schema X' / etc.)")

    # Allowed tools: explicit, no wildcards
    if not spec.allowed_tools:
        errors.append("allowed_tools is empty (must be explicit allowlist)")
    else:
        for tool in spec.allowed_tools:
            if tool == "*" or "*" in tool:
                errors.append(f"wildcard tool not allowed: {tool!r}")

    # Max steps: positive, sane
    if spec.max_steps <= 0:
        errors.append(f"max_steps must be > 0 (got {spec.max_steps})")
    if spec.max_steps > 50:
        errors.append(f"max_steps too high ({spec.max_steps}); cap at 50 per scaling rule")

    # Max tokens: positive
    if spec.max_tokens <= 0:
        errors.append(f"max_tokens must be > 0 (got {spec.max_tokens})")

    # Delegation depth: must respect ceiling
    if spec.delegation_depth >= spec.max_delegation_depth:
        errors.append(
            f"delegation_depth {spec.delegation_depth} >= max {spec.max_delegation_depth}; "
            f"refuse to spawn (recursion guard)"
        )

    # Memory scopes: known values only
    valid_scopes = {"global", "team", "ephemeral", "session", "user"}
    for scope in spec.memory_scopes:
        if scope not in valid_scopes:
            errors.append(f"unknown memory_scope: {scope!r}")

    if errors:
        raise SpecValidationError(
            "Spec validation failed for "
            f"{spec.base_type}/{spec.specialization}: "
            + "; ".join(errors)
        )


# Anthropic's scaling rules — embed verbatim in lead agent's system prompt.
# Quoted from "How we built our multi-agent research system" (May 2025).
ANTHROPIC_SCALING_RULES = """
## Scaling rules for subagent count and depth

When deciding how many subagents to spawn:
- Simple fact-finding (1 fact, 1 source): 1 agent, 3-10 tool calls
- Direct comparison (X vs Y, 2-3 entities): 2-4 subagents, 10-15 calls each
- Complex research (broad topic, multi-source): 10+ subagents with clearly
  divided responsibilities

Anti-patterns (DO NOT do these):
- Spawning >5 subagents for a query that could be answered by 1
- Letting a subagent run >20 tool calls without check-in
- Spawning subagents with vague objectives ("research X") — they will
  duplicate each other's work. Each subagent's objective must be a
  concrete action verb statement of ≥5 words with explicit boundaries.
- Spawning subagents whose output formats overlap — define divergent
  output_format for each so synthesis is clean.

Each subagent spec MUST include: objective (concrete, ≥5 words, action verb),
output_format (markdown table | JSON schema | …), success_criteria (how the
subagent knows it's done), task_boundaries (what NOT to investigate),
allowed_tools (explicit allowlist), max_steps (per scaling table above).
"""
