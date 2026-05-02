"""
Skill Scanner — Three-point security analysis for collective intelligence skills.

Scans skills at three checkpoints:
  1. On publish  — creator PredaCore scans before sharing
  2. At the pool — collective intelligence registry scans on arrival
  3. On receive  — every receiving PredaCore scans before sandbox trial

Static analysis catches dangerous patterns BEFORE execution.
Runtime monitoring catches behavioral drift DURING execution.
Skills are recipes (tool pipelines), not code — so the scanner can
read and understand 100% of what a skill does.

Scan verdicts:
  CLEAN     — no issues found, proceed
  FLAGGED   — suspicious but not definitively bad, needs user review
  REJECTED  — definitively dangerous, auto-blocked
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .skill_genome import (
    SENSITIVE_PATHS,
    TOOL_TIER_MAP,
    CapabilityTier,
    SkillGenome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan verdicts
# ---------------------------------------------------------------------------


class ScanVerdict(str, Enum):
    CLEAN = "clean"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass
class ScanFinding:
    """A single finding from a scan."""
    rule: str                  # e.g., "exfiltration_pattern"
    severity: str              # "low", "medium", "high", "critical"
    description: str
    step_index: int | None = None   # which step triggered it
    tool_name: str = ""


@dataclass
class ScanReport:
    """Complete scan report for a skill genome."""
    genome_id: str
    verdict: ScanVerdict
    findings: list[ScanFinding] = field(default_factory=list)
    scanned_at: float = field(default_factory=time.time)
    scanner_version: str = "1.0.0"
    scan_duration_ms: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "verdict": self.verdict.value,
            "findings": [
                {
                    "rule": f.rule,
                    "severity": f.severity,
                    "description": f.description,
                    "step_index": f.step_index,
                    "tool_name": f.tool_name,
                }
                for f in self.findings
            ],
            "scanned_at": self.scanned_at,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
        }


# ---------------------------------------------------------------------------
# Exfiltration patterns — read + network combos that leak data
# ---------------------------------------------------------------------------

# Tools that read local data
_READ_TOOLS = frozenset({
    "read_file", "list_directory", "git_context", "git_find_files",
    "git_diff_summary", "memory_recall", "semantic_search", "pdf_reader",
    "identity_read",
})

# Tools that send data to network. ``api_call`` was missing here despite
# being the most direct way to POST data outbound — adding it closes a
# coverage hole that the exfiltration-pattern detector would otherwise miss.
_NETWORK_SEND_TOOLS = frozenset({
    "api_call",
    "web_scrape", "browser_automation", "browser_control",
    "speak", "voice_note",
    "image_gen", "multi_agent", "openclaw_delegate",
})

# Tools that write locally (can be used to stage exfiltration)
_WRITE_TOOLS = frozenset({
    "write_file", "run_command", "python_exec", "execute_code",
    "memory_store", "identity_update",
})


# ---------------------------------------------------------------------------
# Skill Scanner
# ---------------------------------------------------------------------------


class SkillScanner:
    """Security scanner for skill genomes.

    Performs static analysis on the skill recipe to detect dangerous patterns
    before the skill ever executes.  Since skills are tool pipelines (not code),
    the scanner can understand 100% of the skill's behavior.
    """

    def __init__(self) -> None:
        self._scan_count = 0
        self._reject_count = 0

    def scan(self, genome: SkillGenome) -> ScanReport:
        """Run full static analysis on a skill genome. Returns ScanReport."""
        start = time.time()
        findings: list[ScanFinding] = []

        # Run all static analysis rules
        findings.extend(self._check_capability_mismatch(genome))
        findings.extend(self._check_exfiltration_pattern(genome))
        findings.extend(self._check_sensitive_paths(genome))
        findings.extend(self._check_undeclared_tools(genome))
        findings.extend(self._check_obfuscation(genome))
        findings.extend(self._check_excessive_scope(genome))
        findings.extend(self._check_signature(genome))
        findings.extend(self._check_empty_or_malformed(genome))

        # Determine verdict
        verdict = self._compute_verdict(findings)

        elapsed = (time.time() - start) * 1000
        self._scan_count += 1
        if verdict == ScanVerdict.REJECTED:
            self._reject_count += 1

        report = ScanReport(
            genome_id=genome.id,
            verdict=verdict,
            findings=findings,
            scan_duration_ms=round(elapsed, 2),
        )

        logger.info(
            "Scan complete: skill=%s verdict=%s findings=%d (%.1fms)",
            genome.name or genome.id,
            verdict.value,
            len(findings),
            elapsed,
        )
        return report

    # -- Static analysis rules ----------------------------------------------

    def _check_capability_mismatch(self, genome: SkillGenome) -> list[ScanFinding]:
        """Check if declared tier matches actual tool usage."""
        findings = []
        actual_tier = CapabilityTier.PURE_LOGIC

        for step in genome.steps:
            tool_tier = TOOL_TIER_MAP.get(step.tool_name, CapabilityTier.LOCAL_READ)
            if tool_tier > actual_tier:
                actual_tier = tool_tier

        if actual_tier > genome.capability_tier:
            findings.append(ScanFinding(
                rule="capability_mismatch",
                severity="critical",
                description=(
                    f"Skill declares tier {genome.capability_tier.name} but uses "
                    f"tools requiring tier {actual_tier.name}. "
                    f"This is either a bug or an attempt to bypass tier restrictions."
                ),
            ))
        return findings

    def _check_exfiltration_pattern(self, genome: SkillGenome) -> list[ScanFinding]:
        """Detect read-then-send patterns that could exfiltrate data.

        The check is **sequence-aware**: a read step + an unrelated network
        step elsewhere in the recipe is NOT exfiltration (a skill might
        legitimately do both in independent contexts). We flag when the
        network step ACTUALLY consumes the read step's output, evidenced by:

          (a) the network step is the immediate next step (data flows by
              implicit pipe), OR
          (b) the network step has ``use_previous=True`` AND the immediately
              preceding step is a read tool, OR
          (c) the network step's parameters reference an output variable
              that resolves to a read step (templated ``{{prev}}`` or
              ``{{step_N}}``).

        The previous stateless detector flagged any read+send combo, even
        unrelated ones, AND missed encode-then-send (since base64/encode
        between them broke the read→send adjacency). The new shape catches
        intent-bearing pipes while reducing false positives.
        """
        findings = []

        for i, step in enumerate(genome.steps):
            if step.tool_name not in _NETWORK_SEND_TOOLS:
                continue

            # Look at the immediately preceding step — only adjacency or
            # explicit use_previous wiring qualifies as exfiltration intent.
            prev_idx = i - 1
            if prev_idx < 0:
                continue
            prev_step = genome.steps[prev_idx]
            prev_is_read = prev_step.tool_name in _READ_TOOLS

            # (a) + (b): adjacency, with use_previous strengthening the signal
            uses_prev = bool(getattr(step, "use_previous", False))
            if prev_is_read and (uses_prev or i - prev_idx == 1):
                findings.append(ScanFinding(
                    rule="exfiltration_pattern",
                    severity="critical",
                    description=(
                        f"Step {i} ({step.tool_name}) sends data to the "
                        f"network immediately after step {prev_idx} "
                        f"({prev_step.tool_name}) read local data"
                        + (" (use_previous=True)" if uses_prev else "")
                        + ". This pattern can exfiltrate sensitive data."
                    ),
                    step_index=i,
                    tool_name=step.tool_name,
                ))
                continue

            # (c): templated reference to an earlier read step's output
            params_str = str(step.parameters)
            for j in range(i):
                if genome.steps[j].tool_name not in _READ_TOOLS:
                    continue
                if (
                    "{{prev}}" in params_str
                    or f"{{{{step_{j}}}}}" in params_str
                    or f"{{{{steps[{j}]}}}}" in params_str
                ):
                    findings.append(ScanFinding(
                        rule="exfiltration_pattern",
                        severity="critical",
                        description=(
                            f"Step {i} ({step.tool_name}) parameters reference "
                            f"the output of read step {j} ({genome.steps[j].tool_name}) "
                            f"and send to the network. Likely exfiltration."
                        ),
                        step_index=i,
                        tool_name=step.tool_name,
                    ))
                    break
        return findings

    def _check_sensitive_paths(self, genome: SkillGenome) -> list[ScanFinding]:
        """Check if any step references sensitive file paths."""
        findings = []
        for i, step in enumerate(genome.steps):
            params_str = str(step.parameters).lower()
            for sensitive in SENSITIVE_PATHS:
                if sensitive.lower() in params_str:
                    findings.append(ScanFinding(
                        rule="sensitive_path_access",
                        severity="critical",
                        description=(
                            f"Step {i} ({step.tool_name}) references sensitive "
                            f"path '{sensitive}'. Skills must never access "
                            f"credentials or secret files."
                        ),
                        step_index=i,
                        tool_name=step.tool_name,
                    ))
        return findings

    def _check_undeclared_tools(self, genome: SkillGenome) -> list[ScanFinding]:
        """Check if steps use tools not in declared_tools."""
        findings = []
        declared = set(genome.declared_tools)

        for i, step in enumerate(genome.steps):
            if step.tool_name not in declared:
                findings.append(ScanFinding(
                    rule="undeclared_tool",
                    severity="high",
                    description=(
                        f"Step {i} uses '{step.tool_name}' which is not in "
                        f"declared_tools {sorted(declared)}. All tools must "
                        f"be declared in the capability manifest."
                    ),
                    step_index=i,
                    tool_name=step.tool_name,
                ))
        return findings

    def _check_obfuscation(self, genome: SkillGenome) -> list[ScanFinding]:
        """Detect obfuscated or encoded content in parameters."""
        findings = []
        for i, step in enumerate(genome.steps):
            params_str = json.dumps(step.parameters, default=str)

            # Check for base64-like strings (long alphanumeric with padding)
            if re.search(r'[A-Za-z0-9+/]{40,}={0,2}', params_str):
                findings.append(ScanFinding(
                    rule="obfuscation_detected",
                    severity="high",
                    description=(
                        f"Step {i} ({step.tool_name}) contains what appears to be "
                        f"base64-encoded content. Skills must be fully readable "
                        f"in plain text."
                    ),
                    step_index=i,
                    tool_name=step.tool_name,
                ))

            # Check for hex-encoded strings
            if re.search(r'\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){10,}', params_str):
                findings.append(ScanFinding(
                    rule="obfuscation_detected",
                    severity="high",
                    description=(
                        f"Step {i} ({step.tool_name}) contains hex-encoded content."
                    ),
                    step_index=i,
                    tool_name=step.tool_name,
                ))

            # Check for URL patterns in non-network tools
            _NETWORK_TOOLS_WITH_URLS = _NETWORK_SEND_TOOLS | {"web_search", "web_scrape", "deep_search", "browser_automation", "browser_control"}
            if step.tool_name not in _NETWORK_TOOLS_WITH_URLS:
                if re.search(r'https?://\S+', params_str):
                    findings.append(ScanFinding(
                        rule="unexpected_url",
                        severity="medium",
                        description=(
                            f"Step {i} ({step.tool_name}) contains a URL but is "
                            f"not a network tool. This could indicate data exfiltration."
                        ),
                        step_index=i,
                        tool_name=step.tool_name,
                    ))
        return findings

    def _check_excessive_scope(self, genome: SkillGenome) -> list[ScanFinding]:
        """Flag skills that request too many tool permissions."""
        findings = []
        if len(genome.declared_tools) > 8:
            findings.append(ScanFinding(
                rule="excessive_scope",
                severity="medium",
                description=(
                    f"Skill declares {len(genome.declared_tools)} tools. "
                    f"Skills should follow least-privilege — request only "
                    f"what's needed. Consider splitting into smaller skills."
                ),
            ))

        # Check for mixed read+write+network (kitchen sink)
        has_read = any(t in _READ_TOOLS for t in genome.declared_tools)
        has_write = any(t in _WRITE_TOOLS for t in genome.declared_tools)
        has_network = any(t in _NETWORK_SEND_TOOLS for t in genome.declared_tools)
        if has_read and has_write and has_network:
            findings.append(ScanFinding(
                rule="kitchen_sink_permissions",
                severity="high",
                description=(
                    "Skill requests read + write + network permissions. "
                    "This is the broadest possible scope and requires "
                    "maximum scrutiny."
                ),
            ))
        return findings

    def _check_signature(self, genome: SkillGenome) -> list[ScanFinding]:
        """Verify the genome's cryptographic signature."""
        findings = []
        if not genome.signature:
            findings.append(ScanFinding(
                rule="missing_signature",
                severity="high",
                description="Skill has no cryptographic signature. Cannot verify integrity.",
            ))
        elif not genome.verify_signature():
            findings.append(ScanFinding(
                rule="invalid_signature",
                severity="critical",
                description=(
                    "Skill signature verification FAILED. The genome has been "
                    "tampered with since it was signed. REJECTING."
                ),
            ))
        return findings

    def _check_empty_or_malformed(self, genome: SkillGenome) -> list[ScanFinding]:
        """Basic sanity checks on genome structure."""
        findings = []
        if not genome.steps:
            findings.append(ScanFinding(
                rule="empty_skill",
                severity="medium",
                description="Skill has no steps. Empty skills should not propagate.",
            ))
        if not genome.name:
            findings.append(ScanFinding(
                rule="missing_name",
                severity="low",
                description="Skill has no name. All shared skills should be named.",
            ))
        if not genome.creator_instance_id:
            findings.append(ScanFinding(
                rule="missing_origin",
                severity="medium",
                description="Skill has no creator instance ID. Origin cannot be verified.",
            ))
        if len(genome.steps) > 20:
            findings.append(ScanFinding(
                rule="excessive_steps",
                severity="medium",
                description=(
                    f"Skill has {len(genome.steps)} steps. Complex skills are "
                    f"harder to audit. Consider splitting."
                ),
            ))
        return findings

    # -- Verdict computation ------------------------------------------------

    def _compute_verdict(self, findings: list[ScanFinding]) -> ScanVerdict:
        """Compute overall verdict from findings."""
        if not findings:
            return ScanVerdict.CLEAN

        severities = {f.severity for f in findings}

        # Any critical finding = auto-reject
        if "critical" in severities:
            return ScanVerdict.REJECTED

        # Multiple high findings = reject
        high_count = sum(1 for f in findings if f.severity == "high")
        if high_count >= 2:
            return ScanVerdict.REJECTED

        # Single high or any medium = flag for review
        if "high" in severities or "medium" in severities:
            return ScanVerdict.FLAGGED

        # Only low findings = clean
        return ScanVerdict.CLEAN

    # -- Runtime monitoring -------------------------------------------------

    def check_runtime_tool_drift(
        self,
        genome: SkillGenome,
        actual_tool_called: str,
        step_index: int,
    ) -> ScanFinding | None:
        """Check if a tool call at runtime matches the declared recipe.

        Called by the sandbox during skill execution.  If the skill calls
        a tool it didn't declare, this returns a finding and the sandbox
        should kill the execution immediately.
        """
        # Check against declared tools
        if actual_tool_called not in genome.declared_tools:
            return ScanFinding(
                rule="runtime_tool_drift",
                severity="critical",
                description=(
                    f"RUNTIME: Step {step_index} called '{actual_tool_called}' "
                    f"which is not in declared tools {sorted(genome.declared_tools)}. "
                    f"Skill killed and quarantined."
                ),
                step_index=step_index,
                tool_name=actual_tool_called,
            )

        # Check if the step matches expected tool
        if step_index < len(genome.steps):
            expected = genome.steps[step_index].tool_name
            if actual_tool_called != expected:
                return ScanFinding(
                    rule="runtime_step_mismatch",
                    severity="high",
                    description=(
                        f"RUNTIME: Step {step_index} expected "
                        f"'{expected}' but got '{actual_tool_called}'. "
                        f"Skill is deviating from its recipe."
                    ),
                    step_index=step_index,
                    tool_name=actual_tool_called,
                )
        return None

    def check_runtime_data_volume(
        self,
        genome: SkillGenome,
        step_index: int,
        output_size_bytes: int,
        threshold_bytes: int = 1_000_000,  # 1MB default
    ) -> ScanFinding | None:
        """Flag if a step produces unusually large output (possible data staging)."""
        if output_size_bytes > threshold_bytes:
            return ScanFinding(
                rule="runtime_data_volume_anomaly",
                severity="high",
                description=(
                    f"RUNTIME: Step {step_index} produced {output_size_bytes:,} bytes "
                    f"of output (threshold: {threshold_bytes:,}). "
                    f"Unusually large output may indicate data staging."
                ),
                step_index=step_index,
            )
        return None

    def check_runtime_timing(
        self,
        genome: SkillGenome,
        step_index: int,
        elapsed_ms: float,
        threshold_ms: float = 30_000,  # 30s default
    ) -> ScanFinding | None:
        """Flag if a step takes unusually long (possible network exfiltration)."""
        if elapsed_ms > threshold_ms:
            return ScanFinding(
                rule="runtime_timing_anomaly",
                severity="medium",
                description=(
                    f"RUNTIME: Step {step_index} took {elapsed_ms:.0f}ms "
                    f"(threshold: {threshold_ms:.0f}ms). "
                    f"Unusually slow execution may indicate hidden operations."
                ),
                step_index=step_index,
            )
        return None

    # -- Stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "total_scans": self._scan_count,
            "total_rejections": self._reject_count,
            "rejection_rate": (
                f"{self._reject_count / self._scan_count * 100:.1f}%"
                if self._scan_count > 0 else "N/A"
            ),
        }
