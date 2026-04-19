"""
Skill Evolution — Crystallization engine for learning tool patterns.

Watches PredaCore's tool execution history and detects repeated patterns.
When a pattern is used successfully N times, it's crystallized into a
reusable SkillGenome — a recipe that can be shared across the Flame.

Flow:
  1. ExecutionHistory records every tool call
  2. SkillCrystallizer scans history for repeated sequences
  3. Pattern detected → candidate SkillGenome created
  4. SkillScanner validates the candidate
  5. User endorses → skill is signed and ready to share
  6. CollectiveIntelligence publishes to the shared pool

This is how PredaCore learns: self-improvement is shared.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .skill_genome import (
    SkillGenome,
    SkillStep,
    TrustLevel,
    TrustScore,
)
from .skill_scanner import ScanVerdict, SkillScanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

# Minimum times a pattern must appear before crystallization
MIN_PATTERN_OCCURRENCES = 3

# Minimum success rate for a pattern to be worth crystallizing
MIN_SUCCESS_RATE = 0.8

# Maximum steps in a pattern (longer patterns are less likely to be reusable)
MAX_PATTERN_LENGTH = 8

# Minimum steps in a pattern
MIN_PATTERN_LENGTH = 2


@dataclass
class DetectedPattern:
    """A tool usage pattern detected in execution history."""
    tool_sequence: tuple[str, ...]    # ("web_search", "web_scrape", "write_file")
    occurrences: int = 0
    successes: int = 0
    failures: int = 0
    avg_elapsed_ms: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sample_args: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    @property
    def pattern_hash(self) -> str:
        return hashlib.md5(
            "|".join(self.tool_sequence).encode(), usedforsecurity=False
        ).hexdigest()[:12]

    @property
    def is_crystallizable(self) -> bool:
        """Check if this pattern is worth turning into a skill."""
        return (
            self.occurrences >= MIN_PATTERN_OCCURRENCES
            and self.success_rate >= MIN_SUCCESS_RATE
            and MIN_PATTERN_LENGTH <= len(self.tool_sequence) <= MAX_PATTERN_LENGTH
        )


# ---------------------------------------------------------------------------
# Skill Crystallizer
# ---------------------------------------------------------------------------


class SkillCrystallizer:
    """Detects repeated tool patterns and crystallizes them into skills.

    Watches execution history for sequences of tool calls that appear
    together repeatedly.  When a pattern is stable and successful, it
    becomes a SkillGenome — a reusable, shareable recipe.
    """

    def __init__(
        self,
        instance_id: str | None = None,
        user: str | None = None,
        data_dir: str | Path | None = None,
    ):
        self._instance_id = instance_id or os.getenv("PREDACORE_INSTANCE_ID", "local")
        self._user = user or os.getenv("USER", "default")
        self._data_dir = Path(data_dir or "~/.predacore/skills/evolved").expanduser()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._scanner = SkillScanner()
        self._patterns: dict[str, DetectedPattern] = {}  # hash → pattern
        self._crystallized: dict[str, SkillGenome] = {}   # genome_id → genome
        self._pending_endorsement: dict[str, SkillGenome] = {}  # genome_id → genome

        # Load previously crystallized skills
        self._load()

    # -- Pattern detection --------------------------------------------------

    def observe(self, execution_records: list[dict[str, Any]]) -> list[DetectedPattern]:
        """Scan execution records for tool patterns.

        Args:
            execution_records: List of dicts with at least:
                - tool: str
                - status: str ("ok" or "error")
                - elapsed_ms: float
                - args_preview: str (JSON)
                - timestamp: float

        Returns:
            List of newly detected/updated patterns.
        """
        if len(execution_records) < MIN_PATTERN_LENGTH:
            return []

        # Extract tool sequences of various window sizes
        tools = [r.get("tool", "") for r in execution_records]
        statuses = [r.get("status", "ok") for r in execution_records]
        elapsed_times = [r.get("elapsed_ms", 0) for r in execution_records]

        updated: list[DetectedPattern] = []

        for window_size in range(MIN_PATTERN_LENGTH, min(MAX_PATTERN_LENGTH + 1, len(tools) + 1)):
            for i in range(len(tools) - window_size + 1):
                seq = tuple(tools[i:i + window_size])

                # Skip if any tool is empty
                if any(not t for t in seq):
                    continue

                # Skip if sequence has duplicate consecutive tools (likely retry, not pattern)
                if any(seq[j] == seq[j + 1] for j in range(len(seq) - 1)):
                    continue

                pattern_hash = hashlib.md5(
                    "|".join(seq).encode(), usedforsecurity=False
                ).hexdigest()[:12]

                if pattern_hash not in self._patterns:
                    self._patterns[pattern_hash] = DetectedPattern(
                        tool_sequence=seq,
                        first_seen=time.time(),
                    )

                pattern = self._patterns[pattern_hash]
                pattern.occurrences += 1
                pattern.last_seen = time.time()

                # Track success/failure
                seq_statuses = statuses[i:i + window_size]
                if all(s == "ok" for s in seq_statuses):
                    pattern.successes += 1
                else:
                    pattern.failures += 1

                # Track timing
                seq_elapsed = elapsed_times[i:i + window_size]
                if seq_elapsed:
                    total_elapsed = sum(seq_elapsed)
                    # Track timing samples separately for accurate average
                    prev_samples = getattr(pattern, '_timing_samples', 0)
                    new_samples = prev_samples + len(seq_elapsed)
                    if prev_samples > 0:
                        pattern.avg_elapsed_ms = (
                            (pattern.avg_elapsed_ms * prev_samples + total_elapsed) / new_samples
                        )
                    else:
                        pattern.avg_elapsed_ms = total_elapsed / len(seq_elapsed)
                    pattern._timing_samples = new_samples

                # Keep sample args (last occurrence)
                sample = []
                for j in range(i, i + window_size):
                    args_str = execution_records[j].get("args_preview", "{}")
                    try:
                        sample.append(json.loads(args_str))
                    except (json.JSONDecodeError, TypeError):
                        sample.append({})
                pattern.sample_args = sample

                updated.append(pattern)

        return updated

    def find_crystallizable(self) -> list[DetectedPattern]:
        """Return all patterns ready to be crystallized into skills."""
        return [p for p in self._patterns.values() if p.is_crystallizable]

    # -- Crystallization ----------------------------------------------------

    def crystallize(self, pattern: DetectedPattern) -> SkillGenome | None:
        """Turn a detected pattern into a SkillGenome.

        Returns the genome if it passes security scanning, None if rejected.
        """
        if not pattern.is_crystallizable:
            logger.debug("Pattern %s not ready for crystallization", pattern.pattern_hash)
            return None

        # Build steps from pattern
        steps = []
        for i, tool_name in enumerate(pattern.tool_sequence):
            step = SkillStep(
                tool_name=tool_name,
                parameters=pattern.sample_args[i] if i < len(pattern.sample_args) else {},
                use_previous=i > 0,  # pipe output from previous step
            )
            steps.append(step)

        # Build genome
        genome = SkillGenome(
            name=self._generate_name(pattern),
            description=self._generate_description(pattern),
            creator_instance_id=self._instance_id,
            creator_user=self._user,
            steps=steps,
            declared_tools=list(pattern.tool_sequence),
            source_pattern="|".join(pattern.tool_sequence),
            tags=self._generate_tags(pattern),
        )

        # Compute capability tier from tools
        genome.compute_tier()

        # Initialize trust
        genome.trust = TrustScore(
            level=TrustLevel.SANDBOXED,
            local_successes=pattern.successes,
            local_failures=pattern.failures,
        )

        # Sign the genome
        genome.sign()

        # Security scan
        report = self._scanner.scan(genome)

        if report.verdict == ScanVerdict.REJECTED:
            logger.warning(
                "Crystallized skill '%s' REJECTED by scanner: %s",
                genome.name,
                "; ".join(f.description for f in report.findings),
            )
            return None

        if report.verdict == ScanVerdict.FLAGGED:
            logger.info(
                "Crystallized skill '%s' FLAGGED — needs user review: %s",
                genome.name,
                "; ".join(f.description for f in report.findings),
            )
            # Still create it, but mark as needing review
            genome.trust.level = TrustLevel.UNTRUSTED

        # Store as pending endorsement
        self._pending_endorsement[genome.id] = genome
        self._save()

        logger.info(
            "Crystallized skill: '%s' (tier=%s, steps=%d, pattern=%s)",
            genome.name,
            genome.capability_tier.name,
            len(genome.steps),
            genome.source_pattern,
        )
        return genome

    def crystallize_all(self) -> list[SkillGenome]:
        """Crystallize all ready patterns. Returns list of new genomes."""
        patterns = self.find_crystallizable()
        genomes = []
        for pattern in patterns:
            # Don't re-crystallize patterns that already have genomes
            existing = any(
                g.source_pattern == "|".join(pattern.tool_sequence)
                for g in list(self._crystallized.values()) + list(self._pending_endorsement.values())
            )
            if existing:
                continue

            genome = self.crystallize(pattern)
            if genome:
                genomes.append(genome)
        return genomes

    # -- Endorsement --------------------------------------------------------

    def endorse(self, genome_id: str) -> SkillGenome | None:
        """User endorses a skill — moves from pending to crystallized, ready to share."""
        genome = self._pending_endorsement.pop(genome_id, None)
        if not genome:
            logger.warning("Genome %s not found in pending endorsements", genome_id)
            return None

        genome.trust.user_endorsed = True
        genome.trust.level = TrustLevel.ENDORSED
        genome.sign()  # re-sign after endorsement

        self._crystallized[genome.id] = genome
        self._save()

        logger.info("User endorsed skill '%s' — ready for Flame", genome.name)
        return genome

    def reject_endorsement(self, genome_id: str) -> bool:
        """User rejects a pending skill — removes it."""
        genome = self._pending_endorsement.pop(genome_id, None)
        if genome:
            self._save()
            logger.info("User rejected skill '%s'", genome.name)
            return True
        return False

    # -- Skill management ---------------------------------------------------

    def get_crystallized(self) -> list[SkillGenome]:
        """Return all crystallized (endorsed) skills."""
        return list(self._crystallized.values())

    def get_pending(self) -> list[SkillGenome]:
        """Return skills pending user endorsement."""
        return list(self._pending_endorsement.values())

    def get_publishable(self) -> list[SkillGenome]:
        """Return skills that meet propagation requirements."""
        return [g for g in self._crystallized.values() if g.can_propagate()]

    def record_execution(self, genome_id: str, success: bool) -> None:
        """Record a skill execution result — updates trust score."""
        genome = self._crystallized.get(genome_id) or self._pending_endorsement.get(genome_id)
        if not genome:
            return
        genome.invocation_count += 1
        if success:
            genome.trust.record_success()
        else:
            genome.trust.record_failure()
        # Auto-quarantine check
        if genome.trust.should_quarantine:
            genome.trust.level = TrustLevel.QUARANTINED
            logger.warning("Skill '%s' auto-quarantined after failures", genome.name)
        self._save()

    # -- Persistence --------------------------------------------------------

    def _save(self) -> None:
        """Persist crystallized skills and pending endorsements to disk."""
        data = {
            "crystallized": {gid: g.to_dict() for gid, g in self._crystallized.items()},
            "pending": {gid: g.to_dict() for gid, g in self._pending_endorsement.items()},
            "patterns": {
                ph: {
                    "tool_sequence": list(p.tool_sequence),
                    "occurrences": p.occurrences,
                    "successes": p.successes,
                    "failures": p.failures,
                    "avg_elapsed_ms": p.avg_elapsed_ms,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                }
                for ph, p in self._patterns.items()
            },
        }
        path = self._data_dir / "evolution_state.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.error("Failed to save evolution state: %s", e)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load(self) -> None:
        """Load persisted evolution state from disk."""
        path = self._data_dir / "evolution_state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)

            for gid, gdata in data.get("crystallized", {}).items():
                self._crystallized[gid] = SkillGenome.from_dict(gdata)

            for gid, gdata in data.get("pending", {}).items():
                self._pending_endorsement[gid] = SkillGenome.from_dict(gdata)

            for ph, pdata in data.get("patterns", {}).items():
                self._patterns[ph] = DetectedPattern(
                    tool_sequence=tuple(pdata["tool_sequence"]),
                    occurrences=pdata.get("occurrences", 0),
                    successes=pdata.get("successes", 0),
                    failures=pdata.get("failures", 0),
                    avg_elapsed_ms=pdata.get("avg_elapsed_ms", 0),
                    first_seen=pdata.get("first_seen", 0),
                    last_seen=pdata.get("last_seen", 0),
                )

            logger.info(
                "Loaded evolution state: %d crystallized, %d pending, %d patterns",
                len(self._crystallized),
                len(self._pending_endorsement),
                len(self._patterns),
            )
        except Exception as e:
            logger.error("Failed to load evolution state: %s", e)

    # -- Name generation helpers --------------------------------------------

    def _generate_name(self, pattern: DetectedPattern) -> str:
        """Generate a human-readable name for a crystallized pattern."""
        tools = pattern.tool_sequence
        if len(tools) == 2:
            return f"{tools[0]}_then_{tools[1]}"
        first = tools[0]
        last = tools[-1]
        return f"{first}_to_{last}_{len(tools)}step"

    def _generate_description(self, pattern: DetectedPattern) -> str:
        """Generate a description for a crystallized pattern."""
        steps = " → ".join(pattern.tool_sequence)
        return (
            f"Auto-learned skill: {steps}. "
            f"Observed {pattern.occurrences} times with "
            f"{pattern.success_rate:.0%} success rate."
        )

    def _generate_tags(self, pattern: DetectedPattern) -> list[str]:
        """Generate tags based on the tools used."""
        tags = ["auto-learned", "crystallized"]
        tool_set = set(pattern.tool_sequence)
        if tool_set & {"web_search", "web_scrape", "deep_search"}:
            tags.append("web")
        if tool_set & {"read_file", "write_file", "list_directory"}:
            tags.append("files")
        if tool_set & {"git_context", "git_find_files", "git_diff_summary"}:
            tags.append("git")
        if tool_set & {"memory_store", "memory_recall"}:
            tags.append("memory")
        if tool_set & {"run_command", "python_exec", "execute_code"}:
            tags.append("execution")
        return tags

    # -- Stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return evolution statistics."""
        return {
            "patterns_detected": len(self._patterns),
            "patterns_crystallizable": len(self.find_crystallizable()),
            "skills_crystallized": len(self._crystallized),
            "skills_pending_endorsement": len(self._pending_endorsement),
            "skills_publishable": len(self.get_publishable()),
            "top_patterns": [
                {
                    "sequence": " → ".join(p.tool_sequence),
                    "occurrences": p.occurrences,
                    "success_rate": f"{p.success_rate:.0%}",
                }
                for p in sorted(
                    self._patterns.values(),
                    key=lambda p: p.occurrences,
                    reverse=True,
                )[:10]
            ],
        }
