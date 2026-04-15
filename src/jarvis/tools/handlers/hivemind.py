"""
Flame tool handlers — skill evolution, scanning, sync, endorsement.

Tools:
  hivemind_status  — show Flame stats
  hivemind_sync    — pull/push skills from shared pool
  skill_evolve     — detect patterns + crystallize skills
  skill_scan       — security scan a genome
  skill_endorse    — approve/reject pending skills
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ._context import ToolContext, ToolError, ToolErrorKind

logger = logging.getLogger(__name__)


def _get_crystallizer(ctx: ToolContext):
    """Lazy-init the SkillCrystallizer singleton."""
    if not hasattr(ctx, "_skill_crystallizer") or ctx._skill_crystallizer is None:
        from jarvis._vendor.common.skill_evolution import SkillCrystallizer
        ctx._skill_crystallizer = SkillCrystallizer()
    return ctx._skill_crystallizer


def _get_flame(ctx: ToolContext):
    """Lazy-init the HiveMind singleton."""
    if not hasattr(ctx, "_skill_flame") or ctx._skill_flame is None:
        from jarvis._vendor.common.skill_hivemind import Flame
        ctx._skill_flame = Flame()
    return ctx._skill_flame


def _get_scanner(ctx: ToolContext):
    """Lazy-init the SkillScanner singleton."""
    if not hasattr(ctx, "_skill_scanner") or ctx._skill_scanner is None:
        from jarvis._vendor.common.skill_scanner import SkillScanner
        ctx._skill_scanner = SkillScanner()
    return ctx._skill_scanner


async def handle_hivemind_status(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Show Flame status: local skills, shared pool, trust distribution."""
    try:
        hivemind = _get_flame(ctx)
        crystallizer = _get_crystallizer(ctx)

        hive_stats = hivemind.stats()
        evo_stats = crystallizer.stats()

        result = {
            "hivemind": hive_stats,
            "evolution": evo_stats,
        }
        return json.dumps(result, indent=2, default=str)
    except (RuntimeError, ImportError, OSError, ValueError) as e:
        logger.error("hivemind_status failed: %s", e)
        raise ToolError(f"Flame status failed: {e}", kind=ToolErrorKind.EXECUTION, tool_name="hivemind_status") from e


async def handle_hivemind_sync(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Sync with the shared skill pool."""
    try:
        hivemind = _get_flame(ctx)
        result = hivemind.sync()
        return json.dumps(result, indent=2, default=str)
    except (RuntimeError, ImportError, OSError, ConnectionError) as e:
        logger.error("hivemind_sync failed: %s", e)
        raise ToolError(f"Flame sync failed: {e}", kind=ToolErrorKind.EXECUTION, tool_name="hivemind_sync") from e


async def handle_skill_evolve(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Detect patterns and crystallize skills."""
    action = str(args.get("action", "stats")).strip().lower()
    crystallizer = _get_crystallizer(ctx)

    try:
        if action == "detect":
            # Get recent execution history and observe patterns
            from ...tools.resilience import ExecutionHistory
            history = getattr(ctx, "execution_history", None)
            if history is None:
                return json.dumps({"status": "no execution history available"})

            records = history.recent(100)
            patterns = crystallizer.observe(records)
            crystallizable = crystallizer.find_crystallizable()

            return json.dumps({
                "patterns_found": len(crystallizer._patterns),
                "crystallizable": [
                    {
                        "sequence": " → ".join(p.tool_sequence),
                        "occurrences": p.occurrences,
                        "success_rate": f"{p.success_rate:.0%}",
                    }
                    for p in crystallizable
                ],
            }, indent=2)

        elif action == "crystallize":
            genomes = crystallizer.crystallize_all()
            return json.dumps({
                "crystallized": len(genomes),
                "skills": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "tier": g.capability_tier.name,
                        "steps": len(g.steps),
                        "pattern": g.source_pattern,
                    }
                    for g in genomes
                ],
            }, indent=2)

        elif action == "list":
            crystallized = crystallizer.get_crystallized()
            pending = crystallizer.get_pending()
            return json.dumps({
                "crystallized": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "tier": g.capability_tier.name,
                        "trust": g.trust.level.value,
                        "successes": g.trust.local_successes,
                        "can_propagate": g.can_propagate(),
                    }
                    for g in crystallized
                ],
                "pending_endorsement": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "tier": g.capability_tier.name,
                        "pattern": g.source_pattern,
                    }
                    for g in pending
                ],
            }, indent=2)

        elif action == "stats":
            return json.dumps(crystallizer.stats(), indent=2, default=str)

        else:
            raise ToolError(f"Unknown action: {action}. Use: detect, crystallize, list, stats", kind=ToolErrorKind.VALIDATION, tool_name="skill_evolve")

    except (RuntimeError, ImportError, OSError, ValueError, KeyError) as e:
        logger.error("skill_evolve failed: %s", e)
        raise ToolError(f"Skill evolution failed: {e}", kind=ToolErrorKind.EXECUTION, tool_name="skill_evolve") from e


async def handle_skill_scan(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Security scan a skill genome."""
    genome_id = args.get("genome_id", "")
    if not genome_id:
        raise ToolError("Missing genome_id parameter", kind=ToolErrorKind.VALIDATION, tool_name="skill_scan")

    try:
        scanner = _get_scanner(ctx)
        crystallizer = _get_crystallizer(ctx)

        # Find the genome
        genome = None
        for g in crystallizer.get_crystallized() + crystallizer.get_pending():
            if g.id == genome_id:
                genome = g
                break

        if not genome:
            # Check Flame
            hivemind = _get_flame(ctx)
            genome = hivemind.get_skill(genome_id)

        if not genome:
            raise ToolError(f"Genome '{genome_id}' not found", kind=ToolErrorKind.NOT_FOUND, tool_name="skill_scan")

        report = scanner.scan(genome)
        return json.dumps(report.to_dict(), indent=2, default=str)

    except (RuntimeError, ImportError, OSError, ValueError) as e:
        logger.error("skill_scan failed: %s", e)
        raise ToolError(f"Skill scan failed: {e}", kind=ToolErrorKind.EXECUTION, tool_name="skill_scan") from e


async def handle_skill_endorse(
    args: dict[str, Any], ctx: ToolContext
) -> str:
    """Endorse or reject a pending skill."""
    genome_id = args.get("genome_id", "")
    action = str(args.get("action", "endorse")).strip().lower()

    if not genome_id:
        raise ToolError("Missing genome_id parameter", kind=ToolErrorKind.VALIDATION, tool_name="skill_scan")

    try:
        crystallizer = _get_crystallizer(ctx)

        if action == "endorse":
            genome = crystallizer.endorse(genome_id)
            if not genome:
                raise ToolError(f"Genome '{genome_id}' not found in pending endorsements", kind=ToolErrorKind.NOT_FOUND, tool_name="skill_endorse")

            # Auto-publish to Flame if it meets propagation requirements
            hivemind = _get_flame(ctx)
            if genome.can_propagate():
                result = hivemind.publish(genome)
                return json.dumps({
                    "endorsed": True,
                    "name": genome.name,
                    "published": result.get("status") == "published",
                    "publish_details": result,
                }, indent=2, default=str)

            return json.dumps({
                "endorsed": True,
                "name": genome.name,
                "published": False,
                "reason": "Skill endorsed but needs more successes before propagation",
                "tier": genome.capability_tier.name,
                "successes": genome.trust.local_successes,
            }, indent=2, default=str)

        elif action == "reject":
            success = crystallizer.reject_endorsement(genome_id)
            return json.dumps({
                "rejected": success,
                "genome_id": genome_id,
            }, indent=2)

        else:
            raise ToolError(f"Unknown action: {action}. Use: endorse, reject", kind=ToolErrorKind.VALIDATION, tool_name="skill_endorse")

    except (RuntimeError, ImportError, OSError, ValueError, KeyError) as e:
        logger.error("skill_endorse failed: %s", e)
        raise ToolError(f"Skill endorsement failed: {e}", kind=ToolErrorKind.EXECUTION, tool_name="skill_endorse") from e
