"""
The Flame — Prometheus's shared skill network.

Fire stolen from the gods, given to all.  Every JARVIS instance that learns
a useful skill can share it through the Flame.  Every other instance
auto-receives verified skills and benefits from the collective learning.
The more agents that learn, the brighter the Flame burns.

Architecture:
  - Local pool:  ~/.prometheus/flame/local/   (per-instance storage)
  - Shared pool: ~/.prometheus/flame/shared/  (collective repository)
  - Sync:        Periodic pull from shared → local, push endorsed → shared

Security: THREE scan points.
  1. On publish  — creator scans before sharing
  2. At the pool — Flame scans on arrival (duplicate detection, reputation check)
  3. On receive  — receiver scans before sandbox trial

Trust: Collective reputation.
  - Each instance reports success/failure for shared skills
  - One quarantine report from ANY instance drops the skill's reputation
  - Below threshold → auto-recalled from all instances

Self-improvement is shared.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .skill_genome import (
    SkillGenome,
    TrustLevel,
    TrustScore,
    TIER_PROPAGATION,
)
from .skill_scanner import ScanReport, ScanVerdict, SkillScanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reputation thresholds
# ---------------------------------------------------------------------------

# Minimum network reputation to stay in the pool
MIN_REPUTATION_SCORE = 30.0

# Quarantine reports needed to auto-recall from all instances
QUARANTINE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------


@dataclass
class SyncStatus:
    """Tracks sync state between local and shared pool."""
    last_push: float = 0.0
    last_pull: float = 0.0
    skills_pushed: int = 0
    skills_pulled: int = 0
    skills_quarantined: int = 0
    skills_recalled: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_push": self.last_push,
            "last_pull": self.last_pull,
            "skills_pushed": self.skills_pushed,
            "skills_pulled": self.skills_pulled,
            "skills_quarantined": self.skills_quarantined,
            "skills_recalled": self.skills_recalled,
        }


# ---------------------------------------------------------------------------
# Hive Mind
# ---------------------------------------------------------------------------


class Flame:
    """Shared skill pool with auto-sync and collective reputation.

    Each JARVIS instance has a local copy of the Flame.  Skills flow:
      Local endorsed → shared pool → other instances' local pools.

    The shared pool can be backed by:
      - Filesystem (default: ~/.prometheus/flame/shared/)
      - Git repository (for distributed teams)
      - HTTP API (for cloud-hosted hive)

    This implementation uses filesystem backend. The interface is designed
    so other backends can be swapped in without changing the rest of the system.
    """

    def __init__(
        self,
        instance_id: str | None = None,
        local_dir: str | Path | None = None,
        shared_dir: str | Path | None = None,
    ):
        self._instance_id = instance_id or os.getenv("JARVIS_INSTANCE_ID", "local")

        # Local pool — this instance's received skills
        self._local_dir = Path(
            local_dir or "~/.prometheus/flame/local"
        ).expanduser()
        self._local_dir.mkdir(parents=True, exist_ok=True)

        # Shared pool — the collective skill repository
        self._shared_dir = Path(
            shared_dir or "~/.prometheus/flame/shared"
        ).expanduser()
        self._shared_dir.mkdir(parents=True, exist_ok=True)

        self._scanner = SkillScanner()
        self._sync_status = SyncStatus()

        # In-memory indexes
        self._local_skills: dict[str, SkillGenome] = {}
        self._reputation: dict[str, dict[str, Any]] = {}  # genome_id → reputation data
        self._shared_index: dict[str, str] = {}  # source_pattern → genome_id

        self._load_local()
        self._load_reputation()
        self._rebuild_shared_index()

    # -- Publishing (creator → shared pool) ---------------------------------

    def publish(self, genome: SkillGenome) -> dict[str, Any]:
        """Publish a skill to the shared pool.

        Scan point 1: Creator-side scan (should already be done).
        Scan point 2: Pool-side scan on arrival.

        Returns dict with status and details.
        """
        # Verify the skill is endorsed and meets propagation requirements
        if not genome.trust.user_endorsed:
            return {"status": "rejected", "reason": "Skill not endorsed by user"}

        if not genome.can_propagate():
            rules = TIER_PROPAGATION.get(genome.capability_tier, {})
            return {
                "status": "rejected",
                "reason": (
                    f"Skill doesn't meet tier {genome.capability_tier.name} "
                    f"propagation requirements: needs {rules.get('min_successes', '?')} "
                    f"successes, has {genome.trust.local_successes}"
                ),
            }

        # Verify signature
        if not genome.verify_signature():
            return {"status": "rejected", "reason": "Invalid signature — genome tampered"}

        # SCAN POINT 2: Pool-side scan
        report = self._scanner.scan(genome)
        if report.verdict == ScanVerdict.REJECTED:
            return {
                "status": "rejected",
                "reason": "Failed pool-side security scan",
                "findings": [f.description for f in report.findings],
            }

        # Check for duplicates
        existing = self._find_in_shared(genome.source_pattern)
        if existing:
            # Keep the one with higher trust
            if existing.trust.network_score >= genome.trust.local_success_rate * 100:
                return {"status": "duplicate", "existing_id": existing.id}

        # Write to shared pool
        self._write_to_shared(genome)
        self._shared_index[genome.source_pattern] = genome.id

        # Initialize reputation
        self._reputation[genome.id] = {
            "creator": self._instance_id,
            "published_at": time.time(),
            "reports": [],  # {instance_id, success: bool, timestamp}
            "quarantine_votes": [],
        }
        self._save_reputation()

        self._sync_status.skills_pushed += 1
        self._sync_status.last_push = time.time()

        logger.info(
            "Published skill '%s' to Flame (tier=%s, trust=%s)",
            genome.name,
            genome.capability_tier.name,
            genome.trust.level.value,
        )

        return {
            "status": "published",
            "genome_id": genome.id,
            "name": genome.name,
            "tier": genome.capability_tier.name,
        }

    # -- Receiving (shared pool → local) ------------------------------------

    def sync(self) -> dict[str, Any]:
        """Pull new skills from shared pool into local.

        Scan point 3: Receiver-side scan before sandbox.
        """
        new_skills = 0
        quarantined = 0
        recalled = 0

        # Scan shared pool for skills not in local
        shared_genomes = self._list_shared()

        for genome in shared_genomes:
            # Skip own skills
            if genome.creator_instance_id == self._instance_id:
                continue

            # Skip already received
            if genome.id in self._local_skills:
                # But check for recalls
                if self._should_recall(genome.id):
                    self._recall_skill(genome.id)
                    recalled += 1
                continue

            # Check reputation before scanning
            rep = self._get_reputation(genome.id)
            if rep.get("score", 50) < MIN_REPUTATION_SCORE:
                logger.info(
                    "Skipping skill '%s' — reputation too low (%.1f)",
                    genome.name,
                    rep.get("score", 0),
                )
                continue

            # SCAN POINT 3: Receiver-side scan
            report = self._scanner.scan(genome)

            if report.verdict == ScanVerdict.REJECTED:
                logger.warning(
                    "Rejected incoming skill '%s' — failed security scan",
                    genome.name,
                )
                self._report_quarantine(genome.id)
                quarantined += 1
                continue

            # Accept into local pool (sandboxed)
            genome.trust = TrustScore(
                level=TrustLevel.SANDBOXED,
                local_successes=0,
                local_failures=0,
                network_successes=genome.trust.network_successes,
                network_failures=genome.trust.network_failures,
                network_quarantines=genome.trust.network_quarantines,
            )

            self._local_skills[genome.id] = genome
            self._save_local_skill(genome)
            new_skills += 1

            logger.info(
                "Received skill '%s' from Flame → sandboxed (tier=%s)",
                genome.name,
                genome.capability_tier.name,
            )

        # Check for skills to recall
        for gid in list(self._local_skills.keys()):
            if self._should_recall(gid):
                self._recall_skill(gid)
                recalled += 1

        self._sync_status.skills_pulled += new_skills
        self._sync_status.skills_quarantined += quarantined
        self._sync_status.skills_recalled += recalled
        self._sync_status.last_pull = time.time()

        return {
            "new_skills": new_skills,
            "quarantined": quarantined,
            "recalled": recalled,
            "total_local": len(self._local_skills),
        }

    # -- Reputation system --------------------------------------------------

    def report_success(self, genome_id: str) -> None:
        """Report a successful skill execution to the Flame."""
        if genome_id not in self._reputation:
            self._reputation[genome_id] = {"reports": [], "quarantine_votes": []}

        self._reputation[genome_id]["reports"].append({
            "instance_id": self._instance_id,
            "success": True,
            "timestamp": time.time(),
        })
        self._save_reputation()

        # Update local trust
        if genome_id in self._local_skills:
            self._local_skills[genome_id].trust.record_success()
            self._local_skills[genome_id].trust.network_successes += 1
            self._promote_if_ready(genome_id)
            self._save_local_skill(self._local_skills[genome_id])

    def report_failure(self, genome_id: str) -> None:
        """Report a failed skill execution to the Flame."""
        if genome_id not in self._reputation:
            self._reputation[genome_id] = {"reports": [], "quarantine_votes": []}

        self._reputation[genome_id]["reports"].append({
            "instance_id": self._instance_id,
            "success": False,
            "timestamp": time.time(),
        })
        self._save_reputation()

        # Update local trust
        if genome_id in self._local_skills:
            self._local_skills[genome_id].trust.record_failure()
            self._local_skills[genome_id].trust.network_failures += 1
            # Check for auto-quarantine
            if self._local_skills[genome_id].trust.should_quarantine:
                self._quarantine_skill(genome_id)
            self._save_local_skill(self._local_skills[genome_id])

    def report_quarantine(self, genome_id: str) -> None:
        """Report that this instance quarantined a skill — affects global reputation."""
        self._report_quarantine(genome_id)

    def _report_quarantine(self, genome_id: str) -> None:
        if genome_id not in self._reputation:
            self._reputation[genome_id] = {"reports": [], "quarantine_votes": []}

        votes = self._reputation[genome_id].setdefault("quarantine_votes", [])
        # Only count one vote per instance
        if not any(v.get("instance_id") == self._instance_id for v in votes):
            votes.append({
                "instance_id": self._instance_id,
                "timestamp": time.time(),
            })

        self._save_reputation()

        # Check if we need to recall globally
        if len(votes) >= QUARANTINE_THRESHOLD:
            logger.warning(
                "Skill %s reached quarantine threshold (%d votes) — recalling globally",
                genome_id,
                len(votes),
            )

    # -- Trust progression --------------------------------------------------

    def _promote_if_ready(self, genome_id: str) -> None:
        """Promote a skill's trust level based on local success."""
        genome = self._local_skills.get(genome_id)
        if not genome:
            return

        trust = genome.trust

        if trust.level == TrustLevel.SANDBOXED and trust.local_successes >= 5:
            trust.level = TrustLevel.LIMITED
            logger.info("Skill '%s' promoted: SANDBOXED → LIMITED", genome.name)

        elif trust.level == TrustLevel.LIMITED and trust.local_successes >= 15:
            if trust.local_success_rate >= 0.9:
                trust.level = TrustLevel.TRUSTED
                logger.info("Skill '%s' promoted: LIMITED → TRUSTED", genome.name)

    def _quarantine_skill(self, genome_id: str) -> None:
        """Quarantine a skill locally and report to Flame."""
        genome = self._local_skills.get(genome_id)
        if genome:
            genome.trust.level = TrustLevel.QUARANTINED
            self._save_local_skill(genome)
            self._report_quarantine(genome_id)
            logger.warning("Quarantined skill '%s' locally", genome.name)

    def _should_recall(self, genome_id: str) -> bool:
        """Check if a skill should be recalled based on global reputation."""
        rep = self._reputation.get(genome_id, {})
        votes = rep.get("quarantine_votes", [])
        return len(votes) >= QUARANTINE_THRESHOLD

    def _recall_skill(self, genome_id: str) -> None:
        """Remove a skill from local pool due to global recall."""
        genome = self._local_skills.pop(genome_id, None)
        if genome:
            # Remove local file
            skill_path = self._local_dir / f"{genome_id}.json"
            skill_path.unlink(missing_ok=True)
            self._sync_status.skills_recalled += 1
            logger.warning(
                "RECALLED skill '%s' — quarantined by %d instances",
                genome.name,
                len(self._reputation.get(genome_id, {}).get("quarantine_votes", [])),
            )

    # -- Local pool management ----------------------------------------------

    def get_local_skills(self) -> list[SkillGenome]:
        """Return all skills in the local Flame pool pool."""
        return list(self._local_skills.values())

    def get_executable_skills(self) -> list[SkillGenome]:
        """Return skills trusted enough to execute (not quarantined/untrusted)."""
        executable_levels = {TrustLevel.SANDBOXED, TrustLevel.LIMITED, TrustLevel.TRUSTED, TrustLevel.ENDORSED}
        return [
            g for g in self._local_skills.values()
            if g.trust.level in executable_levels
        ]

    def get_skill(self, genome_id: str) -> SkillGenome | None:
        """Get a specific skill from local pool."""
        return self._local_skills.get(genome_id)

    # -- Filesystem persistence (local) -------------------------------------

    def _save_local_skill(self, genome: SkillGenome) -> None:
        path = self._local_dir / f"{genome.id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(genome.to_dict(), f, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.error("Failed to save local skill %s: %s", genome.id, e)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_local(self) -> None:
        for path in self._local_dir.glob("*.json"):
            if path.name == "reputation.json":
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                genome = SkillGenome.from_dict(data)
                self._local_skills[genome.id] = genome
            except Exception as e:
                logger.error("Failed to load local skill %s: %s", path.name, e)

        if self._local_skills:
            logger.info("Loaded %d skills from local Flame pool", len(self._local_skills))

    # -- Filesystem persistence (shared pool) -------------------------------

    def _write_to_shared(self, genome: SkillGenome) -> None:
        path = self._shared_dir / f"{genome.id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(genome.to_dict(), f, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.error("Failed to write to shared pool %s: %s", genome.id, e)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _list_shared(self) -> list[SkillGenome]:
        genomes = []
        for path in self._shared_dir.glob("*.json"):
            if path.name == "reputation.json":
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                genomes.append(SkillGenome.from_dict(data))
            except Exception as e:
                logger.error("Failed to load shared skill %s: %s", path.name, e)
        return genomes

    def _rebuild_shared_index(self) -> None:
        """Build in-memory index of shared pool for O(1) duplicate lookups."""
        self._shared_index.clear()
        for path in self._shared_dir.glob("*.json"):
            if path.name == "reputation.json":
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                sp = data.get("source_pattern", "")
                gid = data.get("id", path.stem)
                if sp:
                    self._shared_index[sp] = gid
            except Exception:
                pass

    def _find_in_shared(self, source_pattern: str) -> SkillGenome | None:
        gid = self._shared_index.get(source_pattern)
        if not gid:
            return None
        path = self._shared_dir / f"{gid}.json"
        if not path.exists():
            self._shared_index.pop(source_pattern, None)
            return None
        try:
            with open(path) as f:
                return SkillGenome.from_dict(json.load(f))
        except Exception:
            return None

    # -- Reputation persistence ---------------------------------------------

    def _save_reputation(self) -> None:
        path = self._shared_dir / "reputation.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._reputation, f, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.error("Failed to save reputation: %s", e)
            # Clean up temp file on failure
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_reputation(self) -> None:
        path = self._shared_dir / "reputation.json"
        if path.exists():
            try:
                with open(path) as f:
                    self._reputation = json.load(f)
            except Exception as e:
                logger.error("Failed to load reputation: %s", e)

    def _get_reputation(self, genome_id: str) -> dict[str, Any]:
        rep = self._reputation.get(genome_id, {})
        reports = rep.get("reports", [])
        quarantine_votes = rep.get("quarantine_votes", [])

        total = len(reports)
        successes = sum(1 for r in reports if r.get("success"))
        base_score = (successes / total * 100) if total > 0 else 50.0
        penalty = len(quarantine_votes) * 15

        return {
            "score": max(0, base_score - penalty),
            "total_reports": total,
            "successes": successes,
            "quarantine_votes": len(quarantine_votes),
        }

    # -- Stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return Flame statistics."""
        shared_count = len(list(self._shared_dir.glob("*.json"))) - (
            1 if (self._shared_dir / "reputation.json").exists() else 0
        )
        return {
            "instance_id": self._instance_id,
            "local_skills": len(self._local_skills),
            "shared_pool_skills": shared_count,
            "executable_skills": len(self.get_executable_skills()),
            "sync": self._sync_status.to_dict(),
            "trust_distribution": {
                level.value: sum(
                    1 for g in self._local_skills.values()
                    if g.trust.level == level
                )
                for level in TrustLevel
            },
            "scanner": self._scanner.stats(),
        }
