"""
Skill Genome — Capability manifest, cryptographic signature, and trust scoring.

Every skill in the Flame carries a genome: a complete description of what
it can do, what tools it needs, who created it, and how much the network trusts
it.  The genome is the unit of security — scanners inspect it, the sandbox
enforces it, and trust scores determine propagation speed.

Capability tiers:
  Tier 0 — Pure logic         (no permissions, auto-propagates fast)
  Tier 1 — Local read         (reads files/git, needs user endorsement)
  Tier 2 — Local write        (writes files, needs 20+ successes)
  Tier 3 — Network read       (web search/scrape, manual approve)
  Tier 4 — Network write      (send email/post, both users must approve)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

logger = logging.getLogger(__name__)

# Secret used for HMAC signing skill genomes.
#
# Production deploys MUST set PREDACORE_SKILL_SIGNING_SECRET to a stable
# value shared across the fleet — otherwise genomes signed in dev/CI fail
# to verify in prod (containers vs host vs CI all derive different USER +
# HOME values, so any machine-local fallback would silently invalidate
# signatures across environments).
#
# We retain the machine-derived fallback for single-developer setups so
# `python -c "from predacore._vendor.common.skill_genome import ..."` works
# out of the box, but we LOG LOUDLY when it kicks in, and we REFUSE to
# start with the fallback when PREDACORE_ENV=production.

def _resolve_signing_secret() -> bytes:
    explicit = os.getenv("PREDACORE_SKILL_SIGNING_SECRET", "").encode("utf-8")
    if explicit:
        return explicit
    env = os.getenv("PREDACORE_ENV", "").strip().lower()
    if env in ("production", "prod"):
        raise RuntimeError(
            "PREDACORE_SKILL_SIGNING_SECRET is required in production. "
            "Set it to a stable value shared across all instances; without "
            "it, skill signatures will not verify across containers/CI."
        )
    logger.warning(
        "PREDACORE_SKILL_SIGNING_SECRET not set — falling back to a "
        "machine-derived key (USER + HOME). Skills signed on this machine "
        "will NOT verify on others. Set the env var for fleet-wide signing."
    )
    return hashlib.sha256(
        (os.getenv("USER", "") + "-" + str(pathlib.Path.home())).encode()
    ).digest()


_SIGNING_SECRET = _resolve_signing_secret()


# ---------------------------------------------------------------------------
# Capability tiers
# ---------------------------------------------------------------------------


class CapabilityTier(IntEnum):
    """Risk tiers for skill capabilities.  Higher = more verification needed."""
    PURE_LOGIC = 0       # No tool access, just transforms data
    LOCAL_READ = 1       # Reads local data (files, git, memory)
    LOCAL_WRITE = 2      # Writes local data (files, configs)
    NETWORK_READ = 3     # Reads from network (web search, API GET)
    NETWORK_WRITE = 4    # Writes to network (email, POST, push)


# Map tools to their capability tiers
TOOL_TIER_MAP: dict[str, CapabilityTier] = {
    # Tier 0 — pure logic (no tool access needed)

    # Tier 1 — local read
    "read_file": CapabilityTier.LOCAL_READ,
    "list_directory": CapabilityTier.LOCAL_READ,
    "git_context": CapabilityTier.LOCAL_READ,
    "git_find_files": CapabilityTier.LOCAL_READ,
    "git_diff_summary": CapabilityTier.LOCAL_READ,
    "git_semantic_search": CapabilityTier.LOCAL_READ,
    "memory_recall": CapabilityTier.LOCAL_READ,
    "semantic_search": CapabilityTier.LOCAL_READ,
    "pdf_reader": CapabilityTier.LOCAL_READ,
    "identity_read": CapabilityTier.LOCAL_READ,
    "tool_stats": CapabilityTier.LOCAL_READ,

    # Tier 2 — local write
    "write_file": CapabilityTier.LOCAL_WRITE,
    "run_command": CapabilityTier.LOCAL_WRITE,
    "python_exec": CapabilityTier.LOCAL_WRITE,
    "execute_code": CapabilityTier.LOCAL_WRITE,
    "memory_store": CapabilityTier.LOCAL_WRITE,
    "identity_update": CapabilityTier.LOCAL_WRITE,
    "journal_append": CapabilityTier.LOCAL_WRITE,
    "git_commit_suggest": CapabilityTier.LOCAL_WRITE,

    # Tier 3 — network read
    "web_search": CapabilityTier.NETWORK_READ,
    "web_scrape": CapabilityTier.NETWORK_READ,
    "deep_search": CapabilityTier.NETWORK_READ,
    "browser_automation": CapabilityTier.NETWORK_READ,

    # Tier 4 — network write
    "speak": CapabilityTier.NETWORK_WRITE,
    "voice_note": CapabilityTier.NETWORK_WRITE,
    "image_gen": CapabilityTier.NETWORK_WRITE,
    "desktop_control": CapabilityTier.NETWORK_WRITE,
    "android_control": CapabilityTier.NETWORK_WRITE,
    "multi_agent": CapabilityTier.NETWORK_WRITE,
    "openclaw_delegate": CapabilityTier.NETWORK_WRITE,
}


# Propagation rules per tier
TIER_PROPAGATION: dict[CapabilityTier, dict[str, Any]] = {
    CapabilityTier.PURE_LOGIC: {
        "min_successes": 5,
        "requires_user_endorsement": False,
        "requires_receiver_approval": False,
        "auto_propagate": True,
    },
    CapabilityTier.LOCAL_READ: {
        "min_successes": 10,
        "requires_user_endorsement": True,
        "requires_receiver_approval": False,
        "auto_propagate": True,
    },
    CapabilityTier.LOCAL_WRITE: {
        "min_successes": 20,
        "requires_user_endorsement": True,
        "requires_receiver_approval": False,
        "auto_propagate": False,  # semi-auto
    },
    CapabilityTier.NETWORK_READ: {
        "min_successes": 30,
        "requires_user_endorsement": True,
        "requires_receiver_approval": True,
        "auto_propagate": False,
    },
    CapabilityTier.NETWORK_WRITE: {
        "min_successes": 50,
        "requires_user_endorsement": True,
        "requires_receiver_approval": True,
        "auto_propagate": False,
    },
}


# Sensitive paths that skills should NEVER access
SENSITIVE_PATHS = frozenset({
    ".env", ".env.local", ".env.production",
    "credentials.json", "credentials.yaml",
    "secrets.yaml", "secrets.json",
    ".ssh/", ".gnupg/", ".aws/",
    "id_rsa", "id_ed25519",
    ".predacore/.env",
    "token.json", "keystore",
})


# ---------------------------------------------------------------------------
# Trust Score
# ---------------------------------------------------------------------------


class TrustLevel(str, Enum):
    """Trust levels for skills in the Flame."""
    QUARANTINED = "quarantined"    # Flagged by scanner or failed at runtime
    UNTRUSTED = "untrusted"       # Just arrived, not yet tested
    SANDBOXED = "sandboxed"       # Running in sandbox trial
    LIMITED = "limited"           # Passed sandbox, limited capabilities
    TRUSTED = "trusted"           # Fully trusted, all declared capabilities
    ENDORSED = "endorsed"         # User explicitly endorsed for sharing


@dataclass
class TrustScore:
    """Tracks trust metrics for a skill."""
    level: TrustLevel = TrustLevel.UNTRUSTED
    local_successes: int = 0
    local_failures: int = 0
    network_successes: int = 0     # successes across the Flame
    network_failures: int = 0
    network_quarantines: int = 0   # how many instances quarantined this skill
    user_endorsed: bool = False
    first_seen: float = field(default_factory=time.time)
    last_success: float = 0.0
    last_failure: float = 0.0

    @property
    def local_success_rate(self) -> float:
        total = self.local_successes + self.local_failures
        return self.local_successes / total if total > 0 else 0.0

    @property
    def network_score(self) -> float:
        """0-100 reputation score across the Flame."""
        total = self.network_successes + self.network_failures
        if total == 0:
            return 50.0  # neutral — no data yet
        base = (self.network_successes / total) * 100
        # Each quarantine report drops score by 15 points
        penalty = self.network_quarantines * 15
        return max(0.0, min(100.0, base - penalty))

    @property
    def should_quarantine(self) -> bool:
        """Auto-quarantine if network score drops too low or local failures spike."""
        if self.network_score < 30:
            return True
        if self.local_failures >= 5 and self.local_success_rate < 0.3:
            return True
        return False

    def record_success(self) -> None:
        self.local_successes += 1
        self.last_success = time.time()

    def record_failure(self) -> None:
        self.local_failures += 1
        self.last_failure = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "local_successes": self.local_successes,
            "local_failures": self.local_failures,
            "local_success_rate": f"{self.local_success_rate:.1%}",
            "network_score": round(self.network_score, 1),
            "network_quarantines": self.network_quarantines,
            "user_endorsed": self.user_endorsed,
        }


# ---------------------------------------------------------------------------
# Skill Genome
# ---------------------------------------------------------------------------


@dataclass
class SkillStep:
    """A single step in a skill recipe."""
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None       # optional: only run if condition met
    use_previous: bool = False         # pipe previous step's output as input


@dataclass
class SkillGenome:
    """Complete genome of a skill — the unit of sharing in the Flame.

    A genome is a recipe (sequence of tool calls) + metadata + security info.
    It's NOT code. It's a declarative description of what tools to call and how.
    """
    # Identity
    id: str = field(default_factory=lambda: f"skill_{uuid.uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # Origin
    creator_instance_id: str = ""      # which PredaCore instance created this
    creator_user: str = ""             # which user's PredaCore
    created_at: float = field(default_factory=time.time)

    # The recipe — sequence of tool calls
    steps: list[SkillStep] = field(default_factory=list)

    # Capability manifest — what this skill needs
    declared_tools: list[str] = field(default_factory=list)
    capability_tier: CapabilityTier = CapabilityTier.PURE_LOGIC

    # Security
    signature: str = ""                # HMAC signature of the genome
    trust: TrustScore = field(default_factory=TrustScore)

    # Metadata
    tags: list[str] = field(default_factory=list)
    source_pattern: str = ""           # the execution pattern that was crystallized
    invocation_count: int = 0

    def compute_tier(self) -> CapabilityTier:
        """Compute capability tier from declared tools (highest tier wins)."""
        if not self.declared_tools:
            return CapabilityTier.PURE_LOGIC
        max_tier = CapabilityTier.PURE_LOGIC
        for tool in self.declared_tools:
            tier = TOOL_TIER_MAP.get(tool, CapabilityTier.LOCAL_READ)
            if tier > max_tier:
                max_tier = tier
        self.capability_tier = max_tier
        return max_tier

    def sign(self, secret: bytes | None = None) -> str:
        """Sign the genome with HMAC — proves integrity and origin."""
        secret = secret or _SIGNING_SECRET
        payload = self._signable_payload()
        sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        self.signature = sig
        return sig

    def verify_signature(self, secret: bytes | None = None) -> bool:
        """Verify the genome's signature hasn't been tampered with."""
        secret = secret or _SIGNING_SECRET
        payload = self._signable_payload()
        expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def _signable_payload(self) -> str:
        """Create deterministic string for signing."""
        steps_data = [
            {"tool": s.tool_name, "params": s.parameters, "cond": s.condition}
            for s in self.steps
        ]
        return json.dumps({
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "creator": self.creator_instance_id,
            "steps": steps_data,
            "declared_tools": sorted(self.declared_tools),
            "tier": self.capability_tier.value,
        }, sort_keys=True, default=str)

    def can_propagate(self) -> bool:
        """Check if this skill meets propagation requirements for its tier."""
        rules = TIER_PROPAGATION.get(self.capability_tier, {})
        min_successes = rules.get("min_successes", 50)
        needs_endorsement = rules.get("requires_user_endorsement", True)

        if self.trust.local_successes < min_successes:
            return False
        if needs_endorsement and not self.trust.user_endorsed:
            return False
        if self.trust.should_quarantine:
            return False
        if self.trust.local_success_rate < 0.8:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "creator_instance_id": self.creator_instance_id,
            "creator_user": self.creator_user,
            "created_at": self.created_at,
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "parameters": s.parameters,
                    "condition": s.condition,
                    "use_previous": s.use_previous,
                }
                for s in self.steps
            ],
            "declared_tools": self.declared_tools,
            "capability_tier": self.capability_tier.value,
            "signature": self.signature,
            "trust": self.trust.to_dict(),
            "tags": self.tags,
            "source_pattern": self.source_pattern,
            "invocation_count": self.invocation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillGenome:
        trust_data = data.get("trust", {})
        trust = TrustScore(
            level=TrustLevel(trust_data.get("level", "untrusted")),
            local_successes=trust_data.get("local_successes", 0),
            local_failures=trust_data.get("local_failures", 0),
            network_successes=trust_data.get("network_successes", 0),
            network_failures=trust_data.get("network_failures", 0),
            network_quarantines=trust_data.get("network_quarantines", 0),
            user_endorsed=trust_data.get("user_endorsed", False),
        )
        steps = [
            SkillStep(
                tool_name=s["tool_name"],
                parameters=s.get("parameters", {}),
                condition=s.get("condition"),
                use_previous=s.get("use_previous", False),
            )
            for s in data.get("steps", [])
        ]
        genome = cls(
            id=data["id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            creator_instance_id=data.get("creator_instance_id", ""),
            creator_user=data.get("creator_user", ""),
            created_at=data.get("created_at", 0),
            steps=steps,
            declared_tools=data.get("declared_tools", []),
            capability_tier=CapabilityTier(data.get("capability_tier", 0)),
            signature=data.get("signature", ""),
            trust=trust,
            tags=data.get("tags", []),
            source_pattern=data.get("source_pattern", ""),
            invocation_count=data.get("invocation_count", 0),
        )
        return genome
