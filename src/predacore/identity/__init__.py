"""
PredaCore Self-Evolving Identity System.

Every install ships with a pre-built identity surface (voice, heartbeat
policy, reflection rules, memory skeleton) but no fixed name. The agent
is named by its human on the first real conversation — the default
IDENTITY.md file contains those first-turn instructions, so no separate
bootstrap mode is needed.

Workspaces live at ~/.predacore/agents/{name}/:

- SOUL_SEED.md: Immutable guardrails — ships with source, loaded built-in
- EVENT_HORIZON.md: Recursive growth protocol — ships with source, built-in
- IDENTITY.md: Name, nature, vibe (seeded name-less; agent fills in on first turn)
- SOUL.md: Personality and voice (evolves over time)
- USER.md: Human profile (discovered through interaction)
- MEMORY.md: Curated long-term memory
- TOOLS.md: Local environment notes
- HEARTBEAT.md: Background discipline policy
- REFLECTION.md: Self-reflection rules
- JOURNAL.md: Append-only growth log
- BELIEFS.md + beliefs.json: Crystallization ladder (observation→tested→committed)
- DECISIONS.md: Per-turn reasoning trace
- EVOLUTION.md: Diff log of identity file changes over time
"""

from .engine import IdentityEngine, get_identity_engine

__all__ = ["IdentityEngine", "get_identity_engine"]
