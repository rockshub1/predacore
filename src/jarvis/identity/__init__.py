"""
Prometheus Self-Evolving Identity System.

Each agent lives in ~/.prometheus/agents/{name}/. The identity surface:
- SOUL_SEED.md: Immutable guardrails (security, ethics, trust) — ships with source
- EVENT_HORIZON.md: Recursive growth protocol — ships with source, loaded every session
- BOOTSTRAP.md: First-run self-discovery conversation (archived after) — ships with source
- IDENTITY.md: Agent-discovered name, nature, vibe (written by agent)
- SOUL.md: Agent-grown personality and values (evolves over time)
- USER.md: Agent-learned human profile (discovered through interaction)
- MEMORY.md: Curated long-term memory
- TOOLS.md: Local environment notes
- HEARTBEAT.md: Background discipline policy
- REFLECTION.md: Self-reflection rules
- JOURNAL.md: Append-only growth log (insights, milestones, reflections)
"""

from .engine import IdentityEngine, get_identity_engine

__all__ = ["IdentityEngine", "get_identity_engine"]
