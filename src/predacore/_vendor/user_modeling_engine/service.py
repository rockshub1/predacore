"""
User Modeling Engine (UME) service.

Provides a lightweight persistent user profile layer used by CSC and PredaCore.
Profiles are stored as JSON files under a configurable data directory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(user_id or "default")).strip("_")
    return cleaned or "default"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class UserProfile:
    user_id: str
    preferences: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    knowledge_areas: dict[str, Any] = field(default_factory=dict)
    cognitive_style: str = ""
    notes: str = ""
    last_interaction_at: str = ""
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "goals": self.goals,
            "knowledge_areas": self.knowledge_areas,
            "cognitive_style": self.cognitive_style,
            "notes": self.notes,
            "last_interaction_at": self.last_interaction_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(
            user_id=str(data.get("user_id") or "default"),
            preferences=dict(data.get("preferences") or {}),
            goals=[str(g) for g in (data.get("goals") or [])],
            knowledge_areas=dict(data.get("knowledge_areas") or {}),
            cognitive_style=str(data.get("cognitive_style") or ""),
            notes=str(data.get("notes") or ""),
            last_interaction_at=str(data.get("last_interaction_at") or ""),
            updated_at=str(data.get("updated_at") or _utc_now_iso()),
        )


class UserModelingEngineService:
    """Persistent user modeling service with simple JSON storage."""

    def __init__(self, data_path: str = "data/ume"):
        self._base = Path(data_path).expanduser()
        self._base.mkdir(parents=True, exist_ok=True)
        self._profiles_dir = self._base / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._events_dir = self._base / "events"
        self._events_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, user_id: str) -> Path:
        return self._profiles_dir / f"{_safe_user_id(user_id)}.json"

    def _events_path(self, user_id: str) -> Path:
        return self._events_dir / f"{_safe_user_id(user_id)}.jsonl"

    def get_profile(self, user_id: str) -> UserProfile:
        path = self._profile_path(user_id)
        if not path.exists():
            return UserProfile(user_id=str(user_id or "default"))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return UserProfile.from_dict(raw if isinstance(raw, dict) else {})
        except Exception:
            return UserProfile(user_id=str(user_id or "default"))

    def save_profile(self, profile: UserProfile) -> UserProfile:
        profile.updated_at = _utc_now_iso()
        path = self._profile_path(profile.user_id)
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return profile

    def update_profile(self, user_id: str, patch: dict[str, Any]) -> UserProfile:
        profile = self.get_profile(user_id)
        merged = _deep_merge(profile.to_dict(), patch or {})
        merged["user_id"] = profile.user_id
        updated = UserProfile.from_dict(merged)
        return self.save_profile(updated)

    def record_interaction(
        self,
        user_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now_iso()
        self.update_profile(user_id, {"last_interaction_at": now})
        event = {
            "at": now,
            "user_id": str(user_id or "default"),
            "message": str(message or ""),
            "metadata": dict(metadata or {}),
        }
        path = self._events_path(user_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def build_planning_context(self, user_id: str) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        return {
            "user_id": profile.user_id,
            "preferences": profile.preferences,
            "goals": profile.goals,
            "knowledge_areas": profile.knowledge_areas,
            "cognitive_style": profile.cognitive_style,
            "notes": profile.notes,
            "last_interaction_at": profile.last_interaction_at,
            "updated_at": profile.updated_at,
        }
