"""Per-provider OAuth grant storage.

Each provider's tokens live in ``~/.predacore/oauth/<provider>.json``,
chmod 600, written atomically via temp + rename. Only the user can read.

The schema is small and stable:

    {
      "provider":      "openai_codex",
      "access_token":  "...",
      "refresh_token": "...",
      "expires_at":    1730568000.0,   # unix seconds
      "scope":         "...",
      "account_id":    "...",          # provider-supplied (Codex carries this)
      "obtained_at":   1730565000.0,
      "client_id":     "..."
    }

We deliberately store JSON in plain text on disk under chmod 600 — same
trust model as ``~/.predacore/.env``. If the user wants stronger
protection they can pipe through their OS keychain (future enhancement);
for now consistency with the rest of predacore is the right tradeoff.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OAUTH_DIR = Path.home() / ".predacore" / "oauth"


@dataclass
class OAuthGrant:
    """A single OAuth grant for one provider, persisted to disk."""

    provider: str
    access_token: str
    refresh_token: str
    expires_at: float                     # unix-seconds; 0 = unknown
    scope: str = ""
    account_id: str = ""
    obtained_at: float = field(default_factory=time.time)
    client_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """True when access_token is past expiry. expires_at=0 means unknown."""
        if self.expires_at <= 0:
            return False
        return time.time() >= self.expires_at

    @property
    def expires_within(self) -> float:
        """Seconds until expiry. Negative when already expired. ``inf``
        when the provider didn't supply ``expires_in``."""
        if self.expires_at <= 0:
            return float("inf")
        return self.expires_at - time.time()


class OAuthGrantStore:
    """Filesystem-backed grant store. One JSON file per provider."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir).expanduser() if base_dir else DEFAULT_OAUTH_DIR

    def _path(self, provider: str) -> Path:
        if not provider or "/" in provider or ".." in provider:
            raise ValueError(f"invalid provider name: {provider!r}")
        return self.base_dir / f"{provider}.json"

    def save(self, grant: OAuthGrant) -> Path:
        """Atomically persist a grant to disk with chmod 600.

        Uses ``tempfile`` + ``os.replace`` so a crash mid-write can never
        leave a half-written grant file in place.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        target = self._path(grant.provider)
        # NamedTemporaryFile in the same dir so os.replace stays atomic
        # (cross-device renames aren't atomic on POSIX).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".oauth-", suffix=".tmp", dir=str(self.base_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(grant), f, indent=2, sort_keys=True)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, target)
        except BaseException:
            # Cleanup partially-written tmp on any failure path.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info("oauth: saved grant for %s → %s", grant.provider, target)
        return target

    def load(self, provider: str) -> OAuthGrant | None:
        """Return the stored grant for ``provider`` or ``None`` if absent."""
        path = self._path(provider)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("oauth: corrupt grant file %s — %s", path, exc)
            return None
        # Tolerate extra keys (forward-compat) and missing optionals.
        return OAuthGrant(
            provider=data.get("provider", provider),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=float(data.get("expires_at", 0) or 0),
            scope=data.get("scope", ""),
            account_id=data.get("account_id", ""),
            obtained_at=float(data.get("obtained_at", 0) or 0),
            client_id=data.get("client_id", ""),
            extra=data.get("extra", {}) or {},
        )

    def delete(self, provider: str) -> bool:
        """Remove a stored grant. Returns True if a file was deleted."""
        path = self._path(provider)
        try:
            path.unlink()
            logger.info("oauth: deleted grant for %s", provider)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("oauth: delete failed for %s: %s", provider, exc)
            return False

    def list_providers(self) -> list[str]:
        """Names of all providers with stored grants."""
        if not self.base_dir.is_dir():
            return []
        return sorted(
            p.stem for p in self.base_dir.glob("*.json") if p.is_file()
        )


# Module-level convenience wrappers that use the default store path.
_default_store: OAuthGrantStore | None = None


def _store() -> OAuthGrantStore:
    global _default_store
    if _default_store is None:
        _default_store = OAuthGrantStore()
    return _default_store


def save_grant(grant: OAuthGrant) -> Path:
    """Save a grant via the default ``~/.predacore/oauth/`` store."""
    return _store().save(grant)


def load_grant(provider: str) -> OAuthGrant | None:
    """Load a grant for ``provider`` from the default store."""
    return _store().load(provider)
