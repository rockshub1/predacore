"""
Channel plugin registry — the single source of truth for which channel
adapters exist at runtime.

Three discovery mechanisms, in precedence order (later wins on name clash):

1. **Built-in adapters** — telegram, discord, whatsapp, webchat shipped with
   the core `predacore` package (registered via ``predacore.channels``
   entry points in our own pyproject).
2. **Third-party packages** — any installed package that declares a
   ``predacore.channels`` entry point. Example:

       [project.entry-points."predacore.channels"]
       slack = "predacore_slack.adapter:SlackAdapter"

   Do ``pip install predacore-slack`` and the adapter auto-registers on
   the next daemon start.
3. **User-local plugins** — any ``*.py`` file in ``~/.predacore/channels/``
   that defines a subclass of ``ChannelAdapter`` at module level. Meant
   for quick experiments and site-specific adapters; no install needed.

Adapter classes must be importable + carry a ``channel_name`` class attribute
(matched against ``config.channels.enabled``). The factory builds instances
on demand with ``AdapterCls(config)``.
"""
from __future__ import annotations

import importlib
import importlib.metadata as _meta
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PredaCoreConfig
    from ..gateway import ChannelAdapter

logger = logging.getLogger(__name__)

# Entry-point group scanned for channel plugins.
_ENTRY_POINT_GROUP = "predacore.channels"

# Directory users can drop local channel .py files into. Each file is
# imported; any top-level class that ends with "Adapter" and has a
# ``channel_name`` attribute is registered.
_USER_PLUGIN_DIRNAME = "channels"

# Built-in adapters — always discoverable, even in editable / dev installs
# where entry-point metadata may not yet be registered.
# Maps channel_name -> "module_path:ClassName".
_BUILTIN_ADAPTERS: dict[str, str] = {
    "telegram": "predacore.channels.telegram:TelegramAdapter",
    "discord":  "predacore.channels.discord:DiscordAdapter",
    "whatsapp": "predacore.channels.whatsapp:WhatsAppAdapter",
    "webchat":  "predacore.channels.webchat:WebChatAdapter",
    "slack":    "predacore.channels.slack:SlackAdapter",
    "signal":   "predacore.channels.signal:SignalAdapter",
    "imessage": "predacore.channels.imessage:IMessageAdapter",
    "email":    "predacore.channels.email:EmailAdapter",
}


class ChannelRegistry:
    """Discovers and caches channel-adapter classes by ``channel_name``."""

    def __init__(self, home_dir: Path | str) -> None:
        self._home = Path(home_dir).expanduser()
        self._adapters: dict[str, type[ChannelAdapter]] = {}
        self._sources: dict[str, str] = {}  # name -> source tag for diagnostics
        self._scanned = False

    # ── Discovery ─────────────────────────────────────────────────────

    def scan(self, *, refresh: bool = False) -> None:
        """Populate the registry. Idempotent unless ``refresh=True``."""
        if self._scanned and not refresh:
            return
        if refresh:
            self._adapters.clear()
            self._sources.clear()

        self._load_builtins()       # always-available core adapters
        self._load_entry_points()   # installed third-party packages
        self._load_user_plugins()   # ~/.predacore/channels/*.py
        self._scanned = True

        logger.info(
            "ChannelRegistry scan complete: %d adapter(s) — %s",
            len(self._adapters),
            ", ".join(sorted(self._adapters.keys())) or "none",
        )

    def _load_builtins(self) -> None:
        """Import the core bundled adapters without relying on entry points.

        Entry-point discovery works fine once ``pip install predacore`` has
        run, but fails in editable / dev layouts where the metadata isn't
        registered yet. The four core adapters are always present in the
        package tree, so import them directly as a safety net. Third-party
        and user-local channels still flow through the entry-point and
        plugin-directory paths below.
        """
        for name, target in _BUILTIN_ADAPTERS.items():
            module_path, _, class_name = target.partition(":")
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as exc:
                logger.debug("Built-in channel %s unavailable: %s", name, exc)
                continue
            self._register(cls, source=f"built-in:{module_path}")

    def _load_entry_points(self) -> None:
        """Scan installed packages for ``predacore.channels`` entry points."""
        try:
            eps = _meta.entry_points(group=_ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.10 compat fallback
            eps = _meta.entry_points().get(_ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                cls = ep.load()
            except (ImportError, AttributeError, ModuleNotFoundError) as exc:
                logger.warning(
                    "Channel entry point %s=%s failed to load: %s",
                    ep.name, getattr(ep, "value", "?"), exc,
                )
                continue
            self._register(cls, source=f"entry-point:{ep.name}")

    def _load_user_plugins(self) -> None:
        """Import each ``*.py`` in ``~/.predacore/channels/``."""
        plugin_dir = self._home / _USER_PLUGIN_DIRNAME
        if not plugin_dir.is_dir():
            return
        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"predacore_user_channel_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                logger.warning("User channel plugin %s failed to load: %s", py_file, exc)
                continue
            for attr in dir(module):
                if not attr.endswith("Adapter") or attr.startswith("_"):
                    continue
                candidate = getattr(module, attr)
                if isinstance(candidate, type) and hasattr(candidate, "channel_name"):
                    self._register(candidate, source=f"local:{py_file.name}")

    def _register(self, cls: type[ChannelAdapter], *, source: str) -> None:
        name = getattr(cls, "channel_name", None)
        if not isinstance(name, str) or not name:
            logger.debug(
                "Ignoring class %s — missing/invalid channel_name", cls.__name__,
            )
            return
        self._adapters[name] = cls
        self._sources[name] = source
        logger.debug("Registered channel adapter %s from %s", name, source)

    # ── Public API ────────────────────────────────────────────────────

    def available(self) -> list[str]:
        """Names of all discovered channel adapters."""
        self.scan()
        return sorted(self._adapters.keys())

    def describe(self) -> list[dict]:
        """Diagnostic summary: one dict per discovered adapter."""
        self.scan()
        return [
            {"name": name, "source": self._sources.get(name, "?"), "class": cls.__name__}
            for name, cls in sorted(self._adapters.items())
        ]

    def has(self, name: str) -> bool:
        self.scan()
        return name in self._adapters

    def create(self, name: str, config: PredaCoreConfig) -> ChannelAdapter | None:
        """Instantiate the adapter registered under ``name``."""
        self.scan()
        cls = self._adapters.get(name)
        if cls is None:
            logger.warning("No adapter registered for channel %r", name)
            return None
        try:
            return cls(config)
        except Exception as exc:
            logger.error("Failed to instantiate %s adapter: %s", name, exc, exc_info=True)
            return None


# ── Module-level singleton (one registry per process) ────────────────

_registry: ChannelRegistry | None = None


def get_registry(home_dir: Path | str | None = None) -> ChannelRegistry:
    """Return the process-wide channel registry, creating it if needed."""
    global _registry
    if _registry is None:
        if home_dir is None:
            home_dir = Path.home() / ".predacore"
        _registry = ChannelRegistry(home_dir)
    return _registry


def reset_registry() -> None:
    """Drop the cached registry (for tests)."""
    global _registry
    _registry = None
