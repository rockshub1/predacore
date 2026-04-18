"""
Hot-Reload Config Watcher — watches config files and reloads on change.

Uses filesystem polling (cross-platform) to detect changes to config
files and automatically reload the PredaCoreConfig when modifications
are detected.

Features:
  - File modification time tracking
  - Configurable poll interval
  - Callback-based notifications
  - Graceful error handling (corrupted config doesn't crash PredaCore)
  - Change diffing for logging

Usage:
    watcher = ConfigWatcher(config_path=Path("config.yaml"))
    watcher.on_change(lambda old, new: print("Config changed!"))
    await watcher.start()
    ...
    await watcher.stop()
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """
    Watches a config file for modifications and triggers reload callbacks.

    Uses polling (not inotify) for maximum cross-platform compatibility.
    """

    def __init__(
        self,
        config_path: Path,
        poll_interval: float = 2.0,
    ):
        """
        Args:
            config_path: Path to the config file to watch.
            poll_interval: Seconds between polls. Default: 2.0.
        """
        self._config_path = Path(config_path)
        self._poll_interval = poll_interval
        self._callbacks: list[Callable] = []
        self._last_mtime: float = 0.0
        self._last_hash: str = ""
        self._task: asyncio.Task | None = None
        self._running = False
        self._reload_count = 0

    def on_change(self, callback: Callable) -> None:
        """
        Register a callback to be invoked when the config file changes.

        Callback signature: callback(config_data: dict) -> None
        """
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start watching the config file."""
        if self._running:
            return

        self._running = True

        # Record initial state (file I/O in thread)
        exists = await asyncio.to_thread(self._config_path.exists)
        if exists:
            stat = await asyncio.to_thread(self._config_path.stat)
            self._last_mtime = stat.st_mtime
            self._last_hash = await asyncio.to_thread(self._compute_hash)

        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Config watcher started: %s (poll every %.1fs)",
            self._config_path,
            self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop watching the config file."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Config watcher stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._check_for_changes()
            except asyncio.CancelledError:
                break
            except (OSError, ValueError) as e:
                logger.error("Config watcher error: %s", e)

    async def _check_for_changes(self) -> None:
        """Check if the config file has been modified.

        All filesystem I/O is delegated to a thread so the event loop is
        never blocked by slow or networked filesystems.
        """
        try:
            exists = await asyncio.to_thread(self._config_path.exists)
        except OSError:
            return
        if not exists:
            return

        stat = await asyncio.to_thread(self._config_path.stat)
        current_mtime = stat.st_mtime

        if current_mtime <= self._last_mtime:
            return  # No change

        # Double-check with content hash (mtime can change without content change)
        current_hash = await asyncio.to_thread(self._compute_hash)
        if current_hash == self._last_hash:
            self._last_mtime = current_mtime
            return  # Same content

        # Config has changed!
        self._last_mtime = current_mtime
        self._last_hash = current_hash
        self._reload_count += 1

        logger.info(
            "Config change detected: %s (reload #%d)",
            self._config_path.name,
            self._reload_count,
        )

        # Load the new config (file I/O in thread)
        config_data = await asyncio.to_thread(self._load_config)
        if config_data is None:
            logger.error("Failed to parse changed config — ignoring this change")
            return

        # Notify all callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(config_data)
                else:
                    await asyncio.to_thread(callback, config_data)
            except (OSError, ValueError, TypeError) as e:
                logger.error("Config change callback error: %s", e)

    def _compute_hash(self) -> str:
        """Compute a hash of the config file contents."""
        import hashlib

        try:
            content = self._config_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except OSError:
            return ""

    def _load_config(self) -> dict[str, Any] | None:
        """Load and parse the config file."""
        try:
            content = self._config_path.read_text(encoding="utf-8")

            # Determine format from extension
            suffix = self._config_path.suffix.lower()

            if suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    return yaml.safe_load(content)
                except ImportError:
                    logger.warning("PyYAML not installed, trying JSON parse")
                    import json

                    return json.loads(content)

            elif suffix == ".json":
                import json

                return json.loads(content)

            elif suffix == ".toml":
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                return tomllib.loads(content)

            else:
                # Try YAML first, then JSON
                try:
                    import yaml

                    return yaml.safe_load(content)
                except (ImportError, ValueError):
                    import json

                    return json.loads(content)

        except (OSError, ValueError, ImportError) as e:
            logger.error("Failed to load config %s: %s", self._config_path, e)
            return None

    @property
    def reload_count(self) -> int:
        """Number of successful config reloads."""
        return self._reload_count

    @property
    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "config_path": str(self._config_path),
            "is_running": self._running,
            "reload_count": self._reload_count,
            "poll_interval": self._poll_interval,
            "last_check": self._last_mtime,
        }
