"""
JARVIS Daemon — 24/7 background operation with health monitoring.

The daemon manages JARVIS's lifecycle as a background process:
  - PID file management (write/check/clean)
  - Signal handling (SIGTERM/SIGINT → graceful shutdown)
  - Health endpoint (HTTP GET /health on webhook_port)
  - Auto-restart with exponential backoff
  - Heartbeat logging

Usage:
    # From CLI:
    prometheus start --daemon   # Start daemon
    prometheus stop             # Graceful shutdown
    prometheus status           # Check if running

    # Programmatic:
    daemon = JARVISDaemon(config)
    await daemon.start()
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from .alerting import Alert, AlertManager, AlertSeverity
from ..config import JARVISConfig, load_config

try:
    from jarvis._vendor.ethical_governance_module.persistent_audit import PersistentAuditStore
except ImportError:
    PersistentAuditStore = None  # type: ignore
from .config_watcher import ConfigWatcher
from ..core import JARVISCore, _get_system_prompt
from ..gateway import Gateway
from ..sessions import SessionStore

logger = logging.getLogger(__name__)


# -- PID File Management --


class PIDManager:
    """Manages the daemon PID file at ~/.prometheus/jarvis.pid."""

    def __init__(self, pid_path: str):
        self.pid_path = Path(pid_path)

    def write(self) -> None:
        """Atomically write current PID to file.

        Uses O_CREAT | O_EXCL for atomic creation so two daemons
        cannot race to create the same PID file.
        """
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                str(self.pid_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            # PID file already exists — check if the process is still alive
            if self.is_running():
                raise FileExistsError(
                    f"Daemon already running (PID file {self.pid_path} exists and process is alive)"
                )
            # Stale PID file — remove and retry
            self.cleanup()
            fd = os.open(
                str(self.pid_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        logger.info("PID %d written to %s", os.getpid(), self.pid_path)

    def read(self) -> int | None:
        """Read PID from file, or None if not found."""
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except (ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """Check if the daemon process is alive."""
        pid = self.read()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)  # Signal 0 = existence check
            return True
        except ProcessLookupError:
            self.cleanup()
            return False
        except PermissionError:
            return True  # Process exists but we can't signal it

    def cleanup(self) -> None:
        """Remove PID file."""
        try:
            self.pid_path.unlink(missing_ok=True)
            logger.debug("PID file removed: %s", self.pid_path)
        except OSError as e:
            logger.warning("Failed to remove PID file: %s", e)

    def get_status(self) -> dict[str, Any]:
        """Return daemon status info."""
        pid = self.read()
        running = self.is_running()
        return {
            "pid": pid,
            "running": running,
            "pid_file": str(self.pid_path),
        }


# -- Health Server --


class HealthServer:
    """Lightweight HTTP health endpoint for monitoring."""

    def __init__(self, port: int, daemon: JARVISDaemon):
        self.port = port
        self.daemon = daemon
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the health HTTP server."""
        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        self._app.router.add_get("/status", self._status_handler)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        logger.info(
            "Health endpoint listening on http://127.0.0.1:%d/health", self.port
        )

    async def stop(self) -> None:
        """Stop the health server."""
        if self._runner:
            await self._runner.cleanup()
            logger.debug("Health server stopped")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """GET /health — simple liveness check."""
        return web.json_response(
            {
                "status": "ok",
                "uptime_seconds": self.daemon.uptime,
                "pid": os.getpid(),
            }
        )

    async def _status_handler(self, request: web.Request) -> web.Response:
        """GET /status — detailed daemon status."""
        return web.json_response(self.daemon.get_status())


# -- Daemon --


class JARVISDaemon:
    """
    JARVIS background daemon.

    Manages the full lifecycle:
      1. Write PID file
      2. Start Gateway + registered channels
      3. Start CronEngine (if configured)
      4. Start health endpoint
      5. Run heartbeat loop until SIGTERM/SIGINT
      6. Graceful shutdown (stop channels, clean PID)
    """

    MAX_RESTART_ATTEMPTS = 5
    RESTART_BACKOFF_BASE = 2  # seconds, doubles each attempt

    def __init__(self, config: JARVISConfig):
        self.config = config
        self.pid_manager = PIDManager(str(Path(config.home_dir) / "jarvis.pid"))
        self._started_at: float = 0
        self._shutdown_event = asyncio.Event()
        self._core: JARVISCore | None = None
        self._gateway: Gateway | None = None
        self._cron_engine = None
        self._health_server: HealthServer | None = None
        self._config_watcher: ConfigWatcher | None = None
        self._alert_manager = AlertManager(
            slack_url=os.getenv("JARVIS_ALERT_SLACK_URL", ""),
            pagerduty_key=os.getenv("JARVIS_ALERT_PD_KEY", ""),
            webhook_url=os.getenv("JARVIS_ALERT_WEBHOOK_URL", ""),
        )
        # Enterprise audit store — only when EGM is active
        self._audit_store = None
        egm_mode = os.getenv("EGM_MODE", "off").strip().lower()
        if egm_mode != "off" and PersistentAuditStore is not None:
            audit_dir = Path(config.home_dir) / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            self._audit_store = PersistentAuditStore(str(audit_dir / "compliance.db"))
            logger.info("Enterprise audit store enabled (EGM_MODE=%s)", egm_mode)
        self._heartbeat_count: int = 0
        self._db_server = None

    @property
    def uptime(self) -> float:
        """Seconds since daemon started."""
        if self._started_at == 0:
            return 0
        return time.time() - self._started_at

    def get_status(self) -> dict[str, Any]:
        """Full daemon status."""
        status = {
            "running": not self._shutdown_event.is_set(),
            "pid": os.getpid(),
            "uptime_seconds": round(self.uptime, 1),
            "started_at": self._started_at,
            "heartbeat_count": self._heartbeat_count,
            "config": {
                "mode": self.config.mode,
                "trust_level": self.config.security.trust_level,
                "llm_provider": self.config.llm.provider,
                "channels": self.config.channels.enabled,
            },
        }

        # Add gateway stats if available
        if self._gateway:
            status["gateway"] = self._gateway.get_stats()

        # Add cron stats if available
        if self._cron_engine:
            status["cron"] = {
                "jobs": len(self._cron_engine.jobs),
                "enabled": True,
            }

        return status

    async def start(self) -> None:
        """
        Start the daemon. Blocks until shutdown signal received.

        This is the main entry point — call this from the CLI.
        """
        # Check if already running
        if self.pid_manager.is_running():
            existing_pid = self.pid_manager.read()
            logger.error("JARVIS daemon already running (PID %s)", existing_pid)
            print(f"❌ JARVIS is already running (PID {existing_pid})")
            print(
                f"   Run 'prometheus stop' first, or remove {self.pid_manager.pid_path}"
            )
            return

        # Write PID
        self.pid_manager.write()
        self._started_at = time.time()

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler, sig)

        logger.info("=" * 60)
        logger.info("JARVIS Daemon starting (PID %d)", os.getpid())
        logger.info("=" * 60)

        try:
            # Start DB server (must be first — other subsystems connect to it)
            try:
                from .db_server import DBServer
                home = self.config.home_dir
                mem_dir = self.config.memory.persistence_dir
                db_registry = {
                    "unified_memory": str(Path(mem_dir) / "unified_memory.db"),
                    "outcomes": str(Path(home) / "outcomes.db"),
                    "users": str(Path(home) / "users.db"),
                    "approvals": str(Path(mem_dir) / "approvals.db"),
                    "pipeline_states": str(Path(home) / "pipeline_states.db"),
                    "openclaw_idempotency": str(Path(mem_dir) / "openclaw_idempotency.db"),
                    "memory": str(Path(mem_dir) / "memory.db"),
                    "compliance": str(Path(home) / "audit" / "compliance.db"),
                }
                self._db_server = DBServer(
                    db_registry=db_registry,
                    socket_path=self.config.daemon.db_socket_path,
                )
                await self._db_server.start()
                logger.info("DB server started on %s", self.config.daemon.db_socket_path)
            except (ImportError, OSError) as exc:
                logger.warning("DB server not available (non-fatal): %s", exc)

            # Initialize core
            self._core = JARVISCore(self.config)

            # Initialize gateway
            self._gateway = Gateway(
                config=self.config,
                process_fn=self._core.process,
            )
            self._gateway.core = self._core
            # Share core's OutcomeStore — don't open a second connection
            if self._core._outcome_store is not None:
                self._gateway._outcome_store = self._core._outcome_store

            # Register enabled channels
            await self._register_channels()

            # Start gateway (starts all channels)
            await self._gateway.start()

            # Start cron engine if configured
            if self.config.daemon.cron_file:
                await self._start_cron()

            # Start health endpoint
            self._health_server = HealthServer(
                port=self.config.daemon.webhook_port,
                daemon=self,
            )
            await self._health_server.start()

            # Start config watcher for hot-reload
            config_path = Path(self.config.home_dir) / "config.yaml"
            if config_path.exists():
                self._config_watcher = ConfigWatcher(config_path=config_path)
                self._config_watcher.on_change(self._on_config_change)
                await self._config_watcher.start()

            logger.info("JARVIS Daemon fully started")

            # Heartbeat loop — runs until shutdown signal
            await self._heartbeat_loop()

        except Exception as e:  # catch-all: top-level boundary
            logger.error("Daemon crashed: %s", e, exc_info=True)
            self._alert_manager.fire(
                Alert(
                    title="JARVIS Daemon Crash",
                    message=str(e),
                    severity=AlertSeverity.CRITICAL,
                    source="daemon",
                )
            )
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Signal the daemon to shut down gracefully."""
        logger.info("Daemon shutdown requested")
        self._shutdown_event.set()

    def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown.

        A second signal forces an immediate exit to avoid hanging.
        """
        sig_name = signal.Signals(sig).name
        if self._shutdown_event.is_set():
            logger.warning("Received %s again — forcing immediate exit", sig_name)
            self.pid_manager.cleanup()
            sys.exit(1)
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        self._shutdown_event.set()

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat — health logging and maintenance."""
        interval = self.config.daemon.heartbeat_interval

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
                break  # Shutdown was signaled
            except asyncio.TimeoutError:
                pass  # Normal timeout, do heartbeat

            self._heartbeat_count += 1

            if self._heartbeat_count % 10 == 0:  # Log every 10th heartbeat
                logger.info(
                    "Heartbeat #%d | uptime=%.0fs | pid=%d",
                    self._heartbeat_count,
                    self.uptime,
                    os.getpid(),
                )

    async def _register_channels(self) -> None:
        """Register enabled channel adapters with the gateway."""
        for channel_name in self.config.channels.enabled:
            if channel_name == "cli":
                continue  # CLI is handled separately, not in daemon

            try:
                adapter = self._create_channel_adapter(channel_name)
                if adapter:
                    self._gateway.register_channel(adapter)
                    logger.info("Registered channel: %s", channel_name)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.error("Failed to register channel '%s': %s", channel_name, e)

    def _create_channel_adapter(self, name: str):
        """Factory: create a channel adapter by name."""
        if name == "telegram":
            from ..channels.telegram import TelegramAdapter

            return TelegramAdapter(self.config)
        elif name == "discord":
            from ..channels.discord import DiscordAdapter

            return DiscordAdapter(self.config)
        elif name == "whatsapp":
            from ..channels.whatsapp import WhatsAppAdapter

            return WhatsAppAdapter(self.config)
        elif name == "webchat":
            from ..channels.webchat import WebChatAdapter

            return WebChatAdapter(self.config)
        else:
            logger.warning("Unknown channel: %s", name)
            return None

    async def _start_cron(self) -> None:
        """Initialize and start the cron engine."""
        try:
            from .cron import CronEngine

            self._cron_engine = CronEngine(self.config, self._gateway)
            await self._cron_engine.start()
            logger.info("Cron engine started (%d jobs)", len(self._cron_engine.jobs))
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.error("Cron engine failed to start: %s", e)

    def _on_config_change(self, config_data: dict) -> None:
        """Handle hot-reload of config.yaml — model, prompt, and provider update live."""
        logger.info("Config hot-reload: applying %d top-level keys", len(config_data))
        if self._core is not None:
            try:
                new_cfg = load_config(str(Path(self.config.home_dir) / "config.yaml"))
                old_model = self.config.llm.model
                old_provider = self.config.llm.provider

                # Update system prompt
                self._core._system_prompt = _get_system_prompt(new_cfg)

                # Hot-swap model on the LLM provider (no restart needed)
                if new_cfg.llm.model != old_model:
                    provider = self._core.llm._get_provider_instance(self._core.llm._provider_name)
                    if hasattr(provider, 'config'):
                        provider.config.model = new_cfg.llm.model
                    logger.info("Model hot-swapped: %s → %s", old_model, new_cfg.llm.model)

                # Update config reference (core + gateway + daemon)
                self.config = new_cfg
                self._core.config = new_cfg
                if self._gateway is not None:
                    self._gateway.config = new_cfg
                logger.info("Config hot-reload applied successfully")
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                logger.error("Config hot-reload failed: %s", e)

    async def _cleanup(self, timeout: float = 15.0) -> None:
        """Graceful shutdown — stop everything and clean up.

        Each subsystem gets up to *timeout* seconds to shut down.  If it
        exceeds the deadline the error is logged and we move on so the
        daemon does not hang.
        """
        logger.info("Cleaning up daemon resources...")

        async def _stop(name: str, coro) -> None:
            try:
                await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("%s did not stop within %.0fs — skipping", name, timeout)
            except Exception as exc:  # catch-all: top-level boundary
                logger.error("%s cleanup error: %s", name, exc)

        # Stop config watcher
        if self._config_watcher:
            await _stop("Config watcher", self._config_watcher.stop())

        # Stop health server
        if self._health_server:
            await _stop("Health server", self._health_server.stop())

        # Stop cron engine
        if self._cron_engine:
            await _stop("Cron engine", self._cron_engine.stop())

        # Stop gateway (stops all channels)
        if self._gateway:
            await _stop("Gateway", self._gateway.stop())

        # Stop DB server (last — other subsystems may flush during cleanup)
        if self._db_server:
            await _stop("DB server", self._db_server.stop())

        # Remove PID file
        self.pid_manager.cleanup()

        elapsed = self.uptime
        logger.info(
            "JARVIS Daemon stopped (uptime=%.0fs, heartbeats=%d)",
            elapsed,
            self._heartbeat_count,
        )


# -- CLI Helpers --


def stop_daemon(config: JARVISConfig) -> bool:
    """
    Send SIGTERM to the running daemon.
    Returns True only if the daemon fully stopped.
    """
    pid_manager = PIDManager(str(Path(config.home_dir) / "jarvis.pid"))

    if not pid_manager.is_running():
        return False

    pid = pid_manager.read()
    if pid is None:
        # PID file disappeared between is_running() and read()
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to PID %d", pid)

        # Wait up to 10 seconds for shutdown
        for _ in range(20):
            time.sleep(0.5)
            if not pid_manager.is_running():
                pid_manager.cleanup()
                return True

        logger.warning("Daemon PID %d did not stop in 10s", pid)
        return False

    except ProcessLookupError:
        pid_manager.cleanup()
        return False
    except PermissionError:
        logger.error("Permission denied sending signal to PID %d", pid)
        return False


def generate_launchd_plist(config: JARVISConfig) -> str:
    """
    Generate a macOS launchd plist for auto-starting JARVIS.

    Install with:
        prometheus install
        launchctl load ~/Library/LaunchAgents/com.prometheus.jarvis.plist
    """
    python_path = sys.executable
    home_dir = config.home_dir
    log_dir = config.logs_dir

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.prometheus.jarvis</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>src.jarvis.cli</string>
        <string>start</string>
        <string>--daemon</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{Path(home_dir).parent}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_dir}/daemon.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon.stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>JARVIS_HOME</key>
        <string>{home_dir}</string>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>"""
    return plist


def install_launchd(config: JARVISConfig) -> Path:
    """Install the launchd plist to ~/Library/LaunchAgents/."""
    plist_content = generate_launchd_plist(config)
    target = Path.home() / "Library" / "LaunchAgents" / "com.prometheus.jarvis.plist"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plist_content)

    # Ensure log directory exists
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)

    return target
