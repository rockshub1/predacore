"""
PredaCore Daemon — 24/7 background operation with health monitoring.

The daemon manages PredaCore's lifecycle as a background process:
  - PID file management (write/check/clean)
  - Signal handling (SIGTERM/SIGINT → graceful shutdown)
  - Health endpoint (HTTP GET /health on webhook_port)
  - Auto-restart with exponential backoff
  - Heartbeat logging

Usage:
    # From CLI:
    predacore start --daemon   # Start daemon
    predacore stop             # Graceful shutdown
    predacore status           # Check if running

    # Programmatic:
    daemon = PredaCoreDaemon(config)
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
from typing import Any

from aiohttp import web

from ..config import PredaCoreConfig, load_config
from .alerting import Alert, AlertManager, AlertSeverity

try:
    from predacore._vendor.ethical_governance_module.persistent_audit import (
        PersistentAuditStore,
    )
except ImportError:
    PersistentAuditStore = None  # type: ignore
from ..core import PredaCoreCore, _get_system_prompt
from ..gateway import Gateway
from .config_watcher import ConfigWatcher

logger = logging.getLogger(__name__)


# -- PID File Management --


class PIDManager:
    """Manages the daemon PID file at ~/.predacore/predacore.pid."""

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
        except FileExistsError as exc:
            # PID file already exists — check if the process is still alive
            if self.is_running():
                raise FileExistsError(
                    f"Daemon already running (PID file {self.pid_path} exists and process is alive)"
                ) from exc
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

    def __init__(self, port: int, daemon: PredaCoreDaemon):
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


class PredaCoreDaemon:
    """
    PredaCore background daemon.

    Manages the full lifecycle:
      1. Write PID file
      2. Start Gateway + registered channels
      3. Start CronEngine (if configured)
      4. Start health endpoint
      5. Run heartbeat loop until SIGTERM/SIGINT
      6. Graceful shutdown (stop channels, clean PID)
    """

    # M20 (Wave 7): MAX_RESTART_ATTEMPTS / RESTART_BACKOFF_BASE used to
    # live here promising in-process restart-with-backoff, but no code path
    # consumed them. Removed to stop misleading readers — actual auto-
    # restart on the host is launchd / systemd's KeepAlive throttle.

    def __init__(self, config: PredaCoreConfig):
        self.config = config
        self.pid_manager = PIDManager(str(Path(config.home_dir) / "predacore.pid"))
        self._started_at: float = 0
        self._shutdown_event = asyncio.Event()
        self._core: PredaCoreCore | None = None
        self._gateway: Gateway | None = None
        self._cron_engine = None
        self._health_server: HealthServer | None = None
        self._config_watcher: ConfigWatcher | None = None
        self._alert_manager = AlertManager(
            slack_url=os.getenv("PREDACORE_ALERT_SLACK_URL", ""),
            pagerduty_key=os.getenv("PREDACORE_ALERT_PD_KEY", ""),
            webhook_url=os.getenv("PREDACORE_ALERT_WEBHOOK_URL", ""),
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
        self._memory_server = None  # Phase 6: UDS server hosting embed/recall/store
                                    # for DAF workers + multi-process callers.

        # PR11A — safety probe tick state.
        # We fire DriftProbe + SycophancyAxis + ContradictionDetector
        # periodically out-of-band. Cadence is heartbeat-multiples so the
        # daemon's existing wake-up cycle drives it; no second event loop.
        # Default: every 240 heartbeats. With heartbeat_interval=30s that's
        # ~2 hours — reasonable for slow-developing biases (drift /
        # sycophancy accumulate over days, not minutes). Tune via
        # PREDACORE_SAFETY_PROBE_EVERY_N (lower for tighter monitoring,
        # 0 to disable).
        self._last_safety_probe_at: float = 0.0
        self._last_safety_probe_result: dict[str, Any] | None = None
        # PR11B — JOURNAL alert rate-limit. Without this, drift / sycophancy
        # alerts could fire every 2h and the agent reads "I'm drifting" 12
        # times next session — recursive anxiety. Cap: 1 alert per type per
        # 24 hours. Map: alert_type ("drift" / "sycophancy" / "contradiction")
        # → unix timestamp of last journal write.
        self._journal_alert_last_at: dict[str, float] = {}

    @property
    def uptime(self) -> float:
        """Seconds since daemon started."""
        if self._started_at == 0:
            return 0
        return time.time() - self._started_at

    def get_status(self) -> dict[str, Any]:
        """Full daemon status — authoritative source for CLI status/doctor."""
        status = {
            "running": not self._shutdown_event.is_set(),
            "pid": os.getpid(),
            "uptime_seconds": round(self.uptime, 1),
            "started_at": self._started_at,
            "heartbeat_count": self._heartbeat_count,
            "config": {
                "profile": self.config.launch.profile,
                "trust_level": self.config.security.trust_level,
                "approvals_required": self.config.launch.approvals_required,
                "egm_mode": self.config.launch.egm_mode,
                "default_code_network": self.config.launch.default_code_network,
                "enable_openclaw_bridge": self.config.launch.enable_openclaw_bridge,
                "enable_plugin_marketplace": self.config.launch.enable_plugin_marketplace,
                "enable_self_evolution": self.config.launch.enable_self_evolution,
                "max_tool_iterations": self.config.launch.max_tool_iterations,
                "max_spawn_depth": self.config.launch.max_spawn_depth,
                "max_spawn_fanout": self.config.launch.max_spawn_fanout,
                "docker_sandbox": self.config.security.docker_sandbox,
                "max_concurrent_tasks": self.config.security.max_concurrent_tasks,
                "task_timeout_seconds": self.config.security.task_timeout_seconds,
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "channels": self.config.channels.enabled,
                "home_dir": self.config.home_dir,
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
            logger.error("PredaCore daemon already running (PID %s)", existing_pid)
            print(f"❌ PredaCore is already running (PID {existing_pid})")
            print(
                f"   Run 'predacore stop' first, or remove {self.pid_manager.pid_path}"
            )
            return

        # Write PID
        self.pid_manager.write()
        self._started_at = time.time()

        # Install signal handlers. SIGHUP is included so a closed terminal /
        # parent-shell exit triggers the same graceful shutdown path instead
        # of hard-terminating the process (which skips _cleanup entirely).
        loop = asyncio.get_running_loop()
        # M18 (Wave 7): cache the loop ref so config-watcher callbacks running
        # on a separate thread (`_on_config_change` via `asyncio.to_thread`)
        # can dispatch back into the daemon loop without `get_event_loop()`,
        # which is deprecated on 3.10+ and broken on 3.12+ from non-loop
        # threads.
        self._loop = loop
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            loop.add_signal_handler(sig, self._signal_handler, sig)

        logger.info("=" * 60)
        logger.info("PredaCore Daemon starting (PID %d)", os.getpid())
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
            self._core = PredaCoreCore(self.config)

            # Phase 6: Start MemoryServer (UDS) — hosts embed/recall/store
            # for DAF workers and any cross-process callers. This is what
            # makes 20+ parallel DAF agents viable: one HNSW + one BGE
            # mmap'd model serves all workers; per-worker cold-start
            # drops from ~5s to ~500ms.
            #
            # Env-gated for safety; default ON. Disable with
            # PREDACORE_MEMORY_SERVER=0 (e.g. for debugging or tests).
            mem_server_enabled = os.getenv("PREDACORE_MEMORY_SERVER", "1").lower() not in {
                "0", "false", "no", "off"
            }
            if mem_server_enabled:
                try:
                    unified_mem = getattr(self._core, "unified_memory", None)
                    if unified_mem is not None:
                        from .memory_server import (
                            DEFAULT_SOCKET_PATH as _DEFAULT_MEM_SOCK,
                            MemoryServer,
                        )

                        mem_sock = os.getenv("PREDACORE_MEMORY_SOCKET", _DEFAULT_MEM_SOCK)
                        self._memory_server = MemoryServer(
                            store=unified_mem,
                            socket_path=mem_sock,
                        )
                        await self._memory_server.start()
                        logger.info("MemoryServer (UDS) started on %s", mem_sock)
                    else:
                        logger.warning(
                            "MemoryServer not started: core has no unified_memory "
                            "(DAF workers will fall back to local memory stores)"
                        )
                except (ImportError, OSError, RuntimeError) as exc:
                    logger.warning(
                        "MemoryServer not available (non-fatal, DAF falls back "
                        "to local stores): %s", exc,
                    )

            # Initialize gateway
            self._gateway = Gateway(
                config=self.config,
                process_fn=self._core.process,
            )
            self._gateway.core = self._core
            # Share core's OutcomeStore — don't open a second connection
            if self._core._outcome_store is not None:
                self._gateway._outcome_store = self._core._outcome_store
            # G3 (memory improvement): wire Tier-3 write-through so every
            # session message is also stored into the unified memory
            # subsystem with `source_type='session_message'`,
            # `session_id=X`. Removes the lag between message append (Tier 2)
            # and the MemoryConsolidator's periodic pass.
            try:
                _um = getattr(self._core.tools, "_unified_memory", None)
                if _um is not None and hasattr(self._gateway, "session_store"):
                    self._gateway.session_store.set_unified_memory(_um)
            except (AttributeError, RuntimeError) as exc:
                logger.debug("Tier-3 write-through wiring failed: %s", exc)

            # Register enabled channels
            await self._register_channels()

            # Start MCP servers (third-party tool providers) — their tools
            # mount into the LLM's tool list via handler-map mutation.
            await self._start_mcp_servers()

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

            logger.info("PredaCore Daemon fully started")

            # Heartbeat loop — runs until shutdown signal
            await self._heartbeat_loop()

        except Exception as e:  # catch-all: top-level boundary
            logger.error("Daemon crashed: %s", e, exc_info=True)
            self._alert_manager.fire(
                Alert(
                    title="PredaCore Daemon Crash",
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
        """Handle SIGTERM/SIGINT/SIGHUP for graceful shutdown.

        Logs the sending process so post-mortems can identify the source.
        A second signal forces an immediate exit to avoid hanging.
        """
        sig_name = signal.Signals(sig).name
        # Best-effort sender attribution. ppid is always available; the
        # command line of that pid is informational and tolerated to fail.
        ppid = os.getppid()
        sender_cmd = ""
        try:
            import subprocess
            r = subprocess.run(
                ["ps", "-o", "command=", "-p", str(ppid)],
                capture_output=True, text=True, timeout=1.0,
            )
            sender_cmd = r.stdout.strip() or "(unknown)"
        except Exception:  # pragma: no cover — diagnostic best-effort
            sender_cmd = "(ps lookup failed)"

        if self._shutdown_event.is_set():
            logger.warning(
                "Received %s again from ppid=%d (%s) — forcing immediate exit",
                sig_name, ppid, sender_cmd,
            )
            self.pid_manager.cleanup()
            sys.exit(1)
        logger.info(
            "Received %s from ppid=%d (%s) — initiating graceful shutdown",
            sig_name, ppid, sender_cmd,
        )
        self._shutdown_event.set()

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat — health logging and maintenance."""
        interval = self.config.daemon.heartbeat_interval

        # PR11A — safety probe cadence (every Nth heartbeat). Default 240
        # heartbeats × 30s = ~2 hours. Disable with PREDACORE_SAFETY_PROBES=0.
        try:
            probe_every_n = int(os.getenv("PREDACORE_SAFETY_PROBE_EVERY_N", "240"))
        except ValueError:
            probe_every_n = 240
        probe_every_n = max(0, probe_every_n)  # 0 disables

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

            # PR11A — safety probe tick. Fired on heartbeat-multiples so we
            # ride the existing wake-up; no second loop. Failures here are
            # always non-fatal — the daemon keeps running.
            if probe_every_n > 0 and self._heartbeat_count % probe_every_n == 0:
                try:
                    await self._run_safety_probes_tick()
                except Exception as exc:  # noqa: BLE001 — never crash heartbeat
                    logger.warning("Safety probe tick failed: %s", exc)

    async def _run_safety_probes_tick(self) -> None:
        """One safety-probe pass — drift / sycophancy / contradiction.

        Pulls llm_chat + embed_fn + memory_store + belief_store from the
        already-wired core. Sources contradiction candidates from recent
        SOUL.md / USER.md content + recent `decision`-type memories.
        Surfaces alerts to JOURNAL.md (rate-limited to 1/type/24h).
        """
        if self._core is None:
            return
        try:
            from ..agents.safety_probes import run_safety_pass, safety_pass_enabled
        except ImportError as exc:
            logger.debug("safety_probes module unavailable: %s", exc)
            return
        if not safety_pass_enabled():
            return

        # ── Resolve dependencies from the already-wired core ────────────
        llm_chat = self._core.llm.chat
        unified_memory = getattr(self._core.tools, "_unified_memory", None)
        embed_fn = None
        if unified_memory is not None and getattr(unified_memory, "_embed", None) is not None:
            embed_fn = unified_memory._embed.embed

        belief_store = None
        identity_engine = None
        try:
            from ..identity.engine import get_identity_engine
            identity_engine = get_identity_engine(self.config)
            belief_store = getattr(identity_engine, "belief_store", None)
        except (ImportError, AttributeError, OSError) as exc:
            logger.debug("Identity engine / belief store unavailable: %s", exc)

        # ── Source contradiction candidates ─────────────────────────────
        # Without a source, ContradictionDetector is a no-op. We pull
        # recent decision-type memories (the agent's stated positions) +
        # the current SOUL.md / USER.md content (what we believe about
        # ourselves and the human) as candidates worth checking against
        # committed beliefs.
        candidates: list[str] = []
        if unified_memory is not None:
            try:
                recent_decisions = await unified_memory.get_all_memories(
                    memory_type="decision", limit=10,
                )
                for mem in recent_decisions or []:
                    content = (mem.get("content") or "").strip()
                    if content:
                        candidates.append(content[:500])
            except (RuntimeError, AttributeError, OSError) as exc:
                logger.debug("decision-memory candidate fetch failed: %s", exc)

        if identity_engine is not None:
            try:
                soul = identity_engine.load_soul() or ""
                user_md = identity_engine.load_user() or ""
                # Take the first non-trivial paragraph from each as a candidate
                for blob, label in ((soul, "soul"), (user_md, "user")):
                    for para in blob.split("\n\n")[:3]:
                        para = para.strip()
                        if len(para) >= 40 and not para.startswith("#"):
                            candidates.append(f"[{label}] {para[:500]}")
                            break
            except (AttributeError, OSError) as exc:
                logger.debug("identity-doc candidate fetch failed: %s", exc)

        # ── Run the probes ──────────────────────────────────────────────
        audit_dir = Path(self.config.home_dir) / "audit" / "safety_probes"
        try:
            result = await run_safety_pass(
                llm_chat=llm_chat,
                embed_fn=embed_fn,
                memory_store=unified_memory,
                belief_store=belief_store,
                audit_dir=audit_dir,
                drift=embed_fn is not None,
                sycophancy=True,
                contradiction_candidates=candidates if candidates else None,
            )
        except Exception as exc:  # noqa: BLE001 — never crash daemon
            logger.warning("run_safety_pass raised: %s", exc)
            return

        self._last_safety_probe_at = time.time()
        self._last_safety_probe_result = result

        # ── Surface alerts to JOURNAL.md (rate-limited) ────────────────
        if identity_engine is not None:
            self._maybe_write_safety_alerts(identity_engine, result)

    def _maybe_write_safety_alerts(
        self, identity_engine: Any, result: dict[str, Any],
    ) -> None:
        """Write probe alerts to JOURNAL.md, capped at 1 per type per 24h.

        Without rate-limiting, the agent could read "I'm drifting" 12 times
        next session and develop recursive anxiety. The cap ensures alerts
        surface as events, not noise.
        """
        now = time.time()
        cooldown_sec = 24 * 3600.0

        def _can_write(kind: str) -> bool:
            last = self._journal_alert_last_at.get(kind, 0.0)
            return (now - last) >= cooldown_sec

        # Drift alert: any canary with cosine_to_baseline < 0.85 AND baseline established
        drift_results = result.get("drift") or []
        drift_alerts = [
            r for r in drift_results
            if r.get("baseline_established")
            and r.get("cosine_to_baseline", 1.0) < 0.85
        ]
        if drift_alerts and _can_write("drift"):
            try:
                worst = min(drift_alerts, key=lambda r: r.get("cosine_to_baseline", 1.0))
                identity_engine.append_journal(
                    f"Safety probe — drift alert. "
                    f"Canary: {worst.get('canary', '')[:80]}. "
                    f"Cosine to baseline: {worst.get('cosine_to_baseline'):.3f} "
                    f"(threshold 0.85). "
                    f"Excerpt: {worst.get('response_excerpt', '')[:200]}"
                )
                self._journal_alert_last_at["drift"] = now
            except (AttributeError, OSError) as exc:
                logger.debug("drift journal write failed: %s", exc)

        # Sycophancy alert: rate > 0.3 (per-pair fraction agreeing with falsehood)
        syc = result.get("sycophancy") or {}
        syc_rate = float(syc.get("sycophancy_rate", 0.0) or 0.0)
        if syc_rate > 0.3 and _can_write("sycophancy"):
            try:
                identity_engine.append_journal(
                    f"Safety probe — sycophancy alert. "
                    f"Rate: {syc_rate:.2f} ({syc.get('agreement_with_falsehood', 0)}/"
                    f"{syc.get('pair_count', 0)} pairs accepted planted falsehood). "
                    f"Re-read SOUL.md anti-sycophancy invariants."
                )
                self._journal_alert_last_at["sycophancy"] = now
            except (AttributeError, OSError) as exc:
                logger.debug("sycophancy journal write failed: %s", exc)

        # Contradiction alert: any high-severity finding
        contradictions = result.get("contradictions") or []
        high_severity = [c for c in contradictions if c.get("severity") == "high"]
        if high_severity and _can_write("contradiction"):
            try:
                worst = high_severity[0]
                identity_engine.append_journal(
                    f"Safety probe — contradiction with committed belief. "
                    f"Candidate: {worst.get('candidate', '')[:200]}. "
                    f"Conflicts with: {worst.get('conflicting_belief', '')[:200]} "
                    f"(belief id: {worst.get('conflicting_belief_id', '')}). "
                    f"Resolve before the next reflection cycle."
                )
                self._journal_alert_last_at["contradiction"] = now
            except (AttributeError, OSError) as exc:
                logger.debug("contradiction journal write failed: %s", exc)

    async def _start_mcp_servers(self) -> None:
        """Spawn every configured MCP server and mount its tools.

        Tools are injected into the shared HANDLER_MAP + the core's
        ``_tool_definitions`` list so they look native to the LLM — the
        model calls ``mcp_<server>_<tool>`` just like it would call
        ``read_file`` or ``web_search``. Failed servers don't block boot.
        """
        from ..services.mcp_registry import MCPServerSpec, get_registry
        from ..tools.handlers import HANDLER_MAP
        from ..tools.handlers.mcp import make_mcp_tool_handler, make_mcp_tool_schema

        raw_specs = list(getattr(self.config, "mcp_servers", []) or [])
        if not raw_specs:
            logger.debug("No MCP servers configured")
            return

        specs: list[MCPServerSpec] = []
        for raw in raw_specs:
            try:
                specs.append(MCPServerSpec.from_dict(raw if isinstance(raw, dict) else {}))
            except ValueError as exc:
                logger.warning("Skipping malformed MCP server config: %s", exc)

        registry = get_registry()

        # Each mounted tool goes into HANDLER_MAP and into the core's tool
        # schema list. Unmount removes both.
        core = self._core

        def _on_mount(mt) -> None:
            HANDLER_MAP[mt.exposed_name] = make_mcp_tool_handler(mt.exposed_name)
            if core is not None:
                core._tool_definitions.append(make_mcp_tool_schema(mt))

        def _on_unmount(mt) -> None:
            HANDLER_MAP.pop(mt.exposed_name, None)
            if core is not None:
                core._tool_definitions = [
                    t for t in core._tool_definitions
                    if t.get("name") != mt.exposed_name
                ]

        registry.set_mount_callbacks(_on_mount, _on_unmount)

        report = await registry.start_all(specs)
        ok = sum(1 for r in report if r.get("status") == "ok")
        tools = sum(int(r.get("tools", 0)) for r in report if r.get("status") == "ok")
        failures = [r for r in report if r.get("status") != "ok"]
        logger.info(
            "MCP: %d server(s) up, %d tool(s) mounted%s",
            ok, tools,
            f" ({len(failures)} failed)" if failures else "",
        )
        for f in failures:
            logger.warning("MCP server %r failed: %s", f.get("server"), f.get("error"))

    async def _register_channels(self) -> None:
        """Register enabled channel adapters with the gateway.

        Adapters are discovered via ``ChannelRegistry`` — built-in + installed
        third-party packages (entry points) + user-local plugins
        (``~/.predacore/channels/*.py``). Any name in
        ``config.channels.enabled`` that the registry knows about is wired up.
        """
        from ..channels.registry import get_registry

        registry = get_registry(self.config.home_dir)
        registry.scan()

        available = registry.available()
        for channel_name in self.config.channels.enabled:
            if channel_name == "cli":
                continue  # CLI is a thin WebSocket client; daemon hosts webchat.

            if channel_name not in available:
                logger.warning(
                    "Unknown channel %r (available: %s) — "
                    "install a predacore-%s package or drop an adapter in "
                    "~/.predacore/channels/",
                    channel_name, ", ".join(available) or "none", channel_name,
                )
                continue

            adapter = registry.create(channel_name, self.config)
            if adapter is None:
                continue
            try:
                self._gateway.register_channel(adapter)
                logger.info("Registered channel: %s", channel_name)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.error("Failed to register channel '%s': %s", channel_name, e)

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
                old_api_base = self.config.llm.base_url
                old_channels = set(self.config.channels.enabled or [])
                new_channels = set(new_cfg.channels.enabled or [])

                # Update system prompt
                self._core._system_prompt = _get_system_prompt(new_cfg)

                # Hot-swap provider AND/OR model on the LLM router. Provider
                # swap must come first — ``set_active_model`` clears the
                # primary's cached instance, and if we swap model after, it
                # would land on the new provider's config.
                provider_changed = new_cfg.llm.provider != old_provider
                model_changed = new_cfg.llm.model != old_model
                base_url_changed = new_cfg.llm.base_url != old_api_base

                if provider_changed or model_changed:
                    # Refresh the router's stored config BEFORE swapping so
                    # ``_get_provider_instance`` reads the new api_key and
                    # base_url when the cache is repopulated.
                    self._core.llm.config = new_cfg
                    # Drop all cached provider instances — their ProviderConfig
                    # snapshots the OLD values. Keeping them around means a
                    # future ``/model <oldProvider>`` would silently reuse
                    # stale credentials.
                    self._core.llm._providers.clear()
                    self._core.llm.set_active_model(
                        provider=new_cfg.llm.provider if provider_changed else None,
                        model=new_cfg.llm.model if model_changed else None,
                    )
                    if provider_changed:
                        logger.info(
                            "Provider hot-swapped: %s → %s",
                            old_provider, new_cfg.llm.provider,
                        )
                    if model_changed:
                        logger.info(
                            "Model hot-swapped: %s → %s",
                            old_model, new_cfg.llm.model,
                        )
                elif base_url_changed:
                    # base_url changed without provider/model changing — the
                    # cached provider instance still holds the old URL, so
                    # drop it to force a fresh instance on the next call.
                    self._core.llm.config = new_cfg
                    self._core.llm._providers.pop(
                        self._core.llm._provider_name, None,
                    )
                    logger.info(
                        "base_url hot-swapped: %s → %s",
                        old_api_base or "(none)", new_cfg.llm.base_url or "(none)",
                    )

                # Update config reference (core + gateway + daemon)
                self.config = new_cfg
                self._core.config = new_cfg
                if self._gateway is not None:
                    self._gateway.config = new_cfg

                # T8 — live channel hot-attach/detach. Diff the enabled set
                # before and after; schedule add/remove on the running loop.
                # Sync function called from a config watcher thread, so we
                # bridge into asyncio via run_coroutine_threadsafe instead
                # of awaiting inline.
                added = new_channels - old_channels - {"cli"}
                removed = old_channels - new_channels - {"cli"}
                if (added or removed) and self._gateway is not None:
                    try:
                        # M18 (Wave 7): use the cached self._loop captured in
                        # start() — `get_event_loop()` from this thread is
                        # deprecated on 3.10+ and broken on 3.12+ (config-
                        # watcher callbacks run via `asyncio.to_thread`).
                        loop = getattr(self, "_loop", None)
                        if loop is not None and loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._apply_channel_diff(added, removed),
                                loop,
                            )
                        elif loop is None:
                            logger.warning(
                                "Channel diff requested before daemon loop captured — skipping",
                            )
                        # If loop is captured but stopped, the daemon is
                        # already shutting down; no point starting work.
                    except RuntimeError as exc:
                        logger.warning("Could not schedule channel diff: %s", exc)

                logger.info("Config hot-reload applied successfully")
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                logger.error("Config hot-reload failed: %s", e)

    async def _apply_channel_diff(
        self, added: set[str], removed: set[str],
    ) -> None:
        """Hot-attach newly-enabled channels and detach disabled ones.

        Called from :meth:`_on_config_change` whenever ``channels.enabled``
        changes. Each operation is independent — one adapter failing to
        attach must not block the others.
        """
        if self._gateway is None:
            return
        if removed:
            for name in sorted(removed):
                try:
                    await self._gateway.unregister_channel(name)
                except Exception as exc:  # noqa: BLE001 — never let one channel break the diff
                    logger.error("Hot-detach %s failed: %s", name, exc)
        if added:
            from ..channels.registry import get_registry
            registry = get_registry(self.config.home_dir)
            registry.scan(refresh=True)
            available = registry.available()
            for name in sorted(added):
                if name not in available:
                    logger.warning(
                        "Hot-attach %s skipped: not in registry. "
                        "Run channel_install or drop an adapter in "
                        "~/.predacore/channels/.", name,
                    )
                    continue
                adapter = registry.create(name, self.config)
                if adapter is None:
                    continue
                try:
                    self._gateway.register_channel(adapter)
                    await self._gateway.start_channel(name)
                except (ImportError, RuntimeError, OSError, ValueError) as exc:
                    logger.error("Hot-attach %s failed: %s", name, exc)

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

        # Stop MCP servers (before gateway — their tools may still be in flight)
        try:
            from .mcp_registry import get_registry as _mcp_reg
            await _stop("MCP servers", _mcp_reg().stop_all())
        except ImportError:
            pass

        # Stop gateway (stops all channels)
        if self._gateway:
            await _stop("Gateway", self._gateway.stop())

        # Stop DB server (last — other subsystems may flush during cleanup)
        if self._memory_server:
            await _stop("Memory server", self._memory_server.stop())
        if self._db_server:
            await _stop("DB server", self._db_server.stop())

        # Remove PID file
        self.pid_manager.cleanup()

        elapsed = self.uptime
        logger.info(
            "PredaCore Daemon stopped (uptime=%.0fs, heartbeats=%d)",
            elapsed,
            self._heartbeat_count,
        )


# -- CLI Helpers --


def stop_daemon(config: PredaCoreConfig) -> bool:
    """
    Send SIGTERM to the running daemon.
    Returns True only if the daemon fully stopped.
    """
    pid_manager = PIDManager(str(Path(config.home_dir) / "predacore.pid"))

    if not pid_manager.is_running():
        return False

    pid = pid_manager.read()
    if pid is None:
        # PID file disappeared between is_running() and read()
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to PID %d", pid)

        # Wait up to 10 seconds for graceful shutdown
        for _ in range(20):
            time.sleep(0.5)
            if not pid_manager.is_running():
                pid_manager.cleanup()
                return True

        # Escalate to SIGKILL if the daemon is stuck (e.g. blocked in a
        # long-running subprocess like Gemini CLI). Graceful shutdown
        # can't interrupt a blocking syscall, so force-terminate and
        # clean up the pidfile so the next `predacore` works.
        logger.warning(
            "Daemon PID %d did not respond to SIGTERM in 10s — escalating to SIGKILL",
            pid,
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pid_manager.cleanup()
            return True
        for _ in range(10):
            time.sleep(0.3)
            if not pid_manager.is_running():
                pid_manager.cleanup()
                return True
        logger.error("Daemon PID %d survived SIGKILL — check process state manually", pid)
        return False

    except ProcessLookupError:
        pid_manager.cleanup()
        return False
    except PermissionError:
        logger.error("Permission denied sending signal to PID %d", pid)
        return False


def generate_launchd_plist(config: PredaCoreConfig) -> str:
    """
    Generate a macOS launchd plist for auto-starting PredaCore.

    Install with:
        predacore install
        launchctl load ~/Library/LaunchAgents/com.predacore.plist
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
    <string>com.predacore</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>src.predacore.cli</string>
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
        <key>PREDACORE_HOME</key>
        <string>{home_dir}</string>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>"""
    return plist


def install_launchd(config: PredaCoreConfig) -> Path:
    """Install the launchd plist to ~/Library/LaunchAgents/."""
    plist_content = generate_launchd_plist(config)
    target = Path.home() / "Library" / "LaunchAgents" / "com.predacore.plist"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plist_content)

    # Ensure log directory exists
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)

    return target
