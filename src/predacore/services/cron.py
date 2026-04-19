"""
PredaCore Cron Engine — lightweight scheduled task execution.

Reads job definitions from ~/.predacore/cron.yaml and executes
them on schedule by routing messages through the Gateway.

No heavy dependencies — uses a simple tick-based approach that
checks every 60 seconds whether any jobs should fire.

Job format in cron.yaml:
    jobs:
      - name: daily_summary
        schedule: "0 9 * * *"
        action: "Give me a summary of what happened overnight"
        channel: cli
        enabled: true

      - name: market_check
        schedule: "*/30 * * * *"
        action: "Check BTC, ETH, SOL prices and give a brief update"
        channel: telegram
        enabled: true

Cron expression format (standard 5-field):
    minute hour day_of_month month day_of_week
    ┬      ┬    ┬            ┬     ┬
    │      │    │            │     └─ 0-6 (Sun-Sat)
    │      │    │            └─────── 1-12
    │      │    └──────────────────── 1-31
    │      └───────────────────────── 0-23
    └──────────────────────────────── 0-59

    Special: * (any), */N (every N), N-M (range), N,M (list)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from ..config import PredaCoreConfig

logger = logging.getLogger(__name__)


# ── Cron Expression Parser ───────────────────────────────────────────


class CronExpression:
    """
    Parse and evaluate standard 5-field cron expressions.

    Supports: *, */N, N-M, N,M, and plain integers.
    """

    def __init__(self, expression: str):
        self.expression = expression.strip()
        parts = self.expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields, got {len(parts)}: '{expression}'"
            )

        self.minute = self._parse_field(parts[0], 0, 59)
        self.hour = self._parse_field(parts[1], 0, 23)
        self.day_of_month = self._parse_field(parts[2], 1, 31)
        self.month = self._parse_field(parts[3], 1, 12)
        self.day_of_week = self._parse_field(parts[4], 0, 6)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        return (
            dt.minute in self.minute
            and dt.hour in self.hour
            and dt.day in self.day_of_month
            and dt.month in self.month
            and dt.weekday() in self._convert_dow(self.day_of_week)
        )

    @staticmethod
    def _convert_dow(cron_dow: set) -> set:
        """
        Convert cron day-of-week (0=Sun) to Python weekday (0=Mon).
        Cron: 0=Sun, 1=Mon, ..., 6=Sat
        Python: 0=Mon, 1=Tue, ..., 6=Sun
        """
        python_dow = set()
        for d in cron_dow:
            if d == 0:
                python_dow.add(6)  # Sun
            else:
                python_dow.add(d - 1)
        return python_dow

    @staticmethod
    def _parse_field(field: str, min_val: int, max_val: int) -> set:
        """
        Parse a single cron field into a set of matching values.

        Supports:
          *       → all values
          */N     → every N values
          N       → exact value
          N-M     → range (inclusive)
          N,M,... → list
        """
        result = set()

        for part in field.split(","):
            part = part.strip()

            if part == "*":
                return set(range(min_val, max_val + 1))

            if part.startswith("*/"):
                step = int(part[2:])
                if step <= 0:
                    raise ValueError(f"Step must be positive: {part}")
                return set(range(min_val, max_val + 1, step))

            if "-" in part:
                start, end = part.split("-", 1)
                s, e = int(start), int(end)
                if s < min_val or e > max_val or s > e:
                    raise ValueError(
                        f"Range {part} out of bounds [{min_val}-{max_val}]"
                    )
                result.update(range(s, e + 1))
            else:
                val = int(part)
                if val < min_val or val > max_val:
                    raise ValueError(f"Value {val} out of bounds [{min_val}-{max_val}]")
                result.add(val)

        return result

    def __repr__(self) -> str:
        return f"CronExpression('{self.expression}')"


# ── Cron Job ─────────────────────────────────────────────────────────


@dataclass
class CronJob:
    """A scheduled task definition."""

    name: str
    schedule: str
    action: str
    channel: str = "cli"
    enabled: bool = True
    last_run: float | None = None
    run_count: int = 0

    def __post_init__(self):
        self._expression = CronExpression(self.schedule)

    def should_run(self, now: datetime) -> bool:
        """Check if this job should run at the given time."""
        if not self.enabled:
            return False

        if not self._expression.matches(now):
            return False

        # Prevent running twice in the same minute
        if self.last_run:
            last_run_dt = datetime.fromtimestamp(self.last_run, tz=timezone.utc)
            if (
                last_run_dt.year == now.year
                and last_run_dt.month == now.month
                and last_run_dt.day == now.day
                and last_run_dt.hour == now.hour
                and last_run_dt.minute == now.minute
            ):
                return False

        return True

    def mark_run(self) -> None:
        """Record that the job ran."""
        self.last_run = time.time()
        self.run_count += 1


# ── Cron Engine ──────────────────────────────────────────────────────


class CronEngine:
    """
    Tick-based cron scheduler.

    Checks every 60 seconds whether any scheduled jobs match the
    current minute and fires them through the Gateway.
    """

    TICK_INTERVAL = 60  # seconds

    def __init__(self, config: PredaCoreConfig, gateway):
        self.config = config
        self.gateway = gateway
        self.jobs: list[CronJob] = []
        self._task: asyncio.Task | None = None
        self._job_tasks: set = set()  # Track running job tasks
        self._running = False

        # Load jobs from cron file
        self._load_jobs()

        # Built-in: memory consolidation every 6 hours
        self._register_builtin_jobs()

    def _load_jobs(self) -> None:
        """Load cron jobs from ~/.predacore/cron.yaml."""
        if yaml is None:
            logger.info("PyYAML not installed — cron engine disabled")
            return

        cron_file = (
            Path(self.config.daemon.cron_file) if self.config.daemon.cron_file else None
        )

        if not cron_file:
            # Try default location
            cron_file = Path(self.config.home_dir) / "cron.yaml"

        if not cron_file.exists():
            logger.info("No cron file found at %s — cron engine idle", cron_file)
            return

        try:
            data = yaml.safe_load(cron_file.read_text(encoding="utf-8"))
            if not data or "jobs" not in data:
                logger.warning("Cron file has no 'jobs' key: %s", cron_file)
                return

            for job_def in data["jobs"]:
                try:
                    job = CronJob(
                        name=job_def["name"],
                        schedule=job_def["schedule"],
                        action=job_def["action"],
                        channel=job_def.get("channel", "cli"),
                        enabled=job_def.get("enabled", True),
                    )
                    self.jobs.append(job)
                    logger.info("Loaded cron job: %s (%s)", job.name, job.schedule)
                except (KeyError, ValueError) as e:
                    logger.error("Invalid cron job definition: %s — %s", job_def, e)

            logger.info("Loaded %d cron jobs from %s", len(self.jobs), cron_file)

        except yaml.YAMLError as e:
            logger.error("Failed to parse cron file %s: %s", cron_file, e)
        except OSError as e:
            logger.error("Failed to load cron file %s: %s", cron_file, e)

    def _register_builtin_jobs(self) -> None:
        """Register built-in system jobs (memory consolidation, etc.)."""
        # Skip if a user-defined job with this name already exists
        existing_names = {j.name for j in self.jobs}
        if "memory_consolidation" not in existing_names:
            try:
                self.jobs.append(
                    CronJob(
                        name="memory_consolidation",
                        schedule="0 */6 * * *",  # every 6 hours
                        action="__consolidate_memory__",
                        channel="system",
                        enabled=True,
                    )
                )
                logger.info(
                    "Registered built-in cron job: memory_consolidation (every 6h)"
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Failed to register memory_consolidation job: %s", exc)

    async def start(self) -> None:
        """Start the cron scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("Cron engine started")

    async def stop(self) -> None:
        """Stop the cron scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cron engine stopped")

    async def _tick_loop(self) -> None:
        """Main scheduler loop — checks every TICK_INTERVAL seconds."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for job in self.jobs:
                    if job.should_run(now):
                        # Prevent overlapping execution of the same job
                        if any(t.get_name() == f"cron-{job.name}" for t in self._job_tasks):
                            logger.debug("Skipping cron job %s — still running", job.name)
                            continue
                        task = asyncio.create_task(
                            self._execute_job(job),
                            name=f"cron-{job.name}",
                        )
                        self._job_tasks.add(task)
                        task.add_done_callback(self._job_tasks.discard)

                # Sleep until the next minute boundary (+ small buffer)
                seconds_until_next = 60 - now.second + 1
                await asyncio.sleep(min(seconds_until_next, self.TICK_INTERVAL))

            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Cron tick error: %s", e, exc_info=True)
                await asyncio.sleep(self.TICK_INTERVAL)

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single cron job by routing through the gateway."""
        logger.info("Executing cron job: %s", job.name)
        job.mark_run()

        # Built-in internal actions
        if job.action == "__consolidate_memory__":
            await self._run_memory_consolidation()
            return

        try:
            # Route through gateway as a system-initiated message
            from ..gateway import IncomingMessage
            incoming = IncomingMessage(
                channel=job.channel,
                user_id="cron",
                text=f"[Scheduled Task: {job.name}]\n\n{job.action}",
                metadata={"source": "cron", "job_name": job.name},
            )
            await self.gateway.handle_message(incoming)
            logger.info("Cron job completed: %s (run #%d)", job.name, job.run_count)
        except Exception as e:
            logger.error("Cron job failed: %s — %s", job.name, e, exc_info=True)

    async def _run_memory_consolidation(self) -> None:
        """Run unified memory consolidation (entity extraction, linking, pruning)."""
        try:
            # Access unified memory from gateway's core
            core = getattr(self.gateway, "_core", None) or getattr(
                self.gateway, "core", None
            )
            if core is None:
                logger.debug("Memory consolidation skipped: no core reference")
                return
            um = getattr(core.tools, "_unified_memory", None)
            if um is None:
                logger.debug(
                    "Memory consolidation skipped: unified memory not available"
                )
                return

            from ..memory import MemoryConsolidator

            llm = getattr(core, "llm", None)
            consolidator = MemoryConsolidator(store=um, llm=llm)
            stats = await consolidator.consolidate()
            logger.info(
                "Memory consolidation complete: %s",
                ", ".join(f"{k}={v}" for k, v in stats.items()),
            )
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            logger.warning("Memory consolidation failed: %s", exc, exc_info=True)

    def add_job(self, job: CronJob) -> None:
        """Add a job dynamically."""
        self.jobs.append(job)
        logger.info("Added cron job: %s", job.name)

    def remove_job(self, name: str) -> bool:
        """Remove a job by name."""
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.name != name]
        removed = len(self.jobs) < before
        if removed:
            logger.info("Removed cron job: %s", name)
        return removed

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return a list of all jobs with their status."""
        return [
            {
                "name": job.name,
                "schedule": job.schedule,
                "action": job.action[:80],
                "channel": job.channel,
                "enabled": job.enabled,
                "last_run": job.last_run,
                "run_count": job.run_count,
            }
            for job in self.jobs
        ]
