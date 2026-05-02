"""Cron/scheduled task handler: cron_task (create, list, cancel, status)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    blocked,
    invalid_param,
    missing_param,
    resource_not_found,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dangerous command patterns (shared with shell.py — imported here to avoid
# circular imports; kept as a module constant)
# ---------------------------------------------------------------------------

_DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+(-\S*)?-r\S*\s+/\s*($|[;&|])", re.IGNORECASE),
    re.compile(r"\brm\s+(-\S*)?-r\S*\s+/\*", re.IGNORECASE),
    re.compile(r"\bdd\s+.*\bof\s*=\s*/dev/[sh]d", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", re.IGNORECASE),
    re.compile(r"\bchmod\s+(-\S+\s+)*777\s+/\s*($|[;&|])", re.IGNORECASE),
    re.compile(r">\s*/dev/[sh]d", re.IGNORECASE),
    re.compile(r"\bformat\s+[A-Za-z]:", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# In-memory task store (persisted to ~/.predacore/cron_tasks.json)
# ---------------------------------------------------------------------------

_CRON_TASKS: dict[str, dict[str, Any]] = {}
_CRON_COUNTER = 0
_CRON_TASK_REFS: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
_CRON_MAX_TASKS = 20
_CRON_FILE = Path.home() / ".predacore" / "cron_tasks.json"
_CRON_LOADED = False


def _cron_save() -> None:
    """Persist cron tasks to disk (exclude asyncio refs)."""
    try:
        _CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for tid, task in _CRON_TASKS.items():
            serializable[tid] = {k: v for k, v in task.items() if not k.startswith("_")}
        _CRON_FILE.write_text(json.dumps(serializable, indent=2, default=str))
    except OSError as _exc:
        logger.debug("Failed to save cron tasks: %s", _exc)


def _cron_load() -> None:
    """Load persisted cron tasks from disk (called once on first access)."""
    global _CRON_LOADED, _CRON_COUNTER
    if _CRON_LOADED:
        return
    _CRON_LOADED = True
    try:
        if _CRON_FILE.exists():
            data = json.loads(_CRON_FILE.read_text())
            for tid, task in data.items():
                task["active"] = False  # don't auto-restart
                _CRON_TASKS[tid] = task
            existing_ids = []
            for t in _CRON_TASKS:
                if t.startswith("cron_"):
                    try:
                        existing_ids.append(int(t.replace("cron_", "")))
                    except ValueError:
                        pass
            if existing_ids:
                _CRON_COUNTER = max(existing_ids)
            logger.info("Loaded %d persisted cron tasks from disk", len(data))
    except (OSError, json.JSONDecodeError, ValueError) as _exc:
        logger.debug("Failed to load cron tasks: %s", _exc)


def _cron_prune_dead() -> None:
    """Remove inactive tasks dead for >1 hour to prevent memory leaks."""
    now = time.time()
    dead = [
        tid for tid, t in _CRON_TASKS.items()
        if not t.get("active") and now - (t.get("last_run") or t.get("created_at", now)) > 3600
    ]
    for tid in dead:
        _CRON_TASKS.pop(tid, None)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


async def handle_cron_task(args: dict[str, Any], ctx: ToolContext) -> str:
    """Create, list, cancel, or check status of scheduled recurring tasks."""
    global _CRON_COUNTER
    _cron_load()
    action = str(args.get("action") or "").strip().lower()

    if action == "list":
        _cron_prune_dead()
        if not _CRON_TASKS:
            return "[No scheduled tasks]"
        lines = ["## Scheduled Tasks\n"]
        for tid, task in _CRON_TASKS.items():
            status = "active" if task.get("active") else "cancelled"
            lines.append(
                f"- **{task['name']}** (id={tid}, {status}) "
                f"every {task['interval_min']}m, "
                f"runs: {task.get('run_count', 0)}/{task.get('max_runs', 0) or 'inf'}"
            )
        return "\n".join(lines)

    if action == "cancel":
        task_id = str(args.get("task_id") or "").strip()
        if not task_id or task_id not in _CRON_TASKS:
            raise resource_not_found("Cron task", task_id or "(empty)", tool="cron_task")
        _CRON_TASKS[task_id]["active"] = False
        atask = _CRON_TASKS[task_id].get("_asyncio_task")
        if atask and not atask.done():
            atask.cancel()
        _cron_save()
        return f"Cancelled task: {_CRON_TASKS[task_id]['name']} (id={task_id})"

    if action == "status":
        task_id = str(args.get("task_id") or "").strip()
        if not task_id or task_id not in _CRON_TASKS:
            raise resource_not_found("Cron task", task_id or "(empty)", tool="cron_task")
        task = _CRON_TASKS[task_id]
        safe = {k: v for k, v in task.items() if not k.startswith("_")}
        return json.dumps(safe, indent=2, default=str)

    if action == "create":
        _cron_prune_dead()
        active_count = sum(1 for t in _CRON_TASKS.values() if t.get("active"))
        if active_count >= _CRON_MAX_TASKS:
            raise ToolError(
                f"Too many active cron tasks ({active_count}/{_CRON_MAX_TASKS}). Cancel some first.",
                kind=ToolErrorKind.LIMIT_EXCEEDED,
                tool_name="cron_task",
            )

        name = str(args.get("task_name") or "").strip()
        if not name:
            raise missing_param("task_name", tool="cron_task")
        command = str(args.get("command") or "").strip()
        if not command:
            raise missing_param("command", tool="cron_task")

        for pattern in _DANGEROUS_COMMAND_PATTERNS:
            if pattern.search(command):
                raise blocked(
                    f"cron command matched dangerous pattern: {pattern.pattern}",
                    tool="cron_task",
                )

        interval = max(1, min(int(args.get("interval_minutes") or 10), 1440))
        max_runs = max(0, int(args.get("max_runs") or 0))

        _CRON_COUNTER += 1
        task_id = f"cron_{_CRON_COUNTER}"

        task_record: dict[str, Any] = {
            "id": task_id,
            "name": name,
            "command": command,
            "interval_min": interval,
            "max_runs": max_runs,
            "run_count": 0,
            "active": True,
            "created_at": time.time(),
            "last_run": None,
            "last_output": None,
            "last_error": None,
        }
        _CRON_TASKS[task_id] = task_record
        _cron_save()

        async def _cron_loop():
            try:
                while _CRON_TASKS.get(task_id, {}).get("active", False):
                    await asyncio.sleep(interval * 60)
                    task = _CRON_TASKS.get(task_id)
                    if not task or not task.get("active"):
                        break
                    if max_runs > 0 and task.get("run_count", 0) >= max_runs:
                        task["active"] = False
                        break
                    task["run_count"] = task.get("run_count", 0) + 1
                    task["last_run"] = time.time()
                    logger.info("Cron task %s (%s) — run #%d", task_id, name, task["run_count"])
                    proc = None
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *shlex.split(command),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        # Trust-tiered cron timeout. yolo keeps the 1h ceiling
                        # for long-running batch jobs; ask_everytime caps at 5m
                        # so a runaway `sleep 99999` can't camp on a worker.
                        _trust = getattr(ctx.config.security, "trust_level", "ask_everytime")
                        _cron_timeout = 3600 if _trust == "yolo" else 300
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=_cron_timeout
                        )
                        task["last_output"] = (stdout or b"").decode(errors="replace")[:200_000]
                        task["last_error"] = (
                            (stderr or b"").decode(errors="replace")[:100_000]
                            if proc.returncode != 0 else None
                        )
                    except asyncio.TimeoutError:
                        task["last_error"] = f"[Command timed out after {_cron_timeout}s (trust={_trust})]"
                        if proc is not None:
                            try:
                                proc.kill()
                                await proc.wait()
                            except ProcessLookupError:
                                logger.debug("cron_task: process already exited")
                    except Exception as run_exc:
                        task["last_error"] = str(run_exc)[:5000]
            finally:
                task = _CRON_TASKS.get(task_id)
                if task:
                    task["active"] = False

        atask = asyncio.create_task(_cron_loop())
        task_record["_asyncio_task"] = atask
        _CRON_TASK_REFS.add(atask)
        atask.add_done_callback(_CRON_TASK_REFS.discard)

        return json.dumps({
            "task_id": task_id,
            "name": name,
            "interval_minutes": interval,
            "command": command,
            "max_runs": max_runs or "unlimited",
            "status": "active",
        }, indent=2)

    raise invalid_param(
        "action",
        f"unknown cron action: {action}. Valid: create, list, cancel, status",
        tool="cron_task",
    )
