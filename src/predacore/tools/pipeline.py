"""
PredaCore Tool Pipeline — Chain multiple tools into composable workflows.

Enables declarative tool composition where the output of one tool feeds
into the next. Supports:
  - Sequential chaining: read_file → python_exec (with file content as input)
  - Variable substitution: {{prev}} for previous step's output
  - Conditional steps: only execute if previous step matched a condition
  - Error handling: stop_on_error or continue-and-collect
  - Parallel fan-out: run independent steps concurrently
  - Approval gates: pause execution, return a resume token, continue later
  - Resume tokens: compact token to resume a paused pipeline without re-running

Usage (from LLM or API):
    pipeline = ToolPipeline(dispatcher)
    result = await pipeline.execute([
        {"tool": "read_file", "args": {"path": "data.csv"}},
        {"tool": "python_exec", "args": {"code": "print(len('{{prev}}'.splitlines()))"}},
    ])

Usage (with approval gate):
    result = await pipeline.execute([
        {"tool": "web_search", "args": {"query": "..."}},
        {"tool": "summarize", "args": {"text": "{{prev}}"}, "approval": "required"},
        {"tool": "send_email", "args": {"body": "{{prev}}"}},
    ])
    # result.status == "needs_approval", result.resume_token set
    # Later:
    result = await pipeline.resume(token, approve=True)

Usage (programmatic):
    result = await pipeline.execute([
        PipelineStep(tool="git_context", args={}),
        PipelineStep(tool="python_exec", args={"code": "# analyze: {{prev}}"}),
    ])
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# Maximum steps in a single pipeline (prevent infinite loops)
_MAX_PIPELINE_STEPS = 200  # raised from 20 per "remove all limits"

# Maximum total pipeline execution time (seconds)
_MAX_PIPELINE_TIMEOUT = 3600  # raised from 300 (5min → 1h)

# Resume token expiry (seconds) — paused pipelines expire after 24h
_TOKEN_EXPIRY_SEC = 86400

# Variable pattern: {{prev}}, {{step.0}}, {{step.1.key}}, etc.
_VAR_PATTERN = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


@dataclass
class PipelineStep:
    """A single step in a tool pipeline."""
    tool: str                                    # Tool name
    args: dict[str, Any] = field(default_factory=dict)  # Tool arguments
    name: str = ""                               # Optional step name for referencing
    condition: str = ""                          # Only run if condition matches prev output
    on_error: str = "stop"                       # "stop" or "continue"
    timeout: float = 0                           # Per-step timeout override (0 = default)
    approval: str = ""                           # "required" = pause before this step

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineStep":
        """Create a PipelineStep from a dictionary specification."""
        return cls(
            tool=str(d.get("tool", "")).strip(),
            args=dict(d.get("args", {})),
            name=str(d.get("name", "")).strip(),
            condition=str(d.get("condition", "")).strip(),
            on_error=str(d.get("on_error", "stop")).strip(),
            timeout=float(d.get("timeout", 0)),
            approval=str(d.get("approval", "")).strip().lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool": self.tool, "args": self.args}
        if self.name:
            d["name"] = self.name
        if self.condition:
            d["condition"] = self.condition
        if self.on_error != "stop":
            d["on_error"] = self.on_error
        if self.timeout:
            d["timeout"] = self.timeout
        if self.approval:
            d["approval"] = self.approval
        return d


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""
    ok: bool
    steps_completed: int
    total_steps: int
    results: list[dict[str, Any]]
    final_output: str
    elapsed_ms: float
    aborted_at: int | None = None
    error: str = ""
    # Resume token fields
    status: str = "ok"                # "ok" | "needs_approval" | "cancelled"
    resume_token: str = ""            # Compact token to resume a paused pipeline
    approval_prompt: str = ""         # Human-readable description of what needs approval
    pending_step: int | None = None   # Which step is waiting for approval

    def to_dict(self) -> dict[str, Any]:
        """Serialize this pipeline result to dictionary."""
        d: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "final_output": self.final_output[:2000],
            "elapsed_ms": round(self.elapsed_ms, 1),
            "aborted_at": self.aborted_at,
            "error": self.error,
            "step_results": [
                {
                    "step": r["step"],
                    "tool": r["tool"],
                    "status": r["status"],
                    "elapsed_ms": round(r["elapsed_ms"], 1),
                    "output_preview": r["output"][:500],
                }
                for r in self.results
            ],
        }
        if self.resume_token:
            d["resume_token"] = self.resume_token
            d["approval_prompt"] = self.approval_prompt
            d["pending_step"] = self.pending_step
        return d


# ══════════════════════════════════════════════════════════════════════
# Pipeline State Store — SQLite-backed durable state for resume tokens
# ══════════════════════════════════════════════════════════════════════

_CREATE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS pipeline_states (
    token       TEXT PRIMARY KEY,
    state_json  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'paused'
)
"""


class _PipelineStateStore:
    """Persist paused pipeline state to SQLite for durable resume tokens.

    Tokens are compact random strings. The full pipeline state (completed
    steps, outputs, remaining steps) is stored in SQLite. This means:
    - Tokens survive daemon restarts
    - No state in the token itself (secure — can't be forged)
    - Expired states are auto-cleaned on access
    """

    def __init__(self, db_path: str | None = None, *, db_adapter=None):
        self._db_path = db_path or str(
            Path(os.getenv("PREDACORE_HOME", "~/.predacore")).expanduser() / "pipeline_states.db"
        )
        self._db_adapter = db_adapter  # Optional DBAdapter
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        if self._db_adapter:
            self._db_adapter._direct_executescript(
                "pipeline_states", _CREATE_STATE_TABLE
            )
            return
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(_CREATE_STATE_TABLE)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def save(self, state: dict[str, Any], expiry_sec: float = _TOKEN_EXPIRY_SEC) -> str:
        """Save pipeline state and return a compact resume token."""
        token = "jp_" + secrets.token_urlsafe(24)  # jp_ = predacore pipeline
        now = time.time()
        state_json = json.dumps(state, default=str)
        if self._db_adapter:
            self._db_adapter._direct_execute(
                "pipeline_states",
                "INSERT INTO pipeline_states (token, state_json, created_at, expires_at, status) "
                "VALUES (?, ?, ?, ?, 'paused')",
                [token, state_json, now, now + expiry_sec],
            )
            return token
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO pipeline_states (token, state_json, created_at, expires_at, status) "
                "VALUES (?, ?, ?, ?, 'paused')",
                (token, state_json, now, now + expiry_sec),
            )
            conn.commit()
        finally:
            conn.close()
        return token

    def load(self, token: str) -> dict[str, Any] | None:
        """Load pipeline state by token. Returns None if expired/missing."""
        if self._db_adapter:
            # Auto-clean expired states
            self._db_adapter._direct_execute(
                "pipeline_states",
                "DELETE FROM pipeline_states WHERE expires_at < ?",
                [time.time()],
            )
            rows = self._db_adapter._direct_query_dicts(
                "pipeline_states",
                "SELECT state_json, status FROM pipeline_states WHERE token = ?",
                [token],
            )
            if not rows:
                return None
            if rows[0]["status"] != "paused":
                return None
            return json.loads(rows[0]["state_json"])
        conn = self._conn()
        try:
            # Auto-clean expired states
            conn.execute("DELETE FROM pipeline_states WHERE expires_at < ?", (time.time(),))
            conn.commit()

            row = conn.execute(
                "SELECT state_json, status FROM pipeline_states WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            if row[1] != "paused":
                return None  # Already resumed or cancelled
            return json.loads(row[0])
        finally:
            conn.close()

    def mark_used(self, token: str, status: str = "resumed") -> None:
        """Mark a token as used (resumed or cancelled)."""
        if self._db_adapter:
            self._db_adapter._direct_execute(
                "pipeline_states",
                "UPDATE pipeline_states SET status = ? WHERE token = ?",
                [status, token],
            )
            return
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE pipeline_states SET status = ? WHERE token = ?",
                (status, token),
            )
            conn.commit()
        finally:
            conn.close()

    def cleanup_expired(self) -> int:
        """Remove all expired pipeline states. Returns count removed."""
        if self._db_adapter:
            result = self._db_adapter._direct_execute(
                "pipeline_states",
                "DELETE FROM pipeline_states WHERE expires_at < ?",
                [time.time()],
            )
            return result.get("rowcount", 0)
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM pipeline_states WHERE expires_at < ?", (time.time(),)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


# Singleton state store
_state_store: _PipelineStateStore | None = None


def _get_state_store() -> _PipelineStateStore:
    global _state_store
    if _state_store is None:
        _state_store = _PipelineStateStore()
    return _state_store


# ══════════════════════════════════════════════════════════════════════
# ToolPipeline — Main pipeline engine
# ══════════════════════════════════════════════════════════════════════

class ToolPipeline:
    """Execute multi-step tool pipelines with variable substitution.

    The pipeline engine:
      1. Validates all steps before execution
      2. Executes steps sequentially (respecting dependencies)
      3. Substitutes {{prev}} and {{step.N}} variables in args
      4. Handles errors per step (stop or continue)
      5. Pauses at approval gates and returns resume tokens
      6. Resumes paused pipelines from where they left off
      7. Records full execution trace
    """

    def __init__(self, dispatcher: "ToolDispatcher"):
        self._dispatcher = dispatcher

    async def execute(
        self,
        steps: list[dict[str, Any] | PipelineStep],
        *,
        origin: str = "pipeline",
    ) -> PipelineResult:
        """Execute a tool pipeline.

        Args:
            steps: List of pipeline steps (dicts or PipelineStep objects).
            origin: Origin tag for execution history.

        Returns:
            PipelineResult with all step outputs and final result.
            If an approval gate is hit, status="needs_approval" with a resume_token.
        """
        t0 = time.time()

        # Parse steps
        parsed_steps: list[PipelineStep] = []
        for s in steps:
            if isinstance(s, PipelineStep):
                parsed_steps.append(s)
            elif isinstance(s, dict):
                parsed_steps.append(PipelineStep.from_dict(s))
            else:
                return PipelineResult(
                    ok=False, steps_completed=0, total_steps=len(steps),
                    results=[], final_output="",
                    elapsed_ms=(time.time() - t0) * 1000,
                    error=f"Invalid step type: {type(s).__name__}",
                )

        # Validate
        if not parsed_steps:
            return PipelineResult(
                ok=False, steps_completed=0, total_steps=0,
                results=[], final_output="",
                elapsed_ms=(time.time() - t0) * 1000,
                error="Pipeline has no steps",
            )

        if len(parsed_steps) > _MAX_PIPELINE_STEPS:
            return PipelineResult(
                ok=False, steps_completed=0, total_steps=len(parsed_steps),
                results=[], final_output="",
                elapsed_ms=(time.time() - t0) * 1000,
                error=f"Pipeline too long: {len(parsed_steps)} steps (max {_MAX_PIPELINE_STEPS})",
            )

        for i, step in enumerate(parsed_steps):
            if not step.tool:
                return PipelineResult(
                    ok=False, steps_completed=0, total_steps=len(parsed_steps),
                    results=[], final_output="",
                    elapsed_ms=(time.time() - t0) * 1000,
                    error=f"Step {i} has no tool name",
                )

        # Execute from step 0 with empty state
        return await self._execute_from(
            parsed_steps, start_idx=0, results=[], step_outputs={},
            prev_output="", origin=origin, t0=t0,
        )

    async def resume(
        self,
        token: str,
        *,
        approve: bool = True,
        origin: str = "pipeline_resume",
    ) -> PipelineResult:
        """Resume a paused pipeline from a resume token.

        Args:
            token: The resume token from a paused pipeline.
            approve: True to continue execution, False to cancel.
            origin: Origin tag for execution history.

        Returns:
            PipelineResult — continues from where the pipeline paused.
        """
        t0 = time.time()
        store = _get_state_store()

        state = store.load(token)
        if state is None:
            return PipelineResult(
                ok=False, steps_completed=0, total_steps=0,
                results=[], final_output="",
                elapsed_ms=(time.time() - t0) * 1000,
                error="Invalid or expired resume token",
                status="cancelled",
            )

        if not approve:
            store.mark_used(token, status="cancelled")
            return PipelineResult(
                ok=False,
                steps_completed=state.get("completed", 0),
                total_steps=state.get("total_steps", 0),
                results=state.get("results", []),
                final_output=state.get("prev_output", ""),
                elapsed_ms=(time.time() - t0) * 1000,
                status="cancelled",
                error="Pipeline cancelled by user",
            )

        # Mark token as used
        store.mark_used(token, status="resumed")

        # Reconstruct pipeline state
        all_steps = [PipelineStep.from_dict(s) for s in state["all_steps"]]
        resume_idx = state["resume_idx"]
        results = state.get("results", [])
        step_outputs = state.get("step_outputs", {})
        prev_output = state.get("prev_output", "")

        logger.info(
            "Resuming pipeline from step %d/%d (token=%s...)",
            resume_idx, len(all_steps), token[:12],
        )

        return await self._execute_from(
            all_steps, start_idx=resume_idx, results=results,
            step_outputs=step_outputs, prev_output=prev_output,
            origin=origin, t0=t0,
        )

    async def _execute_from(
        self,
        steps: list[PipelineStep],
        *,
        start_idx: int,
        results: list[dict[str, Any]],
        step_outputs: dict[str, str],
        prev_output: str,
        origin: str,
        t0: float,
    ) -> PipelineResult:
        """Core execution loop — runs steps from start_idx onward.

        Shared by both execute() and resume() to avoid duplicating the loop.
        """
        completed = len([r for r in results if r.get("status") == "ok"])

        for i in range(start_idx, len(steps)):
            step = steps[i]
            step_t0 = time.time()
            step_name = step.name or str(i)

            # Check overall timeout
            if (time.time() - t0) > _MAX_PIPELINE_TIMEOUT:
                return PipelineResult(
                    ok=False, steps_completed=completed,
                    total_steps=len(steps),
                    results=results, final_output=prev_output,
                    elapsed_ms=(time.time() - t0) * 1000,
                    aborted_at=i,
                    error=f"Pipeline timeout exceeded ({_MAX_PIPELINE_TIMEOUT}s)",
                )

            # ── Approval gate ─────────────────────────────────────
            if step.approval == "required":
                # Pause execution, save state, return resume token
                state = {
                    "all_steps": [s.to_dict() for s in steps],
                    "resume_idx": i,  # Resume AT this step (not after)
                    "results": results,
                    "step_outputs": step_outputs,
                    "prev_output": prev_output,
                    "completed": completed,
                    "total_steps": len(steps),
                    "paused_at": time.time(),
                }
                token = _get_state_store().save(state)

                # Build approval prompt
                remaining = len(steps) - i
                prompt = (
                    f"Pipeline paused before step {i + 1}/{len(steps)}: "
                    f"**{step.tool}**"
                )
                if step.name:
                    prompt += f" ({step.name})"
                prompt += f"\n{remaining} step(s) remaining."
                if prev_output:
                    preview = prev_output[:300]
                    prompt += f"\nPrevious output: {preview}"

                logger.info(
                    "Pipeline paused at step %d (%s) — token=%s...",
                    i, step.tool, token[:12],
                )

                return PipelineResult(
                    ok=True,
                    steps_completed=completed,
                    total_steps=len(steps),
                    results=results,
                    final_output=prev_output,
                    elapsed_ms=(time.time() - t0) * 1000,
                    status="needs_approval",
                    resume_token=token,
                    approval_prompt=prompt,
                    pending_step=i,
                )

            # ── Condition check ───────────────────────────────────
            if step.condition:
                if not self._check_condition(step.condition, prev_output):
                    logger.info(
                        "Pipeline step %d (%s) skipped — condition '%s' not met",
                        i, step.tool, step.condition,
                    )
                    results.append({
                        "step": i, "tool": step.tool, "name": step_name,
                        "status": "skipped", "output": "",
                        "elapsed_ms": 0, "reason": f"condition '{step.condition}' not met",
                    })
                    continue

            # ── Substitute variables ──────────────────────────────
            resolved_args = self._substitute_vars(
                step.args, prev_output, step_outputs
            )

            # ── Execute step ──────────────────────────────────────
            logger.info("Pipeline step %d/%d: %s", i + 1, len(steps), step.tool)
            try:
                output = await self._dispatcher.dispatch(
                    step.tool,
                    resolved_args,
                    origin=origin,
                )
                step_elapsed = (time.time() - step_t0) * 1000

                # Check if output indicates an error
                is_error = (
                    isinstance(output, str)
                    and output.startswith("[")
                    and any(kw in output.lower() for kw in ("error", "blocked", "timed out", "denied"))
                )

                status = "error" if is_error else "ok"
                results.append({
                    "step": i, "tool": step.tool, "name": step_name,
                    "status": status, "output": output,
                    "elapsed_ms": step_elapsed,
                })

                if is_error and step.on_error == "stop":
                    return PipelineResult(
                        ok=False, steps_completed=completed,
                        total_steps=len(steps),
                        results=results, final_output=output,
                        elapsed_ms=(time.time() - t0) * 1000,
                        aborted_at=i,
                        error=f"Step {i} ({step.tool}) failed: {output[:200]}",
                    )

                # Store output for variable substitution
                prev_output = output if isinstance(output, str) else str(output)
                step_outputs[step_name] = prev_output
                step_outputs[str(i)] = prev_output
                completed += 1

            except Exception as exc:
                step_elapsed = (time.time() - step_t0) * 1000
                error_msg = f"[Pipeline step error: {exc}]"
                results.append({
                    "step": i, "tool": step.tool, "name": step_name,
                    "status": "error", "output": error_msg,
                    "elapsed_ms": step_elapsed,
                })

                if step.on_error == "stop":
                    return PipelineResult(
                        ok=False, steps_completed=completed,
                        total_steps=len(steps),
                        results=results, final_output=error_msg,
                        elapsed_ms=(time.time() - t0) * 1000,
                        aborted_at=i,
                        error=str(exc),
                    )

                prev_output = error_msg
                step_outputs[step_name] = prev_output
                step_outputs[str(i)] = prev_output

        return PipelineResult(
            ok=True, steps_completed=completed,
            total_steps=len(steps),
            results=results, final_output=prev_output,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    async def execute_parallel(
        self,
        steps: list[dict[str, Any] | PipelineStep],
        *,
        origin: str = "pipeline_parallel",
    ) -> PipelineResult:
        """Execute independent pipeline steps concurrently.

        All steps run in parallel — no variable substitution between them.
        Use for fan-out patterns (e.g., search multiple sources at once).
        """
        t0 = time.time()

        parsed_steps: list[PipelineStep] = []
        for s in steps:
            if isinstance(s, PipelineStep):
                parsed_steps.append(s)
            elif isinstance(s, dict):
                parsed_steps.append(PipelineStep.from_dict(s))

        if not parsed_steps:
            return PipelineResult(
                ok=False, steps_completed=0, total_steps=0,
                results=[], final_output="",
                elapsed_ms=(time.time() - t0) * 1000,
                error="No steps to execute",
            )

        if len(parsed_steps) > _MAX_PIPELINE_STEPS:
            return PipelineResult(
                ok=False, steps_completed=0, total_steps=len(parsed_steps),
                results=[], final_output="",
                elapsed_ms=(time.time() - t0) * 1000,
                error=f"Too many parallel steps: {len(parsed_steps)} (max {_MAX_PIPELINE_STEPS})",
            )

        # Launch all steps concurrently
        async def _run_step(i: int, step: PipelineStep) -> dict[str, Any]:
            step_t0 = time.time()
            try:
                output = await self._dispatcher.dispatch(
                    step.tool, step.args, origin=origin,
                )
                return {
                    "step": i, "tool": step.tool,
                    "name": step.name or str(i),
                    "status": "ok", "output": output,
                    "elapsed_ms": (time.time() - step_t0) * 1000,
                }
            except Exception as exc:
                return {
                    "step": i, "tool": step.tool,
                    "name": step.name or str(i),
                    "status": "error", "output": f"[Error: {exc}]",
                    "elapsed_ms": (time.time() - step_t0) * 1000,
                }

        tasks = [_run_step(i, s) for i, s in enumerate(parsed_steps)]
        results = await asyncio.gather(*tasks)
        results = list(results)  # type: ignore

        completed = sum(1 for r in results if r["status"] == "ok")
        all_ok = completed == len(parsed_steps)

        # Combine outputs
        combined = "\n---\n".join(
            f"[{r['tool']}]\n{r['output']}" for r in results
        )

        return PipelineResult(
            ok=all_ok, steps_completed=completed,
            total_steps=len(parsed_steps),
            results=results, final_output=combined,
            elapsed_ms=(time.time() - t0) * 1000,
        )

    # ── Variable Substitution ────────────────────────────────────

    _MAX_SUBSTITUTION_LEN = 50_000  # Cap substituted values to prevent injection amplification

    def _substitute_vars(
        self,
        args: dict[str, Any],
        prev_output: str,
        step_outputs: dict[str, str],
    ) -> dict[str, Any]:
        """Replace {{prev}}, {{step.N}}, {{step.name}} in arg values.

        Substitution values are truncated to _MAX_SUBSTITUTION_LEN to prevent
        injection amplification from prior step outputs.
        """
        # Truncate sources to prevent unbounded expansion
        prev_output = prev_output[:self._MAX_SUBSTITUTION_LEN] if prev_output else ""
        step_outputs = {
            k: v[:self._MAX_SUBSTITUTION_LEN] if isinstance(v, str) else v
            for k, v in step_outputs.items()
        }
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str):
                resolved[key] = self._substitute_string(value, prev_output, step_outputs)
            elif isinstance(value, dict):
                resolved[key] = self._substitute_vars(value, prev_output, step_outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._substitute_string(v, prev_output, step_outputs) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _substitute_string(
        self,
        text: str,
        prev_output: str,
        step_outputs: dict[str, str],
    ) -> str:
        """Replace variable references in a single string."""
        def _replace(match: re.Match) -> str:
            var_path = match.group(1)

            # {{prev}} — previous step's output
            if var_path == "prev":
                return prev_output

            # {{prev_lines}} — previous output line count
            if var_path == "prev_lines":
                return str(len(prev_output.splitlines()))

            # {{prev_len}} — previous output length
            if var_path == "prev_len":
                return str(len(prev_output))

            # {{step.N}} or {{step.name}} — specific step's output
            if var_path.startswith("step."):
                parts = var_path.split(".", 1)
                if len(parts) == 2:
                    step_ref = parts[1]
                    output = step_outputs.get(step_ref, "")
                    return output

            # Unknown variable — leave as-is
            return match.group(0)

        return _VAR_PATTERN.sub(_replace, text)

    # ── Condition Checking ───────────────────────────────────────

    @staticmethod
    def _check_condition(condition: str, prev_output: str) -> bool:
        """Evaluate a simple condition against previous output.

        Supported conditions:
          - "contains:text"     — prev output contains "text"
          - "not_contains:text" — prev output does NOT contain "text"
          - "starts_with:text"  — prev output starts with "text"
          - "not_empty"         — prev output is not empty
          - "is_error"          — prev output looks like an error
          - "not_error"         — prev output does NOT look like an error
          - "lines>N"           — prev output has more than N lines
          - "lines<N"           — prev output has fewer than N lines
        """
        condition = condition.strip().lower()

        if condition.startswith("contains:"):
            search_text = condition[len("contains:"):].strip()
            return search_text.lower() in prev_output.lower()

        if condition.startswith("not_contains:"):
            search_text = condition[len("not_contains:"):].strip()
            return search_text.lower() not in prev_output.lower()

        if condition.startswith("starts_with:"):
            prefix = condition[len("starts_with:"):].strip()
            return prev_output.lower().startswith(prefix.lower())

        if condition == "not_empty":
            return bool(prev_output.strip())

        if condition == "is_error":
            return prev_output.startswith("[") and any(
                kw in prev_output.lower() for kw in ("error", "blocked", "failed")
            )

        if condition == "not_error":
            return not (prev_output.startswith("[") and any(
                kw in prev_output.lower() for kw in ("error", "blocked", "failed")
            ))

        if condition.startswith("lines>"):
            try:
                n = int(condition[len("lines>"):])
                return len(prev_output.splitlines()) > n
            except ValueError:
                return True

        if condition.startswith("lines<"):
            try:
                n = int(condition[len("lines<"):])
                return len(prev_output.splitlines()) < n
            except ValueError:
                return True

        # Unknown condition — default to True (proceed)
        logger.warning("Unknown pipeline condition: %s — defaulting to True", condition)
        return True
