"""Pipeline handler: tool_pipeline — chain multiple tools into workflows.

Supports:
  - Sequential/parallel execution
  - Approval gates (pause before a step)
  - Resume tokens (continue a paused pipeline without re-running)
  - Save/load reusable workflow templates
  - List and delete saved workflows
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
)

# ── Workflow Library ──────────────────────────────────────────────

_WORKFLOW_DIR = Path(
    os.getenv("PREDACORE_HOME", "~/.predacore")
).expanduser() / "workflows"

# Allowed characters in workflow names (prevent path traversal)
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")


def _validate_name(name: str) -> str:
    """Validate and normalize a workflow name."""
    name = name.strip().lower().replace(" ", "-")
    if not _NAME_PATTERN.match(name):
        raise invalid_param(
            "name",
            "must be 1-64 alphanumeric/dash/underscore chars, start with letter or digit",
            tool="tool_pipeline",
        )
    return name


def _workflow_path(name: str) -> Path:
    return _WORKFLOW_DIR / f"{name}.workflow.json"


def _save_workflow(
    name: str,
    steps: list[dict[str, Any]],
    mode: str = "sequential",
    description: str = "",
) -> dict[str, Any]:
    """Save a pipeline as a reusable workflow template."""
    name = _validate_name(name)
    _WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    path = _workflow_path(name)

    workflow = {
        "name": name,
        "description": description,
        "mode": mode,
        "steps": steps,
        "created_at": time.time(),
        "version": 1,
    }

    path.write_text(json.dumps(workflow, indent=2, default=str))
    return {
        "status": "saved",
        "name": name,
        "path": str(path),
        "steps": len(steps),
        "mode": mode,
    }


def _load_workflow(name: str) -> dict[str, Any]:
    """Load a saved workflow by name."""
    name = _validate_name(name)
    path = _workflow_path(name)
    if not path.exists():
        raise ToolError(
            f"Workflow '{name}' not found",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="tool_pipeline",
            suggestion=f"Use action='list' to see available workflows",
        )
    return json.loads(path.read_text())


def _list_workflows() -> list[dict[str, Any]]:
    """List all saved workflows."""
    if not _WORKFLOW_DIR.exists():
        return []
    workflows = []
    for f in sorted(_WORKFLOW_DIR.glob("*.workflow.json")):
        try:
            data = json.loads(f.read_text())
            workflows.append({
                "name": data.get("name", f.stem.replace(".workflow", "")),
                "description": data.get("description", ""),
                "steps": len(data.get("steps", [])),
                "mode": data.get("mode", "sequential"),
                "created_at": data.get("created_at"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return workflows


def _delete_workflow(name: str) -> dict[str, Any]:
    """Delete a saved workflow."""
    name = _validate_name(name)
    path = _workflow_path(name)
    if not path.exists():
        raise ToolError(
            f"Workflow '{name}' not found",
            kind=ToolErrorKind.NOT_FOUND,
            tool_name="tool_pipeline",
        )
    path.unlink()
    return {"status": "deleted", "name": name}


# ── Main Handler ─────────────────────────────────────────────────

async def handle_tool_pipeline(args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute, resume, save, load, list, or delete tool pipelines.

    Actions:
        (default)  — Execute a pipeline from inline steps
        "resume"   — Resume a paused pipeline from a token
        "save"     — Save a pipeline as a reusable workflow
        "run"      — Load and execute a saved workflow
        "list"     — List all saved workflows
        "delete"   — Delete a saved workflow
        "show"     — Show a saved workflow's definition

    Args (execute — default):
        steps: list of step dicts
        mode: "sequential" (default) or "parallel"

    Args (resume):
        action: "resume"
        token: resume token from a paused pipeline
        approve: true (continue) or false (cancel)

    Args (save):
        action: "save"
        name: workflow name (alphanumeric, dashes, underscores)
        steps: list of step dicts
        mode: "sequential" or "parallel"
        description: optional human-readable description

    Args (run):
        action: "run"
        name: saved workflow name
        args_override: optional dict of arg overrides per step name

    Args (list):
        action: "list"

    Args (delete / show):
        action: "delete" or "show"
        name: workflow name

    Returns:
        JSON summary with results, or workflow metadata.
    """
    action = str(args.get("action", "")).strip().lower()

    # ── Save workflow ─────────────────────────────────────────
    if action == "save":
        name = args.get("name", "")
        if not name:
            raise missing_param("name", tool="tool_pipeline")
        steps = args.get("steps")
        if not steps or not isinstance(steps, list):
            raise missing_param("steps", tool="tool_pipeline")
        mode = str(args.get("mode", "sequential")).strip().lower()
        description = str(args.get("description", "")).strip()
        result = _save_workflow(name, steps, mode, description)
        return json.dumps(result, indent=2)

    # ── List workflows ────────────────────────────────────────
    if action == "list":
        workflows = _list_workflows()
        if not workflows:
            return json.dumps({"workflows": [], "message": "No saved workflows yet"})
        return json.dumps({"workflows": workflows, "count": len(workflows)}, indent=2)

    # ── Show workflow ─────────────────────────────────────────
    if action == "show":
        name = args.get("name", "")
        if not name:
            raise missing_param("name", tool="tool_pipeline")
        workflow = _load_workflow(name)
        return json.dumps(workflow, indent=2)

    # ── Delete workflow ───────────────────────────────────────
    if action == "delete":
        name = args.get("name", "")
        if not name:
            raise missing_param("name", tool="tool_pipeline")
        result = _delete_workflow(name)
        return json.dumps(result, indent=2)

    # ── Run saved workflow ────────────────────────────────────
    if action == "run":
        name = args.get("name", "")
        if not name:
            raise missing_param("name", tool="tool_pipeline")
        workflow = _load_workflow(name)
        steps = workflow.get("steps", [])
        mode = workflow.get("mode", "sequential")

        # Apply arg overrides if provided
        overrides = args.get("args_override", {})
        if overrides and isinstance(overrides, dict):
            for step in steps:
                step_name = step.get("name", "")
                if step_name in overrides and isinstance(overrides[step_name], dict):
                    step["args"].update(overrides[step_name])

        # Fall through to execution with loaded steps
        args = {"steps": steps, "mode": mode}

    # ── Resume paused pipeline ────────────────────────────────
    if action == "resume":
        token = args.get("token", "")
        if not token:
            raise missing_param("token", tool="tool_pipeline")
        approve = bool(args.get("approve", True))

        from ..pipeline import ToolPipeline

        _ref = getattr(ctx, "_dispatcher_ref", None)
        dispatcher = _ref() if callable(_ref) else _ref
        if dispatcher is None:
            raise ToolError(
                "Pipeline requires dispatcher reference",
                kind=ToolErrorKind.UNAVAILABLE,
                tool_name="tool_pipeline",
            )

        pipeline = ToolPipeline(dispatcher)
        result = await pipeline.resume(token, approve=approve)
        return json.dumps(result.to_dict(), indent=2, default=str)

    # ── Execute inline pipeline ───────────────────────────────
    steps = args.get("steps")
    if not steps:
        raise missing_param("steps", tool="tool_pipeline")
    if not isinstance(steps, list):
        raise invalid_param("steps", "must be a list of step objects", tool="tool_pipeline")

    mode = str(args.get("mode") or "sequential").strip().lower()
    if mode not in ("sequential", "parallel"):
        raise invalid_param(
            "mode", "must be 'sequential' or 'parallel'", tool="tool_pipeline"
        )

    from ..pipeline import ToolPipeline

    _ref = getattr(ctx, "_dispatcher_ref", None)
    dispatcher = _ref() if callable(_ref) else _ref
    if dispatcher is None:
        raise ToolError(
            "Pipeline requires dispatcher reference (not available in current context)",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="tool_pipeline",
            suggestion="Pipeline is available when running through the standard dispatch path",
        )

    pipeline = ToolPipeline(dispatcher)

    # Auto-save if name is provided alongside steps
    save_as = str(args.get("save_as", "")).strip()
    if save_as:
        _save_workflow(
            save_as, steps, mode,
            description=str(args.get("description", "")).strip(),
        )

    if mode == "parallel":
        result = await pipeline.execute_parallel(steps, origin="pipeline_parallel")
    else:
        result = await pipeline.execute(steps, origin="pipeline")

    output = result.to_dict()
    if save_as:
        output["saved_as"] = save_as
    return json.dumps(output, indent=2, default=str)
