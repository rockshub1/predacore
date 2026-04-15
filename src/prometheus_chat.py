#!/usr/bin/env python3
"""
Prometheus TUI (minimal)

Connects to the local Orchestrator WebSocket (ws://localhost:8000/ws), lets you
send goals, and streams events (plan_status, citations, approvals, diffs/tests/screenshots).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any


async def _readline(prompt: str = "goal> ") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


def _fmt(evt: dict[str, Any]) -> str:
    t = evt.get("type")
    if t == "assistant":
        return f"[assistant] {evt.get('content', '')}"
    if t == "plan_status":
        st = evt.get("status")
        steps = evt.get("steps") or []
        lines = [
            f"[plan_status] goal={evt.get('goal_id')} status={st} steps={len(steps)}"
        ]
        for s in steps[:10]:
            lines.append(
                "  - "
                f"{str(s.get('id', ''))[:8]} {s.get('action_type', '')} "
                f"{s.get('status', '')} :: {str(s.get('description', ''))[:80]}"
            )
        return "\n".join(lines)
    if t == "citations":
        cits = evt.get("citations") or []
        lines = [f"[citations] step={evt.get('step_id')} count={len(cits)}"]
        for c in cits[:5]:
            lines.append(
                "  - "
                f"{c.get('parent_id') or c.get('id')} "
                f"(score={c.get('score')}) :: {str(c.get('preview', ''))[:80]}"
            )
        return "\n".join(lines)
    if t == "approval_request":
        return (
            f"[approval] step={evt.get('step_id')} tool={evt.get('tool_id')} :: "
            f"{str(evt.get('description', ''))[:80]}"
        )
    if t in (
        "approved",
        "diff",
        "test_result",
        "screenshot",
        "plan",
        "plan_start",
        "exec",
        "hello",
        "done",
        "error",
    ):
        return f"[{t}] {json.dumps({k: v for k, v in evt.items() if k != 'type'})}"
    return json.dumps(evt)


async def _main() -> None:
    try:
        import websockets  # type: ignore
    except ImportError:
        print(
            "[x] Missing dependency: websockets. Install with 'pip install websockets rich'.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        from rich.console import Console
    except ImportError:

        class Console:
            def print(self, *a, **k):
                print(*a)

    console = Console()

    url = os.getenv("PROM_ORCH_WS", "ws://localhost:8000/ws")
    try:
        async with websockets.connect(url, max_size=2**24) as ws:
            console.print(f"[bold green]Connected[/] to {url}")

            async def producer() -> None:
                while True:
                    try:
                        line = await _readline()
                    except EOFError:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower().startswith("simulate:"):
                        goal = line.split(":", 1)[1].strip()
                        await ws.send(json.dumps({"type": "simulate", "goal": goal}))
                    elif line.lower().startswith("approve "):
                        step_id = line.split(" ", 1)[1].strip()
                        await ws.send(
                            json.dumps({"type": "approve", "step_id": step_id})
                        )
                    else:
                        goal = line
                        await ws.send(json.dumps({"type": "goal", "goal": goal}))

            async def consumer() -> None:
                while True:
                    raw = await ws.recv()
                    try:
                        evt = json.loads(raw)
                    except Exception:
                        console.print(raw)
                        continue
                    console.print(_fmt(evt))

            await asyncio.gather(producer(), consumer())
    except Exception as e:
        console.print(f"[red][x][/red] WebSocket error: {e}")
        raise SystemExit(1)


def main() -> int:
    try:
        asyncio.run(_main())
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
