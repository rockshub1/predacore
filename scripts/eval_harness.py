#!/usr/bin/env python3
"""
Evaluation Harness

Modes:
- planning: run text planning tasks using HTN or AB-MCTS.
- sudoku: run a small Sudoku mini-suite to benchmark search reliability.

Usage:
  python scripts/eval_harness.py --suite planning --count 5 --mcts
  python scripts/eval_harness.py --suite sudoku --count 10 --metrics-port 8020
  python scripts/eval_harness.py --suite ale --count 3 --metrics-port 8021
"""
import argparse
import asyncio
import time
from uuid import uuid4

try:
    from prometheus_client import Counter, Histogram, start_http_server
except Exception:  # Optional metrics
    Counter = Histogram = start_http_server = None  # type: ignore


async def eval_once(goal: str, use_mcts: bool) -> dict:
    from src.core_strategic_engine.planner import HierarchicalStrategicPlannerV1
    from src.core_strategic_engine.planner_mcts import ABMCTSPlanner
    planner = ABMCTSPlanner(kn_stub=None) if use_mcts else HierarchicalStrategicPlannerV1(kn_stub=None)
    t0 = time.time()
    plan = await planner.create_plan(uuid4(), goal, {})
    dt = time.time() - t0
    ok = plan is not None and len(getattr(plan, 'steps', [])) > 0
    return {"ok": ok, "latency": dt, "steps": len(getattr(plan, 'steps', [])), "justification": getattr(plan, 'justification', '')}


# --- Sudoku mini-suite ---
SudokuBoard = list[list[int]]

def _sudoku_parse(p: str) -> SudokuBoard:
    s = ''.join(ch for ch in p if ch.isdigit())
    if len(s) != 81:
        raise ValueError("sudoku string must have 81 digits (0 for empty)")
    grid = [[int(s[r*9+c]) for c in range(9)] for r in range(9)]
    return grid

def _sudoku_find_empty(grid: SudokuBoard) -> tuple[int, int] | None:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None

def _sudoku_valid(grid: SudokuBoard, r: int, c: int, v: int) -> bool:
    row_ok = all(grid[r][cc] != v for cc in range(9))
    col_ok = all(grid[rr][c] != v for rr in range(9))
    br, bc = (r//3)*3, (c//3)*3
    box_ok = all(grid[br+dr][bc+dc] != v for dr in range(3) for dc in range(3))
    return row_ok and col_ok and box_ok

def _sudoku_solve(grid: SudokuBoard) -> bool:
    empty = _sudoku_find_empty(grid)
    if not empty:
        return True
    r, c = empty
    for v in range(1, 10):
        if _sudoku_valid(grid, r, c, v):
            grid[r][c] = v
            if _sudoku_solve(grid):
                return True
            grid[r][c] = 0
    return False

def run_sudoku_suite(count: int, enable_metrics: bool = False):
    puzzles = [
        # Easy
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        # Medium
        "009000000080605020501078000000000700706040102004000000000720803090301040000000600",
        # Hard
        "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
        # Evil
        "000900002050000060000006809006800000200070005000009100407100000060000080800002000",
    ]
    # Repeat to match count
    puzzles = (puzzles * ((count + len(puzzles) - 1) // len(puzzles)))[:count]

    m_ok = Counter('eval_sudoku_ok_total', 'Sudoku solves that succeeded') if enable_metrics and Counter else None
    m_lat = Histogram('eval_sudoku_latency_seconds', 'Sudoku solve latency') if enable_metrics and Histogram else None

    oks = 0
    lats: list[float] = []
    for i, p in enumerate(puzzles, 1):
        grid = _sudoku_parse(p)
        t0 = time.time()
        ok = _sudoku_solve(grid)
        dt = time.time() - t0
        oks += 1 if ok else 0
        lats.append(dt)
        if m_ok:
            m_ok.inc(1 if ok else 0)
        if m_lat:
            m_lat.observe(dt)
        print(f"- puzzle#{i:02d} | ok={ok} | t={dt:.3f}s")
    avg = sum(lats)/len(lats) if lats else 0.0
    print(f"\nSudoku Summary: ok={oks}/{len(puzzles)} | avg_latency={avg:.3f}s")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['planning', 'sudoku', 'ale', 'ale_mini'], default='planning')
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--mcts', action='store_true')
    ap.add_argument('--metrics-port', type=int, default=0)
    args = ap.parse_args()

    # Optional metrics server
    if args.metrics_port and start_http_server:
        try:
            start_http_server(args.metrics_port)
            print(f"Prometheus metrics on :{args.metrics_port}")
        except Exception as e:
            print(f"Metrics server failed to start: {e}")

    if args.suite == 'sudoku':
        run_sudoku_suite(args.count, enable_metrics=bool(args.metrics_port))
        return

    if args.suite == 'ale':
        # Minimal placeholder: emulate ALE mini-tasks planning without network.
        # We construct a few goals that would typically drive browser/API/tools.
        goals = [
            "Open example.com and extract the H1 text, then summarize",
            "Fetch EDINET demo page, embed into RAG, answer a question with citations",
            "Scrape pagination from a docs page (2 pages) and export JSON",
        ]
        goals = goals[:args.count]
        results = []
        for g in goals:
            res = await eval_once(g, use_mcts=True)
            results.append(res)
            print(f"- [ALE] {g[:40]:40s} | ok={res['ok']} | t={res['latency']:.2f}s | steps={res['steps']}")
        oks = sum(1 for r in results if r['ok'])
        avg = sum(r['latency'] for r in results) / len(results)
        print(f"\nALE Summary: ok={oks}/{len(results)} | avg_latency={avg:.2f}s (tool execution not invoked in offline mode)")
        return

    if args.suite == 'ale_mini':
        # Deterministic CI-friendly validators for plan structure
        def validate_plan_structure(plan, goal: str) -> bool:
            try:
                steps = getattr(plan, 'steps', [])
                s_json = [dict(action_type=getattr(s, 'action_type', ''), parameters=getattr(s, 'parameters', {})) for s in steps]
                # Expect at least one INVOKE_TOOL with selector_extract or edinet_fetch depending on goal
                has_selector = any(st['action_type'] == 'INVOKE_TOOL' and st['parameters'].get('tool_id') == 'selector_extract' for st in s_json)
                has_edinet = any(st['action_type'] == 'INVOKE_TOOL' and st['parameters'].get('tool_id') == 'edinet_fetch' for st in s_json)
                has_rag = any(st['action_type'] == 'INVOKE_TOOL' and st['parameters'].get('tool_id') in ('rag_embed', 'rag_answer') for st in s_json)
                ok = (has_selector or has_edinet) and has_rag
                # If goal mentions export, ensure export_format is present
                if 'export' in (goal or '').lower() and has_selector:
                    has_export = any(st['parameters'].get('tool_id') == 'selector_extract' and 'export_format' in (st['parameters'] or {}) for st in s_json)
                    ok = ok and has_export
                return ok
            except Exception:
                return False

        tasks = [
            ("Open example.com and extract H1 then answer with RAG", True),
            ("Fetch EDINET filing and summarize key metrics with citations", True),
            ("Extract footer links across 2 pages and export CSV", True),
        ]
        oks = 0
        lats = []
        for goal, require in tasks[:args.count]:
            from uuid import uuid4

            from src.core_strategic_engine.planner_mcts import ABMCTSPlanner
            planner = ABMCTSPlanner(kn_stub=None)
            t0 = time.time()
            plan = await planner.create_plan(uuid4(), goal, {})
            dt = time.time() - t0
            lats.append(dt)
            ok = plan is not None and validate_plan_structure(plan, goal)
            oks += 1 if ok else 0
            print(f"- [ALE_MINI] {goal[:40]:40s} | ok={ok} | t={dt:.2f}s | steps={len(getattr(plan, 'steps', []))}")
        avg = sum(lats)/len(lats) if lats else 0.0
        print(f"\nALE_MINI Summary: ok={oks}/{len(tasks[:args.count])} | avg_latency={avg:.2f}s")
        return

    goals = [
        "Build a Flask app with /health",
        "Scrape the footer links from https://example.com across 3 pages",
        "Ingest this policy text into memory and summarize with citations",
        "Create a CLI tool to convert CSV to JSON",
        "Refactor code to use Postgres and add migrations",
    ]
    goals = goals * ((args.count + len(goals) - 1) // len(goals))
    goals = goals[:args.count]
    results = []
    for g in goals:
        res = await eval_once(g, args.mcts)
        results.append(res)
        print(f"- {g[:40]:40s} | ok={res['ok']} | t={res['latency']:.2f}s | steps={res['steps']}")
    oks = sum(1 for r in results if r['ok'])
    avg = sum(r['latency'] for r in results) / len(results)
    print(f"\nSummary: ok={oks}/{len(results)} | avg_latency={avg:.2f}s")


if __name__ == '__main__':
    asyncio.run(main())
