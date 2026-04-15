#!/usr/bin/env python3
"""
Prometheus CLI: plan, patch (stub), and test flows.

Examples:
  python scripts/prometheus_cli.py plan --goal "Build a Flask app with /health"
  python scripts/prometheus_cli.py test --cmd "pytest -q"
  python scripts/prometheus_cli.py patch --file diff.json
"""
import argparse
import asyncio
import json
import os
from uuid import uuid4


async def cmd_plan(goal: str, mcts: bool):
    import grpc
    from common.protos import knowledge_nexus_pb2_grpc

    from src.core_strategic_engine.planner import HierarchicalStrategicPlannerV1
    from src.core_strategic_engine.planner_mcts import ABMCTSPlanner
    # Try KN channel
    try:
        kn_addr = os.getenv('KN_ADDRESS', 'localhost:50051')
        channel = grpc.aio.insecure_channel(kn_addr)
        kn_stub = knowledge_nexus_pb2_grpc.KnowledgeNexusServiceStub(channel)
    except Exception:
        kn_stub = None
    planner = ABMCTSPlanner(kn_stub=kn_stub) if mcts else HierarchicalStrategicPlannerV1(kn_stub=kn_stub)
    plan = await planner.create_plan(uuid4(), goal, {})
    print(json.dumps(plan.model_dump() if hasattr(plan, 'model_dump') else str(plan), indent=2, default=str))

def cmd_test(cmd: str):
    import shlex
    import subprocess
    try:
        print(f"[+] Running tests: {cmd}")
        res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise SystemExit(res.returncode)
    except Exception as e:
        print(f"[x] Test execution failed: {e}")
        raise SystemExit(1)

def cmd_patch(file: str):
    # Stub: read JSON patches and print; in future, apply to repo and run tests/PR
    try:
        data = json.loads(open(file, encoding='utf-8').read())
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[x] Failed to read patch file: {e}")
        raise SystemExit(1)

async def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_plan = sub.add_parser('plan'); p_plan.add_argument('--goal', required=True); p_plan.add_argument('--mcts', action='store_true')
    p_test = sub.add_parser('test'); p_test.add_argument('--command', required=True)
    p_patch = sub.add_parser('patch'); p_patch.add_argument('--file', required=True)
    args = ap.parse_args()
    if args.cmd == 'plan':
        await cmd_plan(args.goal, args.mcts)
    elif args.cmd == 'test':
        cmd_test(args.command)
    elif args.cmd == 'patch':
        cmd_patch(args.file)

if __name__ == '__main__':
    asyncio.run(main())
