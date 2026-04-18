#!/usr/bin/env python3
"""
Dev runner: starts Orchestrator, KN, WIL, DAF, and CSC in one terminal.
EGM is optional and disabled by default.

Features:
- Spawns each service as a subprocess with its own log under ./logs.
- Optionally enforces approvals (for WIL risky ops) and sets ORCH_URL automatically.
- Passes through your OpenRouter key and optional model name.

Usage:
  python scripts/dev_up.py --approvals --port 8000 \
    --openrouter-key "$OPENROUTER_API_KEY" --openrouter-model "deepseek/deepseek-chat-v3.1:free"
  python scripts/dev_up.py --with-egm

Stop with Ctrl-C; the script will terminate child processes gracefully.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def start(name: str, cmd: list[str], env: dict[str, str], logdir: Path) -> subprocess.Popen:
    logdir.mkdir(parents=True, exist_ok=True)
    log = (logdir / f"{name}.log").open("w", encoding="utf-8")
    print(f"[+] {name:14s} -> {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000, help='Orchestrator port')
    ap.add_argument('--approvals', action='store_true', help='Require approvals for risky WIL ops')
    ap.add_argument('--with-egm', action='store_true', help='Start EGM service and enforce EGM checks')
    ap.add_argument('--reload', action='store_true', help='Enable uvicorn auto-reload for orchestrator')
    ap.add_argument('--openrouter-key', type=str, default=os.getenv('OPENROUTER_API_KEY', ''), help='OpenRouter API key')
    ap.add_argument('--openrouter-model', type=str, default=os.getenv('OPENROUTER_MODEL', ''), help='OpenRouter model id (e.g., deepseek/deepseek-chat-v3.1:free)')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    logdir = repo / 'logs'

    base_env = os.environ.copy()
    # Ensure src/ is on PYTHONPATH so bare imports like 'from common.protos import ...' work
    src_dir = str(repo / 'src')
    existing_pp = base_env.get('PYTHONPATH', '')
    base_env['PYTHONPATH'] = f"{src_dir}:{existing_pp}" if existing_pp else src_dir
    # LLM keys: pass through OpenRouter; mirror to OPENAI_API_KEY if needed by client
    if args.openrouter_key:
        base_env['OPENROUTER_API_KEY'] = args.openrouter_key
        base_env.setdefault('OPENAI_API_KEY', args.openrouter_key)
    if args.openrouter_model:
        base_env['OPENROUTER_MODEL'] = args.openrouter_model
    # Default service addresses for local stack.
    base_env.setdefault('KN_ADDRESS', 'localhost:50051')
    base_env.setdefault('CSC_ADDRESS', 'localhost:50052')
    base_env.setdefault('EGM_ADDRESS', 'localhost:50053')
    base_env.setdefault('WIL_ADDRESS', 'localhost:50054')
    base_env.setdefault('DAF_ADDRESS', 'localhost:50055')
    # EGM behavior: off by default unless explicitly enabled.
    if args.with_egm:
        base_env.setdefault('EGM_MODE', 'strict')
    else:
        base_env['EGM_MODE'] = 'off'

    # Orchestrator
    orch_env = base_env.copy()
    # Default to loopback for safety. Pass --public to bind all interfaces.
    _host = '0.0.0.0' if getattr(args, 'public', False) else '127.0.0.1'
    orch_cmd = [sys.executable, '-m', 'uvicorn', 'src.orchestrator.server:app', '--host', _host, '--port', str(args.port)]
    if args.reload:
        orch_cmd.append('--reload')

    # KN
    kn_env = base_env.copy()
    kn_cmd = [sys.executable, '-m', 'src.knowledge_nexus.service']

    # EGM (optional)
    egm_env = base_env.copy()
    egm_cmd = [sys.executable, '-m', 'src.ethical_governance_module.service']

    # WIL
    wil_env = base_env.copy()
    if args.approvals:
        wil_env['APPROVALS_REQUIRED'] = '1'
        wil_env['ORCH_URL'] = f"http://localhost:{args.port}"
    wil_cmd = [sys.executable, '-m', 'src.world_interaction_layer.service']

    # DAF
    daf_env = base_env.copy()
    daf_cmd = [sys.executable, '-m', 'src.dynamic_agent_fabric.service']

    # CSC
    csc_env = base_env.copy()
    csc_cmd = [sys.executable, '-m', 'src.core_strategic_engine.service']

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        procs.append(('orchestrator', start('orchestrator', orch_cmd, orch_env, logdir)))
        procs.append(('kn',           start('kn',           kn_cmd,   kn_env,   logdir)))
        if args.with_egm:
            procs.append(('egm',      start('egm',          egm_cmd,  egm_env,  logdir)))
        procs.append(('wil',          start('wil',          wil_cmd,  wil_env,  logdir)))
        procs.append(('daf',          start('daf',          daf_cmd,  daf_env,  logdir)))
        procs.append(('csc',          start('csc',          csc_cmd,  csc_env,  logdir)))

        stack = "orchestrator, kn, wil, daf, csc" + (", egm" if args.with_egm else "")
        print(f"\n[mode] Stack: {stack}")
        print(f"[mode] EGM_MODE={base_env.get('EGM_MODE', 'off')}")

        print("\n[info] Services starting. Tails (first 10 lines each):\n")
        for name, _ in procs:
            try:
                path = logdir / f"{name}.log"
                if path.exists():
                    print(f"---- {name}.log ----")
                    print("".join(path.open("r", encoding='utf-8').readlines()[:10]))
            except Exception:
                pass

        print("\n[ready] Orchestrator: http://localhost:%d" % args.port)
        print("[tip] Open a new terminal and run: prometheus chat")
        print("[tip] Or open VS Code → F5 in vscode_extension → Prometheus: Open Chat")
        print("[tip] Stop all with Ctrl-C here; logs in ./logs")

        # Wait until any process exits, then stop all
        while True:
            for name, p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"\n[warn] {name} exited with code {ret}; shutting down others...")
                    raise KeyboardInterrupt
            import time; time.sleep(1)
    except KeyboardInterrupt:
        for name, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for name, p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[done] All services stopped")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
