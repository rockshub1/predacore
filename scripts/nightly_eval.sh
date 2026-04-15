#!/usr/bin/env bash
set -euo pipefail

# Nightly evaluation runner for Prometheus
# - Runs planning (AB-MCTS), Sudoku, and ALE placeholder suites
# - Exposes Prometheus metrics on dedicated ports

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT="$LOG_DIR/eval_${STAMP}.log"

export USE_MCTS_PLANNER=1

echo "[info] Starting nightly eval at $STAMP" | tee -a "$OUT"

echo "[planning] Running 10 goals with AB-MCTS" | tee -a "$OUT"
python "$ROOT_DIR/scripts/eval_harness.py" --suite planning --count 10 --mcts --metrics-port 8025 | tee -a "$OUT" || true

echo "[sudoku] Running 20 puzzles" | tee -a "$OUT"
python "$ROOT_DIR/scripts/eval_harness.py" --suite sudoku --count 20 --metrics-port 8026 | tee -a "$OUT" || true

echo "[ale] Running 5 tasks (placeholder offline)" | tee -a "$OUT"
python "$ROOT_DIR/scripts/eval_harness.py" --suite ale --count 5 --metrics-port 8027 | tee -a "$OUT" || true

echo "[ale_mini] Running 2 mock tasks" | tee -a "$OUT"
python "$ROOT_DIR/scripts/eval_harness.py" --suite ale_mini --count 2 --metrics-port 8028 | tee -a "$OUT" || true

echo "[ale_mini_run] Simulated execution of 2 goals" | tee -a "$OUT"
python "$ROOT_DIR/scripts/eval_ale_mini_run.py" --all | tee -a "$OUT" || true

echo "[done] Results saved to $OUT" | tee -a "$OUT"
