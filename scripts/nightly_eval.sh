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

# Aggregate failures explicitly so the nightly fails when any suite breaks,
# instead of the previous `|| true` swallow that hid every failure.
fail_count=0
run_suite() {
    local label="$1"; shift
    echo "[$label] Running" | tee -a "$OUT"
    if ! "$@" 2>&1 | tee -a "$OUT"; then
        echo "[$label] FAILED" | tee -a "$OUT"
        fail_count=$((fail_count + 1))
    fi
}

run_suite planning python "$ROOT_DIR/scripts/eval_harness.py" --suite planning --count 10 --mcts --metrics-port 8025
run_suite sudoku   python "$ROOT_DIR/scripts/eval_harness.py" --suite sudoku --count 20 --metrics-port 8026
run_suite ale      python "$ROOT_DIR/scripts/eval_harness.py" --suite ale --count 5 --metrics-port 8027
run_suite ale_mini python "$ROOT_DIR/scripts/eval_harness.py" --suite ale_mini --count 2 --metrics-port 8028

if [ "$fail_count" -gt 0 ]; then
    echo "[done] $fail_count suite(s) FAILED. Results: $OUT" | tee -a "$OUT"
    exit 1
fi
echo "[done] All suites passed. Results: $OUT" | tee -a "$OUT"
