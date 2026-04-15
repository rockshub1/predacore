# Evaluation Harness

Prometheus provides a lightweight evaluation harness to benchmark planning quality, tool reliability, and latency. It supports three suites:

- planning: text planning goals (HTN or AB‑MCTS via `--mcts`).
- sudoku: deterministic solver as a reliability/latency probe.
- ale: placeholder mini‑tasks that emulate ALE‑bench goals offline.

## Quick Start

- Planning (AB‑MCTS):
  `USE_MCTS_PLANNER=1 python scripts/eval_harness.py --suite planning --count 10 --mcts`

- Sudoku with metrics:
  `python scripts/eval_harness.py --suite sudoku --count 20 --metrics-port 8020`

- ALE placeholder:
  `python scripts/eval_harness.py --suite ale --count 3 --metrics-port 8021`

## Nightly Run

Use `scripts/nightly_eval.sh` to run all suites and capture logs under `logs/`.

## Metrics

The harness can expose Prometheus metrics (if `prometheus_client` is installed):
- Sudoku: `eval_sudoku_ok_total`, `eval_sudoku_latency_seconds`.
- Use `--metrics-port` to enable an HTTP endpoint.

## Notes

- The ALE suite here is offline; it exercises the planner but does not make network calls. Integrate real ALE‑bench tasks by wiring WIL tools and mocks.
- Ensure environment variables for planner (e.g., `MCTS_BRANCHES`, `MCTS_MAX_DEPTH`, `MCTS_BUDGET`) are tuned for your latency/quality targets.

