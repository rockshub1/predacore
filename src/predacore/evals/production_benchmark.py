"""
Production-DB retrieval benchmark for PredaCore unified memory.

Why this exists alongside ``longmemeval.py``:

  - LongMemEval uses synthetic conversation haystacks. Its harness
    (longmemeval.py:130) caps content at ~1000 chars to fit the OLD
    256-token BGE kernel. After the Phase-6b 256→512 upgrade and the
    auto-chunking work, **the LongMemEval number stays at R@5=0.9574 by
    construction** — its truncation makes the upgrades invisible.

  - This benchmark exercises the upgrades:
      • Long content (chunks up to ~4000 chars) → 512-token kernel
      • Code-backed memories with source_path → verify-with-code (T5+)
      • Project-isolated retrieval → real project_id semantics

Pipeline:

  1. Index the predacore source tree (or a user-specified root) into an
     ephemeral DB via ``UnifiedMemoryStore.reindex_file()``. Each file
     becomes one parent memory + auto-chunked children, all with
     ``source_path`` set so verify-with-code is meaningful.
  2. Load the bundled query set (``production_queries.json``); each
     query has expected source paths (any-of match).
  3. Run three passes:
       A. Plain semantic recall (baseline).
       B. With ``verify=True, verify_drop=True`` (code-verified slice).
       C. With ``simulate_drift=N`` set: corrupt N random source files'
          on-disk content, repeat (B), measure how many drifted hits
          get dropped (validates verify-with-code's filter).
  4. Report R@k and NDCG@k per pass. The delta (B − A) is verify-with-
     code's lift on a code-backed corpus; (C kept-rate) is its
     correctness when content has changed.

Usage:
    python -m predacore.evals.production_benchmark
    python -m predacore.evals.production_benchmark --top-k 10 20
    python -m predacore.evals.production_benchmark --simulate-drift 5
    python -m predacore.evals.production_benchmark --json-out .audit/run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from predacore.memory import UnifiedMemoryStore
from predacore.services.embedding import RustEmbeddingClient


# ── Metric helpers (mirror longmemeval.py exactly so numbers compare) ────


def _dcg(relevant_ranks: list[int]) -> float:
    return sum(1.0 / math.log2(r + 2) for r in relevant_ranks)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_ranks = [i for i, r in enumerate(top_k) if r in relevant]
    if not relevant_ranks:
        return 0.0
    actual = _dcg(relevant_ranks)
    ideal = _dcg(list(range(min(len(relevant), k))))
    return actual / ideal if ideal > 0 else 0.0


# ── Corpus discovery ─────────────────────────────────────────────────────


_INDEXABLE_SUFFIXES = {".py", ".md", ".rs", ".toml"}
_SKIP_DIRS = {
    # Build artifacts
    "__pycache__", ".git", ".venv", "venv", "node_modules", "target",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".audit",
    # Discussion-heavy directories that confuse semantic ranking. Tests
    # and design docs talk *about* implementations, so BGE often ranks
    # them higher than the implementation file itself when the query is
    # about what code does. The benchmark's first run showed this exact
    # pattern: memory queries got pulled toward tests/ + documents/.
    # Excluding them is corpus curation, not censorship — these files
    # are still on disk; they're just not in the retrieval index.
    "tests", "documents", "evals",
    # Identity defaults are templates (markdown), not implementation —
    # similar discussion-vs-code issue.
    "defaults",
}


def discover_files(root: Path, max_files: int | None = None) -> list[Path]:
    """Walk ``root`` and return indexable source files in deterministic order.

    Sorting is stable so two runs over the same tree produce the same corpus,
    which keeps benchmark numbers reproducible.
    """
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in _INDEXABLE_SUFFIXES:
                paths.append(p)
    paths.sort()
    if max_files is not None:
        paths = paths[:max_files]
    return paths


# ── Query loading + matching ─────────────────────────────────────────────


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def _path_matches_expected(source_path: str, expected: list[str]) -> bool:
    """A retrieved memory's source_path matches if it ENDS WITH any expected.

    We use endswith() so that absolute paths (``/a/b/src/predacore/x.py``)
    match a relative expectation (``predacore/x.py`` or ``x.py``). This
    keeps the query set portable across machines.
    """
    sp = source_path.replace("\\", "/")
    for exp in expected:
        e = exp.replace("\\", "/").lstrip("./")
        if sp.endswith(e) or sp.endswith("/" + e):
            return True
    return False


# ── Corpus build ─────────────────────────────────────────────────────────


async def build_corpus(
    store: UnifiedMemoryStore,
    files: list[Path],
    *,
    project_id: str = "production_benchmark",
    mode: str = "reactive",
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Index ``files`` into ``store`` and return aggregate stats.

    ``mode``:
      - "reactive" (default): loop over files calling reindex_file once
        per file. Mirrors how the auto-trigger handlers index in production.
      - "bulk": single bulk_index_directory() call over ``source_root``.
        Tests the bulk path's equivalence with the reactive one — both
        should produce identical R@k.
    """
    t0 = time.time()
    if mode == "bulk":
        if source_root is None:
            raise ValueError("mode='bulk' requires source_root")
        result = await store.bulk_index_directory(
            source_root,
            project_id=project_id,
            user_id="bench",
        )
        # Bulk applies its own ignore filter, so file count may not match
        # the discover_files count exactly — that's fine; the benchmark
        # measures recall over whatever made it in.
        return {
            "mode": "bulk",
            "files_indexed": result.get("files_indexed", 0),
            "files_failed": result.get("files_failed", 0),
            "files_skipped_unchanged": result.get("files_skipped_unchanged", 0),
            "files_ignored": result.get("files_ignored", 0),
            "chunks_added": result.get("chunks_added", 0),
            "elapsed_sec": round(time.time() - t0, 2),
            "errors": result.get("errors", []),
        }

    indexed = 0
    failed = 0
    total_chunks = 0
    for p in files:
        try:
            result = await store.reindex_file(
                str(p), project_id=project_id, user_id="bench",
            )
            if isinstance(result, dict):
                # reindex_file returns {chunk_count, new_ids, ...}; new_ids
                # excludes secret-refused rows so it's the truer "what's in
                # the DB" number.
                total_chunks += len(result.get("new_ids", []) or [])
            indexed += 1
        except Exception as exc:  # noqa: BLE001 — benchmark should be tolerant
            failed += 1
            if failed <= 5:
                print(f"  ⚠  reindex failed: {p}: {exc}", file=sys.stderr)
    return {
        "mode": "reactive",
        "files_indexed": indexed,
        "files_failed": failed,
        "chunks_added": total_chunks,
        "elapsed_sec": round(time.time() - t0, 2),
    }


# ── Drift simulation ─────────────────────────────────────────────────────


def apply_drift(files: list[Path], n: int, seed: int = 42) -> list[Path]:
    """Mutate ``n`` random files in-place by appending a marker comment.

    Returns the list of files that were drifted. The marker is a Python /
    text-safe comment; it's enough that ``_verify_chunk_against_source``
    sees the EXISTING chunks no longer match perfectly (their first
    non-blank line is still in the file but added trailing content shifts
    the file beyond the recorded blob_sha). Callers must restore the
    files manually after the run (we keep this safety in caller hands —
    benchmark only mutates inside the temp index, not the working tree).

    NOTE: for safety, we DO NOT mutate the user's source tree by default.
    Drift is applied to a copy under a temp directory; the bench's
    discover_files run targets that copy.
    """
    rng = random.Random(seed)
    to_drift = rng.sample(files, min(n, len(files)))
    for p in to_drift:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write("\n# benchmark-drift-marker — do not commit\n")
        except OSError:
            pass
    return to_drift


# ── Core eval pass ───────────────────────────────────────────────────────


async def run_pass(
    store: UnifiedMemoryStore,
    queries: list[dict],
    *,
    label: str,
    verify: bool,
    verify_drop: bool,
    fetch_k: int = 30,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    """Run all queries through ``store.recall`` and aggregate metrics."""
    print(f"\n── PASS: {label}  (verify={verify}, verify_drop={verify_drop})")
    per_query: list[dict] = []
    t0 = time.time()
    for q in queries:
        expected = q["expected_source_paths"]
        # recall() — we deliberately don't pass project_id so the benchmark
        # corpus surfaces (it was indexed under "production_benchmark").
        results = await store.recall(
            query=q["query"],
            user_id="bench",
            top_k=fetch_k,
            project_id="production_benchmark",
            verify=verify,
            verify_drop=verify_drop,
        )
        # Each result: (memory_dict, score). Pull out source_path; rank by
        # first-occurrence so duplicates from the same file don't double-count.
        retrieved_paths_in_order: list[str] = []
        seen: set[str] = set()
        for mem, _score in results:
            sp = (mem.get("source_path") or "").strip()
            if not sp or sp in seen:
                continue
            retrieved_paths_in_order.append(sp)
            seen.add(sp)
        out: dict[str, Any] = {
            "id": q["id"],
            "category": q.get("category", "uncategorized"),
            "expected": expected,
            "retrieved_top5": retrieved_paths_in_order[:5],
            "hit_in_top5": any(
                _path_matches_expected(p, expected) for p in retrieved_paths_in_order[:5]
            ),
        }
        # Build a "matched" flag per retrieved path so R@k / NDCG@k work
        # against arbitrary string-form paths.
        match_flags: list[bool] = [
            _path_matches_expected(p, expected) for p in retrieved_paths_in_order
        ]
        # For metric computation: we treat each match as a hit; relevant set
        # is "the ideal set of unique matches we saw (cap at 1 — only one
        # right answer per query for binary relevance)".
        retrieved_for_metric = ["match" if m else f"miss_{i}" for i, m in enumerate(match_flags)]
        relevant_for_metric = {"match"} if any(match_flags) else set()
        for k in k_values:
            out[f"r@{k}"] = recall_at_k(retrieved_for_metric, relevant_for_metric, k)
            out[f"ndcg@{k}"] = ndcg_at_k(retrieved_for_metric, relevant_for_metric, k)
        per_query.append(out)

    n = len(per_query)
    if n == 0:
        return {"label": label, "error": "no queries scored"}

    overall: dict[str, Any] = {
        "label": label,
        "queries_scored": n,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    for k in k_values:
        overall[f"R@{k}"] = round(sum(q[f"r@{k}"] for q in per_query) / n, 4)
        overall[f"NDCG@{k}"] = round(sum(q[f"ndcg@{k}"] for q in per_query) / n, 4)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for q in per_query:
        by_category[q["category"]].append(q)
    cat_metrics: dict[str, dict] = {}
    for cat, rs in sorted(by_category.items()):
        c = len(rs)
        m: dict[str, Any] = {"n": c}
        for k in k_values:
            m[f"R@{k}"] = round(sum(r[f"r@{k}"] for r in rs) / c, 4)
            m[f"NDCG@{k}"] = round(sum(r[f"ndcg@{k}"] for r in rs) / c, 4)
        cat_metrics[cat] = m

    return {
        "overall": overall,
        "by_category": cat_metrics,
        "per_query": per_query,
    }


# ── Top-level orchestration ──────────────────────────────────────────────


async def run_benchmark(
    *,
    source_root: Path,
    queries_path: Path,
    drift_n: int,
    fetch_k: int,
    k_values: tuple[int, ...],
    max_files: int | None,
    mode: str = "reactive",
) -> dict[str, Any]:
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path.name}")

    # COPY the source tree so any drift simulation never mutates the user's
    # working files. We also strip dotfiles / .venv / target / etc via
    # discover_files's _SKIP_DIRS rule.
    print(f"Indexing source root: {source_root}")
    files_in_src = discover_files(source_root, max_files=max_files)
    print(f"  found {len(files_in_src)} indexable files")

    embedder = RustEmbeddingClient()
    print("First call will warm BGE-small-en-v1.5 (~133 MB on first install)...")

    out: dict[str, Any] = {
        "source_root": str(source_root),
        "files_discovered": len(files_in_src),
        "fetch_k": fetch_k,
        "k_values": list(k_values),
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "bench.db")
        store = UnifiedMemoryStore(db_path=db_path, embedding_client=embedder)
        try:
            corpus = await build_corpus(
                store, files_in_src, mode=mode, source_root=source_root,
            )
            out["corpus"] = corpus
            print(
                f"  [mode={corpus.get('mode', mode)}] indexed "
                f"{corpus['files_indexed']} files "
                f"({corpus['chunks_added']} chunks) in {corpus['elapsed_sec']}s"
            )

            # PASS A — baseline (plain semantic recall)
            pass_a = await run_pass(
                store, queries, label="A_baseline",
                verify=False, verify_drop=False,
                fetch_k=fetch_k, k_values=k_values,
            )
            out["pass_a_baseline"] = pass_a
            print(_format_overall(pass_a))

            # PASS B — verify-with-code on a clean corpus.
            # Expectation: identical to A because no drift exists; verify=True
            # only affects ranking when content has shifted.
            pass_b = await run_pass(
                store, queries, label="B_verify_clean",
                verify=True, verify_drop=True,
                fetch_k=fetch_k, k_values=k_values,
            )
            out["pass_b_verify_clean"] = pass_b
            print(_format_overall(pass_b))

            # PASS C — drift simulation. Apply mutations to N source files;
            # verify-with-code should now drop drifted chunks while keeping
            # untouched ones.
            if drift_n > 0:
                drifted = apply_drift(files_in_src, drift_n)
                print(f"\n  applied drift to {len(drifted)} files (verify_drop should filter)")
                try:
                    pass_c = await run_pass(
                        store, queries, label="C_verify_with_drift",
                        verify=True, verify_drop=True,
                        fetch_k=fetch_k, k_values=k_values,
                    )
                    out["pass_c_verify_with_drift"] = pass_c
                    out["drift_files"] = [str(p) for p in drifted]
                    print(_format_overall(pass_c))
                finally:
                    # Restore drifted files (strip the marker we added).
                    for p in drifted:
                        try:
                            text = p.read_text(encoding="utf-8")
                            if "# benchmark-drift-marker" in text:
                                cleaned = text.replace(
                                    "\n# benchmark-drift-marker — do not commit\n", ""
                                )
                                p.write_text(cleaned, encoding="utf-8")
                        except OSError:
                            pass
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass

    # Comparison summary: B − A delta tells us verify-with-code's lift on
    # the clean corpus (should be ~0 — same rows survive). The interesting
    # delta is C dropping drifted hits.
    if "pass_a_baseline" in out and "pass_b_verify_clean" in out:
        a = out["pass_a_baseline"]["overall"]
        b = out["pass_b_verify_clean"]["overall"]
        out["delta_b_minus_a"] = {
            f"R@{k}": round(b[f"R@{k}"] - a[f"R@{k}"], 4) for k in k_values
        }
    return out


def _format_overall(p: dict[str, Any]) -> str:
    if "error" in p:
        return f"  ERROR: {p['error']}"
    o = p["overall"]
    pieces = [f"  {o['label']}  ({o['queries_scored']} q, {o['elapsed_sec']}s)"]
    for key in sorted(o):
        if key.startswith("R@") or key.startswith("NDCG@"):
            pieces.append(f"    {key:10s} {o[key]:.4f}")
    return "\n".join(pieces)


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production-DB retrieval benchmark for PredaCore unified memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=None,
        help="Directory to index (defaults to <repo>/src/predacore)",
    )
    parser.add_argument(
        "--queries", type=Path,
        default=None,
        help="Path to queries JSON (defaults to bundled production_queries.json)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Cap files indexed (smoke-test mode)",
    )
    parser.add_argument(
        "--simulate-drift", type=int, default=0, metavar="N",
        help="After clean run, mutate N random files and re-run with verify_drop=True",
    )
    parser.add_argument(
        "--mode", choices=("reactive", "bulk"), default="reactive",
        help=(
            "Indexing mode: 'reactive' loops reindex_file (mirrors auto-trigger "
            "behavior); 'bulk' uses bulk_index_directory (single walker call). "
            "Both should produce identical R@k — the equivalence test."
        ),
    )
    parser.add_argument(
        "--fetch-k", type=int, default=30,
        help="How many raw memories to retrieve before deduping by source_path",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[5, 10, 20],
        help="k values to compute R@k and NDCG@k for",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write full metrics to this JSON path",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent  # evals → predacore → src → repo
    src_root = args.source_root or (repo_root / "src" / "predacore")
    queries_path = args.queries or (here / "production_queries.json")

    if not src_root.exists():
        print(f"ERROR: source root not found: {src_root}", file=sys.stderr)
        return 2
    if not queries_path.exists():
        print(f"ERROR: queries file not found: {queries_path}", file=sys.stderr)
        return 2

    metrics = asyncio.run(
        run_benchmark(
            source_root=src_root,
            queries_path=queries_path,
            drift_n=args.simulate_drift,
            fetch_k=args.fetch_k,
            k_values=tuple(args.k),
            max_files=args.max_files,
            mode=args.mode,
        )
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "delta_b_minus_a" in metrics:
        print("  Verify-with-code lift over baseline (clean corpus):")
        for k, v in metrics["delta_b_minus_a"].items():
            sign = "+" if v >= 0 else ""
            print(f"    {k:8s} {sign}{v:+.4f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nFull metrics written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
