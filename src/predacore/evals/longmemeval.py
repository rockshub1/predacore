"""
LongMemEval benchmark harness for PredaCore unified memory.

LongMemEval is the industry-standard benchmark for long-term memory in chat
assistants (ICLR 2025, Wu et al., arxiv.org/abs/2410.10813). It evaluates
five core memory abilities: information extraction, multi-session reasoning,
temporal reasoning, knowledge update, and abstention.

This harness:
1. Loads the LongMemEval dataset (500 instances, each with 40+ sessions).
2. For each instance, creates a fresh ephemeral UnifiedMemoryStore, populates
   it with the haystack sessions (each turn becomes a memory, tagged with its
   source session_id), then runs a semantic recall for the question.
3. Deduplicates retrieved memories to unique session_ids (preserving rank),
   and computes Recall@k and NDCG@k against the ground-truth answer sessions.
4. Reports overall and per-category metrics.

Dataset download:
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Usage:
    python -m predacore.evals.longmemeval --dataset ~/datasets/longmemeval_s_cleaned.json
    python -m predacore.evals.longmemeval --dataset ./lme_s.json --max-instances 50
    python -m predacore.evals.longmemeval --dataset ./lme_s.json --top-k 10 --fetch-k 100

Abstention questions (question_id ending in `_abs`) are skipped for retrieval
scoring per the benchmark convention — they test whether the model refuses
to answer when no relevant session exists.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from predacore.memory import UnifiedMemoryStore
from predacore.services.embedding import RustEmbeddingClient


# ── Metric helpers ────────────────────────────────────────────────────────


def _dcg(relevant_ranks: list[int]) -> float:
    """Discounted cumulative gain with binary relevance (rank is 0-indexed)."""
    return sum(1.0 / math.log2(r + 2) for r in relevant_ranks)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary Recall@k: 1.0 if any relevant session appears in top-k, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if any(s in relevant for s in retrieved[:k]) else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@k with binary relevance (session is relevant or not)."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_ranks = [i for i, s in enumerate(top_k) if s in relevant]
    if not relevant_ranks:
        return 0.0
    actual = _dcg(relevant_ranks)
    ideal = _dcg(list(range(min(len(relevant), k))))
    return actual / ideal if ideal > 0 else 0.0


# ── Per-instance evaluation ───────────────────────────────────────────────


async def evaluate_instance(
    instance: dict,
    embedder: Any,
    fetch_k: int = 50,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict | None:
    """
    Run one LongMemEval instance end-to-end.

    Returns None for abstention questions (skipped for retrieval metrics).
    Otherwise returns a dict with r@k and ndcg@k for each k in k_values.
    """
    question_id = instance["question_id"]
    if question_id.endswith("_abs"):
        return None

    question = instance["question"]
    haystack_sessions = instance["haystack_sessions"]
    haystack_session_ids = instance["haystack_session_ids"]
    relevant_sessions = set(instance.get("answer_session_ids", []))
    if not relevant_sessions:
        return None

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "lme.db")
        store = UnifiedMemoryStore(db_path=db_path, embedding_client=embedder)
        try:
            # Populate the store: ONE memory per session (not per turn).
            # The benchmark grades at session granularity (answer_session_ids),
            # so session-level storage matches what's being evaluated AND is
            # ~20x faster than turn-level storage. We concatenate user messages
            # first (they carry the facts being tested), then fall back to
            # the full session content capped to fit BGE-small's 256-token limit.
            for sid, session in zip(haystack_session_ids, haystack_sessions):
                user_parts: list[str] = []
                full_parts: list[str] = []
                for turn in session:
                    content = (turn.get("content") or "").strip()
                    if not content:
                        continue
                    role = turn.get("role", "user")
                    full_parts.append(f"{role}: {content}")
                    if role == "user":
                        user_parts.append(content)

                # Prefer user-message concatenation (denser on tested facts),
                # fall back to full session if no user messages.
                content_for_memory = (
                    "\n".join(user_parts) if user_parts else "\n".join(full_parts)
                )
                if not content_for_memory:
                    continue
                # Cap at ~1000 chars (~250 tokens) — comfortably under
                # BGE-small's 256-token MAX_SEQ_LEN so nothing gets truncated.
                if len(content_for_memory) > 1000:
                    content_for_memory = content_for_memory[:1000]

                await store.store(
                    content=content_for_memory,
                    memory_type="conversation",
                    user_id="lme",
                    session_id=sid,
                )

            # Semantic recall for the question
            results = await store.recall(
                query=question,
                user_id="lme",
                top_k=fetch_k,
            )
        finally:
            store.close()

    # Dedupe to unique session_ids while preserving rank order
    seen: set[str] = set()
    retrieved_session_order: list[str] = []
    for mem, _score in results:
        sid = mem.get("session_id")
        if sid and sid not in seen:
            retrieved_session_order.append(sid)
            seen.add(sid)

    out = {
        "question_id": question_id,
        "question_type": instance.get("question_type", "unknown"),
        "num_relevant": len(relevant_sessions),
        "num_retrieved": len(retrieved_session_order),
    }
    for k in k_values:
        out[f"r@{k}"] = recall_at_k(retrieved_session_order, relevant_sessions, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved_session_order, relevant_sessions, k)
    return out


# ── Benchmark runner ──────────────────────────────────────────────────────


async def run_benchmark(
    dataset_path: Path,
    max_instances: int | None = None,
    fetch_k: int = 50,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Run LongMemEval against PredaCore memory and return aggregated metrics."""
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        instances = json.load(f)

    if max_instances is not None:
        instances = instances[:max_instances]

    n = len(instances)
    print(f"Running {n} instances (fetch_k={fetch_k}, k_values={k_values})")
    print("First call will trigger BGE-small-en-v1.5 model download (~133 MB)...")

    embedder = RustEmbeddingClient()

    all_results: list[dict] = []
    skipped_abs = 0
    t_start = time.time()

    for i, instance in enumerate(instances):
        result = await evaluate_instance(
            instance, embedder, fetch_k=fetch_k, k_values=k_values
        )
        if result is None:
            skipped_abs += 1
            continue
        all_results.append(result)

        if (i + 1) % 25 == 0 or i == n - 1:
            running_r5 = sum(r["r@5"] for r in all_results) / max(len(all_results), 1)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i + 1}/{n}] running R@5={running_r5:.3f}  "
                f"({rate:.1f} inst/s, skipped_abs={skipped_abs})"
            )

    if not all_results:
        return {"error": "no scorable instances (all abstention?)"}

    # Overall metrics (average across scored instances)
    total = len(all_results)
    overall: dict[str, Any] = {
        "scored_instances": total,
        "skipped_abstention": skipped_abs,
        "total_elapsed_sec": round(time.time() - t_start, 1),
    }
    for k in k_values:
        overall[f"R@{k}"] = sum(r[f"r@{k}"] for r in all_results) / total
        overall[f"NDCG@{k}"] = sum(r[f"ndcg@{k}"] for r in all_results) / total

    # Per-category breakdown
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_category[r["question_type"]].append(r)

    category_metrics: dict[str, dict] = {}
    for cat, rs in sorted(by_category.items()):
        c = len(rs)
        cat_m: dict[str, Any] = {"n": c}
        for k in k_values:
            cat_m[f"R@{k}"] = sum(r[f"r@{k}"] for r in rs) / c
            cat_m[f"NDCG@{k}"] = sum(r[f"ndcg@{k}"] for r in rs) / c
        category_metrics[cat] = cat_m

    return {"overall": overall, "by_category": category_metrics}


# ── CLI ───────────────────────────────────────────────────────────────────


def _print_report(metrics: dict) -> None:
    if "error" in metrics:
        print(f"\nERROR: {metrics['error']}")
        return

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    overall = metrics["overall"]
    print(f"  scored instances:     {overall['scored_instances']}")
    print(f"  skipped abstention:   {overall['skipped_abstention']}")
    print(f"  total elapsed (sec):  {overall['total_elapsed_sec']}")
    print()
    for key in sorted(overall):
        if key.startswith("R@") or key.startswith("NDCG@"):
            print(f"  {key:10s} {overall[key]:.4f}")

    print("\n" + "=" * 60)
    print("BY QUESTION TYPE")
    print("=" * 60)
    for cat, m in metrics["by_category"].items():
        print(f"\n  {cat}  (n={m['n']})")
        for key in sorted(m):
            if key == "n":
                continue
            print(f"    {key:10s} {m[key]:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark for PredaCore unified memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to longmemeval_s_cleaned.json (download from HuggingFace)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit to first N instances (useful for smoke testing)",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=50,
        help="How many raw memories to retrieve before deduping to unique sessions",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="k values to compute R@k and NDCG@k for",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full metrics dict to this JSON path",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        print(
            "Download with:\n"
            "  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            file=sys.stderr,
        )
        return 2

    metrics = asyncio.run(
        run_benchmark(
            dataset_path=args.dataset,
            max_instances=args.max_instances,
            fetch_k=args.fetch_k,
            k_values=tuple(args.k),
        )
    )

    _print_report(metrics)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nFull metrics written to {args.json_out}")

    return 0 if "error" not in metrics else 1


if __name__ == "__main__":
    sys.exit(main())
