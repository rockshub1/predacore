from __future__ import annotations

import json
import os
import threading
from typing import Any
from uuid import uuid4


class PlanMotifStore:
    """A lightweight JSONL-backed motif store for plan segments.

    Stores motif entries with fields: id, goal_terms, steps (as simple dicts), ts.
    Retrieval is based on token overlap with the incoming goal.
    """

    def __init__(self, path: str = None):
        self.path = path or os.getenv("MCTS_PLAN_CACHE_PATH", "data/plan_motifs.jsonl")
        parent_dir = os.path.dirname(self.path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._lock = threading.Lock()

    def add_motif(self, goal: str, steps: list[dict[str, Any]]) -> None:
        goal_terms = self._terms(goal)
        entry = {
            "id": str(uuid4()),
            "goal_terms": list(goal_terms),
            "steps": steps,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def retrieve(self, goal: str, top_k: int = 3) -> list[list[dict[str, Any]]]:
        """TF‑IDF-like retrieval of motif steps for a given goal.
        Computes IDF on the fly over stored entries, then scores by sum of IDF of shared terms.
        """
        goal_terms = self._terms(goal)
        entries: list[tuple[set, list[dict[str, Any]]]] = []
        df: dict[str, int] = {}
        N = 0
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        terms = set(obj.get("goal_terms", []))
                        steps = obj.get("steps") or []
                        if not terms:
                            continue
                        N += 1
                        entries.append((terms, steps))
                        for t in terms:
                            df[t] = df.get(t, 0) + 1
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        if N == 0:
            return []
        # Compute IDF
        import math

        idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
        scored: list[tuple[float, list[dict[str, Any]]]] = []
        for terms, steps in entries:
            inter = goal_terms & terms
            if not inter:
                continue
            score = sum(idf.get(t, 0.0) for t in inter) / max(1.0, len(terms))
            scored.append((score, steps))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [steps for _, steps in scored[:top_k]]

    def _terms(self, text: str) -> set:
        return {
            t
            for t in (text or "").lower().split()
            if t.isalpha() or t.replace("-", "").isalpha()
        }
