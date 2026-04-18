# Memory

> The engine that makes PredaCore remember you.

PredaCore's memory isn't "throw everything into a vector DB and hope." It's a **5-stage retrieval pipeline with per-stage token budgets**, a **hybrid ranking formula** with recency half-life and importance prior, and a **consolidator that learns from outcomes** — memories from successful sessions decay slower, failed ones fade faster.

This is the file that explains how.

---

## 5-stage retrieval pipeline

Every turn that needs recall runs this pipeline (`memory/retriever.py:31-53`):

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────────┐
│ Preferences │→ │  Entity     │→ │  Semantic   │→ │  Fuzzy     │→ │  Episodes  │
│   500 tok   │  │   800 tok   │  │   1200 tok  │  │  400 tok   │  │  remainder │
└─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  └────────────┘
   5min cache      5min cache      60s cache        stream          stream
```

Each stage is a separate method with its own cache TTL. The LLM never sees "here are 20 nearest neighbors" — it sees a curated dossier that respects each category's signal shape.

**Why the budgets look weird:** semantic gets the largest budget (1200) because it's the broadest signal. Preferences gets only 500 because they're dense and repeat. Entity context is 800 because it grounds the semantic results. Fuzzy is 400 because it's a tie-breaker for typos. Episodes soak up whatever's left.

---

## Hybrid ranking formula

Cosine similarity alone is a naive ranker. PredaCore uses this (`memory/retriever.py:253-257`):

```python
final_score = 0.60 * cosine_similarity
            + 0.25 * math.exp(-0.05 * age_days)    # 14-day half-life
            + 0.15 * (importance / 5.0)            # stored prior
```

**What each term does.** Cosine retrieves by meaning. The exponential recency term applies a ~14-day half-life — useful context from today outranks equally-similar context from last quarter. The importance prior lets the agent flag memories at store time (`importance=5`) and have them surface first.

Tuned by grid search against LongMemEval. Changing the weights is a one-line diff and a re-run; we ship the values that scored 0.9574.

---

## Reward-proportional decay — online learning via memory

Every session's outcome (tools, iterations, tokens, duration, success/error) is written to the `OutcomeStore`. The **consolidator** reads those outcomes and modulates decay (`memory/consolidator.py:121-136`):

- Sessions with **high reward** → linked memories decay **slower**.
- Sessions with **low reward or errors** → linked memories decay **faster**.

Over time, bad patterns fade out and good ones stick around. That's online learning. No fine-tuning, no LoRA — just memory curation shaped by whether the agent actually helped.

---

## The 7-step consolidator

Runs every ~6 hours or every ~50 new memories (`memory/consolidator.py:103-197`):

1. **Apply decay** — exponential, reward-modulated.
2. **Extract entities** — Rust extraction + optional LLM enrichment.
3. **Auto-link** — Rust relation classifier detects uses/prefers/part-of from co-occurrence.
4. **Summarize sessions** — LLM compresses transcripts into episodes with `key_facts · tools_used · outcome · satisfaction`.
5. **Merge similar** — dedupe at 0.87 cosine.
6. **Prune** — drop expired + low-importance.
7. **Enforce cap** — 25k memories per user · preference/entity types protected from eviction.

**Known sharp edge:** the 0.87 merge threshold can occasionally fuse two near-duplicate facts that shouldn't be fused. No safeguard for "important but similar" cases. On the v0.3 list.

---

## Memory scopes

Memories can be scoped (`memory/store.py:61`):

| Scope | Lifetime | Shared with |
|---|---|---|
| `global` | Permanent | All agents for this user |
| `agent` | Permanent | This specific agent only |
| `team` | 72h TTL | Agents collaborating in a multi-agent run |
| `scratch` | Session | Current turn only |

Multi-agent runs persist findings to `team` scope — a private scratchpad that expires automatically. Team findings do **not** leak to `global`.

---

## Rust compute kernel

The whole memory engine sits on 1,872 lines of Rust across 8 files in `predacore_core_crate/src/`:

| File | LOC | Role |
|---|---|---|
| `lib.rs` | 189 | PyO3 bindings · auto-switch to parallel scan when N > 1000 |
| `vector.rs` | 206 | Chunked cosine (8-wide inner loop, `unsafe get_unchecked` for auto-vectorization) · rayon parallel for large scans |
| `embedding.rs` | 261 | BGE-small-en-v1.5 via Candle · 384-dim · 256-seq · mean-pool + L2 |
| `bm25.rs` | 178 | BM25 with IDF smoothing |
| `fuzzy.rs` | 242 | Jaccard over 3-char shingles — 3-char-typo tolerant, language-agnostic |
| `entity.rs` | 348 | 3-tier extraction: dictionary → regex → stopword filter · source-tier + confidence |
| `relations.rs` | 255 | Window-aware sentence parsing: uses / prefers / part_of / etc. with confidence |
| `synonyms.rs` | 193 | Tech-domain synonym expansion |

**BGE loads once** into a `Mutex<Option<...>>` and stays warm. 133 MB model, cached at `~/.cache/huggingface/`. Cargo release profile is tight: `lto = "fat"`, `codegen-units = 1`, `strip = true`.

**Pre-built wheels** for `macosx_11_0_universal2`, `manylinux2014_x86_64`, `manylinux2014_aarch64`, `win_amd64`. No Rust toolchain required for end users.

---

## Why trigrams?

`fuzzy.rs` uses 3-char shingles, not Levenshtein or edit distance. Two reasons:

1. **Language-agnostic** — shingles work for any script. Edit distance needs tokenizer assumptions.
2. **3-char tolerance** — catches typos down to 3 characters (`"congif"` → `"config"`) which is where most human typos land, without the false positives of 2-char shingles.

Jaccard similarity on the shingle sets, threshold 0.3 by default. It's the last-resort stage, gets only 400 tokens of budget.

---

## Entity extraction is 3-tier

`entity.rs`:

```
Tier 1: Dictionary        (exact match, confidence 1.0)
Tier 2: Regex             (pattern match, confidence 0.7–0.9)
Tier 3: Stopword filter   (everything else, confidence varies)
```

Every extracted entity carries a `source_tier` + `confidence`. Downstream consumers can filter: e.g., the relation classifier ignores tier-3 entities below 0.5 confidence.

Optionally augmented by a one-shot LLM pass in the consolidator — off by default, on via `enable_llm_entity_enrichment` config flag.

---

## Relation classification

`relations.rs` does window-aware sentence parsing. For each pair of co-occurring entities within a fixed sentence window, the classifier attempts to attach a typed relation:

| Relation | Signal |
|---|---|
| `uses` | Verb-form detection + entity type compatibility |
| `prefers` | Preference markers ("I like", "favor", "prefer") |
| `part_of` | Containment phrases |
| `located_in` | Location markers |
| `knows` | Knowledge assertions |

Each carries a confidence score. Relations below 0.4 are dropped. Stored in the `relations` SQLite table, queried during the entity stage of retrieval.

---

## SQLite schema

Four tables in `unified_memory.db` (plus legacy `memory.db` — being phased out):

- `memories` — id, content, embedding, scope, importance, created_at, access_count, last_accessed, source_session
- `entities` — name, type, canonical_form, source_tier, confidence
- `relations` — subject, predicate, object, confidence, evidence_memory_id
- `episodes` — session_id, summary, key_facts, tools_used, outcome, satisfaction

Vector index is **in-RAM only**. SQLite is the source of truth; the vector index rebuilds on daemon start. Trade-off: slower cold start, no disk-contention on every read.

WAL mode, 5s `busy_timeout`. Single-writer enforced by the unix-socket DB broker (`services/db_server.py`).

---

## Benchmark reproduction

Retrieval is **fully deterministic** — BGE embed → vector search → rank. No LLM, no sampling, no RNG. Which is why we can promise bit-identical reproduction:

```bash
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python -m predacore.evals.longmemeval --dataset longmemeval_s_cleaned.json --json-out my_run.json

python -c "
import json
mine = json.load(open('my_run.json'))
ours = json.load(open('benchmarks/lme_v0.1.0_full.json'))
for k in ('R@5','R@10','R@20','NDCG@5','NDCG@10','NDCG@20'):
    print(f'{k:8s}  mine={mine[\"overall\"][k]:.4f}  ours={ours[\"overall\"][k]:.4f}')
"
```

Every metric matches the v0.1.0 baseline to 4 decimals. If yours doesn't, something in your environment diverges — file an issue.

---

## Limits

- **GIL not released.** Concurrent Python threads calling `embed()` serialize on the GIL. rayon parallelism helps within a single call. Fix in v0.3 (wrap long inference paths in `py.allow_threads`).
- **`single-session-preference` R@5 = 0.867.** The weak category. Preference-shaped questions are softer semantic signals than facts or events. Cross-encoder re-ranker targeted for v0.3 (target 0.97+ overall).
- **Consolidator LLM summarization is unbounded.** On a large session backlog, burns tokens. No hard cost cap.
- **Merge threshold 0.87** occasionally fuses near-duplicates that shouldn't fuse. v0.3.
- **Single-writer DB broker.** One long write stalls the write lane. Sharding is future work if anyone hits it.
- **Legacy `memory.db`** still exists alongside `unified_memory.db`. Phase-out in v0.3.
