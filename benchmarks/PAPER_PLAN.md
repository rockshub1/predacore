# Paper Plan — LongMemEval v0.1.0 → Paper-Level

_Plan for turning the JARVIS memory benchmark (`lme_v0.1.0_full.json`, R@5 = 0.9574)
from a GitHub benchmark write-up into a paper-level submission. Honest gap analysis
and a 4-week work plan._

_Written: 2026-04-14. Based on: `benchmarks/README.md` v0.1.0._

---

## TL;DR

**The number is paper-level. The documentation isn't. The work is ~4 weeks away.**

- **0.9574 vs 0.74** is a 22-point delta on a standard public benchmark
  (LongMemEval, ICLR 2025). That beats every published R@5 I know of.
- Current README is a high-quality engineering doc with honest caveats, but it
  is not a paper.
- Gap to paper-level is **scientific rigor** (ablations, stats, multiple
  datasets, controlled baselines, formal math), not more engineering.
- 3-4 weeks of focused work gets this to a main-conference submission at
  SIGIR / EMNLP / NAACL. Current state is workshop-level or TMLR-level as-is.

---

## What is already paper-worthy

### 1. The headline result is strong enough

| System | R@5 | Delta |
|---|---|---|
| **JARVIS v0.1.0** | **0.9574** | — |
| Hippo (Show HN, BM25-only) | 0.74 | +22 points |
| HippoRAG 2 (graph-based, PPR) | 0.70-0.75 | +22 points |
| Raw sentence-transformers baseline | 0.55-0.65 | +35 points |

Published hybrid retrieval on LongMemEval lives in the 0.70-0.85 range.
JARVIS is above the top of that range. A 22-point delta over the next-best
published system is the kind of delta that reviewers pay attention to.

### 2. Architecture has genuine novelty (not just standard IR)

Most of the stack is standard (BGE embeddings, BM25, SQLite, WAL, vector
search). Three components are **not standard** and are publishable contributions:

1. **Reward-proportional decay** — memories tied to positive sessions decay
   slower, memories tied to failed sessions decay faster. Hippo-inspired but
   applied at the consolidation layer, not retrieval. I have not seen this
   in published work.
2. **Session-level indexing + 5-section budgeted context** — one memory per
   session (not per turn), retrieval splits results across 5 category-typed
   sections (preferences, entities, semantic, fuzzy, episodes), each with
   its own token budget. An agent-memory design pattern, not a general IR
   technique.
3. **Rust SIMD + rebuild-on-startup vector index** — engineering, not research,
   but the fact that rebuilding a 100K vector index from SQLite on every
   startup is cheap enough to use is a design choice worth writing up.

Any one alone might not be a paper. All three together, plus the benchmark
result, absolutely is.

### 3. Reproducibility is already documented

- Commit hash, exact command line, exact dataset URL, exact config, Rust
  build steps, expected numbers with ±1% tolerance.
- This is more rigorous than a lot of published benchmark sections already.
- The `README.md` + `lme_v0.1.0_full.json` + `lme_v0.1.0_full.log` triple
  is good reproduction material.

---

## What is NOT paper-level yet

### Gap 1: No ablation study (CRITICAL)

**Reviewers will immediately ask: "which components contribute how much to 0.9574?"**
Without this, the paper gets rejected for inability to isolate the contribution.

**Required ablation table:**

```
Configuration                                R@5     ΔR@5
─────────────────────────────────────────── ─────   ──────
Full JARVIS                                 0.9574    —
  − reward-proportional decay               0.????  -0.???
  − BM25 (semantic only)                    0.????  -0.???
  − semantic (BM25 only)                    0.????  -0.???
  − 5-section budgeted (single pool)        0.????  -0.???
  − session-level indexing (turn-level)     0.????  -0.???
  − fuzzy + synonym expansion               0.????  -0.???
  − in-RAM vector index (SQLite-only)       0.????  -0.???
  − rerank (raw cosine only)                0.????  -0.???
```

**Effort:** 4-5 days. Each row is one modified config run on the full 500-instance
benchmark (~53 minutes per row on M-series hardware). 8 rows × 55 min ≈ 7 hours
of compute, but allow time to implement each ablation cleanly as a config flag.

**Rationale:** this is the single most important gap. Without it, every other
improvement is wasted because the paper cannot be accepted.

### Gap 2: Single dataset

LongMemEval alone is not sufficient for a top-venue submission. Papers typically
cover 2-4 retrieval benchmarks.

**Target datasets:**
- **MTEB retrieval subset** (NQ, TriviaQA, MS MARCO, BEIR) — establishes the
  retrieval layer generalizes beyond LongMemEval's specific structure
- **Alternative memory benchmarks** — MemoryBench, LongBench, or MemN2N eval
  suite — for cross-benchmark validation of the memory-specific claims

**Effort:** 5-6 days. Mostly integration work (plumbing MTEB or MemoryBench
datasets through the existing evals harness) + compute time for full runs.

**Rationale:** single-benchmark wins get extra scrutiny. Reviewers will say
"nice LongMemEval number but how do we know it generalizes?"

### Gap 3: No statistical significance

The current run is single-seed. Papers need:
- **3-5 runs with different seeds**, reporting mean ± std
- **Bootstrap confidence intervals** on the R@5 metric
- **Wilcoxon signed-rank test** on the per-instance R@5 differences between
  systems (JARVIS vs each baseline)
- Pairwise paired tests wherever a comparison is made

For your benchmark specifically, non-determinism comes from floating-point
ordering in parallel Rust vector search for close score ties (the README notes
±1%). Running it 5 times and reporting `0.957 ± 0.003` + a p-value on the
JARVIS-vs-Hippo comparison makes the claim bulletproof.

**Effort:** 2-3 days. Run 5 seeds of the full config (~5 hours of compute),
compute confidence intervals and run the significance tests.

### Gap 4: Missing end-to-end answer accuracy

LongMemEval is a **two-stage benchmark**:
1. **Retrieval** — given a question + haystack, find the right session (R@k)
2. **Generation** — given the retrieved context, generate a correct answer

Current benchmark only measures stage 1. Stage 2 is typically 10-20 points
lower than R@k across systems. Reviewers will ask **"what's the actual task
accuracy?"** because R@5 alone doesn't tell them if the LLM can actually use
what was retrieved.

**Required addition:**
- Plug retrieval output into a generation pipeline (Claude-Opus-4 or GPT-4-Turbo)
- Run LongMemEval's answer-grading script on the generated answers
- Report paired (retrieval, generation) numbers:
  - `R@5 = 0.9574, Answer Accuracy = 0.????`
- Break down by question type (some categories are harder to answer from
  correct context than others)

**Effort:** 5-6 days. Includes integration, compute (generation is slower),
and debugging the answer-grading pipeline.

**Rationale:** a pure-retrieval paper at SIGIR would accept retrieval-only
numbers. A memory-for-agents paper at EMNLP/NAACL/ACL/NeurIPS/ICLR needs
both. The framing of the paper affects this — see "Venue strategy" below.

### Gap 5: Baselines not run under identical conditions

Current README compares to:
- Hippo 0.74 (published Show HN)
- HippoRAG 2 0.70-0.75 (published)
- Raw semantic 0.55-0.65 (typical)

**These are not controlled comparisons — they're published numbers from
different papers using different hardware, seeds, and possibly data splits.**
Reviewers will require head-to-head numbers under identical conditions.

**Required baselines to re-run locally:**
- **Hippo** — use their code, same hardware, same data split
- **HippoRAG 2** — same
- **BGE-M3 hybrid** (dense + sparse) — strong off-the-shelf hybrid retriever
- **ColBERTv2** — strong late-interaction dense retriever
- **Long-context baseline** — feed the entire haystack to Claude-Opus-4 or
  GPT-4-Turbo (200K context window) with no retrieval at all. This is the
  most politically important comparison in 2025-2026 because "just use a
  long-context model" is the default skeptic's question.

**Effort:** 6-8 days. Each baseline requires setup, a full 500-instance run,
and data-collection. Some will require building adapters to feed LongMemEval
haystacks into the baseline system.

### Gap 6: No formal method section

A paper needs explicit math. Reviewers will not accept prose-only descriptions.

**Required equations:**

1. **Reward-proportional decay:**
   ```
   λ_type(t) = λ_base_type × f(reward_session(m))
   where f(r) = exp(-k × (1 - r))  for r in [0, 1]
   ```
   Plus the specific k values per memory type, derivation of the half-life
   transform, and the rationale for reward normalization.

2. **Hybrid retrieval fusion:**
   ```
   score(q, m) = α × sim_cosine(q, m) + (1-α) × BM25(q, m.content)
   ```
   Plus how α is chosen (hyperparameter search results), why not reciprocal
   rank fusion, what happens when one score is zero.

3. **5-section budgeted context builder:**
   ```
   context = concat({
     preferences: top_k(B_pref),
     entities:    top_k(B_ent),
     semantic:    top_k(B_sem),
     fuzzy:       top_k(B_fuzzy),
     episodes:    top_k(B_ep)
   })
   where Σ B_i ≤ total_token_budget
   ```
   Plus the per-section budget allocation and rationale.

4. **Rerank formula:**
   ```
   rerank(m, q) = 0.6 × sim(q, m) + 0.25 × recency(m) + 0.15 × importance(m)
   recency(m)  = exp(-0.05 × age_days(m))   (14-day half-life)
   importance(m) ∈ [0, 1]  (learned or hand-set)
   ```
   Plus how 0.6/0.25/0.15 was chosen — hyperparameter search, ablation, or
   prior — and sensitivity analysis.

**Effort:** 2-3 days. Assuming the engineering exists, this is just writing
the math out cleanly with derivations.

### Gap 7: No related work section

README mentions Hippo and HippoRAG 2 but does not situate JARVIS in the broader
literature. A paper's related work section needs:

- **Memory-augmented agents:** Voyager, Generative Agents, MemGPT, Reflexion,
  CAMEL's memory layer
- **Retrieval-augmented generation:** RAG (Lewis et al.), Self-RAG,
  Fusion-in-Decoder, RETRO
- **Long-context vs retrieval:** the long-context wave from 2024-2025
  (Claude-3 200K, Gemini-1.5 1M, GPT-4-Turbo 128K), and the papers arguing
  "retrieval still wins" (LongBench, InfiniteBench, Needle-in-Haystack)
- **Hippocampal memory models:** Hippo, HippoRAG, HippoRAG 2, SCM3 (spaced
  consolidation)
- **Episodic vs semantic memory:** Tulving's distinction, memory consolidation
  literature
- **Consolidation and decay models:** exponential decay, power-law forgetting,
  spaced repetition (SM-2, Anki), Ebbinghaus curve
- **Hybrid retrieval:** ColBERT, SPLADE, BGE-M3, E5-Mistral, late interaction
- **Knowledge graphs for memory:** GraphRAG, KG-RAG, HippoRAG's personalized
  PageRank approach

**Effort:** 2-3 days. Read broadly, cite thoroughly, identify which systems
are closest to JARVIS and explain the delta.

---

## Venue strategy

### Target venues, ranked by fit

| Venue | Fit | Current state? | With ablations + stats? | With ablations + 2nd dataset + E2E? |
|---|---|---|---|---|
| **Workshop at NeurIPS/ICLR** (e.g., Instruction Workshop, Foundation Models Workshop) | ✅ strong | ✅ accept | ✅ very strong | ✅ accept |
| **TMLR** (rolling review, no page limit) | ✅ strong | ✅ accept | ✅ strong | ✅ strong |
| **SIGIR** (Information Retrieval, loves benchmarks) | ✅ strong | ⚠️ borderline reject | ✅ accept | ✅ strong accept |
| **EMNLP / NAACL / ACL** (NLP venues) | ✅ good | ❌ reject (method rigor) | ✅ likely accept | ✅ accept |
| **NeurIPS / ICLR main** | ⚠️ higher bar | ❌ reject | ⚠️ borderline | ✅ likely accept |
| **COLM** (Conference on Language Models, newer) | ✅ excellent | ⚠️ borderline | ✅ accept | ✅ strong accept |

**Recommended primary target:** **SIGIR 2027** (submission ~Jan 2027) or
**COLM 2026** (submission ~May 2026, shorter timeline). Both are natural fits
for retrieval-focused memory work with ablation-driven contributions.

**Recommended fallback:** **EMNLP 2026 Findings** for faster turnaround, or
**TMLR** for no-deadline rolling review.

**Avoid first-time:** NeurIPS / ICLR main track — the rigor bar is higher and
the "just-use-long-context" skeptic is harder to satisfy without multiple
strong dense baselines.

### Framing choice

The same work can be framed three different ways, each hitting a different venue:

1. **"Fast memory retrieval for AI agents"** — SIGIR / CIKM angle. Emphasizes
   retrieval quality, Rust performance, hybrid fusion. Retrieval-only results
   are enough.

2. **"Memory consolidation via reward-proportional decay"** — NeurIPS / ICLR
   angle. Emphasizes the novel decay mechanism as the contribution. Needs
   deeper theoretical analysis of why reward-weighted decay works, maybe a
   toy model or ablation curve showing the effect of decay rate on R@5 over
   session history length.

3. **"Production-ready long-term memory for LLM agents"** — EMNLP / NAACL /
   ACL / COLM angle. Emphasizes the full agent-memory system (consolidation,
   retrieval, evolution), end-to-end answer accuracy, and the "works in
   production for weeks" story. Needs end-to-end accuracy numbers and a
   deployment case study.

**Recommendation:** Frame as #3 first (agent-memory, end-to-end). It's the
highest-impact framing and the most novel overall. Save framings #1 and #2
as fallbacks if the full scope is too much work.

---

## 4-Week execution plan

### Week 1 — Ablations + statistical rigor

**Goal:** answer the "which components contribute what" question and add
statistical significance to the existing result.

| Day | Task |
|---|---|
| 1 | Implement config flags for each ablation (8 total). Each flag disables one component cleanly without breaking downstream. |
| 2 | Run full 500-instance ablation for 4 configurations (reward decay off, BM25 only, semantic only, 5-section off). ~4 hours of compute. |
| 3 | Run remaining 4 ablations (turn-level indexing, no fuzzy, no in-RAM index, no rerank). |
| 4 | Run the full JARVIS config 5 times with different seeds. Compute mean/std/bootstrap CI. |
| 5 | Run paired Wilcoxon signed-rank tests between JARVIS and each ablation, and between JARVIS and the published baselines. |

**Deliverables:**
- `benchmarks/ablation_v0.1.1.json` — full ablation table with mean ± std and CIs
- `benchmarks/significance_v0.1.1.json` — pairwise p-values
- Updated `README.md` with the ablation table and statistical notes

### Week 2 — Second dataset + head-to-head baselines

**Goal:** show JARVIS generalizes beyond LongMemEval and controls for hardware
/ seed by re-running prior work.

| Day | Task |
|---|---|
| 1 | Build MTEB retrieval adapter (or MemoryBench adapter). Wire into the evals harness. |
| 2 | Run JARVIS on the chosen 2nd dataset. Full run. |
| 3 | Re-run Hippo on LongMemEval under identical conditions (same hardware, same seeds). |
| 4 | Re-run BGE-M3 hybrid and ColBERTv2 on LongMemEval under identical conditions. |
| 5 | Run the long-context baseline: feed full haystacks to Claude-Opus-4 (or GPT-4-Turbo) with no retrieval. Measure answer accuracy (skip R@k — undefined for no-retrieval). |

**Deliverables:**
- `benchmarks/2nd_dataset_v0.1.1.json` — full metrics on the second benchmark
- `benchmarks/head_to_head_v0.1.1.json` — all baselines re-run locally
- `benchmarks/long_context_v0.1.1.json` — the "just use a long-context model" numbers

### Week 3 — End-to-end answer accuracy + paper draft begins

**Goal:** complete the second half of LongMemEval (answer generation) and start
writing.

| Day | Task |
|---|---|
| 1 | Plug retrieval output into Claude-Opus-4 (or GPT-4-Turbo) generation pipeline. Implement LongMemEval's answer-grading rubric. |
| 2 | Run end-to-end generation on the full 470 scored instances. Compute paired (retrieval R@5, answer accuracy) numbers. |
| 3 | Write paper draft: introduction, motivation, contributions list. |
| 4 | Write paper draft: related work (~1500 words, ~40 citations). |
| 5 | Write paper draft: method section with all equations. |

**Deliverables:**
- `benchmarks/e2e_v0.1.1.json` — paired retrieval + answer accuracy numbers
- `paper/draft_v0.1.md` — intro + related work + method sections

### Week 4 — Writeup finish + polish + figures

**Goal:** ship a complete paper draft ready for a venue submission.

| Day | Task |
|---|---|
| 1 | Write paper draft: experimental setup, results section with all tables. |
| 2 | Write paper draft: discussion + limitations + conclusion. |
| 3 | Build figures: architecture diagram, per-category breakdown bar chart, ablation plot, latency histogram, R@k vs k curve. Use matplotlib / tikz for venue-appropriate format. |
| 4 | Second-reader pass. Get a technical reader to review for clarity and rigor. Address feedback. |
| 5 | Final polish: abstract, contributions bullet list, intro punchline, bibliography cleanup, reproducibility checklist. Submit or queue for submission. |

**Deliverables:**
- `paper/draft_v1.md` — complete paper ready for submission
- `paper/figures/` — all figures in PDF/PNG/tikz
- `paper/submission_checklist.md` — venue-specific checklist done

---

## Things NOT worth doing

Clarifying what this plan explicitly does **not** include, to keep scope honest:

- **Multi-user benchmarks.** LongMemEval is single-user. Multi-user memory
  robustness is out of scope — a separate paper.
- **Multi-lingual.** BGE-small-en-v1.5 is English-only. Multi-lingual is a
  separate paper if pursued.
- **Replacing BGE-small with BGE-large.** Larger embedding model would
  plausibly improve the weak `single-session-preference` category, but it
  also doubles memory and slows retrieval. Worth ablating with BGE-large as
  a sensitivity test, but not worth rewriting the primary config.
- **Agent end-to-end case study.** Running JARVIS in production on a real
  user's data for weeks, then measuring outcome quality. Would strengthen
  a "production deployment" framing, but adds a month of time and requires
  user consent. Skip unless Framing #3 is the chosen angle and the reviewer
  flags its absence.
- **New theoretical framework.** Memory decay theory is well-studied. We
  don't need to invent a new theoretical model for reward-proportional decay
  — we just need to document and benchmark the one we have. A theoretical
  contribution could be a sequel paper.
- **Wider comparison sweep.** E5-Mistral, NV-Embed, Nomic-Embed, Jina-Embed,
  MiniLM-L12. Pick 2-3 strong dense retrievers to compare against. Don't
  compare against every dense retriever on HuggingFace. Diminishing returns
  after the first 3.

---

## Decision gates

Before committing to the 4-week plan, answer these:

1. **Do you want a publication credit?**
   - **Yes** → commit to the plan, target SIGIR 2027 or COLM 2026
   - **No** → skip the plan, benchmark stays as excellent internal documentation
     and Show HN material

2. **Is the framing #3 (production agent-memory system) the right angle?**
   - **Yes** → include end-to-end accuracy (Week 3) and possibly a deployment
     case study
   - **No** → cut Week 3's end-to-end work and target SIGIR as framing #1
     (pure retrieval), which saves 5 days

3. **Is the second dataset worth the 5-6 days?**
   - **Yes (for top venue)** → keep Week 2
   - **No (workshop or TMLR target)** → cut Week 2, save 5-6 days, target
     Workshop or TMLR instead — both accept single-dataset submissions

4. **Is the long-context baseline essential?**
   - **Yes** (2025-2026 skeptic check) → keep it in Week 2
   - **No** (can cite existing long-context vs retrieval papers instead)
     → cut 1 day from Week 2

### Minimum-viable plan (2 weeks instead of 4)

If 4 weeks is too much, the minimum-viable version:

- **Week 1** unchanged (ablations + stats, no cuts) — this is non-negotiable
- **Week 2** — only the paper draft + figures + polish + submit to TMLR or a workshop

This gets you a paper-level submission at a lower-bar venue without the
second-dataset and head-to-head work. Roughly doubles acceptance risk but
saves 2 weeks.

---

## Reviewer objection preemption

The objections most likely from reviewers, and how to address each:

1. **"Ablation study missing."**
   - Addressed by Week 1. This is the single most important fix.

2. **"You only tested one dataset."**
   - Addressed by Week 2 (second dataset). Or: acknowledge in limitations if
     targeting workshop / TMLR.

3. **"Comparison baselines not run under identical conditions."**
   - Addressed by Week 2 (head-to-head re-runs). Cite as "our reimplementation"
     where needed.

4. **"Retrieval ≠ answer accuracy."**
   - Addressed by Week 3 (end-to-end). Or: frame as retrieval-only paper at
     SIGIR where that's acceptable.

5. **"Just use a long-context model instead."**
   - Addressed by Week 2 long-context baseline. Show that retrieval is faster,
     cheaper, and at least as accurate as stuffing 200K tokens into a context
     window.

6. **"Why not BGE-large / E5-Mistral?"**
   - Sensitivity analysis in Week 1 or 2. Show the effect of swapping the
     embedding model. Honest framing: BGE-small chosen for local-compute
     reasons, and the ablation shows how much BGE-large would buy you.

7. **"Reward-proportional decay is just reward-weighted LRU."**
   - Not quite — LRU uses recency, we use reward-conditioned decay rate.
     Related but distinct. Make this explicit in related work section with
     a diagram comparing decay curves under different reward values.

8. **"Single-session-preference is 0.8667, that's your weakest category — so
    the system isn't actually robust."**
   - Agree and extend: in Week 1, run the preference-boosting ablation (higher
     importance weight on preference-type memories). If it helps, include that
     as an additional contribution. If not, document in limitations as "the
     known soft spot, here's why, here's what would fix it."

---

## Final note

The number `0.9574` is already impressive enough to defend. The work is already
novel enough to be publishable. The gap is **~4 weeks of writing-and-rigor
work**, not another year of engineering. That makes this plan very doable if
the decision is "yes, let's submit."

If the decision is "no, we're just shipping the product," **the current
benchmark is more than sufficient** as internal documentation + public
marketing material. There is no dishonor in not turning a benchmark into a
paper. The paper pathway exists because the result is paper-worthy, not
because it's required.

---

_Plan committed. Next actions: decide on framing (#1, #2, or #3), decide on
venue timeline (SIGIR 2027 / COLM 2026 / EMNLP 2026 Findings / TMLR), and
start Week 1 or explicitly defer._
