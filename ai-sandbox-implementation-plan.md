# AI Sandbox: Implementation Plan (Lean Edition)

## RAG Experimentation Platform

> A compact, config-driven codebase for systematically comparing RAG strategies — designed to produce portfolio-ready comparisons, not a framework.

---

## 1. Philosophy

This is a **comparison engine**, not a platform. The deliverable is a Streamlit dashboard and a set of write-ups that prove you can systematically evaluate AI strategies and make defensible recommendations. The codebase is the supporting infrastructure — small enough to read in 10 minutes, powerful enough to run any combination from the search space.

### Design Principles

1. **One pipeline, swappable components.** A single `rag_pipeline.py` chains chunking → retrieval → generation. You swap strategies by changing a config dict.
2. **Config-driven experiments.** Every experiment is a YAML file. Change one variable, rerun, compare.
3. **Evaluation is the product.** The comparison dashboard and write-ups are 100% of the portfolio value.
4. **Real domain, real data.** FMCG product knowledge — a domain you can speak to credibly.
5. **Minimal codebase.** If a module isn't needed for one of the 10-12 planned experiments, it doesn't get built.

---

## 2. Repository Structure

```
ai-sandbox/
│
├── shared/
│   ├── __init__.py
│   ├── llm_client.py              # LLM interface + Azure OpenAI + Anthropic (one file)
│   ├── embeddings.py              # Embedding interface + implementations (one file)
│   ├── vector_store.py            # VectorStore interface + ChromaDB + FAISS (one file)
│   ├── data_loader.py             # PDF, CSV, directory loading (one file)
│   ├── cost_tracker.py            # Token + cost accounting
│   └── timer.py                   # Latency measurement
│
├── rag/
│   ├── __init__.py
│   ├── chunking.py                # All chunking strategies in one file
│   ├── retrieval.py               # Dense, sparse, hybrid, reranker — one file
│   ├── query_transform.py         # Rewrite, HyDE, decomposition — one file
│   ├── generation.py              # Stuff, citation-grounded, structured — one file
│   ├── advanced.py                # CRAG, adaptive RAG — one file
│   └── pipeline.py                # Config → pipeline composition → run
│
├── evaluation/
│   ├── __init__.py
│   ├── schemas.py                 # ExperimentResult, QueryResult, QueryMetrics
│   ├── metrics.py                 # Correctness, faithfulness, precision, recall
│   ├── llm_judge.py               # LLM-as-judge for subjective quality
│   └── evaluator.py               # Run eval suite, produce ExperimentResult
│
├── experiments/
│   ├── runner.py                  # Load config → run pipeline → evaluate → save results
│   ├── configs/                   # All experiment YAML files
│   │   ├── 01_baseline_naive.yaml
│   │   ├── 02_better_llm.yaml
│   │   ├── 03_hybrid_retrieval.yaml
│   │   ├── 04_hybrid_rerank.yaml
│   │   ├── 05_citation_grounded.yaml
│   │   ├── 06_cross_provider.yaml
│   │   ├── 07_query_rewrite.yaml
│   │   ├── 08_hyde.yaml
│   │   ├── 09_parent_child.yaml
│   │   └── 10_crag.yaml
│   ├── datasets/
│   │   └── fmcg_product_qa.jsonl  # 30-50 question + ground truth pairs
│   └── results/                   # Auto-generated JSON after each run
│
├── dashboard/
│   ├── app.py                     # Streamlit: main comparison dashboard
│   ├── views/
│   │   ├── summary_table.py       # Side-by-side metrics comparison
│   │   ├── query_drilldown.py     # Per-query deep dive
│   │   └── tradeoff_scatter.py    # Cost vs quality, latency vs quality
│   └── components.py              # Reusable Streamlit widgets
│
├── comparisons/
│   └── fmcg_product_qa.md         # The portfolio write-up
│
├── data/                          # Source documents for ingestion
│   └── fmcg_docs/
│
├── docs/
│   └── decision_framework.md      # When to use which RAG strategy
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_evaluation.py
│   └── test_configs.py
│
├── pyproject.toml
├── .env.example
├── CLAUDE.md
└── README.md
```

**Total Python files: ~20.** Someone can read and understand this repo in one sitting.

---

## 3. The Search Space

Each experiment picks one option per row:

| Component | Options |
|-----------|---------|
| Chunking | Fixed (512) · Semantic · Parent-child |
| Embedding | text-embedding-3-small · text-embedding-3-large · BGE-base |
| Retrieval | Dense · BM25 sparse · Hybrid (dense+sparse) |
| Reranker | None · BGE-reranker-v2 · LLM-as-reranker |
| Top-K | 3 · 5 · 10 · 20 |
| Query Transform | None · Rewrite · HyDE · Decomposition |
| Generation | Stuff · Citation-grounded · Structured output |
| LLM | GPT-4o-mini · GPT-4o · Claude Sonnet · Claude Opus |
| Advanced Pattern | None · CRAG · Adaptive RAG |

---

## 4. The 10 Experiments

Each changes **one variable** from a neighbor to isolate what drives improvement.

| # | Experiment | Variable Changed | Question It Answers |
|---|-----------|-----------------|-------------------|
| 1 | Fixed + Dense + Stuff + GPT-4o-mini | — (baseline) | How bad is the cheapest possible setup? |
| 2 | Fixed + Dense + Stuff + GPT-4o | LLM ↑ | Does a better LLM compensate for weak retrieval? |
| 3 | Semantic + Hybrid + Stuff + GPT-4o | Chunking ↑ Retrieval ↑ | How much does better retrieval matter? |
| 4 | Semantic + Hybrid + BGE rerank + Stuff + GPT-4o | Reranker added | Is reranking worth the extra step? |
| 5 | Semantic + Hybrid + BGE + Citation-grounded + GPT-4o | Generation ↑ | Does citation generation affect accuracy or trust? |
| 6 | Semantic + Hybrid + BGE + Stuff + Claude Sonnet | Provider swap | Cross-provider comparison, same retrieval |
| 7 | Semantic + Hybrid + BGE + Query rewrite + Stuff + GPT-4o | Query transform added | Does rewriting the query before search help? |
| 8 | Semantic + Hybrid + BGE + HyDE + Stuff + GPT-4o | Different query transform | Does HyDE beat simple rewrite? |
| 9 | Parent-child + Hybrid + BGE + Stuff + GPT-4o | Chunking strategy swap | Does richer parent context improve answers? |
| 10 | Semantic + Hybrid + BGE + CRAG + GPT-4o | Advanced pattern | Does self-correcting retrieval justify the overhead? |

---

## 5. Evaluation Metrics

Every experiment produces the same `ExperimentResult` with these metrics:

### Quality Metrics (per query, then averaged)

| Metric | What It Measures | How It's Computed |
|--------|-----------------|------------------|
| Answer Correctness | Does the answer match ground truth? | LLM-as-judge scores 0-1 |
| Faithfulness | Is the answer grounded in retrieved context? | LLM checks if claims trace to chunks |
| Context Precision | Are retrieved chunks relevant to the query? | LLM grades each chunk's relevance |
| Context Recall | Did retrieval find all needed information? | LLM checks if ground truth is covered |

### Operational Metrics (per query, then aggregated)

| Metric | What It Measures |
|--------|-----------------|
| Latency (p50, p95) | End-to-end response time |
| Cost per query | API spend (tokens × price) |
| Total tokens | Input + output across all LLM calls |
| Retrieval time vs generation time | Where is the bottleneck? |

---

## 6. Comparison Dashboard (Streamlit)

Three views, each answering a different question:

### View 1: Summary Table
"Which experiment won?"

Select 2-6 experiments → one row per experiment, all metrics as columns. Color-coded: green for best, red for worst.

### View 2: Per-Query Drill-Down
"Why did it win?"

Select a query → see side-by-side: what each pipeline retrieved, what it generated, where scores differ. This is where insight lives.

### View 3: Tradeoff Scatter
"Is the improvement worth the cost?"

X = cost/query, Y = correctness. Each dot is an experiment. Pareto frontier highlighted. Toggle Y-axis between correctness, faithfulness, latency.

---

## 7. Portfolio Write-Up Template

```markdown
# RAG Strategy Comparison: FMCG Product Knowledge QA

## Setup
- 30 questions about [domain], spanning factual, analytical, and multi-hop
- 10 pipeline configurations tested, each isolating one variable
- All evaluated on correctness, faithfulness, context precision, context recall

## Key Findings

### Retrieval matters more than the LLM
Upgrading GPT-4o-mini → GPT-4o improved correctness by X points.
Upgrading dense → hybrid+rerank improved correctness by Y points.
Better retrieval had [N]x more impact than a better model.

### Reranking is the highest-ROI addition
Adding BGE reranker improved precision by X points at only Y% cost increase.
This was the single biggest quality lever across all experiments.

### Query transforms help on hard questions, hurt on easy ones
Rewriting improved correctness on analytical questions by X points
but added unnecessary latency on simple factual lookups.
Adaptive routing (easy → no transform, hard → rewrite) would be the production approach.

### CRAG: impressive but not justified here
Self-correcting retrieval caught Z% of retrieval failures.
But it doubled latency and cost while improving aggregate correctness by only N points.
Justified for high-stakes domains (medical, legal). Overkill for product QA.

## Recommendation
For FMCG product QA: Semantic chunking + Hybrid retrieval + BGE reranker +
GPT-4o with citation-grounded generation. Best quality/cost tradeoff.

## Interactive Dashboard
[Link to Streamlit app]
```

---

## 8. Build Sequence

### Session 1: Scaffold + Naive RAG End-to-End (Day 1-2)

- [ ] Repo structure, `pyproject.toml`, `.env.example`
- [ ] `shared/llm_client.py` — base + Azure OpenAI implementation
- [ ] `shared/embeddings.py` — base + OpenAI text-embedding-3-small
- [ ] `shared/vector_store.py` — base + ChromaDB
- [ ] `shared/data_loader.py` — PDF + directory loader
- [ ] `shared/cost_tracker.py` + `shared/timer.py`
- [ ] `rag/chunking.py` — fixed-size strategy
- [ ] `rag/retrieval.py` — dense retrieval
- [ ] `rag/generation.py` — stuff strategy
- [ ] `rag/pipeline.py` — compose from config
- [ ] `evaluation/schemas.py` — all dataclasses
- [ ] `evaluation/metrics.py` — correctness + faithfulness
- [ ] `evaluation/evaluator.py` — run eval, produce ExperimentResult
- [ ] `experiments/runner.py` — config → run → evaluate → save
- [ ] Create test dataset (10-15 questions to start)
- [ ] **Run experiment #1 (baseline). Verify results JSON.**

### Session 2: Retrieval Strategies (Day 3-4)

- [ ] `rag/chunking.py` — add semantic chunking
- [ ] `rag/retrieval.py` — add BM25 sparse, hybrid with RRF, BGE reranker
- [ ] `shared/llm_client.py` — add Anthropic client
- [ ] Run experiments #2, #3, #4, #6
- [ ] **Verify all produce comparable ExperimentResult objects**

### Session 3: Query Transforms + Advanced (Day 5-6)

- [ ] `rag/query_transform.py` — rewrite, HyDE, decomposition
- [ ] `rag/generation.py` — add citation-grounded generation
- [ ] `rag/chunking.py` — add parent-child chunking
- [ ] `rag/advanced.py` — CRAG implementation
- [ ] Run experiments #5, #7, #8, #9, #10
- [ ] Expand dataset to 30 questions

### Session 4: Dashboard + Write-Up (Day 7-8)

- [ ] `dashboard/app.py` — main Streamlit app
- [ ] `dashboard/views/summary_table.py`
- [ ] `dashboard/views/query_drilldown.py`
- [ ] `dashboard/views/tradeoff_scatter.py`
- [ ] Write `comparisons/fmcg_product_qa.md`
- [ ] Write `docs/decision_framework.md`
- [ ] Polish `README.md` for public repo

**Total: ~8 working days. Portfolio-ready output.**

---

## 9. Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Python | 3.11+ | Ecosystem compatibility |
| Package mgmt | `uv` + `pyproject.toml` | Fast, modern |
| LLM SDK | `openai` + `anthropic` | Direct SDK, no wrappers |
| Embeddings | `openai` / `sentence-transformers` | Cloud + local |
| Vector store | ChromaDB | Zero config, good enough for experimentation |
| Sparse search | `rank_bm25` | Lightweight BM25 |
| Reranking | `FlagEmbedding` | Open-source BGE reranker |
| Dashboard | Streamlit | Fast, good-looking, shareable |
| Config | PyYAML + Pydantic | Type-safe config loading |
| Logging | `structlog` | Structured, filterable |
| Testing | `pytest` | Standard |

---

## 10. What This Gets You

**For portfolio:** A public repo with ~20 files, a live Streamlit dashboard, and a comparison write-up that proves systematic AI evaluation skills. Someone reviewing it sees: clean architecture, real metrics, defensible recommendations.

**For projects:** A reusable pipeline where you change a YAML config to test a new strategy on a new dataset. When a real use case appears, you add a dataset and run experiments — the infrastructure is already there.

**For interviews:** "I tested 10 RAG configurations on an FMCG dataset and found that retrieval quality matters 3x more than model choice. Here's the dashboard." That's a conversation starter, not a code walkthrough.

---

## 11. Future Extensions (Only If Needed)

These are documented so you know where to go, but **do not build these until a specific use case demands it:**

- Agentic sandbox (ReAct, plan-and-execute, multi-agent) — add when you need RAG vs Agent comparison
- Multi-modal RAG (images, tables) — add when the R&D formulation use case requires it
- Graph RAG — add when entity-relationship queries dominate
- Caching layer — add when you're running the same queries repeatedly in production
- FastAPI deployment wrapper — add when someone needs a live API, not a dashboard
