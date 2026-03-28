# CLAUDE.md — RAG Sandbox

## What This Project Is

A RAG comparison engine. Ten experiments, each changing one variable. Five analysis notebooks that go beyond accuracy scores to explain *why* strategies work or fail. A Streamlit dashboard. A write-up.

The deliverable is the comparison — not the code. LangChain handles the plumbing. The value is in the analysis.

---

## Architecture Rules

1. **LangChain is the plumbing. Analysis is the product.** Don't build what LangChain provides. Spend time on the notebooks.
2. **Config-driven experiments.** Every experiment is a YAML file → LangChain components → `ExperimentResult` JSON.
3. **Evaluation is first-class.** RAGAS when API keys are available, LLM judge second, token-overlap heuristic as offline fallback. Evaluation never disappears.
4. **LangGraph for stateful patterns only.** CRAG uses LangGraph because it's a genuine state machine with conditional retry. Everything else is LCEL.
5. **No unnecessary abstractions.** Don't wrap LangChain in another class. `pipeline.py` exists solely to capture timing and cost metadata alongside the answer.

---

## Project Structure

```
rag-sandbox/
├── rag/
│   ├── pipeline.py          # Config → Pipeline; answer_query() → PipelineQueryOutput
│   ├── chains.py            # LCEL chains (stuff, citation) + LangGraph CRAG
│   └── config_loader.py     # YAML keys → LangChain objects (LLM, embeddings, retriever)
├── evaluation/
│   ├── schemas.py           # ExperimentResult, QueryResult, QueryMetrics — NEVER CHANGE
│   ├── metrics.py           # compute_query_metrics: RAGAS path + heuristic fallback
│   ├── llm_judge.py         # LLM-as-judge (4 prompts) + token-overlap heuristic
│   └── evaluator.py         # evaluate_experiment() → ExperimentResult
├── experiments/
│   ├── runner.py            # CLI: --config / --all / --ragas
│   ├── configs/             # 10 YAML experiment configs
│   ├── datasets/            # fmcg_product_qa.jsonl (10 Q&A pairs, expand to 30)
│   └── results/             # Auto-generated ExperimentResult JSONs
├── analysis/                # THE PORTFOLIO
│   ├── 01_embedding_space.ipynb
│   ├── 02_retrieval_failures.ipynb
│   ├── 03_chunk_boundary_analysis.ipynb
│   ├── 04_cost_quality_pareto.ipynb
│   ├── 05_experiment_comparison.ipynb
│   ├── utils.py             # load_results, results_to_summary_df, pareto_frontier, etc.
│   └── outputs/             # JSON/CSV written by notebooks, consumed by dashboard
├── dashboard/
│   ├── app.py               # 3-tab Streamlit app
│   ├── components.py        # load_result_files, build_summary_dataframe
│   └── views/               # summary_table, query_drilldown, tradeoff_scatter,
│                            # embedding_explorer, failure_analysis
├── data/fmcg_docs/          # Source documents (.txt / .md / .pdf)
├── comparisons/             # Write-ups (fill with notebook findings)
├── pyproject.toml
└── .env.example
```

**No `shared/` directory.** It was deleted in the LangChain migration.

---

## Critical Interfaces

### YAML Config Format

```yaml
experiment_id: "01_baseline_naive"
description: "..."

dataset:
  path: "experiments/datasets/fmcg_product_qa.jsonl"
  format: "question_answer"

pipeline:
  data_source: "data/fmcg_docs"
  llm:       { provider: "azure_openai", model: "gpt-4o-mini", temperature: 0.1, max_tokens: 400 }
  embedding: { provider: "azure_openai", model: "text-embedding-3-small" }
  chunking:  { strategy: "fixed", chunk_size: 128, overlap: 20 }   # fixed | semantic | parent_child
  retrieval: { strategy: "dense", top_k: 3 }                       # dense | sparse | hybrid
  reranker:  { enabled: false }
  query_transform: "none"    # none | rewrite | hyde
  generation: "stuff"        # stuff | citation_grounded
  advanced: "none"           # none | crag

evaluation:
  metrics: [answer_correctness, faithfulness, context_precision, context_recall]
  llm_judge:
    provider: "azure_openai"
    model: "gpt-4o"
```

### Pipeline Output (adapter between LangChain and evaluation)

```python
# rag/pipeline.py

@dataclass
class SearchResult:
    text: str
    score: float      # rank-based: 1/rank (LangChain doesn't surface similarity scores)
    metadata: dict

@dataclass
class PipelineQueryOutput:
    answer: str
    retrieved_results: list[SearchResult]   # evaluator reads .text and .score
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens: int    # from get_openai_callback(); 0 for Anthropic
    cost_usd: float
    metadata: dict
```

### Evaluation Schema (THE BACKBONE — never change)

```python
# evaluation/schemas.py

@dataclass
class ExperimentResult:
    experiment_id: str
    pipeline_config: dict
    timestamp: str
    dataset_id: str
    per_query_results: list[QueryResult]
    aggregate_metrics: AggregateMetrics
```

Every pipeline run must produce an `ExperimentResult`. The dashboard and notebooks consume only this schema.

---

## The 10 Experiments

| # | Config | What It Tests |
|---|--------|---------------|
| 1 | `01_baseline_naive` | Fixed + Dense + Stuff + GPT-4o-mini (cheapest baseline) |
| 2 | `02_better_llm` | Same but GPT-4o — does LLM quality fix weak retrieval? |
| 3 | `03_hybrid_retrieval` | Semantic + Hybrid — retrieval upgrade |
| 4 | `04_hybrid_rerank` | Add cross-encoder reranking |
| 5 | `05_citation_grounded` | Citation-grounded generation |
| 6 | `06_cross_provider` | Claude Sonnet instead of GPT-4o |
| 7 | `07_query_rewrite` | Query rewriting before retrieval |
| 8 | `08_hyde` | HyDE instead of rewrite |
| 9 | `09_parent_child` | Parent-child chunking |
| 10 | `10_crag` | CRAG self-correction via LangGraph |

Each changes **one variable** from its neighbor. This isolates what drives improvement.

---

## Analysis Notebooks (THE PRIORITY)

Notebooks must read like a data science report — markdown cells explaining what's being measured and why, code cells producing findings, visualizations that tell a story.

| Notebook | What It Answers |
|----------|-----------------|
| `01_embedding_space` | Do product categories cluster? Where does retrieval fail by design? |
| `02_retrieval_failures` | Why does retrieval fail? Which strategies fix which failure types? |
| `03_chunk_boundary_analysis` | Does chunking split answers across boundaries? Which strategy preserves information best? |
| `04_cost_quality_pareto` | Which experiments are Pareto-optimal? What's the marginal cost of each quality gain? |
| `05_experiment_comparison` | Which single change drives the most improvement? What's the headline finding? |

Each notebook writes outputs to `analysis/outputs/`. The dashboard's **Embedding Space** and **Failure Analysis** tabs consume those files.

---

## Coding Standards

- **Python 3.11+.** Type hints in `.py` files. Notebooks can be looser.
- **No print statements in `.py` files.** Use `structlog`.
- **Don't wrap LangChain.** If LangChain has a method, call it. `pipeline.py` is the only wrapper and it exists only to capture timing/cost.
- **Notebooks are analysis, not scripts.** No bare loops that do 10 things — one cell, one finding.
- **Evaluation schema is sacred.** `ExperimentResult` in `evaluation/schemas.py` never changes. Every pipeline adapts to produce it.

---

## Commands

```bash
# Run one experiment
python -m experiments.runner --config experiments/configs/01_baseline_naive.yaml

# Run all 10
python -m experiments.runner --all

# Run all with RAGAS scoring (requires API keys, costs more, more accurate)
python -m experiments.runner --all --ragas

# Launch dashboard
streamlit run dashboard/app.py

# Run analysis notebooks
jupyter notebook analysis/
```

---

## Environment Variables

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o
AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-small

ANTHROPIC_API_KEY=...        # experiment 06 only

CHROMA_PERSIST_DIR=./data/chroma
```

---

## What Not To Do

- **Don't build what LangChain provides.** If you're writing a custom retriever, stop and check LangChain first.
- **Don't skip the notebooks.** The analysis is the portfolio. Without it, this is just another LangChain demo.
- **Don't change `evaluation/schemas.py`.** The dashboard, notebooks, and result JSONs all depend on this schema.
- **Don't add experiments beyond the 10.** Add questions to the dataset instead.
- **Don't add features for future use.** If it's not needed for the current 10 experiments, it doesn't exist.
