# RAG Strategy Comparison — FMCG Product Q&A

A systematic evaluation of 10 RAG configurations on an FMCG product knowledge base. Each experiment changes one variable so the effect of every upgrade is isolated and measurable. The result is a set of evidence-backed answers to the questions that actually matter when building RAG in production.

Built with LangChain, LangGraph, RAGAS, and a Streamlit comparison dashboard.

---

## What This Answers

- Does upgrading from GPT-4o-mini to GPT-4o improve answers when retrieval is weak?
- How much does hybrid retrieval improve over dense-only?
- Is reranking worth the added latency and cost?
- Which query transformation strategy — rewrite or HyDE — works better on this domain?
- Does CRAG self-correction via LangGraph justify its overhead?
- How do GPT-4o and Claude Sonnet compare on the same retrieval stack?

---

## The 10 Experiments

Each experiment changes **exactly one variable** from its predecessor.

| # | Experiment | Change From Previous |
|---|------------|----------------------|
| 1 | `01_baseline_naive` | Fixed chunking · dense retrieval · GPT-4o-mini · no reranking |
| 2 | `02_better_llm` | GPT-4o instead of GPT-4o-mini |
| 3 | `03_hybrid_retrieval` | Semantic chunking + hybrid BM25+dense retrieval |
| 4 | `04_hybrid_rerank` | Cross-encoder reranking added |
| 5 | `05_citation_grounded` | Citation-grounded generation prompt |
| 6 | `06_cross_provider` | Claude Sonnet 3.5 instead of GPT-4o |
| 7 | `07_query_rewrite` | LLM query rewriting before retrieval |
| 8 | `08_hyde` | HyDE (hypothetical document) instead of rewrite |
| 9 | `09_parent_child` | Parent-child chunking strategy |
| 10 | `10_crag` | CRAG self-correction loop via LangGraph |

---

## Analysis Notebooks

Beyond accuracy scores, five notebooks explain the *mechanisms* behind what works and what doesn't.

**[01 — Embedding Space](analysis/01_embedding_space.ipynb)**
UMAP visualization of the full corpus. Identifies confusion zones where semantically different chunks land near each other — predicting where dense retrieval will fail before running a single query.

**[02 — Retrieval Failure Taxonomy](analysis/02_retrieval_failures.ipynb)**
Classifies every low-scoring retrieval into one of five failure modes: vocabulary mismatch, information split across chunk boundaries, right topic wrong detail, query ambiguity, or wrong chunk. Shows which strategies fix which failure types.

**[03 — Chunk Boundary Analysis](analysis/03_chunk_boundary_analysis.ipynb)**
Measures *information completeness* — the maximum fraction of a ground-truth answer contained in a single chunk. Compares fixed, semantic, and parent-child chunking. Quantifies how often boundary splits make answers unretrievable regardless of retrieval strategy.

**[04 — Cost-Quality Pareto Frontier](analysis/04_cost_quality_pareto.ipynb)**
Plots all 10 experiments on cost vs. quality. Identifies Pareto-optimal configurations. Computes the marginal cost of each quality increment and projects production costs at 1000 queries/day.

**[05 — Master Comparison](analysis/05_experiment_comparison.ipynb)**
Variable importance ranking — which single change drives the most improvement? Radar charts, latency-quality tradeoffs, and the headline finding synthesized across all 10 runs.

---

## Evaluation Metrics

Every experiment produces four metrics per query, aggregated across the full test set:

| Metric | What It Measures |
|--------|-----------------|
| **Answer Correctness** | Semantic match between generated answer and ground truth |
| **Faithfulness** | Whether the answer is grounded in retrieved context (no hallucination) |
| **Context Precision** | Fraction of retrieved chunks that were actually relevant |
| **Context Recall** | Fraction of relevant information that was retrieved |

Scoring uses [RAGAS](https://github.com/explodinggradients/ragas) LLM-as-judge when API keys are available, with a token-overlap heuristic fallback for offline development.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Three tabs:
- **Experiment Results** — sortable metric table, per-query drilldown, cost-quality scatter
- **Embedding Space** — interactive UMAP explorer (run notebook 01 first)
- **Failure Analysis** — retrieval failure breakdown by type and experiment (run notebook 02 first)

---

## Quick Start

**1. Install dependencies**
```bash
pip install -e ".[dev]"
```

**2. Set credentials**
```bash
cp .env.example .env
# Fill in AZURE_OPENAI_* and ANTHROPIC_API_KEY
```

**3. Add source documents**
Drop `.txt`, `.md`, or `.pdf` files into `data/fmcg_docs/`.

**4. Run experiments**
```bash
# Single experiment
python -m experiments.runner --config experiments/configs/01_baseline_naive.yaml

# All 10
python -m experiments.runner --all

# All 10 with RAGAS scoring (more accurate, requires live API keys)
python -m experiments.runner --all --ragas
```

**5. Run analysis notebooks**
```bash
jupyter notebook analysis/
```
Run them in order — each writes outputs consumed by later notebooks and the dashboard.

**6. Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

---

## Project Layout

```
├── rag/
│   ├── pipeline.py          # Config → Pipeline → PipelineQueryOutput
│   ├── chains.py            # LCEL chains (stuff, citation) + LangGraph CRAG
│   └── config_loader.py     # YAML → LangChain LLM / embeddings / retriever
├── evaluation/
│   ├── schemas.py           # ExperimentResult schema (backbone of everything)
│   ├── metrics.py           # RAGAS + heuristic scoring
│   ├── llm_judge.py         # LLM-as-judge with 4 metric prompts
│   └── evaluator.py         # Runs pipeline over dataset → ExperimentResult
├── experiments/
│   ├── runner.py            # CLI entry point
│   ├── configs/             # 10 YAML experiment configs
│   ├── datasets/            # fmcg_product_qa.jsonl
│   └── results/             # Generated ExperimentResult JSONs
├── analysis/
│   ├── 01_embedding_space.ipynb
│   ├── 02_retrieval_failures.ipynb
│   ├── 03_chunk_boundary_analysis.ipynb
│   ├── 04_cost_quality_pareto.ipynb
│   ├── 05_experiment_comparison.ipynb
│   └── utils.py
├── dashboard/
│   ├── app.py
│   └── views/
├── data/fmcg_docs/          # Source documents
└── comparisons/             # Write-ups with findings
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Azure OpenAI (GPT-4o, GPT-4o-mini) · Anthropic (Claude Sonnet 3.5) |
| Embeddings | Azure OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (via LangChain Community) |
| Reranking | Cross-encoder ms-marco-MiniLM (via sentence-transformers) |
| RAG framework | LangChain + LangGraph |
| Evaluation | RAGAS + custom LLM judge |
| Dashboard | Streamlit + Plotly |
| Analysis | pandas · NumPy · UMAP · scikit-learn · seaborn |

---

## Environment Variables

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o
AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-small

ANTHROPIC_API_KEY=sk-ant-...    # required for experiment 06 only

CHROMA_PERSIST_DIR=./data/chroma
```

See `.env.example` for the full template.
