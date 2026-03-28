# Refactoring Plan: LangChain Migration + Analysis Layer

## What You Have Today

~1,250 lines of custom Python across 20 files. The breakdown:

| Layer | Files | Lines | What It Does |
|-------|-------|-------|-------------|
| `shared/` | 6 files | ~626 | LLM clients, embeddings, vector store, data loading, cost tracking, timing |
| `rag/` | 6 files | ~627 | Chunking, retrieval, query transforms, generation, advanced patterns, pipeline |
| `evaluation/` | 4 files | ~200 | Schemas, metrics, LLM judge, evaluator |
| `experiments/` | 1 file | ~70 | Runner CLI |
| `dashboard/` | 5 files | ~120 | Streamlit comparison views |

Plus 10 YAML configs, 10 result JSONs, 1 dataset (10 questions), and sample FMCG docs.

---

## The Problem

~1,250 lines is custom plumbing for things LangChain already provides. The entire `shared/` layer (LLM clients, embeddings, vector stores) and most of `rag/` (chunking, retrieval, generation) are reimplementations of LangChain primitives. This makes the repo look like a software engineering exercise, not an AI/data science project.

**What to cut:** `shared/` and `rag/` — replace with LangChain + LangGraph.

**What to keep:** `evaluation/`, `experiments/`, `dashboard/` — this is where the analytical value lives. Enhance it significantly.

**What to add:** An `analysis/` layer with notebooks that do the actual data science work — embedding space analysis, retrieval failure taxonomy, cost-quality Pareto frontiers.

---

## Target State

```
ai-sandbox/
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py            # REWRITE: LangChain LCEL chains, ~80 lines
│   ├── chains.py              # NEW: Pre-built chain variants (stuff, citation, CRAG)
│   └── config_loader.py       # REWRITE: YAML → LangChain components, ~60 lines
│
├── evaluation/                # KEEP + ENHANCE
│   ├── __init__.py
│   ├── schemas.py             # KEEP as-is
│   ├── metrics.py             # ENHANCE: add RAGAS integration
│   ├── llm_judge.py           # ENHANCE: real LLM judge + heuristic fallback
│   └── evaluator.py           # MINOR EDIT: adapt to new pipeline interface
│
├── experiments/               # KEEP
│   ├── runner.py              # MINOR EDIT: new pipeline builder
│   ├── configs/               # KEEP all 10 YAMLs (simplify format)
│   ├── datasets/              # ENHANCE: expand to 30 questions
│   └── results/               # KEEP all result JSONs
│
├── analysis/                  # NEW — this is the portfolio value
│   ├── 01_embedding_space.ipynb        # Embedding visualization + cluster analysis
│   ├── 02_retrieval_failures.ipynb     # Failure taxonomy + classification
│   ├── 03_chunk_boundary_analysis.ipynb # Information loss at chunk boundaries
│   ├── 04_cost_quality_pareto.ipynb    # Pareto frontier + marginal cost analysis
│   ├── 05_experiment_comparison.ipynb  # Master comparison with statistical tests
│   └── utils.py                        # Shared plotting + data loading helpers
│
├── dashboard/                 # KEEP + ENHANCE
│   ├── app.py
│   ├── components.py
│   └── views/
│       ├── summary_table.py
│       ├── query_drilldown.py
│       ├── tradeoff_scatter.py
│       ├── embedding_explorer.py   # NEW: interactive embedding viz
│       └── failure_analysis.py     # NEW: retrieval failure breakdown
│
├── data/
│   └── fmcg_docs/             # KEEP
│
├── comparisons/
│   └── fmcg_product_qa.md     # REWRITE: fill with real findings
│
├── pyproject.toml             # UPDATE: swap deps
├── .env.example               # KEEP
├── CLAUDE.md                  # REWRITE
└── README.md                  # REWRITE
```

**Deleted entirely:** `shared/` (all 6 files). LangChain replaces every one of them.

---

## File-by-File Change Plan

### DELETE — `shared/` (entire directory)

| File | Lines | Replaced By |
|------|-------|-------------|
| `shared/llm_client.py` | 194 | `langchain_openai.AzureChatOpenAI`, `langchain_anthropic.ChatAnthropic` |
| `shared/embeddings.py` | 119 | `langchain_openai.AzureOpenAIEmbeddings`, `langchain_community.embeddings.HuggingFaceEmbeddings` |
| `shared/vector_store.py` | 146 | `langchain_chroma.Chroma`, `langchain_community.vectorstores.FAISS` |
| `shared/data_loader.py` | 85 | `langchain_community.document_loaders` (PyPDFLoader, CSVLoader, DirectoryLoader) |
| `shared/cost_tracker.py` | 51 | `langchain_community.callbacks.get_openai_callback` + custom wrapper |
| `shared/timer.py` | 31 | Keep as a tiny utility in `rag/utils.py` or inline |

**Net deletion: ~626 lines of plumbing.**

### REWRITE — `rag/pipeline.py` + new `rag/chains.py` + `rag/config_loader.py`

**Current `rag/pipeline.py`** (193 lines): custom Pipeline dataclass with manual composition of chunking → retrieval → generation → advanced patterns. Delete this entirely.

**New `rag/config_loader.py`** (~60 lines): reads YAML config, returns LangChain components:

```python
# What this file does (pseudocode)
def load_from_config(config: dict):
    llm = AzureChatOpenAI(...) or ChatAnthropic(...)  # from config
    embeddings = AzureOpenAIEmbeddings(...)             # from config
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    if config reranker enabled:
        retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=FlashrankRerank() or CrossEncoderReranker()
        )

    if config retrieval strategy == "hybrid":
        bm25_retriever = BM25Retriever.from_documents(docs)
        retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[alpha, 1-alpha]
        )

    return llm, retriever, embeddings
```

**New `rag/chains.py`** (~80 lines): pre-built LCEL chain variants:

```python
# Stuff chain
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Citation-grounded chain — same but different prompt
# CRAG chain — uses LangGraph for the retry loop
```

**New `rag/pipeline.py`** (~80 lines): thin wrapper that builds a chain from config and exposes a `run(query) -> PipelineOutput` method that the experiment runner calls. Captures latency, tokens, cost alongside the answer.

### DELETE — `rag/chunking.py`, `rag/retrieval.py`, `rag/generation.py`, `rag/query_transform.py`

| File | Lines | Replaced By |
|------|-------|-------------|
| `rag/chunking.py` | 84 | `langchain.text_splitter` (RecursiveCharacterTextSplitter, SemanticChunker, ParentDocumentRetriever) |
| `rag/retrieval.py` | 142 | `langchain` retrievers (vectorstore, BM25, EnsembleRetriever) + `ContextualCompressionRetriever` for reranking |
| `rag/generation.py` | 76 | LCEL chains with different prompts |
| `rag/query_transform.py` | 69 | `langchain.retrievers.MultiQueryRetriever` or custom LCEL chain for HyDE/decomposition |

**Net deletion: ~371 lines replaced by ~50 lines of LangChain config.**

### REWRITE — `rag/advanced.py` → LangGraph

Current `advanced.py` (63 lines) has simple threshold-based CRAG and adaptive routing. Replace with proper LangGraph state machines:

```python
# CRAG as a LangGraph StateGraph
class CRAGState(TypedDict):
    query: str
    documents: list[Document]
    generation: str
    retry_count: int

def grade_documents(state): ...   # LLM grades retrieval quality
def generate(state): ...          # Generate from good docs
def rewrite_query(state): ...     # Transform query for retry
def decide_to_retry(state): ...   # Conditional edge

graph = StateGraph(CRAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_conditional_edges("grade", decide_to_retry, {"generate": "generate", "rewrite": "rewrite"})
```

This is where LangGraph genuinely adds value — the CRAG and adaptive patterns are stateful decision loops, not linear chains. Using LangGraph here is defensible, not just wrapper code.

### KEEP — `evaluation/schemas.py`

No changes. This is the backbone — every pipeline still produces `ExperimentResult`.

### ENHANCE — `evaluation/metrics.py` + `evaluation/llm_judge.py`

**Current state:** `llm_judge.py` uses token-overlap heuristics only. `metrics.py` is a thin wrapper.

**Changes:**

Add RAGAS integration as an optional path. When API keys are available, use RAGAS `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` metrics. Fall back to current heuristics when offline.

```python
# evaluation/metrics.py — add RAGAS path
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def compute_query_metrics(...):
    if RAGAS_AVAILABLE and use_ragas:
        # Use RAGAS evaluation
        ...
    else:
        # Fall back to existing heuristic judge
        ...
```

Also enhance `llm_judge.py` to actually use the LLM when available (currently the `llm_client` parameter is accepted but never used for scoring).

### MINOR EDIT — `evaluation/evaluator.py`

Adapt `evaluate_experiment` to accept the new pipeline interface. The core logic (loop over dataset, compute metrics, aggregate) stays identical. Only the pipeline call signature changes.

### MINOR EDIT — `experiments/runner.py`

Change `build_pipeline(config)` import to new location. Everything else stays.

### SIMPLIFY — `experiments/configs/*.yaml`

Configs get simpler because LangChain handles defaults:

```yaml
# Before (current)
pipeline:
  chunking:
    strategy: "fixed"
    chunk_size: 128
    overlap: 20
  embedding:
    model: "text-embedding-3-small"
    provider: "azure_openai"
  vector_store:
    provider: "chroma"
    collection: "baseline_naive"
  retrieval:
    strategy: "dense"
    top_k: 3
    alpha: 0.5
  reranker:
    enabled: false
    model: "bge-reranker-v2-m3"
    top_k_after_rerank: 3
  query_transform:
    strategy: "none"
  generation:
    strategy: "stuff"
    llm:
      provider: "azure_openai"
      model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 400
  advanced:
    pattern: "none"

# After (LangChain-backed)
pipeline:
  llm: { provider: "azure_openai", model: "gpt-4o-mini", temperature: 0.1 }
  embedding: { provider: "azure_openai", model: "text-embedding-3-small" }
  chunking: { strategy: "fixed", chunk_size: 512, overlap: 50 }
  retrieval: { strategy: "dense", top_k: 3 }
  reranker: { enabled: false }
  query_transform: "none"
  generation: "stuff"
  advanced: "none"
```

### KEEP + ENHANCE — `dashboard/`

Keep all existing views. Add two new views:

**`dashboard/views/embedding_explorer.py`** — interactive embedding visualization. Load pre-computed UMAP/t-SNE projections from analysis notebooks, display in Streamlit with plotly. Color by cluster, source doc, or retrieval success/failure.

**`dashboard/views/failure_analysis.py`** — retrieval failure breakdown. Load failure taxonomy from analysis notebooks, show distribution of failure types per experiment, drill down to specific failing queries.

---

## NEW — `analysis/` (The Portfolio Differentiator)

This is where the repo stops being a software project and becomes a data science project. Each notebook produces findings that go into the comparison write-up and dashboard.

### `01_embedding_space.ipynb` — Embedding Space Analysis

**What it does:**
- Embed the FMCG corpus with 2-3 models (text-embedding-3-small, text-embedding-3-large, BGE-base)
- Compute UMAP/t-SNE 2D projections
- Visualize clusters — do product categories form natural clusters? Where do chunks from different products overlap?
- Measure inter-cluster vs intra-cluster distances
- Identify "problem zones" where semantically different chunks land near each other
- Compare embedding models: which one separates your domain concepts best?

**Key outputs:**
- Embedding space scatter plots (colored by product category, by source doc)
- Distance distribution histograms (same-product vs different-product chunk distances)
- "Confusion zones" — chunk pairs that are semantically different but embedded close together

**Why it matters for portfolio:** This is the kind of analysis that shows you understand *why* retrieval works or fails, not just that it does. Nobody with a RAG framework repo does this.

### `02_retrieval_failures.ipynb` — Retrieval Failure Taxonomy

**What it does:**
- Load per-query results from all 10 experiments
- For queries where context_precision or context_recall is low, classify the failure mode:
  - **Wrong chunk retrieved** — lexical overlap or embedding proximity pulled an irrelevant chunk
  - **Right topic, wrong detail** — chunk is about the right product but doesn't contain the specific fact
  - **Information split across chunks** — answer requires combining information from 2+ chunks that weren't both retrieved
  - **Query ambiguity** — the query could match multiple products/contexts
  - **Vocabulary mismatch** — query uses different terms than the source document
- Count failure types per experiment
- Identify which strategies fix which failure types

**Key outputs:**
- Failure type distribution bar charts per experiment
- Confusion matrix: failure type × strategy effectiveness
- Specific failing query examples with annotated explanations

**Why it matters:** "Hybrid retrieval reduces vocabulary mismatch failures by 60% but doesn't help with information-split failures — that requires query decomposition" is a 10x more useful finding than "hybrid scored 0.82 vs dense 0.71."

### `03_chunk_boundary_analysis.ipynb` — Information Loss at Boundaries

**What it does:**
- For each ground truth answer, identify where in the source docs the supporting facts live
- Measure: does the chunking strategy keep the supporting facts in one chunk, or split them?
- Compute "information completeness" per chunk — what fraction of the ground truth answer can be derived from a single chunk?
- Compare across chunking strategies: fixed vs semantic vs parent-child
- Plot information completeness curves as a function of chunk size

**Key outputs:**
- Information completeness distribution per chunking strategy
- "Split fact" examples — where a key fact spans a chunk boundary
- Optimal chunk size curve for this corpus

**Why it matters:** This is the demand-decomposition mindset applied to chunking. It's empirical, it's quantitative, and it directly explains why certain chunking strategies outperform others.

### `04_cost_quality_pareto.ipynb` — Pareto Frontier Analysis

**What it does:**
- Plot all 10 experiments on cost vs correctness
- Identify the Pareto frontier (experiments where you can't improve quality without increasing cost)
- Compute marginal cost of improvement — "going from 0.75 to 0.85 correctness costs X, but 0.85 to 0.90 costs 3X"
- Sensitivity analysis: which cost component dominates? (embedding calls, retrieval, generation, reranking)
- Compute "efficiency score" = quality / cost for each experiment
- Project: what would it cost to run each strategy at 1000 queries/day?

**Key outputs:**
- Pareto frontier plot with experiment labels
- Marginal cost curve
- Cost breakdown treemaps per experiment
- Production cost projection table

**Why it matters:** This is the throughput maximisation / pricing optimization thinking from POINT, applied to RAG. It shows you think about AI in business terms, not just accuracy terms.

### `05_experiment_comparison.ipynb` — Master Comparison

**What it does:**
- Load all 10 experiment results
- Statistical significance tests: are the differences between experiments meaningful or within noise?
- Stratified analysis: break results by query difficulty (easy factual vs analytical vs multi-hop)
- Ablation summary: for each variable (chunking, retrieval, reranking, LLM, query transform, advanced pattern), compute the average impact across all experiments where it was the changed variable
- Rank variables by impact on each metric

**Key outputs:**
- Variable importance ranking (which lever matters most for correctness? for faithfulness? for cost?)
- Stratified performance table (strategy × query type)
- Statistical significance heatmap
- The "headline finding" — one sentence that captures the most important insight

**Why it matters:** This is the notebook that generates the write-up content. The comparison story writes itself from these outputs.

---

## Dependency Changes

### Remove
```
# No longer needed — LangChain handles these
chromadb          # accessed via langchain_chroma
rank-bm25         # accessed via langchain BM25Retriever
sentence-transformers  # accessed via langchain_community
pypdf             # accessed via langchain_community PyPDFLoader
```

### Add
```
langchain>=0.3
langchain-openai>=0.3
langchain-anthropic>=0.3
langchain-chroma>=0.2
langchain-community>=0.3
langgraph>=0.2

# Analysis
ragas>=0.2
umap-learn>=0.5
scikit-learn>=1.4
plotly>=5.20
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
```

### Keep
```
openai            # still needed as langchain-openai dependency
anthropic         # still needed as langchain-anthropic dependency
pandas
numpy
pydantic
pyyaml
streamlit
structlog
pytest
```

---

## Updated `pyproject.toml`

```toml
[project]
name = "ai-sandbox"
version = "0.2.0"
description = "Config-driven RAG comparison engine with analytical deep dives."
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-chroma>=0.2",
    "langchain-community>=0.3",
    "langgraph>=0.2",
    "ragas>=0.2",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pydantic>=2.7.0",
    "pyyaml>=6.0.1",
    "streamlit>=1.44.0",
    "structlog>=24.1.0",
    "plotly>=5.20",
    "umap-learn>=0.5",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

[project.optional-dependencies]
dev = ["pytest>=8.1.0", "jupyter>=1.0"]
```

---

## Line Count Comparison

| Layer | Before | After | Change |
|-------|--------|-------|--------|
| `shared/` | ~626 lines | 0 (deleted) | -626 |
| `rag/` | ~627 lines | ~220 lines | -407 |
| `evaluation/` | ~200 lines | ~280 lines | +80 (RAGAS integration) |
| `experiments/` | ~70 lines | ~70 lines | 0 |
| `dashboard/` | ~120 lines | ~200 lines | +80 (2 new views) |
| `analysis/` | 0 | ~800 lines across 5 notebooks | +800 |
| **Total** | **~1,643** | **~1,570** | **Similar total, but 50% is now analysis** |

The ratio flips from **80% plumbing / 20% analysis** to **30% plumbing / 70% analysis**.

---

## Build Sequence for Claude Code

### Session 1: LangChain Migration
- Delete `shared/` entirely
- Rewrite `rag/config_loader.py` — YAML → LangChain components
- Rewrite `rag/chains.py` — LCEL chains for stuff, citation-grounded
- Rewrite `rag/pipeline.py` — thin wrapper producing same `PipelineOutput`
- Implement CRAG as LangGraph StateGraph in `rag/chains.py`
- Update `experiments/runner.py` to use new pipeline
- Update `evaluation/evaluator.py` for new interface
- Verify: experiment #1 still runs and produces valid `ExperimentResult`
- Run all 10 experiments, verify results

### Session 2: Evaluation Enhancement
- Add RAGAS integration to `evaluation/metrics.py`
- Enhance `evaluation/llm_judge.py` with real LLM scoring
- Re-run experiments with RAGAS metrics
- Expand dataset to 30 questions

### Session 3: Analysis Notebooks
- `01_embedding_space.ipynb`
- `02_retrieval_failures.ipynb`
- `03_chunk_boundary_analysis.ipynb`
- `04_cost_quality_pareto.ipynb`
- `05_experiment_comparison.ipynb`

### Session 4: Dashboard Enhancement + Write-Up
- Add `embedding_explorer.py` and `failure_analysis.py` views
- Write `comparisons/fmcg_product_qa.md` with real findings from notebooks
- Polish README for public repo

---

## What the Portfolio Now Shows

**Before (software engineering):** "I built a config-driven RAG pipeline with abstract interfaces, strategy registries, and swappable components."

**After (AI/data science):** "I tested 10 RAG strategies on FMCG data. I found that embedding models cluster product categories well but confuse usage instructions with product descriptions — causing 40% of retrieval failures. Hybrid retrieval fixes vocabulary mismatch but not information-split failures. Reranking is the single highest-ROI lever. Here's the Pareto frontier showing diminishing returns above $0.01/query."

That second paragraph is what gets remembered in an interview. The LangChain plumbing is invisible infrastructure. The analysis is the differentiator.
