# RAG Strategy Comparison - FMCG Product Q&A

A config-driven RAG comparison sandbox for testing retrieval and generation strategies on an FMCG product knowledge base.

Built with LangChain, LangGraph, Google Gemini, RAGAS, and a Streamlit dashboard.

## What This Answers

- Does upgrading from Gemini 2.5 Flash Lite to Gemini 2.5 Flash improve answers when retrieval is weak?
- How much does hybrid retrieval improve over dense-only?
- Is reranking worth the added latency?
- Which query transformation strategy works better here: rewrite or HyDE?
- Does CRAG self-correction justify its overhead?
- How does Gemini 2.0 Flash compare with Gemini 2.5 Flash on the same retrieval stack?

## The 10 Experiments

Each experiment changes one variable from its predecessor.

| # | Experiment | Change From Previous |
|---|------------|----------------------|
| 1 | `01_baseline_naive` | Fixed chunking + dense retrieval + Gemini 2.5 Flash Lite |
| 2 | `02_better_llm` | Gemini 2.5 Flash instead of Gemini 2.5 Flash Lite |
| 3 | `03_hybrid_retrieval` | Semantic chunking + hybrid BM25+dense retrieval |
| 4 | `04_hybrid_rerank` | Cross-encoder reranking added |
| 5 | `05_citation_grounded` | Citation-grounded generation prompt |
| 6 | `06_cross_provider` | Gemini 2.0 Flash instead of Gemini 2.5 Flash |
| 7 | `07_query_rewrite` | Query rewriting before retrieval |
| 8 | `08_hyde` | HyDE instead of rewrite |
| 9 | `09_parent_child` | Parent-child chunking |
| 10 | `10_crag` | CRAG self-correction loop |

## Quick Start

1. Install dependencies:

```bash
pip install -e ".[dev]"
```

2. Create `.env` from the template and set `GOOGLE_API_KEY`.

3. Add source documents under `data/fmcg_docs/`.

4. Run an experiment:

```bash
python -m experiments.runner --config experiments/configs/01_baseline_naive.yaml
```

5. Run all experiments:

```bash
python -m experiments.runner --all
```

6. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Environment

```bash
GOOGLE_API_KEY=your-key
CHROMA_PERSIST_DIR=./data/chroma
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini |
| Embeddings | Gemini Embeddings |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| RAG framework | LangChain + LangGraph |
| Evaluation | RAGAS + custom LLM judge |
| Dashboard | Streamlit + Plotly |
