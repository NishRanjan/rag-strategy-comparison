# RAG Component Reference

Every component in this project maps to a YAML config key, a Python implementation, and a testable hypothesis. This document is the single reference for all three.

---

## Table of Contents

1. [Chunking Strategies](#1-chunking-strategies)
2. [Embedding Model](#2-embedding-model)
3. [Vector Store](#3-vector-store)
4. [Retrieval Strategies](#4-retrieval-strategies)
5. [Reranker](#5-reranker)
6. [Query Transforms](#6-query-transforms)
7. [Generation Strategies](#7-generation-strategies)
8. [Advanced Patterns](#8-advanced-patterns)
9. [Evaluation Layer](#9-evaluation-layer)
10. [How Components Compose](#10-how-components-compose)

---

## 1. Chunking Strategies

Chunking is the first irreversible decision in the pipeline. Once documents are split, retrieval can only work with what the chunks contain. A chunk boundary that cuts across a key fact means no retrieval strategy can recover it.

**Config key:** `pipeline.chunking.strategy`

---

### 1.1 Fixed Chunking

**What it does**
Splits documents at character count boundaries using `RecursiveCharacterTextSplitter`. It tries paragraph breaks, then sentence breaks, then word breaks — in that order — before hard-splitting at the character limit. A small overlap window (`overlap` tokens) is repeated at the start of each new chunk to reduce boundary loss.

**Where it's implemented**
`rag/config_loader.py` → `chunk_documents()` → `strategy == "fixed"`
Uses: `langchain_text_splitters.RecursiveCharacterTextSplitter`

**Config parameters**
```yaml
chunking:
  strategy: "fixed"
  chunk_size: 128    # target characters per chunk
  overlap: 20        # characters repeated from previous chunk
```

**Hypothesis**
Fixed chunking is the simplest and cheapest baseline. The assumption is that paragraph-level structure in the source documents naturally aligns with information units — a product description, a financial metric, a policy clause — so character-count splitting approximates semantic splitting without any embedding cost.

**Where it breaks**
Fails when key facts span paragraphs (e.g. a table heading in one paragraph, values in the next). The overlap only partially mitigates this. Experiment 03 tests whether semantic chunking improves on this.

**Tested in:** Experiments 01, 02

---

### 1.2 Semantic Chunking

**What it does**
Embeds every sentence individually, then groups consecutive sentences into chunks by measuring the cosine distance between adjacent sentence embeddings. A new chunk boundary is created where the semantic shift exceeds a threshold. Chunks vary in length — a tight paragraph stays in one chunk, a section that shifts topic frequently produces many small chunks.

**Where it's implemented**
`rag/config_loader.py` → `chunk_documents()` → `strategy == "semantic"`
Uses: `langchain_experimental.text_splitter.SemanticChunker`
Falls back to fixed if `langchain_experimental` is unavailable.

**Config parameters**
```yaml
chunking:
  strategy: "semantic"
  chunk_size: 128    # used only for fallback
  overlap: 20        # used only for fallback
```

**Hypothesis**
If documents contain dense mixed-topic sections (e.g. a quarterly report paragraph covering revenue, then margins, then guidance), fixed chunking will bundle unrelated facts together. Semantic chunking produces chunks that each represent one coherent idea, improving precision — the retrieved chunk is more likely to be entirely about the right topic.

**The cost**
Semantic chunking requires an embedding call per sentence at index time. For a large corpus of annual reports, this is non-trivial. The hypothesis is that better precision justifies this one-time cost.

**Tested in:** Experiments 03–10 (as the standard chunking for the upgraded stack)

---

### 1.3 Parent-Child Chunking

**What it does**
Splits documents twice: first into large "parent" chunks (full `chunk_size`), then splits each parent into smaller "child" chunks (`chunk_size // 2`). Child chunks are what get embedded and indexed for retrieval. When a child chunk is retrieved, its parent's full text is stored in `metadata["parent_content"]` and returned alongside it.

**Where it's implemented**
`rag/config_loader.py` → `chunk_documents()` → `strategy == "parent_child"`
Parent content is stored in `child.metadata["parent_content"]` at index time. The pipeline reads `doc.page_content` for generation, which is the child text — but the parent context is available in metadata for extended answer generation.

**Config parameters**
```yaml
chunking:
  strategy: "parent_child"
  chunk_size: 128    # parent chunk size; children are chunk_size // 2
  overlap: 20        # parent overlap; child overlap is overlap // 2
```

**Hypothesis**
Retrieval precision benefits from small, specific chunks (easier to match the query exactly). Generation quality benefits from larger context (the full paragraph or section). Parent-child decouples these two requirements: retrieve small, generate from large. The hypothesis is that this produces higher faithfulness scores — the answer is generated from richer context — without sacrificing precision.

**Tested in:** Experiment 09

---

## 2. Embedding Model

**What it does**
Converts text (chunks at index time, queries at retrieval time) into dense vectors. Semantic similarity between a query and a chunk is computed as the cosine similarity of their vectors.

**Where it's implemented**
`rag/config_loader.py` → `build_embeddings()`
Uses: `langchain_openai.AzureOpenAIEmbeddings`

**Config parameters**
```yaml
embedding:
  provider: "azure_openai"
  model: "text-embedding-3-small"
```

**What it controls**
The quality of the embedding model determines the ceiling for dense retrieval. A weak embedding model will cluster semantically different chunks close together — creating "confusion zones" where irrelevant chunks are retrieved confidently. Notebook 01 (`01_embedding_space.ipynb`) visualises this directly.

**Hypothesis tested across the project**
All experiments use the same embedding model. This means embedding quality is a fixed constraint — the experiments test whether retrieval strategy and generation can compensate for its weaknesses, not whether a better embedding model helps.

---

## 3. Vector Store

**What it does**
Stores chunk embeddings and retrieves the top-k most similar chunks to a query embedding using approximate nearest-neighbour search.

**Where it's implemented**
`rag/config_loader.py` → `build_vector_store()`
Uses: `langchain_chroma.Chroma`
Each experiment uses its own named collection (keyed by `experiment_id`) to prevent cross-contamination between runs.

**Config parameters**
```yaml
# No explicit config key — collection is derived from experiment_id
# Persist directory from environment:
CHROMA_PERSIST_DIR=./data/chroma
```

**Note**
The vector store is not a variable in any experiment — it's fixed infrastructure. The retrieval *strategy* (dense vs hybrid vs sparse) varies; the underlying store does not.

---

## 4. Retrieval Strategies

Retrieval is the most studied variable in this project. Experiments 01–04 isolate its effect by holding generation constant.

**Config key:** `pipeline.retrieval.strategy`

---

### 4.1 Dense Retrieval

**What it does**
Embeds the query, then retrieves the top-k chunks by cosine similarity to the query embedding. Pure vector search — no keyword matching.

**Where it's implemented**
`rag/config_loader.py` → `build_retriever()` → `strategy == "dense"`
`vector_store.as_retriever(search_kwargs={"k": top_k})`

**Config parameters**
```yaml
retrieval:
  strategy: "dense"
  top_k: 3
```

**Hypothesis**
Dense retrieval captures semantic similarity. "What were GCPL's Home Care margins in Q3?" will match chunks that discuss segment margins even if the words are different — because the embedding space encodes the relationship. The weakness is vocabulary: if the source document uses a term the query doesn't, but doesn't embed it close to the query, retrieval fails silently.

**Tested in:** Experiments 01, 02

---

### 4.2 Sparse Retrieval (BM25)

**What it does**
Scores chunks by the TF-IDF-weighted keyword overlap between query and chunk. Pure lexical matching — no embeddings. Rewards exact term matches.

**Where it's implemented**
`rag/config_loader.py` → `build_retriever()` → `strategy == "sparse"`
Uses: `langchain_community.retrievers.BM25Retriever`

**Config parameters**
```yaml
retrieval:
  strategy: "sparse"
  top_k: 5
```

**Hypothesis**
For domain-specific corpora with precise terminology (product codes, regulatory terms, financial line items), keyword matching outperforms embedding similarity because the query and document use exactly the same terms. BM25 has no standalone experiment here — it features as the sparse component in hybrid.

---

### 4.3 Hybrid Retrieval

**What it does**
Runs dense and sparse retrieval in parallel, then combines their ranked result lists using `EnsembleRetriever` with a weighted score. The `alpha` parameter controls the dense-to-sparse ratio — `alpha=0.6` means 60% weight on dense, 40% on BM25.

**Where it's implemented**
`rag/config_loader.py` → `build_retriever()` → `strategy == "hybrid"`
Uses: `langchain.retrievers.EnsembleRetriever` with `[dense_retriever, BM25Retriever]`

**Config parameters**
```yaml
retrieval:
  strategy: "hybrid"
  top_k: 8        # candidates per retriever before merging
  alpha: 0.6      # weight on dense; (1-alpha) on sparse
```

**Hypothesis**
Dense retrieval fails on vocabulary mismatch. Sparse retrieval fails on paraphrase and synonyms. Hybrid retrieval compensates for both failure modes simultaneously. The expectation is that hybrid improves context recall (more of the right chunks are retrieved) without sacrificing precision.

The higher `top_k: 8` (vs `3` in dense-only) is deliberate — with two retrieval paths, more candidates enter before reranking filters them down.

**Tested in:** Experiments 03–10

---

## 5. Reranker

**What it does**
After initial retrieval, a cross-encoder model scores each candidate chunk against the query. Unlike bi-encoders (which embed query and document independently), a cross-encoder reads query and document together — giving it full attention across both — producing much more accurate relevance scores. The top `top_k_after_rerank` chunks by cross-encoder score are passed to generation.

**Where it's implemented**
`rag/config_loader.py` → `wrap_reranker()`
Uses: `langchain_community.cross_encoders.HuggingFaceCrossEncoder` with `cross-encoder/ms-marco-MiniLM-L-6-v2`
Wrapped in: `langchain.retrievers.ContextualCompressionRetriever`

**Config parameters**
```yaml
reranker:
  enabled: true
  top_k_after_rerank: 5    # chunks passed to generation after reranking
```

**Hypothesis**
Initial retrieval (dense or hybrid) optimises for recall — it finds a broad set of candidates. The reranker optimises precision — it selects the most genuinely relevant subset. This two-stage approach decouples the two objectives. The hypothesis is that reranking raises context precision without reducing recall, and that this translates to higher answer correctness because the LLM sees less noise in its context window.

**The cost**
Cross-encoder scoring runs locally (no API cost) but has latency. Each candidate chunk requires a separate forward pass through the cross-encoder. With `top_k: 8` candidates and `top_k_after_rerank: 5`, eight inference calls happen per query. Notebook 04 quantifies whether the quality gain justifies the latency increase.

**Tested in:** Experiments 04–10

---

## 6. Query Transforms

Query transforms modify the user's question before it hits the retriever. The goal is to bridge the vocabulary gap between how users ask questions and how information is written in source documents.

**Config key:** `pipeline.query_transform`

---

### 6.1 None (Passthrough)

**What it does**
The query is passed to the retriever exactly as entered. No transformation.

**Where it's implemented**
`rag/pipeline.py` → `answer_query()` → `query_transform == "none"`
`effective_query = query` (no LLM call)

**Hypothesis**
The baseline assumption is that the user's query is already well-formed for retrieval. This is valid for structured Q&A datasets with clean questions but breaks on conversational queries or domain-specific terminology gaps.

**Tested in:** Experiments 01–06, 09, 10

---

### 6.2 Query Rewriting

**What it does**
Passes the user's query to the LLM with the prompt: *"Rewrite this question to be more specific and retrieval-friendly."* The rewritten query is used for retrieval; the original query is used for generation.

**Where it's implemented**
`rag/chains.py` → `rewrite_query()`
`rag/pipeline.py` → `answer_query()` → `query_transform == "rewrite"`
Prompt: `QUERY_REWRITE_PROMPT` in `rag/chains.py`

**Config parameters**
```yaml
query_transform: "rewrite"
```

**Hypothesis**
Users ask questions in natural language. Source documents use formal or domain-specific language. The LLM, having seen both styles in training, can translate between them. A question like *"What's GCPL's hair care doing?"* might be rewritten to *"GCPL hair care segment revenue growth and market share performance"* — a much stronger retrieval signal.

**The risk**
Rewriting adds one LLM call per query (latency + cost) and can introduce hallucinated terms if the LLM adds specificity that isn't in the corpus.

**Tested in:** Experiment 07 vs Experiment 08 (HyDE)

---

### 6.3 HyDE (Hypothetical Document Embeddings)

**What it does**
Instead of rewriting the query, the LLM generates a short *hypothetical document* that would answer the question. This hypothetical document — not the original query — is then embedded and used for retrieval. The idea is that the embedding of a document-shaped text is more similar to real document embeddings than the embedding of a question-shaped text.

**Where it's implemented**
`rag/chains.py` → `generate_hyde_query()`
`rag/pipeline.py` → `answer_query()` → `query_transform == "hyde"`
Prompt: `HYDE_PROMPT` in `rag/chains.py`

**Config parameters**
```yaml
query_transform: "hyde"
```

**Hypothesis**
Embedding models are trained on document-to-document similarity. A question and its answer document may not be nearest neighbours in embedding space, even if they're semantically related. By generating a hypothetical answer document first, HyDE shifts the retrieval from query-document matching to document-document matching — where the embedding model is stronger.

**When HyDE wins over rewrite**
HyDE works best when the gap between query style and document style is large (e.g., conversational questions against formal financial reports). Rewriting works better when the query is already specific but uses the wrong vocabulary. Experiment 08 vs 07 tests which dominates on this corpus.

**Tested in:** Experiment 08

---

## 7. Generation Strategies

Generation is the final step — the LLM reads retrieved context and produces an answer. The strategy controls the prompt structure and what it asks the LLM to do with the context.

**Config key:** `pipeline.generation`

---

### 7.1 Stuff (Concatenate All Context)

**What it does**
All retrieved chunks are concatenated into a single context block with numbered separators (`[1] ... [2] ...`), then stuffed into a single prompt with the question. The LLM generates a free-form answer from that context.

**Where it's implemented**
`rag/chains.py` → `answer_with_stuff()`
Prompt: `STUFF_PROMPT`

**Config parameters**
```yaml
generation: "stuff"
```

**Hypothesis**
For small-to-medium context windows (a few chunks), concatenating everything and letting the LLM synthesise is the most straightforward approach. The LLM handles irrelevant context by ignoring it. The hypothesis is that the LLM is capable enough to extract the right information from a mixed context, so the burden of relevance filtering should sit on the retriever (and reranker), not the generator.

**The risk**
As context grows, LLMs suffer from "lost in the middle" — facts in the middle of a long context are attended to less than facts at the start or end. This is why reranking matters: it controls which chunks appear and in what order.

**Tested in:** Experiments 01–04, 06–10

---

### 7.2 Citation-Grounded Generation

**What it does**
The same concatenated context is used, but the prompt explicitly instructs the LLM to cite each factual claim with `[1]`, `[2]`, etc. referencing the numbered context chunks. The answer structure changes: claims are tied to sources.

**Where it's implemented**
`rag/chains.py` → `answer_with_citation()`
Prompt: `CITATION_PROMPT`

**Config parameters**
```yaml
generation: "citation_grounded"
```

**Hypothesis**
Citation grounding has two effects. First, it constrains the LLM's behaviour — the explicit citation instruction discourages hallucination because the LLM must justify each claim. Second, it produces a verifiable answer — a reader can check claim [2] against context chunk [2]. The hypothesis is that citation grounding improves faithfulness (the metric that measures hallucination) at potentially some cost to fluency and correctness score.

**The tradeoff**
Citation prompts can reduce answer quality if the LLM becomes overly conservative — refusing to synthesise across chunks or answering in a stilted, citation-heavy style. Experiment 05 quantifies whether the faithfulness gain is worth it.

**Tested in:** Experiment 05

---

## 8. Advanced Patterns

**Config key:** `pipeline.advanced`

---

### 8.1 None

Standard linear pipeline: transform → retrieve → generate. No loops or conditional branching.

**Tested in:** Experiments 01–09

---

### 8.2 CRAG (Corrective RAG)

**What it does**
CRAG adds a self-correction loop using a LangGraph state machine. After retrieval, each retrieved chunk is individually scored for relevance by the LLM using a binary yes/no grading prompt. If no chunks pass the relevance grade, the query is rewritten and retrieval is attempted again (one retry). Generation only happens once the LLM has confirmed relevant documents exist — or after the retry budget is exhausted.

**Where it's implemented**
`rag/chains.py` → `build_crag_graph()`
Implemented as a `StateGraph` with four nodes:

```
retrieve → grade → [generate | rewrite → retrieve → grade → generate]
```

| Node | What It Does |
|------|-------------|
| `retrieve` | Calls the retriever with the current question |
| `grade` | LLM scores each chunk: relevant or not. Filters to relevant only |
| `generate` | Runs `answer_with_stuff` on filtered documents |
| `rewrite` | LLM rewrites the query; increments retry counter |

The conditional edge at `grade`: if relevant docs exist or retry count ≥ 1, go to `generate`. Otherwise go to `rewrite`.

**Prompts used**
- `CRAG_GRADE_PROMPT` — "Is this document relevant to this question? yes/no"
- `QUERY_REWRITE_PROMPT` — same as experiment 07's rewrite

**Config parameters**
```yaml
advanced: "crag"
```

**Hypothesis**
Standard RAG is open-loop — it retrieves and generates regardless of retrieval quality. CRAG closes the loop by grading the retrieval before committing to generation. The hypothesis is that for queries where initial retrieval fails (wrong chunks, vocabulary mismatch), CRAG's self-correction recovers the answer that standard RAG would miss. The cost is significant: up to N+1 extra LLM calls per query (one per chunk for grading, plus one for rewriting).

**When CRAG wins**
Queries where the first retrieval attempt pulls clearly irrelevant chunks. The LLM grader catches this, rewrites, and the second retrieval succeeds.

**When CRAG loses**
Queries where the answer simply doesn't exist in the corpus. CRAG retries once and then generates from whatever it has — which may be nothing. It also adds latency and cost regardless of whether the retry was needed.

**Tested in:** Experiment 10 vs Experiment 04 (same stack without CRAG)

---

## 9. Evaluation Layer

**What it does**
Every pipeline run produces an `ExperimentResult` — a structured JSON of per-query and aggregate metrics. This schema is the backbone of the entire project: the dashboard reads it, the notebooks analyse it, the comparisons cite it.

**Where it's implemented**
- `evaluation/schemas.py` — `ExperimentResult`, `QueryResult`, `QueryMetrics`, `AggregateMetrics`
- `evaluation/llm_judge.py` — `LLMJudge` (4 metric prompts + token-overlap fallback)
- `evaluation/metrics.py` — `compute_query_metrics()` (RAGAS path + judge path)
- `evaluation/evaluator.py` — `evaluate_experiment()` (loops dataset, calls pipeline, aggregates)

---

### Metrics

| Metric | What It Measures | How It's Computed |
|--------|-----------------|-------------------|
| **Answer Correctness** | Does the answer match the ground truth? | Token overlap (heuristic) or LLM judge comparing answer vs ground truth |
| **Faithfulness** | Is the answer grounded in retrieved context? | Token overlap of answer against combined contexts, or LLM judge |
| **Context Precision** | What fraction of retrieved chunks were relevant? | Mean query-context overlap per chunk, or LLM judge |
| **Context Recall** | Was the information needed to answer actually retrieved? | Ground truth coverage in combined contexts, or LLM judge |

### Scoring paths

**RAGAS** (activated with `--ragas` flag)
Uses the `ragas` library with LLM-as-judge. Most accurate. Requires live API keys and adds cost per query.

**LLM Judge** (activated when `evaluation.llm_judge` is set in YAML config)
Uses `LLMJudge` with four structured prompts. Each metric is one LLM call per query. More accurate than heuristics, cheaper than RAGAS.

**Token-Overlap Heuristic** (default fallback)
No API calls. Pure token set intersection. Fast, offline, reproducible — but blind to paraphrase and semantics. The existing 10 result JSONs were scored this way.

---

## 10. How Components Compose

The pipeline call chain for a single query, end-to-end:

```
YAML config
    ↓
rag/config_loader.py    → build_llm(), build_embeddings(), load_documents(),
                          chunk_documents(), build_vector_store(),
                          build_retriever(), wrap_reranker()
    ↓
rag/pipeline.py         → Pipeline.prepare()   [called once per experiment]
                       → Pipeline.answer_query(query)
    │
    ├── query_transform == "rewrite"  → chains.rewrite_query()
    ├── query_transform == "hyde"     → chains.generate_hyde_query()
    │
    ├── advanced == "crag"            → chains.build_crag_graph().invoke()
    │                                    [retrieve → grade → (rewrite →) generate]
    │
    ├── retriever.invoke(query)       → dense / hybrid / sparse
    │
    ├── generation == "stuff"         → chains.answer_with_stuff()
    └── generation == "citation"      → chains.answer_with_citation()
    ↓
PipelineQueryOutput     → answer, retrieved_results, latency_ms, total_tokens, cost_usd
    ↓
evaluation/evaluator.py → evaluate_experiment()
                          → compute_query_metrics() per query
                          → AggregateMetrics across all queries
    ↓
ExperimentResult JSON   → experiments/results/{experiment_id}.json
    ↓
dashboard / analysis notebooks
```

### Which experiment changes which component

| Experiment | Chunking | Retrieval | Reranker | Query Transform | Generation | Advanced |
|-----------|----------|-----------|----------|-----------------|------------|----------|
| 01 | fixed | dense | ✗ | none | stuff | none |
| 02 | fixed | dense | ✗ | none | stuff | none |
| 03 | **semantic** | **hybrid** | ✗ | none | stuff | none |
| 04 | semantic | hybrid | **✓** | none | stuff | none |
| 05 | semantic | hybrid | ✓ | none | **citation** | none |
| 06 | semantic | hybrid | ✓ | none | stuff | none |
| 07 | semantic | hybrid | ✓ | **rewrite** | stuff | none |
| 08 | semantic | hybrid | ✓ | **hyde** | stuff | none |
| 09 | **parent_child** | hybrid | ✓ | none | stuff | none |
| 10 | semantic | hybrid | ✓ | none | stuff | **crag** |

LLM changes between 01 → 02 (mini → 4o) and 04 → 06 (GPT → Claude). All other variables held constant within each comparison pair.
