# RAG Experiment Components — Logic & Trade-offs

Each experiment changes **one variable** from its predecessor. This document explains what each component does, where it sits in the pipeline, and why it should (or shouldn't) improve on the baseline.

---

## Pipeline Overview

Every query flows through this sequence:

```
Raw Query
   ↓
[Query Transform]   ← none | rewrite | hyde
   ↓
[Retriever]         ← dense | sparse | hybrid
   ↓
[Reranker]          ← enabled | disabled
   ↓
[Generator]         ← stuff | citation_grounded
   ↓
[Advanced]          ← none | crag
   ↓
Answer
```

The baseline (Exp 01) uses the simplest option at every stage. Each subsequent experiment upgrades one stage.

---

## Experiment 01 — Baseline Naive

**Config:** `01_baseline_naive.yaml`
**LLM:** `gemini-2.5-flash-lite` | **Chunking:** fixed-128 | **Retrieval:** dense | **Generation:** stuff

### What it does

Documents are split into fixed 128-token windows with 20-token overlap. The query is embedded and compared against all chunk embeddings using dot-product similarity. The top-3 chunks are concatenated and stuffed into a single prompt.

### Relevant code

**Chunking** — [rag/config_loader.py](../rag/config_loader.py)
```python
# Fixed strategy: deterministic windows, no semantic awareness
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
return splitter.split_documents(documents)
```

**Dense retrieval** — `DenseRetriever` in config_loader.py
```python
# Dot-product similarity against all stored vectors
scores = [float(np.dot(qvec, dvec)) for dvec in self._vectors]
top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self._top_k]
```

**Stuff generation** — [rag/chains.py](../rag/chains.py)
```python
# All retrieved chunks concatenated, single LLM call
context = format_docs(docs)
prompt = f"Context:\n{context}\n\nQuestion: {question}"
```

### Baseline characteristics
- Cheapest option (`gemini-2.5-flash-lite` is lowest cost)
- Fixed chunking frequently splits answers across boundaries
- Dense-only retrieval misses keyword-exact matches
- No reranking means noisy top-K is passed directly to generation

---

## Experiment 02 — Better LLM

**Config:** `02_better_llm.yaml`
**Change:** `gemini-2.5-flash-lite` → `gemini-2.5-flash`
**Everything else:** identical to baseline

### What it tests

Can a stronger generator compensate for weak retrieval? If Exp 02 scores significantly higher than Exp 01 with no retrieval changes, it means the baseline LLM was the bottleneck. If the gap is small, retrieval quality matters more than generation quality.

### Relevant code

**LLM switch** — [rag/config_loader.py](../rag/config_loader.py)
```python
def build_llm(llm_config):
    model = llm_config["model"]  # "gemini-2.5-flash" vs "gemini-2.5-flash-lite"
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens)
```

### Expected improvement
Better instruction following and reasoning over the same (noisy) retrieved context. Faithfulness and answer correctness should improve. Context precision/recall are unaffected — retrieval is identical.

---

## Experiment 03 — Hybrid Retrieval

**Config:** `03_hybrid_retrieval.yaml`
**Changes:** chunking `fixed` → `semantic`; retrieval `dense` → `hybrid` (BM25 + dense, α=0.6); `top_k` 3 → 8

### What it tests

Two changes together: semantic chunking produces more coherent chunks (less boundary splitting), and hybrid retrieval combines vocabulary matching (BM25) with semantic similarity. Product names, ingredient lists, and SKU codes are keyword-heavy — BM25 catches exact matches that dense embeddings dilute.

### Relevant code

**Semantic chunking** — [rag/config_loader.py](../rag/config_loader.py)
```python
# SemanticChunker groups sentences by embedding similarity breakpoints
# rather than fixed token counts
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(embeddings)
return splitter.create_documents([d.page_content for d in documents])
```

**Hybrid retrieval** — `HybridRetriever` in config_loader.py
```python
# alpha=0.6 means 60% dense, 40% BM25
dense_scores = [float(np.dot(qvec, dvec)) for dvec in self._dense_vectors]
sparse_scores = self._bm25.get_scores(query_tokens)

# Normalise each to [0,1] then blend
combined = alpha * norm(dense) + (1 - alpha) * norm(sparse)
```

### Expected improvement
Hybrid retrieval is the single biggest lever in most RAG benchmarks. BM25 recovers exact keyword matches that semantic similarity misses (e.g., a query for "SKU XJ-400" maps poorly in embedding space). Semantic chunking reduces the chance that the answer spans two chunks. Both push context precision up.

---

## Experiment 04 — Hybrid + Reranker

**Config:** `04_hybrid_rerank.yaml`
**Change:** reranker `enabled: false` → `enabled: true, top_k_after_rerank: 5`

### What it tests

Reranking is a two-stage strategy: retrieve broadly (top-8) then re-score with a cross-encoder that jointly encodes the query and each candidate chunk. Cross-encoders are more accurate than bi-encoders but too slow for full-corpus search — so they operate only on the shortlist.

### Relevant code

**Reranker wrapper** — [rag/config_loader.py](../rag/config_loader.py)
```python
def wrap_reranker(retriever, reranker_config):
    if not reranker_config.get("enabled", False):
        return retriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=encoder, top_n=top_k_after_rerank)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
```

### Expected improvement
Reranking eliminates false-positive retrievals where a chunk is semantically adjacent to the query but doesn't actually contain the answer. The cross-encoder scores relevance in the full query+chunk context. Faithfulness should increase (less irrelevant context passed to the LLM). Latency increases due to local cross-encoder inference.

---

## Experiment 05 — Citation-Grounded Generation

**Config:** `05_citation_grounded.yaml`
**Change:** generation `stuff` → `citation_grounded`

### What it tests

Citation-grounded generation constrains the LLM to answer only with claims it can attribute to a specific numbered source chunk. This trades some fluency for verifiability — the answer will be more faithful but potentially less complete if the answer spans chunks.

### Relevant code

**Citation prompt** — [rag/chains.py](../rag/chains.py)
```python
CITATION_SYSTEM = """
You are a precise assistant. Answer using ONLY the provided sources.
For every factual claim, add an inline citation [1], [2], etc.
If the sources do not contain the answer, say "I don't know".
"""

def answer_with_citation(llm, question, docs, rate_limit_config):
    numbered = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    ...
```

### Expected improvement
Faithfulness metric should rise — the LLM cannot hallucinate unsupported facts without violating the citation constraint. Answer correctness may dip slightly if the answer requires synthesis across chunks that the model is reluctant to combine without explicit support.

---

## Experiment 06 — Cross-Provider Model

**Config:** `06_cross_provider.yaml`
**Change:** LLM `gemini-2.5-flash` → `gemini-2.0-flash`

### What it tests

Gemini 2.0 Flash is an older generation model at lower cost. On the same strong retrieval stack (hybrid + rerank), does the model generation quality still hold? This isolates whether the quality gains from Exp 03–05 are robust across LLM versions, or fragile to model capability.

### Relevant code

Same `build_llm()` call as Exp 02 — only the `model` string changes.

### Expected outcome
A cost-quality comparison point. If Gemini 2.0 Flash matches 2.5 Flash on this stack, it suggests the retrieval quality is doing the heavy lifting. If scores drop significantly, generation capability is still a meaningful variable even with good retrieval.

---

## Experiment 07 — Query Rewriting

**Config:** `07_query_rewrite.yaml`
**Change:** `query_transform: "none"` → `"rewrite"`

### What it tests

User queries are often under-specified or phrased differently from how source documents are written. Query rewriting uses the LLM to rephrase the question into a form more likely to match the document vocabulary before retrieval.

### Relevant code

**Rewrite chain** — [rag/chains.py](../rag/chains.py)
```python
QUERY_REWRITE_SYSTEM = """
Rewrite the following question to be more specific and better suited
for document retrieval. Preserve the original intent.
Return only the rewritten question, no explanation.
"""

def rewrite_query(llm, question, rate_limit_config):
    response = invoke_with_retry(llm, [SystemMessage(QUERY_REWRITE_SYSTEM),
                                       HumanMessage(question)], rate_limit_config)
    return response.content.strip()
```

**Pipeline integration** — [rag/pipeline.py](../rag/pipeline.py)
```python
if query_transform == "rewrite":
    effective_query = rewrite_query(self._llm, query, rate_limit_cfg)
elif query_transform == "hyde":
    effective_query = generate_hyde_query(self._llm, query, rate_limit_cfg)
else:
    effective_query = query
```

### Expected improvement
Useful for conversational or ambiguous queries. In an FMCG product domain, rewriting may expand abbreviations or add category context (e.g., "best for oily skin?" → "which moisturiser products are formulated for oily or combination skin?"). Adds one extra LLM call per query.

---

## Experiment 08 — HyDE

**Config:** `08_hyde.yaml`
**Change:** `query_transform: "rewrite"` → `"hyde"`

### What it tests

HyDE (Hypothetical Document Embeddings) takes a different approach to the query mismatch problem. Instead of rewriting the query, it generates a *hypothetical answer document* and embeds that for retrieval. The hypothesis sits closer in embedding space to real answer-containing chunks than the short query does.

### Relevant code

**HyDE chain** — [rag/chains.py](../rag/chains.py)
```python
HYDE_SYSTEM = """
Write a short passage (2-3 sentences) that would be a perfect answer
to the following question, as if it appeared in a product document.
Do not say you are generating a hypothetical document.
"""

def generate_hyde_query(llm, question, rate_limit_config):
    response = invoke_with_retry(llm, [SystemMessage(HYDE_SYSTEM),
                                       HumanMessage(question)], rate_limit_config)
    return response.content.strip()  # This text is embedded, not the original query
```

### HyDE vs Rewrite

| | Query Rewrite | HyDE |
|---|---|---|
| What gets embedded | Rewritten question | Hypothetical answer |
| Works well when | Query is ambiguous | Query is well-formed but short |
| Risk | May still miss vocabulary | Hallucinated hypothesis can mislead retrieval |
| Extra LLM calls | 1 | 1 |

HyDE tends to outperform rewriting on factual Q&A where the embedding space is document-like. For conversational queries, rewriting usually wins.

---

## Experiment 09 — Parent-Child Chunking

**Config:** `09_parent_child.yaml`
**Change:** chunking `semantic` → `parent_child`

### What it tests

Parent-child chunking indexes small child chunks for retrieval precision (so similarity search returns a tight match) but returns the larger parent chunk to the generator. This gives the LLM more context around the matched passage, reducing the chance that the answer is cut off at a chunk boundary.

### Relevant code

**Parent-child strategy** — [rag/config_loader.py](../rag/config_loader.py)
```python
# Child chunks: small, used for embedding & retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
# Parent chunks: 3× larger, returned to the generator
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size * 3, chunk_overlap=overlap)

parent_chunks = parent_splitter.split_documents(documents)
# Each child stores a pointer to its parent
for parent_idx, parent in enumerate(parent_chunks):
    children = child_splitter.split_documents([parent])
    for child in children:
        child.metadata["parent_idx"] = parent_idx
```

### Expected improvement
Retrieval precision stays high (small child chunks match tightly). Generation quality improves because the LLM receives full paragraphs rather than truncated mid-sentence chunks. Context recall should increase. The downside is more tokens sent to generation, increasing cost and latency.

---

## Experiment 10 — CRAG (Self-Correcting RAG)

**Config:** `10_crag.yaml`
**Change:** `advanced: "none"` → `"crag"`

### What it tests

CRAG (Corrective RAG) adds a self-correction loop. After retrieval, a grader LLM evaluates whether each retrieved chunk is actually relevant to the question. If all chunks are graded as irrelevant, the system rewrites the query and retries retrieval rather than generating a hallucinated answer from bad context.

### Relevant code

**CRAG graph** — [rag/chains.py](../rag/chains.py)
```python
class _SimpleCRAGGraph:
    """
    State machine:
      retrieve → grade_docs → [all_relevant] → generate
                            → [some_irrelevant] → rewrite_query → retrieve (retry once)
                            → [all_irrelevant]  → rewrite_query → retrieve (retry once)
    """
    def grade_doc(self, doc: Document, question: str) -> str:
        response = invoke_with_retry(self._llm, [
            SystemMessage(CRAG_GRADE_SYSTEM),
            HumanMessage(f"Document: {doc.page_content}\nQuestion: {question}")
        ], self._rate_limit_config)
        return "relevant" if "yes" in response.content.lower() else "irrelevant"
```

```python
CRAG_GRADE_SYSTEM = """
You are a relevance grader. Given a document and a question,
respond with only 'yes' if the document contains information
useful for answering the question, otherwise 'no'.
"""
```

**Pipeline invocation** — [rag/pipeline.py](../rag/pipeline.py)
```python
if advanced == "crag":
    crag = build_crag_graph(self._llm, self._retriever, rate_limit_cfg)
    result = crag.invoke({"question": effective_query})
    answer = result["answer"]
    retrieved_docs = result["documents"]
```

### Expected improvement
CRAG is most valuable when the document corpus has sparse coverage of some queries — i.e., the first retrieval attempt sometimes returns off-topic chunks. The self-correction loop prevents the LLM from generating confident but unsupported answers. The cost is 1 grading call per retrieved chunk, plus potentially a full retrieval + generation retry. Latency is highest of all 10 experiments.

### Why LangGraph

CRAG has conditional branching (retry vs. proceed) and state that persists across steps (the graded document list). This is a genuine state machine, not a linear chain — LangGraph's `StateGraph` is the right abstraction. All other experiments use LCEL because they're linear transforms.

---

## Summary Table

| # | Experiment | Stage Changed | Key Mechanism | Expected Gain |
|---|---|---|---|---|
| 01 | Baseline | — | Fixed chunks, dense retrieval, stuff gen | Reference point |
| 02 | Better LLM | Generation | `flash-lite` → `flash` | Better reasoning over same context |
| 03 | Hybrid Retrieval | Chunking + Retrieval | Semantic chunks + BM25+dense blend | Fewer boundary splits, keyword recall |
| 04 | Reranker | Post-retrieval | Cross-encoder rescores top-8 → top-5 | Removes false-positive retrievals |
| 05 | Citation Grounded | Generation | LLM constrained to cite sources | Higher faithfulness |
| 06 | Cross-Provider | Generation | `gemini-2.5-flash` → `gemini-2.0-flash` | Cost reduction, quality benchmark |
| 07 | Query Rewrite | Pre-retrieval | LLM rephrases query before embedding | Better vocabulary alignment |
| 08 | HyDE | Pre-retrieval | LLM generates hypothetical answer to embed | Denser embedding space match |
| 09 | Parent-Child | Chunking | Small chunks indexed, large chunks returned | Better context around match |
| 10 | CRAG | Post-retrieval loop | Grade → retry if irrelevant | Avoids generation from bad context |
