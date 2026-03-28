"""Thin pipeline wrapper: YAML config → LangChain components → PipelineQueryOutput."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import structlog

from rag.chains import (
    answer_with_citation,
    answer_with_stuff,
    build_crag_graph,
    format_docs,
    generate_hyde_query,
    rewrite_query,
)
from rag.config_loader import (
    build_embeddings,
    build_llm,
    build_retriever,
    build_vector_store,
    chunk_documents,
    load_documents,
    wrap_reranker,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output schema (consumed by evaluation/evaluator.py)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Minimal retrieval result with the two fields the evaluator needs."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineQueryOutput:
    """Structured output from one pipeline query execution."""

    answer: str
    retrieved_results: list[SearchResult]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens: int
    cost_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

@contextmanager
def _token_callback():
    """Yield a token-tracking callback for AzureOpenAI/OpenAI calls.

    Falls back to a nullcontext (returning None) when langchain_community
    is unavailable or when a non-OpenAI provider is in use.
    """
    try:
        from langchain_community.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            yield cb
    except Exception:
        with nullcontext() as cb:
            yield cb


def _extract_usage(cb) -> tuple[int, float]:
    """Pull total_tokens and total_cost from a callback object (or default to 0)."""
    tokens = getattr(cb, "total_tokens", 0) or 0
    cost = getattr(cb, "total_cost", 0.0) or 0.0
    return int(tokens), float(cost)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Config-driven RAG pipeline backed by LangChain components.

    Call prepare() once to load documents and build retrieval indexes, then
    call answer_query() for each evaluation query. Both steps are separated so
    the experiment runner can time them independently.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the pipeline from a YAML config dict."""
        self.config = config
        self._prepared = False
        self._retriever = None
        self._chunks: list = []
        self._llm = None

    def prepare(self) -> None:
        """Load source documents, chunk, embed, index, and build the retriever."""
        pipe_cfg = self.config["pipeline"]

        embeddings = build_embeddings(pipe_cfg["embedding"])
        docs = load_documents(pipe_cfg["data_source"])
        self._chunks = chunk_documents(docs, pipe_cfg["chunking"], embeddings)

        # Use experiment_id as the Chroma collection name to prevent cross-experiment pollution.
        collection = self.config.get("experiment_id", "default")
        vector_store = build_vector_store(self._chunks, embeddings, collection)

        retriever = build_retriever(pipe_cfg["retrieval"], vector_store, self._chunks)
        self._retriever = wrap_reranker(retriever, pipe_cfg.get("reranker", {"enabled": False}))

        self._llm = build_llm(pipe_cfg["llm"])
        self._prepared = True
        logger.info("pipeline_prepared", docs=len(docs), chunks=len(self._chunks))

    def answer_query(self, query: str) -> PipelineQueryOutput:
        """Run one query through transform → retrieval → generation.

        Args:
            query: Raw user question from the evaluation dataset.

        Returns:
            PipelineQueryOutput with answer, retrieved contexts, latency, and cost.
        """
        if not self._prepared:
            self.prepare()

        pipe_cfg = self.config["pipeline"]
        generation_strategy: str = pipe_cfg.get("generation", "stuff")
        query_transform: str = pipe_cfg.get("query_transform", "none")
        advanced: str = pipe_cfg.get("advanced", "none")

        overall_start = time.perf_counter()
        effective_query = query
        docs = []
        retrieval_ms = 0.0
        gen_ms = 0.0

        with _token_callback() as cb:

            if advanced == "crag":
                graph = build_crag_graph(self._llm, self._retriever)
                state = graph.invoke(
                    {"question": query, "documents": [], "generation": "", "retry_count": 0}
                )
                docs = state["documents"]
                answer = state["generation"]
                # CRAG interleaves retrieval and generation; overall latency captures both.

            else:
                # --- Query transform ---
                retrieval_start = time.perf_counter()
                if query_transform == "rewrite":
                    effective_query = rewrite_query(self._llm, query)
                elif query_transform == "hyde":
                    effective_query = generate_hyde_query(self._llm, query)

                docs = self._retriever.invoke(effective_query)
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

                # --- Generation ---
                gen_start = time.perf_counter()
                if generation_strategy == "citation_grounded":
                    answer = answer_with_citation(self._llm, query, docs)
                else:
                    answer = answer_with_stuff(self._llm, query, docs)
                gen_ms = (time.perf_counter() - gen_start) * 1000

        overall_ms = (time.perf_counter() - overall_start) * 1000
        total_tokens, cost_usd = _extract_usage(cb)

        # Assign rank-based scores (1/rank) because LangChain retrievers don't expose similarity
        # scores through the BaseRetriever interface.
        retrieved_results = [
            SearchResult(
                text=doc.page_content,
                score=1.0 / (i + 1),
                metadata=doc.metadata,
            )
            for i, doc in enumerate(docs)
        ]

        return PipelineQueryOutput(
            answer=answer,
            retrieved_results=retrieved_results,
            latency_ms=overall_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            metadata={
                "effective_query": effective_query,
                "generation_strategy": generation_strategy,
                "query_transform": query_transform,
                "advanced": advanced,
            },
        )


def build_pipeline(config: dict) -> Pipeline:
    """Instantiate a Pipeline from a YAML config dict.

    Args:
        config: Parsed YAML experiment config.

    Returns:
        Unprepared Pipeline (call prepare() before answer_query()).
    """
    return Pipeline(config)
