"""Generation helpers and a lightweight CRAG fallback."""

from __future__ import annotations

import time

import structlog

logger = structlog.get_logger(__name__)


STUFF_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "If the context does not contain enough information to answer, say so clearly."
)

CITATION_SYSTEM = (
    "You are a precise assistant. Answer the question using only the provided context. "
    "For each factual claim, cite the source snippet using [1], [2], etc."
)

QUERY_REWRITE_SYSTEM = (
    "Rewrite the user's question to be more specific and retrieval-friendly. "
    "Return only the rewritten question with no explanation."
)

HYDE_SYSTEM = (
    "Write a short hypothetical document (2-4 sentences) that would directly answer this "
    "question. This hypothetical document will be used to improve retrieval."
)

CRAG_GRADE_SYSTEM = (
    "You are a relevance grader. Is the following document relevant to the question? "
    "Reply with only 'yes' or 'no'."
)


def _text_from_response(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
    return str(response)


def _invoke_llm(llm, prompt: str) -> str:
    return _text_from_response(llm.invoke(prompt))


def format_docs(docs: list) -> str:
    """Format a list of documents into a numbered context block."""
    return "\n\n---\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


def invoke_with_retry(llm, prompt: str, rate_limit_config: dict | None = None) -> str:
    """Invoke an LLM with simple exponential backoff on rate-limit errors."""
    cfg = rate_limit_config or {}
    max_retries = int(cfg.get("max_retries", 3))
    backoff = float(cfg.get("retry_backoff", 2.0))

    for attempt in range(max_retries + 1):
        try:
            return _invoke_llm(llm, prompt)
        except Exception as exc:  # noqa: BLE001
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait_seconds = backoff ** attempt
            logger.warning(
                "rate_limited_retry",
                attempt=attempt + 1,
                max_retries=max_retries,
                wait_seconds=wait_seconds,
                error=str(exc),
            )
            time.sleep(wait_seconds)


def answer_with_stuff(
    llm,
    question: str,
    docs: list,
    rate_limit_config: dict | None = None,
) -> str:
    prompt = (
        f"{STUFF_SYSTEM}\n\n"
        f"Context:\n{format_docs(docs)}\n\n"
        f"Question: {question}"
    )
    return invoke_with_retry(llm, prompt, rate_limit_config)


def answer_with_citation(
    llm,
    question: str,
    docs: list,
    rate_limit_config: dict | None = None,
) -> str:
    prompt = (
        f"{CITATION_SYSTEM}\n\n"
        f"Context:\n{format_docs(docs)}\n\n"
        f"Question: {question}"
    )
    return invoke_with_retry(llm, prompt, rate_limit_config)


def rewrite_query(llm, question: str, rate_limit_config: dict | None = None) -> str:
    prompt = f"{QUERY_REWRITE_SYSTEM}\n\n{question}"
    return invoke_with_retry(llm, prompt, rate_limit_config)


def generate_hyde_query(llm, question: str, rate_limit_config: dict | None = None) -> str:
    prompt = f"{HYDE_SYSTEM}\n\n{question}"
    return invoke_with_retry(llm, prompt, rate_limit_config)


class _SimpleCRAGGraph:
    def __init__(self, llm, retriever, rate_limit_config: dict | None = None) -> None:
        self._llm = llm
        self._retriever = retriever
        self._rate_limit_config = rate_limit_config

    def invoke(self, state: dict) -> dict:
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        docs = self._retriever.invoke(question)
        relevant_docs = []
        for doc in docs:
            verdict = invoke_with_retry(
                self._llm,
                (
                    f"{CRAG_GRADE_SYSTEM}\n\n"
                    f"Question: {question}\n\n"
                    f"Document: {doc.page_content}"
                ),
                self._rate_limit_config,
            )
            if "yes" in verdict.lower():
                relevant_docs.append(doc)

        if not relevant_docs and retry_count < 1:
            rewritten = rewrite_query(
                self._llm,
                question,
                rate_limit_config=self._rate_limit_config,
            )
            return self.invoke(
                {
                    **state,
                    "question": rewritten,
                    "retry_count": retry_count + 1,
                }
            )

        generation = answer_with_stuff(
            self._llm,
            question,
            relevant_docs,
            rate_limit_config=self._rate_limit_config,
        )
        return {**state, "documents": relevant_docs, "generation": generation}


def build_crag_graph(llm, retriever, rate_limit_config: dict | None = None):
    """Build a lightweight CRAG graph facade with the same invoke contract."""
    return _SimpleCRAGGraph(llm, retriever, rate_limit_config)
