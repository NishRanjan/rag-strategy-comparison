"""Metric helpers for converting query outputs into evaluation scores.

Two evaluation paths are available:
  1. RAGAS — uses LLM-as-judge via the ragas library (preferred when API keys are set).
  2. Heuristic fallback — uses the LLMJudge class which itself supports both an LLM
     judge (LangChain BaseChatModel) and token-overlap heuristics.

RAGAS is activated when both:
  - ragas is installed (pip install ragas)
  - use_ragas=True is passed to compute_query_metrics (or set as a module-level default)
"""

from __future__ import annotations

import structlog

from evaluation.llm_judge import LLMJudge
from evaluation.schemas import QueryMetrics

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# RAGAS availability check
# ---------------------------------------------------------------------------

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness as ragas_correctness,
        context_precision as ragas_context_precision,
        context_recall as ragas_context_recall,
        faithfulness as ragas_faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.debug("ragas_unavailable", detail="pip install ragas to enable RAGAS scoring")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(score: float) -> float:
    """Clamp a metric score into the 0–1 range."""
    return max(0.0, min(1.0, score))


def _compute_with_ragas(
    query: str,
    ground_truth: str,
    answer: str,
    retrieved_contexts: list[str],
    llm=None,
    embeddings=None,
) -> QueryMetrics | None:
    """Attempt to score one query using RAGAS.

    Returns None if RAGAS is unavailable or if evaluation fails, so the caller
    can fall back to the heuristic path.

    Args:
        query: User question.
        ground_truth: Reference answer.
        answer: Generated answer.
        retrieved_contexts: List of retrieved text chunks.
        llm: Optional LangChain BaseChatModel for RAGAS (uses its default otherwise).
        embeddings: Optional LangChain Embeddings for RAGAS.
    """
    if not RAGAS_AVAILABLE:
        return None

    try:
        from datasets import Dataset

        data = {
            "question": [query],
            "ground_truth": [ground_truth],
            "answer": [answer],
            "contexts": [retrieved_contexts],
        }
        dataset = Dataset.from_dict(data)

        kwargs: dict = {}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[
                ragas_correctness,
                ragas_faithfulness,
                ragas_context_precision,
                ragas_context_recall,
            ],
            raise_exceptions=False,
            **kwargs,
        )
        row = result.to_pandas().iloc[0]
        return QueryMetrics(
            answer_correctness=clamp(float(row.get("answer_correctness", 0.0))),
            faithfulness=clamp(float(row.get("faithfulness", 0.0))),
            context_precision=clamp(float(row.get("context_precision", 0.0))),
            context_recall=clamp(float(row.get("context_recall", 0.0))),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ragas_eval_failed", error=str(exc), fallback="heuristic")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_query_metrics(
    judge: LLMJudge,
    query: str,
    ground_truth: str,
    answer: str,
    retrieved_contexts: list[str],
    use_ragas: bool = False,
    ragas_llm=None,
    ragas_embeddings=None,
) -> QueryMetrics:
    """Compute the four canonical evaluation metrics for a single query.

    Evaluation path:
      - If use_ragas=True and ragas is installed: use RAGAS LLM judge.
      - Otherwise: use the injected LLMJudge (LLM or heuristic fallback).

    Args:
        judge: LLMJudge instance (used when not using RAGAS).
        query: User question.
        ground_truth: Reference answer from the dataset.
        answer: Generated answer from the pipeline.
        retrieved_contexts: Retrieved text chunks passed to the generator.
        use_ragas: Whether to attempt RAGAS evaluation first.
        ragas_llm: Optional LangChain BaseChatModel for RAGAS.
        ragas_embeddings: Optional LangChain Embeddings for RAGAS.

    Returns:
        QueryMetrics with all four scores in [0, 1].
    """
    if use_ragas:
        ragas_result = _compute_with_ragas(
            query=query,
            ground_truth=ground_truth,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        if ragas_result is not None:
            return ragas_result

    # Heuristic / LLM judge path
    correctness = judge.score_answer_correctness(
        query=query, ground_truth=ground_truth, answer=answer
    )
    faithfulness = judge.score_faithfulness(answer=answer, contexts=retrieved_contexts)
    precision = judge.score_context_precision(query=query, contexts=retrieved_contexts)
    recall = judge.score_context_recall(ground_truth=ground_truth, contexts=retrieved_contexts)

    return QueryMetrics(
        answer_correctness=clamp(correctness.score),
        faithfulness=clamp(faithfulness.score),
        context_precision=clamp(precision.score),
        context_recall=clamp(recall.score),
    )
