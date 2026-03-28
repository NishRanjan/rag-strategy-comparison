"""Evaluation runner that produces canonical ExperimentResult objects."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

import structlog

from evaluation.llm_judge import LLMJudge
from evaluation.metrics import compute_query_metrics
from evaluation.schemas import AggregateMetrics, ExperimentResult, QueryResult

logger = structlog.get_logger(__name__)


def percentile(values: list[float], percentile_value: float) -> float:
    """Compute a simple percentile from a list of numeric values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        round((percentile_value / 100) * (len(sorted_values) - 1)),
    )
    return sorted_values[index]


def evaluate_experiment(
    pipeline,
    dataset: list[dict[str, str]],
    experiment_id: str,
    dataset_id: str,
    use_ragas: bool = False,
    judge_llm=None,
) -> ExperimentResult:
    """Run the pipeline over a dataset and return the canonical result schema.

    Args:
        pipeline: Pipeline instance with answer_query(query) -> PipelineQueryOutput.
        dataset: List of dicts with 'query' and 'ground_truth' keys.
        experiment_id: Unique identifier for this experiment run.
        dataset_id: Identifier for the evaluation dataset (used in the result).
        use_ragas: Whether to use RAGAS LLM-as-judge scoring (requires ragas + API keys).
        judge_llm: Optional LangChain BaseChatModel for the heuristic LLM judge.
                   When None and use_ragas=False, token-overlap heuristics are used.

    Returns:
        ExperimentResult with per-query and aggregate metrics.
    """
    judge = LLMJudge(llm_client=judge_llm)
    query_results: list[QueryResult] = []

    for index, row in enumerate(dataset):
        query = row["query"]
        ground_truth = row["ground_truth"]

        output = pipeline.answer_query(query)
        contexts = [result.text for result in output.retrieved_results]

        metrics = compute_query_metrics(
            judge=judge,
            query=query,
            ground_truth=ground_truth,
            answer=output.answer,
            retrieved_contexts=contexts,
            use_ragas=use_ragas,
        )

        query_results.append(
            QueryResult(
                query_id=row.get("query_id", f"q{index + 1:03d}"),
                query=query,
                ground_truth=ground_truth,
                generated_answer=output.answer,
                retrieved_contexts=contexts,
                retrieval_scores=[result.score for result in output.retrieved_results],
                metrics=metrics,
                latency_ms=output.latency_ms,
                total_tokens=output.total_tokens,
                cost_usd=output.cost_usd,
            )
        )
        logger.info(
            "query_evaluated",
            index=index + 1,
            total=len(dataset),
            correctness=metrics.answer_correctness,
        )

    aggregate_metrics = AggregateMetrics(
        mean_correctness=mean(r.metrics.answer_correctness for r in query_results) if query_results else 0.0,
        mean_faithfulness=mean(r.metrics.faithfulness for r in query_results) if query_results else 0.0,
        mean_context_precision=mean(r.metrics.context_precision for r in query_results) if query_results else 0.0,
        mean_context_recall=mean(r.metrics.context_recall for r in query_results) if query_results else 0.0,
        mean_latency_ms=mean(r.latency_ms for r in query_results) if query_results else 0.0,
        total_cost_usd=sum(r.cost_usd for r in query_results),
        total_tokens=sum(r.total_tokens for r in query_results),
        p95_latency_ms=percentile([r.latency_ms for r in query_results], 95),
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        pipeline_config=pipeline.config,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_id=dataset_id,
        per_query_results=query_results,
        aggregate_metrics=aggregate_metrics,
    )
