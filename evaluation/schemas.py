"""Canonical evaluation schemas consumed by results and dashboard views."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class QueryMetrics:
    """Per-query quality metrics normalized to the 0-1 range."""

    answer_correctness: float
    faithfulness: float
    context_precision: float
    context_recall: float


@dataclass
class QueryResult:
    """Per-query execution and evaluation output."""

    query_id: str
    query: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str]
    retrieval_scores: list[float]
    metrics: QueryMetrics
    latency_ms: float
    total_tokens: int
    cost_usd: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics computed across all experiment queries."""

    mean_correctness: float
    mean_faithfulness: float
    mean_context_precision: float
    mean_context_recall: float
    mean_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    p95_latency_ms: float


@dataclass
class ExperimentResult:
    """Top-level experiment output written to disk and read by the dashboard."""

    experiment_id: str
    pipeline_config: dict
    timestamp: str
    dataset_id: str
    per_query_results: list[QueryResult]
    aggregate_metrics: AggregateMetrics

    def to_dict(self) -> dict:
        """Convert the dataclass tree into a JSON-serializable dictionary."""
        return asdict(self)

