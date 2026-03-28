"""Tests for evaluation schema generation."""

from __future__ import annotations

from evaluation.evaluator import evaluate_experiment
from rag.pipeline import build_pipeline


def test_evaluator_returns_experiment_result() -> None:
    """Ensure the evaluator returns the canonical experiment result shape."""
    config = {
        "experiment_id": "test-eval",
        "dataset": {"path": "experiments/datasets/fmcg_product_qa.jsonl", "format": "question_answer"},
        "pipeline": {
            "type": "rag",
            "data_source": "data/fmcg_docs",
            "chunking": {"strategy": "fixed", "chunk_size": 64, "overlap": 10},
            "embedding": {"provider": "hash", "model": "hash"},
            "vector_store": {"provider": "memory", "collection": "test-eval"},
            "retrieval": {"strategy": "dense", "top_k": 3, "alpha": 0.5},
            "reranker": {"enabled": False, "model": "bge-reranker-v2-m3", "top_k_after_rerank": 2},
            "query_transform": {"strategy": "none"},
            "generation": {
                "strategy": "stuff",
                "llm": {"provider": "heuristic", "model": "heuristic-llm", "temperature": 0.0, "max_tokens": 128},
            },
            "advanced": {"pattern": "none"},
        },
    }
    pipeline = build_pipeline(config)
    pipeline.prepare()
    dataset = [{"query_id": "q1", "query": "What is VitaOil Plus?", "ground_truth": "VitaOil Plus is a fortified cooking oil."}]
    result = evaluate_experiment(pipeline=pipeline, dataset=dataset, experiment_id="test-eval", dataset_id="test")
    assert result.experiment_id == "test-eval"
    assert result.per_query_results
    assert result.aggregate_metrics.total_tokens >= 0

