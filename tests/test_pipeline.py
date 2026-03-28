"""Tests for pipeline construction and query execution."""

from __future__ import annotations

from rag.pipeline import build_pipeline


def test_pipeline_answers_query() -> None:
    """Ensure the configured pipeline can prepare and answer a query."""
    config = {
        "experiment_id": "test-baseline",
        "dataset": {"path": "experiments/datasets/fmcg_product_qa.jsonl", "format": "question_answer"},
        "pipeline": {
            "type": "rag",
            "data_source": "data/fmcg_docs",
            "chunking": {"strategy": "fixed", "chunk_size": 64, "overlap": 10},
            "embedding": {"provider": "hash", "model": "hash"},
            "vector_store": {"provider": "memory", "collection": "test"},
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
    result = pipeline.answer_query("Which product reduces frying odor?")
    assert result.answer
    assert result.retrieved_results

