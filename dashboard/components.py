"""Reusable dashboard helpers."""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd


def load_result_files(results_dir: str | Path = "experiments/results") -> list[dict]:
    """Load all experiment result JSON files from disk."""
    root = Path(results_dir)
    if not root.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def build_summary_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert experiment result dictionaries into a summary dataframe."""
    rows = []
    for result in results:
        aggregate = result["aggregate_metrics"]
        rows.append(
            {
                "experiment_id": result["experiment_id"],
                "correctness": aggregate["mean_correctness"],
                "faithfulness": aggregate["mean_faithfulness"],
                "context_precision": aggregate["mean_context_precision"],
                "context_recall": aggregate["mean_context_recall"],
                "mean_latency_ms": aggregate["mean_latency_ms"],
                "p95_latency_ms": aggregate["p95_latency_ms"],
                "total_cost_usd": aggregate["total_cost_usd"],
                "total_tokens": aggregate["total_tokens"],
            }
        )
    return pd.DataFrame(rows)

