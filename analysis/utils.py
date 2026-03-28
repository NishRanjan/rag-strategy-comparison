"""Shared utilities for analysis notebooks: data loading, plotting, export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results")
OUTPUTS_DIR = Path("analysis/outputs")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str | Path = RESULTS_DIR) -> list[dict]:
    """Load all experiment result JSON files into a list of dicts."""
    root = Path(results_dir)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]


def results_to_summary_df(results: list[dict]) -> pd.DataFrame:
    """Convert experiment results to a flat summary DataFrame.

    Columns: experiment_id, correctness, faithfulness, context_precision,
    context_recall, mean_latency_ms, p95_latency_ms, total_cost_usd, total_tokens.
    """
    rows = []
    for r in results:
        agg = r["aggregate_metrics"]
        rows.append({
            "experiment_id": r["experiment_id"],
            "correctness": agg["mean_correctness"],
            "faithfulness": agg["mean_faithfulness"],
            "context_precision": agg["mean_context_precision"],
            "context_recall": agg["mean_context_recall"],
            "mean_latency_ms": agg["mean_latency_ms"],
            "p95_latency_ms": agg["p95_latency_ms"],
            "total_cost_usd": agg["total_cost_usd"],
            "total_tokens": agg["total_tokens"],
        })
    return pd.DataFrame(rows)


def results_to_query_df(results: list[dict]) -> pd.DataFrame:
    """Expand all per-query results into a long-format DataFrame.

    Columns: experiment_id, query_id, query, ground_truth, generated_answer,
    answer_correctness, faithfulness, context_precision, context_recall,
    latency_ms, total_tokens, cost_usd.
    """
    rows = []
    for r in results:
        exp_id = r["experiment_id"]
        for qr in r["per_query_results"]:
            m = qr["metrics"]
            rows.append({
                "experiment_id": exp_id,
                "query_id": qr["query_id"],
                "query": qr["query"],
                "ground_truth": qr["ground_truth"],
                "generated_answer": qr["generated_answer"],
                "retrieved_contexts": qr["retrieved_contexts"],
                "answer_correctness": m["answer_correctness"],
                "faithfulness": m["faithfulness"],
                "context_precision": m["context_precision"],
                "context_recall": m["context_recall"],
                "latency_ms": qr["latency_ms"],
                "total_tokens": qr["total_tokens"],
                "cost_usd": qr["cost_usd"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

METRIC_COLS = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]


def composite_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Compute a weighted composite score across the four metrics.

    Args:
        df: DataFrame with metric columns (summary or query level).
        weights: Dict mapping metric name → weight. Defaults to equal weights.

    Returns:
        Series of composite scores.
    """
    if weights is None:
        weights = {m: 0.25 for m in METRIC_COLS}
    score = sum(df[m] * w for m, w in weights.items())
    return score


def pareto_frontier(df: pd.DataFrame, quality_col: str = "correctness", cost_col: str = "total_cost_usd") -> pd.DataFrame:
    """Return the Pareto-optimal rows (highest quality for a given cost level).

    Args:
        df: Summary DataFrame with quality and cost columns.
        quality_col: Column name for the quality metric.
        cost_col: Column name for cost.

    Returns:
        Subset of df containing only Pareto-optimal experiments.
    """
    sorted_df = df.sort_values(cost_col).reset_index(drop=True)
    pareto_rows = []
    best_quality = -1.0
    for _, row in sorted_df.iterrows():
        if row[quality_col] >= best_quality:
            pareto_rows.append(row)
            best_quality = row[quality_col]
    return pd.DataFrame(pareto_rows)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def apply_plot_style() -> None:
    """Apply a consistent matplotlib style for analysis notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 6)})


def radar_chart(ax, values: list[float], labels: list[str], title: str = "") -> None:
    """Draw a radar/spider chart for multi-metric comparison.

    Args:
        ax: Matplotlib Axes (must be polar).
        values: Metric values in [0, 1].
        labels: Metric labels (same length as values).
        title: Chart title.
    """
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    ax.plot(angles, values_plot, linewidth=2)
    ax.fill(angles, values_plot, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=15)


# ---------------------------------------------------------------------------
# Output persistence (notebooks write JSON/CSV for the dashboard to consume)
# ---------------------------------------------------------------------------

def save_output(data, filename: str) -> Path:
    """Save analysis output to analysis/outputs/ for dashboard consumption.

    Args:
        data: Dict, DataFrame, or list to serialize.
        filename: Output filename (e.g., 'embedding_projections.json').

    Returns:
        Path of the saved file.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename

    if isinstance(data, pd.DataFrame):
        if filename.endswith(".csv"):
            data.to_csv(path, index=False)
        else:
            data.to_json(path, orient="records", indent=2)
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return path
