"""Retrieval failure taxonomy view for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

FAILURES_PATH = Path("analysis/outputs/retrieval_failures.json")
RATES_PATH = Path("analysis/outputs/failure_rates_by_experiment.json")

FAILURE_DESCRIPTIONS = {
    "vocabulary_mismatch": "Query uses different terms than the source document",
    "information_split": "Answer spans a chunk boundary — no single chunk contains the full answer",
    "right_topic_wrong_detail": "Correct product retrieved but wrong specific fact",
    "query_ambiguity": "Query matches multiple products or contexts equally",
    "wrong_chunk": "Lexical or embedding similarity pulled an irrelevant chunk",
    "no_retrieval": "No documents retrieved for this query",
}


def render_failure_analysis() -> None:
    """Render the retrieval failure breakdown section.

    Loads pre-computed failure taxonomy from analysis/02_retrieval_failures.ipynb.
    Displays a placeholder if the notebook hasn't been run yet.
    """
    st.subheader("Retrieval Failure Analysis")

    if not FAILURES_PATH.exists():
        st.info(
            "Run `analysis/02_retrieval_failures.ipynb` to generate the failure taxonomy. "
            "The failure breakdown will appear here once that output exists."
        )
        return

    try:
        failures = pd.read_json(FAILURES_PATH)

        # --- Failure type legend ---
        with st.expander("Failure type definitions", expanded=False):
            for ftype, desc in FAILURE_DESCRIPTIONS.items():
                st.markdown(f"**{ftype}**: {desc}")

        # --- Overall distribution ---
        col1, col2 = st.columns([1, 2])
        with col1:
            overall_counts = failures["failure_type"].value_counts()
            st.markdown("**Overall failure distribution**")
            st.dataframe(
                overall_counts.rename("count").reset_index().rename(columns={"index": "failure_type"}),
                hide_index=True,
            )

        with col2:
            try:
                import plotly.express as px

                fig = px.pie(
                    values=overall_counts.values,
                    names=overall_counts.index,
                    title="Failure Type Distribution (all experiments)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(overall_counts)

        st.divider()

        # --- Per-experiment breakdown ---
        if RATES_PATH.exists():
            rates_df = pd.read_json(RATES_PATH)
            st.markdown("**Failure rate by experiment and type**")

            try:
                import plotly.express as px

                # Melt for stacked bar
                id_col = "experiment_id"
                value_cols = [c for c in rates_df.columns if c != id_col]
                melted = rates_df.melt(id_vars=id_col, value_vars=value_cols,
                                       var_name="failure_type", value_name="rate")
                fig = px.bar(
                    melted,
                    x=id_col, y="rate", color="failure_type",
                    barmode="stack",
                    title="Failure Rate by Experiment",
                    labels={"rate": "Fraction of queries failing", id_col: "Experiment"},
                    height=420,
                )
                fig.update_layout(xaxis_tickangle=-40)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(rates_df)

        st.divider()

        # --- Query drilldown ---
        st.markdown("**Failing query examples**")
        failure_type_filter = st.selectbox(
            "Filter by failure type",
            options=["all"] + list(failures["failure_type"].unique()),
            key="failure_type_filter",
        )
        exp_filter = st.selectbox(
            "Filter by experiment",
            options=["all"] + sorted(failures["experiment_id"].unique().tolist()),
            key="failure_exp_filter",
        )

        filtered = failures.copy()
        if failure_type_filter != "all":
            filtered = filtered[filtered["failure_type"] == failure_type_filter]
        if exp_filter != "all":
            filtered = filtered[filtered["experiment_id"] == exp_filter]

        display_cols = [c for c in ["experiment_id", "query_id", "query", "failure_type",
                                     "context_precision", "context_recall"] if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].round(3),
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Error loading failure analysis: {exc}")
