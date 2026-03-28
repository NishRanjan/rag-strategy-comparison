"""Tradeoff scatter dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_tradeoff_scatter(summary_df: pd.DataFrame) -> None:
    """Render the cost vs quality tradeoff view."""
    st.subheader("Tradeoff Scatter")
    if summary_df.empty:
        st.info("No results available for scatter plot.")
        return
    y_axis = st.selectbox("Y-axis", ["correctness", "faithfulness", "mean_latency_ms"])
    chart_df = summary_df[["experiment_id", "total_cost_usd", y_axis]].rename(columns={"total_cost_usd": "cost_usd"})
    st.scatter_chart(chart_df, x="cost_usd", y=y_axis, color="experiment_id")

