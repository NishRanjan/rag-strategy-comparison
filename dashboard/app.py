"""Main Streamlit app for comparing experiment results."""

from __future__ import annotations

import streamlit as st

from dashboard.components import build_summary_dataframe, load_result_files
from dashboard.views.embedding_explorer import render_embedding_explorer
from dashboard.views.failure_analysis import render_failure_analysis
from dashboard.views.query_drilldown import render_query_drilldown
from dashboard.views.summary_table import render_summary_table
from dashboard.views.tradeoff_scatter import render_tradeoff_scatter


def main() -> None:
    """Load experiment results and render all comparison views."""
    st.set_page_config(page_title="AI Sandbox RAG Comparison", layout="wide")
    st.title("AI Sandbox: RAG Strategy Comparison")

    tab_results, tab_embed, tab_failures = st.tabs(
        ["Experiment Results", "Embedding Space", "Failure Analysis"]
    )

    results = load_result_files()
    summary_df = build_summary_dataframe(results)

    with tab_results:
        render_summary_table(summary_df)
        st.divider()
        render_query_drilldown(results)
        st.divider()
        render_tradeoff_scatter(summary_df)

    with tab_embed:
        render_embedding_explorer()

    with tab_failures:
        render_failure_analysis()


if __name__ == "__main__":
    main()
