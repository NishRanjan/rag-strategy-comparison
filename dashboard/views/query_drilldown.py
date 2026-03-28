"""Per-query drilldown dashboard view."""

from __future__ import annotations

import streamlit as st


def render_query_drilldown(results: list[dict]) -> None:
    """Render a side-by-side query inspection view across experiments."""
    st.subheader("Per-Query Drilldown")
    if not results:
        st.info("Run an experiment to inspect per-query details.")
        return
    queries = {
        item["query_id"]: item["query"]
        for result in results
        for item in result.get("per_query_results", [])
    }
    if not queries:
        st.info("No per-query results available.")
        return
    selected_query_id = st.selectbox("Query", list(queries.keys()), format_func=lambda query_id: queries[query_id])
    for result in results:
        match = next(
            (item for item in result["per_query_results"] if item["query_id"] == selected_query_id),
            None,
        )
        if match is None:
            continue
        with st.expander(result["experiment_id"], expanded=False):
            st.markdown(f"**Answer**\n\n{match['generated_answer']}")
            st.write("Metrics", match["metrics"])
            st.write("Contexts", match["retrieved_contexts"])

