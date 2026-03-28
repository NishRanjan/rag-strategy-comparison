"""Summary table dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_summary_table(summary_df: pd.DataFrame) -> None:
    """Render the side-by-side comparison summary view."""
    st.subheader("Summary Table")
    if summary_df.empty:
        st.info("No experiment results found yet.")
        return
    st.dataframe(
        summary_df.style.highlight_max(color="#c7f9cc").highlight_min(color="#f8d7da"),
        use_container_width=True,
    )

