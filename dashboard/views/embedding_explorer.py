"""Interactive embedding space explorer view for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECTIONS_PATH = Path("analysis/outputs/embedding_projections.json")
DISTANCES_PATH = Path("analysis/outputs/cluster_distances.json")


def render_embedding_explorer() -> None:
    """Render the interactive embedding space visualization section.

    Loads pre-computed UMAP projections from the analysis notebook output.
    Displays a placeholder if the notebook hasn't been run yet.
    """
    st.subheader("Embedding Space Explorer")

    if not PROJECTIONS_PATH.exists():
        st.info(
            "Run `analysis/01_embedding_space.ipynb` to generate the embedding projections. "
            "The interactive scatter plot will appear here once that output exists."
        )
        return

    try:
        import plotly.express as px

        embed_df = pd.read_json(PROJECTIONS_PATH)

        # Distance stats summary
        if DISTANCES_PATH.exists():
            stats = json.loads(DISTANCES_PATH.read_text())
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean intra-doc distance", f"{stats.get('mean_intra_doc', 0):.3f}")
            col2.metric("Mean inter-doc distance", f"{stats.get('mean_inter_doc', 0):.3f}")
            col3.metric(
                "Separation ratio",
                f"{stats.get('separation_ratio', 0):.2f}x",
                help="Higher = embedding model better separates different documents",
            )
            st.divider()

        # Interactive UMAP scatter
        color_by = st.selectbox(
            "Color by",
            options=[c for c in embed_df.columns if c not in {"x", "y", "text_preview"}],
            key="embed_color_by",
        )
        fig = px.scatter(
            embed_df,
            x="x",
            y="y",
            color=color_by,
            hover_data=["text_preview"] if "text_preview" in embed_df.columns else None,
            title="FMCG Corpus Embedding Space (UMAP 2D)",
            height=550,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Showing {len(embed_df):,} chunks. "
            "Well-separated clusters indicate the embedding model distinguishes document categories effectively."
        )

    except ImportError:
        st.warning("Install plotly to view the interactive chart: `pip install plotly`")
    except Exception as exc:
        st.error(f"Error loading embedding projections: {exc}")
