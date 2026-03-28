# RAG Strategy Comparison: FMCG Product Knowledge QA

## Setup

- FMCG product QA dataset covering product facts, usage guidance, and restrictions.
- Ten config-driven experiments that isolate a single change whenever possible.
- Every run emits the same `ExperimentResult` JSON for dashboard and write-up reuse.

## Expected Findings

### Retrieval matters more than the model

The repo is structured to compare retrieval improvements directly against model upgrades so the portfolio story stays concrete and defensible.

### Reranking is a likely high-ROI addition

Experiment `04_hybrid_rerank` exists specifically to measure the precision lift from reranking against the extra latency and cost.

### Query transforms help selectively

`07_query_rewrite` and `08_hyde` are intended to show whether query transforms help harder questions enough to justify production complexity.

### CRAG may be overkill for FMCG QA

`10_crag` measures whether self-correcting retrieval improves outcomes enough to offset the operational overhead.

## Recommendation Template

Use the dashboard outputs to fill in concrete metrics and choose the best quality-cost tradeoff for the target domain.

