# Decision Framework

## When to Use Which RAG Strategy

Use the baseline when you need a quick sanity check and the retrieval problem is simple.

Use semantic chunking plus hybrid retrieval when documents are varied in structure and keyword overlap alone is not enough.

Add reranking when top-k recall is decent but precision is noisy.

Use citation-grounded generation when trust and inspectability matter to stakeholders.

Use query transforms when your dataset includes analytical or multi-hop questions.

Use CRAG only when retrieval failures are costly enough to justify extra latency and complexity.

