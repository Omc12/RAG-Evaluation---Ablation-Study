## Evaluation & Ablation Study

We evaluated three RAG variants on the same set of stock-news questions:

- Semantic-only retrieval
- Keyword-only retrieval (BM25)
- Hybrid retrieval (Semantic + BM25)

Each answer was scored on:
- Retrieval relevance
- Grounding (faithfulness to context)
- Coverage

Results showed that Hybrid RAG consistently achieved higher scores,
demonstrating improved recall and multi-factor coverage.