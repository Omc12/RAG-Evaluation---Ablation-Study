## Evaluation & Ablation Study

We evaluated three retrieval strategies—semantic retrieval, hybrid retrieval, and hybrid retrieval with MMR re-ranking—on a fixed stock news dataset. Each method was assessed using Gemini-based automatic evaluation across three criteria: retrieval relevance, grounding, and coverage (0–2 scale).

Hybrid retrieval consistently improved coverage compared to semantic retrieval, indicating higher recall of diverse market factors. However, redundancy was observed, with multiple retrieved documents often covering the same event.

Applying MMR re-ranking further improved coverage scores without degrading grounding, suggesting that diversity-aware retrieval helps surface complementary information relevant to stock analysis.

While hybrid retrieval improved recall, it did not eliminate hallucinations in generated answers. This highlights that retrieval diversity alone is insufficient for grounding; strict prompt constraints and evaluation-aware generation remain necessary.

MMR re-ranking improved contextual diversity but did not significantly alter grounding scores, indicating that hallucination is more sensitive to prompt discipline than retrieval strategy.

Additionally, free-tier API constraints influenced experimental design, necessitating reduced evaluation runs and rate limiting. This reflects real-world limitations often overlooked in prototype RAG systems.