# Ablation Study of Retrieval Strategies for Stock-News RAG Systems

This repository contains the code and experimental setup for a research-style ablation study on Retrieval-Augmented Generation (RAG) systems applied to stock-news question answering.

The project investigates how different retrieval strategies affect answer quality, with a particular focus on coverage, grounding, and retrieval relevance.

---

## 🔍 Research Motivation

Large Language Models (LLMs) often hallucinate or provide incomplete answers when queried about real-world information. Retrieval-Augmented Generation (RAG) mitigates this issue by conditioning generation on external documents.

However, retrieval itself remains a major failure mode. In particular:
- Semantic retrieval may miss exact facts
- Hybrid retrieval improves recall but introduces redundancy
- Increased recall does not guarantee grounded answers

This project empirically studies these trade-offs.

---

## 🧪 Experimental Setup

### Retrieval Strategies Compared
- **Semantic Retrieval**: Sentence embeddings and cosine similarity
- **Hybrid Retrieval**: Semantic similarity combined with BM25 keyword scoring
- **Hybrid + MMR**: Hybrid retrieval followed by Maximum Marginal Relevance (MMR) re-ranking for diversity

### Generation
- Gemini (2.5 Flash)
- Strict context-only prompting
- Temperature set to 0 for determinism

### Evaluation
- Automatic evaluation using an LLM-as-a-judge
- Metrics (0–2 scale):
  - Retrieval Relevance
  - Grounding
  - Coverage

---

## 📊 Key Findings

- Hybrid retrieval improves coverage compared to semantic retrieval
- MMR further increases contextual diversity and coverage
- Grounding scores remain largely unchanged across retrieval strategies
- Retrieval diversity alone does not eliminate hallucinations

These results indicate that retrieval optimization and grounding enforcement are complementary but distinct problems.

---

## 📁 Repository Structure

```text
.
├── retrieval.py          # Semantic, hybrid, and MMR-based retrieval
├── generate.py           # Gemini-based generation
├── auto_eval.py          # LLM-as-a-judge evaluation
├── ablation.py           # Ablation experiment runner
├── dataset_snapshot.json # Frozen dataset for reproducibility
├── paper.pdf             # Research paper (compiled)
└── README.md
