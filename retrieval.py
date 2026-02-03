import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents):
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents")

        self.documents = documents
        self.texts = [d["title"] + " " + d["text"] for d in documents]

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.texts)

        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve_semantic(self, query, k=5):
        q = self.model.encode([query])[0]
        scores = np.dot(self.embeddings, q)
        idx = scores.argsort()[-k:][::-1]
        return [self.documents[i] for i in idx]

    def retrieve_keyword(self, query, k=5):
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in idx]

    def retrieve_hybrid(self, query, k=5):
        q = self.model.encode([query])[0]
        sem = np.dot(self.embeddings, q)
        key = self.bm25.get_scores(query.lower().split())
        combined = sem + key
        idx = combined.argsort()[-k:][::-1]
        return [self.documents[i] for i in idx]

    def mmr_rerank(self, query, candidate_docs, k=5, lambda_param=0.7):
        query_vec = self.model.encode([query])[0]

        doc_texts = [d["title"] + " " + d["text"] for d in candidate_docs]
        doc_vecs = self.model.encode(doc_texts)

        selected = []
        selected_indices = []

        similarities = np.dot(doc_vecs, query_vec)

        for _ in range(min(k, len(candidate_docs))):
            if not selected:
                idx = np.argmax(similarities)
            else:
                diversity = np.max(
                    np.dot(doc_vecs, doc_vecs[selected_indices].T),
                    axis=1
                )
                mmr_score = lambda_param * similarities - (1 - lambda_param) * diversity
                idx = np.argmax(mmr_score)

            selected.append(candidate_docs[idx])
            selected_indices.append(idx)

        return selected
    
    def retrieve_hybrid_mmr(self, query, k=5):
        hybrid = self.retrieve_hybrid(query, k=10)
        return self.mmr_rerank(query, hybrid, k=k)
