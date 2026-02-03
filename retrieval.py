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
