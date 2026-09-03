"""
vector_store.py
----------------
Lightweight FAISS-backed vector store used for chunk retrieval (the "R"
in RAG). Uses an inner-product index over L2-normalized vectors, which
is mathematically equivalent to cosine similarity search — entirely
local, no external services required.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


class FAISSVectorStore:
    """A minimal in-memory FAISS index paired with parallel metadata."""

    def __init__(self, dim: int):
        import faiss  # imported lazily so the module loads even without faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Add a batch of embeddings along with their associated metadata
        (e.g. {"source": filename, "chunk_text": chunk})."""
        if embeddings.shape[0] == 0:
            return
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("embeddings and metadatas must be the same length")

        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return the top_k (metadata, similarity_score) pairs for a query vector."""
        if self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype("float32")
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def size(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        self.index = self._faiss.IndexFlatIP(self.dim)
        self.metadata = []
