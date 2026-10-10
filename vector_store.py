"""FAISS-backed vector store that keeps chunk metadata alongside the index.

Kept behind a small interface (add / search / __len__) so a pgvector
implementation can replace it later without changing the RAG pipeline.
"""

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self._metadata)

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        if vectors.shape[0] != len(metadatas):
            raise ValueError("Number of vectors and metadata entries must match.")
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}.")
        self._index.add(vectors)
        self._metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (position, similarity) pairs, highest similarity first."""
        if len(self) == 0:
            return []
        k = min(k, len(self))
        scores, ids = self._index.search(query_vector.reshape(1, -1), k)
        return [(int(idx), float(score)) for score, idx in zip(scores[0], ids[0]) if idx != -1]

    def metadata(self, position: int) -> dict:
        return self._metadata[position]
