"""Sentence Transformers embedding wrapper (CPU, normalized vectors)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, QUERY_PREFIX


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, query_prefix: str = QUERY_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = _load_model(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents. Returns L2-normalized float32 vectors (inner product = cosine)."""
        return self._encode(texts)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        """Embed queries, applying the configured query prefix if one is set."""
        return self._encode([f"{self.query_prefix}{t}" for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
