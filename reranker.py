"""Reranker interface and cross-encoder implementation.

The pipeline depends only on the Reranker protocol, so another model can replace
CrossEncoderReranker without changing retrieval_pipeline.py.
"""

from typing import Protocol

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


class Reranker(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage, higher is more relevant."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device="cpu")

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(question, passage) for passage in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
