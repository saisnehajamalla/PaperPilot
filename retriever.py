"""Hybrid retrieval: MiniLM semantic search (FAISS) blended with TF-IDF keyword scores.

Refusal is decided by two gates, not the combined score alone:
  - the combined score must reach MIN_FINAL_SCORE, and
  - the semantic score of the best candidate must reach MIN_SEMANTIC_SCORE.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    KEYWORD_WEIGHT,
    MIN_FINAL_SCORE,
    MIN_SEMANTIC_SCORE,
    SEMANTIC_WEIGHT,
    TOP_K,
)
from embeddings import Embedder
from vector_store import VectorStore


@dataclass
class Candidate:
    chunk: dict
    semantic: float
    keyword: float
    final: float


@dataclass
class RetrievalResult:
    question: str
    candidates: list[Candidate] = field(default_factory=list)
    accepted: bool = False
    reason: str = ""


NON_RETRIEVABLE_SECTIONS = {"References"}


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict],
        embedder: Embedder | None = None,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        min_final_score: float = MIN_FINAL_SCORE,
        min_semantic_score: float = MIN_SEMANTIC_SCORE,
    ):
        self.chunks = [c for c in chunks if c["section"] not in NON_RETRIEVABLE_SECTIONS]
        self.embedder = embedder or Embedder()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.min_final_score = min_final_score
        self.min_semantic_score = min_semantic_score

        self._store = VectorStore(self.embedder.dimension)
        if self.chunks:
            vectors = self.embedder.embed([c["text"] for c in self.chunks])
            self._store.add(vectors, self.chunks)

        self._tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._chunk_tfidf = (
            self._tfidf.fit_transform([c["text"] for c in self.chunks]) if self.chunks else None
        )

    def _keyword_scores(self, question: str) -> np.ndarray:
        if self._chunk_tfidf is None:
            return np.zeros(0)
        query = self._tfidf.transform([question])
        return (self._chunk_tfidf @ query.T).toarray().flatten()

    def score_all(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        """Return per-chunk semantic and keyword scores for a question."""
        semantic = np.zeros(len(self.chunks))
        if not self.chunks:
            return semantic, semantic
        query_vector = self.embedder.embed_query([question])[0]
        for position, score in self._store.search(query_vector, k=len(self.chunks)):
            semantic[position] = score
        return semantic, self._keyword_scores(question)

    def rank(
        self,
        semantic: np.ndarray,
        keyword: np.ndarray,
        top_k: int = TOP_K,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> list[Candidate]:
        sw = self.semantic_weight if semantic_weight is None else semantic_weight
        kw = self.keyword_weight if keyword_weight is None else keyword_weight
        final = sw * semantic + kw * keyword
        order = np.argsort(-final)[:top_k]
        return [
            Candidate(
                chunk=self.chunks[i],
                semantic=float(semantic[i]),
                keyword=float(keyword[i]),
                final=float(final[i]),
            )
            for i in order
        ]

    def decide(self, candidates: list[Candidate]) -> tuple[bool, str]:
        best = candidates[0]
        if best.final < self.min_final_score:
            return False, f"Best combined score {best.final:.3f} < {self.min_final_score}"
        if best.semantic < self.min_semantic_score:
            return False, f"Best semantic score {best.semantic:.3f} < {self.min_semantic_score}"
        return True, "Passed both gates"

    def retrieve(self, question: str, top_k: int = TOP_K) -> RetrievalResult:
        result = RetrievalResult(question=question)
        if not self.chunks or not question.strip():
            result.reason = "No indexed content or empty question."
            return result

        semantic, keyword = self.score_all(question)
        result.candidates = self.rank(semantic, keyword, top_k=top_k)
        result.accepted, result.reason = self.decide(result.candidates)
        return result
