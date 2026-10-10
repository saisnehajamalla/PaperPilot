"""Two-stage retrieval: first-stage candidates (BGE + FAISS hybrid), then cross-encoder reranking.

Stage 1 is unchanged and only selects candidates. Stage 2 scores each (question, chunk)
pair directly. Refusal depends only on the reranker score of the best reranked chunk.
"""

from dataclasses import dataclass, field

from config import (
    FINAL_TOP_K,
    RERANK_THRESHOLD,
    RERANK_TOP_N,
    SECTION_AWARE_RETRIEVAL,
    SECTION_BOOST_MAX_EXTRA,
)
from reranker import Reranker
from retriever import Candidate, HybridRetriever
from section_prior import boost_candidates


@dataclass
class RankedChunk:
    candidate: Candidate
    rerank_score: float


@dataclass
class PipelineResult:
    question: str
    stage1: list[Candidate] = field(default_factory=list)
    reranked: list[RankedChunk] = field(default_factory=list)
    accepted: bool = False
    reason: str = ""


class RetrievalPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        top_n: int = RERANK_TOP_N,
        final_top_k: int = FINAL_TOP_K,
        threshold: float = RERANK_THRESHOLD,
        section_aware: bool = SECTION_AWARE_RETRIEVAL,
        section_boost_max_extra: int = SECTION_BOOST_MAX_EXTRA,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_n = top_n
        self.final_top_k = final_top_k
        self.threshold = threshold
        self.section_aware = section_aware
        self.section_boost_max_extra = section_boost_max_extra

    def retrieve(self, question: str) -> PipelineResult:
        result = PipelineResult(question=question)
        if not self.retriever.chunks or not question.strip():
            result.reason = "No indexed content or empty question."
            return result

        semantic, keyword = self.retriever.score_all(question)
        pool = self.retriever.rank(semantic, keyword, top_k=self.top_n)

        if self.section_aware:
            boosted = boost_candidates(
                question, self.retriever.chunks, semantic, keyword,
                self.retriever.semantic_weight, self.retriever.keyword_weight,
                max_extra=self.section_boost_max_extra,
            )
            existing = {c.chunk["chunk_id"] for c in pool}
            pool = pool + [c for c in boosted if c.chunk["chunk_id"] not in existing]

        result.stage1 = pool
        scores = self.reranker.score(question, [c.chunk["text"] for c in result.stage1])
        ordered = sorted(zip(result.stage1, scores), key=lambda pair: pair[1], reverse=True)
        result.reranked = [RankedChunk(c, s) for c, s in ordered[: self.final_top_k]]

        best = result.reranked[0].rerank_score
        result.accepted = best >= self.threshold
        result.reason = (
            f"Top reranker score {best:.3f} {'>=' if result.accepted else '<'} {self.threshold}"
        )
        return result
