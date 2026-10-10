"""Section-aware candidate boosting.

Natural questions like "What problem does this paper solve?" often don't resemble the
paper's own wording, so the abstract can rank outside the first-stage top-N purely on
embedding similarity. This module detects a coarse question intent and, if the intent
maps to specific sections, guarantees that the best-scoring chunk(s) from those sections
are included in the candidate pool before reranking — it does not change any score used
for ranking, it only relaxes which chunks reach the reranker.

This is a heuristic, not a parser: it depends on the paper's sections being labeled with
recognizable names (pdf_processor.detect_sections). Papers with unusual section titles
(e.g. a custom "Deep Reinforcement Learning" heading) may not benefit, because the chunk
ends up labeled with whatever standard heading preceded it.
"""

import re

from retriever import Candidate

# Ordered: the first matching intent wins.
INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("future_work", ("future work", "future", "further work", "next step", "remaining challenge")),
    ("limitation", ("limitation", "weakness", "shortcoming", "drawback", "limits")),
    ("contribution", ("contribution", "novelty", "propose", "introduce")),
    ("results", ("result", "accuracy", "performance", "compare", "score", "bleu", "error", "outperform")),
    ("problem", ("problem", "solve", "address", "motivation", "challenge", "objective", "aim")),
    ("dataset", ("dataset", "data", "benchmark", "corpus", "environment", "games")),
    ("methodology", ("method", "architecture", "approach", "model", "algorithm", "optimizer", "train")),
]

# Intent-specific, not applied globally: evaluation showed the embedding model can rank
# the true Abstract/Introduction chunk below every other chunk in its own section for
# "problem"/"contribution"-style questions, so FAISS alone may never surface it. Boosting
# showed no effect for limitation/future-work (their gold chunks were already in the
# unboosted top-20), so those intents are intentionally not included here.
INTENT_SECTIONS: dict[str, tuple[str, ...]] = {
    "problem": ("Abstract", "Introduction"),
    "contribution": ("Abstract", "Introduction"),
}


def detect_intent(question: str) -> str | None:
    text = question.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(re.search(rf"\b{re.escape(k)}", text) for k in keywords):
            return intent
    return None


def boost_candidates(
    question: str,
    chunks: list[dict],
    semantic,
    keyword,
    semantic_weight: float,
    keyword_weight: float,
    max_extra: int = 2,
) -> list[Candidate]:
    """Return up to max_extra candidates from the sections tied to the question's intent."""
    intent = detect_intent(question)
    sections = INTENT_SECTIONS.get(intent or "", ())
    if not sections:
        return []
    final = semantic_weight * semantic + keyword_weight * keyword
    idxs = [i for i, c in enumerate(chunks) if c["section"] in sections]
    idxs.sort(key=lambda i: -final[i])
    return [
        Candidate(chunk=chunks[i], semantic=float(semantic[i]), keyword=float(keyword[i]), final=float(final[i]))
        for i in idxs[:max_extra]
    ]
