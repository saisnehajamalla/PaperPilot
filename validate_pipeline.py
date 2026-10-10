"""Evaluate the two-stage pipeline (BGE+FAISS hybrid candidates, then cross-encoder reranking).

Reranker scores are computed once per question; threshold sweeps reuse them.

Usage:
    python validate_pipeline.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import sys
from collections import defaultdict

import numpy as np

from config import (
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    KEYWORD_WEIGHT,
    MIN_SEMANTIC_SCORE,
    QUERY_PREFIX,
    RERANK_THRESHOLD,
    RERANK_TOP_N,
    RERANKER_MODEL,
    SEMANTIC_WEIGHT,
)
from pdf_processor import build_chunks, clean_pages, extract_pages
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retrieval_testset import TEST_SETS, off_topic_questions
from retriever import HybridRetriever

DEBUG_QUESTIONS = [
    "What methodology was used?",
    "What architecture did the network use?",
    "What problem does the paper solve?",
    "What problem does the paper address?",
    "What were the results?",
    "What is the capital of France?",
    "What is the boiling point of water at sea level?",
]


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def load(name: str, path: str, pipeline_factory) -> RetrievalPipeline:
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    chunks = build_chunks(clean_pages(pages), paper_name=name)
    return pipeline_factory(HybridRetriever(chunks))


def describe(candidate, score=None) -> str:
    score_txt = f" rr={score:+.3f}" if score is not None else ""
    return (
        f"sem={candidate.semantic:.3f} kw={candidate.keyword:.3f} hyb={candidate.final:.3f}{score_txt}"
        f"  p{candidate.chunk['page']} {candidate.chunk['section']:<14} | {candidate.chunk['text'][:70]}"
    )


def print_debug(name: str, question: str, result) -> None:
    print(f"\n--- [{name}] Question: {question}")
    print("Candidate ranking BEFORE reranking (first-stage hybrid, top 3):")
    for i, c in enumerate(result.stage1[:3], 1):
        print(f"  {i}. {describe(c)}")
    print("Ranking AFTER reranking (top 3):")
    for i, rc in enumerate(result.reranked[:3], 1):
        print(f"  {i}. {describe(rc.candidate, rc.rerank_score)}")
    print("Reranker scores (all candidates, in stage-1 order):")
    scores = [rc.rerank_score for rc in sorted(result.reranked, key=lambda r: r.candidate.final, reverse=True)]
    print("  " + ", ".join(f"{s:+.2f}" for s in scores))


def main() -> None:
    specs = [arg.partition("=")[::2] for arg in sys.argv[1:]]
    if not specs:
        print(__doc__)
        sys.exit(1)

    print("Configuration")
    print(f"  first stage     : {EMBEDDING_MODEL} (query prefix {QUERY_PREFIX!r}), "
          f"semantic={SEMANTIC_WEIGHT} keyword={KEYWORD_WEIGHT}")
    print(f"  reranker        : {RERANKER_MODEL}")
    print(f"  candidates/top-K: {RERANK_TOP_N} / {FINAL_TOP_K}")
    print(f"  threshold       : {RERANK_THRESHOLD} (reranker raw score)")
    print(f"  (first-stage semantic gate {MIN_SEMANTIC_SCORE} is not used by the pipeline)\n")

    reranker = CrossEncoderReranker()

    def factory(retriever):
        return RetrievalPipeline(
            retriever,
            reranker,
            top_n=RERANK_TOP_N,
            final_top_k=RERANK_TOP_N,  # keep full ordering so top-K can be cut later
            threshold=RERANK_THRESHOLD,
        )

    trials = []  # dicts with name, category, question, gold, result
    for name, path in specs:
        pipeline = load(name, path, factory)
        print(f"{name}: {len(pipeline.retriever.chunks)} retrievable chunks")
        for category, question, gold in TEST_SETS.get(name, []):
            if gold:
                trials.append(dict(name=name, category=category, question=question, gold=gold,
                                   result=pipeline.retrieve(question)))
        for question in off_topic_questions():
            trials.append(dict(name=name, category="Off-topic", question=question, gold=[],
                               result=pipeline.retrieve(question)))
        for question in DEBUG_QUESTIONS:
            if any(t["name"] == name and t["question"] == question for t in trials):
                print_debug(name, question, next(t["result"] for t in trials
                                                  if t["name"] == name and t["question"] == question))
    print()

    on = [t for t in trials if t["gold"]]
    off = [t for t in trials if not t["gold"]]

    def top_scores(ts):
        return np.array([t["result"].reranked[0].rerank_score for t in ts])

    on_top = top_scores(on)
    off_top = top_scores(off)

    print("=== Reranker top-1 score distribution (raw cross-encoder logits) ===")
    for label, arr in [("on-topic", on_top), ("off-topic", off_top)]:
        q = np.percentile(arr, [0, 25, 50, 75, 100])
        print(f"  {label:<10} n={len(arr):<3} min={q[0]:+.2f} q25={q[1]:+.2f} median={q[2]:+.2f} "
              f"q75={q[3]:+.2f} max={q[4]:+.2f}")
    print(f"  lowest on-topic top score : {on_top.min():+.3f}")
    print(f"  highest off-topic top score: {off_top.max():+.3f}")
    print(f"  separable (gap > 0)       : {on_top.min() > off_top.max()}\n")

    print("=== Threshold sweep (accept if top-1 reranker score >= threshold) ===")
    print(f"{'threshold':>10} {'on accepted':>13} {'false refusals':>15} {'off rejected':>13} {'false accepts':>14}")
    lo, hi = np.floor(min(on_top.min(), off_top.min())), np.ceil(max(on_top.max(), off_top.max()))
    for thr in np.linspace(lo, hi, int(hi - lo) * 2 + 1):
        on_ok = int((on_top >= thr).sum())
        off_rej = int((off_top < thr).sum())
        print(f"{thr:>10.2f} {on_ok:>7}/{len(on_top):<5} {len(on_top) - on_ok:>15} "
              f"{off_rej:>8}/{len(off_top):<4} {len(off_top) - off_rej:>14}")
    print()

    # Category report uses RERANK_THRESHOLD from config.
    totals = defaultdict(lambda: dict(n=0, s1_r1=0, s1_t5=0, r_r1=0, r_t5=0, acc=0, recall20=0))
    for t in trials:
        if not t["gold"]:
            continue
        res = t["result"]
        s1 = res.stage1[:FINAL_TOP_K]
        rr = res.reranked[:FINAL_TOP_K]
        tot = totals[t["category"]]
        tot["n"] += 1
        tot["recall20"] += any(is_relevant(c.chunk["text"], t["gold"]) for c in res.stage1)
        tot["s1_r1"] += is_relevant(s1[0].chunk["text"], t["gold"])
        tot["s1_t5"] += any(is_relevant(c.chunk["text"], t["gold"]) for c in s1)
        tot["r_r1"] += is_relevant(rr[0].candidate.chunk["text"], t["gold"])
        tot["r_t5"] += any(is_relevant(c.candidate.chunk["text"], t["gold"]) for c in rr)
        tot["acc"] += res.reranked[0].rerank_score >= RERANK_THRESHOLD

    print(f"=== Category results at RERANK_THRESHOLD={RERANK_THRESHOLD} ===")
    print(f"{'Category':<14}{'N':>3}{'Rank-1':>9}{'Top-5':>9}{'Accepted':>11}{'False ref.':>12}"
          f"{'Recall@20':>11}   | stage-1 Rank-1 / Top-5")
    all_t = dict(n=0, r_r1=0, r_t5=0, acc=0, s1_r1=0, s1_t5=0, recall20=0)
    for cat, v in totals.items():
        print(f"{cat:<14}{v['n']:>3}{v['r_r1']:>6}/{v['n']:<2}{v['r_t5']:>6}/{v['n']:<2}"
              f"{v['acc']:>8}/{v['n']:<2}{v['n'] - v['acc']:>12}{v['recall20']:>8}/{v['n']:<2}"
              f"   | {v['s1_r1']}/{v['n']} / {v['s1_t5']}/{v['n']}")
        for k in all_t:
            all_t[k] += v[k]
    print(f"{'All on-topic':<14}{all_t['n']:>3}{all_t['r_r1']:>6}/{all_t['n']:<2}{all_t['r_t5']:>6}/{all_t['n']:<2}"
          f"{all_t['acc']:>8}/{all_t['n']:<2}{all_t['n'] - all_t['acc']:>12}{all_t['recall20']:>8}/{all_t['n']:<2}   | "
          f"{all_t['s1_r1']}/{all_t['n']} / {all_t['s1_t5']}/{all_t['n']}")
    off_acc = sum(t["result"].reranked[0].rerank_score >= RERANK_THRESHOLD for t in off)
    print(f"\nOff-topic: rejected {len(off) - off_acc}/{len(off)}, false accepts {off_acc}")


if __name__ == "__main__":
    main()
