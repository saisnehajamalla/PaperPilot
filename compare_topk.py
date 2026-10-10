"""Compare FINAL_TOP_K=5 vs FINAL_TOP_K=10 on the 3-paper test set.

Settings held fixed: BGE, no query reformulation, weights 0.8/0.2, FAISS pool 20,
section-aware retrieval on, cross-encoder reranker, threshold -8.0. Only how many of
the already-ranked candidates are exposed to the (future) LLM differs.

Usage:
    python compare_topk.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import sys
from collections import defaultdict

import numpy as np

from config import RERANK_THRESHOLD, RERANK_TOP_N, SECTION_BOOST_MAX_EXTRA
from pdf_processor import build_chunks, clean_pages, extract_pages
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retrieval_testset import TEST_SETS, off_topic_questions
from retriever import HybridRetriever

TOPKS = (5, 10)


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def main() -> None:
    specs = [arg.partition("=")[::2] for arg in sys.argv[1:]]
    if not specs:
        print(__doc__)
        sys.exit(1)

    reranker = CrossEncoderReranker()
    totals = defaultdict(lambda: defaultdict(int))
    newly_covered = []  # questions with gold absent at top-5 but present at top-10
    off_top1 = []

    for name, path in specs:
        with open(path, "rb") as f:
            chunks = build_chunks(clean_pages(extract_pages(f.read())), paper_name=name)
        retriever = HybridRetriever(chunks)
        pipe = RetrievalPipeline(retriever, reranker, top_n=RERANK_TOP_N, final_top_k=50,
                                 threshold=RERANK_THRESHOLD, section_aware=True,
                                 section_boost_max_extra=SECTION_BOOST_MAX_EXTRA)

        for category, question, gold in TEST_SETS.get(name, []):
            if not gold:
                continue
            res = pipe.retrieve(question)
            rr = res.reranked
            top1 = rr[0].rerank_score
            recall20 = any(is_relevant(c.chunk["text"], gold) for c in res.stage1[:RERANK_TOP_N])
            hit = {k: any(is_relevant(x.candidate.chunk["text"], gold) for x in rr[:k]) for k in TOPKS}
            rank1 = is_relevant(rr[0].candidate.chunk["text"], gold)

            for cat in (category, "ALL"):
                t = totals[cat]
                t["n"] += 1
                t["rank1"] += rank1
                t["recall20"] += recall20
                t["accepted"] += top1 >= RERANK_THRESHOLD
                t["false_refusal"] += top1 < RERANK_THRESHOLD
                for k in TOPKS:
                    t[f"top{k}"] += hit[k]

            if not hit[5] and hit[10]:
                gold_rank = next(i + 1 for i, x in enumerate(rr) if is_relevant(x.candidate.chunk["text"], gold))
                newly_covered.append((name, category, question, gold_rank))

        for question in off_topic_questions():
            off_top1.append(pipe.retrieve(question).reranked[0].rerank_score)

    off_arr = np.array(off_top1)
    false_accepts = int((off_arr >= RERANK_THRESHOLD).sum())

    print(f"Threshold {RERANK_THRESHOLD}, pool {RERANK_TOP_N}, section-aware on\n")
    print(f"{'Category':<14}{'N':>3}{'Rank-1':>8}{'Top-5':>8}{'Top-10':>8}{'Recall@20':>11}"
          f"{'False ref.':>12}")
    for cat, t in totals.items():
        if cat == "ALL":
            continue
        n = t["n"]
        print(f"{cat:<14}{n:>3}{t['rank1']:>5}/{n:<2}{t['top5']:>5}/{n:<2}{t['top10']:>5}/{n:<2}"
              f"{t['recall20']:>8}/{n:<2}{t['false_refusal']:>9}/{n:<2}")
    t = totals["ALL"]
    n = t["n"]
    print(f"{'ALL on-topic':<14}{n:>3}{t['rank1']:>5}/{n:<2}{t['top5']:>5}/{n:<2}{t['top10']:>5}/{n:<2}"
          f"{t['recall20']:>8}/{n:<2}{t['false_refusal']:>9}/{n:<2}")

    print(f"\nOff-topic: {len(off_arr)} trials, false accepts = {false_accepts}/{len(off_arr)} "
          f"(identical for top-5 and top-10: refusal only uses the top-1 score)")

    print(f"\nQuestions where the gold chunk enters coverage between top-5 and top-10: "
          f"{len(newly_covered)}")
    for name, cat, q, rank in newly_covered:
        print(f"  {name:<10} {cat:<12} rank {rank:>2}  {q}")


if __name__ == "__main__":
    main()
