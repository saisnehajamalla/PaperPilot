"""Compare baseline retrieval against section-aware candidate boosting on the 3-paper test set.

Both variants share chunks, embeddings, reranker, and threshold. Only whether the
candidate pool is widened with section-targeted chunks differs.

Usage:
    python compare_section_aware.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import sys
from collections import defaultdict

import numpy as np

from config import RERANK_TOP_N, SECTION_BOOST_MAX_EXTRA
from pdf_processor import build_chunks, clean_pages, extract_pages
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retrieval_testset import TEST_SETS, off_topic_questions
from retriever import HybridRetriever
from section_prior import detect_intent

THRESHOLD = -8.0
TOP_K = 5


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def main() -> None:
    specs = [arg.partition("=")[::2] for arg in sys.argv[1:]]
    if not specs:
        print(__doc__)
        sys.exit(1)

    reranker = CrossEncoderReranker()
    variants = {"baseline": False, "section_aware": True}

    on = defaultdict(lambda: {v: defaultdict(int) for v in variants})
    off_scores = {v: [] for v in variants}
    per_question = []

    for name, path in specs:
        with open(path, "rb") as f:
            chunks = build_chunks(clean_pages(extract_pages(f.read())), paper_name=name)
        retriever = HybridRetriever(chunks)
        pipes = {
            v: RetrievalPipeline(retriever, reranker, top_n=RERANK_TOP_N, final_top_k=RERANK_TOP_N,
                                 threshold=THRESHOLD, section_aware=flag,
                                 section_boost_max_extra=SECTION_BOOST_MAX_EXTRA)
            for v, flag in variants.items()
        }

        for category, question, gold in TEST_SETS.get(name, []):
            if not gold:
                continue
            row = dict(paper=name, category=category, question=question,
                       intent=detect_intent(question) or "-")
            for v, pipe in pipes.items():
                res = pipe.retrieve(question)
                pool, rr = res.stage1, res.reranked
                top1 = rr[0].rerank_score
                for bucket in (on[category][v], on["ALL"][v]):
                    bucket["n"] += 1
                    bucket["rank1"] += is_relevant(rr[0].candidate.chunk["text"], gold)
                    bucket["top5"] += any(is_relevant(x.candidate.chunk["text"], gold) for x in rr[:TOP_K])
                    bucket["recall_pool"] += any(is_relevant(c.chunk["text"], gold) for c in pool)
                    bucket["accepted"] += top1 >= THRESHOLD
                    bucket["false_refusal"] += top1 < THRESHOLD
                pre = next((i + 1 for i, c in enumerate(pool) if is_relevant(c.chunk["text"], gold)), None)
                post = next((i + 1 for i, x in enumerate(rr) if is_relevant(x.candidate.chunk["text"], gold)), None)
                row[v] = (pre, post, len(pool))
            per_question.append(row)

        for question in off_topic_questions():
            for v, pipe in pipes.items():
                off_scores[v].append(pipe.retrieve(question).reranked[0].rerank_score)

    print(f"Threshold {THRESHOLD}, pool {RERANK_TOP_N} (+up to {SECTION_BOOST_MAX_EXTRA} boosted), "
          f"final top-K {TOP_K}\n")
    print(f"{'Category':<12}{'N':>3}   {'Rank-1 base/sec':>18} {'Top-5 base/sec':>16} "
          f"{'Recall@pool base/sec':>23} {'False ref. base/sec':>21}")
    for cat in list(on.keys()):
        if cat == "ALL":
            continue
        b, s = on[cat]["baseline"], on[cat]["section_aware"]
        n = b["n"]
        print(f"{cat:<12}{n:>3}   {b['rank1']:>6}/{n} vs {s['rank1']}/{n}   "
              f"{b['top5']:>6}/{n} vs {s['top5']}/{n}   {b['recall_pool']:>9}/{n} vs {s['recall_pool']}/{n}   "
              f"{b['false_refusal']:>9}/{n} vs {s['false_refusal']}/{n}")
    b, s = on["ALL"]["baseline"], on["ALL"]["section_aware"]
    n = b["n"]
    print(f"{'ALL on-topic':<12}{n:>3}   {b['rank1']:>6}/{n} vs {s['rank1']}/{n}   "
          f"{b['top5']:>6}/{n} vs {s['top5']}/{n}   {b['recall_pool']:>9}/{n} vs {s['recall_pool']}/{n}   "
          f"{b['false_refusal']:>9}/{n} vs {s['false_refusal']}/{n}")

    print("\nOff-topic (threshold applied to top-1 reranker score)")
    for v in variants:
        arr = np.array(off_scores[v])
        fa = int((arr >= THRESHOLD).sum())
        print(f"  {v:<14} trials={len(arr)}  max={arr.max():+.3f}  median={np.median(arr):+.3f}  "
              f"false accepts={fa}/{len(arr)}")

    print("\nPer-question (rank before -> rank after, pool size; '-' = absent)")
    print(f"{'paper':<10}{'category':<12}{'intent':<13}{'baseline':>18}{'section_aware':>18}  question")
    def fmt(pre, post, pool_size):
        return f"{pre or '-'}->{post or '-'} (n={pool_size})"
    for row in per_question:
        print(f"{row['paper']:<10}{row['category']:<12}{row['intent']:<13}"
              f"{fmt(*row['baseline']):>18}{fmt(*row['section_aware']):>18}  {row['question']}")


if __name__ == "__main__":
    main()
