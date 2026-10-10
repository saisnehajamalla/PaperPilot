"""Diagnose candidate recall vs reranking quality, and calibrate the reranker threshold.

Runs the two-stage pipeline with a large candidate pool (RERANK_TOP_N from config/env),
scores every candidate once, then reports:
  - per on-topic question: gold rank before reranking, whether it is in the top 20 / 21-40,
    and its rank after reranking within the 20-candidate and full pools
  - recall, rank-1, top-5, and refusal counts at RERANK_THRESHOLD
  - reranker score distributions and a threshold sweep over all off-topic trials

Usage:
    python diagnose_retrieval.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import sys

import numpy as np

from config import (
    EMBEDDING_MODEL,
    KEYWORD_WEIGHT,
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

TOP_K = 5


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def first_gold_rank(candidates, gold) -> int | None:
    return next((i + 1 for i, c in enumerate(candidates) if is_relevant(c.chunk["text"], gold)), None)


def main() -> None:
    specs = [arg.partition("=")[::2] for arg in sys.argv[1:]]
    if not specs:
        print(__doc__)
        sys.exit(1)

    pool = RERANK_TOP_N
    print("Configuration")
    print(f"  first stage : {EMBEDDING_MODEL} prefix={QUERY_PREFIX!r} sem={SEMANTIC_WEIGHT} kw={KEYWORD_WEIGHT}")
    print(f"  reranker    : {RERANKER_MODEL}")
    print(f"  pool size   : {pool}   threshold: {RERANK_THRESHOLD}\n")

    reranker = CrossEncoderReranker()
    on_rows, off_rows = [], []

    for name, path in specs:
        with open(path, "rb") as f:
            pages = extract_pages(f.read())
        chunks = build_chunks(clean_pages(pages), paper_name=name)
        pipeline = RetrievalPipeline(
            HybridRetriever(chunks), reranker, top_n=pool, final_top_k=pool, threshold=RERANK_THRESHOLD
        )

        for category, question, gold in TEST_SETS.get(name, []):
            if not gold:
                continue
            res = pipeline.retrieve(question)
            stage1 = res.stage1
            scores = {rc.candidate.chunk["chunk_id"]: rc.rerank_score for rc in res.reranked}
            reranked = res.reranked

            pre = first_gold_rank(stage1, gold)
            post_pool = next((i + 1 for i, rc in enumerate(reranked)
                              if is_relevant(rc.candidate.chunk["text"], gold)), None)
            pool20 = sorted(stage1[:20], key=lambda c: scores[c.chunk["chunk_id"]], reverse=True)
            post20 = next((i + 1 for i, c in enumerate(pool20) if is_relevant(c.chunk["text"], gold)), None)

            on_rows.append(dict(
                name=name, category=category, question=question,
                pre=pre, post20=post20, post_pool=post_pool,
                top1=reranked[0].rerank_score,
                rank1=is_relevant(reranked[0].candidate.chunk["text"], gold),
                top5=any(is_relevant(rc.candidate.chunk["text"], gold) for rc in reranked[:TOP_K]),
                top1_text=reranked[0].candidate.chunk["text"][:70],
            ))

        for question in off_topic_questions():
            res = pipeline.retrieve(question)
            off_rows.append(dict(name=name, question=question, top1=res.reranked[0].rerank_score))

    print("=== Per on-topic question: candidate recall vs reranking ===")
    print(f"{'paper':<10}{'category':<12}{'pre-rank':>9}{'in top20':>9}{'in 21-40':>9}"
          f"{'post(20)':>9}{'post(40)':>9}{'in top5':>8}  question")
    for r in on_rows:
        in20 = r["pre"] is not None and r["pre"] <= 20
        in2140 = r["pre"] is not None and 20 < r["pre"] <= 40
        print(f"{r['name']:<10}{r['category']:<12}{str(r['pre'] or '-'):>9}"
              f"{'Y' if in20 else 'n':>9}{'Y' if in2140 else 'n':>9}"
              f"{str(r['post20'] or '-'):>9}{str(r['post_pool'] or '-'):>9}"
              f"{'Y' if r['post_pool'] and r['post_pool'] <= TOP_K else 'n':>8}  {r['question']}")

    n = len(on_rows)
    rec20 = sum(r["pre"] is not None and r["pre"] <= 20 for r in on_rows)
    rec40 = sum(r["pre"] is not None and r["pre"] <= 40 for r in on_rows)
    rank1 = sum(r["rank1"] for r in on_rows)
    top5 = sum(r["top5"] for r in on_rows)
    on_top = np.array([r["top1"] for r in on_rows])
    off_top = np.array([r["top1"] for r in off_rows])
    false_ref = int((on_top < RERANK_THRESHOLD).sum())
    false_acc = int((off_top >= RERANK_THRESHOLD).sum())

    print(f"\n=== Summary (pool={pool}, threshold={RERANK_THRESHOLD}) ===")
    print(f"  Recall@20 (stage 1)          : {rec20}/{n}")
    print(f"  Recall@40 (stage 1)          : {rec40}/{n}")
    print(f"  Rank-1 accuracy (reranked)   : {rank1}/{n}")
    print(f"  Top-5 accuracy (reranked)    : {top5}/{n}")
    print(f"  False refusals (on-topic)    : {false_ref}/{n}")
    print(f"  False accepts (off-topic)    : {false_acc}/{len(off_rows)}")

    print("\n=== Score distributions (top-1 reranker score) ===")
    for label, arr in [("on-topic", on_top), ("off-topic", off_top)]:
        q = np.percentile(arr, [0, 25, 50, 75, 100])
        print(f"  {label:<10} n={len(arr):<3} min={q[0]:+.2f} q25={q[1]:+.2f} median={q[2]:+.2f} "
              f"q75={q[3]:+.2f} max={q[4]:+.2f}")
    print(f"  lowest on-topic top score : {on_top.min():+.3f}")
    print(f"  highest off-topic top score: {off_top.max():+.3f}")

    print("\n=== Threshold sweep (accept if top-1 reranker score >= threshold) ===")
    print(f"{'Threshold':>10} {'On-topic accepted':>18} {'False refusals':>15} "
          f"{'Off-topic rejected':>19} {'False accepts':>14}")
    grid = list(np.arange(-11.5, -9.5, 0.25)) + list(np.arange(-9.0, 4.0, 0.5))
    for thr in grid:
        acc = int((on_top >= thr).sum())
        rej = int((off_top < thr).sum())
        print(f"{thr:>10.2f} {acc:>12}/{len(on_top):<5} {len(on_top) - acc:>15} "
              f"{rej:>13}/{len(off_top):<5} {len(off_top) - rej:>14}")

    print("\n=== Highest-scoring off-topic questions (top-1 reranker score) ===")
    for r in sorted(off_rows, key=lambda r: r["top1"], reverse=True)[:8]:
        print(f"  {r['top1']:+.3f}  {r['name']:<10} {r['question']}")

    print("\n=== On-topic top-1 chunk after reranking (for questions with rank-1 miss) ===")
    for r in on_rows:
        if not r["rank1"]:
            print(f"  {r['top1']:+.3f} {r['name']:<10} {r['question'][:45]:<45} | {r['top1_text']}")


if __name__ == "__main__":
    main()
