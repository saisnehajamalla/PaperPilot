"""Calibrate retrieval weights and semantic refusal threshold against labeled test questions.

Each on-topic question lists gold substrings; a retrieved chunk counts as relevant if it
contains one of them. Off-topic questions have no gold passages and should be refused.

Usage:
    python calibrate_retrieval.py path/to/paper1.pdf path/to/paper2.pdf
"""

import os
import sys

import numpy as np

from pdf_processor import build_chunks, clean_pages, extract_pages
from retrieval_testset import TEST_SETS
from retriever import HybridRetriever

SEMANTIC_THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.55]
WEIGHTS = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
TOP_K = 5


def is_relevant(chunk_text: str, gold: list[str]) -> bool:
    return any(g in chunk_text for g in gold)


def load_paper(path: str, name: str) -> HybridRetriever:
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    chunks = build_chunks(clean_pages(pages), paper_name=name)
    return HybridRetriever(chunks, min_final_score=0.0, min_semantic_score=0.0)


def check_gold(retriever: HybridRetriever, name: str) -> None:
    for category, _, gold in TEST_SETS[name]:
        for g in gold:
            if not any(g in c["text"] for c in retriever.chunks):
                print(f"  WARNING gold not found in {name}: {category}: {g!r}")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    retrievers = {
        "attention": load_paper(sys.argv[1], "attention"),
        "resnet": load_paper(sys.argv[2], "resnet"),
    }
    for name, r in retrievers.items():
        print(f"{name}: {len(r.chunks)} retrievable chunks")
        check_gold(r, name)

    # Score once per question; weights and thresholds only re-rank and gate.
    scored = []
    for name, r in retrievers.items():
        for category, question, gold in TEST_SETS[name]:
            semantic, keyword = r.score_all(question)
            scored.append((name, category, question, gold, r, semantic, keyword))

    print("\n=== Weight sweep (rank-1 / top-5 relevance over 8 on-topic questions) ===")
    print(f"{'sem':>5} {'kw':>5} {'rank1':>7} {'top5':>7}  per-category rank1/top5")
    for sw, kw in WEIGHTS:
        r1 = t5 = total = 0
        per_cat: dict[str, list[str]] = {}
        for name, category, _, gold, r, semantic, keyword in scored:
            if not gold:
                continue
            cands = r.rank(semantic, keyword, top_k=TOP_K, semantic_weight=sw, keyword_weight=kw)
            hit1 = is_relevant(cands[0].chunk["text"], gold)
            hit5 = any(is_relevant(c.chunk["text"], gold) for c in cands)
            total += 1
            r1 += hit1
            t5 += hit5
            per_cat.setdefault(category, []).append(f"{'Y' if hit1 else 'n'}/{'Y' if hit5 else 'n'}")
        cats = "  ".join(f"{k}:{'+'.join(v)}" for k, v in per_cat.items())
        print(f"{sw:>5.1f} {kw:>5.1f} {r1:>3}/{total:<3} {t5:>3}/{total:<3}  {cats}")

    print("\n=== Semantic threshold sweep per weight (gate on top candidate's semantic score) ===")
    print(f"{'sem':>5} {'kw':>5} {'thr':>6} {'accepted on-topic':>18} {'false refusals':>15} {'off-topic rejected':>19} {'false accepts':>14}")
    for sw, kw in WEIGHTS:
        for thr in SEMANTIC_THRESHOLDS:
            on_total = on_ok = off_total = off_ok = 0
            for name, category, _, gold, r, semantic, keyword in scored:
                top = r.rank(semantic, keyword, top_k=1, semantic_weight=sw, keyword_weight=kw)[0]
                accepted = top.semantic >= thr
                if gold:
                    on_total += 1
                    on_ok += accepted
                else:
                    off_total += 1
                    off_ok += not accepted
            print(
                f"{sw:>5.1f} {kw:>5.1f} {thr:>6.2f} {on_ok:>10}/{on_total:<7} "
                f"{on_total - on_ok:>15} {off_ok:>12}/{off_total:<6} {off_total - off_ok:>14}"
            )

    print("\n=== Top-1 semantic scores per question (on-topic vs off-topic) ===")
    for name, category, _, gold, r, semantic, keyword in scored:
        top = r.rank(semantic, keyword, top_k=1, semantic_weight=0.7, keyword_weight=0.3)[0]
        kind = "on " if gold else "off"
        print(f"  {kind} {name:<10} {category:<13} top_sem={top.semantic:.3f}  top_kw={top.keyword:.3f}")


if __name__ == "__main__":
    main()
