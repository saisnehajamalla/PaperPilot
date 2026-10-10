"""Validate the configured retrieval pipeline across labeled papers.

Uses the settings from .env / environment (embedding model, weights, thresholds, query prefix).
Nothing is tuned here.

Usage:
    python validate_retrieval.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import os
import sys
from collections import defaultdict

from config import (
    EMBEDDING_MODEL,
    KEYWORD_WEIGHT,
    MIN_FINAL_SCORE,
    MIN_SEMANTIC_SCORE,
    QUERY_PREFIX,
    SEMANTIC_WEIGHT,
)
from pdf_processor import build_chunks, clean_pages, extract_pages
from retrieval_testset import TEST_SETS, off_topic_questions
from retriever import HybridRetriever

TOP_K = 5


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def load(name: str, path: str) -> HybridRetriever:
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    chunks = build_chunks(clean_pages(pages), paper_name=name)
    return HybridRetriever(chunks)


def main() -> None:
    specs = []
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        specs.append((name, path))
    if not specs:
        print(__doc__)
        sys.exit(1)

    print("Configuration")
    print(f"  embedding model : {EMBEDDING_MODEL}")
    print(f"  query prefix    : {QUERY_PREFIX!r}")
    print(f"  weights         : semantic={SEMANTIC_WEIGHT} keyword={KEYWORD_WEIGHT}")
    print(f"  gates           : min_semantic={MIN_SEMANTIC_SCORE} min_final={MIN_FINAL_SCORE}")
    print(f"  top_k           : {TOP_K}\n")

    totals = defaultdict(lambda: {"n": 0, "rank1": 0, "top5": 0, "accepted": 0})
    off = {"n": 0, "rejected": 0}

    for name, path in specs:
        retriever = load(name, path)
        print(f"=== {name}: {len(retriever.chunks)} retrievable chunks ===")
        if name not in TEST_SETS:
            print("  no test set defined for this paper\n")
            continue

        for category, question, gold in TEST_SETS[name]:
            if not gold:
                continue
            result = retriever.retrieve(question, top_k=TOP_K)
            cands = result.candidates
            rank1 = is_relevant(cands[0].chunk["text"], gold)
            top5 = any(is_relevant(c.chunk["text"], gold) for c in cands)
            hit_rank = next((i + 1 for i, c in enumerate(cands) if is_relevant(c.chunk["text"], gold)), "-")
            t = totals[category]
            t["n"] += 1
            t["rank1"] += rank1
            t["top5"] += top5
            t["accepted"] += result.accepted
            print(
                f"  [{category}] {question}\n"
                f"      accepted={result.accepted!s:<5} top1_sem={cands[0].semantic:.3f} "
                f"rank1={'Y' if rank1 else 'n'} top5={'Y' if top5 else 'n'} first_gold_rank={hit_rank}\n"
                f"      rank1 chunk: p{cands[0].chunk['page']} {cands[0].chunk['section']}: "
                f"{cands[0].chunk['text'][:80]}"
            )

        for question in off_topic_questions():
            result = retriever.retrieve(question, top_k=TOP_K)
            off["n"] += 1
            off["rejected"] += not result.accepted
            print(
                f"  [Off-topic] {question}\n"
                f"      accepted={result.accepted!s:<5} top1_sem={result.candidates[0].semantic:.3f} "
                f"top1_final={result.candidates[0].final:.3f}"
            )
        print()

    on_n = sum(v["n"] for v in totals.values())
    on_r1 = sum(v["rank1"] for v in totals.values())
    on_t5 = sum(v["top5"] for v in totals.values())
    on_acc = sum(v["accepted"] for v in totals.values())

    print("=== Summary by category (all papers) ===")
    print(f"{'Category':<14}{'n':>4}{'rank1':>10}{'top5':>10}{'accepted':>12}{'false refusals':>16}")
    for category, v in totals.items():
        print(
            f"{category:<14}{v['n']:>4}{v['rank1']:>6}/{v['n']:<3}{v['top5']:>6}/{v['n']:<3}"
            f"{v['accepted']:>6}/{v['n']:<4}{v['n'] - v['accepted']:>16}"
        )
    print(
        f"{'ALL on-topic':<14}{on_n:>4}{on_r1:>6}/{on_n:<3}{on_t5:>6}/{on_n:<3}"
        f"{on_acc:>6}/{on_n:<4}{on_n - on_acc:>16}"
    )
    print(
        f"\nOff-topic: rejected {off['rejected']}/{off['n']} "
        f"(false accepts: {off['n'] - off['rejected']})"
    )


if __name__ == "__main__":
    main()
