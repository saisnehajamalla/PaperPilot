"""Print retrieval candidates and scores for a PDF, to tune weights and thresholds.

Usage:
    python evaluate_retrieval.py path/to/paper.pdf [question ...]
"""

import os
import sys

from pdf_processor import build_chunks, clean_pages, extract_pages
from retriever import HybridRetriever

DEFAULT_QUESTIONS = [
    "What dataset was used?",
    "What methodology was used?",
    "What were the results?",
    "What problem does the paper solve?",
    "What is the capital of France?",
]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    questions = sys.argv[2:] or DEFAULT_QUESTIONS
    paper_name = os.path.basename(pdf_path)

    with open(pdf_path, "rb") as f:
        pages = extract_pages(f.read())
    chunks = build_chunks(clean_pages(pages), paper_name=paper_name)
    print(f"{paper_name}: {len(pages)} pages, {len(chunks)} chunks")

    retriever = HybridRetriever(chunks)
    print(
        f"weights semantic={retriever.semantic_weight} keyword={retriever.keyword_weight} "
        f"min_final={retriever.min_final_score} min_semantic={retriever.min_semantic_score}\n"
    )

    for question in questions:
        result = retriever.retrieve(question)
        verdict = "ACCEPT" if result.accepted else "REFUSE"
        print(f"Q: {question}  -> {verdict} ({result.reason})")
        for rank, c in enumerate(result.candidates, 1):
            text = c.chunk["text"][:90]
            print(
                f"  {rank}. sem={c.semantic:.3f} kw={c.keyword:.3f} final={c.final:.3f}"
                f"  p{c.chunk['page']} {c.chunk['section']:<14} | {text}"
            )
        print()


if __name__ == "__main__":
    main()
