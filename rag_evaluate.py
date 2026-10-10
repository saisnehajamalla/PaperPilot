"""First RAG evaluation: retrieval -> LLM -> citation, on the existing labeled test set.

Distinguishes retrieval failure (gold never reached the LLM), generation failure (gold
was in context but the answer is wrong or wrongly refused), and citation failure
(answer looks right but the attached evidence doesn't actually support it).

Correctness against gold substrings is reported automatically where possible; answer
text is also printed in full so a human can judge paraphrased correctness, since an
LLM's answer won't literally contain the gold substring.

Usage:
    python rag_evaluate.py attention=path1.pdf resnet=path2.pdf atari=path3.pdf
"""

import sys

from llm import LLMClient
from pdf_processor import build_chunks, clean_pages, extract_pages
from rag_pipeline import REFUSAL_SENTENCE, RAGPipeline
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retrieval_testset import TEST_SETS, off_topic_questions
from retriever import HybridRetriever


def is_relevant(text: str, gold: list[str]) -> bool:
    return any(g in text for g in gold)


def main() -> None:
    specs = [arg.partition("=")[::2] for arg in sys.argv[1:]]
    if not specs:
        print(__doc__)
        sys.exit(1)

    reranker = CrossEncoderReranker()
    llm_client = LLMClient()

    print("=== On-topic questions ===\n")
    for name, path in specs:
        with open(path, "rb") as f:
            chunks = build_chunks(clean_pages(extract_pages(f.read())), paper_name=name)
        retriever = HybridRetriever(chunks)
        pipeline = RetrievalPipeline(retriever, reranker)
        rag = RAGPipeline(pipeline, llm_client)

        for category, question, gold in TEST_SETS.get(name, []):
            if not gold:
                continue
            retrieval = pipeline.retrieve(question)
            gold_in_context = any(is_relevant(c.candidate.chunk["text"], gold) for c in retrieval.reranked)
            result = rag.answer(question)
            citation_supported = any(is_relevant(s.evidence, gold) for s in result.sources)

            if not gold_in_context:
                layer = "RETRIEVAL FAILURE (gold never reached the LLM)"
            elif result.refused:
                layer = f"GENERATION FAILURE (false refusal, reason={result.reason})"
            elif not citation_supported:
                layer = "CITATION FAILURE (answered, but attached evidence doesn't contain gold text)"
            else:
                layer = "gold in context, answered, citation supported -- check answer text by hand"

            print(f"[{name}] [{category}] {question}")
            print(f"  retrieval accepted={retrieval.accepted}  gold_in_context={gold_in_context}  "
                  f"refused={result.refused} ({result.reason})  citation_supported={citation_supported}")
            print(f"  -> {layer}")
            print(f"  Answer: {result.answer}")
            for s in result.sources:
                print(f"    [{s.source_id}] {s.paper_name} p{s.page} {s.section}: {s.evidence[:90]}")
            print()

    print("\n=== Off-topic sample (first 10 of 60, across all three papers) ===\n")
    pipelines = {}
    for name, path in specs:
        with open(path, "rb") as f:
            chunks = build_chunks(clean_pages(extract_pages(f.read())), paper_name=name)
        pipelines[name] = RAGPipeline(RetrievalPipeline(HybridRetriever(chunks), reranker), llm_client)

    off_questions = off_topic_questions()[:10]
    false_accepts = 0
    checked = 0
    for name, rag in pipelines.items():
        for question in off_questions:
            result = rag.answer(question)
            checked += 1
            correct_refusal = result.refused and result.answer == REFUSAL_SENTENCE
            if not correct_refusal:
                false_accepts += 1
                print(f"  FALSE ACCEPT [{name}] {question} -> {result.answer}")
    print(f"\nOff-topic sample: {checked} trials, {checked - false_accepts} correctly refused, "
          f"{false_accepts} false accepts")


if __name__ == "__main__":
    main()
