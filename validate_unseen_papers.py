"""Phase 2A: validate PaperPilot on unseen papers through the real app.py UI code path.

Uses Streamlit's AppTest to execute the actual app.py script and its real chat_input /
chat_message / expander rendering, with session_state pre-seeded exactly as the sidebar
upload handler would build it (same extract_pages/clean_pages/build_chunks/HybridRetriever/
RetrievalPipeline/RAGPipeline calls). AppTest does not support simulating file_uploader,
so the upload click itself isn't automated; everything downstream of it is the real code.

Requires a live LLM API key and 4 full-paper PDF fixtures (gan.pdf, bert.pdf, unet.pdf,
xgboost.pdf) -- see test_fixtures.py for how to provide them locally. This is a live
validation script, not a deterministic/CI test: it calls the real LLM API.

Usage:
    python validate_unseen_papers.py
"""

import json
import sys

from streamlit.testing.v1 import AppTest

from llm import LLMClient
from pdf_processor import build_chunks, clean_pages, detect_sections, detect_title_authors, extract_pages
from rag_pipeline import RAGPipeline
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retriever import HybridRetriever
from test_fixtures import fixtures_dir, require_fixtures

FIXTURE_FILES = {
    "gan": "gan.pdf",
    "bert": "bert.pdf",
    "unet": "unet.pdf",
    "xgboost": "xgboost.pdf",
}

QUESTIONS = {
    "gan": [
        ("Q1-Problem", "What problem does this paper address?"),
        ("Q2-Dataset", "What datasets were used in this paper?"),
        ("Q3-Methodology", "What methodology or model does the paper propose?"),
        ("Q4-Architecture", "How does the proposed approach work?"),
        ("Q5-Results", "What are the key experimental results?"),
        ("Q6-Limitations", "What limitations or weaknesses are mentioned by the authors?"),
        ("Q7-FutureWork", "What future work or extensions does the paper suggest?"),
        ("Q8-Synthesis", "How does the proposed adversarial framework address the problem of "
                          "intractable probabilistic computation identified in the introduction, "
                          "and what experimental evidence supports its effectiveness?"),
        ("Q9-Fact", "What log-likelihood score did the adversarial nets achieve on the MNIST "
                     "dataset in the quantitative evaluation?"),
        ("Q10-OffTopic", "What is the capital of France?"),
    ],
    "bert": [
        ("Q1-Problem", "What problem does this paper address?"),
        ("Q2-Dataset", "What datasets were used in this paper?"),
        ("Q3-Methodology", "What methodology or model does the paper propose?"),
        ("Q4-Architecture", "How does the proposed approach work?"),
        ("Q5-Results", "What are the key experimental results?"),
        ("Q6-Limitations", "What limitations or weaknesses are mentioned by the authors?"),
        ("Q7-FutureWork", "What future work or extensions does the paper suggest?"),
        ("Q8-Synthesis", "How does BERT's bidirectional pre-training approach address the "
                          "limitation of unidirectional language models described in the "
                          "introduction, and what experimental results support its effectiveness?"),
        ("Q9-Fact", "What GLUE benchmark score did BERT achieve?"),
        ("Q10-OffTopic", "What was the company's revenue in 2025?"),
    ],
    "unet": [
        ("Q1-Problem", "What problem does this paper address?"),
        ("Q2-Dataset", "What datasets were used in this paper?"),
        ("Q3-Methodology", "What methodology or model does the paper propose?"),
        ("Q4-Architecture", "How does the proposed approach work?"),
        ("Q5-Results", "What are the key experimental results?"),
        ("Q6-Limitations", "What limitations or weaknesses are mentioned by the authors?"),
        ("Q7-FutureWork", "What future work or extensions does the paper suggest?"),
        ("Q8-Synthesis", "How does the u-net architecture address the challenge of limited "
                          "annotated training data mentioned in the paper, and what experimental "
                          "results support its effectiveness?"),
        ("Q9-Fact", "How long did it take to train the u-net on the given dataset?"),
        ("Q10-OffTopic", "What GPU should I buy for this model?"),
    ],
    "xgboost": [
        ("Q1-Problem", "What problem does this paper address?"),
        ("Q2-Dataset", "What datasets were used in this paper?"),
        ("Q3-Methodology", "What methodology or model does the paper propose?"),
        ("Q4-Architecture", "How does the proposed approach work?"),
        ("Q5-Results", "What are the key experimental results?"),
        ("Q6-Limitations", "What limitations or weaknesses are mentioned by the authors?"),
        ("Q7-FutureWork", "What future work or extensions does the paper suggest?"),
        ("Q8-Synthesis", "How does XGBoost's system design address the scalability problem "
                          "identified in the introduction, and what experimental evidence "
                          "supports its effectiveness?"),
        ("Q9-Fact", "What is the minimum number of machines XGBoost needed to process the "
                     "entire dataset in the scalability experiment?"),
        ("Q10-OffTopic", "What is the boiling point of water at sea level?"),
    ],
}


def build_paper_state(name: str, path: str, reranker, llm_client):
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    sections = detect_sections(pages)
    meta = detect_title_authors(pages)
    chunks = build_chunks(clean_pages(pages), paper_name=name)
    retriever = HybridRetriever(chunks)
    retrieval_pipeline = RetrievalPipeline(retriever, reranker)
    rag = RAGPipeline(retrieval_pipeline, llm_client)
    return {
        "pages": pages, "sections": sections, "meta": meta, "chunks": chunks,
        "rag": rag, "retrieval_pipeline": retrieval_pipeline, "filename": name,
    }


def seed(at: AppTest, state: dict) -> None:
    at.session_state["pages"] = state["pages"]
    at.session_state["sections"] = state["sections"]
    at.session_state["meta"] = state["meta"]
    at.session_state["chunks"] = state["chunks"]
    at.session_state["rag"] = state["rag"]
    at.session_state["filename"] = state["filename"]
    at.session_state["summary"] = None
    at.session_state["chat_history"] = []
    at.run()


def ask(at: AppTest, question: str) -> dict:
    n_before = len(at.chat_input[0].value or "")
    at.chat_input[0].set_value(question).run()
    exc = list(at.exception)
    if exc:
        return {"exception": str(exc[0]), "answer": None, "sources": []}

    cm = at.chat_message[-1]
    md_blocks = [m.value for m in cm.markdown]
    answer = md_blocks[0] if md_blocks else ""
    sources = []
    for exp in cm.expander:
        evidence = exp.markdown[0].value.lstrip("> ").strip() if exp.markdown else ""
        sources.append({"label": exp.label, "evidence": evidence})
    return {"exception": None, "answer": answer, "sources": sources}


def main() -> None:
    fixtures = require_fixtures(list(FIXTURE_FILES.values()))
    if fixtures is None:
        sys.exit(0)
    papers = {name: fixtures[filename] for name, filename in FIXTURE_FILES.items()}

    reranker = CrossEncoderReranker()
    llm_client = LLMClient()

    results = {}
    for name, path in papers.items():
        print(f"=== {name} ===")
        state = build_paper_state(name, path, reranker, llm_client)
        print(f"  pages={len(state['pages'])} chunks={len(state['chunks'])} "
              f"sections={[s['name'] for s in state['sections']]}")

        at = AppTest.from_file("app.py", default_timeout=90)
        seed(at, state)
        if list(at.exception):
            print("  INITIAL RENDER EXCEPTION:", list(at.exception))

        paper_results = []
        for qtype, question in QUESTIONS[name]:
            retrieval = state["retrieval_pipeline"].retrieve(question)
            out = ask(at, question)
            record = {
                "qtype": qtype,
                "question": question,
                "retrieval_accepted": retrieval.accepted,
                "retrieval_reason": retrieval.reason,
                "top10_sections": [c.candidate.chunk["section"] for c in retrieval.reranked[:10]],
                "top10_pages": [c.candidate.chunk["page"] for c in retrieval.reranked[:10]],
                **out,
            }
            paper_results.append(record)
            print(f"  [{qtype}] accepted={retrieval.accepted} exc={out['exception']}")
            print(f"    Q: {question}")
            print(f"    A: {out['answer']}")
            for s in out["sources"]:
                print(f"      {s['label']}")
                print(f"        {s['evidence'][:160]}")
        results[name] = paper_results
        print()

    with open(fixtures_dir() / "unseen_papers_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved full results to unseen_papers_results.json")


if __name__ == "__main__":
    main()
