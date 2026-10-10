"""Phase 3.1 validation: grounded Paper Overview feature, through the real app.py UI.

Covers: overview correctness/grounding across 4 unseen-to-dev papers (reusing the Phase 2A
set, which already includes a paper with no explicit limitations/future-work section, and
a paper with future-work content outside a literal "Future Work" heading), cross-paper
isolation, and a RAG-chat regression subset to confirm Phase 2A behavior is unchanged.

Requires a live LLM API key and 4 full-paper PDF fixtures (gan.pdf, bert.pdf, unet.pdf,
xgboost.pdf) -- see test_fixtures.py for how to provide them locally. This is a live
validation script, not a deterministic/CI test: it calls the real LLM API.

Usage:
    python validate_overview.py
"""

import sys

from streamlit.testing.v1 import AppTest

from llm import LLMClient
from overview_pipeline import generate_overview
from pdf_processor import build_chunks, clean_pages, detect_sections, detect_title_authors, extract_pages
from rag_pipeline import RAGPipeline
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retriever import HybridRetriever
from test_fixtures import require_fixtures

FIXTURE_FILES = {
    "gan": "gan.pdf",         # has explicit limitations + future work
    "bert": "bert.pdf",       # no explicit limitations/future-work of its own
    "unet": "unet.pdf",       # no explicit limitations section AND no future-work section at all
    "xgboost": "xgboost.pdf", # future-work-like content outside a literal heading
}

REGRESSION_QUESTIONS = {
    "gan": [
        ("problem", "What problem does this paper address?"),
        ("dataset", "What datasets were used in this paper?"),
        ("methodology", "What methodology or model does the paper propose?"),
        ("results", "What are the key experimental results?"),
    ],
    "bert": [
        ("problem", "What problem does this paper address?"),
        ("dataset", "What datasets were used in this paper?"),
        ("methodology", "What methodology or model does the paper propose?"),
        ("results", "What are the key experimental results?"),
        ("synthesis", "How does BERT's bidirectional pre-training approach address the "
                       "limitation of unidirectional language models described in the "
                       "introduction, and what experimental results support its effectiveness?"),
        ("offtopic", "What was the company's revenue in 2025?"),
    ],
}


def build_everything(name, path, reranker, llm_client):
    with open(path, "rb") as f:
        pages = extract_pages(f.read())
    sections = detect_sections(pages)
    meta = detect_title_authors(pages)
    chunks = build_chunks(clean_pages(pages), paper_name=name)
    retrieval_pipeline = RetrievalPipeline(HybridRetriever(chunks), reranker)
    rag = RAGPipeline(retrieval_pipeline, llm_client)
    overview = generate_overview(retrieval_pipeline, llm_client)
    return dict(pages=pages, sections=sections, meta=meta, chunks=chunks,
                rag=rag, overview=overview, overview_error=overview.error, filename=name)


def seed(at: AppTest, state: dict) -> None:
    for k in ["pages", "sections", "meta", "chunks", "rag", "overview", "overview_error", "filename"]:
        at.session_state[k] = state[k]
    at.session_state["chat_history"] = []
    at.run()


def dump_overview_rendering(at: AppTest) -> None:
    print("  Rendered markdown/text/warning in main body:")
    for m in at.markdown:
        print("   MD:", m.value[:150])
    for w in at.warning:
        print("   WARN:", w.value[:200])
    print("  Expanders (evidence) found:", len(at.expander))
    for exp in at.expander:
        print("   -", exp.label)


def main() -> None:
    fixtures = require_fixtures(list(FIXTURE_FILES.values()))
    if fixtures is None:
        sys.exit(0)
    papers = {name: fixtures[filename] for name, filename in FIXTURE_FILES.items()}

    reranker = CrossEncoderReranker()
    llm_client = LLMClient()

    states = {}
    print("=== Generating overviews ===")
    for name, path in papers.items():
        print(f"\n--- {name} ---")
        states[name] = build_everything(name, path, reranker, llm_client)
        ov = states[name]["overview"]
        print(f"ready={ov.ready} error={ov.error}")
        if ov.ready:
            print("SUMMARY:", ov.summary.text)
            print("PROBLEM:", ov.problem.text)
            print("METHODOLOGY:", ov.methodology.text[:200])
            print("CONTRIBUTIONS:", [it.text[:80] for it in ov.contributions])
            print("DATASETS:", [it.text[:80] for it in ov.datasets])
            print("RESULTS:", [it.text[:80] for it in ov.results])
            print("LIMITATIONS:", [(it.text[:80], len(it.sources)) for it in ov.limitations])
            print("FUTURE_WORK:", [(it.text[:80], len(it.sources)) for it in ov.future_work])

    print("\n=== Rendering through the real app.py (AppTest) ===")
    for name in papers:
        print(f"\n--- {name} ---")
        at = AppTest.from_file("app.py", default_timeout=60)
        seed(at, states[name])
        print("  exceptions:", list(at.exception))
        dump_overview_rendering(at)

    print("\n=== Cross-paper isolation test ===")
    at = AppTest.from_file("app.py", default_timeout=60)
    seed(at, states["gan"])
    gan_summary = at.markdown[0].value if at.markdown else None
    for b in at.sidebar.button:
        if b.label == "Clear paper":
            b.click().run()
    print("after clear: pages is None:", at.session_state["pages"] is None,
          "| overview is None:", at.session_state["overview"] is None)
    seed(at, states["xgboost"])
    xgb_markdown = [m.value for m in at.markdown]
    print("gan summary leaked into xgboost render:",
          gan_summary is not None and any(gan_summary == m for m in xgb_markdown))
    print("exceptions after switch:", list(at.exception))

    print("\n=== Regression: subset of existing RAG chat questions ===")
    for name, questions in REGRESSION_QUESTIONS.items():
        at = AppTest.from_file("app.py", default_timeout=60)
        seed(at, states[name])
        for qtype, question in questions:
            at.chat_input[0].set_value(question).run()
            cm = at.chat_message[-1]
            answer = cm.markdown[0].value if cm.markdown else "(blank)"
            print(f"[{name}] [{qtype}] exc={list(at.exception)} A: {answer[:120]}")


if __name__ == "__main__":
    main()
