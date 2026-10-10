"""PaperPilot - AI-Powered Research Paper Analysis and Question Answering System."""

import streamlit as st
from dotenv import load_dotenv

from comparison_pipeline import MAX_PAPERS, MIN_PAPERS, ProcessedPaper, generate_comparison
from config import MAX_PAGES
from llm import LLMClient
from overview_pipeline import generate_overview
from pdf_processor import (
    PDFValidationError,
    build_chunks,
    clean_pages,
    detect_sections,
    detect_title_authors,
    extract_pages_from_doc,
    validate_pdf,
)
from rag_pipeline import RAGPipeline
from reranker import CrossEncoderReranker
from research_gap_pipeline import generate_research_gap_analysis
from retrieval_pipeline import RetrievalPipeline
from retriever import HybridRetriever

load_dotenv()

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Cached heavy resources (shared across reruns and across papers)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@st.cache_resource(show_spinner=False)
def get_llm_client() -> LLMClient:
    return LLMClient()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "pages": None,
    "sections": None,
    "chunks": None,
    "meta": None,
    "overview": None,
    "overview_error": None,
    "gap_analysis": None,
    "gap_analysis_error": None,
    "chat_history": [],
    "filename": None,
    "rag": None,
    # Multi-paper comparison state is intentionally separate from the single-paper state
    # above: clearing one must never affect the other.
    "compare_input_signature": (),
    "compare_papers": None,
    "compare_result": None,
    "compare_error": None,
    "compare_failed": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 PaperPilot")
st.caption("AI-Powered Research Paper Analysis and Question Answering System")

# ---------------------------------------------------------------------------
# Sidebar: single-paper upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
        try:
            with st.spinner("Validating and extracting text..."):
                # getvalue() (not read()) so this is unaffected by any cursor position
                # left over from a previous read of the same UploadedFile object.
                file_bytes = uploaded_file.getvalue()
                # Validates (including the MAX_PAGES limit) before any text extraction,
                # chunking, or embedding happens.
                doc = validate_pdf(file_bytes, MAX_PAGES, filename=uploaded_file.name)
                pages = extract_pages_from_doc(doc)
                if not pages:
                    raise ValueError("No pages could be extracted from this PDF.")
                if not any(p["text"].strip() for p in pages):
                    raise ValueError(
                        "No extractable text was found. This PDF may be a scanned image "
                        "without OCR text."
                    )

                sections = detect_sections(pages)
                meta = detect_title_authors(pages)
                chunks = build_chunks(clean_pages(pages), paper_name=uploaded_file.name)
                if not chunks:
                    raise ValueError("No usable text chunks could be built from this PDF.")

            with st.spinner("Building search index (embeddings + reranker)..."):
                retriever = HybridRetriever(chunks)
                retrieval_pipeline = RetrievalPipeline(retriever, get_reranker())
                rag = RAGPipeline(retrieval_pipeline, get_llm_client())

            st.session_state.pages = pages
            st.session_state.sections = sections
            st.session_state.meta = meta
            st.session_state.chunks = chunks
            st.session_state.rag = rag
            # Overview is generated on demand (see the Paper Overview tab), not here.
            st.session_state.overview = None
            st.session_state.overview_error = None
            st.session_state.gap_analysis = None
            st.session_state.gap_analysis_error = None
            st.session_state.filename = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"Loaded {uploaded_file.name} ({len(pages)} pages)")
        except PDFValidationError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not process this PDF: {e}")

    if st.session_state.pages:
        st.divider()
        st.metric("Pages", len(st.session_state.pages))
        st.metric("Sections detected", len(st.session_state.sections))
        if st.button("Clear paper", use_container_width=True):
            for key in ["pages", "sections", "chunks", "meta", "overview", "overview_error",
                        "gap_analysis", "gap_analysis_error", "chat_history", "filename", "rag"]:
                st.session_state[key] = [] if key == "chat_history" else None
            st.rerun()

# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------
def render_evidence(sources: list) -> None:
    for src in sources:
        with st.expander(f"📄 {src.paper_name} — 📍 Page {src.page} — 📑 {src.section}"):
            st.markdown(f"> {src.evidence}")


def render_overview_section(heading: str, items) -> None:
    st.markdown(f"### {heading}")
    is_list = isinstance(items, list)
    for item in items if is_list else [items]:
        if is_list:
            st.markdown(f"- {item.text}")
        else:
            st.write(item.text)
        render_evidence(item.sources)


def render_answer(answer: str, sources: list) -> None:
    st.write(answer)
    if not sources:
        return
    st.markdown("**Sources**")
    for src in sources:
        with st.expander(f"📄 {src['paper_name']} — 📍 Page {src['page']} — 📑 {src['section']}"):
            st.markdown(f"> {src['evidence']}")


def run_comparison(papers: list[ProcessedPaper]) -> None:
    """Calls generate_comparison and stores the outcome. Never re-reads uploaded files --
    callers always pass already-processed ProcessedPaper objects, so this is safe to call
    again for an explicit regenerate/retry without touching any UploadedFile buffer."""
    with st.spinner(f"Comparing {len(papers)} papers across all dimensions..."):
        try:
            result = generate_comparison(papers, get_llm_client())
            st.session_state.compare_result = result
            st.session_state.compare_error = None if result.ready else result.error
        except Exception as e:
            st.session_state.compare_result = None
            st.session_state.compare_error = str(e)


# ---------------------------------------------------------------------------
# Tabs
#
# All four tab bodies run on every Streamlit rerun regardless of which tab is
# visually active, so a single-paper gate must be an if/else inside each of the
# first three tabs rather than a global st.stop() -- st.stop() would also
# prevent the independent Compare Papers tab from ever rendering.
# ---------------------------------------------------------------------------
tab_overview, tab_gaps, tab_chat, tab_compare = st.tabs(
    ["📋 Paper Overview", "🔬 Research Gap Analysis", "💬 Ask Questions", "📊 Compare Papers"]
)

# ---- Overview tab ---------------------------------------------------------
with tab_overview:
    if not st.session_state.pages:
        st.info("Upload a PDF from the sidebar to get started.")
    else:
        meta = st.session_state.meta
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(meta["title"])
            st.write(f"**Authors:** {meta['authors']}")
        with col2:
            st.metric("Page count", len(st.session_state.pages))

        st.divider()
        overview = st.session_state.overview
        if overview is None:
            st.info(
                "Generate a grounded summary, problem statement, contributions, "
                "methodology, datasets, results, limitations, and future work for this "
                "paper. You can ask questions in the Ask Questions tab without doing this."
            )
            if st.button("Generate Paper Overview"):
                with st.spinner("Analyzing paper (generating grounded overview)..."):
                    try:
                        overview = generate_overview(st.session_state.rag.retrieval_pipeline, get_llm_client())
                        st.session_state.overview = overview
                        st.session_state.overview_error = None if overview.ready else overview.error
                        st.rerun()
                    except Exception as e:
                        st.session_state.overview_error = str(e)
                        st.error(f"Could not generate the paper overview: {e}")
        elif not overview.ready:
            st.warning(
                "A grounded overview could not be generated for this paper"
                + (f": {overview.error}" if overview.error else ".")
                + " You can still ask questions in the Ask Questions tab."
            )
            if st.button("Try again"):
                st.session_state.overview = None
                st.session_state.overview_error = None
                st.rerun()
        else:
            st.markdown("### Summary")
            st.write(overview.summary.text)
            render_evidence(overview.summary.sources)

            st.divider()
            render_overview_section("Problem", overview.problem)
            st.divider()
            render_overview_section("Key Contributions", overview.contributions)
            st.divider()
            render_overview_section("Methodology", overview.methodology)
            st.divider()
            render_overview_section("Datasets", overview.datasets)
            st.divider()
            render_overview_section("Key Results", overview.results)
            st.divider()
            render_overview_section("Limitations", overview.limitations)
            st.divider()
            render_overview_section("Future Work", overview.future_work)

            if st.button("Regenerate overview"):
                st.session_state.overview = None
                st.rerun()

        st.divider()
        st.subheader("Detected Sections")
        if st.session_state.sections:
            for s in st.session_state.sections:
                st.markdown(f"- **{s['name']}** — starts on page {s['page']}")
        else:
            st.write("No standard sections were detected in this document.")

# ---- Research Gap Analysis tab --------------------------------------------
with tab_gaps:
    if not st.session_state.pages:
        st.info("Upload a PDF from the sidebar to get started.")
    else:
        st.subheader("Research Gap Analysis")
        st.caption(
            "A conservative, evidence-grounded analysis in four separate categories: what the "
            "authors themselves state as a limitation, what the authors themselves propose as "
            "future work, gaps PaperPilot infers from the evidence (clearly labeled as "
            "inference, not an author claim), and research directions PaperPilot suggests from "
            "those gaps. This analyzes only the uploaded paper — it cannot verify novelty "
            "against the broader literature."
        )

        gaps = st.session_state.gap_analysis
        if gaps is None:
            if st.button("Analyze Research Gaps"):
                with st.spinner("Analyzing problem, contributions, methodology, evaluation "
                                 "scope, limitations, future work, and conclusions..."):
                    try:
                        gaps = generate_research_gap_analysis(
                            st.session_state.rag.retrieval_pipeline, get_llm_client()
                        )
                        st.session_state.gap_analysis = gaps
                        st.session_state.gap_analysis_error = None if gaps.ready else gaps.error
                        st.rerun()
                    except Exception as e:
                        st.session_state.gap_analysis_error = str(e)
                        st.error(f"Could not generate the research gap analysis: {e}")
        elif not gaps.ready:
            st.warning(f"The research gap analysis could not be generated: {gaps.error}")
            if st.button("Try again"):
                st.session_state.gap_analysis = None
                st.rerun()
        else:
            st.markdown("### Author-Stated Limitations")
            if gaps.limitations:
                for item in gaps.limitations:
                    st.markdown(f"- {item.claim}")
                    render_evidence(item.sources)
            else:
                st.caption("No explicit limitations were identified in the paper.")

            st.divider()
            st.markdown("### Author-Stated Future Work")
            if gaps.future_work:
                for item in gaps.future_work:
                    st.markdown(f"- {item.claim}")
                    render_evidence(item.sources)
            else:
                st.caption("No explicit future work was identified in the paper.")

            st.divider()
            st.markdown("### Potential Research Gaps")
            st.caption("PaperPilot inference — not explicitly stated by the authors")
            if gaps.potential_gaps:
                for item in gaps.potential_gaps:
                    st.markdown(f"**{item.title}**")
                    st.markdown(f"- Observation: {item.observation}")
                    st.markdown(f"- Reasoning: {item.reasoning}")
                    st.markdown(f"- What this paper does not establish: {item.not_established}")
                    render_evidence(item.sources)
            else:
                st.caption("The available evidence in this paper was not sufficient to "
                           "support a well-reasoned potential gap.")

            st.divider()
            st.markdown("### Potential Research Directions")
            st.caption("Suggested direction — novelty not verified")
            if gaps.research_directions:
                for item in gaps.research_directions:
                    st.markdown(f"**{item.title}**")
                    st.markdown(f"- Research question: {item.research_question}")
                    st.markdown(f"- Motivation: {item.motivation}")
                    render_evidence(item.sources)
            else:
                st.caption("No research direction was well-supported enough to suggest.")

            if st.button("Regenerate analysis"):
                st.session_state.gap_analysis = None
                st.rerun()

# ---- Chat tab ---------------------------------------------------------
with tab_chat:
    if not st.session_state.pages:
        st.info("Upload a PDF from the sidebar to get started.")
    else:
        st.subheader("Ask a question about the paper")
        st.caption(
            'Try: "What is the main problem?", "What methodology was used?", '
            '"What dataset was used?", "What are the results?"'
        )

        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                render_answer(entry["answer"], entry["sources"])

        question = st.chat_input("Ask a question about this paper...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag.answer(question)
                    sources = [
                        {
                            "paper_name": s.paper_name,
                            "page": s.page,
                            "section": s.section,
                            "evidence": s.evidence,
                        }
                        for s in result.sources
                    ]
                    render_answer(result.answer, sources)
                    st.session_state.chat_history.append(
                        {"question": question, "answer": result.answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")

# ---- Compare Papers tab ----------------------------------------------------
# Deliberately independent of the single-paper state above: usable with or
# without an active single-paper session, and clearing one never touches the
# other.
with tab_compare:
    st.subheader("Multi-Paper Comparison")
    st.caption(
        "Upload 2-5 research papers to compare their problem statements, contributions, "
        "methodology, datasets, results, limitations and future work. This analyzes only "
        "the uploaded papers and cannot verify novelty against the broader literature."
    )

    compare_files = st.file_uploader(
        "Upload 2-5 PDFs to compare",
        type=["pdf"],
        accept_multiple_files=True,
        key="compare_uploader",
    )

    current_signature = tuple(sorted((f.name, f.size) for f in compare_files)) if compare_files else ()
    if current_signature != st.session_state.compare_input_signature:
        # The input set changed: any previous comparison is now stale.
        st.session_state.compare_result = None
        st.session_state.compare_error = None
        st.session_state.compare_failed = []
        st.session_state.compare_papers = None
        st.session_state.compare_input_signature = current_signature

    if compare_files:
        st.write(f"Selected {len(compare_files)} paper(s):")
        for f in compare_files:
            st.markdown(f"- {f.name} ({f.size / 1024:.0f} KB)")

    n = len(compare_files) if compare_files else 0
    can_compare = MIN_PAPERS <= n <= MAX_PAPERS
    if compare_files and not can_compare:
        st.warning(f"Please upload between {MIN_PAPERS} and {MAX_PAPERS} PDFs (currently {n}).")

    if st.session_state.compare_papers is None:
        # Nothing has been processed yet for this exact file set -- this is the only
        # entry point that reads the uploaded files.
        if can_compare and st.button("Compare Papers"):
            processed: list[ProcessedPaper] = []
            failed: list[str] = []
            seen_names: dict[str, int] = {}
            with st.spinner("Processing papers and building independent search indexes..."):
                for i, f in enumerate(compare_files, start=1):
                    base_name = f.name
                    seen_names[base_name] = seen_names.get(base_name, 0) + 1
                    display_name = (
                        base_name if seen_names[base_name] == 1
                        else f"{base_name} ({seen_names[base_name]})"
                    )
                    try:
                        # getvalue() (not read()) so this can't return an empty/partial
                        # buffer regardless of any prior read of the same UploadedFile.
                        file_bytes = f.getvalue()
                        # Validates (including the MAX_PAGES limit) before any text
                        # extraction, chunking, or embedding happens for this paper.
                        doc = validate_pdf(file_bytes, MAX_PAGES, filename=display_name)
                        pages = extract_pages_from_doc(doc)
                        if not pages or not any(p["text"].strip() for p in pages):
                            raise ValueError(
                                "No extractable text (this may be a scanned PDF without OCR)."
                            )
                        sections = detect_sections(pages)
                        meta = detect_title_authors(pages)
                        chunks = build_chunks(clean_pages(pages), paper_name=display_name)
                        if not chunks:
                            raise ValueError("No usable text chunks could be built.")

                        retriever = HybridRetriever(chunks)
                        paper_retrieval_pipeline = RetrievalPipeline(retriever, get_reranker())
                        processed.append(ProcessedPaper(
                            paper_id=f"paper_{i}",
                            display_name=display_name,
                            pages=pages,
                            sections=sections,
                            meta=meta,
                            chunks=chunks,
                            retrieval_pipeline=paper_retrieval_pipeline,
                        ))
                    except Exception as e:
                        failed.append(f"{display_name}: {e}")

            st.session_state.compare_failed = failed

            if len(processed) < MIN_PAPERS:
                st.session_state.compare_papers = None
                st.session_state.compare_result = None
                st.session_state.compare_error = (
                    f"At least {MIN_PAPERS} papers must process successfully to run a comparison."
                )
            else:
                st.session_state.compare_papers = processed
                run_comparison(processed)
            st.rerun()
    else:
        # Already processed (and generated at least once) for this exact file set.
        # Re-running reuses the existing per-paper indexes already stored in
        # compare_papers and never re-reads the uploaded files, so an explicit
        # regenerate/retry cannot be affected by an already-consumed UploadedFile buffer.
        existing_result = st.session_state.compare_result
        action_label = (
            "Regenerate Comparison" if (existing_result is not None and existing_result.ready)
            else "Try Comparison Again"
        )
        if st.button(action_label):
            run_comparison(st.session_state.compare_papers)
            st.rerun()

    if st.session_state.compare_failed:
        st.warning(
            "These papers could not be processed and were excluded from the comparison:\n"
            + "\n".join(f"- {x}" for x in st.session_state.compare_failed)
        )

    result_not_ready = (
        st.session_state.compare_result is None or not st.session_state.compare_result.ready
    )
    if st.session_state.compare_error and result_not_ready:
        st.warning(f"Comparison could not be generated: {st.session_state.compare_error}")

    result = st.session_state.compare_result
    if result is not None and result.ready:
        st.divider()
        if result.comparable:
            st.success("The authors' reported results appear sufficiently comparable: "
                        + result.comparability_note)
        else:
            st.warning("⚠️ Results are NOT directly comparable across these papers: "
                       + result.comparability_note)

        for paper in st.session_state.compare_papers:
            pdata = result.papers.get(paper.paper_id)
            if pdata is None:
                continue
            st.divider()
            st.markdown(f"## {pdata.display_name}")

            if pdata.generation_error:
                st.error(f"Findings could not be generated for this paper: {pdata.generation_error}")
                continue

            if pdata.problem:
                st.markdown("**Problem**")
                st.write(pdata.problem.text)
                render_evidence(pdata.problem.sources)
            else:
                st.caption("Problem: not established from the uploaded paper.")

            st.markdown("**Contributions**")
            if pdata.contributions:
                for item in pdata.contributions:
                    st.markdown(f"- {item.text}")
                    render_evidence(item.sources)
            else:
                st.caption("Not reported in the paper.")

            if pdata.methodology:
                st.markdown("**Methodology**")
                st.write(pdata.methodology.text)
                render_evidence(pdata.methodology.sources)
            else:
                st.caption("Methodology: not established from the uploaded paper.")

            st.markdown("**Datasets & Experimental Setup**")
            if pdata.datasets_setup:
                for item in pdata.datasets_setup:
                    st.markdown(f"- {item.text}")
                    render_evidence(item.sources)
            else:
                st.caption("Not reported in the paper.")

            st.markdown("**Results**")
            if pdata.results:
                for item in pdata.results:
                    st.markdown(f"- {item.text}")
                    render_evidence(item.sources)
            else:
                st.caption("Not reported in the paper.")

            st.markdown("**Author-Stated Limitations**")
            if pdata.limitations:
                for item in pdata.limitations:
                    st.markdown(f"- {item.text}")
                    render_evidence(item.sources)
            else:
                st.caption("No explicit limitations were identified in the paper.")

            st.markdown("**Author-Stated Future Work**")
            if pdata.future_work:
                for item in pdata.future_work:
                    st.markdown(f"- {item.text}")
                    render_evidence(item.sources)
            else:
                st.caption("No explicit future work was identified in the paper.")

        st.divider()
        st.markdown("## Cross-Paper Synthesis")
        st.caption("PaperPilot synthesis across the uploaded papers — not a claim about the "
                   "broader research field.")
        if result.synthesis_error:
            st.warning(f"Cross-paper synthesis could not be generated: {result.synthesis_error}. "
                       "The per-paper comparisons above are still valid.")
        elif result.synthesis:
            for item in result.synthesis:
                st.markdown(f"**[{item.synthesis_type}]** {item.claim}")
                render_evidence(item.sources)
        else:
            st.caption("No well-supported cross-paper observation was found.")

        if st.button("Clear comparison"):
            st.session_state.compare_result = None
            st.session_state.compare_papers = None
            st.session_state.compare_error = None
            st.session_state.compare_failed = []
            st.rerun()
