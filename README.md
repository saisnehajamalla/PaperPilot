# PaperPilot

AI-powered research paper analysis and question answering. Upload a PDF and get a
grounded, evidence-cited overview, research-gap analysis, and Q&A chat for it, or upload
2-5 PDFs to compare them side by side.

Every claim PaperPilot makes is tied back to a specific page/section/quoted excerpt from
the paper's own text. If the retrieved evidence doesn't support an answer, PaperPilot says
so explicitly instead of guessing.

## Features

- **PDF Upload** — extracts text page by page (page numbers preserved) using PyMuPDF.
- **Paper Overview** — auto-generated on upload: title, authors, page count, a short
  summary, and evidence-grounded sections (Problem, Contributions, Methodology, Datasets,
  Results, Limitations, Future Work), plus the paper's detected section headings.
- **Research Gap Analysis** — on demand (button-triggered): author-stated limitations,
  author-stated future work, and two conservative, clearly-labeled inferences —
  "PaperPilot inference — not explicitly stated by the authors" and "Suggested direction —
  novelty not verified" — kept visually and textually distinct from what the authors
  actually wrote.
- **Ask Questions** — chat interface; every answer is grounded in the paper's own text and
  refuses to answer ("I couldn't find this information in the uploaded paper.") rather than
  guess when retrieval or grounding doesn't support a confident answer.
- **Compare Papers** — upload 2-5 PDFs for a side-by-side comparison. Generation runs in two
  independent stages: one LLM call per paper (so one paper's failure can never erase or
  suppress another paper's results), followed by a separate cross-paper synthesis call. The
  app explicitly states when papers aren't meaningfully comparable rather than forcing a
  ranking.
- **Source / Evidence** — every answer and every overview/gap/comparison item includes the
  page number, section, and a quoted excerpt it was based on, shown in an expandable
  evidence panel.

## Tech Stack

Python · Streamlit · PyMuPDF (PDF extraction) · `sentence-transformers` (BGE embeddings,
`BAAI/bge-small-en-v1.5`) · FAISS (vector index) · scikit-learn (TF-IDF keyword scoring,
blended with semantic search for hybrid retrieval) · a cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second retrieval stage · Groq API
(`openai/gpt-oss-120b`, free tier) for generation.

## Setup

1. **Clone/open the project folder** and create a virtual environment:

   ```bash
   python -m venv venv
   ```

   Activate it:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (cmd): `venv\Scripts\activate.bat`
   - macOS/Linux: `source venv/bin/activate`

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key (free):**

   Get a free key at https://console.groq.com/keys (no credit card required).

   Copy `.env.example` to `.env` and fill in your key:

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your-groq-api-key-here
   ```

   Retrieval, reranking, and generation settings (`TOP_K`, `SEMANTIC_WEIGHT`,
   `RERANK_THRESHOLD`, etc. — see `config.py`) are also configurable via `.env`, but the
   defaults are the calibrated, validated values; change them only if you know what you're
   adjusting.

## Run

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Usage

### Single-paper workflow

1. Upload one research paper PDF from the sidebar. PDFs over the configured page limit
   (see "PDF size limit" below) are rejected before any processing happens.
2. The **Paper Overview** tab shows the detected title, authors, and sections immediately;
   click **Generate Paper Overview** to generate the grounded summary/problem/
   contributions/methodology/datasets/results/limitations/future-work breakdown on demand.
   It's cached after that — switching tabs or asking a chat question doesn't regenerate it.
3. Click **Analyze Research Gaps** in the **Research Gap Analysis** tab for author-stated
   limitations/future work plus clearly-labeled, conservative inferred gaps and directions.
   This, like the overview, is independent of whether you've generated one.
4. Switch to **Ask Questions** to chat with the paper directly — no overview needed first.
5. Every section and chat answer has an expandable **Source** showing the page, section,
   and the exact quoted evidence it's grounded in.
6. **Clear paper** in the sidebar resets the single-paper workflow only — it does not affect
   an in-progress comparison.

### Multi-paper workflow

1. Switch to the **Compare Papers** tab and upload 2-5 PDFs there (this uploader is
   independent of the single-paper sidebar upload). Each file is checked against the same
   page limit as the single-paper upload; an oversized paper is excluded and named
   explicitly rather than silently dropped.
2. Click **Compare Papers**. Each paper is analyzed independently first; a failure on one
   paper is shown as an explicit error for that paper only and never blocks or fabricates
   results for the others.
3. A cross-paper synthesis section follows, or an explicit note if the papers aren't
   comparable or synthesis couldn't be generated.
4. Once a comparison exists for the current file set, the button becomes **Regenerate
   Comparison** (or **Try Comparison Again** if generation failed) — re-running reuses the
   already-processed papers rather than re-reading and re-indexing the uploaded files.
   Changing the selected files invalidates the old result; **Clear comparison** resets it
   explicitly.

## PDF size limit

Uploads are capped at `MAX_PAGES` pages (default 300, configurable via `.env`/environment —
see `config.py`). A PDF is validated — including this page-count check — before any text
extraction, chunking, or embedding happens; an oversized, empty, corrupted, unreadable, or
encrypted PDF is rejected with a clear message rather than partially processed. This applies
identically to the single-paper upload and to every file in a multi-paper comparison.

## Citation and grounding behavior

- Page/section/evidence metadata is never taken from the LLM's own output — it's always
  resolved through the application's own source map (`citation.py`), built from the actual
  retrieved chunks.
- Any claim whose cited source can't be resolved to a real chunk is dropped rather than
  shown uncited.
- In multi-paper comparisons, each paper has its own independent retrieval index; a finding
  can only ever cite its own paper's evidence, even if two uploaded files share a filename.

## Error handling

Provider failures (rate limits, quota exhaustion, timeouts, connection errors, invalid
credentials, oversized requests) are translated centrally (`llm.py`) into short, actionable
messages before they ever reach the UI — the raw provider response (which can include an
account/organization identifier) is never shown to the user, only kept internally for
diagnostics. This applies the same way across chat, Paper Overview, Research Gap Analysis,
and Multi-Paper Comparison, since all four read from the same translated message.

## Repository structure

```
PaperPilot/
├── app.py                        Streamlit UI (all 4 tabs)
├── config.py                     Environment-driven settings (retrieval, LLM, MAX_PAGES)
├── llm.py, citation.py,          Shared infrastructure used by every pipeline
│   context_builder.py,
│   pdf_processor.py
├── embeddings.py, vector_store.py,
│   retriever.py, retrieval_pipeline.py,
│   reranker.py, section_prior.py  Retrieval stack (do not modify without re-validating)
├── rag_pipeline.py, overview_pipeline.py,
│   research_gap_pipeline.py,
│   comparison_pipeline.py         The 4 generation pipelines
│
├── test_*.py                      Deterministic unit/state tests -- no LLM API key, no
│                                   full-paper PDFs required. See "Running the deterministic
│                                   tests" below.
├── test_fixtures.py               Shared fixture-path resolver used by the tests/scripts
│                                   below that accept full-paper PDFs
├── run_tests.py                   Single entry point that runs all deterministic tests
│
├── validate_overview.py,          Optional, non-deterministic: exercise real app.py code
│   validate_unseen_papers.py      paths against real papers through a live LLM call. Need
│                                   both a GROQ_API_KEY and local fixture PDFs (see below).
│
├── calibrate_retrieval.py, compare_section_aware.py, compare_topk.py,
│   diagnose_retrieval.py, evaluate_retrieval.py, rag_evaluate.py,
│   validate_pipeline.py, validate_retrieval.py, retrieval_testset.py
│                                   Retrieval evaluation/calibration scripts from earlier
│                                   tuning phases. Exploratory, not pass/fail tests; take PDF
│                                   paths as CLI arguments (not fixture-directory-based).
│                                   rag_evaluate.py additionally needs a live LLM call.
│                                   Kept for reproducing how the current retrieval config
│                                   (weights, threshold, top-k) was chosen -- not obsolete.
│
└── fixtures/                      Optional local-only full-paper PDFs (see below); README.md
                                    documents what's expected here. Never committed.
```

These files are kept flat at the repository root rather than split into `tests/`/`scripts/`
subdirectories: every file above imports application modules directly (e.g.
`from pdf_processor import ...`), which only resolves correctly today because the repository
root is implicitly on `sys.path` when a script is run directly from it. Moving them would
require either a `sys.path` shim in every file or a package/`conftest.py` setup, for a purely
cosmetic reorganization with no functional benefit — so the categorization above is kept as
documentation instead of a directory move.

## Running the deterministic tests

```bash
python run_tests.py
```

This runs all 5 deterministic test modules and prints one pass/fail line per module, then an
overall summary; it exits non-zero if any module fails. Each module can also be run on its
own, e.g. `python test_pdf_validation.py`. None of them call a real LLM or need an API key:

- `test_pdf_validation.py` — PDF page-limit enforcement and corrupted/empty/encrypted-PDF
  handling, using small synthetic PDFs generated at test time.
- `test_llm_error_translation.py` — provider-error-to-user-message translation, using
  reconstructed provider exceptions (no live request).
- `test_compare_ui_state.py` / `test_overview_ui_state.py` — Streamlit UI state transitions
  (caching, regenerate/retry, invalidation, partial-failure display), using Streamlit's
  `AppTest` with a stubbed generation call.
- `test_comparison_reliability.py` — the comparison pipeline's failure-isolation guarantees,
  using a stubbed LLM client. Checks that don't need full-paper PDFs always run; checks that
  do are **skipped** (not silently passed) with an explicit message if the fixtures below
  aren't present.

## Optional full-paper evaluation fixtures

Some scripts (`test_comparison_reliability.py`'s full-paper checks, `validate_overview.py`,
`validate_unseen_papers.py`) use real papers as retrieval evidence instead of synthetic
text. These PDFs are not committed to the repository (redistribution permissions haven't
been verified). To supply them locally, place `gan.pdf`, `bert.pdf`, `unet.pdf`, and
`xgboost.pdf` in a `fixtures/` folder at the repository root, or point the
`PAPERPILOT_FIXTURES_DIR` environment variable at a directory containing them — see
`fixtures/README.md`. Nothing in the repository downloads these automatically.

`validate_overview.py` and `validate_unseen_papers.py` also call the real Groq API and are
not deterministic — run them only when you want to verify live behavior and have API quota
available.

## Current limitations

- A successful live multi-paper comparison with 3 papers, and a separate one with 5 papers
  (including duplicate-filename handling), have not yet been observed completing with
  actual generated content — prior attempts were blocked by Groq's free-tier daily quota
  before any paper's Stage A call could complete. The failure-handling path itself (every
  paper correctly reporting its own `generation_error`, duplicate filenames staying fully
  isolated even under total failure) has been verified at both scales. 2-paper comparisons
  have completed successfully multiple times.
- There is no automated CI pipeline; `run_tests.py` aggregates the deterministic suite for
  local use, but nothing runs it automatically on a push or pull request.
- Provider-error translation falls back to a generic rate-limit or generic-error message
  when the provider's response doesn't clearly indicate the specific cause (e.g. a 429 whose
  message text doesn't mention a per-day or per-minute limit) — see `llm.py`.
