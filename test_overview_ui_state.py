"""Deterministic tests for the Paper Overview tab's UI state transitions: on-demand
generation, caching across reruns, retry after failure, invalidation when the paper
changes, and that chat/gap-analysis stay usable without an overview.

Seeds st.session_state directly (overview generation no longer happens on upload, so no
file upload needs to be simulated) and stubs overview_pipeline.generate_overview to count
calls -- no LLM or real retrieval index is needed.

Usage:
    python test_overview_ui_state.py
"""

import sys

import overview_pipeline
from overview_pipeline import OverviewItem, OverviewResult
from streamlit.testing.v1 import AppTest

CALL_LOG: list[int] = []
FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


def find_button(at: AppTest, label: str):
    for b in at.button:
        if b.label == label:
            return b
    return None


def stub_generate_overview_success(retrieval_pipeline, llm):
    CALL_LOG.append(1)
    item = OverviewItem(text="stub text", sources=[])
    return OverviewResult(ready=True, error=None, summary=item, problem=item,
                           contributions=[item], methodology=item, datasets=[item],
                           results=[item], limitations=[], future_work=[])


def stub_generate_overview_failure(retrieval_pipeline, llm):
    CALL_LOG.append(1)
    return OverviewResult(ready=False, error="stub overview failure")


class FakeRAG:
    """Stands in for RAGPipeline: only .retrieval_pipeline is ever read in these tests,
    since generate_overview is stubbed and chat_input is never actually submitted."""
    retrieval_pipeline = object()


def seed_single_paper_state(at: AppTest, *, overview, overview_error) -> None:
    at.session_state["pages"] = [{"page": 1, "text": "stub page text"}]
    at.session_state["sections"] = []
    at.session_state["meta"] = {"title": "Stub Paper", "authors": "Stub Author"}
    at.session_state["chunks"] = [{"text": "stub chunk", "page": 1, "section": "Abstract"}]
    at.session_state["rag"] = FakeRAG()
    at.session_state["overview"] = overview
    at.session_state["overview_error"] = overview_error
    at.session_state["gap_analysis"] = None
    at.session_state["gap_analysis_error"] = None
    at.session_state["filename"] = "stub.pdf"
    at.session_state["chat_history"] = []
    at.run()


def test_seeding_pages_alone_does_not_call_overview():
    print("\n=== Test 1: having pages/rag set (as upload would leave it) makes no overview call ===")
    overview_pipeline.generate_overview = stub_generate_overview_success
    CALL_LOG.clear()
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=None, overview_error=None)
    check("no generate_overview call happened just from having pages/rag set", CALL_LOG == [])
    check("the 'Generate Paper Overview' button is shown",
          find_button(at, "Generate Paper Overview") is not None)


def test_button_generates_overview():
    print("\n=== Test 2: clicking 'Generate Paper Overview' generates it ===")
    overview_pipeline.generate_overview = stub_generate_overview_success
    CALL_LOG.clear()
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=None, overview_error=None)
    btn = find_button(at, "Generate Paper Overview")
    btn.click().run()
    check("generate_overview was called exactly once", CALL_LOG == [1])
    check("overview is now stored in session state", at.session_state["overview"] is not None)
    check("overview.ready is True", at.session_state["overview"].ready is True)


def test_ordinary_rerun_does_not_regenerate():
    print("\n=== Test 3: an ordinary rerun after a successful overview makes no new call ===")
    overview_pipeline.generate_overview = stub_generate_overview_success
    item = OverviewItem(text="stub text", sources=[])
    existing = OverviewResult(ready=True, error=None, summary=item, problem=item,
                               contributions=[item], methodology=item, datasets=[item],
                               results=[item], limitations=[], future_work=[])
    CALL_LOG.clear()
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=existing, overview_error=None)
    at.run()
    check("no generate_overview calls happened on ordinary reruns", CALL_LOG == [])
    check("'Regenerate overview' button is shown instead of 'Generate Paper Overview'",
          find_button(at, "Regenerate overview") is not None
          and find_button(at, "Generate Paper Overview") is None)


def test_retry_after_failure():
    print("\n=== Test 4: an explicit retry is possible after a failed overview ===")
    overview_pipeline.generate_overview = stub_generate_overview_failure
    CALL_LOG.clear()
    failed = OverviewResult(ready=False, error="stub overview failure")
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=failed, overview_error="stub overview failure")
    check("'Try again' button is shown for a failed overview", find_button(at, "Try again") is not None)
    btn = find_button(at, "Try again")
    btn.click().run()
    check("overview was reset to None, showing the generate button again",
          find_button(at, "Generate Paper Overview") is not None)

    overview_pipeline.generate_overview = stub_generate_overview_success
    btn2 = find_button(at, "Generate Paper Overview")
    btn2.click().run()
    check("retry after reset successfully generated an overview", CALL_LOG == [1])


def test_changing_paper_invalidates_overview():
    print("\n=== Test 5: changing papers (as the upload handler does) invalidates the cached overview ===")
    item = OverviewItem(text="stub text", sources=[])
    existing = OverviewResult(ready=True, error=None, summary=item, problem=item,
                               contributions=[item], methodology=item, datasets=[item],
                               results=[item], limitations=[], future_work=[])
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=existing, overview_error=None)
    # Simulate what the sidebar upload handler does on a new file.
    at.session_state["overview"] = None
    at.session_state["overview_error"] = None
    at.session_state["filename"] = "other.pdf"
    at.run()
    check("overview is cleared for the new paper", at.session_state["overview"] is None)
    check("the 'Generate Paper Overview' button is shown again",
          find_button(at, "Generate Paper Overview") is not None)


def test_other_features_usable_without_overview():
    print("\n=== Test 6: chat and gap-analysis tabs render without requiring an overview ===")
    at = AppTest.from_file("app.py", default_timeout=30)
    seed_single_paper_state(at, overview=None, overview_error=None)
    check("no exception rendering with pages set but no overview", list(at.exception) == [])
    check("the chat input widget is present", len(at.chat_input) > 0)
    check("the 'Analyze Research Gaps' button is present (independent of overview)",
          find_button(at, "Analyze Research Gaps") is not None)


def main() -> None:
    test_seeding_pages_alone_does_not_call_overview()
    test_button_generates_overview()
    test_ordinary_rerun_does_not_regenerate()
    test_retry_after_failure()
    test_changing_paper_invalidates_overview()
    test_other_features_usable_without_overview()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("\nAll executed checks passed.")


if __name__ == "__main__":
    main()
