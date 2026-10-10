"""Deterministic tests for the Compare Papers tab's UI state transitions: result caching
across ordinary reruns, the explicit regenerate/retry action, invalidation when the file
set changes, and partial-failure rendering.

AppTest cannot simulate st.file_uploader, so these seed st.session_state directly (the same
approach used by validate_overview.py / validate_unseen_papers.py) and stub
comparison_pipeline.generate_comparison to count calls -- no LLM, PDF, or real retrieval
index is needed. Because the (unsimulated) uploader always reports an empty file selection,
compare_input_signature is seeded to () to match it; this is called out explicitly in each
test rather than left implicit.

Usage:
    python test_compare_ui_state.py
"""

import sys

import comparison_pipeline
from comparison_pipeline import ComparisonResult, PaperComparisonData, ProcessedPaper
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


def stub_generate_comparison(papers, llm):
    CALL_LOG.append(len(papers))
    return ComparisonResult(
        ready=True,
        error=None,
        papers={p.paper_id: PaperComparisonData(paper_id=p.paper_id, display_name=p.display_name)
                for p in papers},
        comparable=True,
        comparability_note="stub comparability note",
        synthesis=[],
        synthesis_error=None,
    )


def stub_generate_comparison_failure(papers, llm):
    CALL_LOG.append(len(papers))
    return ComparisonResult(
        ready=False,
        error="stub total failure",
        papers={p.paper_id: PaperComparisonData(paper_id=p.paper_id, display_name=p.display_name,
                                                 generation_error="stub per-paper failure")
                for p in papers},
        comparable=None,
        comparability_note="",
        synthesis=[],
        synthesis_error=None,
    )


def fake_papers(n: int) -> list[ProcessedPaper]:
    return [
        ProcessedPaper(paper_id=f"paper_{i}", display_name=f"paper{i}.pdf", pages=[], sections=[],
                        meta={}, chunks=[], retrieval_pipeline=None)
        for i in range(1, n + 1)
    ]


def success_result(papers) -> ComparisonResult:
    return ComparisonResult(
        ready=True,
        papers={p.paper_id: PaperComparisonData(paper_id=p.paper_id, display_name=p.display_name)
                for p in papers},
        comparable=True, comparability_note="note", synthesis=[], synthesis_error=None,
    )


def seed_compare_state(at: AppTest, *, papers, result) -> None:
    at.session_state["compare_input_signature"] = ()
    at.session_state["compare_papers"] = papers
    at.session_state["compare_result"] = result
    at.session_state["compare_error"] = None if (result is None or result.ready) else result.error
    at.session_state["compare_failed"] = []
    at.run()


def test_ordinary_rerun_no_new_calls_and_result_reused():
    print("\n=== Test 1/2: an ordinary rerun makes no new LLM calls; the result is reused ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison
    CALL_LOG.clear()
    papers = fake_papers(2)
    result = success_result(papers)

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=result)
    at.run()  # a second, ordinary rerun
    check("no generate_comparison calls happened on ordinary reruns", CALL_LOG == [])
    check("the cached result object is still the one we seeded (reused, not regenerated)",
          at.session_state["compare_result"] is result)


def test_initial_button_hidden_once_result_exists():
    print("\n=== Test: once a result exists, the initial 'Compare Papers' action is replaced ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison
    papers = fake_papers(2)
    result = success_result(papers)

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=result)
    check("'Compare Papers' button is not shown", find_button(at, "Compare Papers") is None)
    check("'Regenerate Comparison' button is shown instead",
          find_button(at, "Regenerate Comparison") is not None)


def test_explicit_regenerate_invokes_generation_again():
    print("\n=== Test 3: clicking 'Regenerate Comparison' does invoke generation again ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison
    CALL_LOG.clear()
    papers = fake_papers(2)
    result = success_result(papers)

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=result)
    btn = find_button(at, "Regenerate Comparison")
    btn.click().run()
    check("generate_comparison was called exactly once", CALL_LOG == [2])
    check("compare_papers (the already-processed list) is unchanged, not re-read from files",
          at.session_state["compare_papers"] is papers)


def test_retry_after_total_failure_and_partial_failure_rendering():
    print("\n=== Test 6: partial/total failure is represented correctly, and is retryable ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison_failure
    CALL_LOG.clear()
    papers = fake_papers(2)
    failed_result = ComparisonResult(
        ready=False, error="stub total failure",
        papers={p.paper_id: PaperComparisonData(paper_id=p.paper_id, display_name=p.display_name,
                                                 generation_error="stub per-paper failure")
                for p in papers},
        comparable=None, comparability_note="", synthesis=[], synthesis_error=None,
    )

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=failed_result)
    check("'Try Comparison Again' button is shown (not 'Regenerate Comparison')",
          find_button(at, "Try Comparison Again") is not None
          and find_button(at, "Regenerate Comparison") is None)
    check("the total-failure error is surfaced via st.warning",
          any("stub total failure" in w.value for w in at.warning))

    comparison_pipeline.generate_comparison = stub_generate_comparison
    btn = find_button(at, "Try Comparison Again")
    btn.click().run()
    check("retry invoked generation again", CALL_LOG == [2])
    check("a successful retry now shows 'Regenerate Comparison'",
          find_button(at, "Regenerate Comparison") is not None)


def test_changing_input_set_invalidates_old_result():
    print("\n=== Test 4: changing the file-set signature invalidates the cached result ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison
    papers = fake_papers(2)
    result = success_result(papers)

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=result)
    # Simulate a changed file selection: seed a signature that no longer matches what the
    # (unsimulated) uploader computes -- the same mismatch a real file-set change produces.
    at.session_state["compare_input_signature"] = (("different.pdf", 123),)
    at.run()
    check("compare_result was invalidated", at.session_state["compare_result"] is None)
    check("compare_papers was invalidated", at.session_state["compare_papers"] is None)
    check("compare_input_signature was reset to the (empty, unsimulated) current selection",
          at.session_state["compare_input_signature"] == ())


def test_clear_comparison_resets_state():
    print("\n=== Test 5: 'Clear comparison' resets compare state back to the initial action ===")
    comparison_pipeline.generate_comparison = stub_generate_comparison
    papers = fake_papers(2)
    result = success_result(papers)

    at = AppTest.from_file("app.py", default_timeout=30)
    seed_compare_state(at, papers=papers, result=result)
    btn = find_button(at, "Clear comparison")
    check("'Clear comparison' button is present", btn is not None)
    btn.click().run()
    check("compare_result cleared", at.session_state["compare_result"] is None)
    check("compare_papers cleared", at.session_state["compare_papers"] is None)
    check("compare_failed cleared", at.session_state["compare_failed"] == [])


def main() -> None:
    test_ordinary_rerun_no_new_calls_and_result_reused()
    test_initial_button_hidden_once_result_exists()
    test_explicit_regenerate_invokes_generation_again()
    test_changing_input_set_invalidates_old_result()
    test_clear_comparison_resets_state()
    test_retry_after_total_failure_and_partial_failure_rendering()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("\nAll executed checks passed.")


if __name__ == "__main__":
    main()
