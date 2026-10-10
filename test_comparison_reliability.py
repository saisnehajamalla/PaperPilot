"""Deterministic tests for the two-stage comparison pipeline's failure isolation.

These do not call a real LLM -- they stub LLMClient.complete_json to force each failure
mode directly, so isolation and failure handling are proven without depending on the
model ever actually making a mistake.

Requires 4 full-paper PDF fixtures (gan.pdf, bert.pdf, unet.pdf, xgboost.pdf) to exercise
real retrieval against real evidence -- see test_fixtures.py for how to provide them. If
they are not present, the fixture-dependent checks are skipped with an actionable message;
a missing-fixture skip is not a failure.

Exit code: 0 only if every executed check passed (or all were skipped). Non-zero if any
check failed.

Usage:
    python test_comparison_reliability.py
"""

import json
import sys

from citation import Source
from comparison_pipeline import (
    ComparisonFinding,
    PaperComparisonData,
    ProcessedPaper,
    generate_comparison,
)
from llm import LLMResponse
from pdf_processor import build_chunks, clean_pages, extract_pages
from reranker import CrossEncoderReranker
from retrieval_pipeline import RetrievalPipeline
from retriever import HybridRetriever
from test_fixtures import require_fixtures

FIXTURE_FILES = ["gan.pdf", "bert.pdf", "unet.pdf", "xgboost.pdf"]

GOOD_STAGE_A_JSON = json.dumps({
    "problem": {"text": "fake problem", "source_ids": ["SOURCE_1"]},
    "contributions": [{"text": "fake contribution", "source_ids": ["SOURCE_1"]}],
    "methodology": {"text": "fake methodology", "source_ids": ["SOURCE_1"]},
    "datasets_setup": [{"text": "fake dataset", "source_ids": ["SOURCE_1"]}],
    "results": [{"text": "fake result", "source_ids": ["SOURCE_1"]}],
    "limitations": [],
    "future_work": [],
})

GOOD_SYNTHESIS_JSON = json.dumps({
    "comparable": False,
    "comparability_note": "fake incomparability note",
    "synthesis": [{"claim": "fake synthesis claim", "synthesis_type": "difference", "source_ids": ["P1_SOURCE_1", "P2_SOURCE_1"]}],
})


def build_real_paper(paper_id: str, filename: str, reranker, fixtures: dict) -> ProcessedPaper:
    """Uses the real retrieval stack so Stage A has genuine, retrievable evidence --
    only the LLM response is stubbed."""
    with open(fixtures[filename], "rb") as f:
        pages = extract_pages(f.read())
    chunks = build_chunks(clean_pages(pages), paper_name=filename)
    rp = RetrievalPipeline(HybridRetriever(chunks), reranker)
    return ProcessedPaper(paper_id=paper_id, display_name=filename, pages=pages,
                           sections=[], meta={}, chunks=chunks, retrieval_pipeline=rp)


class StubLLM:
    """Returns a scripted response based on call count, regardless of prompt content."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp


FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


def test_one_paper_api_error_other_succeeds(reranker, fixtures):
    print("\n=== Test 1: paper_2's Stage A call fails (API error); paper_1 must be unaffected ===")
    papers = [
        build_real_paper("paper_1", "gan.pdf", reranker, fixtures),
        build_real_paper("paper_2", "bert.pdf", reranker, fixtures),
    ]
    # paper_1 succeeds, paper_2's Stage A call errors out, synthesis is skipped (< 2 successes)
    llm = StubLLM([
        LLMResponse(ok=True, content=GOOD_STAGE_A_JSON),
        LLMResponse(ok=False, error="Rate limit or quota exceeded: 429"),
    ])
    result = generate_comparison(papers, llm)
    check("overall ready=True (paper_1 succeeded)", result.ready is True)
    p1 = result.papers.get("paper_1")
    p2 = result.papers.get("paper_2")
    check("paper_1 has a result object with no generation_error", p1 is not None and p1.generation_error is None)
    check("paper_1's findings are actually populated", bool(p1 and p1.problem and p1.contributions))
    check("paper_2 has a result object WITH generation_error set", p2 is not None and p2.generation_error is not None)
    check("paper_2's dimensions are empty (not fabricated)", p2 is not None and p2.problem is None and not p2.contributions)
    check("synthesis_error explains why synthesis was skipped", bool(result.synthesis_error))
    check("synthesis list is empty, not fabricated", result.synthesis == [])


def test_malformed_json_for_one_paper(reranker, fixtures):
    print("\n=== Test 2: paper_1 returns malformed JSON; paper_2 succeeds ===")
    papers = [
        build_real_paper("paper_1", "gan.pdf", reranker, fixtures),
        build_real_paper("paper_2", "bert.pdf", reranker, fixtures),
    ]
    llm = StubLLM([
        LLMResponse(ok=True, content="this is not valid json at all"),
        LLMResponse(ok=True, content=GOOD_STAGE_A_JSON),
    ])
    result = generate_comparison(papers, llm)
    p1 = result.papers.get("paper_1")
    p2 = result.papers.get("paper_2")
    check("paper_1 marked with generation_error (malformed response)",
          p1 is not None and p1.generation_error is not None)
    check("paper_2 unaffected by paper_1's malformed response", p2 is not None and p2.generation_error is None)
    check("paper_2's findings are actually populated", bool(p2 and p2.problem))


def test_both_succeed_but_synthesis_fails(reranker, fixtures):
    print("\n=== Test 3: both papers succeed in Stage A; Stage B (synthesis) fails ===")
    papers = [
        build_real_paper("paper_1", "gan.pdf", reranker, fixtures),
        build_real_paper("paper_2", "bert.pdf", reranker, fixtures),
    ]
    llm = StubLLM([
        LLMResponse(ok=True, content=GOOD_STAGE_A_JSON),
        LLMResponse(ok=True, content=GOOD_STAGE_A_JSON),
        LLMResponse(ok=False, error="Could not connect to the LLM API: Connection error."),
    ])
    result = generate_comparison(papers, llm)
    check("overall ready=True despite synthesis failure", result.ready is True)
    p1 = result.papers.get("paper_1")
    p2 = result.papers.get("paper_2")
    check("paper_1's findings preserved", bool(p1 and p1.problem and p1.generation_error is None))
    check("paper_2's findings preserved", bool(p2 and p2.problem and p2.generation_error is None))
    check("synthesis_error is set", result.synthesis_error is not None)
    check("synthesis list is empty, not fabricated", result.synthesis == [])
    check("comparable defaults to False (conservative) on synthesis failure", result.comparable is False)


def test_all_papers_fail(reranker, fixtures):
    print("\n=== Test 4: every paper's Stage A call fails -> overall failure, not a fake partial success ===")
    papers = [
        build_real_paper("paper_1", "gan.pdf", reranker, fixtures),
        build_real_paper("paper_2", "bert.pdf", reranker, fixtures),
    ]
    llm = StubLLM([
        LLMResponse(ok=False, error="boom 1"),
        LLMResponse(ok=False, error="boom 2"),
    ])
    result = generate_comparison(papers, llm)
    check("overall ready=False", result.ready is False)
    check("a clear error message is set", bool(result.error))
    check("both papers still have a result object recording their own error",
          all(result.papers[pid].generation_error is not None for pid in ("paper_1", "paper_2")))


def test_cross_paper_citation_leak_rejected_in_stage_a():
    print("\n=== Test 5: a Stage A response citing a source id that isn't in that paper's own map ===")
    from comparison_pipeline import _make_single
    # Stage A's source_map only ever contains the one paper's own chunks, so a citation
    # to a nonexistent / wrong id is simply absent from the map -- prove it's dropped.
    source_map = {"SOURCE_1": ("paper_1", {"paper_name": "gan.pdf", "page": 1, "section": "Abstract",
                                             "text": "real text", "chunk_id": "g1"})}
    bad = {"text": "a claim citing a source that does not exist in this paper's map", "source_ids": ["SOURCE_99"]}
    result = _make_single(bad, source_map, owning_paper_id="paper_1")
    check("finding with an unresolvable source id is dropped entirely", result is None)


def test_duplicate_filename_distinct_results(reranker, fixtures):
    print("\n=== Test 6: two papers with the same display_name produce fully independent results ===")
    papers = [
        build_real_paper("paper_1", "gan.pdf", reranker, fixtures),
        build_real_paper("paper_2", "gan.pdf", reranker, fixtures),  # same filename, different paper_id
    ]
    llm = StubLLM([LLMResponse(ok=True, content=GOOD_STAGE_A_JSON), LLMResponse(ok=True, content=GOOD_STAGE_A_JSON),
                   LLMResponse(ok=True, content=GOOD_SYNTHESIS_JSON)])
    result = generate_comparison(papers, llm)
    p1 = result.papers.get("paper_1")
    p2 = result.papers.get("paper_2")
    check("both entries exist independently despite identical display_name",
          p1 is not None and p2 is not None and p1 is not p2)
    check("paper_1's sources are all tagged with paper_1's own chunk_ids (own evidence)",
          all(s.chunk_id.startswith("gan.pdf") for s in (p1.problem.sources if p1.problem else [])))


def main() -> None:
    # Fixture-independent: exercises _make_single directly, no PDFs or model needed.
    test_cross_paper_citation_leak_rejected_in_stage_a()

    fixtures = require_fixtures(FIXTURE_FILES)
    if fixtures is None:
        print("\nSkipping Tests 1, 2, 3, 4, 6 (require full-paper fixtures).")
    else:
        reranker = CrossEncoderReranker()
        test_one_paper_api_error_other_succeeds(reranker, fixtures)
        test_malformed_json_for_one_paper(reranker, fixtures)
        test_both_succeed_but_synthesis_fails(reranker, fixtures)
        test_all_papers_fail(reranker, fixtures)
        test_duplicate_filename_distinct_results(reranker, fixtures)

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("\nAll executed checks passed.")


if __name__ == "__main__":
    main()
