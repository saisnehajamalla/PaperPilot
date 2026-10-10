"""Deterministic tests for provider-error translation (llm.py's translate_provider_error)
and its effect across all four pipelines.

No real API key or live provider request is used anywhere in this file. Provider
exceptions are reconstructed faithfully using the real openai SDK's own exception
construction path (OpenAI()._make_status_error_from_response against a synthetic httpx
response), so the test exercises the exact same object shapes a real Groq call would
raise -- not a hand-rolled approximation.

Usage:
    python test_llm_error_translation.py
"""

import json
import sys

import httpx2
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
)

from comparison_pipeline import ProcessedPaper, generate_comparison
from llm import LLMClient, LLMResponse, translate_provider_error
from overview_pipeline import generate_overview
from rag_pipeline import RAGPipeline
from research_gap_pipeline import generate_research_gap_analysis

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# Build real openai-SDK exception objects from synthetic HTTP responses, the same way the
# SDK itself would when the real client receives these bytes from the real API.
# ---------------------------------------------------------------------------
_CLIENT = OpenAI(api_key="fake-key-for-tests-only", base_url="https://example.invalid/v1")
_REQUEST = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


def status_error(status_code: int, body: dict | str):
    content = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
    response = httpx2.Response(status_code, content=content, request=_REQUEST)
    return _CLIENT._make_status_error_from_response(response)


def groq_rate_limit_body(message: str) -> dict:
    return {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}


SENSITIVE_SUBSTRINGS = ["org_", "{", "}", "sk-", "Error code", "api_key", "Authorization"]


def assert_message_is_clean(label: str, message: str) -> None:
    check(f"{label}: message is non-empty", bool(message))
    for marker in SENSITIVE_SUBSTRINGS:
        check(f"{label}: message does not contain '{marker}'", marker not in message)


# ---------------------------------------------------------------------------
# Category 1: translate_provider_error() unit tests
# ---------------------------------------------------------------------------
def test_token_rate_limit_tpm():
    print("\n=== Test: token-rate-limit (TPM) is translated distinctly from daily quota ===")
    e = status_error(429, groq_rate_limit_body(
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_xyz` "
        "on tokens per minute (TPM): Limit 8000, Requested 13800. Please try again in 20s."
    ))
    msg = translate_provider_error(e)
    check("TPM message is the rate-limited (not quota) message",
          msg == "The AI service is temporarily rate-limited. Please wait before trying again.")
    assert_message_is_clean("TPM", msg)


def test_daily_quota_exhaustion():
    print("\n=== Test: daily quota exhaustion (TPD) is translated distinctly from rate limit ===")
    e = status_error(429, groq_rate_limit_body(
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_xyz` "
        "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 196875, "
        "Requested 8450. Please try again in 38m20.4s."
    ))
    msg = translate_provider_error(e)
    check("TPD message is the daily-quota message",
          msg == "The AI provider's usage quota has been reached. Please try again later.")
    assert_message_is_clean("TPD", msg)


def test_request_too_large_413():
    print("\n=== Test: HTTP 413 (request/token-size limit) ===")
    e = status_error(413, groq_rate_limit_body(
        "Request too large for model `openai/gpt-oss-120b`. Limit 8000, Requested 13800 "
        "tokens per minute (TPM)."
    ))
    msg = translate_provider_error(e)
    check("413 message is the request-too-large message",
          msg == "This request exceeds the AI provider's processing limit. Try reducing the amount of content.")
    assert_message_is_clean("413", msg)


def test_connection_error():
    print("\n=== Test: connection error ===")
    e = APIConnectionError(request=_REQUEST)
    msg = translate_provider_error(e)
    check("connection-error message is the connection message",
          msg == "Could not connect to the AI service. Check your connection and try again.")
    assert_message_is_clean("connection", msg)


def test_timeout_error():
    print("\n=== Test: timeout error ===")
    e = APITimeoutError(request=_REQUEST)
    msg = translate_provider_error(e)
    check("timeout message is the timeout message",
          msg == "The AI service took too long to respond. Please try again.")
    assert_message_is_clean("timeout", msg)


def test_authentication_error():
    print("\n=== Test: authentication / invalid API key error ===")
    e = status_error(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                      "code": "invalid_api_key"}})
    msg = translate_provider_error(e)
    check("401 message is the authentication message",
          msg == "The AI service rejected the configured API credentials. Check the application's configuration.")
    assert_message_is_clean("auth", msg)


def test_unknown_provider_error():
    print("\n=== Test: unrecognized provider status error (e.g. 500) falls back to a generic message ===")
    e = status_error(500, {"error": {"message": "internal explosion", "type": "server_error"}})
    msg = translate_provider_error(e)
    check("500 message is the generic provider-error message",
          msg == "The AI service returned an error and could not complete this request.")
    assert_message_is_clean("500", msg)


def test_completely_unknown_exception():
    print("\n=== Test: a non-openai exception falls back to the generic unknown message ===")
    msg = translate_provider_error(RuntimeError("some unrelated local bug"))
    check("non-provider exception gets the generic unknown-error message",
          msg == "An unexpected error occurred while contacting the AI service.")
    assert_message_is_clean("unknown", msg)


def test_rate_limit_with_missing_structured_metadata():
    print("\n=== Test: a 429 whose body doesn't have the expected shape still degrades safely ===")
    # Not valid JSON at all -- body becomes the raw string, not a dict.
    e = status_error(429, "Too Many Requests")
    msg = translate_provider_error(e)
    check("falls back to the generic rate-limit message when metadata is missing",
          msg == "The AI service is temporarily rate-limited. Please wait before trying again.")
    assert_message_is_clean("missing-metadata", msg)

    # Valid JSON, but no "message" key inside "error".
    e2 = status_error(429, {"error": {"type": "tokens", "code": "rate_limit_exceeded"}})
    msg2 = translate_provider_error(e2)
    check("falls back to the generic rate-limit message when 'message' key is absent",
          msg2 == "The AI service is temporarily rate-limited. Please wait before trying again.")


# ---------------------------------------------------------------------------
# Category 2: LLMClient integration -- exception to LLMResponse, end to end
# ---------------------------------------------------------------------------
class _RaisingCompletions:
    def __init__(self, exc: Exception):
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


class _RaisingChat:
    def __init__(self, exc: Exception):
        self.completions = _RaisingCompletions(exc)


class _RaisingClient:
    def __init__(self, exc: Exception):
        self.chat = _RaisingChat(exc)


def llm_client_raising(exc: Exception) -> LLMClient:
    client = LLMClient(api_key="fake-key-for-tests-only")
    client._client = _RaisingClient(exc)  # bypasses _get_client()'s real OpenAI() construction
    return client


def test_llmclient_translates_and_preserves_raw():
    print("\n=== Test: LLMClient.complete_json translates .error and preserves .raw_error ===")
    raw_message = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_secret123` "
        "on tokens per day (TPD): Limit 200000, Used 199999, Requested 10."
    )
    e = status_error(429, groq_rate_limit_body(raw_message))
    client = llm_client_raising(e)
    response = client.complete_json("system", "user")
    check("response.ok is False", response.ok is False)
    check("response.error is the short translated message",
          response.error == "The AI provider's usage quota has been reached. Please try again later.")
    check("response.raw_error preserves the original diagnostic detail (not shown to users)",
          response.raw_error is not None and "org_secret123" in response.raw_error)
    assert_message_is_clean("LLMClient .error", response.error)


def test_llmclient_no_crash_across_all_categories():
    print("\n=== Test: LLMClient never raises for any of the covered exception categories ===")
    cases = [
        status_error(429, groq_rate_limit_body("... on tokens per minute (TPM): ...")),
        status_error(429, groq_rate_limit_body("... on tokens per day (TPD): ...")),
        status_error(413, groq_rate_limit_body("Request too large ...")),
        APIConnectionError(request=_REQUEST),
        APITimeoutError(request=_REQUEST),
        status_error(401, {"error": {"message": "Invalid API Key"}}),
        status_error(500, {"error": {"message": "boom"}}),
        RuntimeError("totally unrelated"),
    ]
    all_ok = True
    for exc in cases:
        client = llm_client_raising(exc)
        try:
            response = client.complete_json("system", "user")
            if response.ok or not response.error:
                all_ok = False
        except Exception:
            all_ok = False
    check("every case returned a non-ok LLMResponse with a non-empty .error, no exception raised", all_ok)


# ---------------------------------------------------------------------------
# Category 3: the four pipelines handle a translated failure without crashing, and
# comparison partial failures stay distinguishable from genuine missing evidence.
# ---------------------------------------------------------------------------
class _FakeCandidate:
    def __init__(self, chunk):
        self.chunk = chunk


class _FakeRankedChunk:
    def __init__(self, chunk):
        self.candidate = _FakeCandidate(chunk)


class _FakeRetrieveResult:
    def __init__(self, accepted, reranked):
        self.accepted = accepted
        self.reranked = reranked


class _FakeRetriever:
    chunks = [{"chunk_id": "stub-c1"}]  # just needs to be truthy


class _FakeRetrievalPipeline:
    """Always returns one accepted, well-formed chunk for any query -- enough for every
    pipeline to reach its LLM call, without any real embedding/reranking/PDF."""

    def __init__(self):
        self.retriever = _FakeRetriever()

    def retrieve(self, question: str):
        chunk = {
            "chunk_id": f"stub-{abs(hash(question)) % 1000}",
            "paper_name": "stub.pdf",
            "page": 1,
            "section": "Abstract",
            "text": "stub evidence text",
        }
        return _FakeRetrieveResult(accepted=True, reranked=[_FakeRankedChunk(chunk)])


class StubLLM:
    """Returns a scripted LLMResponse based on call count, regardless of prompt content."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp

    complete_text = complete_json


TRANSLATED_QUOTA_MSG = "The AI provider's usage quota has been reached. Please try again later."


def test_rag_pipeline_handles_translated_failure():
    print("\n=== Test: rag_pipeline.answer() surfaces a translated error without crashing ===")
    llm = StubLLM([LLMResponse(ok=False, error=TRANSLATED_QUOTA_MSG, raw_error="org_xyz raw dump")])
    rag = RAGPipeline(_FakeRetrievalPipeline(), llm)
    result = rag.answer("What problem does this paper address?")
    check("reason is llm_error", result.reason == "llm_error")
    check("the translated message appears in the answer", TRANSLATED_QUOTA_MSG in result.answer)
    check("no raw diagnostic (org id) leaked into the answer", "org_xyz" not in result.answer)


def test_overview_pipeline_handles_translated_failure():
    print("\n=== Test: overview_pipeline.generate_overview() surfaces a translated error ===")
    llm = StubLLM([LLMResponse(ok=False, error=TRANSLATED_QUOTA_MSG, raw_error="org_xyz raw dump")])
    result = generate_overview(_FakeRetrievalPipeline(), llm)
    check("ready is False", result.ready is False)
    check("error is exactly the translated message (not a raw dump)", result.error == TRANSLATED_QUOTA_MSG)


def test_research_gap_pipeline_handles_translated_failure():
    print("\n=== Test: research_gap_pipeline.generate_research_gap_analysis() surfaces a translated error ===")
    llm = StubLLM([LLMResponse(ok=False, error=TRANSLATED_QUOTA_MSG, raw_error="org_xyz raw dump")])
    result = generate_research_gap_analysis(_FakeRetrievalPipeline(), llm)
    check("ready is False", result.ready is False)
    check("error is exactly the translated message (not a raw dump)", result.error == TRANSLATED_QUOTA_MSG)


def test_comparison_pipeline_partial_failure_stays_distinguishable():
    print("\n=== Test: comparison_pipeline isolates a translated per-paper failure from the other paper ===")
    good_json = json.dumps({
        "problem": {"text": "fake problem", "source_ids": ["SOURCE_1"]},
        "contributions": [{"text": "fake contribution", "source_ids": ["SOURCE_1"]}],
        "methodology": {"text": "fake methodology", "source_ids": ["SOURCE_1"]},
        "datasets_setup": [], "results": [], "limitations": [], "future_work": [],
    })
    papers = [
        ProcessedPaper(paper_id="paper_1", display_name="a.pdf", pages=[], sections=[],
                        meta={}, chunks=[], retrieval_pipeline=_FakeRetrievalPipeline()),
        ProcessedPaper(paper_id="paper_2", display_name="b.pdf", pages=[], sections=[],
                        meta={}, chunks=[], retrieval_pipeline=_FakeRetrievalPipeline()),
    ]
    llm = StubLLM([
        LLMResponse(ok=True, content=good_json),  # paper_1 Stage A succeeds
        LLMResponse(ok=False, error=TRANSLATED_QUOTA_MSG, raw_error="org_xyz raw dump"),  # paper_2 fails
    ])
    result = generate_comparison(papers, llm)
    p1 = result.papers.get("paper_1")
    p2 = result.papers.get("paper_2")
    check("paper_1 succeeded and is unaffected", p1 is not None and p1.generation_error is None
          and p1.problem is not None)
    check("paper_2's generation_error is exactly the translated message (a real failure)",
          p2 is not None and p2.generation_error == TRANSLATED_QUOTA_MSG)
    check("paper_2's failure is NOT rendered as 'not reported' empty dimensions silently -- "
          "generation_error is set, distinguishing it from genuine missing evidence",
          p2.generation_error is not None and p2.problem is None and not p2.contributions)
    check("no raw diagnostic (org id) leaked into the per-paper error",
          "org_xyz" not in (p2.generation_error or ""))


def main() -> None:
    test_token_rate_limit_tpm()
    test_daily_quota_exhaustion()
    test_request_too_large_413()
    test_connection_error()
    test_timeout_error()
    test_authentication_error()
    test_unknown_provider_error()
    test_completely_unknown_exception()
    test_rate_limit_with_missing_structured_metadata()

    test_llmclient_translates_and_preserves_raw()
    test_llmclient_no_crash_across_all_categories()

    test_rag_pipeline_handles_translated_failure()
    test_overview_pipeline_handles_translated_failure()
    test_research_gap_pipeline_handles_translated_failure()
    test_comparison_pipeline_partial_failure_stays_distinguishable()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("\nAll executed checks passed.")


if __name__ == "__main__":
    main()
