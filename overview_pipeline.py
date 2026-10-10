"""Grounded paper-overview generation: category-targeted retrieval -> one structured LLM call.

This is an orchestration layer only. It calls the existing, calibrated RetrievalPipeline
once per overview category (same reranker, threshold, weights — nothing about retrieval
itself changes) and combines the results into a single structured LLM request, rather
than sending the whole paper or making one call per category.

Same defense-in-depth as rag_pipeline.py: if a category's retrieval was never accepted,
its evidence is withheld from the prompt entirely, and the final result for that category
is forced to a canonical "not found" sentence in code — regardless of what the LLM
produces — so the model cannot hallucinate content for a category the paper doesn't have.
"""

import json
from dataclasses import dataclass, field

from citation import Source, build_source, strip_source_labels
from llm import LLMClient
from retrieval_pipeline import RankedChunk, RetrievalPipeline

EVIDENCE_PER_CATEGORY = 5

CATEGORY_QUERIES = {
    "summary": "What is this paper about overall, including its main topic and approach?",
    "problem": "What problem does this paper address?",
    "contributions": "What are the main contributions of this paper?",
    "methodology": "What methodology or model does the paper propose?",
    "datasets": "What datasets or data does the paper use?",
    "results": "What are the key experimental results?",
    "limitations": "What limitations or weaknesses are mentioned by the authors?",
    "future_work": "What future work or extensions does the paper suggest?",
}

ABSENCE_TEXT = {
    "problem": "No explicit problem statement was identified in the paper.",
    "contributions": "No explicit contributions were identified in the paper.",
    "methodology": "No explicit methodology was identified in the paper.",
    "datasets": "No explicit datasets were identified in the paper.",
    "results": "No explicit results were identified in the paper.",
    "limitations": "No explicit limitations were identified in the paper.",
    "future_work": "No explicit future work was identified in the paper.",
}

SINGLE_ITEM_CATEGORIES = {"summary", "problem", "methodology"}
FALLBACK_TEXT = "Could not generate this section."


@dataclass
class OverviewItem:
    text: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class OverviewResult:
    ready: bool = False
    error: str | None = None
    summary: OverviewItem = None
    problem: OverviewItem = None
    contributions: list[OverviewItem] = field(default_factory=list)
    methodology: OverviewItem = None
    datasets: list[OverviewItem] = field(default_factory=list)
    results: list[OverviewItem] = field(default_factory=list)
    limitations: list[OverviewItem] = field(default_factory=list)
    future_work: list[OverviewItem] = field(default_factory=list)


def _gather_evidence(retrieval_pipeline: RetrievalPipeline) -> tuple[dict[str, list[RankedChunk]], list[str]]:
    evidence: dict[str, list[RankedChunk]] = {}
    unavailable: list[str] = []
    for category, query in CATEGORY_QUERIES.items():
        result = retrieval_pipeline.retrieve(query)
        if result.accepted:
            evidence[category] = result.reranked[:EVIDENCE_PER_CATEGORY]
        else:
            evidence[category] = []
            if category in ABSENCE_TEXT:
                unavailable.append(category)

    if not evidence["summary"]:
        # The dedicated summary query found nothing reliable; fall back to whatever
        # other categories did find, rather than leaving the summary with no evidence.
        combined, seen = [], set()
        for category in ("problem", "contributions", "methodology", "results"):
            for rc in evidence.get(category, []):
                cid = rc.candidate.chunk["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    combined.append(rc)
        evidence["summary"] = combined[:EVIDENCE_PER_CATEGORY]

    return evidence, unavailable


def _build_context(evidence: dict[str, list[RankedChunk]]) -> tuple[str, str, dict[str, dict]]:
    source_map: dict[str, dict] = {}
    chunk_to_sid: dict[str, str] = {}
    unique_blocks: list[str] = []
    category_lines: list[str] = []

    for category, chunks in evidence.items():
        if not chunks:
            category_lines.append(f"{category}: (no reliable evidence retrieved)")
            continue
        sids = []
        for rc in chunks:
            chunk = rc.candidate.chunk
            cid = chunk["chunk_id"]
            if cid not in chunk_to_sid:
                sid = f"SOURCE_{len(chunk_to_sid) + 1}"
                chunk_to_sid[cid] = sid
                source_map[sid] = chunk
                unique_blocks.append(
                    f"[{sid}]\nPaper: {chunk['paper_name']}\nPage: {chunk['page']}\nSection: {chunk['section']}\n\n{chunk['text']}"
                )
            sids.append(chunk_to_sid[cid])
        category_lines.append(f"{category}: {', '.join(sids)}")

    return "\n\n".join(unique_blocks), "\n".join(category_lines), source_map


def _build_system_prompt(unavailable: list[str]) -> str:
    absence_rules = "\n".join(f'- {c}: output exactly "{ABSENCE_TEXT[c]}"' for c in unavailable)
    absence_block = (
        f"\nFor the following categories, no reliable evidence was retrieved from the paper. "
        f"For each, you MUST NOT attempt to answer from unrelated excerpts — do not infer that "
        f"the authors must have discussed it just because it would be a logical thing to include. "
        f"Instead:\n{absence_rules}\n"
        if unavailable else ""
    )
    return f"""You are PaperPilot, an AI research-paper assistant generating a structured overview of an uploaded paper.

You are given excerpts from the paper labeled SOURCE_1, SOURCE_2, etc., followed by a category index showing which source ids were retrieved for each overview category. An excerpt may be relevant to more than one category even if it is only indexed under one; use your judgment, but only among the supplied excerpts.

Do not use outside knowledge. Do not invent facts, datasets, results, contributions, limitations, future work, numbers, page numbers, sections, or quotations — page and section metadata is attached separately by the application, not by you.
{absence_block}
For categories with evidence, base your answer only on the supplied excerpts. If the authors explicitly state something (an explicit list of contributions, or an explicit "future work" or "limitations" discussion), prefer their own framing. If no explicit statement exists but relevant information is present in the excerpts, you may produce a concise synthesis, but do not present that synthesis as a direct, verbatim author claim.

For "limitations" specifically: only report something as a limitation if the authors themselves present it negatively — as a weakness, drawback, failure case, constraint, something that does not work well, or something they explicitly call a limitation. Do NOT take a finding the authors present positively (e.g. "scaling up model size improves results", "this architecture is simpler/faster/shared across stages") and reframe it as a limitation by speculating about a downside they did not state (e.g. turning "bigger models help" into "may be limited without scaling"). A footnote explaining an evaluation choice (e.g. excluding one dataset from a benchmark average) is not itself a limitation of the paper's method unless the authors frame it that way. If you cannot find anything the authors present negatively, return exactly the single absence sentence for that category instead of inventing a negative spin on a positive result.

For "future_work" specifically: only report something the authors explicitly propose as a next step, extension, or direction for later work — not a capability the method obviously lacks today.

If a category legitimately has more than one distinct point (e.g. several contributions, datasets, results, or limitations), return multiple items for it, each with its own supporting source_ids. Do not merge unrelated points into one item, and do not cite a source id whose content you are not actually using.

Every "text" field must read as plain prose with no citation markers of any kind — never write "SOURCE_1" or similar inside any text field. Source ids belong only in the separate "source_ids" fields.

The summary should be 3-5 sentences, grounded only in the supplied excerpts, and should not simply repeat the entire paper.

Respond ONLY with valid JSON matching this schema:
{{
  "summary": {{"text": string, "source_ids": [string]}},
  "problem": {{"text": string, "source_ids": [string]}},
  "contributions": [{{"text": string, "source_ids": [string]}}],
  "methodology": {{"text": string, "source_ids": [string]}},
  "datasets": [{{"text": string, "source_ids": [string]}}],
  "results": [{{"text": string, "source_ids": [string]}}],
  "limitations": [{{"text": string, "source_ids": [string]}}],
  "future_work": [{{"text": string, "source_ids": [string]}}]
}}"""


def _make_item(data, source_map: dict[str, dict]) -> OverviewItem:
    if not isinstance(data, dict):
        return OverviewItem(text=FALLBACK_TEXT)
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return OverviewItem(text=FALLBACK_TEXT)
    text = strip_source_labels(text)
    if not text.strip():
        return OverviewItem(text=FALLBACK_TEXT)
    raw_ids = data.get("source_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
    sources = [build_source(sid, source_map[sid]) for sid in raw_ids if isinstance(sid, str) and sid in source_map]
    return OverviewItem(text=text, sources=sources)


def _make_list(data, source_map: dict[str, dict]) -> list[OverviewItem]:
    if not isinstance(data, list) or not data:
        return [OverviewItem(text=FALLBACK_TEXT)]
    items = [_make_item(x, source_map) for x in data]
    return items or [OverviewItem(text=FALLBACK_TEXT)]


def _assemble_result(data: dict, source_map: dict[str, dict], unavailable: list[str]) -> OverviewResult:
    result = OverviewResult(ready=True)
    result.summary = _make_item(data.get("summary"), source_map)
    result.problem = _make_item(data.get("problem"), source_map)
    result.methodology = _make_item(data.get("methodology"), source_map)
    result.contributions = _make_list(data.get("contributions"), source_map)
    result.datasets = _make_list(data.get("datasets"), source_map)
    result.results = _make_list(data.get("results"), source_map)
    result.limitations = _make_list(data.get("limitations"), source_map)
    result.future_work = _make_list(data.get("future_work"), source_map)

    # Hard override: force the canonical "not found" sentence for any category whose
    # retrieval was never accepted, no matter what the LLM produced.
    for category in unavailable:
        forced = OverviewItem(text=ABSENCE_TEXT[category])
        if category in SINGLE_ITEM_CATEGORIES:
            setattr(result, category, forced)
        else:
            setattr(result, category, [forced])
    return result


def generate_overview(retrieval_pipeline: RetrievalPipeline, llm_client: LLMClient) -> OverviewResult:
    if not retrieval_pipeline.retriever.chunks:
        return OverviewResult(ready=False, error="No usable text was extracted from this PDF.")

    evidence, unavailable = _gather_evidence(retrieval_pipeline)
    context, index, source_map = _build_context(evidence)
    if not source_map:
        return OverviewResult(ready=False, error="No reliable evidence could be retrieved from this paper.")

    system_prompt = _build_system_prompt(unavailable)
    user_prompt = f"Category index (source ids retrieved per category):\n{index}\n\nExcerpts:\n\n{context}"
    response = llm_client.complete_json(system_prompt, user_prompt, temperature=0.1)

    if not response.ok:
        return OverviewResult(ready=False, error=response.error)

    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return OverviewResult(ready=False, error="The AI model returned a response in an unexpected format.")
    if not isinstance(data, dict):
        return OverviewResult(ready=False, error="The AI model returned a response in an unexpected format.")

    return _assemble_result(data, source_map, unavailable)
