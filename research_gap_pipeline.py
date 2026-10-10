"""Evidence-grounded research-gap analysis: category-targeted retrieval -> one structured
LLM call -> four strictly-separated categories of finding.

This module does not touch the existing RetrievalPipeline, RAGPipeline, or
overview_pipeline — it only orchestrates additional retrieval calls against the same,
unmodified, calibrated pipeline and builds its own structured prompt.

Four categories, kept separate everywhere (schema, code, UI):
  A. author_stated_limitations   - a weakness the AUTHORS present negatively
  B. author_stated_future_work   - a direction the AUTHORS explicitly propose
  C. potential_gaps              - PaperPilot's own inference, clearly labeled as such
  D. potential_research_directions - a suggestion PaperPilot generates from A/C, labeled
                                      as unverified for novelty

Same defense-in-depth as rag_pipeline.py / overview_pipeline.py: an item survives only if
it cites at least one source id that resolves to a real chunk; categories A/B are hard-
forced to empty if their own retrieval was never accepted, regardless of what the LLM
produced, so a paper with no explicit limitations/future work cannot have them invented.
"""

import json
from dataclasses import dataclass, field

from citation import Source, build_source, strip_source_labels
from llm import LLMClient
from retrieval_pipeline import RankedChunk, RetrievalPipeline

EVIDENCE_PER_CATEGORY = 5

CATEGORY_QUERIES = {
    "problem": "What problem does this paper address?",
    "contributions": "What does this paper demonstrably solve or contribute?",
    "methodology": "What methodology, model, or key assumptions does the paper rely on?",
    "evaluation_scope": "What experiments were performed and what is the scope of the "
                         "evaluation, including datasets, metrics, and conditions tested?",
    "limitations": "What limitations or weaknesses are mentioned by the authors?",
    "future_work": "What future work or extensions does the paper suggest?",
    "conclusions": "What conclusions do the authors draw, and what questions remain "
                    "unresolved or unaddressed?",
}

HARD_OVERRIDE_CATEGORIES = {"limitations", "future_work"}


@dataclass
class LimitationItem:
    claim: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class FutureWorkItem:
    claim: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class PotentialGap:
    title: str
    observation: str
    reasoning: str
    not_established: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class ResearchDirection:
    title: str
    research_question: str
    motivation: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class ResearchGapResult:
    ready: bool = False
    error: str | None = None
    limitations: list[LimitationItem] = field(default_factory=list)
    future_work: list[FutureWorkItem] = field(default_factory=list)
    potential_gaps: list[PotentialGap] = field(default_factory=list)
    research_directions: list[ResearchDirection] = field(default_factory=list)


def _gather_evidence(retrieval_pipeline: RetrievalPipeline) -> tuple[dict[str, list[RankedChunk]], list[str]]:
    evidence: dict[str, list[RankedChunk]] = {}
    unavailable: list[str] = []
    for category, query in CATEGORY_QUERIES.items():
        result = retrieval_pipeline.retrieve(query)
        if result.accepted:
            evidence[category] = result.reranked[:EVIDENCE_PER_CATEGORY]
        else:
            evidence[category] = []
            if category in HARD_OVERRIDE_CATEGORIES:
                unavailable.append(category)
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
    notes = []
    if "limitations" in unavailable:
        notes.append(
            '- No reliable evidence was retrieved for "limitations". Return an empty list '
            'for author_stated_limitations — do not attempt to infer a limitation from '
            'unrelated excerpts.'
        )
    if "future_work" in unavailable:
        notes.append(
            '- No reliable evidence was retrieved for "future_work". Return an empty list '
            'for author_stated_future_work — do not attempt to infer a proposal from '
            'unrelated excerpts.'
        )
    unavailable_block = ("\n" + "\n".join(notes) + "\n") if notes else ""

    return f"""You are PaperPilot, an AI research-paper assistant performing a conservative, evidence-grounded research-gap analysis of an uploaded paper.

You are given excerpts labeled SOURCE_1, SOURCE_2, etc., followed by a category index showing which source ids were retrieved for each evidence category (problem, contributions, methodology, evaluation_scope, limitations, future_work, conclusions). An excerpt may be relevant to more than one output category even if indexed under one.

Do not use outside knowledge about this research area. Do not invent limitations, future work, experimental findings, datasets, metrics, page numbers, sections, or quotations. Never claim that the entire research field has or has not solved something — you only have access to this one paper, which cannot establish field-wide novelty or absence of prior work.
{unavailable_block}
Sort every finding into exactly one of four categories:

1. author_stated_limitations — a weakness, constraint, unresolved issue, or limitation the AUTHORS THEMSELVES present negatively (as a weakness, drawback, failure case, something that does not work well, or something they explicitly call a limitation). Do NOT reframe a positive or neutral finding as a limitation (e.g. do not turn "scaling helps" into "may be limited without scaling"). A footnote explaining an evaluation choice is not itself a limitation unless the authors frame it that way. If you cannot find anything the authors present negatively, return an empty list.

2. author_stated_future_work — an extension or direction the AUTHORS THEMSELVES explicitly propose, anywhere in the paper, not only under a heading literally named "Future Work". Do not invent a direction the authors did not propose. If none exists, return an empty list.

3. potential_gaps — a possible research gap YOU infer by analyzing what the paper investigates, evaluates, assumes, or leaves unresolved. This is PaperPilot's own reasoning, not an author claim. CRITICAL RULE: "the paper did not evaluate X" is NOT by itself evidence that X is a gap — a paper may legitimately have a narrow, intentional scope, and not testing something is not automatically a weakness. Only propose an item when the evidence supports a specific, useful open question. For every item, provide all of: "observation" (what the paper actually says or demonstrates, grounded in the excerpts), "reasoning" (why a specific related question remains unestablished by this evidence — never "they didn't test X" alone), and "what_the_paper_does_not_establish" (phrased as a gap in THIS PAPER's evidence only, e.g. "this paper does not establish whether the method generalizes to Y" — never "no prior work has solved Y" or "the field lacks Y", since one paper cannot establish that; you may note that further literature review would be needed to know whether this is already addressed elsewhere). If the evidence does not support a specific, well-reasoned gap, return an empty list rather than inventing one.

4. potential_research_directions — a suggested investigation, generated BY PAPERPILOT, motivated by a specific item from category 1 or 3 above. For every item, provide "title", a plausible "research_question", and "motivation" naming which specific limitation or gap motivates it, with evidence. Never claim the direction is novel, unsolved across the field, or publication-ready — you have not reviewed the broader literature, so do not make claims that would require it.

For every item in every category, "source_ids" must list only the SOURCE_n labels that genuinely support that specific item — do not cite a source whose content you are not using. If you cannot support an item with at least one real source id, omit that item.

Every free-text field must read as plain prose with no citation markers of any kind in it — never write "SOURCE_1" or similar inside any text field. Source ids belong only in the separate "source_ids" fields.

Respond ONLY with valid JSON matching this schema:
{{
  "author_stated_limitations": [{{"claim": string, "source_ids": [string]}}],
  "author_stated_future_work": [{{"claim": string, "source_ids": [string]}}],
  "potential_gaps": [{{"title": string, "observation": string, "reasoning": string, "what_the_paper_does_not_establish": string, "source_ids": [string]}}],
  "potential_research_directions": [{{"title": string, "research_question": string, "motivation": string, "source_ids": [string]}}]
}}
Each list may be empty. Prefer an empty list over an unsupported or invented item."""


def _valid_sources(raw_ids, source_map: dict[str, dict]) -> list[Source]:
    if not isinstance(raw_ids, list):
        return []
    return [build_source(sid, source_map[sid]) for sid in raw_ids if isinstance(sid, str) and sid in source_map]


def _clean_text(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = strip_source_labels(value)
    return cleaned if cleaned.strip() else None


def _make_limitations(data, source_map) -> list[LimitationItem]:
    if not isinstance(data, list):
        return []
    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        claim = _clean_text(entry.get("claim"))
        sources = _valid_sources(entry.get("source_ids"), source_map)
        if claim and sources:
            items.append(LimitationItem(claim=claim, sources=sources))
    return items


def _make_future_work(data, source_map) -> list[FutureWorkItem]:
    if not isinstance(data, list):
        return []
    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        claim = _clean_text(entry.get("claim"))
        sources = _valid_sources(entry.get("source_ids"), source_map)
        if claim and sources:
            items.append(FutureWorkItem(claim=claim, sources=sources))
    return items


def _make_gaps(data, source_map) -> list[PotentialGap]:
    if not isinstance(data, list):
        return []
    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        title = _clean_text(entry.get("title"))
        observation = _clean_text(entry.get("observation"))
        reasoning = _clean_text(entry.get("reasoning"))
        not_established = _clean_text(entry.get("what_the_paper_does_not_establish"))
        sources = _valid_sources(entry.get("source_ids"), source_map)
        if title and observation and reasoning and not_established and sources:
            items.append(PotentialGap(
                title=title, observation=observation, reasoning=reasoning,
                not_established=not_established, sources=sources,
            ))
    return items


def _make_directions(data, source_map) -> list[ResearchDirection]:
    if not isinstance(data, list):
        return []
    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        title = _clean_text(entry.get("title"))
        question = _clean_text(entry.get("research_question"))
        motivation = _clean_text(entry.get("motivation"))
        sources = _valid_sources(entry.get("source_ids"), source_map)
        if title and question and motivation and sources:
            items.append(ResearchDirection(
                title=title, research_question=question, motivation=motivation, sources=sources,
            ))
    return items


def generate_research_gap_analysis(retrieval_pipeline: RetrievalPipeline, llm_client: LLMClient) -> ResearchGapResult:
    if not retrieval_pipeline.retriever.chunks:
        return ResearchGapResult(ready=False, error="No usable text was extracted from this PDF.")

    evidence, unavailable = _gather_evidence(retrieval_pipeline)
    context, index, source_map = _build_context(evidence)
    if not source_map:
        return ResearchGapResult(ready=False, error="No reliable evidence could be retrieved from this paper.")

    system_prompt = _build_system_prompt(unavailable)
    user_prompt = f"Category index (source ids retrieved per evidence category):\n{index}\n\nExcerpts:\n\n{context}"
    response = llm_client.complete_json(system_prompt, user_prompt, temperature=0.1)

    if not response.ok:
        return ResearchGapResult(ready=False, error=response.error)

    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return ResearchGapResult(ready=False, error="The AI model returned a response in an unexpected format.")
    if not isinstance(data, dict):
        return ResearchGapResult(ready=False, error="The AI model returned a response in an unexpected format.")

    result = ResearchGapResult(ready=True)
    result.limitations = _make_limitations(data.get("author_stated_limitations"), source_map)
    result.future_work = _make_future_work(data.get("author_stated_future_work"), source_map)
    result.potential_gaps = _make_gaps(data.get("potential_gaps"), source_map)
    result.research_directions = _make_directions(data.get("potential_research_directions"), source_map)

    # Hard override: a category whose retrieval was never accepted must not surface any
    # items, no matter what the LLM produced.
    if "limitations" in unavailable:
        result.limitations = []
    if "future_work" in unavailable:
        result.future_work = []

    return result
