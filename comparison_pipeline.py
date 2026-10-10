"""Evidence-grounded multi-paper comparison, generated in two independent stages.

Root cause of the earlier reliability defect: a single LLM call asked to produce findings
for every paper across all seven dimensions sometimes silently omitted an entire paper's
entry from the "papers" object in its JSON response -- confirmed by inspecting the raw
response directly (valid, short JSON that simply never included a "paper_2" key, not a
truncation or parsing failure). One shared call gave the model room to drop a paper with
no way to tell "the model found nothing" apart from "the model forgot this paper exists".

Fix: split generation into two independent stages.

  Stage A: one LLM call PER PAPER, using only that paper's own evidence and source ids.
           A failure here (API error, malformed JSON, or no usable evidence) is recorded
           on that paper's own PaperComparisonData.generation_error and does not touch
           any other paper's result.
  Stage B: one LLM call for cross-paper synthesis, built only from papers that succeeded
           in Stage A, reusing their already-retrieved evidence (trimmed) rather than
           re-retrieving. A Stage B failure is recorded in ComparisonResult.synthesis_error
           and never erases the Stage A results already collected.

Isolation guarantee unchanged: each paper has its own HybridRetriever/RetrievalPipeline
built only from that paper's own chunks, so there is no shared index a finding could leak
through. Source labels are paper-prefixed where more than one paper shares a context
(Stage B); the source map stores the owning paper_id alongside each chunk, so a finding
can be validated to only ever cite its own paper's evidence, even if two uploaded files
share a filename.
"""

import json
from dataclasses import dataclass, field

from citation import Source, build_source, strip_source_labels
from llm import LLMClient
from retrieval_pipeline import RankedChunk, RetrievalPipeline

# Stage A is always exactly one paper, so its budget doesn't need to shrink as the overall
# comparison grows -- it matches the already-proven per-category budget used by
# overview_pipeline.py / research_gap_pipeline.py for single-paper structured extraction.
STAGE_A_EVIDENCE_PER_CATEGORY = 5

MIN_PAPERS = 2
MAX_PAPERS = 5


def _evidence_budget(num_papers: int) -> int:
    """Chunks per dimension per paper for Stage B's (multi-paper) synthesis context.

    A live 5-paper run of the old combined-call design measured ~13,800 tokens against
    this provider's 8,000 tokens-per-minute limit at 3 chunks/dimension/paper; 1
    chunk/dimension/paper stayed comfortably under it. Stage B's job is lighter than the
    old combined call (it reasons over already-extracted findings, not raw extraction),
    so this budget only needs to supply a few citable excerpts, not full coverage.
    """
    if num_papers <= 2:
        return 3
    if num_papers == 3:
        return 2
    return 1


DIMENSION_QUERIES = {
    # Deliberately short, single-clause questions: a compound multi-clause query was
    # measured to push the reranker score below the acceptance threshold on chunks the
    # short form retrieves comfortably -- extra clauses dilute the query embedding's
    # similarity to any single passage. Guidance on what to extract belongs in the system
    # prompt, not here.
    "problem": "What problem does this paper address?",
    "contributions": "What are the main contributions of this paper?",
    "methodology": "What methodology or model does the paper propose?",
    "datasets_setup": "What datasets or data does the paper use?",
    "results": "What are the key experimental results?",
    "limitations": "What limitations or weaknesses are mentioned by the authors?",
    "future_work": "What future work or extensions does the paper suggest?",
}

HARD_OVERRIDE_DIMENSIONS = {"limitations", "future_work"}

SYNTHESIS_TYPES = {
    "shared_goal", "similarity", "difference", "complementary",
    "tradeoff", "evaluation_difference", "unestablished_gap",
}


@dataclass
class ProcessedPaper:
    """A single uploaded paper, fully processed and independently indexed."""
    paper_id: str
    display_name: str
    pages: list[dict]
    sections: list[dict]
    meta: dict
    chunks: list[dict]
    retrieval_pipeline: RetrievalPipeline


@dataclass
class ComparisonFinding:
    text: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class PaperComparisonData:
    paper_id: str
    display_name: str
    problem: ComparisonFinding | None = None
    contributions: list[ComparisonFinding] = field(default_factory=list)
    methodology: ComparisonFinding | None = None
    datasets_setup: list[ComparisonFinding] = field(default_factory=list)
    results: list[ComparisonFinding] = field(default_factory=list)
    limitations: list[ComparisonFinding] = field(default_factory=list)
    future_work: list[ComparisonFinding] = field(default_factory=list)
    # Set only when Stage A's own LLM call failed or returned something unusable for THIS
    # paper -- distinct from a dimension that is simply, correctly, empty. The UI must
    # show these differently: "could not be generated" vs "not reported in the paper".
    generation_error: str | None = None


@dataclass
class SynthesisItem:
    claim: str
    synthesis_type: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class ComparisonResult:
    ready: bool = False
    error: str | None = None
    papers: dict[str, PaperComparisonData] = field(default_factory=dict)
    comparable: bool | None = None
    comparability_note: str = ""
    synthesis: list[SynthesisItem] = field(default_factory=list)
    # Set only when Stage B failed outright; per-paper results in `papers` remain valid
    # and must still be shown.
    synthesis_error: str | None = None


def _gather_paper_evidence(
    paper: ProcessedPaper, evidence_per_category: int
) -> tuple[dict[str, list[RankedChunk]], list[str]]:
    evidence: dict[str, list[RankedChunk]] = {}
    unavailable: list[str] = []
    for dimension, query in DIMENSION_QUERIES.items():
        result = paper.retrieval_pipeline.retrieve(query)
        if result.accepted:
            evidence[dimension] = result.reranked[:evidence_per_category]
        else:
            evidence[dimension] = []
            if dimension in HARD_OVERRIDE_DIMENSIONS:
                unavailable.append(dimension)
    return evidence, unavailable


def _build_paper_block(
    paper: ProcessedPaper, evidence: dict[str, list[RankedChunk]], prefix: str
) -> tuple[str, dict[str, tuple[str, dict]]]:
    """Build one paper's labeled evidence block.

    prefix="" gives bare SOURCE_n labels (Stage A: always exactly one paper in context,
    so no disambiguation is needed). A non-empty prefix (e.g. "P1") gives paper-prefixed
    labels (Stage B: multiple papers share one context and must stay disambiguated).
    """
    chunk_to_label: dict[str, str] = {}
    source_map: dict[str, tuple[str, dict]] = {}
    source_blocks: list[str] = []
    dimension_lines: list[str] = []
    label_prefix = f"{prefix}_SOURCE_" if prefix else "SOURCE_"

    for dimension, chunks in evidence.items():
        if not chunks:
            dimension_lines.append(f"  {dimension}: (no reliable evidence retrieved)")
            continue
        labels = []
        for rc in chunks:
            chunk = rc.candidate.chunk
            cid = chunk["chunk_id"]
            if cid not in chunk_to_label:
                label = f"{label_prefix}{len(chunk_to_label) + 1}"
                chunk_to_label[cid] = label
                source_map[label] = (paper.paper_id, chunk)
                source_blocks.append(
                    f"[{label}]\nPage: {chunk['page']}\nSection: {chunk['section']}\n\n{chunk['text']}"
                )
            labels.append(chunk_to_label[cid])
        dimension_lines.append(f"  {dimension}: {', '.join(labels)}")

    header = f"=== {paper.paper_id} (\"{paper.display_name}\") ===\n" if prefix else ""
    block = header + "Dimension index:\n" + "\n".join(dimension_lines) + "\n\n" + "\n\n".join(source_blocks)
    return block, source_map


STAGE_A_SYSTEM_PROMPT = """You are PaperPilot, extracting a structured, evidence-grounded summary of ONE uploaded research paper, as preparation for a later multi-paper comparison.

You are given excerpts labeled SOURCE_1, SOURCE_2, etc., followed by a dimension index showing which source ids were retrieved for each dimension.

Do not use outside knowledge. Do not invent datasets, metrics, numbers, page numbers, sections, or quotations.

Report, using ONLY the provided source ids:
- problem: the problem, objective, assumptions, or scope this paper itself states.
- contributions: what this paper's authors explicitly claim to contribute (one or more items).
- methodology: the core approach, architecture, key components, training/inference strategy, and design assumptions.
- datasets_setup: datasets, dataset size/composition, baselines, evaluation protocol, conditions, and metrics used (one or more items). If something is not stated, omit it — do not invent it.
- results: reported findings and numbers, preserving exact metric names, values, units, dataset names, and conditions (one or more items).
- limitations: ONLY weaknesses THIS PAPER'S OWN AUTHORS present negatively about THEIR OWN work. Never attribute a limitation of an earlier method, a baseline, or the field in general to this paper. Never infer a limitation merely because an experiment, dataset, or metric is absent — a narrow scope is not automatically a weakness. If none, return an empty list.
- future_work: ONLY directions THIS PAPER'S OWN AUTHORS explicitly propose, anywhere in the paper, not only under a heading literally named "Future Work". If none, return an empty list.

Every "source_ids" list must contain only ids that genuinely support that specific item — do not cite an id whose content you are not using, and do not invent an item merely to fill a dimension. Every free-text field must be plain prose with no citation markers inside it — never write "SOURCE_1" or similar inside a text field. Source ids belong only in "source_ids" fields.

Respond ONLY with valid JSON matching this schema:
{
  "problem": {"text": string, "source_ids": [string]},
  "contributions": [{"text": string, "source_ids": [string]}],
  "methodology": {"text": string, "source_ids": [string]},
  "datasets_setup": [{"text": string, "source_ids": [string]}],
  "results": [{"text": string, "source_ids": [string]}],
  "limitations": [{"text": string, "source_ids": [string]}],
  "future_work": [{"text": string, "source_ids": [string]}]
}"""


def _clean_text(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = strip_source_labels(value)
    return cleaned if cleaned.strip() else None


def _sources_for_paper(raw_ids, source_map: dict[str, tuple[str, dict]], owning_paper_id: str) -> list[Source]:
    """Resolve source ids, keeping only those that both exist AND belong to owning_paper_id."""
    if not isinstance(raw_ids, list):
        return []
    sources = []
    for sid in raw_ids:
        if not isinstance(sid, str) or sid not in source_map:
            continue
        paper_id, chunk = source_map[sid]
        if paper_id != owning_paper_id:
            continue  # cross-paper citation leak -- discard, never trust the model here
        sources.append(build_source(sid, chunk))
    return sources


def _sources_any_paper(raw_ids, source_map: dict[str, tuple[str, dict]]) -> list[Source]:
    if not isinstance(raw_ids, list):
        return []
    return [
        build_source(sid, source_map[sid][1])
        for sid in raw_ids
        if isinstance(sid, str) and sid in source_map
    ]


def _make_single(data, source_map, owning_paper_id) -> ComparisonFinding | None:
    if not isinstance(data, dict):
        return None
    text = _clean_text(data.get("text"))
    sources = _sources_for_paper(data.get("source_ids"), source_map, owning_paper_id)
    if text and sources:
        return ComparisonFinding(text=text, sources=sources)
    return None


def _make_list(data, source_map, owning_paper_id) -> list[ComparisonFinding]:
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = _clean_text(entry.get("text"))
        sources = _sources_for_paper(entry.get("source_ids"), source_map, owning_paper_id)
        if text and sources:
            out.append(ComparisonFinding(text=text, sources=sources))
    return out


def _make_synthesis(data, source_map) -> list[SynthesisItem]:
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        claim = _clean_text(entry.get("claim"))
        synthesis_type = entry.get("synthesis_type")
        if not isinstance(synthesis_type, str) or synthesis_type not in SYNTHESIS_TYPES:
            synthesis_type = "synthesis"
        sources = _sources_any_paper(entry.get("source_ids"), source_map)
        if claim and sources:
            out.append(SynthesisItem(claim=claim, synthesis_type=synthesis_type, sources=sources))
    return out


def generate_paper_findings(
    paper: ProcessedPaper,
    llm_client: LLMClient,
    evidence_per_category: int = STAGE_A_EVIDENCE_PER_CATEGORY,
) -> tuple[PaperComparisonData, dict[str, list[RankedChunk]]]:
    """Stage A: one independent LLM call using only this paper's own evidence.

    Returns (findings, evidence) -- the raw evidence is returned too so Stage B can reuse
    it without a second round of retrieval. Any failure here (no evidence, API error,
    malformed response) is captured entirely within this one paper's PaperComparisonData
    and never touches any other paper.
    """
    evidence, unavailable = _gather_paper_evidence(paper, evidence_per_category)
    block, source_map = _build_paper_block(paper, evidence, prefix="")

    if not source_map:
        return (
            PaperComparisonData(
                paper_id=paper.paper_id,
                display_name=paper.display_name,
                generation_error="No reliable evidence could be retrieved from this paper.",
            ),
            evidence,
        )

    response = llm_client.complete_json(STAGE_A_SYSTEM_PROMPT, block, temperature=0.1)
    if not response.ok:
        return (
            PaperComparisonData(
                paper_id=paper.paper_id, display_name=paper.display_name, generation_error=response.error
            ),
            evidence,
        )

    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        data = None
    if not isinstance(data, dict):
        return (
            PaperComparisonData(
                paper_id=paper.paper_id,
                display_name=paper.display_name,
                generation_error="The AI model returned a response in an unexpected format.",
            ),
            evidence,
        )

    pd = PaperComparisonData(paper_id=paper.paper_id, display_name=paper.display_name)
    pd.problem = _make_single(data.get("problem"), source_map, paper.paper_id)
    pd.methodology = _make_single(data.get("methodology"), source_map, paper.paper_id)
    pd.contributions = _make_list(data.get("contributions"), source_map, paper.paper_id)
    pd.datasets_setup = _make_list(data.get("datasets_setup"), source_map, paper.paper_id)
    pd.results = _make_list(data.get("results"), source_map, paper.paper_id)
    pd.limitations = _make_list(data.get("limitations"), source_map, paper.paper_id)
    pd.future_work = _make_list(data.get("future_work"), source_map, paper.paper_id)

    if "limitations" in unavailable:
        pd.limitations = []
    if "future_work" in unavailable:
        pd.future_work = []

    return pd, evidence


def _trim_evidence(evidence: dict[str, list[RankedChunk]], k: int) -> dict[str, list[RankedChunk]]:
    return {dimension: chunks[:k] for dimension, chunks in evidence.items()}


def _summarize_findings(pd: PaperComparisonData) -> str:
    """Compact prose summary of one paper's validated Stage A findings, for Stage B's
    reasoning context only -- Stage B must still cite real excerpts, never this summary."""
    lines = []
    if pd.problem:
        lines.append(f"Problem: {pd.problem.text}")
    if pd.contributions:
        lines.append("Contributions: " + "; ".join(c.text for c in pd.contributions))
    if pd.methodology:
        lines.append(f"Methodology: {pd.methodology.text}")
    if pd.datasets_setup:
        lines.append("Datasets/setup: " + "; ".join(c.text for c in pd.datasets_setup))
    if pd.results:
        lines.append("Results: " + "; ".join(c.text for c in pd.results))
    if pd.limitations:
        lines.append("Limitations: " + "; ".join(c.text for c in pd.limitations))
    if pd.future_work:
        lines.append("Future work: " + "; ".join(c.text for c in pd.future_work))
    return "\n".join(lines) if lines else "(no validated findings for this paper)"


def _build_synthesis_system_prompt(paper_ids: list[str]) -> str:
    id_list = ", ".join(paper_ids)
    return f"""You are PaperPilot, synthesizing a comparison across {len(paper_ids)} already-analyzed research papers ({id_list}).

For context, you are given each paper's already-extracted findings (for your reasoning only — these are not directly citable), followed by a smaller set of raw excerpts labeled with paper-specific source ids such as P1_SOURCE_1, P2_SOURCE_3, etc. — the prefix tells you which paper that excerpt belongs to.

CRITICAL RULE: every synthesis item's "source_ids" must cite only real excerpt ids from the excerpts below, and may span multiple papers' ids since synthesis is explicitly about comparing papers. Do not invent a source id, and do not cite an id whose content you are not using.

Do not use outside knowledge. Do not invent claims about what the broader research field has or has not solved — you have only read these specific papers.

First, assess comparability: set "comparable" to true only if the papers' reported results share a sufficiently similar task, dataset, metric definition, evaluation protocol, and experimental conditions to be fairly ranked against each other. In "comparability_note", explain your reasoning in 1-3 sentences. Papers on different tasks, datasets, metrics, or protocols are common and expected — in that case set "comparable" to false and say so plainly. Do not produce or imply a performance ranking when "comparable" is false.

Then provide "synthesis": a list of evidence-supported cross-paper observations. Each item needs a "synthesis_type" of exactly one of: shared_goal, similarity, difference, complementary, tradeoff, evaluation_difference, unestablished_gap. For "unestablished_gap" items specifically: phrase them as something the SELECTED PAPERS do not establish or address together — never as "the research field lacks X" or "no prior work has solved X", since you have only read these specific papers. Do not generate an item merely to fill the list — an empty list is fine if nothing is well-supported.

Every free-text field must be plain prose with no citation markers inside it — never write "P1_SOURCE_1" or similar inside a text/claim field. Source ids belong only in "source_ids" fields.

Respond ONLY with valid JSON matching this schema:
{{
  "comparable": boolean,
  "comparability_note": string,
  "synthesis": [{{"claim": string, "synthesis_type": string, "source_ids": [string]}}]
}}"""


def generate_synthesis(
    papers_with_data: list[tuple[ProcessedPaper, PaperComparisonData, dict[str, list[RankedChunk]]]],
    llm_client: LLMClient,
) -> tuple[bool, str, list[SynthesisItem], str | None]:
    """Stage B: one call for cross-paper synthesis, reusing Stage A's already-fetched
    evidence (trimmed) rather than re-retrieving. Returns (comparable, note, synthesis, error).

    Only ever called with papers that already succeeded in Stage A. A failure here is
    reported via the returned error string; it must never be used to discard the Stage A
    results the caller already collected.
    """
    if len(papers_with_data) < 2:
        return (
            False,
            "Synthesis requires at least 2 successfully analyzed papers.",
            [],
            "Fewer than 2 papers had successful findings; synthesis was skipped.",
        )

    evidence_per_category = _evidence_budget(len(papers_with_data))
    source_map: dict[str, tuple[str, dict]] = {}
    blocks = []
    for idx, (paper, pd, evidence) in enumerate(papers_with_data, start=1):
        trimmed = _trim_evidence(evidence, evidence_per_category)
        block, local_map = _build_paper_block(paper, trimmed, prefix=f"P{idx}")
        source_map.update(local_map)
        blocks.append(
            f"=== {paper.paper_id} (\"{paper.display_name}\") ===\n"
            f"Already-extracted findings (for reasoning only, not directly citable):\n"
            f"{_summarize_findings(pd)}\n\n{block}"
        )

    context = "\n\n".join(blocks)
    system_prompt = _build_synthesis_system_prompt([p.paper_id for p, _, _ in papers_with_data])
    response = llm_client.complete_json(system_prompt, context, temperature=0.1)

    if not response.ok:
        return False, "", [], response.error

    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        data = None
    if not isinstance(data, dict):
        return False, "", [], "The AI model returned a response in an unexpected format."

    comparable = data.get("comparable")
    comparable = comparable if isinstance(comparable, bool) else False
    note = _clean_text(data.get("comparability_note")) or (
        "Comparability could not be reliably determined from the model's response; "
        "treat any numerical comparison with caution."
    )
    synthesis = _make_synthesis(data.get("synthesis"), source_map)
    return comparable, note, synthesis, None


def generate_comparison(papers: list[ProcessedPaper], llm_client: LLMClient) -> ComparisonResult:
    if len(papers) < MIN_PAPERS:
        return ComparisonResult(ready=False, error=f"At least {MIN_PAPERS} papers are required.")
    if len(papers) > MAX_PAPERS:
        return ComparisonResult(ready=False, error=f"At most {MAX_PAPERS} papers can be compared at once.")
    if any(not p.retrieval_pipeline.retriever.chunks for p in papers):
        empty = [p.display_name for p in papers if not p.retrieval_pipeline.retriever.chunks]
        return ComparisonResult(ready=False, error=f"No usable text was extracted from: {', '.join(empty)}.")

    result = ComparisonResult(ready=False)
    papers_with_data: list[tuple[ProcessedPaper, PaperComparisonData, dict[str, list[RankedChunk]]]] = []

    # Stage A: one independent attempt per paper. No internal retry loop -- a failure here
    # is recorded on that paper alone and does not block or discard any other paper.
    for paper in papers:
        pd, evidence = generate_paper_findings(paper, llm_client)
        result.papers[paper.paper_id] = pd
        if pd.generation_error is None:
            papers_with_data.append((paper, pd, evidence))

    if not papers_with_data:
        result.error = "Findings could not be generated for any of the uploaded papers."
        return result

    # At least one paper has usable findings -- show it, even if others failed or
    # synthesis fails next.
    result.ready = True

    comparable, note, synthesis, synthesis_error = generate_synthesis(papers_with_data, llm_client)
    result.comparable = comparable
    result.comparability_note = note
    result.synthesis = synthesis
    result.synthesis_error = synthesis_error
    return result
