"""Formats reranked chunks into a labeled context block and a SOURCE_n -> chunk mapping.

The mapping is the single source of truth for citations. The LLM never supplies page,
section, or evidence text itself — it only ever refers to a SOURCE_n label.
"""

from retrieval_pipeline import RankedChunk


def build_context(reranked: list[RankedChunk]) -> tuple[str, dict[str, dict]]:
    blocks = []
    mapping: dict[str, dict] = {}
    for i, rc in enumerate(reranked, 1):
        source_id = f"SOURCE_{i}"
        chunk = rc.candidate.chunk
        mapping[source_id] = chunk
        blocks.append(
            f"[{source_id}]\n"
            f"Paper: {chunk['paper_name']}\n"
            f"Page: {chunk['page']}\n"
            f"Section: {chunk['section']}\n\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks), mapping
