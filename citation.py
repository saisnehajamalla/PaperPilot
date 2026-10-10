"""Shared citation representation, used by both the chat RAG pipeline and the paper
overview pipeline so there is exactly one way evidence gets attached to an answer."""

import re
from dataclasses import dataclass


@dataclass
class Source:
    source_id: str
    paper_name: str
    page: int
    section: str
    evidence: str
    chunk_id: str


def build_source(source_id: str, chunk: dict) -> Source:
    return Source(
        source_id=source_id,
        paper_name=chunk["paper_name"],
        page=chunk["page"],
        section=chunk["section"],
        evidence=chunk["text"],
        chunk_id=chunk["chunk_id"],
    )


def strip_source_labels(text: str) -> str:
    """Remove any stray SOURCE_n tokens the model left in a text field, as a safety net
    independent of prompt compliance — the user must never see raw source labels."""
    text = re.sub(r"[\(\[][^()\[\]]{0,80}SOURCE_\d+[^()\[\]]{0,80}[\)\]]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"SOURCE_\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()
