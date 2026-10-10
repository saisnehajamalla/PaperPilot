"""PDF text extraction and lightweight structural analysis for PaperPilot."""

import re

import fitz  # PyMuPDF


class PDFValidationError(Exception):
    """A PDF could not be validated for processing (empty, corrupted, unreadable, or
    encrypted). Safe to show str(e) directly to the user -- it never includes a raw
    traceback, only a short description."""


class PDFTooLargeError(PDFValidationError):
    """The PDF is a readable, valid PDF, but exceeds the configured page limit."""

    def __init__(self, page_count: int, max_pages: int):
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(
            f"This PDF contains {page_count} pages, exceeding the configured limit of "
            f"{max_pages} pages. Please upload a shorter PDF."
        )

SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "methods",
    "materials and methods",
    "experiments",
    "experimental setup",
    "experiments and results",
    "model architecture",
    "architecture",
    "training",
    "training data and batching",
    "evaluation",
    "analysis",
    "why self-attention",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
]

_HEADING_RE = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?(" + "|".join(SECTION_HEADINGS) + r")\s*$",
    re.IGNORECASE,
)


def validate_pdf(file_bytes: bytes, max_pages: int, filename: str | None = None) -> "fitz.Document":
    """Opens and validates a PDF *before* any expensive processing (chunking, embedding,
    indexing) begins. Does not extract page text.

    On success, returns the opened fitz.Document so the caller can pass it straight to
    extract_pages_from_doc() without reopening the same bytes a second time. The caller is
    responsible for closing it (extract_pages_from_doc() does this).

    Raises PDFTooLargeError if the page count exceeds max_pages, or PDFValidationError for
    anything else that makes the file unusable (empty, corrupted, unreadable, encrypted).
    Both are safe to display to the user directly.
    """
    label = f" ({filename})" if filename else ""

    if not file_bytes:
        raise PDFValidationError(f"The uploaded file{label} is empty.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFValidationError(
            f"This file{label} could not be read as a PDF. It may be corrupted or in a "
            "different format."
        ) from e

    try:
        if doc.needs_pass:
            raise PDFValidationError(
                f"This PDF{label} is password-protected and cannot be processed."
            )
        page_count = doc.page_count
    except PDFValidationError:
        doc.close()
        raise
    except Exception as e:
        doc.close()
        raise PDFValidationError(
            f"This file{label} could not be read as a PDF. It may be corrupted."
        ) from e

    if page_count == 0:
        doc.close()
        raise PDFValidationError(f"This PDF{label} contains no pages.")

    if page_count > max_pages:
        doc.close()
        raise PDFTooLargeError(page_count, max_pages)

    return doc


def extract_pages_from_doc(doc: "fitz.Document") -> list[dict]:
    """Extract text page by page from an already-opened, already-validated document
    (see validate_pdf()), preserving page numbers (1-indexed). Closes doc when done."""
    pages = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text})
    finally:
        doc.close()
    return pages


def extract_pages(file_bytes: bytes) -> list[dict]:
    """Extract text page by page, preserving page numbers (1-indexed).

    Does not enforce MAX_PAGES or any other validation -- callers that need the configured
    page-limit check should use validate_pdf() + extract_pages_from_doc() instead. Kept
    as-is for callers (tests, scripts) that only need raw extraction of a known-good file.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return extract_pages_from_doc(doc)


_BOILERPLATE_SENTENCE_RE = re.compile(
    r"[^.]*\b(?:permission\s+to\s+reproduce|all\s+rights\s+reserved|licensed\s+under|copyright|©)[^.]*\.?",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"\barXiv:\s*\S+", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\d{1,4}$")


def clean_pages(pages: list[dict], repeat_ratio: float = 0.5) -> list[dict]:
    """Remove obvious boilerplate (copyright, arXiv IDs, repeated headers/footers, page numbers).

    Only removes lines or sentences that match these patterns; other content is kept.
    """
    line_counts: dict[str, int] = {}
    for page in pages:
        seen = {l.strip() for l in page["text"].split("\n") if l.strip()}
        for line in seen:
            if len(line) <= 80:
                line_counts[line] = line_counts.get(line, 0) + 1

    min_repeats = max(3, int(len(pages) * repeat_ratio)) if len(pages) >= 4 else None

    cleaned = []
    for page in pages:
        lines = [l for l in page["text"].split("\n") if l.strip()]
        edge_idx = {0, len(lines) - 1}
        kept_lines = []
        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if idx in edge_idx and _BARE_NUMBER_RE.match(line):
                continue
            if min_repeats and len(line) <= 80 and line_counts.get(line, 0) >= min_repeats:
                continue
            kept_lines.append(raw_line)
        text = "\n".join(kept_lines)
        text = _ARXIV_ID_RE.sub(" ", text)
        text = _BOILERPLATE_SENTENCE_RE.sub(" ", text)
        cleaned.append({**page, "text": text})
    return cleaned


def detect_title_authors(pages: list[dict]) -> dict:
    """Heuristically guess the title and authors from the first page."""
    if not pages:
        return {"title": "Unknown Title", "authors": "Unknown Authors"}

    first_page_text = pages[0]["text"]
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    if not lines:
        return {"title": "Unknown Title", "authors": "Unknown Authors"}

    # Title: the first substantial, non-heading line (skips stray running headers/page numbers).
    title = lines[0]
    title_idx = 0
    for idx, line in enumerate(lines[:5]):
        if len(line) >= 10 and len(line) < 200 and not _HEADING_RE.match(line):
            title = line
            title_idx = idx
            break

    # Authors: look at the line(s) right after the title for name-like patterns
    # (commas, "and", superscript markers, or an email/affiliation nearby).
    authors = "Not detected"
    for line in lines[title_idx + 1 : title_idx + 4]:
        if re.search(r"@|university|department|abstract", line, re.IGNORECASE):
            continue
        if "," in line or " and " in line.lower() or re.match(r"^[A-Z][a-z]+ [A-Z]", line):
            authors = line
            break

    return {"title": title, "authors": authors}


def detect_sections(pages: list[dict]) -> list[dict]:
    """Find section headings and the page on which each starts."""
    sections = []
    for page in pages:
        for raw_line in page["text"].split("\n"):
            line = raw_line.strip()
            if not line or len(line) > 60:
                continue
            match = _HEADING_RE.match(line)
            if match:
                name = match.group(1).title()
                if not sections or sections[-1]["name"] != name:
                    sections.append({"name": name, "page": page["page"]})
    return sections


def _heading_offsets(text: str) -> list[tuple[int, str]]:
    offsets = []
    pos = 0
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        match = _HEADING_RE.match(line) if line and len(line) <= 60 else None
        if match:
            offsets.append((pos, match.group(1).title()))
        pos += len(raw_line) + 1
    return offsets


def build_chunks(
    pages: list[dict],
    paper_name: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[dict]:
    """Split pages into overlapping chunks that never cross a section heading."""
    chunks = []
    current_section = "Front Matter"
    for page in pages:
        text = page["text"]
        if not text.strip():
            continue
        headings = _heading_offsets(text)
        boundaries = [(0, current_section)] + headings + [(len(text), None)]
        for (seg_start, seg_section), (seg_end, _) in zip(boundaries, boundaries[1:]):
            section = seg_section or current_section
            start = seg_start
            while start < seg_end:
                end = min(start + chunk_size, seg_end)
                chunk_text = " ".join(text[start:end].split())
                if chunk_text:
                    chunks.append(
                        {
                            "chunk_id": f"{paper_name}-p{page['page']}-c{len(chunks)}",
                            "paper_name": paper_name,
                            "page": page["page"],
                            "section": section,
                            "text": chunk_text,
                        }
                    )
                if end == seg_end:
                    break
                start = end - overlap
            current_section = section
    return chunks
