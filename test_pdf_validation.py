"""Deterministic tests for PDF page-limit enforcement: the validate_pdf()/
extract_pages_from_doc() helper in pdf_processor.py, plus a structural check that both
upload paths in app.py actually call validate_pdf() before any chunking/embedding.

All PDFs are small synthetic documents created in-memory with PyMuPDF -- nothing is
downloaded. max_pages is passed explicitly to validate_pdf() in every test, so these never
need to touch config.py or the MAX_PAGES environment variable.

AppTest cannot simulate st.file_uploader, so the upload-handler branches in app.py (both
the single-paper and Compare Papers paths) cannot be driven end-to-end here -- see the
structural check below and the final report for what that leaves unverified.

Usage:
    python test_pdf_validation.py
"""

import sys
from pathlib import Path

import fitz

from pdf_processor import (
    PDFTooLargeError,
    PDFValidationError,
    extract_pages_from_doc,
    validate_pdf,
)

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


def make_pdf(num_pages: int) -> bytes:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 50), f"stub text page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def test_valid_pdf_below_limit_accepted():
    print("\n=== Test 1: a valid PDF below the limit is accepted ===")
    data = make_pdf(5)
    doc = validate_pdf(data, max_pages=10)
    check("validate_pdf returned an open document", doc is not None)
    check("page_count matches what was created", doc.page_count == 5)
    pages = extract_pages_from_doc(doc)
    check("extract_pages_from_doc returned 5 pages", len(pages) == 5)
    check("page text was actually extracted", "stub text page 1" in pages[0]["text"])


def test_pdf_exactly_at_limit_accepted():
    print("\n=== Test 2: a PDF exactly at the limit is accepted ===")
    data = make_pdf(10)
    doc = validate_pdf(data, max_pages=10)
    check("a document with page_count == max_pages is accepted, not rejected",
          doc is not None and doc.page_count == 10)
    doc.close()


def test_pdf_exceeding_limit_rejected():
    print("\n=== Test 3: a PDF exceeding the limit is rejected ===")
    data = make_pdf(11)
    try:
        validate_pdf(data, max_pages=10)
        check("PDFTooLargeError was raised for an 11-page PDF with max_pages=10", False)
    except PDFTooLargeError as e:
        check("PDFTooLargeError was raised", True)
        check("e.page_count is the actual page count (11)", e.page_count == 11)
        check("e.max_pages is the configured limit passed in (10)", e.max_pages == 10)
        check("the message names the actual page count and the actual configured limit",
              "11 pages" in str(e) and "limit of 10 pages" in str(e))
    except Exception as e:
        check(f"PDFTooLargeError was raised (got {type(e).__name__} instead)", False)


def test_empty_bytes_rejected():
    print("\n=== Test 4: empty bytes are rejected ===")
    try:
        validate_pdf(b"", max_pages=10)
        check("PDFValidationError was raised for empty bytes", False)
    except PDFValidationError:
        check("PDFValidationError was raised for empty bytes", True)


def test_invalid_pdf_bytes_rejected():
    print("\n=== Test 5: invalid (non-PDF) bytes are rejected ===")
    try:
        validate_pdf(b"this is plainly not a PDF file at all", max_pages=10)
        check("PDFValidationError was raised for non-PDF bytes", False)
    except PDFValidationError:
        check("PDFValidationError was raised for non-PDF bytes", True)


def test_corrupted_pdf_handled_safely():
    print("\n=== Test 6: a corrupted/unreadable PDF is handled safely (no crash, no traceback leak) ===")
    data = make_pdf(3)
    truncated = data[:20]  # severe truncation -- not a recoverable PDF structure
    try:
        validate_pdf(truncated, max_pages=10)
        check("PDFValidationError was raised for a severely truncated PDF", False)
    except PDFValidationError as e:
        check("PDFValidationError was raised for a severely truncated PDF", True)
        check("the error message does not contain a raw Python traceback",
              "Traceback" not in str(e) and "File \"" not in str(e))
    except Exception as e:
        check(f"PDFValidationError was raised (got unhandled {type(e).__name__} instead)", False)


def test_max_pages_easily_overridden_without_touching_config():
    print("\n=== Test: max_pages is a plain parameter, not read from config.py ===")
    data = make_pdf(5)
    try:
        validate_pdf(data, max_pages=3)
        check("a tiny max_pages override (3) correctly rejects a 5-page PDF", False)
    except PDFTooLargeError as e:
        check("a tiny max_pages override (3) correctly rejects a 5-page PDF", True)
        check("the override value (3), not config.MAX_PAGES, was used", e.max_pages == 3)


def test_upload_paths_validate_before_chunking():
    print("\n=== Test 7/8 (structural): both upload paths call validate_pdf() before build_chunks() ===")
    source = Path(__file__).parent.joinpath("app.py").read_text(encoding="utf-8")

    single_start = source.index("# Sidebar: single-paper upload")
    single_end = source.index("# Shared rendering helpers")
    single_block = source[single_start:single_end]
    check("single-paper handler calls validate_pdf()", "validate_pdf(" in single_block)
    check("single-paper handler calls build_chunks()", "build_chunks(" in single_block)
    check("single-paper handler validates before chunking",
          "validate_pdf(" in single_block and "build_chunks(" in single_block
          and single_block.index("validate_pdf(") < single_block.index("build_chunks("))

    compare_start = source.index("with tab_compare:")
    compare_block = source[compare_start:]
    check("compare handler calls validate_pdf()", "validate_pdf(" in compare_block)
    check("compare handler calls build_chunks()", "build_chunks(" in compare_block)
    check("compare handler validates before chunking",
          "validate_pdf(" in compare_block and "build_chunks(" in compare_block
          and compare_block.index("validate_pdf(") < compare_block.index("build_chunks("))
    check("the compare handler's validate_pdf call is inside a per-file try/except "
          "(so one oversized paper cannot abort the whole batch)",
          "except Exception as e:\n                        failed.append(" in compare_block)


def main() -> None:
    test_valid_pdf_below_limit_accepted()
    test_pdf_exactly_at_limit_accepted()
    test_pdf_exceeding_limit_rejected()
    test_empty_bytes_rejected()
    test_invalid_pdf_bytes_rejected()
    test_corrupted_pdf_handled_safely()
    test_max_pages_easily_overridden_without_touching_config()
    test_upload_paths_validate_before_chunking()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("\nAll executed checks passed.")


if __name__ == "__main__":
    main()
