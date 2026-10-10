"""Deterministic test runner for PaperPilot.

Runs every deterministic test module (no LLM API key, no full-paper PDFs required, no live
provider requests) as a separate subprocess and aggregates the results into one verdict.

Each module already prints its own PASS/FAIL/SKIPPED lines and returns a non-zero exit code
if any of its own checks failed (see each test_*.py's check()/FAILURES/sys.exit(1) pattern).
This runner does not reinterpret that output -- it forwards it live and exits non-zero if
any module's process exited non-zero, so a failure here can never be masked.

pytest is not used here: these modules use a plain assertion-tracking helper rather than
`assert`, so running them under pytest as-is would report every test function as passed
regardless of internal check failures (pytest only fails a test on a raised exception, and
these never raise). Converting them would mean restructuring all 5 files; this runner gives
a correct, zero-dependency non-zero-on-failure entry point without that risk.

Usage:
    python run_tests.py
"""

import subprocess
import sys

DETERMINISTIC_TESTS = [
    "test_pdf_validation.py",
    "test_llm_error_translation.py",
    "test_compare_ui_state.py",
    "test_overview_ui_state.py",
    "test_comparison_reliability.py",
]


def main() -> int:
    results: list[tuple[str, int]] = []
    for test_file in DETERMINISTIC_TESTS:
        print(f"\n{'=' * 70}\nRunning {test_file}\n{'=' * 70}")
        proc = subprocess.run([sys.executable, test_file])
        results.append((test_file, proc.returncode))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for name, code in results:
        print(f"  {'FAIL' if code != 0 else 'PASS'}  {name}")

    failed = [name for name, code in results if code != 0]
    if failed:
        print(f"\n{len(failed)} of {len(results)} test module(s) FAILED: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(results)} deterministic test modules passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
