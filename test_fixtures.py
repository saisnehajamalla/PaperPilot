"""Shared fixture-path resolution for validation/test scripts that need full research-paper
PDFs (gan.pdf, bert.pdf, unet.pdf, xgboost.pdf).

These papers are not committed to the repository -- their redistribution permissions have
not been verified. Provide them locally via the PAPERPILOT_FIXTURES_DIR environment variable,
or by placing them in the default `fixtures/` folder at the repository root. Nothing in this
module downloads papers automatically.
"""

import os
from pathlib import Path

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def fixtures_dir() -> Path:
    return Path(os.getenv("PAPERPILOT_FIXTURES_DIR", DEFAULT_FIXTURES_DIR))


def require_fixtures(filenames: list[str]) -> dict[str, Path] | None:
    """Returns {filename: path} if every file exists, else prints an actionable skip
    message and returns None so the caller can skip the full-paper checks cleanly."""
    d = fixtures_dir()
    missing = [f for f in filenames if not (d / f).exists()]
    if missing:
        print(f"SKIPPED: missing fixture PDF(s) {missing} in {d}")
        print(f"  Set PAPERPILOT_FIXTURES_DIR, or place these files in {DEFAULT_FIXTURES_DIR}, "
              "to run the full-paper checks in this script.")
        return None
    return {f: d / f for f in filenames}
