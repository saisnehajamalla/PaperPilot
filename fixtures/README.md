# Fixtures

Some validation/test scripts (`test_comparison_reliability.py`, `validate_overview.py`,
`validate_unseen_papers.py`) use real research papers as retrieval evidence. These PDFs are
not committed here because their redistribution permissions have not been verified.

To run the full-paper checks in those scripts, place these files in this directory (or point
`PAPERPILOT_FIXTURES_DIR` at a directory containing them):

- `gan.pdf`
- `bert.pdf`
- `unet.pdf`
- `xgboost.pdf`

If they are missing, the affected scripts print a skip message naming what's missing and
continue (or exit 0) rather than failing or downloading anything automatically.
