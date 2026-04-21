#!/usr/bin/env bash
# Verification script for python-fastapi-postgres template.
# Runs from inside scaffold/ after the Coder has finished customising.
# Exit 0 = scaffold is valid, non-zero = something is broken.
#
# Diagnostics go to stdout; the Coder reads this output to decide what
# to fix on the next iteration. Keep messages compact.

set -eu

echo "▸ installing scaffold dev dependencies (quiet)…"
pip install --quiet --upgrade pip >/dev/null
pip install --quiet -e ".[dev]" >/dev/null

echo "▸ ruff check ."
ruff check .

echo "▸ pytest -q"
pytest -q

echo "✓ verification passed"
