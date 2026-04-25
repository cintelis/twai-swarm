#!/usr/bin/env bash
# Verification script for python-fastapi-postgres template.
# Runs from inside scaffold/ after the Coder has finished customising.
#
# Stages (called as `verify.sh <stage>`; "all" runs the full pipeline,
# which is the actual definition of done):
#   lint      — ruff format + ruff check (style + syntax)
#   typecheck — mypy (catches signature drift)
#   smoke     — `python -c "import app"` (catches import-time errors)
#   test      — pytest (functional correctness)
#   all       — install deps then run lint + typecheck + smoke + test in order
#
# Diagnostics go to stdout; the Coder reads this output to decide what
# to fix on the next iteration. Keep messages compact.

set -eu

stage="${1:-all}"

_install_deps() {
    if [ -z "${_DEPS_INSTALLED:-}" ]; then
        echo "▸ installing scaffold dev dependencies (quiet)…"
        pip install --quiet --upgrade pip >/dev/null
        pip install --quiet -e ".[dev]" >/dev/null
        export _DEPS_INSTALLED=1
    fi
}

_run_lint() {
    echo "▸ ruff format --check ."
    ruff format --check .
    echo "▸ ruff check ."
    ruff check .
}

_run_typecheck() {
    if command -v mypy >/dev/null 2>&1; then
        echo "▸ mypy ."
        mypy .
    else
        echo "▸ mypy not installed — skipping (add to [dev] deps to enable)"
    fi
}

_run_smoke() {
    echo "▸ python -c 'import app'"
    python -c "import app"
}

_run_test() {
    echo "▸ pytest -q"
    pytest -q
}

case "$stage" in
    lint)
        _install_deps
        _run_lint
        ;;
    typecheck)
        _install_deps
        _run_typecheck
        ;;
    smoke)
        _install_deps
        _run_smoke
        ;;
    test)
        _install_deps
        _run_test
        ;;
    all)
        _install_deps
        _run_lint
        _run_typecheck
        _run_smoke
        _run_test
        echo "✓ verification passed"
        ;;
    *)
        echo "error: unknown stage '$stage' (expected: lint, typecheck, smoke, test, all)" >&2
        exit 64
        ;;
esac
