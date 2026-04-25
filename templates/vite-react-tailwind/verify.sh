#!/usr/bin/env bash
# Verification script for vite-react-tailwind template.
# Stages: lint | typecheck | smoke | test | all (default).

set -eu

stage="${1:-all}"

_install_deps() {
    if [ -z "${_DEPS_INSTALLED:-}" ]; then
        echo "▸ npm install (quiet)…"
        npm install --no-audit --no-fund --prefer-offline >/dev/null
        export _DEPS_INSTALLED=1
    fi
}

_run_lint() {
    echo "▸ eslint ."
    npx eslint .
}

_run_typecheck() {
    echo "▸ tsc --noEmit"
    npx tsc --noEmit
}

_run_smoke() {
    # `vite build` runs the production bundler — fastest reliable smoke
    # for an SPA. Catches missing imports, syntax errors, asset issues.
    echo "▸ vite build"
    npx vite build
}

_run_test() {
    echo "▸ vitest run"
    npx vitest run
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
