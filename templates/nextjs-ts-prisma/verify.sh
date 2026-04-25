#!/usr/bin/env bash
# Verification script for nextjs-ts-prisma template.
# Stages: lint | typecheck | smoke | test | all (default).
#
# Each stage is intentionally cheap-first. Coder iterates against the fast
# stages (lint, typecheck) while editing, then runs `all` to gate done.

set -eu

stage="${1:-all}"

_install_deps() {
    if [ -z "${_DEPS_INSTALLED:-}" ]; then
        echo "▸ npm install (quiet)…"
        # --no-audit / --no-fund silence noise; --prefer-offline reuses cache.
        npm install --no-audit --no-fund --prefer-offline >/dev/null
        # Generate the Prisma client; needed for typecheck + build.
        echo "▸ prisma generate"
        npx prisma generate
        export _DEPS_INSTALLED=1
    fi
}

_run_lint() {
    echo "▸ next lint"
    npx next lint
}

_run_typecheck() {
    echo "▸ tsc --noEmit"
    npx tsc --noEmit
}

_run_smoke() {
    # `next build` is the de-facto smoke for Next.js — it compiles every
    # route, runs the type-checker again as a side effect, and surfaces
    # any import-time errors. Slow-ish but the most signal you'll get
    # without spinning a real server.
    echo "▸ next build"
    npx next build
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
