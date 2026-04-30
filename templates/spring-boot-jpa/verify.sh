#!/usr/bin/env bash
# Verification script for the spring-boot-jpa template.
# Runs INSIDE the customised scaffold/ directory.
# Stages: lint | typecheck | smoke | test | all (default).
# Exit 0 = green; non-zero = broken.

set -euo pipefail

STAGE="${1:-all}"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "verify: required tool not on PATH: $1" >&2
    exit 2
  }
}

stage_lint() {
  require mvn
  echo "[lint] mvn compile"
  mvn --batch-mode -q compile

  if [ -d frontend ]; then
    require node
    require npm
    echo "[lint] frontend tsc -b"
    (cd frontend && [ -d node_modules ] || npm ci --no-audit --no-fund)
    (cd frontend && npx tsc -b)
  fi
}

stage_typecheck() {
  # For this template, typecheck == lint (Java compile + tsc).
  stage_lint
}

stage_smoke() {
  require mvn
  echo "[smoke] HelloControllerTest (boots Spring context, hits /api/hello)"
  mvn --batch-mode -q -Dtest=HelloControllerTest test
}

stage_test() {
  require mvn
  echo "[test] mvn test"
  mvn --batch-mode -q test

  if [ -d frontend ]; then
    require node
    require npm
    echo "[test] frontend vitest"
    (cd frontend && [ -d node_modules ] || npm ci --no-audit --no-fund)
    (cd frontend && npm test --silent)
  fi
}

stage_all() {
  stage_lint
  stage_smoke
  stage_test
}

case "$STAGE" in
  lint) stage_lint ;;
  typecheck) stage_typecheck ;;
  smoke) stage_smoke ;;
  test) stage_test ;;
  all) stage_all ;;
  *)
    echo "verify: unknown stage '$STAGE' (expected: lint|typecheck|smoke|test|all)" >&2
    exit 2
    ;;
esac

echo "verify: $STAGE OK"
