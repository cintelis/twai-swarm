#!/usr/bin/env bash
# Sync a template from TotallyWildAi/templates into this repo's templates/<name>/.
#
# Usage:
#   bash scripts/sync_template.sh                         # syncs all templates that exist in the upstream
#   bash scripts/sync_template.sh spring-boot-jpa         # syncs one template
#   bash scripts/sync_template.sh spring-boot-jpa --check # exit 1 if the local copy has drifted from upstream
#
# Env overrides:
#   TEMPLATES_REPO  default https://github.com/TotallyWildAi/templates.git
#   TEMPLATES_REF   default main
#
# Single source of truth lives at the upstream repo. This script ensures the
# vendored copy under templates/<name>/ matches it exactly.

set -euo pipefail

TEMPLATES_REPO="${TEMPLATES_REPO:-https://github.com/TotallyWildAi/templates.git}"
TEMPLATES_REF="${TEMPLATES_REF:-main}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CHECK_ONLY=0
TEMPLATE_NAME=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) TEMPLATE_NAME="$arg" ;;
  esac
done

echo "Cloning $TEMPLATES_REPO @ $TEMPLATES_REF ..."
git clone --depth 1 --branch "$TEMPLATES_REF" "$TEMPLATES_REPO" "$TMP_DIR/templates" >/dev/null 2>&1

if [ -n "$TEMPLATE_NAME" ]; then
  TARGETS=("$TEMPLATE_NAME")
else
  # Sync every directory the upstream provides that also exists locally.
  TARGETS=()
  for d in "$TMP_DIR/templates"/*/; do
    name="$(basename "$d")"
    if [ -d "$REPO_ROOT/templates/$name" ] || [ "$CHECK_ONLY" -eq 0 ]; then
      TARGETS+=("$name")
    fi
  done
fi

drift=0
for name in "${TARGETS[@]}"; do
  src="$TMP_DIR/templates/$name"
  dst="$REPO_ROOT/templates/$name"

  if [ ! -d "$src" ]; then
    echo "skip: '$name' not present in upstream" >&2
    continue
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    if [ ! -d "$dst" ]; then
      echo "drift: '$name' exists upstream but not locally"
      drift=1
      continue
    fi
    if ! diff -r --brief "$src" "$dst" >/dev/null 2>&1; then
      echo "drift: '$name' differs from upstream"
      diff -r --brief "$src" "$dst" || true
      drift=1
    else
      echo "ok:    '$name' matches upstream"
    fi
  else
    echo "sync:  '$name'"
    rm -rf "$dst"
    mkdir -p "$dst"
    cp -R "$src/." "$dst/"
    # verify.sh comes from a Linux source repo; ensure it's executable locally too.
    if [ -f "$dst/verify.sh" ]; then chmod +x "$dst/verify.sh"; fi
  fi
done

if [ "$CHECK_ONLY" -eq 1 ] && [ "$drift" -ne 0 ]; then
  echo
  echo "Drift detected. Run 'bash scripts/sync_template.sh' to update vendored copies." >&2
  exit 1
fi

echo "Done."
