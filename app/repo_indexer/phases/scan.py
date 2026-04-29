"""Scan phase — pre-walks the repo to populate `ctx.repo_files`.

Only runs the pre-walk when TS or JS is in scope; the TS extractor uses
the file set to resolve relative imports (`./bar` → `app/foo/bar.ts`).
Pure Python scans skip this entirely. Sprint 10g made the pre-walk
path-only so we don't pay 2× the I/O on TS+JS scans.
"""
from __future__ import annotations

import time

from .. import walker
from ..runner import PhaseContext


class ScanPhase:
    name = "scan"

    def run(self, ctx: PhaseContext) -> None:
        pre_start = time.monotonic()
        if any(lang in ctx.languages for lang in ("typescript", "javascript")):
            for rel_path, _lang in walker.walk_paths(
                ctx.repo_root,
                languages=ctx.languages,
                additional_skip_dirs=ctx.additional_skip_dirs,
            ):
                ctx.repo_files.add(rel_path)
        pre_secs = time.monotonic() - pre_start
        if ctx.repo_files:
            ctx.progress(
                f"[indexer] pre-walked {len(ctx.repo_files)} files in {pre_secs:.1f}s "
                f"(TS/JS path resolution)"
            )
