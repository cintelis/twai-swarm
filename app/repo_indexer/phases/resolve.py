"""Resolve phase — runs the cross-file resolver over the accumulated batch.

Rewrites Calls/Inheritance edges to point at in-repo Functions/Classes
when possible; emits Symbol nodes only for truly external targets.

Sprint 12b: default path is `scope_resolution.finalize.finalize_batch`
(Tarjan SCC over imports + wildcard expansion + param-type method
resolution). `--legacy-resolver` (PhaseContext.legacy_resolver=True)
falls back to the pre-12b `resolver.resolve_batch`. Sprint 13 deletes
the legacy module + the flag.
"""
from __future__ import annotations

import time

from ..runner import PhaseContext


class ResolvePhase:
    name = "resolve"

    def run(self, ctx: PhaseContext) -> None:
        resolve_start = time.monotonic()
        if ctx.legacy_resolver:
            from .. import resolver
            resolver.resolve_batch(ctx.batch)
        else:
            from ..scope_resolution.finalize import finalize_batch
            finalize_batch(ctx.batch)
        resolve_secs = time.monotonic() - resolve_start
        ctx.progress(f"[indexer] resolved cross-file refs in {resolve_secs:.2f}s")
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
