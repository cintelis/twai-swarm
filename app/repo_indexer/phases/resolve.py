"""Resolve phase — runs the cross-file resolver over the accumulated batch.

Rewrites Calls/Inheritance edges to point at in-repo Functions/Classes
when possible; emits Symbol nodes only for truly external targets.
Backed by `scope_resolution.finalize.finalize_batch` — Tarjan SCC over
imports + re-export closures + wildcard expansion + param-type method
resolution + self/super dispatch through inheritance.
"""
from __future__ import annotations

import time

from ..runner import PhaseContext


class ResolvePhase:
    name = "resolve"

    def run(self, ctx: PhaseContext) -> None:
        from ..scope_resolution.finalize import finalize_batch
        resolve_start = time.monotonic()
        finalize_batch(ctx.batch)
        resolve_secs = time.monotonic() - resolve_start
        ctx.progress(f"[indexer] resolved cross-file refs in {resolve_secs:.2f}s")
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
