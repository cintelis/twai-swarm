"""Resolve phase — runs the cross-file resolver over the accumulated batch.

Rewrites Calls/Inheritance edges to point at in-repo Functions/Classes
when possible; emits Symbol nodes only for truly external targets.
Sprint 12 will replace the resolver internals with a finalize-algorithm
port; this phase wrapper stays.
"""
from __future__ import annotations

import time

from .. import resolver
from ..runner import PhaseContext


class ResolvePhase:
    name = "resolve"

    def run(self, ctx: PhaseContext) -> None:
        resolve_start = time.monotonic()
        resolver.resolve_batch(ctx.batch)
        resolve_secs = time.monotonic() - resolve_start
        ctx.progress(f"[indexer] resolved cross-file refs in {resolve_secs:.2f}s")
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
