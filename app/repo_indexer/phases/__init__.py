"""Indexer pipeline phases.

Each phase is a class with a `name` attribute and a `run(ctx)` method
(see `runner.Phase`). `DEFAULT_PHASES` is the ordering used by
`__main__.cmd_scan`; tests / future tooling can build their own tuples.
"""
from __future__ import annotations

from .community_detect import CommunityDetectPhase
from .parse import ParsePhase
from .resolve import ResolvePhase
from .scan import ScanPhase

# Sprint 13a: CommunityDetectPhase runs AFTER ResolvePhase — it needs the
# resolved CALLS / IMPORTS / INHERITS_FROM edges to find clusters worth
# labelling. Removing community detection is a one-line edit (Cross-cutting
# Invariant from sprint-11-to-14-plan.md).
DEFAULT_PHASES = (ScanPhase(), ParsePhase(), ResolvePhase(), CommunityDetectPhase())

__all__ = [
    "ScanPhase",
    "ParsePhase",
    "ResolvePhase",
    "CommunityDetectPhase",
    "DEFAULT_PHASES",
]
