"""Indexer pipeline phases.

Each phase is a class with a `name` attribute and a `run(ctx)` method
(see `runner.Phase`). `DEFAULT_PHASES` is the ordering used by
`__main__.cmd_scan`; tests / future tooling can build their own tuples.
"""
from __future__ import annotations

from .community_detect import CommunityDetectPhase
from .parse import ParsePhase
from .process_extract import ProcessExtractPhase
from .resolve import ResolvePhase
from .scan import ScanPhase

# Sprint 13a/b: CommunityDetectPhase and ProcessExtractPhase both run AFTER
# ResolvePhase. Community detection labels clusters; process extraction then
# walks the resolved CALLS edges and surfaces chains that cross those cluster
# boundaries. Removing either is a one-line edit (Cross-cutting Invariant
# from sprint-11-to-14-plan.md): both phases are reversible.
DEFAULT_PHASES = (
    ScanPhase(),
    ParsePhase(),
    ResolvePhase(),
    CommunityDetectPhase(),
    ProcessExtractPhase(),
)

__all__ = [
    "ScanPhase",
    "ParsePhase",
    "ResolvePhase",
    "CommunityDetectPhase",
    "ProcessExtractPhase",
    "DEFAULT_PHASES",
]
