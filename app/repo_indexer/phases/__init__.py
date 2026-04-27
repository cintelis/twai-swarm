"""Indexer pipeline phases.

Each phase is a class with a `name` attribute and a `run(ctx)` method
(see `runner.Phase`). `DEFAULT_PHASES` is the ordering used by
`__main__.cmd_scan`; tests / future tooling can build their own tuples.

Opt-in phases (NOT in DEFAULT_PHASES)
-------------------------------------
* `EmbedPhase` (Sprint 14a) — generates per-symbol embeddings via
  `app.embeddings`. Network-bound and per-symbol; can be slow on large
  repos. Wired in by `__main__` only when `--with-embeddings` is set,
  which also flips `PhaseContext.embed_enabled` so the phase doesn't
  run if it sneaks into a custom phase tuple by accident.
"""
from __future__ import annotations

from .community_detect import CommunityDetectPhase
from .embed import EmbedPhase
from .parse import ParsePhase
from .process_extract import ProcessExtractPhase
from .resolve import ResolvePhase
from .scan import ScanPhase

# Sprint 13a/b: CommunityDetectPhase and ProcessExtractPhase both run AFTER
# ResolvePhase. Community detection labels clusters; process extraction then
# walks the resolved CALLS edges and surfaces chains that cross those cluster
# boundaries. Removing either is a one-line edit (Cross-cutting Invariant
# from sprint-11-to-14-plan.md): both phases are reversible.
#
# Sprint 14a's EmbedPhase is intentionally NOT here — opt-in only. See the
# module docstring for rationale.
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
    "EmbedPhase",
    "DEFAULT_PHASES",
]
