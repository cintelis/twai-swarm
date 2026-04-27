"""Indexer pipeline phases.

Each phase is a class with a `name` attribute and a `run(ctx)` method
(see `runner.Phase`). `DEFAULT_PHASES` is the ordering used by
`__main__.cmd_scan`; tests / future tooling can build their own tuples.
"""
from __future__ import annotations

from .parse import ParsePhase
from .resolve import ResolvePhase
from .scan import ScanPhase

DEFAULT_PHASES = (ScanPhase(), ParsePhase(), ResolvePhase())

__all__ = [
    "ScanPhase",
    "ParsePhase",
    "ResolvePhase",
    "DEFAULT_PHASES",
]
