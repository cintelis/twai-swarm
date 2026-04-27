"""Phase runner — orchestrates the indexer pipeline.

Sprint 11a: extracted from `__main__.cmd_scan` so future phases (SHA
short-circuit, multiprocessing, community detect, embeddings, …) plug in
without churning the CLI. See `repo-indexer-future-state.md` §4.2.

This module deliberately stays small: a typed `PhaseContext`, a `Phase`
protocol, and a no-frills `run_pipeline` driver. Ordering smarts live in
the phase list passed by the caller; the runner just iterates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .actions import IndexBatch, Language, RepoNode


@dataclass
class PhaseContext:
    """Shared state passed through every phase. Not frozen — phases mutate
    `batch`, `repo_files`, and may attach extra fields in later sprints
    (e.g. SHA cache for 11b).
    """
    repo: RepoNode
    repo_root: Path
    languages: tuple[Language, ...]
    batch: IndexBatch
    repo_files: set[str] = field(default_factory=set)
    py_parser: Any = None
    ts_parsers: dict | None = None
    progress: Callable[[str], None] = print
    # Sprint 11b: SHA short-circuit. `driver` is a Neo4j Driver (typed Any
    # so the runner doesn't drag neo4j into the import path); `prior_shas`
    # holds {rel_path: sha} from the last scan. ParsePhase populates it on
    # entry when driver is set, then skips files whose on-disk SHA matches.
    # Tests pre-seed `prior_shas` directly with `driver=None`.
    driver: Any = None
    prior_shas: dict[str, str] = field(default_factory=dict)
    skipped_files: int = 0
    # Sprint 11c: parallel parse. <=1 → sequential (default, byte-identical to
    # pre-11c). >=2 → multiprocessing.Pool, one tree-sitter parser set per
    # worker (parsers aren't picklable). CLI default is cpu_count()//2 but the
    # PhaseContext default stays 1 so existing call sites and unit tests don't
    # silently spawn pools.
    parse_workers: int = 1
    # Sprint 12b: when True, `phases.resolve` falls back to the legacy
    # `resolver.resolve_batch`. Default False uses the new
    # `scope_resolution.finalize.finalize_batch` path (Tarjan SCC over imports
    # + wildcard expansion + param-type method resolution). The flag exists
    # for one sprint only; Sprint 13 deletes the legacy resolver and this
    # field along with it.
    legacy_resolver: bool = False


class Phase(Protocol):
    name: str

    def run(self, ctx: PhaseContext) -> None: ...


def run_pipeline(ctx: PhaseContext, phases: Iterable[Phase]) -> None:
    """Iterate phases in order, calling `run(ctx)` on each. No retries, no
    cancellation, no parallel scheduling — those belong on the phases
    themselves or in a richer driver if/when we need one.
    """
    for phase in phases:
        phase.run(ctx)
