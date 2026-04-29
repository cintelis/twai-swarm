"""CLI: `python -m app.repo_indexer scan <repo_path> [--name X] [--commit-sha SHA]`.

Run from inside the worker container (the Coder will eventually call this
via a Temporal activity). For local dev, ensure NEO4J_URL + NEO4J_PASSWORD
are exported.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .actions import IndexBatch, RepoNode
from .loader import (
    driver_from_env,
    ensure_constraints,
    fetch_unembedded_symbols,
    prune_stale,
    write_batch,
    write_embeddings_only,
)
from .package_roots import detect_package_roots
from .phases import DEFAULT_PHASES, EmbedPhase
from .runner import PhaseContext, run_pipeline

logger = logging.getLogger("repo_indexer")


def _parser_for_python():
    """Build a tree-sitter Parser for Python. Lazy-imported so the module
    loads fine even when tree-sitter isn't installed (e.g. in tests that
    only touch actions / walker)."""
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    return Parser(Language(tspython.language()))


def _parsers_for_typescript():
    """Build separate parsers for .ts/.js (typescript grammar) and .tsx/.jsx
    (TSX grammar). They share most node types but TSX is needed for files
    using JSX syntax — using the wrong grammar produces parse errors on
    `<Foo />` literals."""
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser
    return {
        "typescript": Parser(Language(tsts.language_typescript())),
        "tsx":        Parser(Language(tsts.language_tsx())),
    }


def cmd_scan(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_path).resolve()
    if not repo_root.is_dir():
        print(f"error: {repo_root} is not a directory", file=sys.stderr)
        return 1

    repo_name = args.name or repo_root.name
    repo = RepoNode(
        name=repo_name,
        url=args.url or "",
        commit_sha=args.commit_sha or "",
        tenant_id=args.tenant_id,
    )

    # --embed-only: backfill embeddings on an already-scanned repo without
    # re-running the file pipeline. Bypasses the per-file SHA short-circuit
    # that would otherwise skip every file when the commit-sha matches.
    if getattr(args, "embed_only", False):
        return cmd_embed_only(repo, repo_name)

    print(f"[indexer] scanning {repo_root}  ->  Neo4j repo={repo_name!r}")

    additional_skip_dirs = frozenset(args.exclude_dirs or ())
    if additional_skip_dirs:
        print(f"[indexer] excluding dirs: {', '.join(sorted(additional_skip_dirs))}")

    package_roots = tuple(detect_package_roots(repo_root))
    # Filter package roots that fall under excluded dirs — otherwise we'd
    # detect a pyproject in templates/ but never actually scan files there.
    if additional_skip_dirs:
        package_roots = tuple(
            r for r in package_roots
            if not any(part in additional_skip_dirs for part in r.fs_root.split("/"))
        )
    if package_roots:
        summary = ", ".join(r.fs_root or "<repo root>" for r in package_roots[:5])
        if len(package_roots) > 5:
            summary += f", +{len(package_roots) - 5} more"
        print(f"[indexer] detected {len(package_roots)} Python package root(s): {summary}")
    else:
        print("[indexer] no Python package roots detected — using repo-relative qns")

    py_parser = _parser_for_python()
    ts_parsers = _parsers_for_typescript()

    languages = tuple(args.languages) if args.languages else ("python", "typescript", "javascript")

    # Sprint 11c: resolve worker count. Default cpu_count()//2 (forward-looking
    # for the 13K-file case; small repos pay spawn overhead). `--parse-workers
    # 1` forces the sequential path.
    parse_workers = args.parse_workers
    if parse_workers is None:
        parse_workers = max(1, (os.cpu_count() or 1) // 2)

    aggregate = IndexBatch(repo=repo)

    # Sprint 14a — opt-in embeddings. Compose a new phase tuple rather than
    # mutating DEFAULT_PHASES (so callers that import DEFAULT_PHASES directly
    # see the unchanged default ordering). EmbedPhase runs LAST: it depends
    # on resolved qualified names but doesn't feed any other phase.
    embed_enabled = bool(getattr(args, "with_embeddings", False))
    phases = DEFAULT_PHASES + (EmbedPhase(),) if embed_enabled else DEFAULT_PHASES

    if args.dry_run:
        ctx = PhaseContext(
            repo=repo,
            repo_root=repo_root,
            languages=languages,
            batch=aggregate,
            py_parser=py_parser,
            ts_parsers=ts_parsers,
            driver=None,
            parse_workers=parse_workers,
            embed_enabled=embed_enabled,
            package_roots=package_roots,
            additional_skip_dirs=additional_skip_dirs,
            extract_routes=bool(getattr(args, "with_routes", False)),
        )
        run_pipeline(ctx, phases)
        print("[indexer] --dry-run: skipping Neo4j write")
        return 0

    with driver_from_env() as driver:
        # ensure_constraints MUST run before fetch_file_shas — guarantees
        # the Repo / File constraints exist on a fresh database.
        ensure_constraints(driver)
        deleted = prune_stale(driver, repo_name, repo.commit_sha)
        if deleted:
            print(f"[indexer] pruned {deleted} stale nodes from previous commit")
        ctx = PhaseContext(
            repo=repo,
            repo_root=repo_root,
            languages=languages,
            batch=aggregate,
            py_parser=py_parser,
            ts_parsers=ts_parsers,
            driver=driver,
            parse_workers=parse_workers,
            embed_enabled=embed_enabled,
            package_roots=package_roots,
            additional_skip_dirs=additional_skip_dirs,
            extract_routes=bool(getattr(args, "with_routes", False)),
        )
        run_pipeline(ctx, phases)
        write_start = time.monotonic()
        write_batch(driver, aggregate)
        write_secs = time.monotonic() - write_start
        print(f"[indexer] wrote to Neo4j in {write_secs:.1f}s")
    return 0


def cmd_embed_only(repo: RepoNode, repo_name: str) -> int:
    """Backfill embeddings on an already-scanned repo.

    Workflow:
    1. Open Neo4j driver.
    2. Query Function / Class nodes that lack `embedding`.
    3. Reconstruct FunctionNode / ClassNode dataclasses.
    4. Run EmbedPhase only — same code path as a full scan, but the batch
       has no Files / Modules / etc. to write.
    5. write_embeddings_only persists the new vectors. The Repo node's
       commit_sha is left untouched.
    """
    print(f"[indexer] embed-only: targeting Neo4j repo={repo_name!r}")
    aggregate = IndexBatch(repo=repo)
    with driver_from_env() as driver:
        ensure_constraints(driver)
        fns, classes = fetch_unembedded_symbols(driver, repo_name, repo.tenant_id)
        if not fns and not classes:
            print("[indexer] embed-only: 0 symbols need embedding (repo already covered)")
            return 0
        aggregate.functions.extend(fns)
        aggregate.classes.extend(classes)
        print(f"[indexer] embed-only: {len(fns)} functions + {len(classes)} classes to embed")

        ctx = PhaseContext(
            repo=repo,
            repo_root=Path("."),  # unused — no file pipeline
            languages=(),
            batch=aggregate,
            py_parser=None,
            ts_parsers={},
            driver=driver,
            parse_workers=1,
            embed_enabled=True,
        )
        EmbedPhase().run(ctx)

        write_start = time.monotonic()
        write_embeddings_only(driver, aggregate)
        print(f"[indexer] wrote {len(aggregate.embeddings)} embeddings to Neo4j in "
              f"{time.monotonic() - write_start:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(prog="python -m app.repo_indexer")
    sub = p.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Index a local source tree into Neo4j")
    scan.add_argument("repo_path", help="Path to the repo's root directory")
    scan.add_argument("--name", help="Repo name override (default: dir name)")
    scan.add_argument("--url", default="", help="Canonical https URL of the repo")
    scan.add_argument("--commit-sha", default="", help="Git commit SHA being scanned")
    scan.add_argument("--tenant-id", default="default", help="Tenant scope (multi-tenant forward-compat)")
    scan.add_argument("--dry-run", action="store_true", help="Parse + report counts without writing to Neo4j")
    scan.add_argument(
        "--languages", nargs="+", default=None,
        choices=["python", "typescript", "javascript"],
        help="Languages to extract. Default: all supported.",
    )
    scan.add_argument(
        "--parse-workers", type=int, default=None,
        help="Parse files in N worker processes. Default: cpu_count()//2 "
             "(or 1 if cpu_count() is 1). Set to 1 to force sequential.",
    )
    scan.add_argument(
        "--with-embeddings", action="store_true", default=False,
        help="Embed every Function + Class symbol via app/embeddings.py and "
             "write vectors to Neo4j. Skipped by default to keep scans fast.",
    )
    scan.add_argument(
        "--embed-only", action="store_true", default=False,
        help="Skip the file pipeline entirely; query Function/Class nodes "
             "in Neo4j that lack `embedding` and embed them in place. "
             "Useful for adding embeddings to an already-scanned repo "
             "without forcing a full re-scan via prune_stale.",
    )
    scan.add_argument(
        "--exclude-dirs", nargs="+", default=None,
        help="Extra directory NAMES (not paths) to skip during the walk. "
             "Augments the built-in denylist (.venv, node_modules, etc.). "
             "Use for self-scans where bundled scaffolds (e.g. `templates`) "
             "shouldn't be indexed alongside the agent's own code.",
    )
    scan.add_argument(
        "--with-routes", action="store_true", default=False,
        help="Sprint 15a — extract HTTP route definitions (FastAPI, Flask) "
             "into Route nodes with HANDLED_BY edges to the handler "
             "Function. Disabled by default to keep scans fast when "
             "the caller doesn't need framework-pattern data.",
    )
    scan.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
