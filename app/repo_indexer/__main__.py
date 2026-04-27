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
from .loader import driver_from_env, ensure_constraints, prune_stale, write_batch
from .phases import DEFAULT_PHASES
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

    print(f"[indexer] scanning {repo_root}  ->  Neo4j repo={repo_name!r}")

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
        )
        run_pipeline(ctx, DEFAULT_PHASES)
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
        )
        run_pipeline(ctx, DEFAULT_PHASES)
        write_start = time.monotonic()
        write_batch(driver, aggregate)
        write_secs = time.monotonic() - write_start
        print(f"[indexer] wrote to Neo4j in {write_secs:.1f}s")
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
    scan.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
