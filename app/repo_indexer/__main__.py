"""CLI: `python -m app.repo_indexer scan <repo_path> [--name X] [--commit-sha SHA]`.

Run from inside the worker container (the Coder will eventually call this
via a Temporal activity). For local dev, ensure NEO4J_URL + NEO4J_PASSWORD
are exported.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .actions import IndexBatch, RepoNode
from .loader import driver_from_env, ensure_constraints, prune_stale, write_batch
from .resolver import resolve_batch
from .walker import walk_repo

logger = logging.getLogger("repo_indexer")


def _parser_for_python():
    """Build a tree-sitter Parser for Python. Lazy-imported so the module
    loads fine even when tree-sitter isn't installed (e.g. in tests that
    only touch actions / walker)."""
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    py_language = Language(tspython.language())
    parser = Parser(py_language)
    return parser


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
    parser = _parser_for_python()

    # Lazy-import the extractor too — keeps the CLI startup fast for
    # `--help` / arg-validation paths.
    from .extractor_python import extract_python_file

    aggregate = IndexBatch(repo=repo)
    start = time.monotonic()
    file_count = 0
    for rel_path, source, language, sha in walk_repo(repo_root, languages=("python",)):
        if language != "python":
            continue
        try:
            fragment = extract_python_file(repo, rel_path, source, sha, parser)
        except Exception as e:
            # Don't let one weird file kill the whole scan — log and continue.
            logger.warning("extractor failed on %s: %s", rel_path, e)
            continue
        aggregate.extend(fragment)
        file_count += 1

    walk_secs = time.monotonic() - start
    print(f"[indexer] parsed {file_count} files in {walk_secs:.1f}s")
    for k, v in aggregate.counts().items():
        print(f"  {k:20s} {v}")

    # Cross-file resolution pass — rewrites Calls/Inheritance to point at
    # in-repo Functions/Classes when possible, emits Symbol nodes only
    # for truly external targets.
    resolve_start = time.monotonic()
    resolve_batch(aggregate)
    resolve_secs = time.monotonic() - resolve_start
    print(f"[indexer] resolved cross-file refs in {resolve_secs:.2f}s")
    for k, v in aggregate.counts().items():
        print(f"  {k:20s} {v}")

    if args.dry_run:
        print("[indexer] --dry-run: skipping Neo4j write")
        return 0

    write_start = time.monotonic()
    with driver_from_env() as driver:
        ensure_constraints(driver)
        deleted = prune_stale(driver, repo_name, repo.commit_sha)
        if deleted:
            print(f"[indexer] pruned {deleted} stale nodes from previous commit")
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
    scan.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
