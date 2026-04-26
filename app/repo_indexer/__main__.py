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
from .walker import walk_paths, walk_repo

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

    # Lazy-import extractors — keeps `--help` fast and means tests that
    # only touch actions/walker don't drag in tree-sitter at import time.
    from .extractor_python import extract_python_file
    from .extractor_typescript import extract_typescript_file

    py_parser = _parser_for_python()
    ts_parsers = _parsers_for_typescript()

    languages = tuple(args.languages) if args.languages else ("python", "typescript", "javascript")

    # Pre-walk to build the file set — the TS extractor uses it to resolve
    # relative imports (`./bar` -> `app/foo/bar.ts`). Sprint 10g: this used
    # to read every file's bytes via walk_repo; now it's path-only so we
    # don't pay 2× the I/O on TS+JS scans.
    pre_start = time.monotonic()
    repo_files: set[str] = set()
    if any(lang in languages for lang in ("typescript", "javascript")):
        for rel_path, _lang in walk_paths(repo_root, languages=languages):
            repo_files.add(rel_path)
    pre_secs = time.monotonic() - pre_start
    if repo_files:
        print(f"[indexer] pre-walked {len(repo_files)} files in {pre_secs:.1f}s (TS/JS path resolution)")

    aggregate = IndexBatch(repo=repo)
    start = time.monotonic()
    file_count = 0
    PROGRESS_EVERY = 200
    for rel_path, source, language, sha in walk_repo(repo_root, languages=languages):
        try:
            if language == "python":
                fragment = extract_python_file(repo, rel_path, source, sha, py_parser)
            elif language in ("typescript", "javascript"):
                # TSX files need the TSX grammar; .ts/.js use the regular TS grammar.
                use_tsx = rel_path.endswith((".tsx", ".jsx"))
                parser = ts_parsers["tsx"] if use_tsx else ts_parsers["typescript"]
                fragment = extract_typescript_file(
                    repo, rel_path, source, sha, parser,
                    repo_files=repo_files, language=language,
                )
            else:
                continue
        except Exception as e:
            # Don't let one weird file kill the whole scan — log and continue.
            logger.warning("extractor failed on %s: %s", rel_path, e)
            continue
        aggregate.extend(fragment)
        file_count += 1
        if file_count % PROGRESS_EVERY == 0:
            elapsed = time.monotonic() - start
            rate = file_count / elapsed if elapsed > 0 else 0
            print(f"[indexer]   parsed {file_count} files ({rate:.0f}/s)", flush=True)

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
    scan.add_argument(
        "--languages", nargs="+", default=None,
        choices=["python", "typescript", "javascript"],
        help="Languages to extract. Default: all supported.",
    )
    scan.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
