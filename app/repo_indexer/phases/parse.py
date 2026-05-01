"""Parse phase — walks the repo, dispatches each file to the right
extractor, and accumulates fragments into `ctx.batch`.

Lazy-imports the extractors so tests that only touch actions/walker
don't drag tree-sitter into the import path. Per-file failures are
logged and skipped — one weird file shouldn't kill the whole scan.

Sprint 11b — per-file SHA short-circuit. On entry, when `ctx.driver` is
set and `ctx.prior_shas` is empty, we fetch the previously-stored SHAs
for this repo from Neo4j. Files whose on-disk SHA already matches the
stored value are skipped: they contribute nothing to `ctx.batch`, so
the loader's MERGE leaves the existing Neo4j data intact.

Caveat: 11b's short-circuit means a file's nodes/edges are not
re-emitted when its SHA matches. This means stale edges from PRIOR
scans of the file persist in Neo4j until commit_sha changes (which
triggers `prune_stale`). This is a pre-existing limitation (the
current loader's MERGE never deletes), not introduced by 11b. Full
per-file diffing is deferred.

Sprint 11c — opt-in multiprocessing. Sequential by default
(`ctx.parse_workers <= 1`). When >=2, files are dispatched to a
`multiprocessing.Pool` whose workers each build their own tree-sitter
parsers (parsers aren't picklable, and `Parser` is not thread-safe).
Pool overhead means parallel mode is a regression on small repos; tune
via `--parse-workers`. The SHA short-circuit still runs on the main
process before submission, so unchanged files never cross the pickle
boundary.
"""
from __future__ import annotations

import logging
import time

from .. import walker
from ..runner import PhaseContext

logger = logging.getLogger("repo_indexer")

PROGRESS_EVERY = 200

# Per-worker globals populated by `_pool_init`. Tree-sitter Parser objects
# are NOT picklable, so each worker process must build its own from scratch.
_PY_PARSER = None
_TS_PARSERS: dict | None = None


_CPP_PARSER = None
_JAVA_PARSER = None


def _pool_init() -> None:
    """Worker-process initializer. Builds a Python parser + a TS/TSX parser
    dict + a C++ parser + a Java parser and stashes them in module
    globals. Runs once per worker, not once per task — that's the point
    of using `initializer` instead of building parsers inside
    `_pool_extract`.

    The cpp / java parser imports are best-effort: if the grammar isn't
    installed, we log nothing here (parsers run lazily on first use) and
    the corresponding branch in `_pool_extract` skips with an error.
    """
    global _PY_PARSER, _TS_PARSERS, _CPP_PARSER, _JAVA_PARSER
    import tree_sitter_python as tspython
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser

    _PY_PARSER = Parser(Language(tspython.language()))
    _TS_PARSERS = {
        "typescript": Parser(Language(tsts.language_typescript())),
        "tsx":        Parser(Language(tsts.language_tsx())),
    }
    try:
        import tree_sitter_cpp as tscpp
        _CPP_PARSER = Parser(Language(tscpp.language()))
    except Exception:  # noqa: BLE001
        _CPP_PARSER = None
    try:
        import tree_sitter_java as tsjava
        _JAVA_PARSER = Parser(Language(tsjava.language()))
    except Exception:  # noqa: BLE001
        _JAVA_PARSER = None


def _pool_extract(args: tuple) -> tuple:
    """Worker-side per-file extraction. Args:
        (rel_path, source, language, sha, repo, repo_files, package_roots,
         extract_routes, extract_mcp_tools, extract_orm)

    Returns `(rel_path, fragment_or_None, error_str_or_None)`. Catches all
    exceptions so one weird file never crashes the pool. The `args` tuple
    is everything we need to be picklable — parsers come from module
    globals populated by `_pool_init`.
    """
    (rel_path, source, language, sha, repo, repo_files, package_roots,
     extract_routes, extract_mcp_tools, extract_orm) = args
    try:
        # Lazy-import inside the worker — keeps the parent process from
        # paying the import cost when sequential mode is used.
        if language == "python":
            from ..extractor_python import extract_python_file
            fragment = extract_python_file(
                repo, rel_path, source, sha, _PY_PARSER,
                package_roots=package_roots,
                extract_routes=extract_routes,
                extract_mcp_tools=extract_mcp_tools,
                extract_orm=extract_orm,
            )
        elif language in ("typescript", "javascript"):
            from ..extractor_typescript import extract_typescript_file
            use_tsx = rel_path.endswith((".tsx", ".jsx"))
            assert _TS_PARSERS is not None  # set by _pool_init
            parser = _TS_PARSERS["tsx"] if use_tsx else _TS_PARSERS["typescript"]
            fragment = extract_typescript_file(
                repo, rel_path, source, sha, parser,
                repo_files=repo_files, language=language,
                extract_routes=extract_routes,
            )
        elif language == "cpp":
            if _CPP_PARSER is None:
                return (rel_path, None, "tree_sitter_cpp not installed in worker")
            from ..extractor_cpp import extract_cpp_file
            fragment = extract_cpp_file(
                repo, rel_path, source, sha, _CPP_PARSER,
                repo_files=repo_files,
            )
        elif language == "java":
            if _JAVA_PARSER is None:
                return (rel_path, None, "tree_sitter_java not installed in worker")
            from ..extractor_java import extract_java_file
            fragment = extract_java_file(
                repo, rel_path, source, sha, _JAVA_PARSER,
                repo_files=repo_files,
                extract_routes=extract_routes,
            )
        else:
            return (rel_path, None, None)
    except Exception as e:  # noqa: BLE001 — pool boundary catch-all
        return (rel_path, None, f"{type(e).__name__}: {e}")
    return (rel_path, fragment, None)


class ParsePhase:
    name = "parse"

    def run(self, ctx: PhaseContext) -> None:
        # Lazy-import — keeps tree-sitter out of the import path for
        # tests that only need actions/walker.
        from ..extractor_python import extract_python_file
        from ..extractor_typescript import extract_typescript_file

        # Sprint 11b: prefetch prior file SHAs once per scan when a driver
        # is wired in and the test harness hasn't pre-seeded the cache.
        # `driver is None` (dry-run / unit tests) leaves the cache empty
        # and the short-circuit naturally disabled.
        # Sprint 17 post-deploy: `ctx.force_reindex` skips the prefetch so
        # files cached by a prior extractor version get re-extracted even
        # when their on-disk SHA is unchanged.
        if ctx.driver is not None and not ctx.prior_shas and not ctx.force_reindex:
            from ..loader import fetch_file_shas
            ctx.prior_shas = fetch_file_shas(
                ctx.driver, ctx.repo.name, ctx.repo.tenant_id,
            )

        # Generator that walks the repo, applies the SHA short-circuit, and
        # yields work items for the parse step. Bumping `ctx.skipped_files`
        # here keeps the skip count accurate whether we run sequentially
        # or hand items to a pool.
        def _work_items():
            for rel_path, source, language, sha in walker.walk_repo(
                ctx.repo_root,
                languages=ctx.languages,
                additional_skip_dirs=ctx.additional_skip_dirs,
            ):
                prior = ctx.prior_shas.get(rel_path)
                if prior == sha:
                    ctx.skipped_files += 1
                    continue
                yield (rel_path, source, language, sha)

        start = time.monotonic()
        file_count = 0

        if ctx.parse_workers <= 1:
            # Sequential path — byte-identical to pre-11c behavior.
            for rel_path, source, language, sha in _work_items():
                try:
                    if language == "python":
                        fragment = extract_python_file(
                            ctx.repo, rel_path, source, sha, ctx.py_parser,
                            package_roots=ctx.package_roots,
                            extract_routes=ctx.extract_routes,
                            extract_mcp_tools=ctx.extract_mcp_tools,
                            extract_orm=ctx.extract_orm,
                        )
                    elif language in ("typescript", "javascript"):
                        # TSX files need the TSX grammar; .ts/.js use the regular TS grammar.
                        use_tsx = rel_path.endswith((".tsx", ".jsx"))
                        parser = ctx.ts_parsers["tsx"] if use_tsx else ctx.ts_parsers["typescript"]
                        fragment = extract_typescript_file(
                            ctx.repo, rel_path, source, sha, parser,
                            repo_files=ctx.repo_files, language=language,
                            extract_routes=ctx.extract_routes,
                        )
                    elif language == "cpp":
                        if ctx.cpp_parser is None:
                            logger.warning("cpp parser unavailable; skipping %s", rel_path)
                            continue
                        from ..extractor_cpp import extract_cpp_file
                        fragment = extract_cpp_file(
                            ctx.repo, rel_path, source, sha, ctx.cpp_parser,
                            repo_files=ctx.repo_files,
                        )
                    elif language == "java":
                        if ctx.java_parser is None:
                            logger.warning("java parser unavailable; skipping %s", rel_path)
                            continue
                        from ..extractor_java import extract_java_file
                        fragment = extract_java_file(
                            ctx.repo, rel_path, source, sha, ctx.java_parser,
                            repo_files=ctx.repo_files,
                            extract_routes=ctx.extract_routes,
                        )
                    else:
                        continue
                except Exception as e:
                    # Don't let one weird file kill the whole scan — log and continue.
                    logger.warning("extractor failed on %s: %s", rel_path, e)
                    continue
                ctx.batch.extend(fragment)
                file_count += 1
                if file_count % PROGRESS_EVERY == 0:
                    elapsed = time.monotonic() - start
                    rate = file_count / elapsed if elapsed > 0 else 0
                    print(f"[indexer]   parsed {file_count} files ({rate:.0f}/s)", flush=True)
        else:
            # Parallel path — multiprocessing.Pool. Imported here, not at
            # module top, so dry-run / unit tests that never use a pool
            # don't fork-related side effects.
            import multiprocessing

            ctx.progress(f"[indexer] parse phase: {ctx.parse_workers} workers")

            # Lazy generator of args tuples. Don't materialize the full
            # list — source bytes for 13K files would balloon RSS. Pool's
            # imap_unordered consumes lazily (with internal buffering).
            args_iter = (
                (rel_path, source, language, sha, ctx.repo, ctx.repo_files,
                 ctx.package_roots, ctx.extract_routes, ctx.extract_mcp_tools,
                 ctx.extract_orm)
                for (rel_path, source, language, sha) in _work_items()
            )

            with multiprocessing.Pool(
                ctx.parse_workers, initializer=_pool_init,
            ) as pool:
                for rel_path, fragment, err in pool.imap_unordered(
                    _pool_extract, args_iter, chunksize=8,
                ):
                    if err is not None:
                        logger.warning("extractor failed on %s: %s", rel_path, err)
                        continue
                    if fragment is None:
                        # Unsupported language — already filtered by the
                        # walker, but defensive.
                        continue
                    ctx.batch.extend(fragment)
                    file_count += 1
                    if file_count % PROGRESS_EVERY == 0:
                        elapsed = time.monotonic() - start
                        rate = file_count / elapsed if elapsed > 0 else 0
                        print(f"[indexer]   parsed {file_count} files ({rate:.0f}/s)", flush=True)

        walk_secs = time.monotonic() - start
        if ctx.skipped_files:
            ctx.progress(
                f"[indexer] skipped {ctx.skipped_files} unchanged files (SHA match)"
            )
        ctx.progress(f"[indexer] parsed {file_count} files in {walk_secs:.1f}s")
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
