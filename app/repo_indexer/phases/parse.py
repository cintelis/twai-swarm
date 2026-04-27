"""Parse phase — walks the repo, dispatches each file to the right
extractor, and accumulates fragments into `ctx.batch`.

Lazy-imports the extractors so tests that only touch actions/walker
don't drag tree-sitter into the import path. Per-file failures are
logged and skipped — one weird file shouldn't kill the whole scan.
"""
from __future__ import annotations

import logging
import time

from .. import walker
from ..runner import PhaseContext

logger = logging.getLogger("repo_indexer")

PROGRESS_EVERY = 200


class ParsePhase:
    name = "parse"

    def run(self, ctx: PhaseContext) -> None:
        # Lazy-import — keeps tree-sitter out of the import path for
        # tests that only need actions/walker.
        from ..extractor_python import extract_python_file
        from ..extractor_typescript import extract_typescript_file

        start = time.monotonic()
        file_count = 0
        for rel_path, source, language, sha in walker.walk_repo(
            ctx.repo_root, languages=ctx.languages
        ):
            try:
                if language == "python":
                    fragment = extract_python_file(
                        ctx.repo, rel_path, source, sha, ctx.py_parser,
                    )
                elif language in ("typescript", "javascript"):
                    # TSX files need the TSX grammar; .ts/.js use the regular TS grammar.
                    use_tsx = rel_path.endswith((".tsx", ".jsx"))
                    parser = ctx.ts_parsers["tsx"] if use_tsx else ctx.ts_parsers["typescript"]
                    fragment = extract_typescript_file(
                        ctx.repo, rel_path, source, sha, parser,
                        repo_files=ctx.repo_files, language=language,
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

        walk_secs = time.monotonic() - start
        ctx.progress(f"[indexer] parsed {file_count} files in {walk_secs:.1f}s")
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
