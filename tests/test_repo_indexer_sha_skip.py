"""Sprint 11b — per-file SHA short-circuit tests.

Drives `ParsePhase` directly with a synthetic on-disk repo, no Neo4j and
no driver. `ctx.prior_shas` is pre-seeded by the test (in real usage
ParsePhase prefetches via `fetch_file_shas` — that path requires a live
driver and is exercised by integration tests).

Tree-sitter is required for actual extraction; we lazy-skip the module
the same way `tests/test_repo_indexer.py` does so the suite still
collects when only the walker tests are wanted.
"""
from __future__ import annotations

import hashlib

import pytest

from app.repo_indexer.actions import IndexBatch, RepoNode
from app.repo_indexer.runner import PhaseContext

try:
    import tree_sitter_python  # noqa: F401
    from app.repo_indexer.__main__ import _parser_for_python
    from app.repo_indexer.phases.parse import ParsePhase
    HAS_TS = True
except ImportError:
    HAS_TS = False


REPO = RepoNode(name="testrepo", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter / tree-sitter-python not installed")
    return _parser_for_python()


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_repo(tmp_path) -> dict[str, str]:
    """Build a 2-file synthetic Python repo. Returns {rel_path: sha}."""
    a = tmp_path / "a.py"
    a.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("def beta():\n    return 2\n", encoding="utf-8")
    return {"a.py": _sha(a), "b.py": _sha(b)}


def _ctx(tmp_path, parser, *, prior_shas=None) -> PhaseContext:
    return PhaseContext(
        repo=REPO,
        repo_root=tmp_path,
        languages=("python",),
        batch=IndexBatch(repo=REPO),
        py_parser=parser,
        ts_parsers=None,
        driver=None,                      # disables the SHA prefetch
        prior_shas=dict(prior_shas or {}),
    )


def test_no_prior_shas_extracts_everything(tmp_path, parser):
    """Empty prior_shas + driver=None means no skips, full extraction."""
    _make_repo(tmp_path)
    ctx = _ctx(tmp_path, parser)
    ParsePhase().run(ctx)

    assert ctx.skipped_files == 0
    assert len(ctx.batch.files) == 2
    qns = {fn.qualified_name for fn in ctx.batch.functions}
    assert qns == {"a.alpha", "b.beta"}


def test_all_shas_match_skips_everything(tmp_path, parser):
    """When every on-disk SHA matches prior, batch stays empty."""
    shas = _make_repo(tmp_path)
    ctx = _ctx(tmp_path, parser, prior_shas=shas)
    ParsePhase().run(ctx)

    assert ctx.skipped_files == 2
    assert ctx.batch.files == []
    assert ctx.batch.functions == []
    assert ctx.batch.modules == []


def test_mixed_skips_unchanged_only(tmp_path, parser):
    """Pre-seed only one file's SHA — the other should still be extracted."""
    shas = _make_repo(tmp_path)
    # Pretend a.py is unchanged; b.py is "new" (no prior entry).
    ctx = _ctx(tmp_path, parser, prior_shas={"a.py": shas["a.py"]})
    ParsePhase().run(ctx)

    assert ctx.skipped_files == 1
    assert len(ctx.batch.files) == 1
    assert ctx.batch.files[0].path == "b.py"
    qns = {fn.qualified_name for fn in ctx.batch.functions}
    assert qns == {"b.beta"}
