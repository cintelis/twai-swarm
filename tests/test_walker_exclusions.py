"""Walker exclusion tests — `additional_skip_dirs` parameter.

Sprint 14e follow-up. Adds the `--exclude-dirs` CLI flag's plumbing
through walk_paths / walk_repo. Used by twai-swarm self-scans to skip
`templates/` (bundled scaffolds aren't the agent's own code).
"""
from __future__ import annotations

from pathlib import Path

from app.repo_indexer.walker import walk_paths, walk_repo


def _write(repo: Path, rel_path: str, content: str = "") -> Path:
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return full


def test_walk_paths_default_includes_everything(tmp_path):
    """Default behaviour preserved — no exclusions means all parseable
    files are yielded."""
    _write(tmp_path, "app/main.py", "x = 1")
    _write(tmp_path, "templates/scaffold/app/main.py", "y = 2")

    rel_paths = sorted(rel for rel, _ in walk_paths(tmp_path))
    assert "app/main.py" in rel_paths
    assert "templates/scaffold/app/main.py" in rel_paths


def test_walk_paths_skips_additional_dirs(tmp_path):
    """When templates is in additional_skip_dirs, files under it are
    excluded — same mechanism as the built-in SKIP_DIRS."""
    _write(tmp_path, "app/main.py", "x = 1")
    _write(tmp_path, "templates/scaffold/app/main.py", "y = 2")
    _write(tmp_path, "templates/other/foo.py", "z = 3")

    rel_paths = sorted(rel for rel, _ in walk_paths(
        tmp_path, additional_skip_dirs=frozenset({"templates"})
    ))
    assert rel_paths == ["app/main.py"]


def test_walk_repo_skips_additional_dirs(tmp_path):
    """walk_repo (the variant that reads file bytes + computes SHAs)
    honours additional_skip_dirs identically to walk_paths."""
    _write(tmp_path, "app/main.py", "x = 1\n")
    _write(tmp_path, "templates/scaffold/app/main.py", "y = 2\n")

    rel_paths = sorted(rel for rel, _, _, _ in walk_repo(
        tmp_path, additional_skip_dirs=frozenset({"templates"})
    ))
    assert rel_paths == ["app/main.py"]


def test_walk_paths_skip_matches_any_directory_with_that_name(tmp_path):
    """SKIP_DIRS-style: a directory NAME match excludes EVERY directory
    of that name, anywhere in the tree. This is intentional — same
    contract as the built-in `node_modules` / `.venv` exclusions —
    callers wanting a path-prefix match should use .gitignore instead."""
    _write(tmp_path, "templates/foo.py", "x = 1")
    _write(tmp_path, "src/templates/foo.py", "x = 2")  # nested templates dir
    _write(tmp_path, "app/main.py", "y = 3")

    rel_paths = sorted(rel for rel, _ in walk_paths(
        tmp_path, additional_skip_dirs=frozenset({"templates"})
    ))
    assert rel_paths == ["app/main.py"]


def test_walk_paths_empty_additional_skip_dirs_is_default(tmp_path):
    """Passing an empty frozenset matches the no-arg default — no
    behaviour change."""
    _write(tmp_path, "templates/foo.py", "x = 1")
    rel_default = sorted(rel for rel, _ in walk_paths(tmp_path))
    rel_empty = sorted(rel for rel, _ in walk_paths(
        tmp_path, additional_skip_dirs=frozenset()
    ))
    assert rel_default == rel_empty
