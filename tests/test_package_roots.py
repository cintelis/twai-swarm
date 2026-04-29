"""Sprint 14e — package-root detection tests.

Synthetic fs fixtures via pytest's tmp_path. No tree-sitter / Neo4j —
the module is pure stdlib (`tomllib` + `pathlib`).
"""
from __future__ import annotations

from pathlib import Path

from app.repo_indexer.package_roots import (
    PackageRoot,
    detect_package_roots,
    module_qn_for,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _write(repo: Path, rel_path: str, content: str = "") -> Path:
    """Create a file at `repo/rel_path`, parents auto-created."""
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return full


# ─── detect_package_roots ───────────────────────────────────────────────────

def test_empty_repo_returns_empty_list(tmp_path):
    """No pyproject.toml anywhere → no roots."""
    assert detect_package_roots(tmp_path) == []


def test_single_package_repo_one_root(tmp_path):
    """Twai-swarm-shape: pyproject at repo root with hatch packages, app/
    is the package directory. fs_root="" (repo IS the anchor)."""
    _write(tmp_path, "pyproject.toml", """
[project]
name = "myapp"
version = "0.1.0"

[tool.hatch.build.targets.wheel]
packages = ["app"]
""")
    _write(tmp_path, "app/__init__.py")
    _write(tmp_path, "app/foo.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0] == PackageRoot(fs_root="", src_relative="")


def test_monorepo_langgraph_shape_two_roots_sorted(tmp_path):
    """Langgraph-shape: two libs/X/pyproject.toml's, each declaring its
    own package. Returned longest-fs-root first so module_qn_for picks
    the most-specific match."""
    _write(tmp_path, "libs/langgraph/pyproject.toml", """
[project]
name = "langgraph"

[tool.hatch.build.targets.wheel]
packages = ["langgraph"]
""")
    _write(tmp_path, "libs/langgraph/langgraph/__init__.py")
    _write(tmp_path, "libs/langgraph/langgraph/graph/__init__.py")

    _write(tmp_path, "libs/checkpoint/pyproject.toml", """
[project]
name = "langgraph-checkpoint"

[tool.hatch.build.targets.wheel]
include = ["langgraph"]
""")
    _write(tmp_path, "libs/checkpoint/langgraph/checkpoint/__init__.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 2
    # Both have depth 2 ("libs/langgraph", "libs/checkpoint"); alpha tie-break.
    assert [r.fs_root for r in roots] == ["libs/checkpoint", "libs/langgraph"]


def test_nested_pyprojects_sorted_most_specific_first(tmp_path):
    """A repo with a top-level pyproject AND a nested one — module_qn_for
    must consult the nested one first for files under it."""
    _write(tmp_path, "pyproject.toml", """
[project]
name = "outer"
[tool.hatch.build.targets.wheel]
packages = ["outer"]
""")
    _write(tmp_path, "outer/__init__.py")

    _write(tmp_path, "examples/myproj/pyproject.toml", """
[project]
name = "innerproj"
[tool.hatch.build.targets.wheel]
packages = ["innerproj"]
""")
    _write(tmp_path, "examples/myproj/innerproj/__init__.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 2
    # Nested first (depth 2 > depth 0).
    assert roots[0].fs_root == "examples/myproj"
    assert roots[1].fs_root == ""


# ─── src/ layout detection ──────────────────────────────────────────────────

def test_poetry_src_layout_detected(tmp_path):
    """Poetry's [{include = "X", from = "src"}] form sets src_relative."""
    _write(tmp_path, "pyproject.toml", """
[tool.poetry]
name = "myapp"
version = "0.1.0"
packages = [{include = "myapp", from = "src"}]
""")
    _write(tmp_path, "src/myapp/__init__.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].src_relative == "src"


def test_setuptools_src_layout_detected(tmp_path):
    """Setuptools package-dir = {"" = "src"} sets src_relative."""
    _write(tmp_path, "pyproject.toml", """
[project]
name = "myapp"

[tool.setuptools]
package-dir = {"" = "src"}
""")
    _write(tmp_path, "src/myapp/__init__.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].src_relative == "src"


# ─── tooling-only pyprojects skipped ────────────────────────────────────────

def test_tooling_only_pyproject_skipped(tmp_path):
    """A pyproject that only configures ruff/pytest/black is NOT a package
    definition; we shouldn't mistake it for one and shorten qns of files
    that aren't actually under any real package."""
    _write(tmp_path, "pyproject.toml", """
[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
""")
    assert detect_package_roots(tmp_path) == []


# ─── exclusion of vendored trees ────────────────────────────────────────────

def test_pyprojects_in_excluded_dirs_skipped(tmp_path):
    """We shouldn't descend into .venv / node_modules looking for
    pyproject.toml. Same exclusion rules as the walker."""
    # Real package
    _write(tmp_path, "libs/real/pyproject.toml", """
[project]
name = "real"
[tool.hatch.build.targets.wheel]
packages = ["real"]
""")
    _write(tmp_path, "libs/real/real/__init__.py")

    # Vendored — should be ignored
    _write(tmp_path, ".venv/lib/site-packages/wheelpkg/pyproject.toml", """
[project]
name = "wheelpkg"
[tool.hatch.build.targets.wheel]
packages = ["wheelpkg"]
""")
    _write(tmp_path, "node_modules/something/pyproject.toml", """
[project]
name = "junk"
[tool.hatch.build.targets.wheel]
packages = ["junk"]
""")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].fs_root == "libs/real"


def test_malformed_pyproject_doesnt_crash(tmp_path):
    """Garbage-in pyproject.toml is silently skipped — don't break the
    whole scan because one repo has a broken file."""
    _write(tmp_path, "pyproject.toml", "this is not [valid toml")
    # Real package alongside it should still be picked up.
    _write(tmp_path, "libs/good/pyproject.toml", """
[project]
name = "good"
[tool.hatch.build.targets.wheel]
packages = ["good"]
""")
    _write(tmp_path, "libs/good/good/__init__.py")

    roots = detect_package_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].fs_root == "libs/good"


# ─── module_qn_for ──────────────────────────────────────────────────────────

def test_module_qn_for_no_roots_falls_back_to_dotted_path():
    """Pre-14e behavior preserved when no package roots match."""
    assert module_qn_for("app/repo_indexer/walker.py", []) == "app.repo_indexer.walker"
    assert module_qn_for("a/b/c.py", []) == "a.b.c"


def test_module_qn_for_init_collapses_to_package():
    """`pkg/__init__.py` → "pkg", not "pkg.__init__"."""
    assert module_qn_for("pkg/__init__.py", []) == "pkg"
    assert module_qn_for("a/b/__init__.py", []) == "a.b"


def test_module_qn_for_picks_most_specific_root():
    """When a file falls under multiple roots, the deepest fs_root wins."""
    roots = [
        PackageRoot(fs_root="examples/myproj", src_relative=""),
        PackageRoot(fs_root="", src_relative=""),
    ]
    # File under nested root → uses inner package qn
    assert module_qn_for("examples/myproj/innerproj/foo.py", roots) == "innerproj.foo"
    # File under outer root only → uses outer
    assert module_qn_for("outer/bar.py", roots) == "outer.bar"


def test_module_qn_for_strips_libs_prefix():
    """The langgraph case: libs/langgraph/langgraph/graph/state.py →
    langgraph.graph.state."""
    roots = [PackageRoot(fs_root="libs/langgraph", src_relative="")]
    qn = module_qn_for("libs/langgraph/langgraph/graph/state.py", roots)
    assert qn == "langgraph.graph.state"


def test_module_qn_for_namespace_package_path():
    """Namespace-package case: libs/checkpoint/langgraph/checkpoint/state.py
    → langgraph.checkpoint.state. The pyproject_dir-as-anchor approach
    handles this without explicit namespace-package logic."""
    roots = [PackageRoot(fs_root="libs/checkpoint", src_relative="")]
    qn = module_qn_for("libs/checkpoint/langgraph/checkpoint/state.py", roots)
    assert qn == "langgraph.checkpoint.state"


def test_module_qn_for_src_layout_strips_src():
    """src/-layout: src_relative="src" gets stripped along with fs_root."""
    roots = [PackageRoot(fs_root="", src_relative="src")]
    assert module_qn_for("src/myapp/foo.py", roots) == "myapp.foo"
    # Combined: nested root + src layout
    roots = [PackageRoot(fs_root="libs/foo", src_relative="src")]
    assert module_qn_for("libs/foo/src/myapp/x.py", roots) == "myapp.x"


def test_module_qn_for_file_at_root_init_collapses():
    """A file at fs_root/X/__init__.py with package_qn=X collapses to X."""
    roots = [PackageRoot(fs_root="libs/langgraph", src_relative="")]
    assert (
        module_qn_for("libs/langgraph/langgraph/__init__.py", roots)
        == "langgraph"
    )


def test_module_qn_for_file_outside_any_root_uses_fallback():
    """A file that doesn't fall under any declared package root keeps
    the dotted repo-relative path. (E.g. a top-level script in a monorepo
    that isn't part of any package.)"""
    roots = [PackageRoot(fs_root="libs/foo", src_relative="")]
    assert module_qn_for("scripts/migrate.py", roots) == "scripts.migrate"


# ─── end-to-end: detect + module_qn_for round-trip ──────────────────────────

def test_end_to_end_langgraph_shape(tmp_path):
    """Walk the langgraph-shape fixture, detect roots, and confirm
    `module_qn_for` produces the import qns we'd see in real Python."""
    _write(tmp_path, "libs/langgraph/pyproject.toml", """
[project]
name = "langgraph"
[tool.hatch.build.targets.wheel]
packages = ["langgraph"]
""")
    _write(tmp_path, "libs/langgraph/langgraph/__init__.py")
    _write(tmp_path, "libs/langgraph/langgraph/graph/__init__.py")
    _write(tmp_path, "libs/langgraph/langgraph/graph/state.py")

    _write(tmp_path, "libs/checkpoint/pyproject.toml", """
[project]
name = "langgraph-checkpoint"
[tool.hatch.build.targets.wheel]
include = ["langgraph"]
""")
    _write(tmp_path, "libs/checkpoint/langgraph/checkpoint/__init__.py")
    _write(tmp_path, "libs/checkpoint/langgraph/checkpoint/sqlite.py")

    roots = detect_package_roots(tmp_path)

    cases = {
        "libs/langgraph/langgraph/__init__.py":             "langgraph",
        "libs/langgraph/langgraph/graph/__init__.py":       "langgraph.graph",
        "libs/langgraph/langgraph/graph/state.py":          "langgraph.graph.state",
        "libs/checkpoint/langgraph/checkpoint/__init__.py": "langgraph.checkpoint",
        "libs/checkpoint/langgraph/checkpoint/sqlite.py":   "langgraph.checkpoint.sqlite",
    }
    for rel_path, expected in cases.items():
        assert module_qn_for(rel_path, roots) == expected, (
            f"{rel_path} -> got {module_qn_for(rel_path, roots)!r}, want {expected!r}"
        )
