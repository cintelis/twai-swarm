"""Package-root detection for Python module-qn construction.

Sprint 14e. Fixes the cross-package call resolution gap on monorepo
codebases (langgraph: 95% unresolved → ~88% pre-fix). The bug is one
function — `extractor_python._module_qn_from_path` builds the qn from
the file path relative to the *repo root*, but a Python file's actual
import qn comes from the project's `pyproject.toml`-declared package
layout. Concrete: `libs/langgraph/langgraph/graph/state.py` should map
to `langgraph.graph.state` (matches `from langgraph.graph.state import
StateGraph`), not `libs.langgraph.langgraph.graph.state`.

Approach
--------
Walk the repo for `pyproject.toml` files; treat each one's parent
directory as a package-root anchor. A file under that anchor has its
qn computed relative to the anchor, not the repo root.

Two adjustments are needed:

1. `src/`-layout. Poetry's `[{include = "X", from = "src"}]` and
   setuptools' `package-dir = {"" = "src"}` shift the source under
   a `src/` subdir. Strip that.
2. Tooling-only pyprojects. Some repos vend a `pyproject.toml` that
   only configures ruff/pytest/black with no build-system or project
   metadata. Skip these so we don't shorten qns for files outside any
   real package.

For namespace packages (PEP 420 — directory without `__init__.py` whose
nested subdirs are real packages, like `langgraph-checkpoint`'s
`libs/checkpoint/langgraph/checkpoint/`), the pyproject_dir-as-anchor
approach Just Works: stripping `libs/checkpoint/` from
`libs/checkpoint/langgraph/checkpoint/state.py` yields
`langgraph/checkpoint/state.py` → `langgraph.checkpoint.state`. No
explicit namespace-package handling required.

Out of scope for v1
-------------------
- `find_packages()` / `[tool.setuptools.packages.find]` auto-discovery
- TypeScript / `package.json` workspace handling (`extractor_typescript`
  uses path-based `repo_files` resolution, different bug shape)
- Local-variable type inference (the `builder.add_node()` family of
  unresolved calls — Sprint 12c-deferred)
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Same exclusions the walker uses — we don't want to descend into vendored
# trees just to find pyproject.toml's there. Keep in sync if walker.py
# updates its list.
_EXCLUDED_DIRS = frozenset({
    ".venv", "venv", "env", ".env",
    "node_modules", ".next", "dist", "build", "out",
    "__pycache__", ".git", ".tox", ".pytest_cache",
    ".ruff_cache", ".mypy_cache",
})


@dataclass(frozen=True)
class PackageRoot:
    """Maps a filesystem subtree to its Python-import qn anchor.

    `fs_root` is the directory (repo-relative, posix-style) that acts as
    the package anchor. Files at `fs_root/X/Y.py` import as `X.Y`.

    `src_relative` is the layout-shift between fs_root and the actual
    source dir. For most repos it's `""`. For `src/`-layout it's
    `"src"` (so files at `fs_root/src/X/Y.py` import as `X.Y`).
    """
    fs_root: str       # repo-relative, posix-style; "" means repo root itself
    src_relative: str  # "", "src", etc. — relative to fs_root


def _is_package_pyproject(data: dict) -> bool:
    """Heuristic: does this pyproject.toml define a Python package vs.
    just tooling config (ruff / pytest / black-only)?

    A pyproject IS a package definition if it has any of:
    - `[project]` table (PEP 621)
    - `[tool.poetry]` table
    - `[tool.hatch.build]` or `[tool.hatch.version]`
    - `[tool.setuptools]` with anything in it

    A tooling-only pyproject (just `[tool.ruff]` / `[tool.pytest]`) returns
    False — we don't want to shorten qns for files in a directory that
    isn't actually a Python distribution.
    """
    if "project" in data:
        return True
    tool = data.get("tool", {})
    if "poetry" in tool:
        return True
    hatch = tool.get("hatch", {})
    if "build" in hatch or "version" in hatch:
        return True
    if tool.get("setuptools"):
        return True
    return False


def _detect_src_layout(data: dict) -> str:
    """Return the `src/`-layout subdir relative to pyproject_dir, or `""`
    if no src layout is configured.

    Recognises:
    - Poetry: `[tool.poetry].packages = [{include = "X", from = "src"}]`
    - Setuptools: `[tool.setuptools].package-dir = {"" = "src"}`

    Hatch supports src layouts via different mechanisms but they're rare
    enough in practice that we punt — the fallback is "".
    """
    tool = data.get("tool", {})

    poetry_pkgs = tool.get("poetry", {}).get("packages", [])
    if isinstance(poetry_pkgs, list):
        for pkg in poetry_pkgs:
            if isinstance(pkg, dict) and isinstance(pkg.get("from"), str):
                return pkg["from"]

    pkg_dir = tool.get("setuptools", {}).get("package-dir", {})
    if isinstance(pkg_dir, dict) and isinstance(pkg_dir.get(""), str):
        return pkg_dir[""]

    return ""


def detect_package_roots(repo_root: Path) -> list[PackageRoot]:
    """Walk `repo_root` for `pyproject.toml` files; return one PackageRoot
    per file that declares a real Python package.

    Sorted by fs_root depth descending so `module_qn_for` picks the most
    specific match first (a file under `libs/foo/` should be addressed
    via `libs/foo/`'s pyproject, not the parent repo's).
    """
    roots: list[PackageRoot] = []
    for pyproject in repo_root.rglob("pyproject.toml"):
        if any(part in _EXCLUDED_DIRS for part in pyproject.parts):
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
            continue
        if not _is_package_pyproject(data):
            continue

        pyproject_dir = pyproject.parent
        try:
            fs_rel = pyproject_dir.relative_to(repo_root)
        except ValueError:
            continue

        # repo_root's own pyproject -> fs_root="" (empty), meaning the
        # repo root is the package anchor.
        fs_root_str = str(fs_rel).replace("\\", "/")
        if fs_root_str == ".":
            fs_root_str = ""

        roots.append(PackageRoot(
            fs_root=fs_root_str,
            src_relative=_detect_src_layout(data),
        ))

    # Most-specific first. Tie-break alphabetically for determinism.
    roots.sort(key=lambda r: (-_depth(r.fs_root), r.fs_root))
    return roots


def _depth(fs_root: str) -> int:
    """Path-segment count for sort priority. "" is depth 0; "libs/foo"
    is depth 2. Used so nested package roots win in module_qn_for."""
    return 0 if not fs_root else fs_root.count("/") + 1


def module_qn_for(rel_path: str, roots: list[PackageRoot]) -> str:
    """Build a Python module qn for `rel_path` (repo-relative posix-style)
    using the most-specific matching package root. Falls back to dotted
    repo-relative path when no root matches.

    `__init__.py` collapses to its package's qn — `pkg/__init__.py` →
    `"pkg"`, not `"pkg.__init__"`.
    """
    parts = rel_path.removesuffix(".py").split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return ""

    for root in roots:
        prefix = _root_prefix_parts(root)
        if len(parts) > len(prefix) and parts[:len(prefix)] == prefix:
            return ".".join(parts[len(prefix):])

    # No package root matched — current pre-14e behavior.
    return ".".join(parts)


def _root_prefix_parts(root: PackageRoot) -> list[str]:
    """The path-segment prefix that `module_qn_for` strips from a file
    path. Combines fs_root + src_relative."""
    parts: list[str] = []
    if root.fs_root:
        parts.extend(root.fs_root.split("/"))
    if root.src_relative:
        parts.extend(root.src_relative.split("/"))
    return parts
