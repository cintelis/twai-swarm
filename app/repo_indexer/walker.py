"""File walker — yields source files from a repo.

Respects:
    - .gitignore (one level — top-level only; rare to nest in a real repo)
    - hardcoded denylist of dependency / build / cache dirs
    - file-extension allowlist per language

The walker doesn't parse anything; it just decides what's parseable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from .actions import Language

# Directory names we never recurse into. Independent of .gitignore so a
# repo without a .gitignore still gets sane defaults.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", ".next", "dist", "build", "out",
    ".tox", ".nox", ".cache",
    "target",  # rust / java
    ".idea", ".vscode",
    "coverage", ".nyc_output",
    "egg-info",
})

# Extension → language mapping. Sprint 10b adds .ts / .tsx / .js / .jsx.
EXT_LANGUAGE: dict[str, Language] = {
    ".py": "python",
}


def _read_gitignore_patterns(repo_root: Path) -> list[str]:
    """Top-level .gitignore patterns. Skipped if the file is missing."""
    gi = repo_root / ".gitignore"
    if not gi.is_file():
        return []
    out: list[str] = []
    for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Strip leading slash — we treat patterns as relative-from-root.
        if s.startswith("/"):
            s = s[1:]
        # Strip trailing slash on dir-only patterns; we match dirs and files
        # the same way at this level.
        if s.endswith("/"):
            s = s[:-1]
        out.append(s)
    return out


def _matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Cheap fnmatch-style match. Good enough for the common cases —
    dotfile prefixes, dir names, `*.foo`. We don't need full gitignore
    semantics; the hardcoded SKIP_DIRS already covers most generated junk."""
    import fnmatch
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        # Match directory components too — `dist` should hit `dist/foo.py`.
        for part in rel_path.split("/"):
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def _file_sha(path: Path) -> str:
    """SHA-256 of file bytes — drives diff-skip on re-scan."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_repo(
    repo_root: Path,
    languages: tuple[Language, ...] = ("python",),
) -> Iterator[tuple[str, bytes, Language, str]]:
    """Yield (rel_path, source_bytes, language, sha) for every parseable file.

    `rel_path` is a posix-style relative path from `repo_root`. `sha` is the
    SHA-256 hex digest of the file contents — used by the loader to skip
    re-MERGE on unchanged files.
    """
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo root {repo_root} is not a directory")

    gi_patterns = _read_gitignore_patterns(repo_root)
    allowed_exts = {ext for ext, lang in EXT_LANGUAGE.items() if lang in languages}

    # Manual walk so we can prune SKIP_DIRS in-place without recursion cost.
    def _walk(d: Path) -> Iterator[Path]:
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name in SKIP_DIRS:
                continue
            if entry.is_dir():
                yield from _walk(entry)
            elif entry.is_file():
                yield entry

    for path in _walk(repo_root):
        ext = path.suffix.lower()
        if ext not in allowed_exts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if _matches_gitignore(rel, gi_patterns):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        # Skip files that are too large to be sensible source — usually
        # generated bundles / fixtures sneak in.
        if len(data) > 2 * 1024 * 1024:  # 2 MB
            continue
        yield rel, data, EXT_LANGUAGE[ext], _file_sha(path)
