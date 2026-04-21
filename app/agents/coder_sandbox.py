"""
Per-workflow sandbox directory for the agentic Coder.

The Coder's tools (list/read/write/run_verify) all operate through
`Sandbox`. It's the ONE place we enforce:

- paths stay inside the workspace root (no `..` escapes, no abs paths)
- writes have a size cap (no accidental 10MB blobs)
- reads are bounded (so one huge file can't blow the prompt budget)

Keep this file boring. The only defence against a model talking us into
trashing the container FS is the checks below.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

MAX_WRITE_BYTES = 200 * 1024       # 200 KB per file; template scaffolds never need more
MAX_READ_BYTES = 50 * 1024         # 50 KB per read; longer files get tailed


class SandboxError(Exception):
    """Raised when a tool call would escape the workspace or violate a bound."""


@dataclass(frozen=True)
class Sandbox:
    """A scoped workspace dir. Create via `Sandbox.create(workflow_id)`."""

    root: Path

    @classmethod
    def create(cls, workflow_id: str, base: str | os.PathLike = "/tmp/coder") -> "Sandbox":
        # Sanitise workflow_id — we stick it in a path, so ban anything funky.
        safe = "".join(c for c in workflow_id if c.isalnum() or c in ("-", "_"))
        if not safe:
            raise SandboxError(f"workflow_id {workflow_id!r} has no safe chars for path")
        root = Path(base) / safe
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root.resolve())

    def destroy(self) -> None:
        """Remove the workspace. Safe to call twice."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def resolve(self, rel_path: str) -> Path:
        """Resolve a relative path to an absolute path inside the sandbox.

        Rejects absolute paths, `..` segments that escape root, and any path
        that resolves outside `self.root`. The model is told to use
        forward-slashes and relative paths — this is the enforcement.
        """
        if not rel_path or not isinstance(rel_path, str):
            raise SandboxError("path must be a non-empty string")
        # Normalise slashes — the model may send either.
        rel = rel_path.replace("\\", "/").lstrip("/")
        if not rel:
            raise SandboxError("path is empty after normalisation")
        candidate = (self.root / rel).resolve()
        # Containment check — candidate must be at or below root.
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise SandboxError(f"path {rel_path!r} escapes the workspace")
        return candidate

    def write(self, rel_path: str, content: str) -> int:
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise SandboxError(
                f"write to {rel_path!r} is {len(data)} bytes, over the {MAX_WRITE_BYTES}-byte cap"
            )
        full = self.resolve(rel_path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        return len(data)

    def read(self, rel_path: str) -> tuple[str, bool]:
        """Return (text, truncated). Reads beyond MAX_READ_BYTES are head-truncated."""
        full = self.resolve(rel_path)
        if not full.exists():
            raise SandboxError(f"path {rel_path!r} does not exist")
        if not full.is_file():
            raise SandboxError(f"path {rel_path!r} is not a file")
        size = full.stat().st_size
        with full.open("rb") as f:
            raw = f.read(MAX_READ_BYTES)
        text = raw.decode("utf-8", errors="replace")
        return text, size > MAX_READ_BYTES

    def list_files(self, max_entries: int = 500) -> list[str]:
        """Relative posix paths, sorted, with common junk directories skipped."""
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache",
                ".ruff_cache", ".mypy_cache", "dist", "build", ".tox"}
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune skip dirs in-place so os.walk doesn't recurse into them.
            dirnames[:] = [d for d in dirnames if d not in skip]
            for name in filenames:
                rel = Path(dirpath, name).relative_to(self.root).as_posix()
                out.append(rel)
                if len(out) >= max_entries:
                    return sorted(out)
        return sorted(out)

    def copy_in(self, src_dir: str | os.PathLike) -> None:
        """Copy a template's scaffold/ contents into the workspace."""
        src = Path(src_dir)
        if not src.is_dir():
            raise SandboxError(f"template source {src} is not a directory")
        # dirs_exist_ok=True lets us copy onto a freshly-created empty root.
        shutil.copytree(src, self.root, dirs_exist_ok=True)
