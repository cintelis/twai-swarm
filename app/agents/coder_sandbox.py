"""
Per-workflow sandbox directory for the agentic Coder.

The Coder's tools (list/read/write/run_verify/bash_exec) all operate
through `Sandbox`. It's the ONE place we enforce:

- paths stay inside the workspace root (no `..` escapes, no abs paths)
- writes have a size cap (no accidental 10MB blobs)
- reads are bounded (so one huge file can't blow the prompt budget)
- bash subprocess env is stripped of provider/AWS/Temporal secrets

Keep this file boring. The only defence against a model talking us into
trashing the container FS is the checks below.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_WRITE_BYTES = 200 * 1024       # 200 KB per file; template scaffolds never need more
MAX_READ_BYTES = 50 * 1024         # 50 KB per read; longer files get tailed

# bash_exec ceilings. 60s default per-call covers pip install + a smoke
# command; the 300s ceiling matches verify.sh.
BASH_DEFAULT_TIMEOUT_SECONDS = 60
BASH_MAX_TIMEOUT_SECONDS = 300
BASH_MAX_OUTPUT_BYTES = 10 * 1024  # per stream; truncate middle if longer

# Env var names allowlisted into the Coder bash subprocess. Threat model:
# prompt injection convinces the model to `printenv > exfil`. Allowlisting
# is safer than denying — new secret types get caught automatically.
#
# None of these reveal app secrets. Windows entries (SYSTEMROOT et al.) are
# required by bash.exe / WSL shim to launch at all; on Linux containers in
# production they're absent and harmless.
ENV_ALLOWLIST = frozenset({
    # POSIX essentials
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
    "PWD", "SHLVL", "TMPDIR", "TMP", "TEMP",
    # Tool-chain caches — speeds up repeat installs without leaking creds.
    "PIP_CACHE_DIR", "NPM_CONFIG_CACHE", "CARGO_HOME",
    # Windows essentials — subprocess can't even start without these on Win.
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "USERPROFILE", "USERNAME", "COMPUTERNAME",
    "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS",
    "OS", "PATHEXT",
})


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

    @classmethod
    def wrap(cls, existing_path: str | os.PathLike) -> "Sandbox":
        """Wrap an existing directory as a Sandbox without mkdir.

        Used by the RepoTaskWorkflow (Sprint 10e) — the cloned repo is
        already on disk and we want the same path-validation + tooling
        primitives without the create-empty-dir semantics."""
        root = Path(existing_path).resolve()
        if not root.is_dir():
            raise SandboxError(f"sandbox wrap target {root} is not a directory")
        return cls(root=root)

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

    async def run_bash(
        self,
        command: str,
        timeout_seconds: int = BASH_DEFAULT_TIMEOUT_SECONDS,
    ) -> tuple[int, str, str]:
        """Run an arbitrary shell command in the workspace.

        Returns (exit_code, stdout, stderr). Each stream is capped at
        BASH_MAX_OUTPUT_BYTES with a middle-truncation marker.

        Env is scrubbed to ENV_ALLOWLIST plus AWS_EC2_METADATA_DISABLED=true,
        so the subprocess can't read provider keys, RDS DSN, GitHub App key,
        etc. via printenv or talk to IMDS for IAM creds. Coder doesn't need
        any of those to write code; if a legitimate use surfaces, add it
        explicitly to the allowlist.
        """
        if not isinstance(command, str) or not command.strip():
            raise SandboxError("command must be a non-empty string")
        if timeout_seconds < 1:
            raise SandboxError("timeout_seconds must be >= 1")
        timeout_seconds = min(timeout_seconds, BASH_MAX_TIMEOUT_SECONDS)

        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
        # Defense in depth even though we strip AWS_* — boto3 would otherwise
        # fall back to IMDS to pick up the task role.
        env["AWS_EC2_METADATA_DISABLED"] = "true"

        def _run() -> tuple[int, str, str]:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return proc.returncode, _cap_output(proc.stdout), _cap_output(proc.stderr)

        try:
            return await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as e:
            partial_out = _cap_output((e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
            partial_err = _cap_output((e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
            return -1, partial_out, f"timed out after {timeout_seconds}s\n{partial_err}"


def _cap_output(text: str, limit: int = BASH_MAX_OUTPUT_BYTES) -> str:
    """Cap a stream to `limit` bytes, keeping head + tail with a middle marker."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    half = limit // 2
    head = data[:half].decode("utf-8", errors="replace")
    tail = data[-half:].decode("utf-8", errors="replace")
    return f"{head}\n[…{len(data) - limit} bytes elided…]\n{tail}"
