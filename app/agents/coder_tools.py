"""
Coder tools — the four things the model can do inside the sandbox.

All four are thin wrappers over `Sandbox`. They return **strings** because
the Anthropic tool-use API serialises tool_result content as text; any
structured shape gets JSON-encoded into a string.

Factory pattern: `build_tools(sandbox)` returns four `@beta_async_tool`
instances that close over `sandbox`. We need the factory because the tool
runner's decorator introspects the function signature to build a schema —
so the sandbox can't be a parameter, it has to be captured.
"""
from __future__ import annotations

import asyncio
import json
import subprocess

from anthropic import beta_async_tool

from .coder_sandbox import Sandbox, SandboxError

# subprocess timeout for verify.sh — pip install + ruff + pytest on a
# small scaffold shouldn't need more than this; something is wedged if it does.
VERIFY_TIMEOUT_SECONDS = 300
# Keep the tail bounded so one chatty verify run doesn't blow the context.
VERIFY_LOG_TAIL_BYTES = 8 * 1024


def _tail(text: str, limit: int = VERIFY_LOG_TAIL_BYTES) -> str:
    """Last `limit` bytes of `text`, with a prefix marker if truncated."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return f"[…{len(data) - limit} earlier bytes elided…]\n" + data[-limit:].decode(
        "utf-8", errors="replace"
    )


def build_tools(sandbox: Sandbox) -> tuple[list, dict]:
    """Return (tools_list, stats_dict). The stats dict is mutated in place
    by the tools as they're called, so the caller can report usage."""

    stats = {
        "list_files_calls": 0,
        "read_file_calls": 0,
        "write_file_calls": 0,
        "run_verify_calls": 0,
        "bytes_written": 0,
        "last_verify_exit": None,
        "last_verify_stdout": "",
        "last_verify_stderr": "",
    }

    @beta_async_tool
    async def list_files() -> str:
        """List every file in the workspace (relative posix paths, one per line).

        Skips common junk (.git, __pycache__, node_modules, .venv, etc.). Hard
        cap of 500 entries — call read_file on specific paths rather than
        trying to get exhaustive listings.
        """
        stats["list_files_calls"] += 1
        try:
            files = sandbox.list_files()
        except SandboxError as e:
            return f"error: {e}"
        if not files:
            return "(workspace is empty)"
        return "\n".join(files)

    @beta_async_tool
    async def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Relative path from the workspace root (forward slashes, no leading /).
                  Example: "app/main.py", "pyproject.toml", "tests/test_items.py".
        """
        stats["read_file_calls"] += 1
        try:
            text, truncated = sandbox.read(path)
        except SandboxError as e:
            return f"error: {e}"
        prefix = f"(file truncated — only first {len(text)} chars shown)\n" if truncated else ""
        return prefix + text

    @beta_async_tool
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file inside the workspace.

        Parent directories are created automatically. 200KB size cap per file —
        if you need a larger file, split it (the limit keeps prompts sane).

        Args:
            path: Relative path from workspace root. Forward slashes, no leading /.
                  Example: "app/api/routes.py", "README.md", "tests/test_items.py".
            content: Full file content as a UTF-8 string. Overwrites any existing file.
        """
        stats["write_file_calls"] += 1
        try:
            n = sandbox.write(path, content)
        except SandboxError as e:
            return f"error: {e}"
        stats["bytes_written"] += n
        return f"wrote {n} bytes to {path}"

    @beta_async_tool
    async def run_verify() -> str:
        """Run the scaffold's `verify.sh` and return its output.

        The verify script is the definition of "done" for this scaffold — it
        installs dev deps, lints, and runs tests. Call this after every
        meaningful edit. Returns exit code + tailed stdout/stderr so you can
        see what broke.

        On exit 0, you're done — stop editing and return a final summary.
        """
        stats["run_verify_calls"] += 1
        verify_path = sandbox.root / "verify.sh"
        if not verify_path.exists():
            # Fallback: templates keep verify.sh at the template root (alongside
            # scaffold/), so we may need to fetch it. Callers copy both in; if
            # this trips, the template matcher forgot to stage verify.sh.
            msg = "error: verify.sh not found in workspace — cannot verify"
            stats["last_verify_exit"] = -1
            stats["last_verify_stderr"] = msg
            return msg

        # Make sure verify.sh is executable (Windows-origin copies lose +x).
        try:
            verify_path.chmod(0o755)
        except OSError:
            pass

        def _run() -> tuple[int, str, str]:
            proc = subprocess.run(
                ["bash", "verify.sh"],
                cwd=str(sandbox.root),
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT_SECONDS,
            )
            return proc.returncode, proc.stdout, proc.stderr

        try:
            exit_code, stdout, stderr = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            stats["last_verify_exit"] = -2
            stats["last_verify_stderr"] = f"timed out after {VERIFY_TIMEOUT_SECONDS}s"
            return f"error: verify.sh timed out after {VERIFY_TIMEOUT_SECONDS}s"
        except FileNotFoundError:
            # `bash` not on PATH — should never happen in the container, does
            # on dev boxes without bash in PATH.
            stats["last_verify_exit"] = -3
            stats["last_verify_stderr"] = "bash not found on PATH"
            return "error: `bash` interpreter not found on PATH"

        stats["last_verify_exit"] = exit_code
        stats["last_verify_stdout"] = stdout
        stats["last_verify_stderr"] = stderr
        tail_out = _tail(stdout)
        tail_err = _tail(stderr)
        return json.dumps({
            "exit_code": exit_code,
            "stdout_tail": tail_out,
            "stderr_tail": tail_err,
        })

    return [list_files, read_file, write_file, run_verify], stats
