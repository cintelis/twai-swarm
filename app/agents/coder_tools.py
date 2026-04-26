"""
Coder tools — the things the model can do inside the sandbox.

All tools are thin wrappers over `Sandbox`. They return **strings** because
the Anthropic tool-use API serialises tool_result content as text; any
structured shape gets JSON-encoded into a string.

Factory pattern: `build_tools(sandbox, ...)` returns `@beta_async_tool`
instances that close over `sandbox` (and optionally a Neo4j driver +
repo_name when running against a pre-indexed existing repo). We need the
factory because the tool runner's decorator introspects the function
signature to build a schema — so the sandbox can't be a parameter, it
has to be captured.

Tool surface:
  - list_files / read_file / write_file / run_verify / bash_exec
        Always available. The "edit + verify" loop.
  - repo_search / repo_find_definition / repo_find_callers
        Only added when the caller passes a Neo4j driver + repo_name —
        i.e. the Coder is working on an *existing* repo that's already
        been indexed by app.repo_indexer. These let the model navigate
        the call graph (Sprint 10c) instead of cold-reading every file.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from anthropic import beta_async_tool

from .coder_sandbox import Sandbox, SandboxError

# subprocess timeout for verify.sh — pip install + ruff + pytest on a
# small scaffold shouldn't need more than this; something is wedged if it does.
VERIFY_TIMEOUT_SECONDS = 300
# Keep the tail bounded so one chatty verify run doesn't blow the context.
VERIFY_LOG_TAIL_BYTES = 8 * 1024

# Stages templates may expose via `verify.sh <stage>`. Coder picks one to
# narrow feedback during iteration; "all" runs the full pipeline.
VERIFY_STAGES = ("lint", "typecheck", "smoke", "test", "all")


def _tail(text: str, limit: int = VERIFY_LOG_TAIL_BYTES) -> str:
    """Last `limit` bytes of `text`, with a prefix marker if truncated."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return f"[…{len(data) - limit} earlier bytes elided…]\n" + data[-limit:].decode(
        "utf-8", errors="replace"
    )


def build_tools(
    sandbox: Sandbox,
    neo4j_driver: Any = None,
    repo_name: str | None = None,
) -> tuple[list, dict]:
    """Return (tools_list, stats_dict). The stats dict is mutated in place
    by the tools as they're called, so the caller can report usage.

    Args:
        sandbox: workspace dir + safe FS / bash primitives.
        neo4j_driver: optional Neo4j driver for graph queries. Pass when
            the Coder is working on a pre-indexed repo.
        repo_name: identifies which repo's nodes to query in Neo4j (the
            same `--name` you passed to `python -m app.repo_indexer scan`).
            Required when `neo4j_driver` is set.
    """
    if neo4j_driver is not None and not repo_name:
        raise ValueError("repo_name is required when neo4j_driver is provided")

    stats = {
        "list_files_calls": 0,
        "read_file_calls": 0,
        "write_file_calls": 0,
        "run_verify_calls": 0,
        "bash_exec_calls": 0,
        "repo_search_calls": 0,
        "repo_find_definition_calls": 0,
        "repo_find_callers_calls": 0,
        "bytes_written": 0,
        "last_verify_exit": None,
        "last_verify_stdout": "",
        "last_verify_stderr": "",
        "last_verify_stage": None,
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
    async def run_verify(stage: str = "all") -> str:
        """Run the scaffold's `verify.sh` and return its output.

        The verify script is the definition of "done" for this scaffold. It
        runs in stages so you can iterate fast: lint a syntax fix without
        waiting for the full test suite. The full pipeline (`stage="all"`)
        is the gate — exit 0 there means you're done.

        Args:
            stage: One of "lint" (style/syntax), "typecheck" (mypy/tsc),
                "smoke" (entry-point sanity check), "test" (unit tests),
                "all" (full pipeline — the actual definition of done).
                Defaults to "all".

        Returns JSON with exit_code, stdout_tail, stderr_tail, stage.
        Templates that don't implement per-stage routing fall back to running
        verify.sh with no args — same as stage="all".
        """
        stage = (stage or "all").strip().lower()
        if stage not in VERIFY_STAGES:
            return f"error: stage must be one of {VERIFY_STAGES}, got {stage!r}"

        stats["run_verify_calls"] += 1
        verify_path = sandbox.root / "verify.sh"
        if not verify_path.exists():
            msg = "error: verify.sh not found in workspace — cannot verify"
            stats["last_verify_exit"] = -1
            stats["last_verify_stderr"] = msg
            return msg

        try:
            verify_path.chmod(0o755)
        except OSError:
            pass

        # Pass the stage as an arg. Templates that don't honour it just run
        # the full pipeline either way — backwards-compatible.
        cmd = ["bash", "verify.sh", stage] if stage != "all" else ["bash", "verify.sh"]

        def _run() -> tuple[int, str, str]:
            proc = subprocess.run(
                cmd,
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
            stats["last_verify_stage"] = stage
            return f"error: verify.sh ({stage}) timed out after {VERIFY_TIMEOUT_SECONDS}s"
        except FileNotFoundError:
            stats["last_verify_exit"] = -3
            stats["last_verify_stderr"] = "bash not found on PATH"
            return "error: `bash` interpreter not found on PATH"

        stats["last_verify_exit"] = exit_code
        stats["last_verify_stdout"] = stdout
        stats["last_verify_stderr"] = stderr
        stats["last_verify_stage"] = stage
        return json.dumps({
            "stage": stage,
            "exit_code": exit_code,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        })

    @beta_async_tool
    async def bash_exec(command: str, timeout_seconds: int = 60) -> str:
        """Run an arbitrary shell command in the workspace.

        Use this for things the read/write/verify tools don't cover: pip
        install, npm install, git status, running a single test, dumping a
        file's last 50 lines, etc. Runs `bash -c <command>` in the workspace
        with a scrubbed env (no provider keys, no AWS/Temporal/GitHub creds,
        no IMDS access) — so don't try to print secrets, they're not there.

        Args:
            command: The shell command to run. Wrapped in `bash -c` so pipes,
                redirects, $(...), etc. all work as expected.
            timeout_seconds: Per-call timeout. Default 60s, max 300s. Long
                installs (a fresh `npm install` of a Next.js scaffold) need
                higher; one-shot greps don't.

        Returns JSON with exit_code, stdout, stderr (each capped at 10KB
        with middle-truncation if longer).
        """
        stats["bash_exec_calls"] += 1
        try:
            exit_code, stdout, stderr = await sandbox.run_bash(command, timeout_seconds)
        except SandboxError as e:
            return f"error: {e}"
        except FileNotFoundError:
            return "error: `bash` interpreter not found on PATH"
        return json.dumps({
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        })

    tools = [list_files, read_file, write_file, run_verify, bash_exec]

    # ─── Graph tools (only when working on an indexed existing repo) ────
    if neo4j_driver is not None:
        # Imported lazily so test paths that don't touch graph tools don't
        # need the neo4j client surface to be importable.
        from app import repo_query

        @beta_async_tool
        async def repo_search(query: str, limit: int = 25) -> str:
            """Find Functions / Classes / Modules in this repo by name (fuzzy).

            Use this as the entry point when you know roughly what you're
            looking for but not the qualified name. Match is case-insensitive
            substring against the bare name (e.g. `parse_args`, `Sandbox`).

            Args:
                query: substring to look for. Short, specific names work best;
                    very common words ("get", "data") return many matches.
                limit: max results to return (1-100, default 25).

            Returns JSON: [{qualified_name, kind, file_path, line_start}, ...].
            Pass the qualified_name back into repo_find_definition or
            repo_find_callers to drill in.
            """
            stats["repo_search_calls"] += 1
            limit = max(1, min(100, int(limit)))
            results = await asyncio.to_thread(
                repo_query.find_symbol, neo4j_driver, repo_name, query, limit
            )
            return json.dumps([
                {
                    "qualified_name": r.qualified_name,
                    "kind": r.kind,
                    "file_path": r.file_path,
                    "line_start": r.line_start,
                }
                for r in results
            ])

        @beta_async_tool
        async def repo_find_definition(qualified_name: str) -> str:
            """Look up where a Function or Class is defined in this repo.

            Args:
                qualified_name: dotted path to the symbol (e.g.
                    "app.agents.coder_sandbox.Sandbox.run_bash"). Get this
                    from repo_search when you don't already know it.

            Returns JSON: {qualified_name, kind, file_path, line_start,
            line_end, docstring} — or null if the symbol isn't in the graph
            (could be external / from a third-party library).
            """
            stats["repo_find_definition_calls"] += 1
            d = await asyncio.to_thread(
                repo_query.find_definition, neo4j_driver, repo_name, qualified_name
            )
            if d is None:
                return "null"
            return json.dumps({
                "qualified_name": d.qualified_name,
                "kind": d.kind,
                "file_path": d.file_path,
                "line_start": d.line_start,
                "line_end": d.line_end,
                "docstring": d.docstring,
            })

        @beta_async_tool
        async def repo_find_callers(qualified_name: str) -> str:
            """List every function in this repo that calls `qualified_name`.

            Critical before any refactor: tells you the blast radius. If the
            list is empty, the function is unused (or only called externally
            / via dynamic dispatch we couldn't resolve). If it's long, the
            change is high-risk and likely needs broader updates.

            Args:
                qualified_name: dotted path to the callee. Same format as
                    repo_find_definition.

            Returns JSON: [{caller_qn, file_path, line}, ...] where `line`
            is the line number of the call site inside the caller's file.
            """
            stats["repo_find_callers_calls"] += 1
            sites = await asyncio.to_thread(
                repo_query.find_callers, neo4j_driver, repo_name, qualified_name
            )
            return json.dumps([
                {
                    "caller_qn": s.caller_qn,
                    "file_path": s.file_path,
                    "line": s.line,
                }
                for s in sites
            ])

        tools.extend([repo_search, repo_find_definition, repo_find_callers])

    return tools, stats
