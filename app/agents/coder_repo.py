"""Repo-aware Coder loop — Sprint 10e.

Cousin of `coder_agentic.run_agentic_coder`, but for editing an
*existing* repo instead of bootstrapping a new one. The repo has been
cloned onto disk by `clone_repo_activity` and indexed into Neo4j by
`index_repo_activity`; this loop hands the Coder both the sandbox tools
(read/write/bash) AND the graph tools (repo_search/find_definition/
find_callers) so it can navigate the codebase before editing.

Differences from `run_agentic_coder`:
  - No template selection — the repo is already there.
  - No fresh-from-scratch system prompt; the model is told its job is
    to MODIFY existing code, not generate a project.
  - Output captures `git diff` instead of a full files snapshot — for
    existing repos the deltas are what matter, not the world state.
  - Built-tool surface includes the three repo_* graph tools (Sprint 10c).

Returns:
    {
      summary: str,
      iterations: int,
      stop_reason: str,
      diff: str,                 # git diff vs commit_sha (the cloned HEAD)
      files_changed: [str, ...], # relative paths the Coder modified
      tool_calls: dict,
      _tokens_in/_tokens_out/_cost_usd/_provider/_model: as in run_agentic_coder
    }
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from app import config
from .coder_sandbox import Sandbox
from .coder_tools import build_tools

logger = logging.getLogger(__name__)

# Same caps as run_agentic_coder (Sprint 9 numbers).
MAX_ITERATIONS = 15
MAX_TOKENS_PER_TURN = 16384
CODER_MODEL = "claude-opus-4-7"

REPO_CODER_SYSTEM_PROMPT = """You are an agentic Coder working on an EXISTING repository.

The full repo is already in your sandbox. You have the standard editing tools (list_files, read_file, write_file, run_verify, bash_exec) plus three graph-aware tools backed by a pre-built code knowledge graph:

- repo_search(query) — fuzzy lookup of Function/Class/Module names. Use this when you know roughly what you're looking for but not the qualified name.
- repo_find_definition(qualified_name) — jump to where a Function or Class is defined (file_path + line_start/line_end + docstring).
- repo_find_callers(qualified_name) — list every Function in the repo that calls qualified_name. CRITICAL before any refactor: tells you the blast radius of a change.

Workflow:
1. Start with repo_search to locate the relevant area of the codebase. Don't read files at random.
2. For each candidate: repo_find_definition to see the surrounding code; repo_find_callers to understand how it's used elsewhere.
3. Plan the minimal change. Edits should be SURGICAL — modify the smallest set of files that satisfies the brief. The repo's existing patterns and style are the source of truth.
4. Apply edits with write_file (overwriting whole files) or bash_exec (sed/grep for targeted changes).
5. Use bash_exec to run focused checks: pytest tests/test_x.py, npm test specific files, ruff check on a directory.
6. When you're done, return a short final summary describing what you changed and why.

Hard rules:
- Do NOT generate new files unless the brief explicitly requires them.
- Do NOT delete or rename files unless explicitly asked.
- Preserve the repo's existing code style, framework choices, and module structure.
- Always run repo_find_callers before changing a function's signature.
- bash_exec runs in a scrubbed env. Don't try to read secrets — they're not there.
- If a change has too-broad blast radius (>5 callers), pause and reconsider; mention the risk in your summary.
"""


def _build_user_message(brief: str, repo_name: str) -> str:
    return (
        f"## Task brief\n{brief.strip()}\n\n"
        f"## Repo\nThe repository `{repo_name}` is already cloned in your workspace and its call graph is indexed.\n"
        f"Start with `list_files` to see the layout, then `repo_search` to locate the relevant code.\n"
        f"Apply the smallest change that satisfies the brief.\n"
    )


def _capture_diff(repo_root: Path) -> tuple[str, list[str]]:
    """Return (unified_diff, files_changed) using git from the repo root.

    files_changed is the list of relative paths git reports as modified
    (added/deleted/renamed all bucket into the same list — caller can
    reparse the diff for finer granularity if needed).
    """
    try:
        diff_proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        names_proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "HEAD", "--name-only"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ("", [])
    diff = diff_proc.stdout if diff_proc.returncode == 0 else ""
    files = (
        [line for line in names_proc.stdout.splitlines() if line.strip()]
        if names_proc.returncode == 0 else []
    )
    return diff, files


async def run_agentic_repo_coder(
    workflow_id: str,
    repo_root: Path,
    repo_name: str,
    brief: str,
    neo4j_driver: Any,
    heartbeat: Any = None,
    tenant_id: str = "default",
) -> dict:
    """Run the agentic Coder loop on an already-cloned + already-indexed repo.

    `neo4j_driver` is required — the graph tools (repo_search/find_definition/
    find_callers) need a connection to query against. The driver lives in
    the activity that calls this function so its lifecycle is tied to the
    activity, not the workflow.
    """
    from app import observability

    sandbox = Sandbox.wrap(repo_root)
    tools, stats = build_tools(sandbox, neo4j_driver=neo4j_driver, repo_name=repo_name)
    user_message = _build_user_message(brief, repo_name)

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
    runner = client.beta.messages.tool_runner(
        model=CODER_MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=REPO_CODER_SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": user_message}],
        thinking={"type": "adaptive"},
    )

    iterations = 0
    total_input_tokens = 0
    total_output_tokens = 0
    stop_reason: str | None = None
    last_text = ""

    tenant_ctx = observability.tenant_scope(tenant_id)
    tenant_ctx.__enter__()
    try:
        async for message in runner:
            iterations += 1
            if heartbeat is not None:
                try:
                    heartbeat(f"repo coder iteration {iterations}")
                except Exception:
                    pass
            if getattr(message, "usage", None) is not None:
                total_input_tokens += int(getattr(message.usage, "input_tokens", 0) or 0)
                total_output_tokens += int(getattr(message.usage, "output_tokens", 0) or 0)
            for block in (message.content or []):
                if getattr(block, "type", None) == "text":
                    t = getattr(block, "text", "") or ""
                    if t.strip():
                        last_text = t
            stop_reason = getattr(message, "stop_reason", None)
            if iterations >= MAX_ITERATIONS:
                logger.warning("repo coder hit MAX_ITERATIONS=%d, halting", MAX_ITERATIONS)
                break
    finally:
        tenant_ctx.__exit__(None, None, None)

    diff, files_changed = _capture_diff(repo_root)

    input_cost = total_input_tokens * 5.0 / 1_000_000
    output_cost = total_output_tokens * 25.0 / 1_000_000

    return {
        "summary": (last_text or "").strip()[:2000],
        "iterations": iterations,
        "stop_reason": stop_reason,
        "diff": diff,
        "files_changed": files_changed,
        "tool_calls": {
            "list_files":          stats["list_files_calls"],
            "read_file":           stats["read_file_calls"],
            "write_file":          stats["write_file_calls"],
            "run_verify":          stats["run_verify_calls"],
            "bash_exec":           stats["bash_exec_calls"],
            "repo_search":         stats["repo_search_calls"],
            "repo_find_definition": stats["repo_find_definition_calls"],
            "repo_find_callers":   stats["repo_find_callers_calls"],
        },
        "_tokens_in":  total_input_tokens,
        "_tokens_out": total_output_tokens,
        "_cost_usd":   round(input_cost + output_cost, 6),
        "_provider":   "anthropic",
        "_model":      CODER_MODEL,
    }
