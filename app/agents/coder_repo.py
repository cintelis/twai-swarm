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

# Iteration cap raised from 15 → 30 in Sprint 18a after the "Refresh Tokens"
# repo-task brief truncated mid-task (PR #8 closed unmerged): the Coder ran
# out of turns while still drafting tests. Repo-aware briefs do more work
# per turn than greenfield (graph lookups + read + plan + edit + verify),
# so 15 is too tight. Greenfield Coder (`coder_agentic.py`) keeps 15 because
# template-customisation finishes well under that ceiling in practice.
MAX_ITERATIONS = 30
MAX_TOKENS_PER_TURN = 16384
# See coder_agentic.CODER_MODEL — same Anthropic-SDK constraint applies here.
CODER_MODEL = "claude-haiku-4-5"

REPO_CODER_SYSTEM_PROMPT = """You are an agentic Coder working on an EXISTING repository.

The full repo is already in your sandbox. You have the standard editing tools (list_files, read_file, write_file, run_verify, bash_exec) plus graph-aware tools backed by a pre-built code knowledge graph:

- repo_semantic_search(query) — hybrid BM25 + embedding search over Functions and Classes. Use this when the brief is conceptual ("the auth flow", "where do we handle errors", "config loading") and you don't yet know the symbol names. This is the right DEFAULT for unfamiliar repos; it's fuzzier than repo_search but understands intent.
- repo_search(query) — fuzzy lookup of Function/Class/Module names by exact substring match. Use this when you already know (or can guess) the symbol name; faster and more precise than repo_semantic_search for that case.
- repo_find_definition(qualified_name) — jump to where a Function or Class is defined (file_path + line_start/line_end + docstring).
- repo_find_callers(qualified_name) — list every Function in the repo that calls qualified_name. CRITICAL before any refactor: tells you the blast radius of a change.
- repo_find_processes(query) — list execution flows (chains of calls that cross module boundaries — e.g. workflow runs, CLI commands, API handlers). Use this when the brief is high-level ("how does X work?") and you want to trace the path before reading code.
- repo_find_modules() — list the major modules (community clusters) of the codebase. Use this on first contact with an unfamiliar repo to learn its shape; then drill in with repo_search on a cluster's sample members.

Workflow:
1. If the brief is high-level or you're unfamiliar with this repo, start with repo_find_modules / repo_find_processes / repo_semantic_search to map the territory. Otherwise jump to repo_search. When the user message starts with a "Repo recon" block, the modules and processes are pre-loaded — use them as your map. You may still drill in with `repo_find_callers` / `repo_find_definition` on specific symbols.
2. Use repo_search to locate the relevant area of the codebase. Don't read files at random.
3. For each candidate: repo_find_definition to see the surrounding code; repo_find_callers to understand how it's used elsewhere.
4. Plan the minimal change. Edits should be SURGICAL — modify the smallest set of files that satisfies the brief. The repo's existing patterns and style are the source of truth.
5. Apply edits with write_file (overwriting whole files) or bash_exec (sed/grep for targeted changes).
6. Use bash_exec to run focused checks: pytest tests/test_x.py, npm test specific files, ruff check on a directory.
7. When you're done, return a short final summary describing what you changed and why.

Hard rules:
- Do NOT generate new files unless the brief explicitly requires them.
- Do NOT delete or rename files unless explicitly asked.
- Preserve the repo's existing code style, framework choices, and module structure.
- Always run repo_find_callers before changing a function's signature.
- bash_exec runs in a scrubbed env. Don't try to read secrets — they're not there.
- If a change has too-broad blast radius (>5 callers), pause and reconsider; mention the risk in your summary.

Budget awareness: You operate inside a fixed iteration budget of 30 turns (raised from 15 in Sprint 18a). Each turn is one model response that may include multiple tool calls. Track your own progress against the brief: if you're at iteration 20+ and haven't started writing tests yet, prioritize tests over additional refactoring. A 5-step refactor at iteration 28 will not finish. If you receive a heartbeat or operator note indicating "completion mode" (typically around iteration 24, i.e. 80% of the budget), stop opening new lines of work — instead, list the brief asks NOT yet addressed and complete only those before returning your final summary.

Architect plan handling (Sprint 18b): When an "## Architect plan" section is present in your user message, treat the listed `acceptance_criteria` as a CONTRACT. The Architect already investigated the repo and decided what "done" looks like — your job is to satisfy every acceptance_criterion, not to re-litigate scope. If you complete the brief but skip an acceptance_criterion, that is a failure. If you genuinely disagree with the plan (it misses a file, mis-identifies a pattern, picks the wrong abstraction), surface the disagreement EXPLICITLY in your final summary — do NOT silently expand or contract scope. The "## Risks" section enumerates blast-radius warnings the Architect already flagged; respect them (especially "don't change this signature" notes).
"""


# Recon caps — keep total budget under ~1500 tokens of output. Modules cap
# (15) is the dominant lever; samples-per-module (3) trims wide clusters.
_RECON_MODULE_CAP = 15
_RECON_PROCESS_CAP = 10
_RECON_SAMPLES_PER_MODULE = 3


def _format_recon_block(modules: list, processes: list) -> str:
    """Render a 'Repo recon' markdown block for the user message.

    `modules` is a list of objects with `.label`, `.size`, `.sample_member_qns`
    (i.e. `ModuleSummary` from `repo_query`, but any duck-typed shape works
    for tests). `processes` is a list of objects with `.name`, `.step_count`,
    `.member_qns`.

    Empty inputs return ""; callers detect the empty string and skip injection
    so the user message doesn't get a meaningless header on small repos.
    """
    if not modules and not processes:
        return ""

    lines: list[str] = ["## Repo recon (auto-generated)"]

    if modules:
        shown = list(modules)[:_RECON_MODULE_CAP]
        lines.append("")
        lines.append(f"### Modules ({len(shown)})")
        for m in shown:
            samples = list(getattr(m, "sample_member_qns", []) or [])[:_RECON_SAMPLES_PER_MODULE]
            sample_str = ", ".join(samples) if samples else "(no sample members)"
            lines.append(f"- `{m.label}` ({m.size} symbols): {sample_str}")
        if len(modules) > _RECON_MODULE_CAP:
            lines.append(f"- …and {len(modules) - _RECON_MODULE_CAP} more")

    if processes:
        shown_p = list(processes)[:_RECON_PROCESS_CAP]
        lines.append("")
        lines.append(f"### Top processes ({len(shown_p)})")
        for p in shown_p:
            members = list(p.member_qns or [])
            if not members:
                chain = "(empty)"
            elif len(members) == 1:
                chain = members[0]
            elif len(members) == 2:
                chain = f"{members[0]} → {members[1]}"
            else:
                chain = f"{members[0]} → … → {members[-1]}"
            lines.append(f"- `{p.name}` ({p.step_count} steps): {chain}")
        if len(processes) > _RECON_PROCESS_CAP:
            lines.append(f"- …and {len(processes) - _RECON_PROCESS_CAP} more")

    lines.append("")
    lines.append(
        "_Use `repo_find_modules` / `repo_find_processes` for full detail, "
        "or `repo_search`/`repo_find_callers` to drill in._"
    )
    return "\n".join(lines)


def _build_user_message(
    brief: str,
    repo_name: str,
    recon_block: str = "",
    architect_plan: dict | None = None,
) -> str:
    """Build the Coder's user-side message.

    Sprint 18b: when an Architect plan is provided, prepend a structured
    "## Architect plan" section + a "## Risks" section above the brief.
    Per D1 the plan's `acceptance_criteria` are the contract the Coder
    is judged against; rendering them prominently gives the Coder a
    visible checklist instead of a buried hint.

    Section ordering:  recon → architect plan → risks → brief → repo
    The model reads top-down; recon is panoramic, plan is the scope, the
    brief is the ground-truth ask. Brief stays even when a plan is
    present so the model can flag plan/brief disagreement explicitly
    (per the system prompt's "surface the disagreement" clause).
    """
    parts: list[str] = []
    if recon_block:
        parts.append(recon_block)
    if architect_plan:
        # Lazy import — keeps coder_repo importable without architect_repo
        # in environments that only need the legacy single-Coder path.
        from .architect_repo import render_architect_plan_section, render_risk_section
        plan_section = render_architect_plan_section(architect_plan)
        if plan_section:
            parts.append(plan_section)
        risks = render_risk_section(architect_plan)
        if risks:
            parts.append(risks)
    parts.append(f"## Task brief\n{brief.strip()}")
    parts.append(
        f"## Repo\nThe repository `{repo_name}` is already cloned in your workspace and its call graph is indexed.\n"
        f"Start with `list_files` to see the layout, then `repo_search` to locate the relevant code.\n"
        f"Apply the smallest change that satisfies the brief.\n"
    )
    return "\n\n".join(parts)


def _capture_diff(repo_root: Path) -> tuple[str, list[str]]:
    """Return (unified_diff, files_changed) using git from the repo root.

    files_changed is the list of relative paths git reports as modified
    (added/deleted/renamed all bucket into the same list — caller can
    reparse the diff for finer granularity if needed).

    Untracked new files are promoted to intent-to-add (`git add -N`) before
    diffing so brand-new files the Coder created appear in both the unified
    diff and the name list. Without this, `git diff HEAD` silently omits
    untracked paths and the push activity ships an incomplete PR — only
    the wiring edits, not the new classes/tests they reference.
    """
    try:
        untracked_proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30,
        )
        untracked = [p for p in untracked_proc.stdout.splitlines() if p.strip()]
        if untracked:
            subprocess.run(
                ["git", "-C", str(repo_root), "add", "-N", "--", *untracked],
                capture_output=True, text=True, timeout=30,
            )
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
    architect_plan: dict | None = None,
) -> dict:
    """Run the agentic Coder loop on an already-cloned + already-indexed repo.

    `neo4j_driver` is required — the graph tools (repo_search/find_definition/
    find_callers) need a connection to query against. The driver lives in
    the activity that calls this function so its lifecycle is tied to the
    activity, not the workflow.

    `architect_plan` (Sprint 18b) is the dict-shaped output of
    `app.agents.architect_repo.run_architect_repo` — when provided, it's
    rendered into the user message above the brief and the Coder treats
    its `acceptance_criteria` as a contract (per D1). Default None
    preserves the pre-18b single-Coder behaviour for callers that haven't
    been updated.
    """
    from app import observability

    sandbox = Sandbox.wrap(repo_root)
    tools, stats = build_tools(sandbox, neo4j_driver=neo4j_driver, repo_name=repo_name)

    # Pre-seed the user message with a panoramic recon block so the Coder
    # has the modules + processes in context without needing to choose to
    # call the graph tools first. Highly-specific briefs (which name a
    # class/algorithm) used to skip these tools entirely; injecting the
    # output makes the choice "use this map" rather than "decide whether
    # to fetch a map." Recon is best-effort — Neo4j hiccups don't fail
    # the activity.
    recon_block = ""
    try:
        from app import repo_query
        modules = await asyncio.to_thread(
            repo_query.find_modules, neo4j_driver, repo_name, _RECON_MODULE_CAP, False,
        )
        processes = await asyncio.to_thread(
            repo_query.find_processes, neo4j_driver, repo_name, None, _RECON_PROCESS_CAP, False,
        )
        recon_block = _format_recon_block(modules, processes)
    except Exception as e:  # noqa: BLE001 — best-effort; log and proceed.
        logger.warning("repo recon queries failed, proceeding without recon block: %s", e)

    user_message = _build_user_message(
        brief, repo_name,
        recon_block=recon_block,
        architect_plan=architect_plan,
    )

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
    runner_kwargs: dict = dict(
        model=CODER_MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=REPO_CODER_SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": user_message}],
    )
    # See coder_agentic for the rationale — adaptive thinking is Opus-only.
    if CODER_MODEL.startswith("claude-opus-"):
        runner_kwargs["thinking"] = {"type": "adaptive"}
    runner = client.beta.messages.tool_runner(**runner_kwargs)

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
            # Sprint 18a: at 80% of MAX_ITERATIONS, surface a "completion mode"
            # banner via the heartbeat so operators have visibility when the
            # Coder is approaching the cap. Mid-stream injection of a Coder-
            # facing message isn't supported by the Anthropic SDK's
            # `tool_runner` (it manages its own message history end-to-end);
            # the system prompt now declares the budget up front instead. If
            # the SDK later exposes a per-turn injection hook, swap this
            # heartbeat-only signal for an actual system-side note.
            completion_mode = iterations >= int(MAX_ITERATIONS * 0.8)
            if heartbeat is not None:
                try:
                    if completion_mode:
                        heartbeat(
                            f"repo coder iteration {iterations} — completion mode "
                            f"({iterations}/{MAX_ITERATIONS}, "
                            f"{MAX_ITERATIONS - iterations} turns left)"
                        )
                    else:
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

    # Capture the post-Coder content of each modified file so the workflow's
    # push activity can open a PR without needing same-worker disk access.
    # Binary or non-UTF-8 files are skipped with a logged warning — the PR
    # will be incomplete but the rest of the change still lands.
    # Deleted paths (in files_changed but not on disk) are likewise skipped;
    # push_files_as_branch only adds/modifies, so deletions don't round-trip
    # through this MVP path.
    files_with_content: list[dict] = []
    for rel in files_changed:
        full = repo_root / rel
        if not full.is_file():
            logger.warning("skipping %s in push payload: not a file (deleted?)", rel)
            continue
        try:
            content = full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("skipping %s in push payload: %s", rel, e)
            continue
        files_with_content.append({"path": rel, "content": content})

    input_cost = total_input_tokens * 5.0 / 1_000_000
    output_cost = total_output_tokens * 25.0 / 1_000_000

    return {
        "summary": (last_text or "").strip()[:2000],
        "iterations": iterations,
        "stop_reason": stop_reason,
        "diff": diff,
        "files_changed": files_changed,
        "files_with_content": files_with_content,
        "tool_calls": {
            "list_files":          stats["list_files_calls"],
            "read_file":           stats["read_file_calls"],
            "write_file":          stats["write_file_calls"],
            "run_verify":          stats["run_verify_calls"],
            "bash_exec":           stats["bash_exec_calls"],
            "repo_search":         stats["repo_search_calls"],
            "repo_find_definition": stats["repo_find_definition_calls"],
            "repo_find_callers":   stats["repo_find_callers_calls"],
            "repo_find_processes": stats["repo_find_processes_calls"],
            "repo_find_modules":   stats["repo_find_modules_calls"],
            "repo_semantic_search": stats["repo_semantic_search_calls"],
        },
        "_tokens_in":  total_input_tokens,
        "_tokens_out": total_output_tokens,
        "_cost_usd":   round(input_cost + output_cost, 6),
        "_provider":   "anthropic",
        "_model":      CODER_MODEL,
    }
