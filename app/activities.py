"""
Activities are where all I/O happens (DB, Anthropic API).
Workflows must stay deterministic, so they can ONLY call into here.

Rule of thumb: if it touches the network, a clock, or randomness, it's an activity.
"""
import asyncio
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Optional
from temporalio import activity

from . import db, agents, telemetry


async def _embed_task_output(
    task_id: str,
    role: str,
    title: str,
    output: dict,
    tenant_id: str = "default",
) -> None:
    """Embed and store a completed task's output for later kNN retrieval.

    Best-effort: any failure (no OpenAI key, rate limit, network) logs and
    swallows. Embeddings are additive — losing one doesn't break workflows.
    """
    try:
        from app.embeddings import embed_text, task_to_embedding_text
        content = task_to_embedding_text(role, title, output)
        embedding = await embed_text(content)
        await db.upsert_task_embedding(task_id, content, embedding, tenant_id=tenant_id)
    except Exception as e:
        logger.warning(
            "embedding failed for task %s (role=%s): %s; skipping",
            task_id, role, e,
        )

logger = logging.getLogger(__name__)


async def _keep_alive(interval: float = 20.0, message: str = "calling LLM") -> None:
    """Background heartbeat so long LLM calls don't trip Temporal's timeout.

    Web search grounding on BA/architect easily takes >60s; without a
    background heartbeat the activity looks stuck even though it's making
    progress. Cancelled via asyncio when the awaited work completes.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            activity.heartbeat(message)
    except asyncio.CancelledError:
        pass

@dataclass
class AgentTaskInput:
    project_id: str
    role: str
    title: str
    description: str
    parent_task_id: Optional[str] = None
    complexity_hint: int = 1
    # Tenant identity. Threaded from ProjectInput through the workflow; every
    # DB write and LLM trace carries it. Defaults to "default" — forward-compat
    # with Temporal workflow inputs that were serialised before this field existed.
    tenant_id: str = "default"

@dataclass
class AgentTaskResult:
    task_id: str
    output: dict
    model: str
    tier: str

@activity.defn
async def create_task_record(input: AgentTaskInput) -> str:
    """Create DB row BEFORE running the agent, so it shows up in dashboards immediately."""
    return await db.create_task(
        project_id=input.project_id,
        role=input.role,
        title=input.title,
        description=input.description,
        parent_task_id=input.parent_task_id,
        tenant_id=input.tenant_id,
    )

@activity.defn
async def run_agent_activity(task_id: str, input: AgentTaskInput) -> AgentTaskResult:
    """Run one agent. Heartbeats keep Temporal from marking long LLM calls as stuck."""
    activity_started_at = time.monotonic()
    span_ctx = telemetry.span(
        "twai_swarm.activity.run_agent",
        **{
            "swarm.role": input.role,
            "swarm.task_id": task_id,
            "swarm.project_id": input.project_id,
            "swarm.tenant_id": input.tenant_id,
        },
    )
    span_ctx.__enter__()
    await db.update_task_running(task_id)

    # get_context_for_task = parent-walk ancestors + (for synthesis roles)
    # kNN over similar prior task outputs in the same project. Each entry
    # is tagged with _source = "ancestor" | "similar" so the prompt can
    # frame them differently. Falls back to ancestors-only on embedding failure.
    context = await db.get_context_for_task(task_id)

    # Fire an initial heartbeat, then hand it off to a background task that
    # keeps pinging Temporal every 20s while we're awaiting the LLM.
    activity.heartbeat("calling LLM")
    hb_task = asyncio.create_task(_keep_alive())

    try:
        result = await agents.run_agent(
            role=input.role,
            task_description=input.description,
            context=context,
            complexity_hint=input.complexity_hint,
            tenant_id=input.tenant_id,
        )
    except Exception as e:
        logger.error(
            "run_agent_activity failed (task_id=%s): %s\n%s",
            task_id, e, traceback.format_exc(),
        )
        await db.fail_task(task_id, str(e))
        raise
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    # Sidecar grounding metadata (web_search citations etc.) lands inside the
    # output JSON under `_citations` so it's queryable without a schema change
    # or sidecar table. UI strips it when pretty-printing the agent's payload.
    output_to_persist = result["output"]
    citations = result.get("citations") or []
    if isinstance(output_to_persist, dict) and citations:
        output_to_persist = {**output_to_persist, "_citations": citations}

    await db.complete_task(
        task_id,
        output=output_to_persist,
        provider=result["provider"],
        model=result["model"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        cost_usd=result["cost_usd"],
    )

    # Embed the task output for downstream kNN retrieval. Best-effort —
    # logged and swallowed on failure so embedding cost doesn't fail the workflow.
    await _embed_task_output(task_id, input.role, input.title, result["output"], tenant_id=input.tenant_id)

    # Record activity duration + close the span we opened above.
    telemetry.histogram_record(
        "agent_activity_duration",
        time.monotonic() - activity_started_at,
        role=input.role,
        tenant_id=input.tenant_id,
    )
    span_ctx.__exit__(None, None, None)

    return AgentTaskResult(
        task_id=task_id,
        output=result["output"],
        model=result["model"],
        tier=result["model_key"],
    )

@activity.defn
async def create_project_record(
    name: str,
    brief: str,
    workflow_id: str,
    tenant_id: str = "default",
) -> str:
    return await db.create_project(name, brief, workflow_id, tenant_id=tenant_id)


@activity.defn
async def run_coder_activity(task_id: str, input: AgentTaskInput, workflow_id: str) -> AgentTaskResult:
    """Run the agentic Coder: tool-using loop over Claude Opus 4.7 in a
    per-workflow sandbox. Produces the same `{files: [...]}` shape as the
    one-shot coder so /download keeps working.

    On any exception, falls back to the one-shot coder so the workflow
    still ships something. The fallback path is logged at ERROR.
    """
    from . import config
    from .agents import coder_agentic

    activity_started_at = time.monotonic()
    span_ctx = telemetry.span(
        "twai_swarm.activity.run_coder",
        **{
            "swarm.role": "coder",
            "swarm.task_id": task_id,
            "swarm.project_id": input.project_id,
            "swarm.workflow_id": workflow_id,
            "swarm.tenant_id": input.tenant_id,
        },
    )
    span_ctx.__enter__()
    await db.update_task_running(task_id)
    context = await db.get_ancestor_outputs(task_id)

    # Pull architecture / SE plan / documenter output out of the context list.
    def _find(role: str) -> dict | None:
        for c in context:
            if c.get("role") == role:
                out = c.get("output")
                return out if isinstance(out, dict) else None
        return None

    architecture = _find("architect")
    se_plan = _find("se")
    documenter = _find("documenter")

    activity.heartbeat("coder starting")
    hb_task = asyncio.create_task(_keep_alive(message="coder running"))

    try:
        if config.CODER_MODE == "oneshot":
            # Explicit opt-out — use the old path directly.
            result = await agents.run_agent(
                role="coder",
                task_description=input.description,
                context=context,
                complexity_hint=input.complexity_hint,
                tenant_id=input.tenant_id,
            )
        else:
            try:
                coder_out = await coder_agentic.run_agentic_coder(
                    workflow_id=workflow_id,
                    brief=input.description,
                    architecture=architecture,
                    se_plan=se_plan,
                    documenter=documenter,
                    heartbeat=activity.heartbeat,
                    tenant_id=input.tenant_id,
                )
                # Repackage into the shape run_agent returns so the DB
                # write + return dataclass below don't need a branch.
                result = {
                    "output": {k: v for k, v in coder_out.items() if not k.startswith("_")},
                    "provider": coder_out["_provider"],
                    "model": coder_out["_model"],
                    "model_key": "anthropic/claude-haiku-4-5",
                    "route_reason": "coder hardcoded to Haiku 4.5 (agentic)",
                    "tokens_in": coder_out["_tokens_in"],
                    "tokens_out": coder_out["_tokens_out"],
                    "cost_usd": coder_out["_cost_usd"],
                    "citations": [],
                }
            except Exception as e:
                logger.error(
                    "agentic coder failed (task_id=%s), falling back to oneshot: %s\n%s",
                    task_id, e, traceback.format_exc(),
                )
                result = await agents.run_agent(
                    role="coder",
                    task_description=input.description,
                    context=context,
                    complexity_hint=input.complexity_hint,
                )
    except Exception as e:
        logger.error(
            "run_coder_activity failed (task_id=%s): %s\n%s",
            task_id, e, traceback.format_exc(),
        )
        await db.fail_task(task_id, str(e))
        raise
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    output_to_persist = result["output"]
    citations = result.get("citations") or []
    if isinstance(output_to_persist, dict) and citations:
        output_to_persist = {**output_to_persist, "_citations": citations}

    await db.complete_task(
        task_id,
        output=output_to_persist,
        provider=result["provider"],
        model=result["model"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        cost_usd=result["cost_usd"],
    )

    # Embed the Coder output too — useful as kNN context for future projects'
    # SE/Reviewer/Documenter ("we've coded something like this before").
    # The embedding text strips the files[] array (see task_to_embedding_text).
    await _embed_task_output(task_id, input.role, input.title, result["output"], tenant_id=input.tenant_id)

    # Coder-specific metrics + close span
    iters = result.get("output", {}).get("iterations") if isinstance(result.get("output"), dict) else None
    if isinstance(iters, int):
        verify_passed = bool(result["output"].get("verify_passed", False))
        telemetry.histogram_record(
            "coder_iterations", float(iters),
            verify_passed=verify_passed,
            tenant_id=input.tenant_id,
        )
    telemetry.histogram_record(
        "agent_activity_duration",
        time.monotonic() - activity_started_at,
        role="coder",
        tenant_id=input.tenant_id,
    )
    span_ctx.__exit__(None, None, None)

    return AgentTaskResult(
        task_id=task_id,
        output=result["output"],
        model=result["model"],
        tier=result["model_key"],
    )



# ─── Sprint 10e: RepoTaskWorkflow activities ─────────────────────────────────
# Three activities glued together by app.workflows.repo_task.RepoTaskWorkflow:
#   clone_repo_activity       — git clone --depth 1 to /tmp/repo-tasks/<wf-id>
#   index_repo_activity       — runs the repo_indexer over the cloned tree
#   run_repo_coder_activity   — agentic Coder loop with Sprint 10c graph tools
#
# Each activity is a single Temporal-friendly entry point: input is a small
# typed payload (or primitives), output is JSON-serialisable.

@activity.defn
async def clone_repo_activity(
    repo_url: str,
    branch: str,
    workflow_id: str,
) -> dict:
    """Shallow-clone `repo_url` at `branch` into /tmp/repo-tasks/<workflow_id>.

    Returns {"path": "<absolute path>", "commit_sha": "<40-char>"}.

    v1: HTTPS-only, no auth (public repos OR url with embedded token).
    Sprint 10f layers GitHub App token injection for private repos.
    """
    import shutil
    from pathlib import Path

    safe_id = "".join(c for c in workflow_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        raise ValueError(f"workflow_id {workflow_id!r} produced empty safe path")
    dest = Path("/tmp/repo-tasks") / safe_id
    if dest.exists():
        # Activity retry — start fresh. Some platforms refuse `git clone`
        # into an already-existing dir even when it's empty, so we let
        # git create dest itself.
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    activity.heartbeat("cloning")
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth", "1", "--branch", branch, repo_url, str(dest),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git clone failed (rc={proc.returncode}): {stderr.decode('utf-8', 'replace')[:500]}"
        )

    sha_proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(dest), "rev-parse", "HEAD",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    sha_out, _ = await sha_proc.communicate()
    commit_sha = sha_out.decode("utf-8").strip()
    if not commit_sha:
        raise RuntimeError("git rev-parse HEAD returned empty")

    return {"path": str(dest), "commit_sha": commit_sha}


@activity.defn
async def index_repo_activity(
    repo_path: str,
    repo_name: str,
    commit_sha: str,
    tenant_id: str = "default",
    force_reindex: bool = False,
) -> dict:
    """Run the repo indexer over `repo_path` and write to Neo4j.

    Returns the IndexBatch counts dict so the workflow output can include
    "we indexed X files / Y functions before the Coder ran."

    `force_reindex=True` bypasses the per-file SHA short-circuit so files
    indexed by a prior extractor version get re-extracted (e.g. after
    Sprint 17 added Java support, JS/TS files cached from earlier scans
    needed re-extraction to pick up new edges).
    """
    from pathlib import Path
    from app.repo_indexer.actions import IndexBatch, RepoNode
    from app.repo_indexer.loader import (
        driver_from_env, ensure_constraints, prune_stale, write_batch,
    )
    from app.repo_indexer.phases import DEFAULT_PHASES
    from app.repo_indexer.runner import PhaseContext, run_pipeline
    import tree_sitter_python as tspython
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser

    # Sprint 17 — best-effort cpp/java parsers. Mirrors the `_pool_init`
    # pattern in phases/parse.py: if the grammar isn't installed, leave
    # the parser as None and the parse phase logs+skips those files.
    try:
        import tree_sitter_cpp as tscpp
    except Exception:  # noqa: BLE001
        tscpp = None
    try:
        import tree_sitter_java as tsjava
    except Exception:  # noqa: BLE001
        tsjava = None

    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo_path {repo_root} is not a directory")

    repo = RepoNode(name=repo_name, url="", commit_sha=commit_sha, tenant_id=tenant_id)
    py_parser = Parser(Language(tspython.language()))
    ts_parsers = {
        "typescript": Parser(Language(tsts.language_typescript())),
        "tsx":        Parser(Language(tsts.language_tsx())),
    }
    try:
        cpp_parser = Parser(Language(tscpp.language())) if tscpp is not None else None
    except Exception:  # noqa: BLE001
        cpp_parser = None
    try:
        java_parser = Parser(Language(tsjava.language())) if tsjava is not None else None
    except Exception:  # noqa: BLE001
        java_parser = None

    aggregate = IndexBatch(repo=repo)

    # Route phase milestones through Temporal heartbeats so a slow scan
    # doesn't time out. The progress callback fires once per phase boundary
    # (scan / parse / resolve / community_detect / process_extract); the
    # per-200-file rate prints in ParsePhase still go to stdout via raw
    # `print` — they're not captured here.
    def _progress(msg: str) -> None:
        activity.heartbeat(msg)
        print(msg, flush=True)

    with driver_from_env() as driver:
        ensure_constraints(driver)
        prune_stale(driver, repo_name, commit_sha)
        ctx = PhaseContext(
            repo=repo,
            repo_root=repo_root,
            # Java + cpp added in Sprint 17 — walker filters by this list,
            # so omitting them silently drops every .java/.cpp file from
            # the parse pipeline. (That's how we shipped a "Java extractor"
            # that never fired in the Temporal activity path.)
            languages=("python", "typescript", "javascript", "cpp", "java"),
            batch=aggregate,
            py_parser=py_parser,
            ts_parsers=ts_parsers,
            cpp_parser=cpp_parser,
            java_parser=java_parser,
            progress=_progress,
            driver=driver,
            # Sequential parsing inside a Temporal activity. multiprocessing.Pool
            # from inside an activity has cross-platform sharp edges (Windows
            # spawn re-imports the workflow context, fork on Linux can deadlock
            # with Temporal's worker threads). Re-evaluate if scan time on a
            # 13K-file repo becomes a real Coder UX problem.
            parse_workers=1,
            # Embeddings are opt-in even for Temporal scans. Activity callers
            # that want them set this flag via a separate mechanism (env var
            # or workflow input field) — out of scope for this refactor.
            embed_enabled=False,
            # Routes are universally cheap to extract (a handful of decorator
            # / method-call patterns per file) so we always turn them on
            # for repo-task scans. Affects FastAPI/Flask/Express (Sprint 15a)
            # and Spring (Sprint 17f).
            extract_routes=True,
            # Sprint 17 post-deploy fix: opt-in cache bust for callers that
            # need to re-extract files whose on-disk SHA hasn't changed
            # (e.g. extractor-version bump). Default False keeps incremental
            # scans fast.
            force_reindex=force_reindex,
        )
        run_pipeline(ctx, DEFAULT_PHASES)
        activity.heartbeat("writing to Neo4j")
        write_batch(driver, aggregate)

    file_count = len(aggregate.files)
    return {"file_count": file_count, **aggregate.counts()}


@activity.defn
async def architect_repo_task_activity(
    repo_path: str,
    repo_name: str,
    brief: str,
    tenant_id: str,
    workflow_id: str,
) -> dict:
    """Run the planning Architect against the cloned + indexed repo.

    Sprint 18b: pre-step before `run_repo_coder_activity`. Mirrors the
    Coder activity's shape (own Neo4j driver, recon-block construction,
    heartbeat keep-alive) but invokes `run_architect_repo` instead — the
    Architect is a pure planner with the graph tools but no write tools.

    Returns `dataclasses.asdict(ArchitectRepoOutput)`. The workflow then
    threads this dict into `run_repo_coder_activity` as
    `architect_plan: dict | None` so the Coder can render its narrative +
    subtasks + acceptance_criteria above the brief (per D1).

    Failure mode: the Architect call wraps its own tool-runner exceptions
    and returns a degraded ArchitectRepoOutput rather than raising, so a
    single bad turn doesn't fail the whole workflow — the Coder falls
    through to a no-plan run.
    """
    from pathlib import Path
    from app.agents.architect_repo import (
        architect_output_to_dict, run_architect_repo,
    )
    from app.agents.coder_repo import (
        _RECON_MODULE_CAP, _RECON_PROCESS_CAP, _format_recon_block,
    )
    from app.repo_indexer.loader import driver_from_env

    activity.heartbeat("architect starting")
    hb_task = asyncio.create_task(_keep_alive(message="architect running"))
    try:
        with driver_from_env() as driver:
            # Build the recon block here (workflow-side) so the Architect
            # gets the same panoramic map the Coder would. Keeps both
            # roles looking at the same evidence — important if Critic
            # later wants to compare what the Architect saw vs what the
            # Coder did.
            recon_block = ""
            try:
                from app import repo_query
                modules = await asyncio.to_thread(
                    repo_query.find_modules,
                    driver, repo_name, _RECON_MODULE_CAP, False,
                )
                processes = await asyncio.to_thread(
                    repo_query.find_processes,
                    driver, repo_name, None, _RECON_PROCESS_CAP, False,
                )
                recon_block = _format_recon_block(modules, processes)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "architect recon queries failed, proceeding without "
                    "recon block: %s", e,
                )

            arch_out = await run_architect_repo(
                workflow_id=workflow_id,
                repo_root=Path(repo_path),
                repo_name=repo_name,
                brief=brief,
                neo4j_driver=driver,
                recon_block=recon_block,
                heartbeat=activity.heartbeat,
                tenant_id=tenant_id,
            )
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
    return architect_output_to_dict(arch_out)


@activity.defn
async def critic_repo_task_activity(
    repo_path: str,
    repo_name: str,
    architect_plan: dict | None,
    coder_diff: str,
    files_with_content: list[dict],
    tenant_id: str,
    workflow_id: str,
) -> dict:
    """Validate the Coder's diff against the Architect's checklist.

    Sprint 18c: post-step after `run_repo_coder_activity`. Runs a
    deterministic gate (ruff / compileall / mvn / npm — best-effort,
    silent skip on absent tooling) and an LLM-judge stage that grades
    each `acceptance_criterion` from the Architect plan against the
    diff. If any block-severity criterion fails OR the deterministic
    gate flags errors, the result includes a structured
    `continuation_prompt` (D7 handoff doc) for the workflow's continuation
    loop.

    Returns `dataclasses.asdict(CriticRepoOutput)`. Failure modes (no
    plan, judge API errors) all surface as a `CriticRepoOutput` rather
    than an activity-level exception — the workflow keeps moving even
    if the Critic stage degrades.
    """
    from pathlib import Path
    from app.agents.critic_repo import critic_output_to_dict, run_critic_repo

    # Tenant_id and repo_name are accepted for symmetry with the other
    # repo-task activities but the Critic doesn't need a Neo4j driver
    # (gates run on the disk; the LLM judge gets the plan + diff
    # directly). Keeping them in the signature keeps the workflow's
    # call site uniform with the Architect/Coder activities.
    _ = tenant_id, repo_name, workflow_id

    activity.heartbeat("critic starting")
    hb_task = asyncio.create_task(_keep_alive(message="critic running"))
    try:
        result = await run_critic_repo(
            architect_plan=architect_plan,
            coder_diff=coder_diff,
            files_with_content=files_with_content,
            repo_root=Path(repo_path),
        )
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
    return critic_output_to_dict(result)


@activity.defn
async def run_repo_coder_activity(
    repo_path: str,
    repo_name: str,
    brief: str,
    tenant_id: str,
    workflow_id: str,
    architect_plan: dict | None = None,
) -> dict:
    """Run the agentic Coder against the cloned + indexed repo.

    Opens its own Neo4j driver — workflow-side I/O is forbidden by
    Temporal, so the driver lifecycle is tied to this activity.

    Sprint 18b: `architect_plan` is the dict produced by
    `architect_repo_task_activity`. Threaded through to
    `run_agentic_repo_coder` which renders it as a "## Architect plan"
    section in the Coder's user message. Default None preserves the
    pre-18b call shape — older callers / replayed histories still work.

    Sprint 18c: when the workflow's continuation loop fires, it passes
    the Critic's structured handoff doc (D7) AS the `brief` parameter.
    No code change is needed in this activity — `_build_user_message`
    already renders the brief verbatim, and the handoff doc is just
    a markdown string with extra sections. The Architect plan is
    threaded through unchanged so the continuation Coder still sees
    the original acceptance_criteria, not just the gap list.
    """
    from pathlib import Path
    from app.agents.coder_repo import run_agentic_repo_coder
    from app.repo_indexer.loader import driver_from_env

    activity.heartbeat("repo coder starting")
    hb_task = asyncio.create_task(_keep_alive(message="repo coder running"))
    try:
        with driver_from_env() as driver:
            result = await run_agentic_repo_coder(
                workflow_id=workflow_id,
                repo_root=Path(repo_path),
                repo_name=repo_name,
                brief=brief,
                neo4j_driver=driver,
                heartbeat=activity.heartbeat,
                tenant_id=tenant_id,
                architect_plan=architect_plan,
            )
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
    return result


@activity.defn
async def push_repo_changes_activity(
    repo_url: str,
    files: list[dict],
    workflow_id: str,
    brief: str,
    tenant_id: str,
) -> dict:
    """Push the Coder's edits as a new branch + PR via the GitHub App.

    Best-effort: returns {"pr_url": None, "error": "..."} for graceful
    failure modes (no installation has access, repo not found, push API
    rejects). Hard failures (network, auth) propagate so Temporal can
    surface them in the workflow result.

    The branch name is derived from the workflow_id so re-runs collide
    cleanly (push_files_as_branch force-updates the ref). PR title +
    body include the brief so a reviewer can see intent without
    leaving the PR.
    """
    import re
    from app import db, github_app

    if not files:
        return {"pr_url": None, "branch_name": None, "error": "no files to push"}

    # https://github.com/<owner>/<name>(.git)?
    m = re.match(r"https?://github\.com/([^/]+)/([^/.]+?)(?:\.git)?/?$", repo_url)
    if not m:
        return {"pr_url": None, "branch_name": None, "error": f"could not parse repo_url: {repo_url}"}
    owner, name = m.group(1), m.group(2)

    # Find which installation can see this repo. The user can have multiple
    # installations connected (across orgs); we pick the first one whose
    # repo list includes the target. iter on demand — repo lists can be
    # large for prolific orgs, so don't materialise all of them.
    installations = await db.get_github_installations(tenant_id=tenant_id)
    matched_inst_id: int | None = None
    for inst in installations:
        try:
            repos = await github_app.list_installation_repos(inst["installation_id"])
        except Exception as e:
            activity.logger.warning(
                "list_installation_repos failed for installation %s: %s",
                inst["installation_id"], e,
            )
            continue
        for r in repos:
            if r["owner"].lower() == owner.lower() and r["name"].lower() == name.lower():
                matched_inst_id = inst["installation_id"]
                break
        if matched_inst_id is not None:
            break

    if matched_inst_id is None:
        return {
            "pr_url": None, "branch_name": None,
            "error": f"no GitHub App installation has access to {owner}/{name}",
        }

    short = workflow_id.removeprefix("repo-task-")[:8]
    branch_name = f"swarm/{short}"
    one_line = " ".join(brief.split())
    pr_title = f"Swarm: {one_line[:80]}{'…' if len(one_line) > 80 else ''}"
    pr_body = (
        f"Automated change from twai-swarm workflow `{workflow_id}`.\n\n"
        f"## Brief\n\n{brief}\n\n"
        f"## Files changed ({len(files)})\n\n"
        + "\n".join(f"- `{f['path']}`" for f in files)
    )
    commit_message = f"swarm: {one_line[:72]}"

    push = await github_app.push_files_as_branch(
        installation_id=matched_inst_id,
        repo_owner=owner,
        repo_name=name,
        branch=branch_name,
        files=files,
        commit_message=commit_message,
        open_pr=True,
        pr_title=pr_title,
        pr_body=pr_body,
    )
    return {
        "pr_url": getattr(push, "pr_url", None),
        "pr_number": getattr(push, "pr_number", None),
        "branch_name": branch_name,
        "installation_id": matched_inst_id,
    }
