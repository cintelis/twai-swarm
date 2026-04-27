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
                    "model_key": "anthropic/claude-opus-4-7",
                    "route_reason": "coder hardcoded to Opus 4.7 (agentic)",
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
) -> dict:
    """Run the repo indexer over `repo_path` and write to Neo4j.

    Returns the IndexBatch counts dict so the workflow output can include
    "we indexed X files / Y functions before the Coder ran."
    """
    from pathlib import Path
    from app.repo_indexer.actions import IndexBatch, RepoNode
    from app.repo_indexer.loader import (
        driver_from_env, ensure_constraints, prune_stale, write_batch,
    )
    from app.repo_indexer.scope_resolution.finalize import finalize_batch
    from app.repo_indexer.walker import walk_paths, walk_repo
    from app.repo_indexer.extractor_python import extract_python_file
    from app.repo_indexer.extractor_typescript import extract_typescript_file
    import tree_sitter_python as tspython
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser

    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo_path {repo_root} is not a directory")

    repo = RepoNode(name=repo_name, url="", commit_sha=commit_sha, tenant_id=tenant_id)
    py_parser = Parser(Language(tspython.language()))
    ts_parsers = {
        "typescript": Parser(Language(tsts.language_typescript())),
        "tsx":        Parser(Language(tsts.language_tsx())),
    }

    # Sprint 10g: path-only pre-walk (was reading bytes for nothing).
    activity.heartbeat("indexing — pre-walking file set")
    repo_files: set[str] = set()
    for rel_path, _lang in walk_paths(repo_root):
        repo_files.add(rel_path)

    aggregate = IndexBatch(repo=repo)
    file_count = 0
    for rel_path, source, language, sha in walk_repo(repo_root):
        try:
            if language == "python":
                fragment = extract_python_file(repo, rel_path, source, sha, py_parser)
            elif language in ("typescript", "javascript"):
                use_tsx = rel_path.endswith((".tsx", ".jsx"))
                parser = ts_parsers["tsx"] if use_tsx else ts_parsers["typescript"]
                fragment = extract_typescript_file(
                    repo, rel_path, source, sha, parser,
                    repo_files=repo_files, language=language,
                )
            else:
                continue
        except Exception as e:
            logger.warning("index_repo_activity: extractor failed on %s: %s", rel_path, e)
            continue
        aggregate.extend(fragment)
        file_count += 1
        if file_count % 50 == 0:
            activity.heartbeat(f"indexed {file_count} files")

    # Sprint 13 cleanup: legacy resolve_batch deleted; finalize_batch is the
    # only resolution path. NOTE: this activity bypasses runner.run_pipeline,
    # so CommunityDetectPhase / ProcessExtractPhase don't run here yet —
    # Temporal-driven scans miss the discoverability data. Tracked as a
    # Sprint 14-prep refactor: switch this body to run_pipeline(ctx, DEFAULT_PHASES).
    finalize_batch(aggregate)

    activity.heartbeat("writing to Neo4j")
    with driver_from_env() as driver:
        ensure_constraints(driver)
        prune_stale(driver, repo_name, commit_sha)
        write_batch(driver, aggregate)

    return {"file_count": file_count, **aggregate.counts()}


@activity.defn
async def run_repo_coder_activity(
    repo_path: str,
    repo_name: str,
    brief: str,
    tenant_id: str,
    workflow_id: str,
) -> dict:
    """Run the agentic Coder against the cloned + indexed repo.

    Opens its own Neo4j driver — workflow-side I/O is forbidden by
    Temporal, so the driver lifecycle is tied to this activity.
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
            )
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
    return result
