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
