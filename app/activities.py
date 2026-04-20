"""
Activities are where all I/O happens (DB, Anthropic API).
Workflows must stay deterministic, so they can ONLY call into here.

Rule of thumb: if it touches the network, a clock, or randomness, it's an activity.
"""
import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Optional
from temporalio import activity

from . import db, agents

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
    )

@activity.defn
async def run_agent_activity(task_id: str, input: AgentTaskInput) -> AgentTaskResult:
    """Run one agent. Heartbeats keep Temporal from marking long LLM calls as stuck."""
    await db.update_task_running(task_id)

    context = await db.get_ancestor_outputs(task_id)

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

    return AgentTaskResult(
        task_id=task_id,
        output=result["output"],
        model=result["model"],
        tier=result["model_key"],
    )

@activity.defn
async def create_project_record(name: str, brief: str, workflow_id: str) -> str:
    return await db.create_project(name, brief, workflow_id)
