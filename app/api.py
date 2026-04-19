"""
Thin intake API. Three endpoints:
- GET  /health        -> ALB health check target
- POST /projects      -> start a workflow
- GET  /projects/{id} -> poll status
"""
import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from temporalio.client import Client

from app import config, db
from app.workflows import ProjectWorkflow, ProjectInput

app = FastAPI(title="Lean Agent Framework")
_temporal: Client | None = None


@app.on_event("startup")
async def _validate_config() -> None:
    config.validate_runtime()

def _connect_kwargs():
    """Temporal Cloud uses TLS + API key; local dev uses plaintext."""
    if not config.TEMPORAL_TLS:
        return {}
    return {"tls": True, "api_key": config.TEMPORAL_API_KEY}

async def temporal() -> Client:
    global _temporal
    if _temporal is None:
        _temporal = await Client.connect(
            config.TEMPORAL_HOST,
            namespace=config.TEMPORAL_NAMESPACE,
            **_connect_kwargs(),
        )
    return _temporal

@app.get("/health")
async def health(deep: str | None = None):
    # ALB uses the shallow form. Keep the default cheap -- don't round-trip to
    # DB or Temporal every time. Pass ?deep=true (or deep=1) for a real probe.
    if deep not in ("true", "1"):
        return {"status": "ok"}

    temporal_state = "unknown"
    postgres_state = "unknown"
    errors = []

    # Temporal check — lightweight system-info RPC with a 3s timeout.
    try:
        from temporalio.api.workflowservice.v1 import GetSystemInfoRequest
        client = await temporal()
        await asyncio.wait_for(
            client.workflow_service.get_system_info(GetSystemInfoRequest()),
            timeout=3.0,
        )
        temporal_state = "ok"
    except Exception as e:
        temporal_state = "down"
        errors.append(f"temporal: {type(e).__name__}: {e}")

    # Postgres check — SELECT 1 with a 3s timeout.
    try:
        pool = await db.get_pool()
        await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=3.0)
        postgres_state = "ok"
    except Exception as e:
        postgres_state = "down"
        errors.append(f"postgres: {type(e).__name__}: {e}")

    if temporal_state == "ok" and postgres_state == "ok":
        return {"status": "ok", "temporal": "ok", "postgres": "ok"}

    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "temporal": temporal_state,
            "postgres": postgres_state,
            "error": "; ".join(errors) or "unknown",
        },
    )

class CreateProjectReq(BaseModel):
    name: str
    brief: str
    auto_approve: bool = False

class CreateProjectResp(BaseModel):
    workflow_id: str

class RejectReq(BaseModel):
    reason: str = ""

@app.post("/projects", response_model=CreateProjectResp)
async def create_project(req: CreateProjectReq):
    client = await temporal()
    workflow_id = f"project-{uuid.uuid4()}"
    await client.start_workflow(
        ProjectWorkflow.run,
        ProjectInput(name=req.name, brief=req.brief, auto_approve=req.auto_approve),
        id=workflow_id,
        task_queue="project-workflows",
    )
    return CreateProjectResp(workflow_id=workflow_id)

@app.post("/projects/{workflow_id}/approve")
async def approve_project(workflow_id: str):
    """Release the approval gate. Idempotent: signalling twice is fine."""
    client = await temporal()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(ProjectWorkflow.approve)
    return {"status": "approved", "workflow_id": workflow_id}

@app.post("/projects/{workflow_id}/reject")
async def reject_project(workflow_id: str, req: RejectReq):
    """Reject at the gate. Workflow short-circuits and returns."""
    client = await temporal()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(ProjectWorkflow.reject, req.reason)
    return {"status": "rejected", "workflow_id": workflow_id, "reason": req.reason}

@app.get("/projects/{workflow_id}/costs")
async def project_costs(workflow_id: str):
    """Spend breakdown for a project. Useful for router tuning over time."""
    pool = await db.get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM projects WHERE workflow_id=$1", workflow_id
    )
    if not row:
        raise HTTPException(404, "project not found")

    rows = await pool.fetch(
        """
        SELECT role, provider, model_used,
               SUM(tokens_in) AS tokens_in,
               SUM(tokens_out) AS tokens_out,
               SUM(cost_usd) AS cost_usd,
               COUNT(*) AS calls
        FROM tasks WHERE project_id=$1 AND status='done'
        GROUP BY role, provider, model_used
        ORDER BY cost_usd DESC NULLS LAST
        """,
        row["id"],
    )
    total = sum(float(r["cost_usd"] or 0) for r in rows)
    return {
        "workflow_id": workflow_id,
        "total_usd": round(total, 4),
        "breakdown": [dict(r) for r in rows],
    }

@app.get("/projects/{workflow_id}")
async def get_project(workflow_id: str):
    client = await temporal()
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    status = desc.status.name

    # Query the workflow for approval-gate state. Safe even if already complete.
    approval_state = None
    try:
        approval_state = await handle.query(ProjectWorkflow.status)
    except Exception:
        # Workflow may have finished/failed before this is meaningful
        pass

    pool = await db.get_pool()
    row = await pool.fetchrow(
        "SELECT id, name, brief FROM projects WHERE workflow_id=$1", workflow_id
    )
    if not row:
        raise HTTPException(404, "project not found")

    tasks = await db.get_project_tasks(str(row["id"]))

    # Compute awaiting_approval: architect is done but SE hasn't started
    has_architect = any(t["role"] == "architect" and t["status"] == "done" for t in tasks)
    has_se = any(t["role"] == "se" for t in tasks)
    awaiting_approval = (
        status == "RUNNING"
        and has_architect
        and not has_se
        and approval_state is not None
        and not approval_state.get("approved", False)
        and not approval_state.get("rejected", False)
    )

    return {
        "workflow_id": workflow_id,
        "status": status,
        "awaiting_approval": awaiting_approval,
        "approval_state": approval_state,
        "project": {"id": str(row["id"]), "name": row["name"], "brief": row["brief"]},
        "tasks": [
            {**t, "id": str(t["id"]), "parent_task_id": str(t["parent_task_id"]) if t["parent_task_id"] else None}
            for t in tasks
        ],
    }
