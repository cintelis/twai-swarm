"""
Thin intake API + bundled UI. Endpoints:
- GET  /                   -> redirects to /ui/
- GET  /ui/                -> static SPA (app/static/index.html)
- GET  /health             -> ALB health check target
- POST /projects           -> start a workflow
- GET  /projects/{id}      -> poll status
- GET  /projects/{id}/costs
- GET  /projects/{id}/download -> zip of the coder agent's scaffold
- POST /projects/{id}/approve
- POST /projects/{id}/reject
"""
import asyncio
import io
import pathlib
import re
import uuid
import zipfile
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from temporalio.client import Client

from app import auth, config, db, telemetry
from app.workflows import ProjectWorkflow, ProjectInput, RepoTaskWorkflow, RepoTaskInput

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# Per-IP rate limiter. Generous defaults for a single-team dev tool;
# slowapi's in-memory store is fine for one container — tighten + use
# Redis if we ever scale the API horizontally beyond two tasks.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(title="Lean Agent Framework")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
_temporal: Client | None = None

# Initialise OTel before instrumenting the FastAPI app — order matters,
# otherwise the auto-instrumentation hooks attach to a not-yet-set provider.
telemetry.init(role="api")
telemetry.instrument_fastapi(app)


@app.on_event("startup")
async def _validate_config() -> None:
    config.validate_runtime()


@app.get("/auth/check", include_in_schema=False)
async def auth_check(_: None = Depends(auth.require_auth)) -> dict:
    """UI calls this on load to decide whether the stored token is still valid.
    Returns 200 if (auth disabled) OR (valid token supplied), 401 otherwise."""
    return {"ok": True, "auth_required": auth.auth_enabled()}


@app.get("/auth/status", include_in_schema=False)
async def auth_status() -> dict:
    """Tells the UI whether to show the token-entry modal at all."""
    return {"auth_required": auth.auth_enabled()}

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
    # Tenant identity. Optional — omitted → "default" (single-tenant dev).
    # When the auth middleware lands, tenant_id will be resolved from the JWT
    # and this field becomes a safety check (must match the JWT's claim).
    tenant_id: str | None = None

class CreateProjectResp(BaseModel):
    workflow_id: str

class RejectReq(BaseModel):
    reason: str = ""

@app.post("/projects", response_model=CreateProjectResp, dependencies=[Depends(auth.require_auth)])
@limiter.limit("10/minute")
async def create_project(request: Request, req: CreateProjectReq):
    from app import tenant
    # Resolve + validate tenant_id. Defaults to "default" for single-tenant
    # dev; validate to reject bad input early even though we don't have
    # auth middleware yet.
    tenant_id = req.tenant_id or tenant.DEFAULT_TENANT_ID
    try:
        tenant.validate_tenant_id(tenant_id)
    except tenant.InvalidTenantIdError as e:
        raise HTTPException(400, str(e))

    client = await temporal()
    workflow_id = f"project-{uuid.uuid4()}"
    await client.start_workflow(
        ProjectWorkflow.run,
        ProjectInput(
            name=req.name,
            brief=req.brief,
            auto_approve=req.auto_approve,
            tenant_id=tenant_id,
        ),
        id=workflow_id,
        task_queue="project-workflows",
    )
    return CreateProjectResp(workflow_id=workflow_id)


# ─── Sprint 10e — RepoTaskWorkflow (work on existing code) ──────────────────

class CreateRepoTaskReq(BaseModel):
    repo_url: str            # public https git URL (Sprint 10f adds GitHub App auth)
    branch: str = "main"
    brief: str
    repo_name: str | None = None      # default: derive from URL
    tenant_id: str | None = None


class CreateRepoTaskResp(BaseModel):
    workflow_id: str


@app.post("/repo-tasks", response_model=CreateRepoTaskResp, dependencies=[Depends(auth.require_auth)])
@limiter.limit("10/minute")
async def create_repo_task(request: Request, req: CreateRepoTaskReq):
    """Kick off a RepoTaskWorkflow: clone -> index -> graph-aware Coder.

    Returns the workflow_id so the caller can poll
    `GET /repo-tasks/{workflow_id}` (added in 10f) for status + diff.
    For now use the existing Temporal handle pattern to await completion.
    """
    from app import tenant
    tenant_id = req.tenant_id or tenant.DEFAULT_TENANT_ID
    try:
        tenant.validate_tenant_id(tenant_id)
    except tenant.InvalidTenantIdError as e:
        raise HTTPException(400, str(e))

    client = await temporal()
    workflow_id = f"repo-task-{uuid.uuid4()}"
    await client.start_workflow(
        RepoTaskWorkflow.run,
        RepoTaskInput(
            repo_url=req.repo_url,
            branch=req.branch,
            brief=req.brief,
            repo_name=req.repo_name or "",
            tenant_id=tenant_id,
        ),
        id=workflow_id,
        task_queue="project-workflows",
    )

    # Land a row in `projects` so the UI's recent-list + detail page work
    # for repo-task workflows. The detail page uses workflow_id prefix
    # ("repo-task-" vs "project-") to switch rendering — see GET /projects/{id}.
    repo_name = (
        req.repo_name
        or req.repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    )
    one_line = " ".join(req.brief.split())
    display_name = f"{repo_name}: {one_line[:80]}{'…' if len(one_line) > 80 else ''}"
    await db.create_project(
        name=display_name,
        brief=req.brief,
        workflow_id=workflow_id,
        tenant_id=tenant_id,
    )
    return CreateRepoTaskResp(workflow_id=workflow_id)


@app.post("/projects/{workflow_id}/approve", dependencies=[Depends(auth.require_auth)])
async def approve_project(workflow_id: str):
    """Release the approval gate. Idempotent: signalling twice is fine."""
    client = await temporal()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(ProjectWorkflow.approve)
    return {"status": "approved", "workflow_id": workflow_id}

@app.post("/projects/{workflow_id}/reject", dependencies=[Depends(auth.require_auth)])
async def reject_project(workflow_id: str, req: RejectReq):
    """Reject at the gate. Workflow short-circuits and returns."""
    client = await temporal()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(ProjectWorkflow.reject, req.reason)
    return {"status": "rejected", "workflow_id": workflow_id, "reason": req.reason}

async def _resolve_workflow_status(workflow_id: str) -> str | None:
    """Ask Temporal for the authoritative workflow status.

    Returns None if we can't talk to Temporal or the workflow doesn't exist —
    callers fall back to whatever the DB says.
    """
    try:
        client = await temporal()
        handle = client.get_workflow_handle(workflow_id)
        desc = await asyncio.wait_for(handle.describe(), timeout=2.0)
        return desc.status.name  # RUNNING / COMPLETED / FAILED / CANCELED / TERMINATED / TIMED_OUT
    except Exception:
        return None


@app.get("/projects", dependencies=[Depends(auth.require_auth)])
async def list_projects(limit: int = 50):
    """Recent projects for the UI listing view.

    The `projects.status` column is initialised to 'running' on workflow start
    and isn't updated anywhere — Temporal owns the real workflow state. So
    for anything the DB still says is 'running', we reconcile by querying
    Temporal (concurrently, 2s timeout each) and return the real status.

    Opportunistic write-back: rows that have transitioned to a terminal
    Temporal state get their DB column updated so future list requests skip
    the Temporal round-trip.
    """
    pool = await db.get_pool()
    rows = await pool.fetch(
        """
        SELECT p.id, p.name, p.brief, p.workflow_id, p.status, p.created_at,
               COUNT(t.id)                                             AS task_count,
               COUNT(t.id) FILTER (WHERE t.status = 'done')            AS done_count,
               COUNT(t.id) FILTER (WHERE t.status = 'failed')          AS failed_count,
               COALESCE(SUM(t.cost_usd), 0)                            AS total_cost_usd
        FROM projects p
        LEFT JOIN tasks t ON t.project_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        LIMIT $1
        """,
        max(1, min(limit, 200)),
    )

    # Reconcile 'running' rows against Temporal in parallel.
    # Note: `projects.status` legacy is lowercase ('running'); Temporal returns
    # uppercase status names (RUNNING, COMPLETED, ...). Normalise on output.
    def _is_stale_running(db_status: str | None) -> bool:
        return (db_status or "").lower() == "running"

    stale_workflow_ids = [r["workflow_id"] for r in rows if _is_stale_running(r["status"])]
    resolved: dict[str, str] = {}
    if stale_workflow_ids:
        results = await asyncio.gather(
            *(_resolve_workflow_status(wid) for wid in stale_workflow_ids),
            return_exceptions=True,
        )
        for wid, res in zip(stale_workflow_ids, results):
            if isinstance(res, str):
                resolved[wid] = res

        # Write-back terminal states so future list requests skip this path.
        terminals = {
            wid: s for wid, s in resolved.items()
            if s in ("COMPLETED", "FAILED", "CANCELED", "TERMINATED", "TIMED_OUT")
        }
        if terminals:
            await pool.executemany(
                "UPDATE projects SET status=$2 WHERE workflow_id=$1",
                [(wid, s) for wid, s in terminals.items()],
            )

    def _final_status(r) -> str:
        wid = r["workflow_id"]
        if wid in resolved:
            return resolved[wid]
        # Normalise legacy lowercase DB values ('running') to the Temporal shape.
        return (r["status"] or "UNKNOWN").upper()

    return {
        "projects": [
            {
                "workflow_id": r["workflow_id"],
                "name": r["name"],
                "brief": (r["brief"] or "")[:160] + ("…" if r["brief"] and len(r["brief"]) > 160 else ""),
                "status": _final_status(r),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "task_count": r["task_count"],
                "done_count": r["done_count"],
                "failed_count": r["failed_count"],
                "total_cost_usd": float(r["total_cost_usd"] or 0),
            }
            for r in rows
        ]
    }


@app.get("/projects/{workflow_id}/costs", dependencies=[Depends(auth.require_auth)])
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

def _salvage_files_from_truncated(raw_text: str) -> list[dict]:
    """Extract complete {path, content} entries from a truncated coder JSON.

    When the coder's output exceeds max_tokens the response gets cut mid-JSON,
    the runner fails json.loads, and the row is stored as
    {"raw_text": "...", "parse_error": True}. This helper finds every
    brace-balanced `{"path": ..., "content": ...}` object that parses
    cleanly and returns it — so a 40-file scaffold with a truncated 41st
    file still yields 40 recovered files instead of a 422.
    """
    import json as _json
    import re as _re

    files: list[dict] = []
    for m in _re.finditer(r'"path"\s*:\s*"', raw_text):
        brace = raw_text.rfind("{", 0, m.start())
        if brace == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(brace, len(raw_text)):
            c = raw_text[j]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end == -1:
            break  # truncated here
        try:
            parsed = _json.loads(raw_text[brace:end])
        except _json.JSONDecodeError:
            break
        if isinstance(parsed, dict) and "path" in parsed and "content" in parsed:
            files.append(parsed)
    return files


@app.get("/projects/{workflow_id}/download", dependencies=[Depends(auth.require_auth)])
async def download_project(workflow_id: str):
    """Bundle the Coder agent's file tree into a zip and stream it back."""
    pool = await db.get_pool()
    project_row = await pool.fetchrow(
        "SELECT id, name FROM projects WHERE workflow_id=$1", workflow_id
    )
    if not project_row:
        raise HTTPException(404, "project not found")

    coder_row = await pool.fetchrow(
        """
        SELECT output FROM tasks
        WHERE project_id=$1 AND role='coder' AND status='done'
        ORDER BY created_at DESC LIMIT 1
        """,
        project_row["id"],
    )
    if not coder_row or not coder_row["output"]:
        raise HTTPException(404, "no coder output yet — workflow may still be running")

    import json as _json
    output = _json.loads(coder_row["output"])
    files = output.get("files") if isinstance(output, dict) else None
    if not isinstance(files, list) or not files:
        # Salvage path: if the coder output was truncated mid-JSON (token
        # overflow), the runner stored it as {"raw_text": "...", "parse_error": True}.
        # Extract every complete {path, content} pair from the raw text
        # so the user gets the intact prefix instead of a 422.
        raw_text = output.get("raw_text") if isinstance(output, dict) else None
        if isinstance(raw_text, str) and raw_text:
            files = _salvage_files_from_truncated(raw_text)
        if not files:
            raise HTTPException(422, "coder output has no `files` array (and no salvageable raw_text)")

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", project_row["name"]).strip("-") or "project"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Drop a small marker so users know what produced the zip.
        zf.writestr(
            f"{safe_name}/.swarm-info",
            f"workflow_id: {workflow_id}\nproject: {safe_name}\nlanguage: {output.get('language', 'unknown')}\n",
        )
        for f in files:
            path = (f.get("path") or "").lstrip("/").replace("\\", "/")
            content = f.get("content") or ""
            if not path or ".." in path.split("/"):
                continue
            zf.writestr(f"{safe_name}/{path}", content)

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}-scaffold.zip"',
        },
    )


@app.get("/projects/{workflow_id}", dependencies=[Depends(auth.require_auth)])
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

    # Repo-task workflows don't run BA/Architect/SE/etc., so `tasks` is empty
    # and the agent-task UI is meaningless. Surface the workflow's result
    # (diff + files_changed + summary) instead so the user can actually see
    # what the Coder did. Skipped for still-running workflows; the Temporal
    # describe() above already reports status=RUNNING in that case.
    repo_task_result = None
    if workflow_id.startswith("repo-task-") and status == "COMPLETED":
        try:
            result = await asyncio.wait_for(handle.result(), timeout=5.0)
            # `handle.result()` returns a dict here, not a RepoTaskOutput
            # dataclass — `get_workflow_handle(workflow_id)` is untyped, so
            # the SDK has no return-type hint to deserialise into. Reading
            # via getattr() silently returned the default for every field
            # (iterations=0, tokens=0, cost=0.0), which is what the UI
            # rendered as "0 iterations / No files changed / $0.0000" even
            # for workflows that completed successfully with real numbers.
            # _get supports both shapes so a future switch to a typed handle
            # doesn't regress this.
            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    val = obj.get(key, default)
                else:
                    val = getattr(obj, key, default)
                return default if val is None else val
            repo_task_result = {
                "commit_sha": _get(result, "commit_sha"),
                "files_changed": list(_get(result, "files_changed", []) or []),
                "diff": _get(result, "diff", "") or "",
                "iterations": _get(result, "iterations", 0),
                "summary": _get(result, "summary", "") or "",
                "tokens_in": _get(result, "tokens_in", 0),
                "tokens_out": _get(result, "tokens_out", 0),
                "cost_usd": float(_get(result, "cost_usd", 0.0) or 0.0),
                # Auto-PR step output (added Sprint 10f). Older workflow
                # outputs without these keys degrade to None silently via
                # _get defaults, so the field is always present.
                "pr_url": _get(result, "pr_url"),
                "pr_number": _get(result, "pr_number"),
                "branch_name": _get(result, "branch_name"),
                "push_error": _get(result, "push_error"),
            }
        except Exception:
            # Workflow result unavailable (timeout, serialisation mismatch, etc.).
            # The detail page degrades gracefully — status + project info still render.
            pass

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
        "repo_task_result": repo_task_result,
        "is_repo_task": workflow_id.startswith("repo-task-"),
    }


# ─── GitHub App ──────────────────────────────────────────
# v1: single-tenant — all installs go under tenant_id='default' (the schema's
# default). Greenfield will resolve tenant_id from JWT claims via auth middleware
# without changing the storage layer.

class GitHubPushReq(BaseModel):
    installation_id: int
    repo_owner: str
    repo_name: str
    branch: str | None = None         # auto-generated if None
    open_pr: bool = True
    # When the repo doesn't exist and the installation is on an Organization,
    # create the repo (Administration: Write permission required on the App).
    # User-type installations can't auto-create; the push fails with a clear
    # error in that case.
    create_if_missing: bool = True
    repo_private: bool = True
    repo_description: str | None = None


@app.get("/github/install-url", dependencies=[Depends(auth.require_auth)])
async def github_install_url():
    """The public install URL — UI 'Connect GitHub' button opens this."""
    if not config.GITHUB_APP_INSTALL_URL:
        raise HTTPException(503, "GITHUB_APP_INSTALL_URL is not configured")
    return {"install_url": config.GITHUB_APP_INSTALL_URL}


@app.post("/github/webhook", include_in_schema=False)
async def github_webhook(request: Request):
    """Receives GitHub App webhook events.

    No bearer-token auth — GitHub signs the payload with HMAC-SHA256 using
    the configured webhook secret; we verify that instead.

    Returns 503 if the webhook secret isn't configured (or still UNSET).
    Returns 401 on signature mismatch. Returns 200 with a small JSON body
    on every successful event (even "ignored" ones) so GitHub stops retrying.
    """
    import json as _json
    from app import github_app, github_webhook as webhook_mod

    if not config.GITHUB_APP_WEBHOOK_SECRET or config.GITHUB_APP_WEBHOOK_SECRET == "UNSET":
        raise HTTPException(503, "GITHUB_APP_WEBHOOK_SECRET not configured")

    body = await request.body()
    sig_header = request.headers.get("X-Hub-Signature-256")
    event_type = request.headers.get("X-GitHub-Event", "")

    try:
        webhook_mod.verify_signature(config.GITHUB_APP_WEBHOOK_SECRET, body, sig_header)
    except webhook_mod.WebhookVerificationError as e:
        raise HTTPException(401, f"webhook verification failed: {e}")

    try:
        payload = _json.loads(body)
    except _json.JSONDecodeError:
        raise HTTPException(400, "invalid JSON body")

    return await webhook_mod.handle_event(
        event_type, payload,
        db_module=db,
        github_app_module=github_app,
    )


@app.get("/github/callback", include_in_schema=False)
async def github_callback(installation_id: int, setup_action: str = "install"):
    """GitHub redirects here after the user installs the App.

    No auth on this endpoint (GitHub is the caller, not a UI client). We fetch
    the install metadata from GitHub and persist it. The redirect target then
    lands the user back in the SPA where they can pick a repo to push to.
    """
    from app import github_app
    if setup_action not in ("install", "update"):
        raise HTTPException(400, f"unexpected setup_action: {setup_action}")
    try:
        meta = await github_app.fetch_installation_metadata(installation_id)
    except github_app.GitHubAppError as e:
        raise HTTPException(502, f"GitHub install lookup failed: {e}")

    await db.upsert_github_installation(
        installation_id=meta["installation_id"],
        account_login=meta["account_login"],
        account_type=meta["account_type"],
    )
    # Bounce back into the UI; the SPA polls /github/installations on load.
    return RedirectResponse(url="/ui/#/github/connected", status_code=302)


@app.get("/github/installations", dependencies=[Depends(auth.require_auth)])
async def github_list_installations():
    rows = await db.get_github_installations()
    return {"installations": [
        {
            "installation_id": r["installation_id"],
            "account_login": r["account_login"],
            "account_type": r["account_type"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]}


@app.delete("/github/installations/{installation_id}", dependencies=[Depends(auth.require_auth)])
async def github_delete_installation(installation_id: int):
    """Drop a stale install row. Use when an App was uninstalled on GitHub
    and the DB row is now pointing at a dead installation_id (causes
    token-mint 404s on subsequent pushes).
    """
    from app import github_app
    deleted = await db.delete_github_installation(installation_id)
    # Also clear any cached token for this installation.
    github_app._token_cache.pop(installation_id, None)
    return {"deleted": deleted, "installation_id": installation_id}


@app.get("/github/installations/{installation_id}/repos", dependencies=[Depends(auth.require_auth)])
async def github_list_repos(installation_id: int):
    from app import github_app
    install = await db.get_github_installation(installation_id)
    if not install:
        raise HTTPException(404, "installation not found")
    try:
        repos = await github_app.list_installation_repos(installation_id)
    except github_app.GitHubAppError as e:
        raise HTTPException(502, f"GitHub repo list failed: {e}")
    return {"repos": repos}


@app.post("/projects/{workflow_id}/github-push", dependencies=[Depends(auth.require_auth)])
async def github_push(workflow_id: str, req: GitHubPushReq):
    """Push the project's coder scaffold to a GitHub repo as a new branch (+ PR)."""
    from app import github_app
    import time as _time
    pool = await db.get_pool()
    project_row = await pool.fetchrow(
        "SELECT id, name FROM projects WHERE workflow_id=$1", workflow_id
    )
    if not project_row:
        raise HTTPException(404, "project not found")

    # Reuse the same coder-output extraction the /download endpoint uses,
    # including the truncated-output salvage path.
    coder_row = await pool.fetchrow(
        """
        SELECT output FROM tasks
        WHERE project_id=$1 AND role='coder' AND status='done'
        ORDER BY created_at DESC LIMIT 1
        """,
        project_row["id"],
    )
    if not coder_row or not coder_row["output"]:
        raise HTTPException(404, "no coder output yet — workflow may still be running")

    import json as _json
    output = _json.loads(coder_row["output"])
    files = output.get("files") if isinstance(output, dict) else None
    if not isinstance(files, list) or not files:
        raw_text = output.get("raw_text") if isinstance(output, dict) else None
        if isinstance(raw_text, str) and raw_text:
            files = _salvage_files_from_truncated(raw_text)
        if not files:
            raise HTTPException(422, "coder output has no files to push")

    install = await db.get_github_installation(req.installation_id)
    if not install:
        raise HTTPException(404, "installation not found")

    # ── Verify install is still alive on GitHub ──────────────────────
    # If you uninstalled the App, our DB row is stale. Auto-clean it and
    # surface a 410 the UI can use to prompt re-install.
    try:
        meta = await github_app.fetch_installation_metadata(req.installation_id)
    except github_app.GitHubAppError as e:
        if "404" in str(e):
            await db.delete_github_installation(req.installation_id)
            github_app._token_cache.pop(req.installation_id, None)
            raise HTTPException(
                410,
                f"Installation {req.installation_id} no longer exists on GitHub "
                f"(it was uninstalled). DB row cleaned up — re-install the App and try again.",
            )
        raise HTTPException(502, f"installation metadata fetch failed: {e}")

    # ── Does the target repo already exist? ──────────────────────────
    # Determines what permissions we'll need: pushing to an existing repo
    # only needs base perms; creating a new org repo also needs
    # organization_administration. Doing this BEFORE the preflight means
    # User-account installs (which can never have org perms) work fine
    # for pushes to existing repos.
    try:
        exists = await github_app.repo_exists(
            req.installation_id, req.repo_owner, req.repo_name,
        )
    except github_app.GitHubAppError as e:
        raise HTTPException(502, f"repo existence check failed: {e}")

    needs_repo_create = False
    if not exists:
        if not req.create_if_missing:
            raise HTTPException(
                404,
                f"repo {req.repo_owner}/{req.repo_name} does not exist (and create_if_missing=false)",
            )
        if install["account_type"] != "Organization":
            raise HTTPException(
                400,
                f"cannot auto-create {req.repo_owner}/{req.repo_name}: installation is on a "
                f"User account (not Organization). GitHub Apps can't create repos under "
                f"user accounts; create the repo manually on GitHub and retry.",
            )
        if req.repo_owner.lower() != install["account_login"].lower():
            raise HTTPException(
                400,
                f"repo owner {req.repo_owner!r} does not match installation account "
                f"{install['account_login']!r} — auto-create only supported for the installed org",
            )
        needs_repo_create = True

    # ── Permission preflight (operation-aware) ───────────────────────
    # Push-to-existing needs only BASE_PERMISSIONS. Create-then-push needs
    # organization_administration on top of those. Surfacing the precise
    # missing perm with a deep link beats the cryptic GitHub 403.
    required = (
        github_app.ORG_CREATE_PERMISSIONS if needs_repo_create
        else github_app.BASE_PERMISSIONS
    )
    missing = github_app.missing_permissions(meta.get("permissions", {}), required)
    if missing:
        install_url_for_accept = (
            f"https://github.com/organizations/{meta['account_login']}/settings/installations/{req.installation_id}"
            if meta["account_type"] == "Organization"
            else f"https://github.com/settings/installations/{req.installation_id}"
        )
        raise HTTPException(
            400,
            f"Installation is missing required permissions: {', '.join(missing)}. "
            f"Go to {install_url_for_accept} and click 'Accept new permissions' to fix.",
        )

    # ── Create repo if needed ────────────────────────────────────────
    repo_was_created = False
    if needs_repo_create:
        try:
            await github_app.create_org_repo(
                installation_id=req.installation_id,
                org=req.repo_owner,
                name=req.repo_name,
                description=req.repo_description or "",
                private=req.repo_private,
            )
            repo_was_created = True
        except github_app.GitHubAppError as e:
            raise HTTPException(502, f"repo creation failed: {e}")

    branch = req.branch or f"swarm/{workflow_id[:12]}-{int(_time.time())}"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", project_row["name"]).strip("-") or "project"
    commit_msg = f"Initial scaffold from twai-swarm: {safe_name}\n\nWorkflow: {workflow_id}"

    try:
        result = await github_app.push_files_as_branch(
            installation_id=req.installation_id,
            repo_owner=req.repo_owner,
            repo_name=req.repo_name,
            branch=branch,
            files=[
                {"path": (f.get("path") or "").lstrip("/").replace("\\", "/"),
                 "content": f.get("content") or ""}
                for f in files
                if f.get("path") and ".." not in (f.get("path") or "").split("/")
            ],
            commit_message=commit_msg,
            open_pr=req.open_pr,
            pr_title=f"twai-swarm scaffold: {safe_name}",
            pr_body=f"Generated by the twai-swarm agentic Coder.\n\n**Workflow:** `{workflow_id}`",
        )
    except github_app.GitHubAppError as e:
        raise HTTPException(502, f"push failed: {e}")

    push_id = await db.record_github_push(
        project_id=str(project_row["id"]),
        installation_id=req.installation_id,
        repo_owner=req.repo_owner,
        repo_name=req.repo_name,
        branch=result.branch,
        commit_sha=result.commit_sha,
        pr_url=result.pr_url,
        pr_number=result.pr_number,
        files_pushed=result.files_pushed,
    )
    return {
        "push_id": push_id,
        "repo": f"{req.repo_owner}/{req.repo_name}",
        "repo_created": repo_was_created,
        "repo_initialised": result.repo_initialised,
        "branch": result.branch,
        "commit_sha": result.commit_sha,
        "pr_url": result.pr_url,
        "pr_number": result.pr_number,
        "files_pushed": result.files_pushed,
    }


@app.get("/projects/{workflow_id}/github-pushes", dependencies=[Depends(auth.require_auth)])
async def github_pushes_for_project(workflow_id: str):
    pool = await db.get_pool()
    row = await pool.fetchrow("SELECT id FROM projects WHERE workflow_id=$1", workflow_id)
    if not row:
        raise HTTPException(404, "project not found")
    pushes = await db.list_github_pushes_for_project(str(row["id"]))
    return {"pushes": pushes}


# ─── UI ─────────────────────────────────────────────────
# Mount AFTER all API routes so they win on the path resolver.
@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
