"""
Thin async Postgres wrapper. One pool per process.

Activities call these functions directly -- no ORM, no repository pattern.
The tasks table IS the domain model at this stage.
"""
import asyncpg
import json
from typing import Optional
from uuid import UUID
from . import config

_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(config.PG_DSN, min_size=2, max_size=10)
    return _pool

async def create_project(name: str, brief: str, workflow_id: str) -> str:
    pool = await get_pool()
    row = await pool.fetchrow(
        "INSERT INTO projects (name, brief, workflow_id) VALUES ($1, $2, $3) RETURNING id",
        name, brief, workflow_id,
    )
    return str(row["id"])

async def create_task(
    project_id: str,
    role: str,
    title: str,
    description: str,
    parent_task_id: Optional[str] = None,
    input_data: Optional[dict] = None,
) -> str:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO tasks (project_id, parent_task_id, role, title, description, input)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        UUID(project_id),
        UUID(parent_task_id) if parent_task_id else None,
        role, title, description,
        json.dumps(input_data) if input_data else None,
    )
    return str(row["id"])

async def update_task_running(task_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE tasks SET status='running', updated_at=now() WHERE id=$1",
        UUID(task_id),
    )

async def complete_task(
    task_id: str,
    output: dict,
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
):
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE tasks
        SET status='done', output=$2, provider=$3, model_used=$4,
            tokens_in=$5, tokens_out=$6, cost_usd=$7, updated_at=now()
        WHERE id=$1
        """,
        UUID(task_id), json.dumps(output), provider, model,
        tokens_in, tokens_out, cost_usd,
    )

async def fail_task(task_id: str, error: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE tasks SET status='failed', output=$2, updated_at=now() WHERE id=$1",
        UUID(task_id), json.dumps({"error": error}),
    )

async def get_ancestor_outputs(task_id: str) -> list[dict]:
    """
    Walk up the task tree to collect all completed ancestor outputs.
    This is the cheap, no-embedding-needed version of context retrieval.
    Use this first; add vector search only when this stops being enough.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        WITH RECURSIVE ancestors AS (
            SELECT id, parent_task_id, role, title, output, 0 AS depth
            FROM tasks WHERE id = $1
            UNION ALL
            SELECT t.id, t.parent_task_id, t.role, t.title, t.output, a.depth + 1
            FROM tasks t JOIN ancestors a ON t.id = a.parent_task_id
        )
        SELECT role, title, output FROM ancestors
        WHERE depth > 0 AND output IS NOT NULL
        ORDER BY depth ASC
        """,
        UUID(task_id),
    )
    return [
        {"role": r["role"], "title": r["title"], "output": json.loads(r["output"])}
        for r in rows
    ]

async def get_project_tasks(project_id: str) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, parent_task_id, role, title, status, provider, model_used,
               tokens_in, tokens_out, cost_usd, output,
               created_at, updated_at
        FROM tasks WHERE project_id=$1 ORDER BY created_at ASC
        """,
        UUID(project_id),
    )
    out = []
    for r in rows:
        d = dict(r)
        # asyncpg returns JSONB as a string; UI needs structured data.
        if d.get("output") is not None:
            d["output"] = json.loads(d["output"])
        # Numeric is a Decimal; JSON encoder won't serialise that.
        if d.get("cost_usd") is not None:
            d["cost_usd"] = float(d["cost_usd"])
        out.append(d)
    return out
