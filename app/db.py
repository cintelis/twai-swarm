"""
Thin async Postgres wrapper. One pool per process.

Activities call these functions directly -- no ORM, no repository pattern.
The tasks table IS the domain model at this stage.
"""
import asyncpg
import json
import logging
from typing import Optional
from uuid import UUID
from . import config

logger = logging.getLogger(__name__)

# Roles that benefit from kNN context augmentation (synthesise-from-priors
# rather than research-from-scratch). Researcher / BA / Architect / Coder
# get only the parent-walk because they're meant to drive the search themselves.
_SYNTHESIS_ROLES = {"se", "estimator", "reviewer", "documenter"}

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

    Returned dicts include `task_id` so callers can dedupe against kNN matches.
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
        SELECT id, role, title, output FROM ancestors
        WHERE depth > 0 AND output IS NOT NULL
        ORDER BY depth ASC
        """,
        UUID(task_id),
    )
    return [
        {
            "task_id": str(r["id"]),
            "role": r["role"],
            "title": r["title"],
            "output": json.loads(r["output"]),
        }
        for r in rows
    ]


async def upsert_task_embedding(task_id: str, content: str, embedding: list[float]) -> None:
    """Store the embedding for a completed task. Upsert so re-embeds replace.

    `embedding` is a 1536-dim float vector — matches the vector(1536) column.
    `content` is the source text we embedded (kept so we can re-embed without
    re-deriving the prompt from output JSON).
    """
    from app.embeddings import vector_literal
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO task_embeddings (task_id, content, embedding)
        VALUES ($1, $2, $3::vector)
        ON CONFLICT (task_id) DO UPDATE
          SET content = EXCLUDED.content,
              embedding = EXCLUDED.embedding,
              created_at = now()
        """,
        UUID(task_id), content, vector_literal(embedding),
    )


async def get_similar_task_outputs(
    project_id: str,
    embedding: list[float],
    exclude_task_ids: list[str],
    limit: int = 3,
    min_similarity: float = 0.30,
) -> list[dict]:
    """kNN over `task_embeddings`, scoped to one project, excluding given IDs.

    `min_similarity` filters out matches that are nominally close in vector
    space but semantically irrelevant — kNN always returns N rows even if
    they're noise. 0.30 is a conservative floor; tighten if matches feel weak.
    """
    from app.embeddings import vector_literal
    pool = await get_pool()
    excludes = [UUID(tid) for tid in exclude_task_ids]
    # asyncpg can't bind an empty list; guard.
    rows = await pool.fetch(
        """
        SELECT t.id, t.role, t.title, t.output,
               1 - (te.embedding <=> $1::vector) AS similarity
        FROM task_embeddings te
        JOIN tasks t ON t.id = te.task_id
        WHERE t.project_id = $2
          AND ($3::uuid[] = '{}' OR t.id <> ALL($3::uuid[]))
          AND t.output IS NOT NULL
          AND 1 - (te.embedding <=> $1::vector) >= $4
        ORDER BY te.embedding <=> $1::vector
        LIMIT $5
        """,
        vector_literal(embedding), UUID(project_id), excludes, min_similarity, limit,
    )
    return [
        {
            "task_id": str(r["id"]),
            "role": r["role"],
            "title": r["title"],
            "output": json.loads(r["output"]),
            "similarity": float(r["similarity"]),
        }
        for r in rows
    ]


async def get_context_for_task(task_id: str) -> list[dict]:
    """Combined context: parent-walk ancestors + (for synthesis roles) kNN.

    Always returns ancestors. For synthesis roles (SE / Estimator / Reviewer /
    Documenter) it also pulls the top-3 most-similar prior outputs from the
    same project, excluding direct ancestors. Each entry is tagged with
    `_source` ("ancestor" or "similar") so the agent prompt can frame them
    differently.

    Falls back to ancestors-only on any embedding failure — kNN is additive,
    not load-bearing.
    """
    ancestors = await get_ancestor_outputs(task_id)

    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT project_id, role, title, description FROM tasks WHERE id=$1",
        UUID(task_id),
    )
    if not row:
        return ancestors

    role = row["role"]
    if role not in _SYNTHESIS_ROLES:
        for a in ancestors:
            a["_source"] = "ancestor"
        return ancestors

    # Build query embedding from this task's input. If embedding fails (no key,
    # rate limit, etc.) we silently skip kNN and return ancestors as-is.
    try:
        from app.embeddings import embed_text, task_to_embedding_text
        query_text = task_to_embedding_text(role, row["title"], {"description": row["description"]})
        query_embedding = await embed_text(query_text)
        ancestor_ids = [a["task_id"] for a in ancestors]
        similar = await get_similar_task_outputs(
            project_id=str(row["project_id"]),
            embedding=query_embedding,
            exclude_task_ids=ancestor_ids,
        )
    except Exception as e:
        logger.warning(
            "kNN context augmentation failed for task %s (role=%s): %s; using ancestors only",
            task_id, role, e,
        )
        similar = []

    for a in ancestors:
        a["_source"] = "ancestor"
    for s in similar:
        s["_source"] = "similar"

    return ancestors + similar

async def upsert_github_installation(
    installation_id: int,
    account_login: str,
    account_type: str,
    tenant_id: str = "default",
) -> None:
    """Record a GitHub App install. Upsert so re-installs replace the row."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO github_installations (installation_id, account_login, account_type, tenant_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (installation_id) DO UPDATE
          SET account_login = EXCLUDED.account_login,
              account_type  = EXCLUDED.account_type,
              tenant_id     = EXCLUDED.tenant_id,
              updated_at    = now()
        """,
        installation_id, account_login, account_type, tenant_id,
    )


async def get_github_installations(tenant_id: str = "default") -> list[dict]:
    """Every GitHub installation for the given tenant, newest first."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT installation_id, account_login, account_type, tenant_id, created_at
        FROM github_installations
        WHERE tenant_id = $1
        ORDER BY updated_at DESC
        """,
        tenant_id,
    )
    return [dict(r) for r in rows]


async def get_github_installation(installation_id: int) -> Optional[dict]:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT installation_id, account_login, account_type, tenant_id FROM github_installations WHERE installation_id=$1",
        installation_id,
    )
    return dict(row) if row else None


async def record_github_push(
    project_id: str,
    installation_id: int,
    repo_owner: str,
    repo_name: str,
    branch: str,
    commit_sha: Optional[str],
    pr_url: Optional[str],
    pr_number: Optional[int],
    files_pushed: int,
    tenant_id: str = "default",
) -> str:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO github_pushes
          (project_id, installation_id, tenant_id, repo_owner, repo_name,
           branch, commit_sha, pr_url, pr_number, files_pushed)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
        """,
        UUID(project_id), installation_id, tenant_id,
        repo_owner, repo_name, branch, commit_sha, pr_url, pr_number, files_pushed,
    )
    return str(row["id"])


async def list_github_pushes_for_project(project_id: str) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, repo_owner, repo_name, branch, commit_sha, pr_url, pr_number,
               files_pushed, pushed_at
        FROM github_pushes
        WHERE project_id = $1
        ORDER BY pushed_at DESC
        """,
        UUID(project_id),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["pushed_at"] = d["pushed_at"].isoformat() if d["pushed_at"] else None
        out.append(d)
    return out


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
