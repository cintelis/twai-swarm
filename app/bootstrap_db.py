"""
One-shot DB bootstrap. Runs the init SQL against the configured PG_DSN.

Idempotent: uses CREATE IF NOT EXISTS where possible. Safe to re-run.

The init.sql lives in ./db/init.sql in the repo; we ship it inside the image
so this script can find it at /app/db/init.sql. But because the Dockerfile
only COPYs `app`, we instead inline the SQL OR copy it in. Simpler: inline
the schema here so there's one source of truth for the container.

Note: this mirrors db/init.sql. If you change one, change the other -- or
better, refactor to read from a packaged resource. For a lean starter, dupe is fine.
"""
import asyncio
import sys
import asyncpg

from app import config

# Kept identical to db/init.sql but authored to work against an already-created
# database (the RDS instance was created by Terraform with db_name=agentdb).
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    brief TEXT NOT NULL,
    workflow_id TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL,
    parent_task_id UUID REFERENCES tasks(id),
    role TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    input JSONB,
    output JSONB,
    provider TEXT,
    model_used TEXT,
    tokens_in INT,
    tokens_out INT,
    cost_usd NUMERIC(10, 6),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS task_embeddings (
    task_id UUID PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_task_embeddings_vec ON task_embeddings
    USING hnsw (embedding vector_cosine_ops);
"""

async def main() -> int:
    print(f"[bootstrap] connecting to DB...")
    try:
        conn = await asyncpg.connect(config.PG_DSN)
    except Exception as e:
        print(f"[bootstrap] connection failed: {e}", file=sys.stderr)
        return 1

    try:
        print("[bootstrap] applying schema...")
        await conn.execute(SCHEMA_SQL)

        # Quick sanity check -- list our tables so you can see them in CW logs
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        )
        print(f"[bootstrap] tables: {[r['tablename'] for r in rows]}")

        # Confirm pgvector is actually installed
        ext = await conn.fetchrow("SELECT extversion FROM pg_extension WHERE extname='vector'")
        print(f"[bootstrap] pgvector version: {ext['extversion'] if ext else 'NOT INSTALLED'}")

        print("[bootstrap] ✅ done")
        return 0
    except Exception as e:
        print(f"[bootstrap] schema apply failed: {e}", file=sys.stderr)
        return 1
    finally:
        await conn.close()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
