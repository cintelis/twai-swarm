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
import os
import sys
import asyncpg

# Bootstrap reads PG_DSN directly instead of importing app.config — the
# config module validates LLM keys at import time, which the bootstrap
# task doesn't have (and shouldn't need).
PG_DSN = os.environ.get("PG_DSN")
if not PG_DSN:
    print("[bootstrap] PG_DSN env var is required", file=sys.stderr)
    sys.exit(1)

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

-- GitHub App installations. One row per tenant×GitHub-account-they-installed-on.
-- tenant_id is forward-compat: today everyone is 'default'; greenfield's tenant
-- middleware will set this from JWT claims. Don't drop the column when migrating.
CREATE TABLE IF NOT EXISTS github_installations (
    installation_id BIGINT PRIMARY KEY,
    account_login   TEXT NOT NULL,         -- 'cintelis' or 'acme-corp'
    account_type    TEXT NOT NULL,         -- 'Organization' or 'User'
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_github_installations_tenant
    ON github_installations(tenant_id);

-- Push history per project. Lets the UI show "this scaffold was pushed to X
-- on date Y" without re-querying GitHub on every page render.
CREATE TABLE IF NOT EXISTS github_pushes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    installation_id BIGINT NOT NULL REFERENCES github_installations(installation_id),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    repo_owner      TEXT NOT NULL,
    repo_name       TEXT NOT NULL,
    branch          TEXT NOT NULL,
    commit_sha      TEXT,
    pr_url          TEXT,
    pr_number       INT,
    files_pushed    INT,
    pushed_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_github_pushes_project
    ON github_pushes(project_id, pushed_at DESC);
"""

async def _setup_langfuse_db(admin_conn: asyncpg.Connection) -> None:
    """Create the `langfuse` database + `langfuse_app` user if they don't exist.

    Runs against the admin connection (default `postgres` database). Idempotent:
    uses `IF NOT EXISTS` / `CREATE USER IF NOT EXISTS` patterns. Safe to run
    on every bootstrap; does nothing when the setup is already complete.

    Requires LANGFUSE_DB_PASSWORD env var (set by ECS from SM). Skips silently
    if unset — supports local dev without Langfuse.
    """
    password = os.environ.get("LANGFUSE_DB_PASSWORD")
    if not password:
        print("[bootstrap] LANGFUSE_DB_PASSWORD unset — skipping Langfuse DB setup")
        return

    # Check if user exists
    user_exists = await admin_conn.fetchval(
        "SELECT 1 FROM pg_roles WHERE rolname='langfuse_app'"
    )
    if not user_exists:
        # CREATE USER doesn't support $1 parameter binding for passwords; safely
        # quote the password using pg's literal quoting. The password comes from
        # SM (32 chars, no special chars because random_password.special=false).
        safe_pw = password.replace("'", "''")
        await admin_conn.execute(
            f"CREATE USER langfuse_app WITH PASSWORD '{safe_pw}'"
        )
        print("[bootstrap] created langfuse_app user")

    # CREATE DATABASE can't run inside a transaction; asyncpg runs each execute
    # in its own implicit txn unless we say otherwise. This should work via
    # asyncpg's direct query path.
    db_exists = await admin_conn.fetchval(
        "SELECT 1 FROM pg_database WHERE datname='langfuse'"
    )
    if not db_exists:
        await admin_conn.execute(
            "CREATE DATABASE langfuse OWNER langfuse_app"
        )
        print("[bootstrap] created langfuse database")
    else:
        # If the DB already exists but was owned by postgres (first-time race
        # or manual creation), re-assign ownership so migrations succeed.
        await admin_conn.execute("ALTER DATABASE langfuse OWNER TO langfuse_app")

    # Grant the app user full perms on the database.
    await admin_conn.execute(
        "GRANT ALL PRIVILEGES ON DATABASE langfuse TO langfuse_app"
    )
    print("[bootstrap] ✅ Langfuse DB ready")


async def main() -> int:
    print("[bootstrap] connecting to DB...")
    try:
        conn = await asyncpg.connect(PG_DSN)
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

        # Langfuse DB setup — runs against the admin connection (default
        # `postgres` database, where CREATE DATABASE is allowed).
        # Derive admin DSN by swapping the database name in PG_DSN.
        admin_dsn = PG_DSN.rsplit("/", 1)[0] + "/postgres"
        try:
            admin_conn = await asyncpg.connect(admin_dsn)
            try:
                await _setup_langfuse_db(admin_conn)
            finally:
                await admin_conn.close()
        except Exception as e:
            # Don't fail the whole bootstrap if Langfuse setup has issues —
            # the swarm's own schema is already applied above.
            print(f"[bootstrap] Langfuse DB setup failed (non-fatal): {e}", file=sys.stderr)

        print("[bootstrap] ✅ done")
        return 0
    except Exception as e:
        print(f"[bootstrap] schema apply failed: {e}", file=sys.stderr)
        return 1
    finally:
        await conn.close()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
