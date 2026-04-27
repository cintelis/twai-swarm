"""MCP server entry point — `python -m app.mcp_server`.

Sprint 14c. Exposes the indexed repo data as MCP resources + tools so
external clients (Claude Code, Claude Desktop, other agents) can read
the graph without coupling to twai-swarm's internal Coder.

Run as `python -m app.mcp_server` after exporting:
    NEO4J_URL, NEO4J_PASSWORD       — graph connection (required)
    TWAI_REPO_NAME                   — repo to expose (required)
    TWAI_TENANT_ID                   — tenant scope (default: 'default')

Connect from Claude Code:
    claude mcp add twai-swarm -- python -m app.mcp_server

Architecture decisions (per sprint plan):
    * Stdio transport only. No HTTP, no SSE, no auth — the server is a
      child process of an MCP client; the client manages access.
    * Single-tenant, single-repo per process. Multi-tenant routing is a
      v2 concern; would need URI templates with tenant prefixes plus auth.
    * Read-only. Every resource and tool is a Neo4j read; no writes.
    * Backed by `app.repo_query` — the server gets a Driver via
      `loader.driver_from_env` and passes it through. No direct loader
      session use, so any future tenant-filtering in repo_query is
      inherited automatically.
"""
from __future__ import annotations

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from app.mcp_server import resources, tools
from app.repo_indexer.loader import driver_from_env

logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    """Read an env var or exit with a helpful message.

    Stdio MCP servers fail unhelpfully when started without their env —
    the client just gets "server exited" with no diagnostic. We exit
    with a clear stderr message so the operator sees the actual problem.
    """
    value = os.getenv(name, "").strip()
    if not value:
        print(
            f"[twai-mcp] missing required env var: {name}\n"
            f"          set NEO4J_URL, NEO4J_PASSWORD, TWAI_REPO_NAME "
            f"(and optionally TWAI_TENANT_ID) before launching.",
            file=sys.stderr,
        )
        sys.exit(2)
    return value


def main() -> None:
    """Build the FastMCP app, register resources + tools, run stdio."""
    # Validate env early — fail fast before opening the driver.
    repo = _required_env("TWAI_REPO_NAME")
    _required_env("NEO4J_URL")
    _required_env("NEO4J_PASSWORD")
    tenant_id = os.getenv("TWAI_TENANT_ID", "default").strip() or "default"

    logger.info(
        "starting twai-swarm MCP server: repo=%s tenant=%s", repo, tenant_id,
    )

    # Open the Neo4j driver via the loader's env-driven helper. The
    # `with` block in `driver_from_env` closes the pool on shutdown;
    # we want the driver to live for the server's lifetime, so we
    # enter the contextmanager manually and let it tear down on
    # process exit.
    drv_cm = driver_from_env()
    driver = drv_cm.__enter__()

    app = FastMCP(
        name="twai-swarm",
        instructions=(
            "twai-swarm exposes a repository's indexed knowledge graph "
            f"(Functions / Classes / Communities / Processes) for repo "
            f"{repo!r}. Resources at twai://repo/{repo}/* are read-only "
            "URIs; tools wrap free-form queries. Read twai://repo/"
            f"{repo}/context for the full catalog."
        ),
    )

    # ─── Resources ────────────────────────────────────────────────────────
    # FastMCP `@app.resource(uri)` decorator registers a handler. URI
    # templates with `{placeholder}` are matched against the path
    # segments and passed positionally to the function. Templates only
    # support string params — we coerce ints elsewhere if needed.
    #
    # We bind to `repo` from env at registration time, so the actual
    # URIs the client sees are fully qualified (e.g. `twai://repo/
    # twai-swarm/context`, not `twai://repo/{name}/context`). Mirrors
    # the GitNexus shape but instantiated for this server's scope.

    repo_uri = f"twai://repo/{repo}"

    @app.resource("twai://repos")
    def _repos() -> str:
        return resources.build_repos_yaml(driver)

    @app.resource(f"{repo_uri}/context")
    def _context() -> str:
        return resources.build_context_yaml(driver, repo)

    @app.resource(f"{repo_uri}/clusters")
    def _clusters() -> str:
        return resources.build_clusters_yaml(driver, repo)

    @app.resource(f"{repo_uri}/cluster/{{label}}")
    def _cluster_detail(label: str) -> str:
        return resources.build_cluster_detail_yaml(driver, repo, label)

    @app.resource(f"{repo_uri}/processes")
    def _processes() -> str:
        return resources.build_processes_yaml(driver, repo)

    @app.resource(f"{repo_uri}/process/{{name}}")
    def _process_detail(name: str) -> str:
        return resources.build_process_detail_yaml(driver, repo, name)

    # ─── Tools ────────────────────────────────────────────────────────────
    # FastMCP `@app.tool()` infers the JSON schema from type hints +
    # default values, so the docstrings here are user-facing — Claude
    # Code surfaces them in tool-discovery.

    @app.tool(
        name="query",
        description=(
            "Hybrid BM25 + vector search across the indexed repo. Use this "
            "when you have a topic ('auth', 'rate limiting') but not a "
            "qualified name. Returns ranked Functions/Classes."
        ),
    )
    def _query(query: str, limit: int = 10) -> str:
        return tools.query_tool(driver, repo, query, limit=limit)

    @app.tool(
        name="context",
        description=(
            "Full context for one qualified name: definition + callers + "
            "callees. Use this after `query` or `find_symbol` to get the "
            "graph neighbourhood of a specific Function or Class."
        ),
    )
    def _context_tool(qualified_name: str) -> str:
        return tools.context_tool(driver, repo, qualified_name)

    @app.tool(
        name="find_symbol",
        description=(
            "Fuzzy substring lookup across Functions/Classes/Modules. Use "
            "when you have a name fragment ('parse_args') and need to "
            "discover the qualified name for follow-up `context` calls."
        ),
    )
    def _find_symbol(name: str, limit: int = 10) -> str:
        return tools.find_symbol_tool(driver, repo, name, limit=limit)

    # FastMCP.run() blocks; transport defaults to stdio. The driver
    # contextmanager teardown happens after .run() returns (i.e. on
    # client disconnect / EOF on stdin).
    try:
        app.run(transport="stdio")
    finally:
        drv_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
