"""Tool handlers for the Sprint 14c MCP server.

Three tools:
    query        — hybrid BM25 + vector search (semantic_search)
    context      — definition + callers + callees for one qualified name
    find_symbol  — fuzzy name lookup across Functions/Classes/Modules

Each tool function is a pure callable: it takes a Neo4j `Driver` plus
the tool args and returns a YAML string. The MCP framing (decorators,
JSON schema generation) lives in `__main__.py`. Keeping the handlers
framework-free means they can be unit-tested without the stdio
runtime — see `tests/test_mcp_server.py`.

Tools (callable) versus Resources (read-by-URI): the cut here mirrors
GitNexus's MCP server. List/detail views with deterministic URIs are
resources; anything that takes a free-form parameter (a query string,
a partial name) is a tool.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import yaml
from neo4j import Driver

from app import repo_query

logger = logging.getLogger(__name__)


def _dump(payload: dict[str, Any] | list[Any]) -> str:
    """Stable YAML dump — same shape as resources._dump (kept private to
    each module rather than shared so tools.py and resources.py have no
    cross-import; the duplication is one line)."""
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def query_tool(
    driver: Driver,
    repo: str,
    query: str,
    limit: int = 10,
) -> str:
    """Hybrid BM25 + vector search. Returns top-`limit` `SemanticHit`s as YAML.

    Wraps `repo_query.semantic_search`. Empty query returns an empty list
    YAML (no Neo4j round-trip — semantic_search does this short-circuit
    internally and we surface its empty result faithfully).
    """
    hits = repo_query.semantic_search(driver, repo, query, k=int(limit))
    payload = {
        "repo": repo,
        "query": query,
        "count": len(hits),
        "hits": [
            {
                "qualified_name": h.qualified_name,
                "name": h.name,
                "kind": h.kind,
                "file_path": h.file_path,
                "line_start": h.line_start,
                "docstring": h.docstring,
                "rrf_score": round(h.rrf_score, 6),
            }
            for h in hits
        ],
    }
    return _dump(payload)


def context_tool(
    driver: Driver,
    repo: str,
    qualified_name: str,
) -> str:
    """Definition + callers + callees for one qualified name.

    The tool aggregates three `repo_query` calls — `find_definition`,
    `find_callers`, `find_callees` — into a single payload so an MCP
    client gets the full context for a symbol in one round-trip.

    Returns `found: false` when the qualified name has no Function/Class
    node — the symbol may exist as a `Symbol` placeholder elsewhere, but
    that's not actionable for the agent.
    """
    definition = repo_query.find_definition(driver, repo, qualified_name)
    if definition is None:
        payload = {
            "repo": repo,
            "qualified_name": qualified_name,
            "found": False,
            "message": (
                f"no Function/Class with qualified_name {qualified_name!r} "
                f"in repo {repo!r}"
            ),
        }
        return _dump(payload)

    callers = repo_query.find_callers(driver, repo, qualified_name)
    callees = repo_query.find_callees(driver, repo, qualified_name)

    payload = {
        "repo": repo,
        "qualified_name": qualified_name,
        "found": True,
        "definition": {
            "qualified_name": definition.qualified_name,
            "kind": definition.kind,
            "file_path": definition.file_path,
            "line_start": definition.line_start,
            "line_end": definition.line_end,
            "docstring": definition.docstring,
        },
        "callers": [asdict(c) for c in callers],
        "callees": [asdict(c) for c in callees],
    }
    return _dump(payload)


def find_symbol_tool(
    driver: Driver,
    repo: str,
    name: str,
    limit: int = 10,
) -> str:
    """Fuzzy substring lookup across Function / Class / Module.

    Wraps `repo_query.find_symbol`. The Coder's primary "I have a name,
    what is it?" entry point — the resource layer can't expose this
    because the input is a free-form fuzzy match, not a stable URI.
    """
    matches = repo_query.find_symbol(driver, repo, name, limit=int(limit))
    payload = {
        "repo": repo,
        "name": name,
        "count": len(matches),
        "matches": [
            {
                "qualified_name": m.qualified_name,
                "kind": m.kind,
                "file_path": m.file_path,
                "line_start": m.line_start,
            }
            for m in matches
        ],
    }
    return _dump(payload)
