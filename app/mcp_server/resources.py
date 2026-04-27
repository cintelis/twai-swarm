"""Resource handlers for the Sprint 14c MCP server.

Each `build_*_yaml` function is a pure callable — it takes a Neo4j
`Driver` plus path params and returns a YAML string. The MCP framing
(URI templates, registration) lives in `__main__.py`; keeping the
handlers framework-free makes them unit-testable without spinning up
the stdio runtime.

Resource catalog (mirrors GitNexus):
    twai://repos                              -> all Repo nodes registered
    twai://repo/{name}/context                -> overview + tool catalog
    twai://repo/{name}/clusters               -> all Communities, ranked
    twai://repo/{name}/cluster/{label}        -> one Community's detail
    twai://repo/{name}/processes              -> all Processes, ranked
    twai://repo/{name}/process/{name}         -> one Process's full trace

YAML format because Claude Desktop renders it nicely. We use `pyyaml`
(already a transitive dep via langfuse / temporal — verified before
adding any usage). `yaml.safe_dump(..., sort_keys=False)` keeps our
deliberately-ordered dicts intact (e.g. context comes out as
`name → stats → tools`, not alphabetized).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import yaml
from neo4j import Driver

from app import repo_query

logger = logging.getLogger(__name__)


# Tool catalog string — included in the `context` resource so an MCP
# client (or human reading it) sees the full surface in one place
# without having to call list_tools separately.
_TOOL_CATALOG: list[dict[str, str]] = [
    {
        "name": "query",
        "summary": "Hybrid BM25 + vector search for code about a topic.",
    },
    {
        "name": "context",
        "summary": "Definition + callers + callees for one qualified name.",
    },
    {
        "name": "find_symbol",
        "summary": "Fuzzy name lookup across Functions, Classes, Modules.",
    },
]


def _dump(payload: dict[str, Any] | list[Any]) -> str:
    """Stable YAML dump.

    `sort_keys=False` because we order keys deliberately (name first,
    stats second, tools last in `context`); `default_flow_style=False`
    so lists come out block-style — easier to read in a terminal.
    """
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def build_repos_yaml(driver: Driver) -> str:
    """`twai://repos` — list every `Repo` node in this Neo4j.

    Used by multi-repo MCP clients to discover what's available; the
    server itself is single-repo per process, but a human inspecting
    the graph from Claude Desktop wants the full inventory.
    """
    cypher = """
        MATCH (r:Repo)
        RETURN r.name        AS name,
               coalesce(r.url, '')        AS url,
               coalesce(r.commit_sha, '') AS commit_sha,
               coalesce(r.tenant_id, '')  AS tenant_id
        ORDER BY r.name
    """
    with driver.session() as session:
        rows = session.run(cypher).data()
    payload = {
        "repos": [
            {
                "name": r["name"],
                "url": r["url"],
                "commit_sha": r["commit_sha"],
                "tenant_id": r["tenant_id"],
            }
            for r in rows
        ],
    }
    return _dump(payload)


def _repo_exists(driver: Driver, repo: str) -> bool:
    """True iff a `Repo {name: $repo}` node exists."""
    with driver.session() as session:
        rec = session.run(
            "MATCH (r:Repo {name: $repo}) RETURN r.name AS name LIMIT 1",
            repo=repo,
        ).single()
    return rec is not None


def _repo_stats(driver: Driver, repo: str) -> dict[str, int]:
    """File / Module / Class / Function / Community / Process counts.

    All in one Cypher round-trip via OPTIONAL MATCH count() aggregations
    to keep the context resource cheap.
    """
    cypher = """
        OPTIONAL MATCH (f:File {repo: $repo})        WITH count(f)  AS files
        OPTIONAL MATCH (m:Module {repo: $repo})      WITH files, count(m)  AS modules
        OPTIONAL MATCH (c:Class {repo: $repo})       WITH files, modules, count(c)  AS classes
        OPTIONAL MATCH (fn:Function {repo: $repo})   WITH files, modules, classes, count(fn) AS functions
        OPTIONAL MATCH (cm:Community {repo: $repo})  WITH files, modules, classes, functions, count(cm) AS communities
        OPTIONAL MATCH (p:Process {repo: $repo})
        RETURN files, modules, classes, functions, communities, count(p) AS processes
    """
    with driver.session() as session:
        rec = session.run(cypher, repo=repo).single()
    if rec is None:
        return {
            "files": 0, "modules": 0, "classes": 0,
            "functions": 0, "communities": 0, "processes": 0,
        }
    return {
        "files": int(rec["files"] or 0),
        "modules": int(rec["modules"] or 0),
        "classes": int(rec["classes"] or 0),
        "functions": int(rec["functions"] or 0),
        "communities": int(rec["communities"] or 0),
        "processes": int(rec["processes"] or 0),
    }


def build_context_yaml(driver: Driver, repo: str) -> str:
    """`twai://repo/{name}/context` — a top-level overview YAML.

    Includes node counts (files, modules, classes, functions,
    communities, processes) and the tool catalog so an MCP client lands
    here first and learns everything else it can ask for.

    Missing-repo handling: emits `available: false` plus the empty
    counts dict rather than crashing, so the MCP client gets a useful
    rendering instead of an MCP error.
    """
    if not _repo_exists(driver, repo):
        payload = {
            "name": repo,
            "available": False,
            "message": f"no Repo node named {repo!r} in this Neo4j",
            "tools": _TOOL_CATALOG,
        }
        return _dump(payload)

    stats = _repo_stats(driver, repo)
    payload = {
        "name": repo,
        "available": True,
        "stats": stats,
        "tools": _TOOL_CATALOG,
    }
    return _dump(payload)


def build_clusters_yaml(driver: Driver, repo: str) -> str:
    """`twai://repo/{name}/clusters` — all Communities, ranked size DESC.

    Backed by `repo_query.find_modules` with a high limit (200) since
    this is the full list view; the Coder's tool form is paginated, but
    the resource is meant to be the canonical "what modules exist" view.
    """
    modules = repo_query.find_modules(driver, repo, limit=200)
    payload = {
        "repo": repo,
        "count": len(modules),
        "clusters": [
            {
                "label": m.label,
                "cohesion": round(m.cohesion, 4),
                "size": m.size,
                "sample_member_qns": list(m.sample_member_qns),
            }
            for m in modules
        ],
    }
    return _dump(payload)


def build_cluster_detail_yaml(
    driver: Driver,
    repo: str,
    label: str,
) -> str:
    """`twai://repo/{name}/cluster/{label}` — one Community's full detail.

    Returns the COMPLETE member list (not the 5-element sample). When
    the cluster doesn't exist, emits a `found: false` payload — keeps
    the MCP client's error path uniform across "missing repo" and
    "missing cluster".
    """
    detail: Optional[repo_query.ModuleDetail] = repo_query.find_module_detail(
        driver, repo, label,
    )
    if detail is None:
        payload = {
            "repo": repo,
            "label": label,
            "found": False,
            "message": f"no Community with label {label!r} in repo {repo!r}",
        }
        return _dump(payload)

    payload = {
        "repo": repo,
        "label": detail.label,
        "found": True,
        "cohesion": round(detail.cohesion, 4),
        "size": detail.size,
        "member_qns": list(detail.member_qns),
    }
    return _dump(payload)


def build_processes_yaml(driver: Driver, repo: str) -> str:
    """`twai://repo/{name}/processes` — all Processes, ranked step_count DESC.

    Same `find_processes` call the Coder uses, but with a high limit
    (200) for the full list-view shape.
    """
    processes = repo_query.find_processes(driver, repo, limit=200)
    payload = {
        "repo": repo,
        "count": len(processes),
        "processes": [
            {
                "name": p.name,
                "summary": p.summary,
                "step_count": p.step_count,
                "member_qns": list(p.member_qns),
            }
            for p in processes
        ],
    }
    return _dump(payload)


def build_process_detail_yaml(
    driver: Driver,
    repo: str,
    name: str,
) -> str:
    """`twai://repo/{name}/process/{name}` — one Process's full step trace.

    Each step carries `step`, `member_qn`, `file_path`, `line_start` so
    an MCP client can render a clickable trace. Steps are 0..N-1 ordered.
    Missing-process emits `found: false` — same shape as cluster detail.
    """
    detail: Optional[repo_query.ProcessDetail] = repo_query.find_process_detail(
        driver, repo, name,
    )
    if detail is None:
        payload = {
            "repo": repo,
            "process": name,
            "found": False,
            "message": f"no Process named {name!r} in repo {repo!r}",
        }
        return _dump(payload)

    payload = {
        "repo": repo,
        "process": detail.name,
        "found": True,
        "summary": detail.summary,
        "step_count": len(detail.steps),
        "steps": [
            {
                "step": s.step,
                "member_qn": s.member_qn,
                "file_path": s.file_path,
                "line_start": s.line_start,
            }
            for s in detail.steps
        ],
    }
    return _dump(payload)
