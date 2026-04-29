"""Typed Cypher wrappers for the repo-knowledge graph.

Used by the Coder (Sprint 10b) and any future RepoExplorer agent. Kept
here as a flat module rather than under repo_indexer/ because querying
is a separate concern from writing — different consumer, different test
shape.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from neo4j import Driver

logger = logging.getLogger(__name__)


__all__ = [
    "Definition",
    "CallSite",
    "SymbolMatch",
    "ProcessSummary",
    "ModuleSummary",
    "ModuleDetail",
    "ProcessStep",
    "ProcessDetail",
    "SemanticHit",
    "find_definition",
    "find_callers",
    "find_callees",
    "subclass_tree",
    "find_symbol",
    "find_processes",
    "find_modules",
    "find_module_detail",
    "find_process_detail",
    "semantic_search",
]


@dataclass(frozen=True)
class Definition:
    qualified_name: str
    kind: str            # "function" | "class"
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""


@dataclass(frozen=True)
class CallSite:
    caller_qn: str       # always a Function
    callee_qn: str
    file_path: str       # caller's file
    line: int            # call-site line


@dataclass(frozen=True)
class SymbolMatch:
    qualified_name: str
    kind: str            # "function" | "class" | "module"
    file_path: str
    line_start: Optional[int] = None


# ─── Sprint 13c — discoverability surface ────────────────────────────────────

@dataclass(frozen=True)
class ProcessSummary:
    """One row from `find_processes`: a named execution flow + its members.

    `member_qns` are in step order — index i corresponds to STEP_IN_PROCESS.step
    = i. `step_count` mirrors `len(member_qns)` and is precomputed for ordering.
    """
    name: str
    summary: str
    step_count: int
    member_qns: tuple[str, ...]   # in step order, 0..N-1


@dataclass(frozen=True)
class ModuleSummary:
    """One row from `find_modules`: a Community node + sample of its members.

    `sample_member_qns` is up to 5 members, sorted lexicographically for
    deterministic output (the underlying graph's member ordering is not stable).
    """
    label: str
    cohesion: float
    size: int
    sample_member_qns: tuple[str, ...]  # up to ~5 representative members


# ─── Sprint 14c — MCP resource layer detail views ────────────────────────────
#
# `find_modules` / `find_processes` are the *list* views (samples + summaries).
# The MCP resource layer also needs *detail* views for one-cluster-by-label
# and one-process-by-name URIs — those want the full member list, not a
# 5-element sample. Same dataclass-and-Cypher style as the list helpers.


@dataclass(frozen=True)
class ModuleDetail:
    """Full detail for a single Community. Backs `twai://repo/{n}/cluster/{label}`.

    Unlike `ModuleSummary`, `member_qns` is the complete list — sorted
    lexicographically for determinism (graph member ordering is not stable
    across runs).
    """
    label: str
    cohesion: float
    size: int
    member_qns: tuple[str, ...]   # ALL members, lexicographically sorted


@dataclass(frozen=True)
class ProcessStep:
    """One ordered step inside a Process flow.

    `step` is the integer index from STEP_IN_PROCESS.step (0..N-1).
    `member_qn` / `file_path` / `line_start` are the projected fields
    from the underlying Function node — enough for an MCP client to
    render a hyperlink to source.
    """
    step: int
    member_qn: str
    file_path: str
    line_start: int


@dataclass(frozen=True)
class ProcessDetail:
    """Full detail for a single Process. Backs `twai://repo/{n}/process/{name}`.

    `steps` are in order — step 0, step 1, ..., step N-1 — so a client
    can read the flow front-to-back without re-sorting.
    """
    name: str
    summary: str
    steps: tuple["ProcessStep", ...]


# ─── Sprint 14b — hybrid semantic search ─────────────────────────────────────

@dataclass(frozen=True)
class SemanticHit:
    """One result from `semantic_search`: a Function or Class plus the fused
    score from BM25 + vector reciprocal-rank-fusion.

    `rrf_score` is exposed for debugging / tunability — the agent doesn't use
    it directly (it gets the ranked list and the score is opaque), but it
    matters to humans iterating on the relevance set.
    """
    qualified_name: str
    name: str
    kind: str             # "function" | "class"
    file_path: str
    line_start: int       # 0 for classes if not stored
    docstring: str        # truncated to ~200 chars
    rrf_score: float


# Truncate docstrings before they cross the wire to the Coder. Long module
# docstrings are wasteful to send and the BM25 leg already used the full
# text for ranking; the agent only needs a sniff for relevance.
_DOCSTRING_TRUNC_CHARS = 200

# Standard RRF damping factor (Cormack et al. 2009). 60 is the canonical
# choice; small enough that rank-1 dominates rank-2 but large enough that
# the long tail still contributes a fractional vote.
_RRF_K_DEFAULT = 60


def find_definition(driver: Driver, repo: str, qualified_name: str) -> Optional[Definition]:
    """Return the definition of `qualified_name` in `repo`, or None.

    Looks under both Function and Class — caller doesn't know which without
    querying, and we don't want to require that knowledge.
    """
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (n {repo: $repo, qualified_name: $qn})
            WHERE n:Function OR n:Class
            RETURN n, labels(n) AS labels
            LIMIT 1
            """,
            repo=repo, qn=qualified_name,
        ).single()
    if rec is None:
        return None
    n = rec["n"]
    kind = "function" if "Function" in rec["labels"] else "class"
    return Definition(
        qualified_name=n["qualified_name"],
        kind=kind,
        file_path=n["file_path"],
        line_start=int(n["line_start"]),
        line_end=int(n["line_end"]),
        docstring=n.get("docstring", "") or "",
    )


def find_callers(driver: Driver, repo: str, qualified_name: str) -> list[CallSite]:
    """Return every CallSite where `qualified_name` is the callee.

    Direction: incoming edges. Use this when refactoring — "what depends
    on this function?" is the question that lets you change a signature
    safely.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (caller:Function)-[r:CALLS]->(callee {repo: $repo, qualified_name: $qn})
            WHERE caller.repo = $repo
            RETURN caller.qualified_name AS caller_qn,
                   callee.qualified_name AS callee_qn,
                   caller.file_path     AS file_path,
                   r.line               AS line
            ORDER BY caller.qualified_name, r.line
            """,
            repo=repo, qn=qualified_name,
        )
        return [CallSite(**row) for row in result.data()]


def find_callees(driver: Driver, repo: str, qualified_name: str) -> list[CallSite]:
    """Outgoing direction — what does `qualified_name` call?

    Use this for blast-radius analysis: "if I change this function,
    what other functions am I leaning on that might also need attention?"
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (caller:Function {repo: $repo, qualified_name: $qn})-[r:CALLS]->(callee)
            RETURN caller.qualified_name AS caller_qn,
                   callee.qualified_name AS callee_qn,
                   caller.file_path      AS file_path,
                   r.line                AS line
            ORDER BY r.line
            """,
            repo=repo, qn=qualified_name,
        )
        return [CallSite(**row) for row in result.data()]


def subclass_tree(driver: Driver, repo: str, root_class_qn: str) -> list[str]:
    """Transitive descendants of `root_class_qn` via INHERITS_FROM.

    Returns a flat list of qualified names; caller can build a tree shape
    if they care. Includes the root class as the first element.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH path = (descendant:Class {repo: $repo})-[:INHERITS_FROM*0..]->(root:Class {repo: $repo, qualified_name: $qn})
            RETURN DISTINCT descendant.qualified_name AS qn
            ORDER BY qn
            """,
            repo=repo, qn=root_class_qn,
        )
        return [row["qn"] for row in result.data()]


# Cypher predicate flagging a `file_path` string as a test-rooted path.
# Bound to a single variable name (`fp`) inside the query so the OR chain
# is the only thing that varies. False positives are fine — this is a
# discoverability filter, not a security boundary. (Sprint 13c.)
_TEST_PATH_PREDICATE = (
    "(fp STARTS WITH 'tests/' OR fp STARTS WITH 'test/' "
    "OR fp CONTAINS '/tests/' OR fp CONTAINS '/test_' "
    "OR fp STARTS WITH 'test_')"
)


def _test_path_predicate_for(var: str) -> str:
    """Return the test-path OR-chain bound to a different Cypher variable.

    `_TEST_PATH_PREDICATE` is hardcoded to `fp`; some queries reference
    the file path under a different alias (`node.file_path`,
    `m.file`, etc.). This helper rewrites the variable so the same
    semantics apply across `find_processes` / `find_modules` /
    `semantic_search`. Sprint 14b extracted this so the predicate
    lives in one place.
    """
    return (
        f"({var} STARTS WITH 'tests/' OR {var} STARTS WITH 'test/' "
        f"OR {var} CONTAINS '/tests/' OR {var} CONTAINS '/test_' "
        f"OR {var} STARTS WITH 'test_')"
    )


def find_symbol(driver: Driver, repo: str, name: str, limit: int = 25) -> list[SymbolMatch]:
    """Fuzzy lookup by bare name across Function / Class / Module.

    Use this as the entry point when you don't know the qualified name
    yet — "find_symbol('parse_args')" gets you the candidates, then you
    follow up with find_definition / find_callers using the QN.

    Match is case-insensitive substring.
    """
    pattern = name.lower()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n {repo: $repo})
            WHERE (n:Function OR n:Class OR n:Module)
              AND toLower(coalesce(n.name, n.qualified_name)) CONTAINS $pattern
            RETURN n.qualified_name AS qualified_name,
                   labels(n)[0]     AS kind,
                   coalesce(n.file_path, '') AS file_path,
                   n.line_start     AS line_start
            ORDER BY size(n.qualified_name), n.qualified_name
            LIMIT $limit
            """,
            repo=repo, pattern=pattern, limit=limit,
        )
        out: list[SymbolMatch] = []
        for row in result.data():
            kind = (row["kind"] or "").lower()
            out.append(SymbolMatch(
                qualified_name=row["qualified_name"],
                kind=kind,
                file_path=row["file_path"],
                line_start=row["line_start"],
            ))
        return out


def find_processes(
    driver: Driver,
    repo: str,
    query: Optional[str] = None,
    limit: int = 10,
    include_tests: bool = False,
) -> list[ProcessSummary]:
    """List Processes (execution flows) in `repo` with their member chains.

    A Process is a chain of CALLS that crosses Community boundaries — see
    `phases/process_extract.py`. Each row carries the chain in step order
    so the Coder can read the flow front-to-back without a follow-up query.

    Args:
        driver: Neo4j driver.
        repo: Repo node `name`.
        query: Optional case-insensitive substring filter on `name` or `summary`.
            Empty/None means no filter.
        limit: Max processes to return. Sorted by step_count DESC, name ASC.
        include_tests: When False (default), drop processes whose chain has
            any test-rooted member (per `_TEST_PATH_PREDICATE`). Set True
            to inspect raw extractor output.
    """
    has_query = bool(query) and query.strip() != ""

    where_clauses: list[str] = []
    if has_query:
        where_clauses.append(
            "(toLower(p.name) CONTAINS toLower($query) "
            "OR toLower(coalesce(p.summary, '')) CONTAINS toLower($query))"
        )
    if not include_tests:
        # `members` is built below as a list of {qn, file}; flag a process
        # for exclusion if ANY member sits under a test path.
        where_clauses.append(
            "none(member IN members WHERE "
            "member.file STARTS WITH 'tests/' "
            "OR member.file STARTS WITH 'test/' "
            "OR member.file CONTAINS '/tests/' "
            "OR member.file CONTAINS '/test_' "
            "OR member.file STARTS WITH 'test_')"
        )
    where_block = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        MATCH (p:Process {{repo: $repo}})-[r:STEP_IN_PROCESS]->(fn:Function)
        WITH p, fn, r.step AS step
        ORDER BY p.name, step
        WITH p, collect({{qn: fn.qualified_name, file: coalesce(fn.file_path, '')}}) AS members
        {where_block}
        WITH p,
             [m IN members | m.qn] AS member_qns,
             size(members) AS step_count
        RETURN p.name AS name,
               coalesce(p.summary, '') AS summary,
               step_count,
               member_qns
        ORDER BY step_count DESC, name ASC
        LIMIT $limit
    """

    params: dict[str, object] = {"repo": repo, "limit": int(limit)}
    if has_query:
        params["query"] = query

    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = result.data()

    return [
        ProcessSummary(
            name=row["name"],
            summary=row["summary"],
            step_count=int(row["step_count"]),
            member_qns=tuple(row["member_qns"]),
        )
        for row in rows
    ]


def find_modules(
    driver: Driver,
    repo: str,
    limit: int = 20,
    include_tests: bool = False,
) -> list[ModuleSummary]:
    """List Communities (modules) in `repo` with up to 5 sample members each.

    A Community is a Louvain cluster over CALLS / IMPORTS / INHERITS_FROM —
    see `phases/community_detect.py`. Singletons (size <= 1) are excluded;
    they're not "modules" in any useful sense.

    Args:
        driver: Neo4j driver.
        repo: Repo node `name`.
        limit: Max communities to return. Sorted by size DESC, label ASC.
        include_tests: When False (default), drop communities whose members
            ALL sit under test paths. A community with even one non-test
            member is kept (mixed clusters are useful to the Coder).
    """
    where_clauses = ["size(members) > 1"]
    if not include_tests:
        # Keep the community if AT LEAST ONE member is non-test. We invert
        # the test predicate per-member then ask `any(...)` over the list.
        where_clauses.append(
            "any(member IN members WHERE NOT ("
            "member.file STARTS WITH 'tests/' "
            "OR member.file STARTS WITH 'test/' "
            "OR member.file CONTAINS '/tests/' "
            "OR member.file CONTAINS '/test_' "
            "OR member.file STARTS WITH 'test_'))"
        )
    where_block = "WHERE " + " AND ".join(where_clauses)

    cypher = f"""
        MATCH (c:Community {{repo: $repo}})<-[:MEMBER_OF]-(m)
        WITH c,
             collect({{qn: m.qualified_name, file: coalesce(m.file_path, '')}}) AS members
        {where_block}
        WITH c,
             [x IN members | x.qn] AS member_qns
        RETURN c.label                       AS label,
               coalesce(c.cohesion, 0.0)     AS cohesion,
               coalesce(c.size, size(member_qns)) AS size,
               member_qns                    AS member_qns
        ORDER BY size DESC, label ASC
        LIMIT $limit
    """

    with driver.session() as session:
        result = session.run(cypher, repo=repo, limit=int(limit))
        rows = result.data()

    out: list[ModuleSummary] = []
    for row in rows:
        all_qns = list(row["member_qns"])
        sample = tuple(sorted(all_qns)[:5])
        out.append(ModuleSummary(
            label=row["label"],
            cohesion=float(row["cohesion"]),
            size=int(row["size"]),
            sample_member_qns=sample,
        ))
    return out


# ─── Sprint 14c — single-detail lookups for the MCP resource layer ──────────


def find_module_detail(
    driver: Driver,
    repo: str,
    label: str,
) -> Optional[ModuleDetail]:
    """Return the full detail of a single Community (cluster), or None.

    Backs the `twai://repo/{name}/cluster/{label}` MCP resource. Unlike
    `find_modules` (which returns a sample), this returns ALL members
    of the cluster sorted lexicographically.

    Returns None when no Community with that (repo, label) pair exists —
    let the caller decide how to render "not found" to the MCP client.
    """
    cypher = """
        MATCH (c:Community {repo: $repo, label: $label})
        OPTIONAL MATCH (c)<-[:MEMBER_OF]-(m)
        WITH c, collect(m.qualified_name) AS member_qns
        RETURN c.label                       AS label,
               coalesce(c.cohesion, 0.0)     AS cohesion,
               coalesce(c.size, size(member_qns)) AS size,
               member_qns                    AS member_qns
        LIMIT 1
    """
    with driver.session() as session:
        rec = session.run(cypher, repo=repo, label=label).single()
    if rec is None:
        return None
    raw = list(rec["member_qns"] or [])
    # Drop any None entries from the OPTIONAL MATCH (cluster with zero
    # members shouldn't happen post-13a, but guard anyway).
    raw = [qn for qn in raw if qn]
    members = tuple(sorted(raw))
    return ModuleDetail(
        label=rec["label"],
        cohesion=float(rec["cohesion"]),
        size=int(rec["size"]),
        member_qns=members,
    )


def find_process_detail(
    driver: Driver,
    repo: str,
    name: str,
) -> Optional[ProcessDetail]:
    """Return the full ordered step trace of a single Process, or None.

    Backs the `twai://repo/{name}/process/{name}` MCP resource. Steps come
    out in `STEP_IN_PROCESS.step` order — i.e. 0, 1, 2, ... — so the MCP
    client can render the chain in execution order without re-sorting.

    Returns None when no Process with that (repo, name) pair exists.
    """
    cypher = """
        MATCH (p:Process {repo: $repo, name: $name})
        OPTIONAL MATCH (p)-[r:STEP_IN_PROCESS]->(fn:Function)
        WITH p, r.step AS step, fn
        ORDER BY step
        WITH p, collect({step: step,
                         qn: fn.qualified_name,
                         file: coalesce(fn.file_path, ''),
                         line_start: coalesce(fn.line_start, 0)}) AS steps
        RETURN p.name                  AS name,
               coalesce(p.summary, '') AS summary,
               steps                   AS steps
        LIMIT 1
    """
    with driver.session() as session:
        rec = session.run(cypher, repo=repo, name=name).single()
    if rec is None:
        return None
    raw_steps = list(rec["steps"] or [])
    # OPTIONAL MATCH yields a single sentinel row when there are no steps;
    # filter rows whose `qn` is None / empty.
    cleaned: list[ProcessStep] = []
    for s in raw_steps:
        qn = s.get("qn") if isinstance(s, dict) else None
        if not qn:
            continue
        cleaned.append(ProcessStep(
            step=int(s.get("step") or 0),
            member_qn=qn,
            file_path=s.get("file") or "",
            line_start=int(s.get("line_start") or 0),
        ))
    return ProcessDetail(
        name=rec["name"],
        summary=rec["summary"] or "",
        steps=tuple(cleaned),
    )


# ─── Sprint 14b — semantic_search ────────────────────────────────────────────


def _embed_query_sync(query: str) -> Optional[list[float]]:
    """Embed `query` synchronously, returning None on any failure.

    The query layer is sync (matches the rest of `repo_query`), but
    `app.embeddings.embed_text` is async. We bridge with `asyncio.run`.
    Failures (no API key, network error, missing openai package) return
    None so the caller can fall back to BM25-only — semantic search is
    a best-effort discoverability surface, not load-bearing.
    """
    try:
        from app.embeddings import embed_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_search: app.embeddings not importable (%s); BM25-only", exc)
        return None
    try:
        return asyncio.run(embed_text(query))
    except RuntimeError as exc:
        # If we're already inside a running event loop (e.g. called from
        # an async tool), `asyncio.run` raises. Schedule on a worker thread
        # via `asyncio.run` from a fresh loop. Fall back to BM25-only if
        # that path also fails — the Coder tool wraps this in
        # `asyncio.to_thread` so a fresh thread should mean a fresh loop.
        logger.warning("semantic_search: embed_text scheduling failed (%s); BM25-only", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_search: embed_text failed (%s); BM25-only", exc)
        return None


def _bm25_leg(
    driver: Driver,
    repo: str,
    query: str,
    candidate_limit: int,
    include_tests: bool,
) -> tuple[Optional[list[dict]], Optional[Exception]]:
    """Run the full-text leg. Returns (rows, None) on success or
    (None, exc) if the index is missing / the query failed.

    Sorted by `score DESC` in Python before the caller assigns ranks.
    """
    test_filter = "" if include_tests else f" AND NOT {_test_path_predicate_for('node.file_path')}"
    cypher = f"""
        CALL db.index.fulltext.queryNodes('function_text', $query) YIELD node, score
        WHERE node.repo = $repo{test_filter}
        RETURN node.qualified_name AS qualified_name,
               node.name AS name,
               coalesce(node.file_path, '') AS file_path,
               coalesce(node.line_start, 0) AS line_start,
               coalesce(node.docstring, '') AS docstring,
               score AS score,
               'function' AS kind
        LIMIT $candidate_limit
        UNION
        CALL db.index.fulltext.queryNodes('class_text', $query) YIELD node, score
        WHERE node.repo = $repo{test_filter}
        RETURN node.qualified_name AS qualified_name,
               node.name AS name,
               coalesce(node.file_path, '') AS file_path,
               coalesce(node.line_start, 0) AS line_start,
               coalesce(node.docstring, '') AS docstring,
               score AS score,
               'class' AS kind
        LIMIT $candidate_limit
    """
    try:
        with driver.session() as session:
            # Parameters passed as a dict to avoid colliding with
            # session.run's own `query` positional parameter (the cypher
            # string). Passing `query=query` as kwarg would re-bind that
            # positional and raise TypeError("multiple values for query").
            result = session.run(
                cypher,
                {"repo": repo, "query": query, "candidate_limit": int(candidate_limit)},
            )
            rows = result.data()
    except Exception as exc:  # noqa: BLE001
        return None, exc
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, None


def _vector_leg(
    driver: Driver,
    repo: str,
    query_vec: list[float],
    candidate_limit: int,
    include_tests: bool,
) -> tuple[Optional[list[dict]], Optional[Exception]]:
    """Run the vector leg. Returns (rows, None) on success or (None, exc)
    if the vector index is missing / the query failed.

    Sorted by `score DESC` in Python before the caller assigns ranks.
    """
    test_filter = "" if include_tests else f" AND NOT {_test_path_predicate_for('node.file_path')}"
    cypher = f"""
        CALL db.index.vector.queryNodes('function_embedding', $candidate_limit, $query_vec)
            YIELD node, score
        WHERE node.repo = $repo{test_filter}
        RETURN node.qualified_name AS qualified_name,
               node.name AS name,
               coalesce(node.file_path, '') AS file_path,
               coalesce(node.line_start, 0) AS line_start,
               coalesce(node.docstring, '') AS docstring,
               score AS score,
               'function' AS kind
        UNION
        CALL db.index.vector.queryNodes('class_embedding', $candidate_limit, $query_vec)
            YIELD node, score
        WHERE node.repo = $repo{test_filter}
        RETURN node.qualified_name AS qualified_name,
               node.name AS name,
               coalesce(node.file_path, '') AS file_path,
               coalesce(node.line_start, 0) AS line_start,
               coalesce(node.docstring, '') AS docstring,
               score AS score,
               'class' AS kind
    """
    try:
        with driver.session() as session:
            result = session.run(
                cypher, repo=repo, query_vec=query_vec,
                candidate_limit=int(candidate_limit),
            )
            rows = result.data()
    except Exception as exc:  # noqa: BLE001
        return None, exc
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, None


def semantic_search(
    driver: Driver,
    repo: str,
    query: str,
    k: int = 10,
    include_tests: bool = False,
    rrf_k: int = _RRF_K_DEFAULT,
) -> list[SemanticHit]:
    """Hybrid BM25 + vector search. Returns top-k results ranked by RRF.

    Both legs run sequentially (the Neo4j driver doesn't trivially
    parallelise two queries from one session — measure and revisit if
    it's a hot path). Each leg pulls top 2*k candidates so the fusion
    has headroom; the final list is truncated to k.

    Args:
        driver: Neo4j driver.
        repo: Repo node `name` to filter on.
        query: Natural-language query string. Empty / whitespace-only
            returns an empty list immediately (no Neo4j round-trip).
        k: Max results to return. Default 10.
        include_tests: When False (default), drop hits whose `file_path`
            matches `_test_path_predicate_for(...)`. Power users can
            override at this layer; the Coder tool always passes False.
        rrf_k: Reciprocal-rank-fusion damping factor. Default 60
            (`_RRF_K_DEFAULT`); standard choice from the literature.
            Smaller values give rank-1 a heavier weight; larger values
            flatten the curve. The agent never sees this — exposed for
            relevance-tuning experiments.

    Edge cases:
        * Empty query → `[]`.
        * Embedding call fails (no API key, network, etc.) → BM25-only
          result with a logged warning.
        * Full-text index missing → vector-only result with a logged
          warning.
        * Both indexes missing → `[]` with a logged warning.

    Returns:
        Up to `k` `SemanticHit` records, ranked by fused RRF score
        descending. Docstrings are truncated to ~200 chars.
    """
    if not query or not query.strip():
        return []
    candidate_limit = max(1, int(k) * 2)

    # 1. Embed (best-effort).
    query_vec = _embed_query_sync(query)

    # 2. BM25 leg.
    bm25_rows, bm25_err = _bm25_leg(driver, repo, query, candidate_limit, include_tests)
    if bm25_err is not None:
        logger.warning(
            "semantic_search: BM25 leg failed (%s); falling back to vector-only",
            bm25_err,
        )
        bm25_rows = []

    # 3. Vector leg (only if we have a query vector).
    if query_vec is None:
        vector_rows: list[dict] = []
        vector_err: Optional[Exception] = None
    else:
        vector_rows_or_none, vector_err = _vector_leg(
            driver, repo, query_vec, candidate_limit, include_tests,
        )
        if vector_err is not None:
            logger.warning(
                "semantic_search: vector leg failed (%s); falling back to BM25-only",
                vector_err,
            )
            vector_rows = []
        else:
            vector_rows = vector_rows_or_none or []

    # 4. Both legs empty AND at least one had an error → nothing usable.
    if not bm25_rows and not vector_rows:
        if bm25_err is not None or vector_err is not None or query_vec is None:
            logger.warning(
                "semantic_search: no results and at least one leg unavailable "
                "(bm25_err=%s vector_err=%s query_vec_present=%s)",
                bm25_err, vector_err, query_vec is not None,
            )
        return []

    # 5. Reciprocal Rank Fusion. For each unique node (keyed on
    #    qualified_name + kind), sum 1 / (rrf_k + rank_i) over the legs
    #    in which it appears. Standard formula.
    #
    #    Worked example (rrf_k=60):
    #       BM25 ranks: A=1, B=2, C=3
    #       Vector ranks: B=1, A=2, D=3
    #       A: 1/(60+1) + 1/(60+2) ≈ 0.01639 + 0.01613 ≈ 0.03252
    #       B: 1/(60+2) + 1/(60+1) ≈ 0.01613 + 0.01639 ≈ 0.03252
    #       C: 1/(60+3)            ≈ 0.01587
    #       D: 1/(60+3)            ≈ 0.01587
    #    A and B tie because they swap ranks symmetrically; C and D tie
    #    for the same reason. Tie-broken by qualified_name asc below.
    fused: dict[tuple[str, str], dict] = {}
    for rank, row in enumerate(bm25_rows, start=1):
        key = (row["qualified_name"], row["kind"])
        contrib = 1.0 / (rrf_k + rank)
        if key in fused:
            fused[key]["rrf"] += contrib
        else:
            fused[key] = {"row": row, "rrf": contrib}

    for rank, row in enumerate(vector_rows, start=1):
        key = (row["qualified_name"], row["kind"])
        contrib = 1.0 / (rrf_k + rank)
        if key in fused:
            fused[key]["rrf"] += contrib
        else:
            fused[key] = {"row": row, "rrf": contrib}

    # 6. Sort by RRF desc; tie-break on qualified_name asc for determinism.
    ranked = sorted(
        fused.values(),
        key=lambda e: (-e["rrf"], e["row"]["qualified_name"]),
    )[: int(k)]

    out: list[SemanticHit] = []
    for entry in ranked:
        row = entry["row"]
        docstring = (row.get("docstring") or "")[:_DOCSTRING_TRUNC_CHARS]
        out.append(SemanticHit(
            qualified_name=row["qualified_name"],
            name=row.get("name") or row["qualified_name"].rsplit(".", 1)[-1],
            kind=row["kind"],
            file_path=row.get("file_path") or "",
            line_start=int(row.get("line_start") or 0),
            docstring=docstring,
            rrf_score=float(entry["rrf"]),
        ))
    return out
