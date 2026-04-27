"""Typed Cypher wrappers for the repo-knowledge graph.

Used by the Coder (Sprint 10b) and any future RepoExplorer agent. Kept
here as a flat module rather than under repo_indexer/ because querying
is a separate concern from writing — different consumer, different test
shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from neo4j import Driver


__all__ = [
    "Definition",
    "CallSite",
    "SymbolMatch",
    "ProcessSummary",
    "ModuleSummary",
    "find_definition",
    "find_callers",
    "find_callees",
    "subclass_tree",
    "find_symbol",
    "find_processes",
    "find_modules",
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
