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
