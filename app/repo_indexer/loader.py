"""Neo4j loader — write an IndexBatch via batched MERGE statements.

One transaction per node-type and per edge-type, all via UNWIND $rows
parameterised queries. This keeps round-trips low (~10 per scan) without
needing the extractor to know any Cypher.

Idempotency: every MERGE is keyed on (repo, qualified_name) for nodes
and on (repo, caller, callee) for edges, so re-running the loader on
the same data is a no-op. Stale-deletion is the caller's job (see
prune_stale_for_commit) — the loader only writes.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator

from neo4j import Driver, GraphDatabase, basic_auth

from .actions import IndexBatch

logger = logging.getLogger(__name__)


@contextmanager
def driver_from_env() -> Iterator[Driver]:
    """Yield a Neo4j driver wired to the worker's NEO4J_URL/PASSWORD env.

    Caller is responsible for the `with` block scope — the driver pools
    connections internally; one driver per process is the right pattern.
    """
    url = os.getenv("NEO4J_URL", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not url or not password:
        raise RuntimeError(
            "NEO4J_URL and NEO4J_PASSWORD must be set; check terraform secrets are wired"
        )
    drv = GraphDatabase.driver(url, auth=basic_auth("neo4j", password))
    try:
        yield drv
    finally:
        drv.close()


def ensure_constraints(driver: Driver) -> None:
    """Idempotent uniqueness constraints. Cheap to re-run on every scan."""
    stmts = [
        # Composite uniqueness — same qualified_name across different repos
        # is allowed and expected (forks, multiple tenants, etc.).
        "CREATE CONSTRAINT repo_name IF NOT EXISTS FOR (r:Repo) REQUIRE r.name IS UNIQUE",
        "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE (f.repo, f.path) IS UNIQUE",
        "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE (m.repo, m.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE (c.repo, c.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE (fn.repo, fn.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (s:Symbol) REQUIRE (s.repo, s.qualified_name) IS UNIQUE",
    ]
    with driver.session() as session:
        for stmt in stmts:
            session.run(stmt)


WRITE_CHUNK_SIZE = 1000


def _chunked_write(session, query: str, rows: list, chunk_size: int = WRITE_CHUNK_SIZE) -> None:
    """Run a parameterised Cypher query in chunks of `chunk_size` rows.

    Single-shot writes of 100K+ rows can blow Bolt's packet size and trip
    NLB-fronted TLS connections (we saw `An existing connection was forcibly
    closed` mid-write on the 13K-file OpenClaw scan). Chunking keeps each
    write under ~1 MB even for the chunky CallEdge payloads.
    """
    if not rows:
        return
    for i in range(0, len(rows), chunk_size):
        session.run(query, rows=rows[i:i + chunk_size])


def write_batch(driver: Driver, batch: IndexBatch) -> None:
    """Write one IndexBatch via batched UNWIND-MERGE writes.

    Each node/edge type is one query, called multiple times when the row
    count exceeds WRITE_CHUNK_SIZE — keeps Bolt round-trips low without
    overflowing any single packet.
    """
    repo = batch.repo

    with driver.session() as session:
        # Repo node first — everything else FKs to it.
        session.run(
            """
            MERGE (r:Repo {name: $name})
            SET r.url = $url,
                r.commit_sha = $commit_sha,
                r.tenant_id = $tenant_id,
                r.scanned_at = datetime()
            """,
            name=repo.name, url=repo.url, commit_sha=repo.commit_sha,
            tenant_id=repo.tenant_id,
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (r:Repo {name: row.repo})
            MERGE (f:File {repo: row.repo, path: row.path})
            SET f.language = row.language, f.sha = row.sha
            MERGE (r)-[:CONTAINS]->(f)
            """,
            [asdict(f) for f in batch.files],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (f:File {repo: row.repo, path: row.file_path})
            MERGE (m:Module {repo: row.repo, qualified_name: row.qualified_name})
            MERGE (f)-[:DEFINES]->(m)
            """,
            [asdict(m) for m in batch.modules],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MERGE (c:Class {repo: row.repo, qualified_name: row.qualified_name})
            SET c.name = row.name,
                c.file_path = row.file_path,
                c.line_start = row.line_start,
                c.line_end = row.line_end,
                c.docstring = row.docstring
            WITH c, row
            OPTIONAL MATCH (m:Module {repo: row.repo})<-[:DEFINES]-(:File {repo: row.repo, path: row.file_path})
            FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
              MERGE (m)-[:DEFINES]->(c)
            )
            """,
            [asdict(c) for c in batch.classes],
        )

        # Functions get two writes (define + module link). Pre-build the
        # asdict rows ONCE so we don't burn CPU on the rebuild for chunk #2.
        function_rows = [{**asdict(fn), "params": list(fn.params)} for fn in batch.functions]
        _chunked_write(session,
            """
            UNWIND $rows AS row
            MERGE (fn:Function {repo: row.repo, qualified_name: row.qualified_name})
            SET fn.name = row.name,
                fn.file_path = row.file_path,
                fn.line_start = row.line_start,
                fn.line_end = row.line_end,
                fn.is_async = row.is_async,
                fn.is_method = row.is_method,
                fn.params = row.params,
                fn.docstring = row.docstring
            WITH fn, row
            FOREACH (_ IN CASE WHEN row.parent_class_qn = '' THEN [] ELSE [1] END |
              MERGE (parent:Class {repo: row.repo, qualified_name: row.parent_class_qn})
              MERGE (parent)-[:DEFINES]->(fn)
            )
            """,
            function_rows,
        )
        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (fn:Function {repo: row.repo, qualified_name: row.qualified_name})
            MATCH (f:File {repo: row.repo, path: row.file_path})-[:DEFINES]->(m:Module)
            WHERE row.parent_class_qn = ''
            MERGE (m)-[:DEFINES]->(fn)
            """,
            function_rows,
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MERGE (s:Symbol {repo: row.repo, qualified_name: row.qualified_name})
            SET s.name = row.name
            """,
            [asdict(s) for s in batch.symbols],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (child:Class {repo: row.repo, qualified_name: row.child_qn})
            OPTIONAL MATCH (parent_class:Class {repo: row.repo, qualified_name: row.parent_qn})
            OPTIONAL MATCH (parent_sym:Symbol {repo: row.repo, qualified_name: row.parent_qn})
            FOREACH (p IN CASE WHEN parent_class IS NOT NULL THEN [parent_class]
                               WHEN parent_sym   IS NOT NULL THEN [parent_sym]
                               ELSE [] END |
              MERGE (child)-[:INHERITS_FROM]->(p)
            )
            """,
            [asdict(e) for e in batch.inherits],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (caller:Function {repo: row.repo, qualified_name: row.caller_qn})
            OPTIONAL MATCH (callee_fn:Function {repo: row.repo, qualified_name: row.callee_qn})
            OPTIONAL MATCH (callee_sym:Symbol {repo: row.repo, qualified_name: row.callee_qn})
            FOREACH (c IN CASE WHEN callee_fn  IS NOT NULL THEN [callee_fn]
                               WHEN callee_sym IS NOT NULL THEN [callee_sym]
                               ELSE [] END |
              MERGE (caller)-[r:CALLS {line: row.line}]->(c)
            )
            """,
            [asdict(e) for e in batch.calls],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (f:File {repo: row.repo, path: row.file_path})
            OPTIONAL MATCH (target_mod:Module {repo: row.repo, qualified_name: row.target_qn})
            FOREACH (_ IN CASE WHEN target_mod IS NOT NULL THEN [1] ELSE [] END |
              MERGE (f)-[:IMPORTS]->(target_mod)
            )
            FOREACH (_ IN CASE WHEN target_mod IS NULL THEN [1] ELSE [] END |
              MERGE (s:Symbol {repo: row.repo, qualified_name: row.target_qn})
              ON CREATE SET s.name = split(row.target_qn, '.')[-1]
              MERGE (f)-[:IMPORTS]->(s)
            )
            """,
            [asdict(e) for e in batch.imports],
        )


def prune_stale(driver: Driver, repo_name: str, current_commit_sha: str) -> int:
    """Delete nodes from earlier scans of this repo that aren't in the
    current scan. Identifies by repo + commit_sha mismatch on the Repo
    node — files/modules don't carry commit_sha themselves (saves space),
    so we just rely on detach-delete cascading from the Repo node.

    Returns the number of nodes deleted. v1 is a no-op when the repo
    matches the current commit; full multi-commit retention comes in 10b.
    """
    if not current_commit_sha:
        # Ad-hoc local scan — no commit-based pruning.
        return 0
    with driver.session() as session:
        # If the repo's stored commit_sha matches, the scan is idempotent and
        # there's nothing to prune. The MERGEs above already updated everything
        # in place. Multi-commit history is a Sprint 10b feature.
        result = session.run(
            """
            MATCH (r:Repo {name: $name})
            RETURN r.commit_sha AS sha
            """,
            name=repo_name,
        ).single()
        if result is None or result["sha"] == current_commit_sha:
            return 0
        # Different commit: nuke all nodes for this repo and let the writer
        # re-create. Brutal but correct; preserves no orphans.
        result = session.run(
            """
            MATCH (n {repo: $name})
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            name=repo_name,
        ).single()
        return int(result["deleted"]) if result else 0
