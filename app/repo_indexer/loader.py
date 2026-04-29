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

from .actions import ClassNode, FunctionNode, IndexBatch

logger = logging.getLogger(__name__)


# Sprint 14a — embedding dimensionality for the Neo4j vector index.
#
# Source of truth: `app.embeddings.EMBEDDING_DIMS` (currently 1536, OpenAI
# `text-embedding-3-small`). We import lazily inside `ensure_constraints`
# to avoid forcing the openai client into the module's import path — tests
# that touch only the loader's chunked-write helper shouldn't need an
# OPENAI_API_KEY. The import is gated so a missing-openai-package install
# still lets the constraint setup proceed (the vector index just won't be
# created); the embed phase itself is opt-in and will raise its own clear
# error if invoked without the dep.
#
# If `app.embeddings.EMBEDDING_DIMS` ever changes (different model), drop
# the existing vector indexes (`DROP INDEX function_embedding`,
# `DROP INDEX class_embedding`) before re-running ensure_constraints —
# Neo4j can't resize a vector index in place.
EMBEDDING_SIMILARITY_FUNCTION = "cosine"


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
    """Idempotent uniqueness constraints + indexes. Cheap to re-run on every scan.

    Sprint 14a adds two vector indexes (Function.embedding, Class.embedding)
    sized to `app.embeddings.EMBEDDING_DIMS`. Vector indexes require Neo4j
    server >=5.13. The deployed `neo4j:5-community` image is well past that
    (5.27+); local dev DBs older than 5.13 will fail this CREATE silently
    via `IF NOT EXISTS` only if the syntax itself is rejected. Verify before
    enabling `--with-embeddings` against an older instance.
    """
    stmts = [
        # Composite uniqueness — same qualified_name across different repos
        # is allowed and expected (forks, multiple tenants, etc.).
        "CREATE CONSTRAINT repo_name IF NOT EXISTS FOR (r:Repo) REQUIRE r.name IS UNIQUE",
        "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE (f.repo, f.path) IS UNIQUE",
        "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE (m.repo, m.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE (c.repo, c.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (fn:Function) REQUIRE (fn.repo, fn.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (s:Symbol) REQUIRE (s.repo, s.qualified_name) IS UNIQUE",
        # Sprint 13a — derived community structure. tenant_id stays on the
        # node but NOT in the uniqueness key, same convention as Repo/File
        # (the multi-tenant gap is tracked in
        # `production-multitenant-architecture.md` and is a separate fix).
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE (c.repo, c.label) IS UNIQUE",
        # Sprint 13b — Process nodes (execution flows). Same tenant_id
        # convention as Community: on the node, not in the uniqueness key.
        "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE (p.repo, p.name) IS UNIQUE",
        # Sprint 15a — HTTP route nodes. Composite key is (repo, path, method)
        # so re-indexing the same route definition (across runs, across
        # different handler_qn resolutions) doesn't fan out into multiple
        # Route nodes. handler_qn lives on the HANDLED_BY edge, not the key.
        "CREATE CONSTRAINT route_id IF NOT EXISTS FOR (r:Route) REQUIRE (r.repo, r.path, r.method) IS UNIQUE",
    ]
    with driver.session() as session:
        for stmt in stmts:
            session.run(stmt)

        # Sprint 14b — full-text (BM25) indexes on Function/Class. These
        # power the keyword leg of `repo_query.semantic_search`. They are
        # NOT gated on the `app.embeddings` import — full-text indexing
        # has no embeddings dependency, and the BM25 leg is useful even
        # when the repo was never indexed `--with-embeddings`.
        for label, index_name in (
            ("Function", "function_text"),
            ("Class", "class_text"),
        ):
            session.run(
                f"""
                CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON EACH [n.name, n.docstring]
                """
            )

        # Sprint 14a — vector indexes. Created here (not as constraints —
        # they're indexes, not uniqueness rules) so the embed phase can
        # write into them on the very first opt-in scan. Sourced dim from
        # `app.embeddings.EMBEDDING_DIMS` to keep one source of truth; if
        # the import fails (no openai installed), we skip the vector index
        # creation rather than crashing the whole scan — without the embed
        # phase the indexes aren't needed anyway.
        try:
            from app.embeddings import EMBEDDING_DIMS as _dim
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "skipping vector index creation — app.embeddings not importable (%s); "
                "this is fine when not using --with-embeddings", exc,
            )
            return

        # Cypher 5 vector index syntax. Backticked option keys per Neo4j
        # 5.13+ docs. `IF NOT EXISTS` makes this idempotent across runs.
        for label in ("Function", "Class"):
            index_name = f"{label.lower()}_embedding"
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: $sim
                }}}}
                """,
                dim=_dim, sim=EMBEDDING_SIMILARITY_FUNCTION,
            )


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

        # Sprint 13a — Community nodes (one per detected cluster) and
        # MEMBER_OF edges from each Function/Class to its community.
        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (r:Repo {name: row.repo})
            MERGE (c:Community {repo: row.repo, label: row.label})
            SET c.tenant_id = row.tenant_id,
                c.cohesion = row.cohesion,
                c.size = row.size
            """,
            [asdict(c) for c in batch.communities],
        )

        # MemberOfEdge — member can be Function or Class. Use FOREACH/CASE
        # to MERGE the edge against whichever type matched (mirrors how
        # the inherits / calls writes pick between Function/Symbol).
        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (c:Community {repo: row.repo, label: row.community_label})
            OPTIONAL MATCH (fn:Function {repo: row.repo, qualified_name: row.member_qn})
            OPTIONAL MATCH (cls:Class {repo: row.repo, qualified_name: row.member_qn})
            FOREACH (m IN CASE WHEN fn IS NOT NULL THEN [fn]
                               WHEN cls IS NOT NULL THEN [cls]
                               ELSE [] END |
              MERGE (m)-[:MEMBER_OF]->(c)
            )
            """,
            [asdict(e) for e in batch.member_of],
        )

        # Sprint 13b — Process nodes (execution flows: chains of CALLS that
        # cross community boundaries) + STEP_IN_PROCESS edges with `step`
        # index. Process points to Function (per the schema in
        # repo-indexer-future-state.md §4.1).
        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (r:Repo {name: row.repo})
            MERGE (p:Process {repo: row.repo, name: row.name})
            SET p.tenant_id = row.tenant_id,
                p.summary = row.summary
            """,
            [asdict(p) for p in batch.processes],
        )

        _chunked_write(session,
            """
            UNWIND $rows AS row
            MATCH (p:Process {repo: row.repo, name: row.process_name})
            MATCH (fn:Function {repo: row.repo, qualified_name: row.member_qn})
            MERGE (p)-[r:STEP_IN_PROCESS]->(fn)
            SET r.step = row.step
            """,
            [asdict(e) for e in batch.step_in_process],
        )

        # Sprint 15a — Route nodes + HANDLED_BY edges + DEFINES edges
        # (File→Route for discoverability). MERGE on (repo, path, method)
        # so re-indexing collapses to one node per route definition.
        if batch.routes:
            _chunked_write(session,
                """
                UNWIND $rows AS row
                MERGE (r:Route {repo: row.repo, path: row.path, method: row.method})
                SET r.tenant_id = row.tenant_id,
                    r.framework = row.framework,
                    r.raw_path = row.raw_path,
                    r.file_path = row.file_path,
                    r.line_start = row.line_start
                """,
                [asdict(r) for r in batch.routes],
            )
            # File→Route edge (matches the `(File)-[:DEFINES]->(Module|Class|Function)`
            # convention the rest of the schema uses).
            _chunked_write(session,
                """
                UNWIND $rows AS row
                MATCH (f:File {repo: row.repo, path: row.file_path})
                MATCH (r:Route {repo: row.repo, path: row.path, method: row.method})
                MERGE (f)-[:DEFINES]->(r)
                """,
                [asdict(r) for r in batch.routes],
            )
        if batch.route_edges:
            # HANDLED_BY edges. Skip rows where handler_qn is empty
            # (inline lambdas / unresolvable handlers — RouteNode still
            # carries file_path + line_start for jump-to-source).
            _chunked_write(session,
                """
                UNWIND $rows AS row
                MATCH (r:Route {repo: row.repo, path: row.path, method: row.method})
                MATCH (fn:Function {repo: row.repo, qualified_name: row.handler_qn})
                MERGE (r)-[:HANDLED_BY]->(fn)
                """,
                [asdict(e) for e in batch.route_edges if e.handler_qn],
            )

        # Sprint 14a — embedding writes. Delegated to _write_embeddings so
        # the --embed-only path can reuse the same Cypher.
        _write_embeddings(session, batch)


# Sprint 14a — embedding writes. SET the `embedding` property on the
# matching Function or Class node. Same Function/Class fan-out pattern as
# MEMBER_OF (loader picks the right label via OPTIONAL MATCH + FOREACH/CASE).
# `embedding` is a tuple of float in Python; asdict serializes it to a list,
# which Neo4j stores as a LIST<FLOAT> property the vector index can read.
_EMBEDDING_WRITE_CYPHER = """
UNWIND $rows AS row
OPTIONAL MATCH (fn:Function {repo: row.repo, qualified_name: row.qualified_name})
OPTIONAL MATCH (cls:Class {repo: row.repo, qualified_name: row.qualified_name})
FOREACH (n IN CASE WHEN row.target_kind = 'function' AND fn IS NOT NULL THEN [fn]
                   WHEN row.target_kind = 'class'    AND cls IS NOT NULL THEN [cls]
                   ELSE [] END |
  SET n.embedding = row.embedding
)
"""


def _write_embeddings(session, batch: IndexBatch) -> None:
    """Persist `batch.embeddings` as `embedding` properties on existing
    Function / Class nodes. Caller owns the session."""
    _chunked_write(
        session,
        _EMBEDDING_WRITE_CYPHER,
        [{**asdict(e), "embedding": list(e.embedding)} for e in batch.embeddings],
    )


def write_embeddings_only(driver: Driver, batch: IndexBatch) -> None:
    """Public entry point for the --embed-only path: writes the embedding
    payload without touching any other graph state. Assumes the Function /
    Class nodes already exist (otherwise the OPTIONAL MATCH pair returns
    null and the SET is a no-op for that row)."""
    with driver.session() as session:
        _write_embeddings(session, batch)


def fetch_file_shas(driver: Driver, repo_name: str, tenant_id: str = "default") -> dict[str, str]:
    """Return {rel_path: sha} for every File previously written under this repo.
    Empty dict if the repo isn't in Neo4j yet (first scan).

    `tenant_id` is on the Repo node today, not File; it's accepted here for
    forward-compat with the multi-tenant fix tracked in
    `production-multitenant-architecture.md`, but we don't filter on it yet.
    """
    out: dict[str, str] = {}
    with driver.session() as session:
        result = session.run(
            "MATCH (f:File {repo: $repo}) RETURN f.path AS path, f.sha AS sha",
            repo=repo_name,
        )
        for rec in result:
            path = rec["path"]
            sha = rec["sha"]
            if path is None or sha is None:
                continue
            out[path] = sha
    return out


def fetch_unembedded_symbols(
    driver: Driver, repo_name: str, tenant_id: str = "default",
) -> tuple[list[FunctionNode], list[ClassNode]]:
    """Return (functions, classes) for nodes in `repo_name` that lack the
    `embedding` property. Used by the `--embed-only` CLI path to backfill
    embeddings on an already-scanned repo without re-running parse / resolve
    / community / process.

    `param_types` isn't persisted on Function nodes (the resolver consumes
    it during scan, then it's discarded), so the reconstructed FunctionNode
    has `param_types=()`. Embedding text generation doesn't use it, so the
    output is byte-identical to a fresh scan's embedding text.

    Methods' `parent_class_qn` is recovered via the `(Class)-[:DEFINES]->(Function)`
    edge — it's not a Function-node property today, but the relationship
    carries the same information.
    """
    fns: list[FunctionNode] = []
    classes: list[ClassNode] = []
    with driver.session() as session:
        fn_rows = session.run(
            """
            MATCH (f:Function {repo: $repo})
            WHERE f.embedding IS NULL
            OPTIONAL MATCH (parent:Class {repo: $repo})-[:DEFINES]->(f)
            RETURN f.qualified_name AS qualified_name,
                   f.name           AS name,
                   coalesce(f.file_path, '')  AS file_path,
                   coalesce(f.line_start, 0)  AS line_start,
                   coalesce(f.line_end, 0)    AS line_end,
                   coalesce(f.is_async, false)   AS is_async,
                   coalesce(f.is_method, false)  AS is_method,
                   coalesce(parent.qualified_name, '') AS parent_class_qn,
                   coalesce(f.params, [])     AS params,
                   coalesce(f.docstring, '')  AS docstring
            """,
            repo=repo_name,
        ).data()
        for row in fn_rows:
            fns.append(FunctionNode(
                repo=repo_name,
                qualified_name=row["qualified_name"],
                name=row["name"],
                file_path=row["file_path"],
                line_start=int(row["line_start"]),
                line_end=int(row["line_end"]),
                is_async=bool(row["is_async"]),
                is_method=bool(row["is_method"]),
                parent_class_qn=row["parent_class_qn"],
                params=tuple(row["params"]),
                param_types=(),  # not persisted; not needed for embedding text
                docstring=row["docstring"],
            ))

        cls_rows = session.run(
            """
            MATCH (c:Class {repo: $repo})
            WHERE c.embedding IS NULL
            RETURN c.qualified_name AS qualified_name,
                   c.name           AS name,
                   coalesce(c.file_path, '')  AS file_path,
                   coalesce(c.line_start, 0)  AS line_start,
                   coalesce(c.line_end, 0)    AS line_end,
                   coalesce(c.docstring, '')  AS docstring
            """,
            repo=repo_name,
        ).data()
        for row in cls_rows:
            classes.append(ClassNode(
                repo=repo_name,
                qualified_name=row["qualified_name"],
                name=row["name"],
                file_path=row["file_path"],
                line_start=int(row["line_start"]),
                line_end=int(row["line_end"]),
                docstring=row["docstring"],
            ))
    _ = tenant_id  # forward-compat for the multi-tenant fix
    return fns, classes


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
