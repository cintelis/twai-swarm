"""Tests for the --embed-only backfill path (fetch_unembedded_symbols
+ write_embeddings_only + cmd_embed_only).

The first two are loader functions (one read, one write); the third is
the CLI plumbing that ties them to EmbedPhase. All tested with the
same fake-driver / fake-session pattern used elsewhere in the suite.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.repo_indexer.actions import (
    ClassNode,
    EmbeddingUpdate,
    FunctionNode,
    IndexBatch,
    RepoNode,
)
from app.repo_indexer.loader import fetch_unembedded_symbols, write_embeddings_only

REPO = "twai-swarm"


# ─── Fakes ──────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def data(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    """Returns a queue of pre-built result sets per `session.run` call.

    fetch_unembedded_symbols runs two queries (functions, then classes);
    write_embeddings_only runs one (embedding upsert). Tests load the
    queue accordingly."""

    def __init__(self, results: list[list[dict[str, Any]]] | None = None):
        self._queue = list(results or [])
        self.queries_seen: list[str] = []
        self.params_seen: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        cypher = args[0] if args else ""
        merged: dict[str, Any] = {}
        if len(args) >= 2 and isinstance(args[1], dict):
            merged.update(args[1])
        merged.update(params)
        self.queries_seen.append(cypher)
        self.params_seen.append(merged)
        rows = self._queue.pop(0) if self._queue else []
        return _FakeResult(rows)


class _FakeDriver:
    def __init__(self, session: _FakeSession):
        self._session = session

    def session(self):
        return self._session


# ─── fetch_unembedded_symbols ───────────────────────────────────────────────

def test_fetch_unembedded_constructs_function_node():
    """One fn row → FunctionNode populated with all the persisted fields."""
    session = _FakeSession([
        [
            {
                "qualified_name": "app.foo.bar",
                "name": "bar",
                "file_path": "app/foo.py",
                "line_start": 10,
                "line_end": 20,
                "is_async": True,
                "is_method": False,
                "parent_class_qn": "",
                "params": ["x", "y"],
                "docstring": "Adds two things.",
            },
        ],
        [],  # empty classes result
    ])
    fns, classes = fetch_unembedded_symbols(_FakeDriver(session), REPO)

    assert classes == []
    assert len(fns) == 1
    fn = fns[0]
    assert fn.qualified_name == "app.foo.bar"
    assert fn.name == "bar"
    assert fn.file_path == "app/foo.py"
    assert fn.line_start == 10
    assert fn.line_end == 20
    assert fn.is_async is True
    assert fn.is_method is False
    assert fn.parent_class_qn == ""
    assert fn.params == ("x", "y")
    assert fn.param_types == ()  # not persisted
    assert fn.docstring == "Adds two things."


def test_fetch_unembedded_method_recovers_parent_class_qn():
    """For methods, the OPTIONAL MATCH on (Class)-[:DEFINES]->(Function)
    populates parent_class_qn — required so embedding_text_for_function
    emits `in app.foo.MyClass` instead of falling back to file_path."""
    session = _FakeSession([
        [
            {
                "qualified_name": "app.foo.MyClass.greet",
                "name": "greet",
                "file_path": "app/foo.py",
                "line_start": 12,
                "line_end": 14,
                "is_async": False,
                "is_method": True,
                "parent_class_qn": "app.foo.MyClass",
                "params": ["self", "name"],
                "docstring": "",
            },
        ],
        [],
    ])
    fns, _ = fetch_unembedded_symbols(_FakeDriver(session), REPO)
    assert fns[0].is_method is True
    assert fns[0].parent_class_qn == "app.foo.MyClass"


def test_fetch_unembedded_constructs_class_node():
    session = _FakeSession([
        [],  # no functions
        [
            {
                "qualified_name": "app.foo.MyClass",
                "name": "MyClass",
                "file_path": "app/foo.py",
                "line_start": 5,
                "line_end": 30,
                "docstring": "Holds things.",
            },
        ],
    ])
    fns, classes = fetch_unembedded_symbols(_FakeDriver(session), REPO)
    assert fns == []
    assert len(classes) == 1
    cls = classes[0]
    assert cls.qualified_name == "app.foo.MyClass"
    assert cls.docstring == "Holds things."


def test_fetch_unembedded_returns_empty_when_repo_fully_covered():
    """All Function/Class nodes already have embeddings → both queries
    return zero rows. Caller short-circuits without invoking the embedder."""
    session = _FakeSession([[], []])
    fns, classes = fetch_unembedded_symbols(_FakeDriver(session), REPO)
    assert fns == []
    assert classes == []
    # Both queries ran (functions, then classes) and both filtered on $repo.
    assert len(session.queries_seen) == 2
    assert all(p["repo"] == REPO for p in session.params_seen)


# ─── write_embeddings_only ──────────────────────────────────────────────────

def test_write_embeddings_only_no_op_on_empty_batch():
    """Empty embeddings list → no Cypher executed (the chunked-write
    helper returns immediately when rows is empty)."""
    session = _FakeSession()
    repo = RepoNode(name=REPO, url="", commit_sha="")
    batch = IndexBatch(repo=repo)
    write_embeddings_only(_FakeDriver(session), batch)
    assert session.queries_seen == []


def test_write_embeddings_only_runs_embedding_cypher():
    """One embedding row → exactly one chunked write that targets the
    Function/Class fan-out cypher."""
    session = _FakeSession()
    repo = RepoNode(name=REPO, url="", commit_sha="")
    batch = IndexBatch(repo=repo)
    batch.embeddings.append(EmbeddingUpdate(
        repo=REPO,
        tenant_id="default",
        target_kind="function",
        qualified_name="app.foo.bar",
        embedding=tuple([0.1] * 1536),
    ))
    write_embeddings_only(_FakeDriver(session), batch)
    assert len(session.queries_seen) == 1
    cypher = session.queries_seen[0]
    assert "OPTIONAL MATCH (fn:Function" in cypher
    assert "OPTIONAL MATCH (cls:Class" in cypher
    assert "SET n.embedding = row.embedding" in cypher

    # Embedding tuple is materialised as a list (Neo4j stores LIST<FLOAT>).
    rows = session.params_seen[0]["rows"]
    assert len(rows) == 1
    assert isinstance(rows[0]["embedding"], list)
    assert rows[0]["target_kind"] == "function"


# ─── cmd_embed_only smoke ───────────────────────────────────────────────────

def test_cmd_embed_only_short_circuits_when_repo_fully_covered():
    """When fetch_unembedded_symbols returns nothing, cmd_embed_only exits
    0 without calling EmbedPhase or write_embeddings_only — the embed
    call would burn OpenAI quota on an already-embedded repo."""
    from app.repo_indexer.__main__ import cmd_embed_only

    repo = RepoNode(name=REPO, url="", commit_sha="")
    session = _FakeSession([[], []])  # 0 functions, 0 classes
    driver = _FakeDriver(session)

    # `with driver_from_env() as driver` -> fake the context manager.
    fake_cm = type("CM", (), {
        "__enter__": lambda self: driver,
        "__exit__": lambda *a: None,
    })()

    with patch("app.repo_indexer.__main__.driver_from_env", lambda: fake_cm), \
         patch("app.repo_indexer.__main__.ensure_constraints", lambda d: None), \
         patch("app.repo_indexer.__main__.write_embeddings_only") as fake_write, \
         patch("app.repo_indexer.phases.embed.EmbedPhase.run") as fake_embed:
        rc = cmd_embed_only(repo, REPO)

    assert rc == 0
    fake_embed.assert_not_called()
    fake_write.assert_not_called()


def test_cmd_embed_only_invokes_embed_phase_when_symbols_present():
    """When fetch_unembedded_symbols returns rows, EmbedPhase runs and
    write_embeddings_only persists. We don't actually run the embedder
    (mocked) — just verify the wiring."""
    from app.repo_indexer.__main__ import cmd_embed_only

    repo = RepoNode(name=REPO, url="", commit_sha="")
    session = _FakeSession([
        [{
            "qualified_name": "app.foo.bar",
            "name": "bar",
            "file_path": "app/foo.py",
            "line_start": 1,
            "line_end": 5,
            "is_async": False,
            "is_method": False,
            "parent_class_qn": "",
            "params": [],
            "docstring": "",
        }],
        [],  # no classes
    ])
    driver = _FakeDriver(session)

    fake_cm = type("CM", (), {
        "__enter__": lambda self: driver,
        "__exit__": lambda *a: None,
    })()

    with patch("app.repo_indexer.__main__.driver_from_env", lambda: fake_cm), \
         patch("app.repo_indexer.__main__.ensure_constraints", lambda d: None), \
         patch("app.repo_indexer.__main__.write_embeddings_only") as fake_write, \
         patch("app.repo_indexer.phases.embed.EmbedPhase.run") as fake_embed:
        rc = cmd_embed_only(repo, REPO)

    assert rc == 0
    fake_embed.assert_called_once()
    fake_write.assert_called_once()
    # The batch passed to write_embeddings_only carries the function we fetched.
    written_batch = fake_write.call_args[0][1]
    assert len(written_batch.functions) == 1
    assert written_batch.functions[0].qualified_name == "app.foo.bar"
