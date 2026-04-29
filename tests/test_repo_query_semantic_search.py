"""Sprint 14b — `semantic_search` query-layer tests.

Same fake-driver pattern as `test_repo_query_processes.py` /
`test_repo_query_modules.py`: no live Neo4j, no live OpenAI. We
monkeypatch `app.repo_query._embed_query_sync` so tests don't need
an OPENAI_API_KEY, and a synthetic Driver/Session returns canned rows
keyed by which Cypher leg is being executed (BM25 vs vector).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from app import repo_query
from app.repo_query import SemanticHit, semantic_search


# ─── Synthetic driver helpers ───────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    """Returns BM25 rows when the cypher mentions `db.index.fulltext`,
    vector rows when it mentions `db.index.vector`. Either source can be
    set to raise (mimicking missing-index errors from Neo4j). The fake
    also re-applies the test-path predicate in Python so we can verify
    the exclusion logic without booting Cypher.
    """

    def __init__(
        self,
        bm25_rows: Optional[list[dict[str, Any]]] = None,
        vector_rows: Optional[list[dict[str, Any]]] = None,
        bm25_raises: Optional[Exception] = None,
        vector_raises: Optional[Exception] = None,
    ):
        self._bm25_rows = bm25_rows or []
        self._vector_rows = vector_rows or []
        self._bm25_raises = bm25_raises
        self._vector_raises = vector_raises
        self.queries_seen: list[str] = []
        self.params_seen: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        cypher = args[0] if args else ""
        # Mirror neo4j.Session.run: accept parameters either as a dict
        # in args[1] or as kwargs. The real driver merges both.
        merged: dict[str, Any] = {}
        if len(args) >= 2 and isinstance(args[1], dict):
            merged.update(args[1])
        merged.update(params)
        self.queries_seen.append(cypher)
        self.params_seen.append(merged)

        if "db.index.fulltext.queryNodes" in cypher:
            if self._bm25_raises is not None:
                raise self._bm25_raises
            rows = list(self._bm25_rows)
        elif "db.index.vector.queryNodes" in cypher:
            if self._vector_raises is not None:
                raise self._vector_raises
            rows = list(self._vector_rows)
        else:
            rows = []

        # Mirror Cypher's test-path filter so include_tests=False excludes
        # rows whose file_path is test-rooted. The production query has
        # the predicate inline; we mimic it here so we can verify the
        # function under test passes through clean output.
        if "STARTS WITH 'tests/'" in cypher:
            def _is_test(fp: str) -> bool:
                return (
                    fp.startswith("tests/")
                    or fp.startswith("test/")
                    or "/tests/" in fp
                    or "/test_" in fp
                    or fp.startswith("test_")
                )
            rows = [r for r in rows if not _is_test(r.get("file_path", ""))]

        return _FakeResult(rows)


class _FakeDriver:
    def __init__(self, session: _FakeSession):
        self._session = session

    def session(self):
        return self._session


def _hit_row(qn: str, kind: str, score: float, file_path: str = "app/x.py",
             docstring: str = "", name: Optional[str] = None,
             line_start: int = 1) -> dict:
    return {
        "qualified_name": qn,
        "name": name or qn.rsplit(".", 1)[-1],
        "file_path": file_path,
        "line_start": line_start,
        "docstring": docstring,
        "score": score,
        "kind": kind,
    }


# ─── tests ──────────────────────────────────────────────────────────────────


def test_semantic_search_empty_query_returns_empty(monkeypatch):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)
    driver = _FakeDriver(_FakeSession(bm25_rows=[], vector_rows=[]))

    assert semantic_search(driver, "r", "") == []
    assert semantic_search(driver, "r", "   ") == []
    # No queries hit Neo4j on empty input.
    assert driver._session.queries_seen == []


def test_semantic_search_rrf_orders_by_fused_score(monkeypatch):
    """Synthetic ranks: BM25 [A,B,C] vector [D,A,B].
    With rrf_k=60:
      A: 1/61 + 1/62  ≈ 0.03252
      B: 1/62 + 1/63  ≈ 0.03200
      C: 1/63         ≈ 0.01587
      D: 1/61         ≈ 0.01639
    So order is A > B > D > C.
    """
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    bm25 = [
        _hit_row("a.A", "function", score=10.0),
        _hit_row("b.B", "function", score=9.0),
        _hit_row("c.C", "function", score=8.0),
    ]
    vector = [
        _hit_row("d.D", "function", score=0.99),
        _hit_row("a.A", "function", score=0.95),
        _hit_row("b.B", "function", score=0.90),
    ]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=vector))

    out = semantic_search(driver, "r", "anything", k=10)
    assert [h.qualified_name for h in out] == ["a.A", "b.B", "d.D", "c.C"]
    # RRF score floats — assert ordering monotonically decreasing.
    scores = [h.rrf_score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_dedupes_by_qualified_name(monkeypatch):
    """Same node in both legs ⇒ one entry with summed RRF contributions."""
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    bm25 = [_hit_row("dup.X", "function", score=10.0)]   # rank 1 → 1/61
    vector = [_hit_row("dup.X", "function", score=0.99)]  # rank 1 → 1/61
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=vector))

    out = semantic_search(driver, "r", "x", k=10)
    assert len(out) == 1
    expected = (1 / 61) + (1 / 61)
    assert out[0].rrf_score == pytest.approx(expected)


def test_semantic_search_falls_back_to_bm25_when_embedding_fails(
    monkeypatch, caplog,
):
    """Embedding raises ⇒ vector leg skipped; BM25 results still return."""
    def boom(_q):
        return None  # _embed_query_sync swallows exceptions and returns None
    monkeypatch.setattr(repo_query, "_embed_query_sync", boom)

    bm25 = [_hit_row("a.A", "function", score=10.0)]
    vector = [_hit_row("zzz.NotReturned", "function", score=0.99)]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=vector))

    out = semantic_search(driver, "r", "auth")
    assert [h.qualified_name for h in out] == ["a.A"]
    # Verify only the BM25 leg ran.
    cyphers = " ".join(driver._session.queries_seen)
    assert "db.index.fulltext.queryNodes" in cyphers
    assert "db.index.vector.queryNodes" not in cyphers


def test_semantic_search_falls_back_to_vector_when_fulltext_missing(
    monkeypatch, caplog,
):
    """Fulltext index missing → BM25 leg raises → result has vector hits."""
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    vector = [_hit_row("v.V", "function", score=0.95)]
    driver = _FakeDriver(_FakeSession(
        bm25_raises=RuntimeError("There is no procedure with the name ..."),
        vector_rows=vector,
    ))

    with caplog.at_level(logging.WARNING, logger="app.repo_query"):
        out = semantic_search(driver, "r", "anything")
    assert [h.qualified_name for h in out] == ["v.V"]
    assert any("BM25 leg failed" in rec.message for rec in caplog.records)


def test_semantic_search_returns_empty_when_both_indexes_missing(
    monkeypatch, caplog,
):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)
    driver = _FakeDriver(_FakeSession(
        bm25_raises=RuntimeError("no fulltext index"),
        vector_raises=RuntimeError("no vector index"),
    ))

    with caplog.at_level(logging.WARNING, logger="app.repo_query"):
        out = semantic_search(driver, "r", "anything")
    assert out == []
    # We don't crash; we log warnings. The "no results" warning fires too.
    assert any("BM25 leg failed" in rec.message for rec in caplog.records)
    assert any("vector leg failed" in rec.message for rec in caplog.records)


def test_semantic_search_excludes_test_paths_by_default(monkeypatch):
    """Same predicate as find_processes / find_modules — test-rooted paths
    are stripped before fusion. The fake session re-applies the predicate
    when the cypher contains the literal sentinel ('STARTS WITH \\'tests/\\'')."""
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    bm25 = [
        _hit_row("prod.A", "function", score=10.0, file_path="app/a.py"),
        _hit_row("test.B", "function", score=9.0, file_path="tests/test_b.py"),
        _hit_row("nest.C", "function", score=8.0, file_path="src/tests/test_c.py"),
        _hit_row("tprefix.D", "function", score=7.0, file_path="test_d.py"),
    ]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=[]))

    out = semantic_search(driver, "r", "anything")
    assert [h.qualified_name for h in out] == ["prod.A"]


def test_semantic_search_emits_test_path_predicate_in_cypher():
    """The Cypher MUST mention the test-path predicate when
    include_tests=False (default). Same load-bearing filter as 13c."""
    driver = _FakeDriver(_FakeSession(bm25_rows=[], vector_rows=[]))
    semantic_search(driver, "r", "anything")
    cyphers = " ".join(driver._session.queries_seen)
    assert "STARTS WITH 'tests/'" in cyphers
    assert "STARTS WITH 'test/'" in cyphers
    assert "CONTAINS '/tests/'" in cyphers
    assert "CONTAINS '/test_'" in cyphers
    assert "STARTS WITH 'test_'" in cyphers


def test_semantic_search_limit_respected(monkeypatch):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    bm25 = [_hit_row(f"a.{i}", "function", score=100 - i) for i in range(20)]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=[]))

    out = semantic_search(driver, "r", "anything", k=5)
    assert len(out) == 5


def test_semantic_search_kind_field_correct(monkeypatch):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    bm25 = [
        _hit_row("a.MyClass", "class", score=10.0),
        _hit_row("a.my_fn", "function", score=9.0),
    ]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=[]))

    out = semantic_search(driver, "r", "anything")
    by_qn = {h.qualified_name: h for h in out}
    assert by_qn["a.MyClass"].kind == "class"
    assert by_qn["a.my_fn"].kind == "function"


def test_semantic_search_truncates_long_docstrings(monkeypatch):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)

    long_doc = "x" * 2000
    bm25 = [_hit_row("a.A", "function", score=10.0, docstring=long_doc)]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=[]))

    out = semantic_search(driver, "r", "anything")
    assert len(out) == 1
    # Truncation cap is 200 chars (private constant in repo_query).
    assert len(out[0].docstring) <= 250
    assert len(out[0].docstring) < len(long_doc)


def test_semantic_search_returns_semantichit_dataclasses(monkeypatch):
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)
    bm25 = [_hit_row("a.A", "function", score=10.0)]
    driver = _FakeDriver(_FakeSession(bm25_rows=bm25, vector_rows=[]))

    out = semantic_search(driver, "r", "anything")
    assert isinstance(out, list)
    assert all(isinstance(h, SemanticHit) for h in out)


def test_semantic_search_passes_query_string_to_bm25_leg(monkeypatch):
    """The BM25 leg's $query param must be the user's query string."""
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)
    driver = _FakeDriver(_FakeSession(bm25_rows=[], vector_rows=[]))
    semantic_search(driver, "r", "auth handlers")
    bm25_params = next(
        p for c, p in zip(driver._session.queries_seen, driver._session.params_seen)
        if "db.index.fulltext.queryNodes" in c
    )
    assert bm25_params["query"] == "auth handlers"
    assert bm25_params["repo"] == "r"


def test_semantic_search_candidate_limit_is_2k(monkeypatch):
    """Each leg pulls 2*k candidates so fusion has headroom."""
    monkeypatch.setattr(repo_query, "_embed_query_sync", lambda q: [0.0] * 1536)
    driver = _FakeDriver(_FakeSession(bm25_rows=[], vector_rows=[]))
    semantic_search(driver, "r", "anything", k=7)
    bm25_params = next(
        p for c, p in zip(driver._session.queries_seen, driver._session.params_seen)
        if "db.index.fulltext.queryNodes" in c
    )
    assert bm25_params["candidate_limit"] == 14
