"""Sprint 13c — `find_modules` query-layer tests.

Same shape as test_repo_query_processes: synthetic driver/session, no
live Neo4j. The fake replays the Cypher's filter / sort / limit logic
in Python so we can assert on the function's projected output without
exercising a real graph.
"""
from __future__ import annotations

import random
from typing import Any

import pytest

from app.repo_query import ModuleSummary, find_modules


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.last_query: str | None = None
        self.last_params: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        cypher = args[0] if args else params.pop("_cypher", "")
        self.last_query = cypher
        self.last_params = params
        query = cypher  # alias for keyword reuse below

        rows = list(self._rows)

        # Singletons (size <= 1) are filtered by the Cypher.
        rows = [r for r in rows if len(r["member_qns"]) > 1]

        # When include_tests=False the Cypher injects `any(member.file ...)`.
        # The fake honours that by reading the query string.
        if "any(member IN members WHERE NOT" in query:
            def _is_test(fp: str) -> bool:
                return (
                    fp.startswith("tests/")
                    or fp.startswith("test/")
                    or "/tests/" in fp
                    or "/test_" in fp
                    or fp.startswith("test_")
                )
            rows = [
                r for r in rows
                if any(not _is_test(f) for f in r["_member_files"])
            ]

        rows = sorted(rows, key=lambda r: (-len(r["member_qns"]), r["label"]))

        limit = int(params.get("limit", 20))
        rows = rows[:limit]

        cleaned = [
            {
                "label": r["label"],
                "cohesion": r.get("cohesion", 0.0),
                "size": r.get("size", len(r["member_qns"])),
                "member_qns": r["member_qns"],
            }
            for r in rows
        ]
        return _FakeResult(cleaned)


class _FakeDriver:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.session_obj = _FakeSession(rows)

    def session(self):
        return self.session_obj


def _community(label: str, member_qns: list[str], member_files: list[str] | None = None,
               cohesion: float = 0.5) -> dict:
    if member_files is None:
        member_files = ["app/x.py"] * len(member_qns)
    return {
        "label": label,
        "cohesion": cohesion,
        "size": len(member_qns),
        "member_qns": member_qns,
        "_member_files": member_files,
    }


# ─── tests ──────────────────────────────────────────────────────────────────


def test_find_modules_orders_by_size_desc():
    driver = _FakeDriver([
        _community("alpha", [f"a.{i}" for i in range(8)]),
        _community("beta", [f"b.{i}" for i in range(3)]),
        _community("gamma", [f"g.{i}" for i in range(12)]),
    ])
    out = find_modules(driver, repo="r")
    assert [m.label for m in out] == ["gamma", "alpha", "beta"]
    assert [m.size for m in out] == [12, 8, 3]


def test_find_modules_skips_singletons():
    driver = _FakeDriver([
        _community("singleton", ["only.one"]),
        _community("real", ["a.b", "a.c"]),
    ])
    out = find_modules(driver, repo="r")
    assert [m.label for m in out] == ["real"]


def test_find_modules_excludes_test_only_communities():
    driver = _FakeDriver([
        _community("prod_cluster", ["app.a", "app.b"], ["app/a.py", "app/b.py"]),
        _community("all_test_cluster", ["t.a", "t.b", "t.c"],
                   ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]),
        _community("nested_test_cluster", ["x.a", "x.b"],
                   ["src/tests/test_a.py", "src/tests/test_b.py"]),
        _community("prefix_test_cluster", ["p.a", "p.b"],
                   ["test_a.py", "test_b.py"]),
    ])
    out = find_modules(driver, repo="r")
    assert [m.label for m in out] == ["prod_cluster"]


def test_find_modules_partial_test_kept():
    """A cluster with one non-test member survives the include_tests=False filter."""
    driver = _FakeDriver([
        _community("mixed", ["app.a", "tests.t"], ["app/a.py", "tests/test_x.py"]),
    ])
    out = find_modules(driver, repo="r")
    assert len(out) == 1
    assert out[0].label == "mixed"


def test_find_modules_sample_members_deterministic():
    """Driver returns members in a randomised order; the function returns
    the first 5 sorted lexicographically every time."""
    members = [f"app.mod_{i}" for i in range(20)]
    shuffled = list(members)
    random.Random(42).shuffle(shuffled)
    driver = _FakeDriver([
        _community("big", shuffled),
    ])
    out = find_modules(driver, repo="r")
    assert len(out[0].sample_member_qns) == 5
    expected = tuple(sorted(members)[:5])
    assert out[0].sample_member_qns == expected


def test_find_modules_limit_respected():
    rows = [
        _community(f"c{i:03d}", [f"x.{i}.{j}" for j in range(i + 2)])
        for i in range(30)
    ]
    driver = _FakeDriver(rows)
    out = find_modules(driver, repo="r", limit=10)
    assert len(out) == 10


def test_find_modules_returns_modulesummary_dataclasses():
    driver = _FakeDriver([
        _community("c", ["a.x", "a.y"], cohesion=0.75),
    ])
    out = find_modules(driver, repo="r")
    assert isinstance(out, list)
    assert all(isinstance(m, ModuleSummary) for m in out)
    assert out[0].cohesion == pytest.approx(0.75)


def test_find_modules_emits_test_path_predicate_in_cypher():
    driver = _FakeDriver([])
    find_modules(driver, repo="r")
    q = driver.session_obj.last_query or ""
    assert "STARTS WITH 'tests/'" in q
    assert "STARTS WITH 'test/'" in q
    assert "CONTAINS '/tests/'" in q
    assert "CONTAINS '/test_'" in q
    assert "STARTS WITH 'test_'" in q
    # And the singleton skip.
    assert "size(members) > 1" in q


def test_find_modules_omits_test_predicate_when_include_tests_true():
    driver = _FakeDriver([])
    find_modules(driver, repo="r", include_tests=True)
    q = driver.session_obj.last_query or ""
    assert "STARTS WITH 'tests/'" not in q
    # Singleton skip is unconditional.
    assert "size(members) > 1" in q
