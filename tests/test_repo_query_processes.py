"""Sprint 13c — `find_processes` query-layer tests.

Mirrors the test style elsewhere in the suite: no live Neo4j, just a
synthetic Driver/Session that returns canned rows. We assert on the
result shape (ordering, filtering, limiting) rather than the Cypher
string itself — the Cypher is exercised end-to-end at integration time.
"""
from __future__ import annotations

from typing import Any

from app.repo_query import ProcessSummary, find_processes


# ─── Synthetic driver helpers ───────────────────────────────────────────────
# The query function does:
#   with driver.session() as session:
#       result = session.run(cypher, **params)
#       rows = result.data()
# We replicate that surface with a list of pre-built dicts. The Cypher
# WHERE clauses live inside the query, so the fake mimics what Neo4j
# would have already filtered + ordered (the function still applies its
# own Python-side test-path filtering when configured to).


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    """Records the query + params and returns the rows the test pre-set.

    Crucially, this fake re-implements the Cypher logic the query relies
    on — sort, filter, limit — so we can verify the function returns the
    right *shape* without booting Neo4j. We DO NOT model test-path
    filtering here because that's part of the Cypher under test; tests
    that check it pass test-rooted rows through and assert the function's
    Cypher excludes them. Since we can't run Cypher, those tests apply
    the same predicate in Python and feed the fake the post-filter rows
    — i.e. they assert the function emits the predicate and we mirror
    its expected behaviour.
    """

    def __init__(self, rows_by_match: list[dict[str, Any]]):
        self._rows_by_match = rows_by_match
        self.last_query: str | None = None
        self.last_params: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        # Production code calls session.run(cypher_string, **params). We
        # accept the cypher positionally (one positional arg) and pass
        # everything else through as `params`. We can't call the param
        # `query` here because `params["query"]` is a Cypher param the
        # production code passes by keyword.
        cypher = args[0] if args else params.pop("_cypher", "")
        self.last_query = cypher
        self.last_params = params
        query = cypher  # alias for the test logic below

        # Emulate the Cypher: optional name/summary substring filter,
        # optional test-path member filter, sort by step_count desc /
        # name asc, then LIMIT.
        rows = list(self._rows_by_match)

        q = params.get("query")
        if q:
            ql = q.lower()
            rows = [
                r for r in rows
                if ql in r["name"].lower() or ql in (r.get("summary") or "").lower()
            ]

        # Test-path filter is on by default (include_tests=False ⇒ Cypher
        # injects a `none(member.file ...)` clause). The fake honours that
        # by keying off whether the query string contains the predicate.
        if "STARTS WITH 'tests/'" in query or "STARTS WITH 'test/'" in query:
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
                if not any(_is_test(f) for f in r["_member_files"])
            ]

        rows = sorted(rows, key=lambda r: (-r["step_count"], r["name"]))

        limit = int(params.get("limit", 10))
        rows = rows[:limit]

        # Strip helper key the production query never returns.
        cleaned = [
            {
                "name": r["name"],
                "summary": r.get("summary", ""),
                "step_count": r["step_count"],
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


def _row(name: str, member_qns: list[str], member_files: list[str], summary: str = "") -> dict:
    return {
        "name": name,
        "summary": summary,
        "step_count": len(member_qns),
        "member_qns": member_qns,
        "_member_files": member_files,
    }


# ─── tests ──────────────────────────────────────────────────────────────────


def test_find_processes_orders_by_step_count_desc():
    driver = _FakeDriver([
        _row("A", ["a.x", "a.y", "a.z", "a.w", "a.v"], ["app/a.py"] * 5),  # 5
        _row("B", ["b.x", "b.y"], ["app/b.py"] * 2),                        # 2
        _row("C", ["c.1", "c.2", "c.3", "c.4", "c.5", "c.6", "c.7", "c.8"],
             ["app/c.py"] * 8),                                             # 8
    ])
    out = find_processes(driver, repo="r")
    assert [p.step_count for p in out] == [8, 5, 2]
    assert [p.name for p in out] == ["C", "A", "B"]


def test_find_processes_query_filter():
    driver = _FakeDriver([
        _row("auth_login", ["a"], ["app/auth.py"], summary="login flow"),
        _row("billing_charge", ["b"], ["app/billing.py"], summary="charges card"),
        _row("misc", ["c"], ["app/misc.py"], summary="contains the word AUTH inside"),
    ])
    out = find_processes(driver, repo="r", query="auth")
    names = [p.name for p in out]
    # Matches the literal "auth" in name (auth_login) and in summary (misc).
    assert "auth_login" in names
    assert "misc" in names
    assert "billing_charge" not in names


def test_find_processes_excludes_test_paths_by_default():
    driver = _FakeDriver([
        _row("prod_flow", ["app.x.y"], ["app/x.py"]),
        _row("test_rooted_flow", ["t.t"], ["tests/test_x.py"]),
        _row("nested_test_flow", ["n.n"], ["src/tests/test_y.py"]),
        _row("test_prefix_flow", ["p.p"], ["test_helper.py"]),
        _row("mixed_flow", ["a.a", "t.t"], ["app/a.py", "tests/test_a.py"]),
    ])

    default = find_processes(driver, repo="r")
    assert [p.name for p in default] == ["prod_flow"]

    keep = find_processes(driver, repo="r", include_tests=True)
    names = [p.name for p in keep]
    assert "prod_flow" in names
    assert "test_rooted_flow" in names
    assert "nested_test_flow" in names
    assert "test_prefix_flow" in names
    assert "mixed_flow" in names


def test_find_processes_limit_respected():
    rows = [
        _row(f"flow_{i}", [f"x.{i}.{j}" for j in range(i + 1)], ["app/x.py"] * (i + 1))
        for i in range(20)
    ]
    driver = _FakeDriver(rows)
    out = find_processes(driver, repo="r", limit=5)
    assert len(out) == 5
    # And ordered by step count desc.
    assert out[0].step_count >= out[-1].step_count


def test_find_processes_member_qns_in_step_order():
    """Members come out in the order Cypher produced them (step asc).

    The production query orders by p.name, step before collect(), so the
    callback receives chains pre-sorted. We assert the function preserves
    that ordering when it builds the ProcessSummary tuple.
    """
    driver = _FakeDriver([
        _row("flow", ["m.0", "m.1", "m.2", "m.3"], ["app/m.py"] * 4),
    ])
    out = find_processes(driver, repo="r")
    assert len(out) == 1
    assert out[0].member_qns == ("m.0", "m.1", "m.2", "m.3")
    assert out[0].step_count == 4


def test_find_processes_returns_processsummary_dataclasses():
    driver = _FakeDriver([
        _row("flow", ["a.b"], ["app/a.py"], summary="hello"),
    ])
    out = find_processes(driver, repo="r")
    assert isinstance(out, list)
    assert all(isinstance(p, ProcessSummary) for p in out)
    assert out[0].summary == "hello"


def test_find_processes_emits_test_path_predicate_in_cypher():
    """The Cypher MUST mention all five test-path predicates when
    include_tests=False (the default). This is the load-bearing filter."""
    driver = _FakeDriver([])
    find_processes(driver, repo="r")  # default: include_tests=False
    q = driver.session_obj.last_query or ""
    assert "STARTS WITH 'tests/'" in q
    assert "STARTS WITH 'test/'" in q
    assert "CONTAINS '/tests/'" in q
    assert "CONTAINS '/test_'" in q
    assert "STARTS WITH 'test_'" in q


def test_find_processes_omits_test_predicate_when_include_tests_true():
    driver = _FakeDriver([])
    find_processes(driver, repo="r", include_tests=True)
    q = driver.session_obj.last_query or ""
    # The member-file predicate must be absent. (`tests/` only appears
    # inside that filter; if the filter is gone the literal can't show
    # up either.)
    assert "STARTS WITH 'tests/'" not in q


def test_find_processes_passes_query_param_when_provided():
    driver = _FakeDriver([])
    find_processes(driver, repo="r", query="auth")
    assert driver.session_obj.last_params.get("query") == "auth"


def test_find_processes_does_not_pass_query_param_when_blank():
    driver = _FakeDriver([])
    find_processes(driver, repo="r", query="")
    assert "query" not in driver.session_obj.last_params
    find_processes(driver, repo="r", query=None)
    assert "query" not in driver.session_obj.last_params
