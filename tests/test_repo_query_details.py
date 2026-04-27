"""Sprint 14c — `find_module_detail` and `find_process_detail` tests.

Mirrors the synthetic-driver style of `test_repo_query_modules.py` /
`test_repo_query_processes.py`: no live Neo4j; a fake Session.run that
returns canned `single()` records.
"""
from __future__ import annotations

from typing import Any

from app.repo_query import (
    ModuleDetail,
    ProcessDetail,
    ProcessStep,
    find_module_detail,
    find_process_detail,
)


# ─── Synthetic driver helpers ───────────────────────────────────────────────
# Production code calls:
#     with driver.session() as session:
#         rec = session.run(cypher, **params).single()
# The detail helpers expect either a record dict or None; we replicate
# both paths.


class _FakeSingleResult:
    """Models the `Result` object returned by `session.run(...)`. We only
    need `.single()` here because both detail helpers use it."""

    def __init__(self, record: dict[str, Any] | None):
        self._record = record

    def single(self):
        # neo4j returns a `Record` (dict-like) or None when no rows.
        return self._record


class _FakeSession:
    def __init__(self, record: dict[str, Any] | None):
        self._record = record
        self.last_query: str | None = None
        self.last_params: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        cypher = args[0] if args else ""
        self.last_query = cypher
        self.last_params = params
        return _FakeSingleResult(self._record)


class _FakeDriver:
    def __init__(self, record: dict[str, Any] | None):
        self._record = record
        self.session_obj = _FakeSession(record)

    def session(self):
        return self.session_obj


# ─── find_module_detail ─────────────────────────────────────────────────────


def test_find_module_detail_missing_returns_none():
    """No matching Community ⇒ None (the resource layer renders found:false)."""
    driver = _FakeDriver(None)
    assert find_module_detail(driver, repo="r", label="nope") is None


def test_find_module_detail_returns_full_member_list():
    """The detail view returns ALL members, not a 5-element sample.

    `find_modules` (the list view) caps `sample_member_qns` at 5; this
    new helper must NOT — it's the resource-layer answer to "show me
    everything in cluster X".
    """
    members = [f"app.mod_{i}" for i in range(20)]
    driver = _FakeDriver({
        "label": "big_cluster",
        "cohesion": 0.42,
        "size": 20,
        "member_qns": members,
    })
    detail = find_module_detail(driver, repo="r", label="big_cluster")
    assert detail is not None
    assert isinstance(detail, ModuleDetail)
    assert detail.label == "big_cluster"
    assert detail.cohesion == 0.42
    assert detail.size == 20
    assert len(detail.member_qns) == 20  # full list, not capped at 5


def test_find_module_detail_member_qns_sorted_lexicographically():
    """Graph-side member ordering isn't stable; the helper sorts for
    deterministic output. Same convention as `find_modules` sample."""
    raw = ["zeta.x", "alpha.x", "mu.x", "beta.x"]
    driver = _FakeDriver({
        "label": "c",
        "cohesion": 0.0,
        "size": 4,
        "member_qns": raw,
    })
    detail = find_module_detail(driver, repo="r", label="c")
    assert detail is not None
    assert detail.member_qns == ("alpha.x", "beta.x", "mu.x", "zeta.x")


# ─── find_process_detail ────────────────────────────────────────────────────


def test_find_process_detail_missing_returns_none():
    driver = _FakeDriver(None)
    assert find_process_detail(driver, repo="r", name="ghost") is None


def test_find_process_detail_returns_steps_in_order():
    """Steps come out in step-asc order (the Cypher already ORDER BYs them
    pre-collect, but the helper must preserve that)."""
    driver = _FakeDriver({
        "name": "checkout_flow",
        "summary": "user purchases an item",
        "steps": [
            {"step": 0, "qn": "app.api.start_checkout", "file": "app/api.py", "line_start": 10},
            {"step": 1, "qn": "app.cart.lock", "file": "app/cart.py", "line_start": 22},
            {"step": 2, "qn": "app.payments.charge", "file": "app/payments.py", "line_start": 88},
        ],
    })
    detail = find_process_detail(driver, repo="r", name="checkout_flow")
    assert detail is not None
    assert isinstance(detail, ProcessDetail)
    assert detail.name == "checkout_flow"
    assert detail.summary == "user purchases an item"
    assert len(detail.steps) == 3
    # Step indices come out 0, 1, 2 — the order of the list itself.
    assert [s.step for s in detail.steps] == [0, 1, 2]
    assert all(isinstance(s, ProcessStep) for s in detail.steps)
    assert detail.steps[0].member_qn == "app.api.start_checkout"
    assert detail.steps[2].file_path == "app/payments.py"
    assert detail.steps[2].line_start == 88


def test_find_process_detail_handles_zero_step_process():
    """OPTIONAL MATCH against STEP_IN_PROCESS yields a sentinel `[{...}]`
    when the process has no steps. The helper must filter those out
    rather than emitting a fake step entry."""
    driver = _FakeDriver({
        "name": "empty_flow",
        "summary": "",
        # Sentinel: a step row with all-None values from OPTIONAL MATCH.
        "steps": [{"step": None, "qn": None, "file": None, "line_start": None}],
    })
    detail = find_process_detail(driver, repo="r", name="empty_flow")
    assert detail is not None
    assert detail.name == "empty_flow"
    assert detail.steps == ()
