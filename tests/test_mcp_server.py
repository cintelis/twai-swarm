"""Sprint 14c — MCP server resource + tool handler tests.

We test the resource and tool handlers as pure functions, NOT through
the FastMCP stdio runtime. Testing through stdio is end-to-end work that
belongs in an integration suite; here we only need to know that each
handler emits the right YAML for the right Cypher result.

All tests skip if the `mcp` SDK isn't importable — the established
established pattern for optional-dep gates in this repo (mirrors
`test_repo_indexer_embed.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# Skip entire module if mcp SDK is unavailable. The handlers themselves
# don't import mcp, but the package's `__init__.py` is imported by the
# tests indirectly through `app.mcp_server.resources` — and importing
# `app.mcp_server.__main__` requires `mcp`. We gate on the dep so a
# bare `pip install -e .` without [mcp] still passes the rest of the
# suite.
mcp = pytest.importorskip("mcp")  # noqa: F841

from app import repo_query  # noqa: E402
from app.mcp_server import resources, tools  # noqa: E402


# ─── Synthetic Neo4j driver ─────────────────────────────────────────────────
# Each handler only ever calls `driver.session()` and uses the result as
# a context manager. The MCP server tests need richer fakes than the
# repo_query tests because some handlers run multiple Cypher queries
# (e.g. `build_context_yaml` runs the existence check + the stats
# aggregation). We model that with a list of canned results consumed
# in order.


@dataclass
class _Canned:
    """One pre-built Cypher response: either rows (for `.data()`) or a
    single record (for `.single()`). The handler picks one path based
    on which Cypher it ran; we don't try to match cypher strings."""
    rows: list[dict[str, Any]] = field(default_factory=list)
    single: dict[str, Any] | None = None


class _FakeResult:
    def __init__(self, canned: _Canned):
        self._canned = canned

    def data(self):
        return list(self._canned.rows)

    def single(self):
        return self._canned.single


class _FakeSession:
    """Returns canned responses in order. `last_query` and `query_log`
    are introspectable for assertions."""

    def __init__(self, queue: list[_Canned]):
        self._queue = list(queue)
        self.last_query: str | None = None
        self.last_params: dict[str, Any] = {}
        self.query_log: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **params):
        cypher = args[0] if args else ""
        self.last_query = cypher
        self.last_params = params
        self.query_log.append((cypher, dict(params)))
        if not self._queue:
            return _FakeResult(_Canned(rows=[], single=None))
        return _FakeResult(self._queue.pop(0))


class _FakeDriver:
    def __init__(self, queue: list[_Canned] | None = None):
        self._session = _FakeSession(queue or [])

    def session(self):
        return self._session


# ─── Resource handler tests ─────────────────────────────────────────────────


def test_resource_repos_lists_all():
    """`twai://repos` returns every Repo node, sorted by name."""
    driver = _FakeDriver([
        _Canned(rows=[
            {"name": "alpha", "url": "https://x/alpha", "commit_sha": "aaa", "tenant_id": "t1"},
            {"name": "bravo", "url": "https://x/bravo", "commit_sha": "bbb", "tenant_id": "t1"},
        ]),
    ])
    out = resources.build_repos_yaml(driver)
    parsed = yaml.safe_load(out)
    assert "repos" in parsed
    assert [r["name"] for r in parsed["repos"]] == ["alpha", "bravo"]
    assert parsed["repos"][0]["url"] == "https://x/alpha"


def test_resource_context_emits_stats():
    """`twai://repo/{name}/context` includes file/function/class counts."""
    driver = _FakeDriver([
        # _repo_exists check
        _Canned(single={"name": "twai-swarm"}),
        # _repo_stats aggregation
        _Canned(single={
            "files": 42, "modules": 12, "classes": 18,
            "functions": 137, "communities": 5, "processes": 3,
        }),
    ])
    out = resources.build_context_yaml(driver, "twai-swarm")
    parsed = yaml.safe_load(out)
    assert parsed["name"] == "twai-swarm"
    assert parsed["available"] is True
    assert parsed["stats"]["files"] == 42
    assert parsed["stats"]["functions"] == 137
    # Tool catalog must be present.
    assert "tools" in parsed
    tool_names = [t["name"] for t in parsed["tools"]]
    assert {"query", "context", "find_symbol"} <= set(tool_names)


def test_resource_clusters_yaml_shape():
    """`twai://repo/{name}/clusters` is well-formed YAML with expected keys."""
    fake_modules = [
        repo_query.ModuleSummary(label="alpha", cohesion=0.7, size=10,
                                 sample_member_qns=("a.x", "a.y")),
        repo_query.ModuleSummary(label="beta", cohesion=0.4, size=4,
                                 sample_member_qns=("b.x",)),
    ]
    with patch.object(repo_query, "find_modules", return_value=fake_modules):
        out = resources.build_clusters_yaml(_FakeDriver(), "myrepo")
    parsed = yaml.safe_load(out)
    assert parsed["repo"] == "myrepo"
    assert parsed["count"] == 2
    assert len(parsed["clusters"]) == 2
    first = parsed["clusters"][0]
    # All expected keys present.
    assert set(first.keys()) >= {"label", "cohesion", "size", "sample_member_qns"}
    assert first["label"] == "alpha"
    assert first["sample_member_qns"] == ["a.x", "a.y"]


def test_resource_cluster_detail_full_member_list():
    """`cluster/{label}` resource returns ALL members, not a sample."""
    full_members = tuple(f"mod.{i}" for i in range(15))
    fake_detail = repo_query.ModuleDetail(
        label="big",
        cohesion=0.55,
        size=15,
        member_qns=full_members,
    )
    with patch.object(repo_query, "find_module_detail", return_value=fake_detail):
        out = resources.build_cluster_detail_yaml(_FakeDriver(), "myrepo", "big")
    parsed = yaml.safe_load(out)
    assert parsed["found"] is True
    assert parsed["label"] == "big"
    assert parsed["size"] == 15
    # Full list — sample-cap of 5 must NOT apply.
    assert len(parsed["member_qns"]) == 15
    assert parsed["member_qns"][0] == "mod.0"


def test_resource_cluster_detail_missing_returns_found_false():
    """Missing cluster yields `found: false` payload, not a crash."""
    with patch.object(repo_query, "find_module_detail", return_value=None):
        out = resources.build_cluster_detail_yaml(_FakeDriver(), "myrepo", "ghost")
    parsed = yaml.safe_load(out)
    assert parsed["found"] is False
    assert "ghost" in parsed["message"]


def test_resource_processes_ordered_by_step_count():
    """List of processes ranked. (We rely on `find_processes` to do the
    ordering; the resource just passes through.)"""
    fake_procs = [
        repo_query.ProcessSummary(name="big", summary="", step_count=10,
                                  member_qns=tuple(f"x.{i}" for i in range(10))),
        repo_query.ProcessSummary(name="small", summary="", step_count=2,
                                  member_qns=("a.b", "a.c")),
    ]
    with patch.object(repo_query, "find_processes", return_value=fake_procs):
        out = resources.build_processes_yaml(_FakeDriver(), "myrepo")
    parsed = yaml.safe_load(out)
    assert parsed["count"] == 2
    assert [p["name"] for p in parsed["processes"]] == ["big", "small"]
    assert parsed["processes"][0]["step_count"] == 10


def test_resource_process_detail_steps_ordered():
    """Single-process resource has steps in 0..N-1 order."""
    fake_detail = repo_query.ProcessDetail(
        name="checkout",
        summary="purchase flow",
        steps=(
            repo_query.ProcessStep(step=0, member_qn="api.start", file_path="api.py", line_start=10),
            repo_query.ProcessStep(step=1, member_qn="cart.lock", file_path="cart.py", line_start=22),
            repo_query.ProcessStep(step=2, member_qn="pay.charge", file_path="pay.py", line_start=88),
        ),
    )
    with patch.object(repo_query, "find_process_detail", return_value=fake_detail):
        out = resources.build_process_detail_yaml(_FakeDriver(), "myrepo", "checkout")
    parsed = yaml.safe_load(out)
    assert parsed["found"] is True
    assert parsed["process"] == "checkout"
    assert parsed["step_count"] == 3
    assert [s["step"] for s in parsed["steps"]] == [0, 1, 2]
    assert parsed["steps"][2]["member_qn"] == "pay.charge"


def test_resource_process_detail_missing_returns_found_false():
    """Same edge case as cluster detail: missing process is graceful."""
    with patch.object(repo_query, "find_process_detail", return_value=None):
        out = resources.build_process_detail_yaml(_FakeDriver(), "myrepo", "missing")
    parsed = yaml.safe_load(out)
    assert parsed["found"] is False
    assert "missing" in parsed["message"]


def test_resource_handlers_handle_missing_repo():
    """Querying a repo that isn't in Neo4j returns sensible empty / not-found.

    `_repo_exists` returns None ⇒ `build_context_yaml` reports
    `available: false` plus the empty stats catalog. No crash."""
    driver = _FakeDriver([
        _Canned(single=None),  # _repo_exists ⇒ None
    ])
    out = resources.build_context_yaml(driver, "nonexistent-repo")
    parsed = yaml.safe_load(out)
    assert parsed["name"] == "nonexistent-repo"
    assert parsed["available"] is False
    assert "nonexistent-repo" in parsed["message"]
    # Tool catalog still present so the client can still discover tools.
    assert "tools" in parsed


# ─── Tool handler tests ─────────────────────────────────────────────────────


def test_tool_query_routes_to_semantic_search():
    """The `query` tool delegates to `repo_query.semantic_search` with
    the right args (driver, repo, query, k=limit)."""
    fake_hits = [
        repo_query.SemanticHit(
            qualified_name="app.auth.login", name="login", kind="function",
            file_path="app/auth.py", line_start=10, docstring="logs in",
            rrf_score=0.0312,
        ),
    ]
    with patch.object(repo_query, "semantic_search", return_value=fake_hits) as mock:
        out = tools.query_tool(_FakeDriver(), "myrepo", "auth login", limit=5)
    # Args check: positional driver+repo+query, k=limit kw.
    args, kwargs = mock.call_args
    assert args[1] == "myrepo"
    assert args[2] == "auth login"
    assert kwargs.get("k") == 5

    parsed = yaml.safe_load(out)
    assert parsed["query"] == "auth login"
    assert parsed["count"] == 1
    assert parsed["hits"][0]["qualified_name"] == "app.auth.login"
    # rrf_score is rounded to 6 places — make sure it survives the round-trip.
    assert parsed["hits"][0]["rrf_score"] == pytest.approx(0.0312, abs=1e-6)


def test_tool_context_aggregates_definition_callers_callees():
    """The `context` tool fans out to all three repo_query helpers."""
    fake_def = repo_query.Definition(
        qualified_name="app.auth.login", kind="function",
        file_path="app/auth.py", line_start=10, line_end=30, docstring="docs",
    )
    fake_callers = [
        repo_query.CallSite(caller_qn="app.api.handler", callee_qn="app.auth.login",
                            file_path="app/api.py", line=42),
    ]
    fake_callees = [
        repo_query.CallSite(caller_qn="app.auth.login", callee_qn="app.db.q",
                            file_path="app/auth.py", line=20),
    ]
    with patch.object(repo_query, "find_definition", return_value=fake_def) as m_def, \
         patch.object(repo_query, "find_callers", return_value=fake_callers) as m_in, \
         patch.object(repo_query, "find_callees", return_value=fake_callees) as m_out:
        out = tools.context_tool(_FakeDriver(), "myrepo", "app.auth.login")

    # All three repo_query helpers were called once each, with the same
    # repo + qualified_name pair.
    assert m_def.call_count == 1
    assert m_in.call_count == 1
    assert m_out.call_count == 1
    for mock in (m_def, m_in, m_out):
        args = mock.call_args.args
        assert args[1] == "myrepo"
        assert args[2] == "app.auth.login"

    parsed = yaml.safe_load(out)
    assert parsed["found"] is True
    assert parsed["definition"]["kind"] == "function"
    assert parsed["definition"]["line_start"] == 10
    assert len(parsed["callers"]) == 1
    assert len(parsed["callees"]) == 1
    assert parsed["callers"][0]["caller_qn"] == "app.api.handler"


def test_tool_context_missing_definition_skips_caller_lookups():
    """When `find_definition` returns None we report `found:false` and
    skip the (expensive) callers/callees follow-ups."""
    with patch.object(repo_query, "find_definition", return_value=None) as m_def, \
         patch.object(repo_query, "find_callers") as m_in, \
         patch.object(repo_query, "find_callees") as m_out:
        out = tools.context_tool(_FakeDriver(), "myrepo", "ghost.qn")

    assert m_def.call_count == 1
    assert m_in.call_count == 0
    assert m_out.call_count == 0

    parsed = yaml.safe_load(out)
    assert parsed["found"] is False
    assert parsed["qualified_name"] == "ghost.qn"


def test_tool_find_symbol_routes_correctly():
    """The find_symbol tool wraps repo_query.find_symbol."""
    fake_matches = [
        repo_query.SymbolMatch(
            qualified_name="app.auth.login", kind="function",
            file_path="app/auth.py", line_start=10,
        ),
    ]
    with patch.object(repo_query, "find_symbol", return_value=fake_matches) as mock:
        out = tools.find_symbol_tool(_FakeDriver(), "myrepo", "login", limit=7)
    args, kwargs = mock.call_args
    assert args[1] == "myrepo"
    assert args[2] == "login"
    assert kwargs.get("limit") == 7

    parsed = yaml.safe_load(out)
    assert parsed["count"] == 1
    assert parsed["matches"][0]["qualified_name"] == "app.auth.login"
