"""Coder tool behaviour: stats tracking, verify subprocess plumbing.

We don't exercise the @beta_async_tool decorator's tool-runner integration
here — that's covered by the tool_runner loop tests. These tests call the
underlying async functions directly.
"""
from __future__ import annotations

import asyncio
import json
import shutil

import pytest

from app.agents.coder_sandbox import Sandbox
from app.agents.coder_tools import build_tools


def _sandbox(tmp_path, wid="wf-tools"):
    return Sandbox.create(wid, base=tmp_path)


async def _call(tool, **kwargs):
    """Unwrap @beta_async_tool to call the underlying function directly.

    The decorator stores the original function at `.__wrapped__` (or a
    similar attr depending on SDK version). We test the callable itself.
    """
    fn = getattr(tool, "__wrapped__", None) or getattr(tool, "function", None) or tool
    # If the SDK exposes a coroutine call directly on the decorated object
    # (e.g., tool(**kwargs) returning a coroutine), fall through to that.
    if callable(fn) and not asyncio.iscoroutinefunction(fn):
        # Some decorator versions wrap with a sync-callable that returns a coro.
        result = fn(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    return await fn(**kwargs)


def test_build_tools_returns_five_without_neo4j(tmp_path):
    """Default build_tools call (no Neo4j driver) returns the 5 sandbox tools."""
    sb = _sandbox(tmp_path)
    tools, stats = build_tools(sb)
    assert len(tools) == 5
    for key in ("list_files_calls", "read_file_calls", "write_file_calls",
                "run_verify_calls", "bash_exec_calls"):
        assert stats[key] == 0


def test_build_tools_adds_graph_tools_with_neo4j(tmp_path):
    """When neo4j_driver + repo_name are passed, the repo_* tool family joins.

    Sprint 10c added 3 graph tools (search/find_definition/find_callers);
    Sprint 13c added 2 more (find_processes/find_modules) for high-level
    discoverability. So 5 sandbox + 5 graph = 10 total.
    """
    sb = _sandbox(tmp_path)
    fake_driver = object()
    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="repo")
    assert len(tools) == 10
    for key in (
        "repo_search_calls", "repo_find_definition_calls", "repo_find_callers_calls",
        "repo_find_processes_calls", "repo_find_modules_calls",
    ):
        assert stats[key] == 0


def test_repo_discoverability_tools_only_when_driver_and_repo_name(tmp_path):
    """Sprint 13c: repo_find_processes / repo_find_modules must be opt-in.

    Without a driver, build_tools returns only the 5 sandbox tools (no
    repo_* tools at all). With both, the 5 graph tools — including the
    two new discoverability ones — are present.
    """
    sb = _sandbox(tmp_path)

    bare_tools, _ = build_tools(sb)
    bare_names = {t.name for t in bare_tools}
    assert "repo_find_processes" not in bare_names
    assert "repo_find_modules" not in bare_names

    full_tools, _ = build_tools(sb, neo4j_driver=object(), repo_name="repo")
    full_names = {t.name for t in full_tools}
    assert "repo_find_processes" in full_names
    assert "repo_find_modules" in full_names


def test_build_tools_requires_repo_name_with_driver(tmp_path):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    with pytest.raises(ValueError, match="repo_name is required"):
        build_tools(sb, neo4j_driver=fake_driver)


@pytest.mark.asyncio
async def test_list_write_read_cycle(tmp_path):
    sb = _sandbox(tmp_path)
    (list_files, read_file, write_file, run_verify, bash_exec), stats = _unpack(build_tools(sb))

    out = await _call(list_files)
    assert "(workspace is empty)" in out

    out = await _call(write_file, path="hi.txt", content="hello")
    assert "wrote 5 bytes" in out
    assert stats["bytes_written"] == 5

    out = await _call(list_files)
    assert "hi.txt" in out

    out = await _call(read_file, path="hi.txt")
    assert out == "hello"

    assert stats["list_files_calls"] == 2
    assert stats["write_file_calls"] == 1
    assert stats["read_file_calls"] == 1


@pytest.mark.asyncio
async def test_path_escape_returns_error_string(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, write_file, _, _), _ = _unpack(build_tools(sb))
    out = await _call(write_file, path="../escape.txt", content="x")
    # Tools return error strings rather than raising — the model sees the
    # error as tool_result content and can recover.
    assert out.startswith("error:")


@pytest.mark.asyncio
async def test_run_verify_reports_exit_code(tmp_path):
    sb = _sandbox(tmp_path)
    # Minimal passing verify.sh — write bytes so Windows keeps LF and the bash
    # shebang line doesn't get a trailing CR (which makes bash exit 2).
    (sb.root / "verify.sh").write_bytes(b"#!/usr/bin/env bash\necho ok\nexit 0\n")
    (_, _, _, run_verify, _), stats = _unpack(build_tools(sb))

    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH — skipping verify subprocess test")

    out = await _call(run_verify)
    parsed = json.loads(out)
    assert parsed["exit_code"] == 0
    assert parsed["stage"] == "all"
    assert "ok" in parsed["stdout_tail"]
    assert stats["last_verify_exit"] == 0
    assert stats["last_verify_stage"] == "all"
    assert stats["run_verify_calls"] == 1


@pytest.mark.asyncio
async def test_run_verify_captures_failure(tmp_path):
    sb = _sandbox(tmp_path)
    (sb.root / "verify.sh").write_bytes(b"#!/usr/bin/env bash\necho oops >&2\nexit 7\n")
    (_, _, _, run_verify, _), stats = _unpack(build_tools(sb))

    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH — skipping verify subprocess test")

    out = await _call(run_verify)
    parsed = json.loads(out)
    assert parsed["exit_code"] == 7
    assert "oops" in parsed["stderr_tail"]
    assert stats["last_verify_exit"] == 7


@pytest.mark.asyncio
async def test_run_verify_passes_stage_arg(tmp_path):
    """Stage arg should be passed through to verify.sh as $1."""
    sb = _sandbox(tmp_path)
    (sb.root / "verify.sh").write_bytes(
        b"#!/usr/bin/env bash\necho \"stage=$1\"\nexit 0\n"
    )
    (_, _, _, run_verify, _), stats = _unpack(build_tools(sb))
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")

    out = await _call(run_verify, stage="lint")
    parsed = json.loads(out)
    assert parsed["stage"] == "lint"
    assert "stage=lint" in parsed["stdout_tail"]
    assert stats["last_verify_stage"] == "lint"


@pytest.mark.asyncio
async def test_run_verify_rejects_bad_stage(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, run_verify, _), _ = _unpack(build_tools(sb))
    out = await _call(run_verify, stage="bogus")
    assert out.startswith("error: stage must be one of")


@pytest.mark.asyncio
async def test_run_verify_missing_script(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, run_verify, _), _ = _unpack(build_tools(sb))
    out = await _call(run_verify)
    assert "verify.sh not found" in out


@pytest.mark.asyncio
async def test_bash_exec_runs_command(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, _, bash_exec), stats = _unpack(build_tools(sb))
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")

    out = await _call(bash_exec, command="echo hello && pwd")
    parsed = json.loads(out)
    assert parsed["exit_code"] == 0
    assert "hello" in parsed["stdout"]
    # cwd should be the sandbox root (resolve to handle macOS /private/var symlink)
    assert str(sb.root.resolve()) in parsed["stdout"] or sb.root.name in parsed["stdout"]
    assert stats["bash_exec_calls"] == 1


@pytest.mark.asyncio
async def test_bash_exec_strips_secrets_from_env(tmp_path, monkeypatch):
    """Coder must not be able to read provider/AWS/Temporal secrets via printenv."""
    sb = _sandbox(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shouldnotleak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIA-shouldnotleak")
    monkeypatch.setenv("PG_DSN", "postgres://shouldnotleak")
    (_, _, _, _, bash_exec), _ = _unpack(build_tools(sb))
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")

    out = await _call(bash_exec, command="env | sort")
    parsed = json.loads(out)
    combined = parsed["stdout"] + parsed["stderr"]
    # Threat-model assertion: secrets must not be visible to the subprocess.
    # We don't assert AWS_EC2_METADATA_DISABLED is in `env` output because
    # the WSL bridge swallows custom env on Windows-hosted bash; in the
    # Linux production container it's set correctly.
    assert "shouldnotleak" not in combined


@pytest.mark.asyncio
async def test_bash_exec_timeout(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, _, bash_exec), _ = _unpack(build_tools(sb))
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")

    out = await _call(bash_exec, command="sleep 5", timeout_seconds=1)
    parsed = json.loads(out)
    assert parsed["exit_code"] == -1
    assert "timed out" in parsed["stderr"]


@pytest.mark.asyncio
async def test_bash_exec_rejects_empty_command(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, _, bash_exec), _ = _unpack(build_tools(sb))
    out = await _call(bash_exec, command="   ")
    assert out.startswith("error:")


def _unpack(built):
    """build_tools returns (list, stats); split the list into the 5 tools."""
    tools, stats = built
    assert len(tools) == 5
    return tuple(tools), stats


# ─── Graph tools (Sprint 10c) ───────────────────────────────────────────────
# We pass a sentinel `fake_driver` and monkeypatch app.repo_query to return
# canned responses. That keeps these tests fast (no Neo4j needed) while
# proving the tool wiring + JSON shape.

@pytest.mark.asyncio
async def test_repo_search_returns_json(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    captured = {}
    def fake_find_symbol(driver, repo, name, limit):
        captured["args"] = (driver, repo, name, limit)
        return [
            repo_query.SymbolMatch(qualified_name="app.foo.bar", kind="function",
                                   file_path="app/foo.py", line_start=10),
        ]
    monkeypatch.setattr(repo_query, "find_symbol", fake_find_symbol)

    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    repo_search = tools[5]   # 5 sandbox tools first, then 3 graph tools

    out = await _call(repo_search, query="bar", limit=10)
    parsed = json.loads(out)
    assert len(parsed) == 1
    assert parsed[0]["qualified_name"] == "app.foo.bar"
    assert parsed[0]["kind"] == "function"
    assert captured["args"] == (fake_driver, "r", "bar", 10)
    assert stats["repo_search_calls"] == 1


@pytest.mark.asyncio
async def test_repo_search_clamps_limit(tmp_path, monkeypatch):
    """limit is clamped to [1, 100] — caller can't request unbounded results."""
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    seen_limit = {}
    def fake_find_symbol(driver, repo, name, limit):
        seen_limit["v"] = limit
        return []
    monkeypatch.setattr(repo_query, "find_symbol", fake_find_symbol)

    tools, _ = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    repo_search = tools[5]
    await _call(repo_search, query="x", limit=10000)
    assert seen_limit["v"] == 100


@pytest.mark.asyncio
async def test_repo_find_definition_returns_json(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    def fake_find_definition(driver, repo, qn):
        return repo_query.Definition(
            qualified_name=qn, kind="function",
            file_path="app/foo.py", line_start=10, line_end=20,
            docstring="hello",
        )
    monkeypatch.setattr(repo_query, "find_definition", fake_find_definition)

    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    repo_find_definition = tools[6]

    out = await _call(repo_find_definition, qualified_name="app.foo.bar")
    parsed = json.loads(out)
    assert parsed["qualified_name"] == "app.foo.bar"
    assert parsed["kind"] == "function"
    assert parsed["line_start"] == 10
    assert parsed["docstring"] == "hello"
    assert stats["repo_find_definition_calls"] == 1


@pytest.mark.asyncio
async def test_repo_find_definition_returns_null_when_missing(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query
    monkeypatch.setattr(repo_query, "find_definition", lambda d, r, qn: None)

    tools, _ = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    repo_find_definition = tools[6]

    out = await _call(repo_find_definition, qualified_name="nope")
    assert out == "null"


@pytest.mark.asyncio
async def test_repo_find_callers_returns_json(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    def fake_find_callers(driver, repo, qn):
        return [
            repo_query.CallSite(caller_qn="app.x.use_it", callee_qn=qn,
                                file_path="app/x.py", line=42),
            repo_query.CallSite(caller_qn="app.y.also", callee_qn=qn,
                                file_path="app/y.py", line=10),
        ]
    monkeypatch.setattr(repo_query, "find_callers", fake_find_callers)

    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    repo_find_callers = tools[7]

    out = await _call(repo_find_callers, qualified_name="app.foo.bar")
    parsed = json.loads(out)
    assert len(parsed) == 2
    assert parsed[0]["caller_qn"] == "app.x.use_it"
    assert parsed[0]["line"] == 42
    assert stats["repo_find_callers_calls"] == 1


# ─── Discoverability tools (Sprint 13c) ─────────────────────────────────────


def _tools_by_name(tools):
    return {t.name: t for t in tools}


@pytest.mark.asyncio
async def test_repo_find_processes_tool_calls_query_layer(tmp_path, monkeypatch):
    """The Coder tool routes args through to repo_query.find_processes
    and serialises the dataclass output as JSON the model can consume."""
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    captured: dict = {}

    def fake_find_processes(driver, repo, query, limit, include_tests):
        captured["args"] = (driver, repo, query, limit, include_tests)
        return [
            repo_query.ProcessSummary(
                name="login -> handler",
                summary="auth login flow",
                step_count=3,
                member_qns=("app.auth.login", "app.auth.verify", "app.auth.handler"),
            ),
        ]
    monkeypatch.setattr(repo_query, "find_processes", fake_find_processes)

    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    tool = _tools_by_name(tools)["repo_find_processes"]

    out = await _call(tool, query="auth", limit=5)
    parsed = json.loads(out)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "login -> handler"
    assert parsed[0]["step_count"] == 3
    assert parsed[0]["member_qns"] == ["app.auth.login", "app.auth.verify", "app.auth.handler"]
    # include_tests must be False by default — the Coder shouldn't see noise.
    assert captured["args"] == (fake_driver, "r", "auth", 5, False)
    assert stats["repo_find_processes_calls"] == 1


@pytest.mark.asyncio
async def test_repo_find_processes_blank_query_passes_none(tmp_path, monkeypatch):
    """Empty/whitespace query strings should reach the query layer as None
    so the Cypher's substring filter is dropped entirely (not run with '')."""
    sb = _sandbox(tmp_path)
    from app import repo_query

    seen: dict = {}
    monkeypatch.setattr(repo_query, "find_processes",
                        lambda d, r, q, lim, it: seen.setdefault("q", q) or [])

    tools, _ = build_tools(sb, neo4j_driver=object(), repo_name="r")
    tool = _tools_by_name(tools)["repo_find_processes"]

    await _call(tool, query="   ", limit=10)
    assert seen["q"] is None


@pytest.mark.asyncio
async def test_repo_find_processes_clamps_limit(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    from app import repo_query

    seen: dict = {}
    def fake(driver, repo, query, limit, include_tests):
        seen["limit"] = limit
        return []
    monkeypatch.setattr(repo_query, "find_processes", fake)

    tools, _ = build_tools(sb, neo4j_driver=object(), repo_name="r")
    tool = _tools_by_name(tools)["repo_find_processes"]
    await _call(tool, query="x", limit=10000)
    assert seen["limit"] == 100


@pytest.mark.asyncio
async def test_repo_find_modules_tool_calls_query_layer(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    fake_driver = object()
    from app import repo_query

    captured: dict = {}

    def fake_find_modules(driver, repo, limit, include_tests):
        captured["args"] = (driver, repo, limit, include_tests)
        return [
            repo_query.ModuleSummary(
                label="auth",
                cohesion=0.82,
                size=7,
                sample_member_qns=("app.auth.login", "app.auth.logout", "app.auth.verify"),
            ),
        ]
    monkeypatch.setattr(repo_query, "find_modules", fake_find_modules)

    tools, stats = build_tools(sb, neo4j_driver=fake_driver, repo_name="r")
    tool = _tools_by_name(tools)["repo_find_modules"]

    out = await _call(tool, limit=15)
    parsed = json.loads(out)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["label"] == "auth"
    assert parsed[0]["size"] == 7
    assert parsed[0]["cohesion"] == pytest.approx(0.82)
    assert parsed[0]["sample_member_qns"] == [
        "app.auth.login", "app.auth.logout", "app.auth.verify",
    ]
    assert captured["args"] == (fake_driver, "r", 15, False)
    assert stats["repo_find_modules_calls"] == 1


@pytest.mark.asyncio
async def test_repo_find_modules_clamps_limit(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    from app import repo_query

    seen: dict = {}
    def fake(driver, repo, limit, include_tests):
        seen["limit"] = limit
        return []
    monkeypatch.setattr(repo_query, "find_modules", fake)

    tools, _ = build_tools(sb, neo4j_driver=object(), repo_name="r")
    tool = _tools_by_name(tools)["repo_find_modules"]
    await _call(tool, limit=99999)
    assert seen["limit"] == 100
