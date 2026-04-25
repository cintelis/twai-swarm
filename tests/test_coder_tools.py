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


def test_build_tools_returns_five(tmp_path):
    sb = _sandbox(tmp_path)
    tools, stats = build_tools(sb)
    assert len(tools) == 5
    for key in ("list_files_calls", "read_file_calls", "write_file_calls",
                "run_verify_calls", "bash_exec_calls"):
        assert stats[key] == 0


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
