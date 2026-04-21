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


def test_build_tools_returns_four(tmp_path):
    sb = _sandbox(tmp_path)
    tools, stats = build_tools(sb)
    assert len(tools) == 4
    for key in ("list_files_calls", "read_file_calls", "write_file_calls", "run_verify_calls"):
        assert stats[key] == 0


@pytest.mark.asyncio
async def test_list_write_read_cycle(tmp_path):
    sb = _sandbox(tmp_path)
    (list_files, read_file, write_file, run_verify), stats = _unpack(build_tools(sb))

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
    (_, _, write_file, _), _ = _unpack(build_tools(sb))
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
    (_, _, _, run_verify), stats = _unpack(build_tools(sb))

    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH — skipping verify subprocess test")

    out = await _call(run_verify)
    parsed = json.loads(out)
    assert parsed["exit_code"] == 0
    assert "ok" in parsed["stdout_tail"]
    assert stats["last_verify_exit"] == 0
    assert stats["run_verify_calls"] == 1


@pytest.mark.asyncio
async def test_run_verify_captures_failure(tmp_path):
    sb = _sandbox(tmp_path)
    (sb.root / "verify.sh").write_bytes(b"#!/usr/bin/env bash\necho oops >&2\nexit 7\n")
    (_, _, _, run_verify), stats = _unpack(build_tools(sb))

    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH — skipping verify subprocess test")

    out = await _call(run_verify)
    parsed = json.loads(out)
    assert parsed["exit_code"] == 7
    assert "oops" in parsed["stderr_tail"]
    assert stats["last_verify_exit"] == 7


@pytest.mark.asyncio
async def test_run_verify_missing_script(tmp_path):
    sb = _sandbox(tmp_path)
    (_, _, _, run_verify), _ = _unpack(build_tools(sb))
    out = await _call(run_verify)
    assert "verify.sh not found" in out


def _unpack(built):
    """build_tools returns (list, stats); split the list into the 4 tools."""
    tools, stats = built
    assert len(tools) == 4
    return tuple(tools), stats
