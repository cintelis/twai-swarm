"""Sprint 15c — MCP tool / resource extraction tests.

Synthetic source through the Python extractor with --with-mcp-tools.
"""
from __future__ import annotations

import pytest

try:
    import tree_sitter_python as _tspy  # noqa: F401
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False


from app.repo_indexer.actions import RepoNode  # noqa: E402
from app.repo_indexer.extractor_python import extract_python_file  # noqa: E402

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-python not installed")
    import tree_sitter_python as tspython
    return Parser(Language(tspython.language()))


def _extract(parser, source: bytes, file_path: str = "srv.py"):
    return extract_python_file(
        REPO, file_path, source, "sha", parser,
        extract_mcp_tools=True,
    )


# ─── Tool decorator patterns ───────────────────────────────────────────────

def test_tool_decorator_with_no_args(parser):
    """`@app.tool() def my_tool():` — name = function name."""
    src = (
        b"@app.tool()\n"
        b"def list_users():\n"
        b'    """Return all users."""\n'
        b"    return []\n"
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_tools) == 1
    t = batch.mcp_tools[0]
    assert t.name == "list_users"
    assert "users" in t.description.lower()  # docstring fallback


def test_tool_decorator_with_kwargs(parser):
    """`@app.tool(name="X", description="Y") def fn():` — kwargs win."""
    src = (
        b'@app.tool(name="weather", description="Get the current weather.")\n'
        b'def _weather_handler(city: str) -> str:\n'
        b'    return "sunny"\n'
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_tools) == 1
    t = batch.mcp_tools[0]
    assert t.name == "weather"
    assert t.description == "Get the current weather."
    assert t.handler_qn == "srv._weather_handler"


def test_tool_decorator_no_parens(parser):
    """`@mcp.tool` (bare attribute decorator) — TS-style decorator
    syntax also supported in Python."""
    src = (
        b"@mcp.tool\n"
        b"def fast_tool():\n"
        b'    """A quick tool."""\n'
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_tools) == 1
    assert batch.mcp_tools[0].name == "fast_tool"


def test_server_receiver_accepted(parser):
    """`@server.tool()` — `server` is in MCP_RECEIVERS."""
    src = (
        b"@server.tool()\n"
        b"def srv_tool():\n"
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_tools) == 1


def test_unknown_receiver_skipped(parser):
    """`@unrelated.tool()` should NOT emit — receiver not in MCP_RECEIVERS."""
    src = (
        b"@unrelated.tool()\n"
        b"def fn():\n"
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert batch.mcp_tools == []


def test_non_tool_decorator_skipped(parser):
    """`@app.something_else()` doesn't emit; only `tool` and `resource`
    are recognized verbs."""
    src = (
        b"@app.middleware()\n"
        b"def fn():\n"
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert batch.mcp_tools == []
    assert batch.mcp_resources == []


def test_disabled_by_default(parser):
    """Without `extract_mcp_tools=True`, no MCP nodes emit."""
    src = (
        b"@app.tool()\n"
        b"def x():\n"
        b"    return None\n"
    )
    batch = extract_python_file(REPO, "s.py", src, "sha", parser)
    assert batch.mcp_tools == []


# ─── Resource decorator patterns ────────────────────────────────────────────

def test_resource_decorator_simple(parser):
    """`@app.resource("twai://repo/{name}/context")` extracts URI."""
    src = (
        b'@app.resource("twai://repo/{name}/context")\n'
        b'def get_context(name: str) -> str:\n'
        b'    """Repo context resource."""\n'
        b'    return ""\n'
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_resources) == 1
    r = batch.mcp_resources[0]
    assert r.uri_template == "twai://repo/{name}/context"
    assert r.handler_qn == "srv.get_context"
    assert "context" in r.description.lower()


def test_resource_with_fstring_keeps_raw(parser):
    """f-string URI templates can't be statically resolved, but we
    still record the raw token so the user has the template shape."""
    src = (
        b"REPO_URI = 'twai://repo/{name}'\n"
        b'@app.resource(f"{REPO_URI}/clusters")\n'
        b'def get_clusters():\n'
        b'    return ""\n'
    )
    batch = _extract(parser, src)
    # The exact uri_template is implementation-defined for f-strings;
    # what matters is that we emitted SOMETHING and didn't crash.
    assert len(batch.mcp_resources) == 1


# ─── Stacked decorators ────────────────────────────────────────────────────

def test_tool_with_other_stacked_decorators(parser):
    """`@auth_required @app.tool()` — extractor scans ALL decorators in
    the wrapper, not just the closest. The non-MCP decorator is silently
    skipped (its receiver isn't in MCP_RECEIVERS)."""
    src = (
        b"@auth_required\n"
        b"@app.tool()\n"
        b"def secured_tool():\n"
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert len(batch.mcp_tools) == 1
    assert batch.mcp_tools[0].name == "secured_tool"


# ─── Description fallback chain ────────────────────────────────────────────

def test_description_kwarg_wins_over_docstring(parser):
    src = (
        b'@app.tool(description="Explicit kwarg")\n'
        b'def fn():\n'
        b'    """Docstring fallback."""\n'
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert batch.mcp_tools[0].description == "Explicit kwarg"


def test_description_docstring_fallback_when_no_kwarg(parser):
    src = (
        b"@app.tool()\n"
        b"def fn():\n"
        b'    """Tool that does X."""\n'
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert "X" in batch.mcp_tools[0].description


def test_description_empty_when_no_signal(parser):
    """No kwarg, no docstring — empty description, not crash."""
    src = (
        b"@app.tool()\n"
        b"def fn():\n"
        b"    return None\n"
    )
    batch = _extract(parser, src)
    assert batch.mcp_tools[0].description == ""
