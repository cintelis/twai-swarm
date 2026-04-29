"""Sprint 15a — HTTP route extraction tests for the Python path.

Two layers:
  1. Direct tests of `extract_routes_from_decorators` against synthetic
     tree-sitter ASTs (no extractor wrapping, no IndexBatch).
  2. End-to-end through `extract_python_file(...,extract_routes=True)`
     with `--with-routes`-equivalent behaviour.
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
from app.repo_indexer.domain_extractors.routes_common import normalize_path  # noqa: E402
from app.repo_indexer.extractor_python import extract_python_file  # noqa: E402

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-python not installed")
    import tree_sitter_python as tspython
    return Parser(Language(tspython.language()))


# ─── Path normalisation ────────────────────────────────────────────────────

def test_normalize_path_basics():
    assert normalize_path("/users") == "/users"
    assert normalize_path("/users/") == "/users"
    assert normalize_path("/Users") == "/users"  # case-folded
    assert normalize_path("users") == "/users"   # leading slash added
    assert normalize_path("") == "/"
    assert normalize_path("/") == "/"


def test_normalize_path_quoted_input_is_stripped():
    """Caller may pass `"/users"` literally; strip the surrounding quotes."""
    assert normalize_path('"/users"') == "/users"
    assert normalize_path("'/users'") == "/users"


def test_normalize_path_collapses_double_slashes():
    assert normalize_path("/api//users") == "/api/users"


def test_normalize_path_preserves_param_syntax():
    assert normalize_path("/users/{id}") == "/users/{id}"
    assert normalize_path("/items/:id") == "/items/:id"


# ─── End-to-end extraction (decorator unwrap + route emission) ──────────────

def test_decorator_unwrap_emits_function(parser):
    """Pre-15a regression: decorated top-level functions were silently
    dropped. Verify they now get FunctionNodes even without --with-routes."""
    src = b'@app.get("/users")\ndef list_users():\n    return []\n'
    batch = extract_python_file(REPO, "api.py", src, "sha", parser)
    fn_names = {f.name for f in batch.functions}
    assert "list_users" in fn_names


def test_decorator_unwrap_emits_method(parser):
    """Same fix applies inside class bodies — `@property` etc. methods
    were silently dropped pre-fix."""
    src = (
        b"class S:\n"
        b"    @property\n"
        b"    def x(self):\n"
        b"        return 1\n"
    )
    batch = extract_python_file(REPO, "s.py", src, "sha", parser)
    fn_names = {f.name for f in batch.functions}
    assert "x" in fn_names


def test_routes_disabled_by_default(parser):
    """--with-routes off ⇒ no RouteNodes emitted even when decorators
    look like routes."""
    src = b'@app.get("/users")\ndef list_users():\n    return []\n'
    batch = extract_python_file(REPO, "api.py", src, "sha", parser)
    assert batch.routes == []
    assert batch.route_edges == []


def test_fastapi_get_emits_route(parser):
    src = b'@app.get("/users")\ndef list_users():\n    return []\n'
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert len(batch.routes) == 1
    r = batch.routes[0]
    assert r.path == "/users"
    assert r.method == "GET"
    assert r.framework == "fastapi"
    assert r.handler_qn == "api.list_users"
    assert r.raw_path == "/users"
    assert len(batch.route_edges) == 1
    e = batch.route_edges[0]
    assert e.path == "/users" and e.method == "GET" and e.handler_qn == "api.list_users"


def test_fastapi_router_post_with_path_param(parser):
    src = (
        b'@router.post("/items/{id}")\n'
        b"def update_item(id: str):\n"
        b"    return {}\n"
    )
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/items/{id}"
    assert batch.routes[0].method == "POST"


def test_flask_route_default_method(parser):
    src = b'@app.route("/users")\ndef list_users():\n    return []\n'
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert len(batch.routes) == 1
    assert batch.routes[0].framework == "flask"
    assert batch.routes[0].method == "GET"  # Flask default


def test_flask_route_with_methods_kwarg_fans_out(parser):
    """`@app.route("/x", methods=["GET", "POST"])` → two RouteNodes."""
    src = (
        b'@app.route("/users", methods=["GET", "POST"])\n'
        b"def list_or_create():\n"
        b"    return []\n"
    )
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    methods = sorted(r.method for r in batch.routes)
    assert methods == ["GET", "POST"]
    # Both edges point at the same handler.
    handler_qns = {e.handler_qn for e in batch.route_edges}
    assert handler_qns == {"api.list_or_create"}


def test_non_route_decorator_emits_nothing(parser):
    """`@property` / `@dataclass` / etc. don't produce RouteNodes."""
    src = (
        b"class S:\n"
        b"    @property\n"
        b"    def x(self):\n"
        b"        return 1\n"
    )
    batch = extract_python_file(REPO, "s.py", src, "sha", parser, extract_routes=True)
    assert batch.routes == []


def test_unknown_receiver_emits_nothing(parser):
    """A receiver name not in PYTHON_ROUTE_RECEIVERS — silently skipped.
    This is the primary defence against `axios.get("/users")` and
    similar HTTP-CLIENT calls."""
    src = b'@axios.get("/external/endpoint")\ndef _wrapped(): return None\n'
    batch = extract_python_file(REPO, "x.py", src, "sha", parser, extract_routes=True)
    assert batch.routes == []


def test_path_concatenation_is_skipped(parser):
    """`@app.get(BASE_PATH + "/users")` — first arg isn't a literal
    string, so we don't fabricate a path. The function still emits as
    a FunctionNode (decorator unwrap), just no Route."""
    src = (
        b'@app.get(BASE_PATH + "/users")\n'
        b"def list_users():\n"
        b"    return []\n"
    )
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert batch.routes == []
    fn_names = {f.name for f in batch.functions}
    assert "list_users" in fn_names


def test_two_decorators_one_route(parser):
    """Stacked decorators: only the route one should emit a Route. The
    other (auth_required, etc.) is harmless — its receiver isn't in our
    set, so the route extractor ignores it."""
    src = (
        b"@auth_required\n"
        b'@app.post("/login")\n'
        b"def login():\n"
        b"    return {}\n"
    )
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/login"
    assert batch.routes[0].method == "POST"


def test_method_decorated_route(parser):
    """A class method decorated with `@app.get(...)` — same treatment as
    free functions, but handler_qn includes the class."""
    src = (
        b"class API:\n"
        b'    @app.get("/users")\n'
        b"    def list_users(self):\n"
        b"        return []\n"
    )
    batch = extract_python_file(REPO, "api.py", src, "sha", parser, extract_routes=True)
    assert len(batch.routes) == 1
    assert batch.routes[0].handler_qn == "api.API.list_users"
