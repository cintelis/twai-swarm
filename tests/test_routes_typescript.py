"""Sprint 15a.2 — HTTP route extraction tests for the TypeScript path.

Covers Express/Hono call-pattern detection and Next.js App Router
file-based detection. NestJS class decorators deferred to 15a.3.
"""
from __future__ import annotations

import pytest

try:
    import tree_sitter_typescript as _tsts  # noqa: F401
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False


from app.repo_indexer.actions import RepoNode  # noqa: E402
from app.repo_indexer.domain_extractors.routes_typescript import (  # noqa: E402
    is_nextjs_route_file,
)
from app.repo_indexer.extractor_typescript import extract_typescript_file  # noqa: E402

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def ts_parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-typescript not installed")
    import tree_sitter_typescript as tsts
    return Parser(Language(tsts.language_typescript()))


# ─── Express / Hono call patterns ──────────────────────────────────────────

def test_express_get_emits_route(ts_parser):
    src = b'app.get("/users", listUsers);\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    r = batch.routes[0]
    assert r.path == "/users"
    assert r.method == "GET"
    assert r.framework == "express"


def test_express_post_with_path_param(ts_parser):
    src = b'app.post("/items/:id", updateItem);\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/items/:id"
    assert batch.routes[0].method == "POST"


def test_router_call_emits_route(ts_parser):
    src = b'router.put("/items/:id", updateItem);\n'
    batch = extract_typescript_file(
        REPO, "router.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    assert batch.routes[0].method == "PUT"


def test_axios_get_is_blocklisted(ts_parser):
    """Critical false-positive guard. `axios.get("/users")` is an HTTP
    CLIENT call, not a server route. The HTTP_CLIENT_RECEIVERS filter
    must reject it."""
    src = b'const data = axios.get("/api/users");\n'
    batch = extract_typescript_file(
        REPO, "client.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert batch.routes == []


def test_fetch_post_is_blocklisted(ts_parser):
    src = b'fetch.post("/api/login", body);\n'
    batch = extract_typescript_file(
        REPO, "client.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert batch.routes == []


def test_template_string_path_static(ts_parser):
    """Template strings without ${} interpolations should resolve."""
    src = b'app.get(`/users`, listUsers);\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/users"


def test_template_string_path_with_interpolation_skipped(ts_parser):
    """Template strings with ${} can't be unquoted to a literal —
    skip rather than fabricate."""
    src = b'app.get(`${PREFIX}/users`, listUsers);\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert batch.routes == []


def test_routes_disabled_by_default(ts_parser):
    src = b'app.get("/users", listUsers);\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        # extract_routes=False is the default
    )
    assert batch.routes == []


def test_unknown_verb_skipped(ts_parser):
    """`app.listen(...)` and `app.use(...)` look like the call pattern
    but aren't HTTP-method names. Should be ignored."""
    src = b'app.listen(3000);\napp.use(cors());\n'
    batch = extract_typescript_file(
        REPO, "server.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert batch.routes == []


# ─── Next.js App Router (file-based) ────────────────────────────────────────

def test_is_nextjs_route_file_detection():
    assert is_nextjs_route_file("app/users/route.ts") is True
    assert is_nextjs_route_file("app/api/v1/items/[id]/route.ts") is True
    assert is_nextjs_route_file("app/users/page.tsx") is False
    assert is_nextjs_route_file("pages/api/users.ts") is False  # Pages Router, not App Router
    assert is_nextjs_route_file("src/utils/route.ts") is False  # not under app/


def test_nextjs_app_router_get_export(ts_parser):
    """`app/users/route.ts` exporting `GET` becomes a Route with path
    derived from the filename."""
    src = b"export async function GET(req) {\n  return [];\n}\n"
    batch = extract_typescript_file(
        REPO, "app/users/route.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    r = batch.routes[0]
    assert r.path == "/users"
    assert r.method == "GET"
    assert r.framework == "nextjs"


def test_nextjs_app_router_multiple_verbs(ts_parser):
    """One file with multiple verb exports → one Route per verb."""
    src = (
        b"export async function GET(req) { return []; }\n"
        b"export async function POST(req) { return {}; }\n"
        b"export async function DELETE(req) { return null; }\n"
    )
    batch = extract_typescript_file(
        REPO, "app/api/items/route.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    methods = sorted(r.method for r in batch.routes)
    assert methods == ["DELETE", "GET", "POST"]
    paths = {r.path for r in batch.routes}
    assert paths == {"/api/items"}


def test_nextjs_dynamic_segment(ts_parser):
    """`[id]` becomes `{id}` in the stored path (FastAPI-style param
    syntax) for consistency with our Python-side route paths."""
    src = b"export async function GET(req) { return null; }\n"
    batch = extract_typescript_file(
        REPO, "app/users/[id]/route.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/users/{id}"


def test_nextjs_catchall_segment(ts_parser):
    """`[...slug]` → `{*slug}` (catch-all syntax)."""
    src = b"export async function GET(req) { return null; }\n"
    batch = extract_typescript_file(
        REPO, "app/docs/[...slug]/route.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert len(batch.routes) == 1
    assert batch.routes[0].path == "/docs/{*slug}"


def test_nextjs_non_verb_export_ignored(ts_parser):
    """A `route.ts` that exports `helperFn` (not a verb) should not
    produce a Route. Only HTTP-verb function names trigger the
    extraction."""
    src = b"export function helperFn(x) { return x; }\n"
    batch = extract_typescript_file(
        REPO, "app/users/route.ts", src, "sha", ts_parser, repo_files=set(),
        extract_routes=True,
    )
    assert batch.routes == []
