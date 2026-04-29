"""Sprint 15a.2 — HTTP route extraction for TypeScript / JavaScript.

TS frameworks mostly use call patterns rather than decorators:

    app.get("/users", listUsers)         // Express, Hono
    app.post("/login", loginHandler)
    app.get("/users", c => c.json([]))   // inline Hono

Plus filesystem-based for Next.js App Router:

    app/users/route.ts:
        export async function GET(req) { ... }
        export async function POST(req) { ... }

NestJS uses class decorators (`@Controller("/api")` + `@Get(":id")`)
which we defer to 15a.3 — different AST shape, needs class-decorator
unwrap that the existing TS extractor doesn't yet do.

The big trap: `axios.get("/users", opts)` and `httpService.get(...)`
match Express's pattern at the AST level. The HTTP_CLIENT_RECEIVERS
blocklist (shared with Python) screens them out. GitNexus's
parse-worker.ts:874 ships exactly this list; we mirror it.
"""
from __future__ import annotations

import re
from typing import Any

from ..actions import RouteEdge, RouteNode
from .routes_common import HTTP_CLIENT_RECEIVERS, HTTP_METHODS, normalize_path


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _string_literal_text(source: bytes, string_node: Any) -> str | None:
    """Pull the inner text of a TS `string` AST node. Returns None for
    template strings with `${...}` interpolation (we can't recover a
    literal there). Plain backtick template strings without interpolation
    DO unwrap cleanly."""
    if string_node.type == "string":
        for c in string_node.children:
            if c.type == "string_fragment":
                return _node_text(source, c)
        return ""
    if string_node.type == "template_string":
        # No interpolations means it's a constant; with any
        # `template_substitution` we can't recover a static path.
        for c in string_node.children:
            if c.type == "template_substitution":
                return None
        # All children are string fragments / backticks; concat fragments.
        parts: list[str] = []
        for c in string_node.children:
            if c.type == "string_fragment":
                parts.append(_node_text(source, c))
        return "".join(parts)
    return None


def _flatten_member_chain(source: bytes, n: Any) -> str | None:
    """Flatten `a.b.c` → "a.b.c". Returns None for non-flattenable
    receivers (call returns, this, etc.)."""
    if n.type == "identifier":
        return _node_text(source, n)
    if n.type == "member_expression":
        obj = n.child_by_field_name("object")
        prop = n.child_by_field_name("property")
        if obj is None or prop is None:
            return None
        base = _flatten_member_chain(source, obj)
        if base is None:
            return None
        return f"{base}.{_node_text(source, prop)}"
    return None


def extract_routes_from_call(
    source: bytes,
    call_node: Any,
    enclosing_fn_qn: str,
    file_path: str,
    repo_name: str,
    tenant_id: str,
) -> list[tuple[RouteNode, RouteEdge]]:
    """Try to interpret `call_node` as `<receiver>.<verb>(path, handler)`.

    Returns [] when the pattern doesn't match or the receiver is in
    HTTP_CLIENT_RECEIVERS (axios/fetch/etc.).

    `enclosing_fn_qn` is the function/method that CONTAINS the call —
    used as the handler-qn fallback when the second arg is an inline
    arrow function with no name we can resolve.
    """
    fn_node = call_node.child_by_field_name("function")
    args_node = call_node.child_by_field_name("arguments")
    if fn_node is None or args_node is None:
        return []
    if fn_node.type != "member_expression":
        return []

    obj = fn_node.child_by_field_name("object")
    prop = fn_node.child_by_field_name("property")
    if obj is None or prop is None:
        return []
    if prop.type != "property_identifier":
        return []

    verb = _node_text(source, prop).lower()
    if verb not in HTTP_METHODS:
        return []

    # Receiver text — flatten dotted forms (`router.api.get(...)`) but
    # the BLOCKLIST check happens on the LAST segment because that's
    # what carries the framework identity (`api.client.get` → "client").
    receiver_chain = _flatten_member_chain(source, obj)
    if receiver_chain is None:
        return []
    receiver_last = receiver_chain.split(".")[-1].lower()
    if receiver_last in HTTP_CLIENT_RECEIVERS:
        return []

    # First positional arg = path. Second positional = handler (if a
    # name we can resolve). Walk arguments by named-child position
    # because tree-sitter exposes `,` as anonymous siblings.
    positional: list[Any] = []
    for c in args_node.children:
        if c.type in ("(", ")", ","):
            continue
        positional.append(c)
    if not positional:
        return []
    path_node = positional[0]
    path_raw = _string_literal_text(source, path_node)
    if path_raw is None:
        return []

    handler_qn = ""
    if len(positional) >= 2:
        h = positional[-1]   # last arg is conventionally the handler
        if h.type == "identifier":
            # Named handler — resolver patches qn at finalize time. For
            # 15a.2 we store the bare name; the loader's MATCH on
            # qualified_name will fail for unqualified names but the
            # RouteNode itself still emits.
            handler_qn = ""  # leave empty; named-handler resolution is 15a.3
        # Inline arrow / function expression — use enclosing fn as a hint
        # only if the arg shape is non-resolvable.

    line_start = call_node.start_point[0] + 1
    framework = _detect_framework(receiver_chain)
    return [_make_route(
        repo_name, tenant_id, path_raw, verb.upper(), framework,
        handler_qn, file_path, line_start,
    )]


def _detect_framework(receiver_chain: str) -> str:
    """Best-effort framework attribution from the receiver name. We
    can't tell Express from Hono syntactically; both produce
    `framework="express"` for now. NestJS and Next.js have their own
    patterns and won't reach this function."""
    return "express"


# Next.js App Router: `app/<path-segments>/route.ts` files where
# segments like `[id]` become `:id` (or `{id}`) in the route. Each
# exported verb function (GET / POST / etc.) is a separate route.

_NEXTJS_ROUTE_FILENAME_RE = re.compile(r"(?:^|/)route\.tsx?$")
_NEXTJS_ROUTE_DIR_RE = re.compile(r"^app/(.*)/route\.tsx?$")


def is_nextjs_route_file(rel_path: str) -> bool:
    """True when `rel_path` is a Next.js App Router route handler
    (e.g. `app/users/route.ts`, `app/api/v1/items/[id]/route.ts`)."""
    if not _NEXTJS_ROUTE_FILENAME_RE.search(rel_path):
        return False
    # Must be under `app/` (App Router) — the `pages/` router is
    # different and doesn't use `route.ts`.
    return rel_path.startswith("app/") or "/app/" in rel_path


def _nextjs_path_from_file(rel_path: str) -> str:
    """Convert `app/api/users/[id]/route.ts` → `/api/users/{id}`.
    Strips the `app/` prefix and the trailing `route.ts(x?)`. Brackets
    map to FastAPI-style param syntax for consistency with the
    Python-side stored shape."""
    # Find `app/` and start after it.
    m = re.search(r"(?:^|/)app/(.*)/route\.tsx?$", rel_path)
    if m is None:
        return "/"
    inner = m.group(1)
    # Convert `[id]` and `[...slug]` (catch-all) to `{id}` / `{*slug}`.
    inner = re.sub(r"\[\.\.\.([^\]]+)\]", r"{*\1}", inner)
    inner = re.sub(r"\[([^\]]+)\]", r"{\1}", inner)
    return "/" + inner


def extract_routes_nextjs_app_router(
    source: bytes,
    root_node: Any,
    rel_path: str,
    repo_name: str,
    tenant_id: str,
    file_module_qn: str,
) -> list[tuple[RouteNode, RouteEdge]]:
    """For an App Router `route.ts` file, walk top-level exported
    function declarations whose name is an HTTP verb (GET/POST/etc.)
    and emit one Route per match. Path comes from filename, NOT source."""
    if not is_nextjs_route_file(rel_path):
        return []

    path = _nextjs_path_from_file(rel_path)
    out: list[tuple[RouteNode, RouteEdge]] = []
    for child in root_node.children:
        # Look through `export_statement` wrappers.
        target = child
        if target.type == "export_statement":
            for sub in target.children:
                if sub.type in ("function_declaration", "lexical_declaration"):
                    target = sub
                    break
        if target.type == "function_declaration":
            name_node = target.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(source, name_node)
            if name.upper() not in {m.upper() for m in HTTP_METHODS}:
                continue
            handler_qn = f"{file_module_qn}.{name}" if file_module_qn else name
            line_start = target.start_point[0] + 1
            out.append(_make_route(
                repo_name, tenant_id, path, name.upper(), "nextjs",
                handler_qn, rel_path, line_start,
            ))
    return out


def _make_route(
    repo_name: str,
    tenant_id: str,
    raw_path: str,
    method: str,
    framework: str,
    handler_qn: str,
    file_path: str,
    line_start: int,
) -> tuple[RouteNode, RouteEdge]:
    path = normalize_path(raw_path)
    return (
        RouteNode(
            repo=repo_name,
            tenant_id=tenant_id,
            path=path,
            method=method,
            framework=framework,
            handler_qn=handler_qn,
            file_path=file_path,
            line_start=line_start,
            raw_path=raw_path,
        ),
        RouteEdge(
            repo=repo_name,
            tenant_id=tenant_id,
            path=path,
            method=method,
            handler_qn=handler_qn,
        ),
    )


__all__ = [
    "extract_routes_from_call",
    "extract_routes_nextjs_app_router",
    "is_nextjs_route_file",
]
