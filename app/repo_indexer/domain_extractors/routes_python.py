"""Sprint 15a — HTTP route extraction for Python sources.

Recognises FastAPI and Flask decorator patterns, emits RouteNode +
RouteEdge records into the IndexBatch.

Called from `extractor_python._emit_function` (in-extractor hook) when
the routes-enabled flag is set on the PhaseContext. Pre-15a the existing
extractor's decorator-unwrap fix gave us access to the decorator nodes;
this module turns them into Route records.

Patterns covered (15a.1):

    FastAPI direct        @app.get("/users")
    FastAPI router        @router.post("/items/{id}")
    Flask                 @app.route("/users", methods=["GET", "POST"])
                          @app.route("/users")  (defaults to GET)

NOT covered (15a.2 follow-ups):
    - Django URL config (path / re_path) — different shape (function call,
      not decorator), requires a separate visitor
    - APIRouter prefix composition (`app.include_router(r, prefix="/api/v1")`)
    - Path concatenation (`app.get(BASE_PATH + "/users")` keeps raw token)
    - NestJS-style class controllers (TS path)
"""
from __future__ import annotations

from typing import Any

from ..actions import RouteEdge, RouteNode
from .routes_common import HTTP_METHODS, normalize_path


# FastAPI / Flask decorator-receiver names we trust. Pre-14g this was
# the only signal; with 14g we COULD additionally check that the
# receiver's typeBinding is a FastAPI/APIRouter/Flask instance — but
# the literal-name check covers all real-world code we've seen and
# avoids false negatives when the user names their app `myapp`.
PYTHON_ROUTE_RECEIVERS = frozenset({
    "app", "router", "blueprint", "bp", "myapp", "api",
    # Plus a few common FastAPI patterns
    "fastapi_app", "router_v1", "router_v2",
})


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _string_literal_text(source: bytes, string_node: Any) -> str | None:
    """Pull the inner content of a Python `string` AST node, or None
    if the node has interpolation / concatenation we can't unquote."""
    if string_node.type != "string":
        return None
    parts: list[str] = []
    for c in string_node.children:
        if c.type == "string_start" or c.type == "string_end":
            continue
        if c.type == "string_content":
            parts.append(_node_text(source, c))
        elif c.type == "interpolation":
            # f-string with embedded code — we can't recover a literal.
            return None
        else:
            # Escape sequences etc.
            parts.append(_node_text(source, c))
    return "".join(parts)


def _list_string_literals(source: bytes, list_node: Any) -> list[str]:
    """Extract string literals from a list AST node. Used for Flask's
    `methods=["GET", "POST"]` kwarg. Non-string elements are dropped."""
    out: list[str] = []
    if list_node.type != "list":
        return out
    for c in list_node.children:
        if c.type != "string":
            continue
        text = _string_literal_text(source, c)
        if text is not None:
            out.append(text)
    return out


def extract_routes_from_decorators(
    source: bytes,
    decorator_nodes: list[Any],
    fn_qn: str,
    fn_file_path: str,
    repo_name: str,
    tenant_id: str,
) -> list[tuple[RouteNode, RouteEdge]]:
    """Return [(RouteNode, RouteEdge)] for every recognised route
    decorator on a function.

    `decorator_nodes` is the list of `decorator` AST children of a
    `decorated_definition` wrapper. Each one we recognise produces one
    or more (route, edge) pairs (Flask's multi-method form fans out).

    `fn_qn` is the wrapped function's qualified name — already computed
    by the caller. `fn_file_path` is the source file (repo-relative
    posix). `repo_name` and `tenant_id` come from the IndexBatch.

    Unrecognised decorators (`@property`, `@dataclass`, etc.) return
    nothing — caller should iterate ALL decorators and accept zero
    matches as "no routes emitted".
    """
    out: list[tuple[RouteNode, RouteEdge]] = []
    for dec in decorator_nodes:
        results = _route_from_decorator(
            source, dec, fn_qn, fn_file_path, repo_name, tenant_id,
        )
        out.extend(results)
    return out


def _route_from_decorator(
    source: bytes,
    decorator_node: Any,
    fn_qn: str,
    fn_file_path: str,
    repo_name: str,
    tenant_id: str,
) -> list[tuple[RouteNode, RouteEdge]]:
    """Process a single decorator. Decorators have the shape
    `@<receiver>.<verb>(<path_string>, ...)`. We accept the call
    when:
        - receiver is in PYTHON_ROUTE_RECEIVERS, AND
        - the called attribute is in HTTP_METHODS or `route`
    """
    # decorator AST: `decorator > "@" + call`. Find the inner call.
    call_node = None
    for c in decorator_node.children:
        if c.type == "call":
            call_node = c
            break
    if call_node is None:
        return []

    fn_node = call_node.child_by_field_name("function")
    args_node = call_node.child_by_field_name("arguments")
    if fn_node is None or fn_node.type != "attribute" or args_node is None:
        return []

    obj = fn_node.child_by_field_name("object")
    attr = fn_node.child_by_field_name("attribute")
    if obj is None or obj.type != "identifier":
        return []
    if attr is None or attr.type != "identifier":
        return []
    receiver = _node_text(source, obj)
    verb = _node_text(source, attr)

    if receiver not in PYTHON_ROUTE_RECEIVERS:
        return []

    line_start = decorator_node.start_point[0] + 1

    if verb in HTTP_METHODS:
        # `@app.get("/users")` — single-method decorator.
        path_raw = _first_string_arg(source, args_node)
        if path_raw is None:
            return []
        return [_make_route(
            repo_name, tenant_id, path_raw, verb.upper(), "fastapi",
            fn_qn, fn_file_path, line_start,
        )]

    if verb == "route":
        # `@app.route("/users", methods=["GET", "POST"])` — Flask form.
        path_raw = _first_string_arg(source, args_node)
        if path_raw is None:
            return []
        methods = _flask_methods_kwarg(source, args_node)
        if not methods:
            methods = ["GET"]   # Flask default
        results: list[tuple[RouteNode, RouteEdge]] = []
        for method in methods:
            results.append(_make_route(
                repo_name, tenant_id, path_raw, method.upper(), "flask",
                fn_qn, fn_file_path, line_start,
            ))
        return results

    return []


def _first_string_arg(source: bytes, args_node: Any) -> str | None:
    """Return the first POSITIONAL string-literal arg from an
    `argument_list`. Returns None for non-literal first args (variables,
    f-strings with interpolations, concatenations) — caller should
    treat as "unrecognised path; skip"."""
    for c in args_node.children:
        if c.type == "(" or c.type == ")" or c.type == ",":
            continue
        if c.type == "keyword_argument":
            continue
        if c.type == "string":
            return _string_literal_text(source, c)
        # First positional that isn't a string — give up.
        return None
    return None


def _flask_methods_kwarg(source: bytes, args_node: Any) -> list[str]:
    """Extract `methods=["GET", "POST"]` from a Flask route call.
    Returns empty list when the kwarg is absent or non-literal."""
    for c in args_node.children:
        if c.type != "keyword_argument":
            continue
        name = c.child_by_field_name("name")
        value = c.child_by_field_name("value")
        if name is None or value is None:
            continue
        if _node_text(source, name) != "methods":
            continue
        if value.type != "list":
            return []
        return _list_string_literals(source, value)
    return []


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
    """Build a (RouteNode, RouteEdge) pair from extracted fields."""
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


__all__ = ["extract_routes_from_decorators"]
