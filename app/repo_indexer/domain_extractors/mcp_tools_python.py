"""Sprint 15c — MCP tool/resource detection for Python sources.

Recognises the FastMCP / mcp.server SDK decorators:

    @app.tool(name="X", description="Y") def fn(...): ...
    @app.tool() def fn(...): ...                              # name=fn name
    @mcp.tool                                                  # no parens
    @app.resource("twai://repo/{name}/context") def fn(...): ...

Plus programmatic registration:

    app.add_tool(my_func)

Description fallback order: decorator `description=` kwarg → first
positional string arg (for `tool`) → function docstring → empty.

Reuses 15a.1's decorator-unwrap path — `_emit_function` already feeds
us the decorator subtree so this module just inspects them and emits
MCPToolNode / MCPResourceNode records.
"""
from __future__ import annotations

from typing import Any

from ..actions import MCPResourceNode, MCPToolNode

# Receivers we accept as MCP server objects. Could be tightened via
# 14g typeBindings (`app = FastMCP(...)` produces a binding) — for
# 15c.1 the literal-name set covers all real-world code we've seen.
MCP_RECEIVERS = frozenset({
    "mcp", "app", "server", "fastmcp", "fast_mcp",
})


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _string_literal_text(source: bytes, string_node: Any) -> str | None:
    """Pull inner text of a Python `string` AST node. Returns None for
    f-strings with interpolations (we can't recover a literal)."""
    if string_node.type != "string":
        return None
    parts: list[str] = []
    for c in string_node.children:
        if c.type == "string_start" or c.type == "string_end":
            continue
        if c.type == "interpolation":
            return None
        if c.type == "string_content":
            parts.append(_node_text(source, c))
        else:
            parts.append(_node_text(source, c))
    return "".join(parts)


def _string_or_fstring_raw(source: bytes, string_node: Any) -> str:
    """Get the source token of a string node — INCLUDING f-string
    interpolations. Used for `uri_template` storage where we want the
    raw template back-trip even if we can't fully resolve `${...}`."""
    return _node_text(source, string_node).strip("\"'")


def _first_positional_string(source: bytes, args_node: Any) -> str | None:
    """Return the first positional string-literal arg from `argument_list`."""
    for c in args_node.children:
        if c.type in ("(", ")", ","):
            continue
        if c.type == "keyword_argument":
            continue
        if c.type == "string":
            return _string_literal_text(source, c)
        return None
    return None


def _string_kwarg(source: bytes, args_node: Any, name: str) -> str | None:
    """Return the value of `name=` if present and a string literal.

    Peeks through `parenthesized_expression` wrappers — Python's
    canonical multi-line string idiom is
        description=("first line "
                     "second line")
    which parses as `parenthesized_expression > string`. Without the
    unwrap we'd report no description for every kwarg-on-its-own-line
    decorator — which is exactly how twai-swarm's MCP server is
    written.
    """
    for c in args_node.children:
        if c.type != "keyword_argument":
            continue
        kw_name = c.child_by_field_name("name")
        kw_value = c.child_by_field_name("value")
        if kw_name is None or kw_value is None:
            continue
        if _node_text(source, kw_name) != name:
            continue
        # Unwrap `parenthesized_expression` wrappers — Python's canonical
        # idiom for multi-line strings:
        #     description=("first line "
        #                  "second line")
        # parses as parenthesized_expression > concatenated_string >
        # [string, string]. Handle both layers so kwargs declared this
        # way are extractable.
        v = kw_value
        while v.type == "parenthesized_expression":
            inner = None
            for ch in v.children:
                if ch.type not in ("(", ")"):
                    inner = ch
                    break
            if inner is None:
                return None
            v = inner
        if v.type == "concatenated_string":
            # `"a" "b"` — Python implicit string concatenation.
            parts: list[str] = []
            for ch in v.children:
                if ch.type == "string":
                    p = _string_literal_text(source, ch)
                    if p is None:
                        return None
                    parts.append(p)
            return "".join(parts)
        if v.type != "string":
            return None
        return _string_literal_text(source, v)
    return None


def extract_mcp_from_decorators(
    source: bytes,
    decorator_nodes: list[Any],
    fn_name: str,
    fn_qn: str,
    fn_file_path: str,
    fn_docstring: str,
    repo_name: str,
    tenant_id: str,
) -> tuple[list[MCPToolNode], list[MCPResourceNode]]:
    """Process a function's decorators for `@mcp.tool` / `@mcp.resource`
    patterns. Returns (tool_nodes, resource_nodes) — typically each
    function has at most ONE such decorator.
    """
    tools: list[MCPToolNode] = []
    resources: list[MCPResourceNode] = []
    for dec in decorator_nodes:
        result = _process_one_decorator(
            source, dec, fn_name, fn_qn, fn_file_path, fn_docstring,
            repo_name, tenant_id,
        )
        if result is None:
            continue
        kind, node = result
        if kind == "tool":
            tools.append(node)  # type: ignore[arg-type]
        elif kind == "resource":
            resources.append(node)  # type: ignore[arg-type]
    return tools, resources


def _process_one_decorator(
    source: bytes,
    decorator_node: Any,
    fn_name: str,
    fn_qn: str,
    fn_file_path: str,
    fn_docstring: str,
    repo_name: str,
    tenant_id: str,
) -> tuple[str, MCPToolNode | MCPResourceNode] | None:
    """Process one decorator. Returns ("tool", node) or ("resource", node)
    or None if the decorator isn't an MCP-style one."""
    # Two shapes:
    #   - @mcp.tool                     (decorator > attribute)
    #   - @mcp.tool(...)                (decorator > call > attribute)
    receiver: str | None = None
    verb: str | None = None
    args_node: Any = None
    for child in decorator_node.children:
        if child.type == "attribute":
            # Bare `@mcp.tool` form (no parens)
            obj = child.child_by_field_name("object")
            attr = child.child_by_field_name("attribute")
            if obj is not None and obj.type == "identifier" and attr is not None and attr.type == "identifier":
                receiver = _node_text(source, obj)
                verb = _node_text(source, attr)
            break
        if child.type == "call":
            fn_node = child.child_by_field_name("function")
            args_node = child.child_by_field_name("arguments")
            if fn_node is None or fn_node.type != "attribute":
                return None
            obj = fn_node.child_by_field_name("object")
            attr = fn_node.child_by_field_name("attribute")
            if obj is None or obj.type != "identifier":
                return None
            if attr is None or attr.type != "identifier":
                return None
            receiver = _node_text(source, obj)
            verb = _node_text(source, attr)
            break
    if receiver is None or verb is None:
        return None
    if receiver not in MCP_RECEIVERS:
        return None

    line_start = decorator_node.start_point[0] + 1

    if verb == "tool":
        # Name: kwarg `name=` first; fall back to function name.
        name = None
        description = ""
        if args_node is not None:
            name = _string_kwarg(source, args_node, "name")
            description = _string_kwarg(source, args_node, "description") or ""
            if not description:
                # Positional string for `@app.tool("description-text")`
                positional = _first_positional_string(source, args_node)
                if positional is not None:
                    description = positional
        if not name:
            name = fn_name
        if not description and fn_docstring:
            # Take the first paragraph; cap at 500 chars.
            description = fn_docstring.split("\n\n")[0].strip()[:500]
        return (
            "tool",
            MCPToolNode(
                repo=repo_name,
                tenant_id=tenant_id,
                name=name,
                description=description,
                handler_qn=fn_qn,
                file_path=fn_file_path,
                line_start=line_start,
            ),
        )

    if verb == "resource":
        # Resource requires a positional URI literal.
        if args_node is None:
            return None
        uri_template: str | None = None
        for c in args_node.children:
            if c.type in ("(", ")", ","):
                continue
            if c.type == "keyword_argument":
                continue
            if c.type == "string":
                # Try clean unquote first; if that fails (f-string), fall
                # back to the raw source token so we still record SOMETHING.
                clean = _string_literal_text(source, c)
                if clean is not None:
                    uri_template = clean
                else:
                    uri_template = _string_or_fstring_raw(source, c)
            break
        if uri_template is None:
            return None
        description = ""
        if fn_docstring:
            description = fn_docstring.split("\n\n")[0].strip()[:500]
        return (
            "resource",
            MCPResourceNode(
                repo=repo_name,
                tenant_id=tenant_id,
                uri_template=uri_template,
                description=description,
                handler_qn=fn_qn,
                file_path=fn_file_path,
                line_start=line_start,
            ),
        )

    return None


__all__ = ["extract_mcp_from_decorators"]
