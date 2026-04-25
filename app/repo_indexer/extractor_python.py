"""Tree-sitter Python extractor.

Walks the AST of one Python file and emits an IndexBatch fragment.

Resolution model:
    - Anything DEFINED in this file (functions, classes, methods) becomes
      a FunctionNode/ClassNode keyed on `<module>.<name>`.
    - Calls and inheritance to names DEFINED in this same file resolve
      directly to those nodes' qualified names.
    - Calls/inheritance to anything else (stdlib, third-party, anything
      from another file in the repo) become SymbolNodes with the
      best-effort dotted name. Sprint 10b will add cross-file resolution
      using the import map; today's pass is intentionally single-file
      so we get a working graph end-to-end first.

Why tree-sitter and not Python's `ast` module:
    - Same library handles TypeScript / JS / Go / Rust later — one AST
      framework instead of one per language.
    - Resilient to syntax errors (recovers and keeps producing nodes),
      so a broken file in the repo doesn't kill the whole scan.
"""
from __future__ import annotations

from typing import Any

from .actions import (
    CallEdge,
    ClassNode,
    FileNode,
    FunctionNode,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    ModuleNode,
    RepoNode,
    SymbolNode,
)


def _module_qn_from_path(rel_path: str) -> str:
    """`app/repo_indexer/walker.py` → `app.repo_indexer.walker`. `__init__.py`
    files map to their containing package."""
    parts = rel_path.removesuffix(".py").split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _docstring(source: bytes, body_node: Any) -> str:
    """First string-literal child of a body, if any. Stripped to first line."""
    if body_node is None:
        return ""
    for child in body_node.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type == "string":
                    txt = _node_text(source, sub).strip("\"' \n")
                    return txt.split("\n", 1)[0][:200]
        # Only the very first statement counts as a docstring.
        if child.type not in ("comment", "newline"):
            break
    return ""


def _function_params(source: bytes, params_node: Any) -> tuple[str, ...]:
    if params_node is None:
        return ()
    out: list[str] = []
    for child in params_node.children:
        if child.type in ("identifier", "typed_parameter", "default_parameter",
                          "typed_default_parameter", "list_splat_pattern",
                          "dictionary_splat_pattern"):
            # Pull just the name token out — drop type annotations + defaults.
            name_node = child if child.type == "identifier" else child.child_by_field_name("name")
            if name_node is None:
                # Fallback: first identifier child.
                for c in child.children:
                    if c.type == "identifier":
                        name_node = c
                        break
            if name_node is not None:
                out.append(_node_text(source, name_node))
    return tuple(out)


def _walk_calls(source: bytes, body_node: Any) -> list[tuple[str, int]]:
    """Return [(callee_dotted_name, line)] for every call expression in body.

    Best-effort: handles `foo()`, `obj.method()`, `pkg.mod.func()`. Skips
    calls where the function expression isn't a name/attribute (e.g.
    `(lambda: 1)()` — those are noise).
    """
    found: list[tuple[str, int]] = []

    def _flatten_attribute(n: Any) -> str | None:
        """`a.b.c` → "a.b.c"; returns None for anything not chainable."""
        if n.type == "identifier":
            return _node_text(source, n)
        if n.type == "attribute":
            obj = n.child_by_field_name("object")
            attr = n.child_by_field_name("attribute")
            base = _flatten_attribute(obj) if obj is not None else None
            if base is None or attr is None:
                return None
            return f"{base}.{_node_text(source, attr)}"
        return None

    def _visit(n: Any) -> None:
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is not None:
                dotted = _flatten_attribute(fn)
                if dotted:
                    # tree-sitter Point uses 0-indexed rows; humans count from 1.
                    found.append((dotted, n.start_point[0] + 1))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _walk_imports(source: bytes, root: Any) -> list[str]:
    """Return [imported_module_qn] from `import x` and `from x import y` lines."""
    out: list[str] = []
    for child in root.children:
        if child.type == "import_statement":
            # `import a, b.c` — children are dotted_name siblings.
            for sub in child.children:
                if sub.type == "dotted_name":
                    out.append(_node_text(source, sub))
                elif sub.type == "aliased_import":
                    name = sub.child_by_field_name("name")
                    if name is not None:
                        out.append(_node_text(source, name))
        elif child.type == "import_from_statement":
            mod = child.child_by_field_name("module_name")
            if mod is not None:
                out.append(_node_text(source, mod))
    return out


def extract_python_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
) -> IndexBatch:
    """Parse one .py file and return its IndexBatch fragment."""
    batch = IndexBatch(repo=repo)

    module_qn = _module_qn_from_path(rel_path)
    batch.files.append(FileNode(repo=repo.name, path=rel_path, language="python", sha=sha))
    if module_qn:
        batch.modules.append(ModuleNode(repo=repo.name, qualified_name=module_qn, file_path=rel_path))

    tree = parser.parse(source)
    root = tree.root_node

    # Track names defined in this file so we can resolve same-file calls
    # without going through SymbolNode.
    local_names: set[str] = set()

    def _is_async_function(node: Any) -> bool:
        """tree-sitter-python represents `async def` as a function_definition
        with an `async` keyword child rather than a distinct node type."""
        return any(c.type == "async" for c in node.children)

    # First pass: defined classes + functions (top-level + methods).
    def _emit_function(node: Any, parent_class_qn: str = "") -> None:
        is_async = _is_async_function(node)
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        params = node.child_by_field_name("parameters")
        if name_node is None:
            return
        name = _node_text(source, name_node)
        is_method = bool(parent_class_qn)
        qn = f"{parent_class_qn}.{name}" if is_method else (
            f"{module_qn}.{name}" if module_qn else name
        )
        local_names.add(name)
        batch.functions.append(FunctionNode(
            repo=repo.name,
            qualified_name=qn,
            name=name,
            file_path=rel_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            is_async=is_async,
            is_method=is_method,
            parent_class_qn=parent_class_qn,
            params=_function_params(source, params),
            docstring=_docstring(source, body),
        ))
        # Calls inside this function become CallEdges (resolved or symbol).
        for callee_dotted, line in _walk_calls(source, body):
            head = callee_dotted.split(".", 1)[0]
            if head in local_names and "." not in callee_dotted:
                # Same-file call to a top-level def — resolve directly.
                callee_qn = f"{module_qn}.{callee_dotted}" if module_qn else callee_dotted
            else:
                # External or cross-file — record as a symbol; cross-file
                # resolution happens in Sprint 10b via the import map.
                callee_qn = callee_dotted
                batch.symbols.append(SymbolNode(
                    repo=repo.name,
                    qualified_name=callee_dotted,
                    name=callee_dotted.rsplit(".", 1)[-1],
                ))
            batch.calls.append(CallEdge(
                repo=repo.name,
                caller_qn=qn,
                callee_qn=callee_qn,
                line=line,
            ))

    def _emit_class(node: Any) -> None:
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node is None:
            return
        name = _node_text(source, name_node)
        qn = f"{module_qn}.{name}" if module_qn else name
        local_names.add(name)
        batch.classes.append(ClassNode(
            repo=repo.name,
            qualified_name=qn,
            name=name,
            file_path=rel_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=_docstring(source, body),
        ))
        # Inheritance — superclasses are in the argument_list child.
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for sub in superclasses.children:
                if sub.type in ("identifier", "attribute"):
                    parent_dotted = _node_text(source, sub)
                    parent_qn = parent_dotted  # always recorded as-seen; resolution in 10b
                    batch.symbols.append(SymbolNode(
                        repo=repo.name,
                        qualified_name=parent_dotted,
                        name=parent_dotted.rsplit(".", 1)[-1],
                    ))
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name, child_qn=qn, parent_qn=parent_qn,
                    ))
        # Methods.
        if body is not None:
            for child in body.children:
                if child.type == "function_definition":
                    _emit_function(child, parent_class_qn=qn)

    for child in root.children:
        if child.type == "function_definition":
            _emit_function(child)
        elif child.type == "class_definition":
            _emit_class(child)

    # Imports — file-level edges.
    for target in _walk_imports(source, root):
        batch.imports.append(ImportEdge(
            repo=repo.name, file_path=rel_path, target_qn=target,
        ))

    return batch
