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
    LocalVarBinding,
    ModuleNode,
    RepoNode,
)


def _module_qn_from_path(rel_path: str, package_roots: tuple = ()) -> str:
    """`app/repo_indexer/walker.py` → `app.repo_indexer.walker`. `__init__.py`
    files map to their containing package.

    `package_roots` is a tuple of `PackageRoot` from `package_roots.py`.
    When provided, files under a declared package root are addressed
    relative to that root (Sprint 14e — fixes monorepo qn construction).
    Default `()` falls back to the dotted repo-relative path so existing
    callers and tests keep working unchanged.
    """
    if package_roots:
        from .package_roots import module_qn_for
        return module_qn_for(rel_path, list(package_roots))
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


def _function_params(source: bytes, params_node: Any) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return (param_names, param_types).

    param_types is a tuple of (name, type_text) pairs — only includes params
    that have type annotations. Used by the resolver to turn `param.method()`
    calls into Function edges instead of Symbol edges.
    """
    if params_node is None:
        return (), ()
    names: list[str] = []
    types: list[tuple[str, str]] = []
    for child in params_node.children:
        if child.type not in ("identifier", "typed_parameter", "default_parameter",
                              "typed_default_parameter", "list_splat_pattern",
                              "dictionary_splat_pattern"):
            continue

        # Param name — children layout differs across param kinds.
        name_node = child if child.type == "identifier" else child.child_by_field_name("name")
        if name_node is None:
            for c in child.children:
                if c.type == "identifier":
                    name_node = c
                    break
        if name_node is None:
            continue
        param_name = _node_text(source, name_node)
        names.append(param_name)

        # Type annotation — only present on typed_parameter / typed_default_parameter.
        # tree-sitter exposes it as a `type` field child.
        type_node = child.child_by_field_name("type")
        if type_node is not None:
            type_text = _node_text(source, type_node).strip()
            if type_text:
                types.append((param_name, type_text))

    return tuple(names), tuple(types)


def _walk_calls(source: bytes, body_node: Any) -> list[tuple[str, int]]:
    """Return [(callee_dotted_name, line)] for every call expression in body.

    Best-effort: handles `foo()`, `obj.method()`, `pkg.mod.func()`. Skips
    calls where the function expression isn't a name/attribute (e.g.
    `(lambda: 1)()` — those are noise).
    """
    found: list[tuple[str, int]] = []

    def _flatten_attribute(n: Any) -> str | None:
        """`a.b.c` → "a.b.c"; returns None for anything not chainable.

        Special case for `super()`: tree-sitter parses the receiver of
        `super().method` as a `call` node whose function is the identifier
        `super`. We collapse that to the bare token `super` so the head of
        the dotted name matches what `scope_resolution.finalize._resolve_callee`
        looks for in its super() resolution branch. Other call-typed receivers
        (e.g. `foo().bar()`) stay None — chasing them through call returns
        needs flow / return-type inference and is deferred.
        """
        if n.type == "identifier":
            return _node_text(source, n)
        if n.type == "attribute":
            obj = n.child_by_field_name("object")
            attr = n.child_by_field_name("attribute")
            base = _flatten_attribute(obj) if obj is not None else None
            if base is None or attr is None:
                return None
            return f"{base}.{_node_text(source, attr)}"
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "identifier" and _node_text(source, fn) == "super":
                return "super"
        return None

    def _visit(n: Any) -> None:
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is not None:
                dotted = _flatten_attribute(fn)
                # Skip bare `super()` — the visitor recurses into every call
                # node, so for `super().method()` we'd otherwise emit BOTH
                # `super.method` (from the outer call) and `super` (from the
                # inner super() call). The bare-super edge has no useful
                # semantics; finalize.py's super branch only fires on
                # dotted shapes.
                if dotted and dotted != "super":
                    # tree-sitter Point uses 0-indexed rows; humans count from 1.
                    found.append((dotted, n.start_point[0] + 1))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _walk_assignments(source: bytes, body_node: Any) -> list[tuple[str, str, int]]:
    """Sprint 14g — return [(var_name, type_raw_name, line)] for every
    `var = SomeClass(...)` constructor assignment in `body_node`.

    Case 7 (simple typeBinding) only: bare-name LHS, bare-identifier
    callee. Skips:
      - Multi-target (`x, y = ...`)
      - Augmented (`x += ...`)
      - Dotted callee (`x = models.User()`) — case 5; deferred
      - Function-call returns (`x = func()`) — return-type tracking; deferred
      - Method calls (`x = obj.method()`) — compound chain; deferred to 14g.2

    The returned tuples feed `LocalVarBinding` records on the IndexBatch;
    the resolver's `LocalVarTypeIndex` consumes them.
    """
    found: list[tuple[str, str, int]] = []

    def _visit(n: Any) -> None:
        # Walk INTO function and class bodies so nested scopes also get
        # their bindings extracted. The CALLER (`_emit_function`) limits
        # the body it passes us to a single function's body, so we don't
        # cross function boundaries here.
        if n.type == "assignment":
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if (left is not None and left.type == "identifier"
                    and right is not None and right.type == "call"):
                callee = right.child_by_field_name("function")
                if callee is not None and callee.type == "identifier":
                    var_name = _node_text(source, left)
                    type_raw_name = _node_text(source, callee)
                    line = n.start_point[0] + 1
                    found.append((var_name, type_raw_name, line))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _walk_module_level_assignments(
    source: bytes, root: Any,
) -> list[tuple[str, str, int]]:
    """Sprint 14i — return [(var_name, type_raw_name, line)] for every
    `var = SomeClass(...)` assignment at the MODULE level only.

    Mirrors `_walk_assignments` shape but does NOT recurse into function
    or class bodies — those are handled separately. Only direct children
    of the module root are inspected (typically `expression_statement`
    nodes containing an `assignment`).

    Catches the canonical FastAPI/Flask/Django pattern:

        # at module top
        app = FastAPI()
        db = SQLAlchemy(app)

    These bindings are stored on the file's Module scope so any function
    or class in the same file finds them via the scope-chain walk.
    """
    found: list[tuple[str, str, int]] = []
    for child in root.children:
        # Module-level assignments are wrapped in `expression_statement`
        # in tree-sitter-python's grammar.
        if child.type != "expression_statement":
            continue
        for sub in child.children:
            if sub.type != "assignment":
                continue
            left = sub.child_by_field_name("left")
            right = sub.child_by_field_name("right")
            if (left is not None and left.type == "identifier"
                    and right is not None and right.type == "call"):
                callee = right.child_by_field_name("function")
                if callee is not None and callee.type == "identifier":
                    var_name = _node_text(source, left)
                    type_raw_name = _node_text(source, callee)
                    line = sub.start_point[0] + 1
                    found.append((var_name, type_raw_name, line))
    return found


def _walk_self_field_assignments(
    source: bytes, body_node: Any,
) -> list[tuple[str, str, int]]:
    """Sprint 14g.2 — return [(field_name, type_raw_name, line)] for every
    `self.<field> = SomeClass(...)` assignment in `body_node`.

    Caller emits these as LocalVarBindings ON THE CLASS SCOPE (not the
    method's function scope), so methods later doing `self.<field>.method()`
    find the binding via the scope-chain walk from method scope up to
    class scope. Mirrors GitNexus's class-field typeBindings (stored
    on the class scope's `typeBindings` map).

    Same constructor-only restriction as `_walk_assignments`. Pattern:

        self.x = SomeClass(...)         ✓ emitted
        self.x = obj.method()           ✗ deferred (return-type tracking)
        self.x = models.User(...)       ✗ deferred (case 5 — namespace)
        self.x: SomeClass = ...         ✗ deferred (annotated; needs separate visitor)

    `var_name` is the field name only ("x"), NOT "self.x" — so the
    receiver-resolution branch can do `find(class_scope, "x", tree)` to
    look up `self.x`'s type from a method.
    """
    found: list[tuple[str, str, int]] = []

    def _visit(n: Any) -> None:
        if n.type == "assignment":
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if (left is not None and left.type == "attribute"
                    and right is not None and right.type == "call"):
                obj = left.child_by_field_name("object")
                attr = left.child_by_field_name("attribute")
                callee = right.child_by_field_name("function")
                if (obj is not None and obj.type == "identifier"
                        and _node_text(source, obj) == "self"
                        and attr is not None and attr.type == "identifier"
                        and callee is not None and callee.type == "identifier"):
                    field_name = _node_text(source, attr)
                    type_raw_name = _node_text(source, callee)
                    line = n.start_point[0] + 1
                    found.append((field_name, type_raw_name, line))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _walk_imports(source: bytes, root: Any) -> list[tuple[str, str, str]]:
    """Return [(target_qn, local_name, kind)] for every import in the file.

    kind is "module" for `import a` / `import a.b` (binds a module/package
    in this file's namespace) or "symbol" for `from x import y` (binds a
    single symbol — function/class/constant — that lives inside module x).

    Local-name normalisation:
      `import a.b`              -> ("a.b", "a", "module")
      `import a.b as foo`       -> ("a.b", "foo", "module")
      `from x import y`         -> ("x.y", "y", "symbol")
      `from x import y as foo`  -> ("x.y", "foo", "symbol")
      `from x import y, z`      -> emits two entries
      `from x import *`         -> ("x", "*", "module") — finalize.py's
                                    _is_wildcard branch picks this up
                                    and unions x's exports into the file's
                                    binding scope.
    """
    out: list[tuple[str, str, str]] = []
    for child in root.children:
        if child.type == "import_statement":
            # `import a, b.c, d as foo` — siblings are dotted_name | aliased_import.
            for sub in child.children:
                if sub.type == "dotted_name":
                    target = _node_text(source, sub)
                    # Local binding for `import a.b` is `a` (the package root),
                    # not `a.b` — that's how Python attribute access works.
                    local = target.split(".", 1)[0]
                    out.append((target, local, "module"))
                elif sub.type == "aliased_import":
                    name = sub.child_by_field_name("name")
                    alias = sub.child_by_field_name("alias")
                    if name is not None and alias is not None:
                        out.append((_node_text(source, name), _node_text(source, alias), "module"))
        elif child.type == "import_from_statement":
            mod = child.child_by_field_name("module_name")
            if mod is None:
                continue
            mod_qn = _node_text(source, mod)
            # Imported names are dotted_name / aliased_import siblings AFTER
            # the literal `import` keyword token. Tracking the keyword is more
            # robust than `is`-comparing against `mod`, which can fail when
            # tree-sitter returns new Node wrappers per attribute access.
            seen_import_kw = False
            for sub in child.children:
                if sub.type == "import":
                    seen_import_kw = True
                    continue
                if not seen_import_kw:
                    continue
                if sub.type == "dotted_name":
                    name = _node_text(source, sub)
                    out.append((f"{mod_qn}.{name}", name, "symbol"))
                elif sub.type == "aliased_import":
                    name_node = sub.child_by_field_name("name")
                    alias_node = sub.child_by_field_name("alias")
                    if name_node is not None and alias_node is not None:
                        name = _node_text(source, name_node)
                        alias = _node_text(source, alias_node)
                        out.append((f"{mod_qn}.{name}", alias, "symbol"))
                elif sub.type == "wildcard_import":
                    # `from x import *`: target_qn is the module itself,
                    # local_name="*" matches finalize._is_wildcard's gate.
                    out.append((mod_qn, "*", "module"))
    return out


def extract_python_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
    package_roots: tuple = (),
) -> IndexBatch:
    """Parse one .py file and return its IndexBatch fragment.

    `package_roots` (Sprint 14e) corrects module qns on monorepos. See
    `_module_qn_from_path`.
    """
    batch = IndexBatch(repo=repo)

    module_qn = _module_qn_from_path(rel_path, package_roots)
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
    def _emit_function(
        node: Any,
        parent_class_qn: str = "",
        parent_class_line_start: int = 0,
        parent_class_line_end: int = 0,
    ) -> None:
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
        param_names, param_types = _function_params(source, params)
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
            params=param_names,
            param_types=param_types,
            docstring=_docstring(source, body),
        ))
        # Calls inside this function become CallEdges. Same-file local calls
        # resolve here; everything else is left as the raw dotted name and
        # the resolver decides post-pass whether it lands on a Function
        # (cross-file) or a Symbol (truly external).
        for callee_dotted, line in _walk_calls(source, body):
            head = callee_dotted.split(".", 1)[0]
            if head in local_names and "." not in callee_dotted:
                callee_qn = f"{module_qn}.{callee_dotted}" if module_qn else callee_dotted
            else:
                # Defer — resolver will rewrite or wrap in Symbol.
                callee_qn = callee_dotted
            batch.calls.append(CallEdge(
                repo=repo.name,
                caller_qn=qn,
                callee_qn=callee_qn,
                line=line,
            ))

        # Sprint 14g — local var typeBindings. Constructor-style assignments
        # `x = SomeClass(...)` become LocalVarBinding records the resolver
        # uses to dispatch `x.method(...)` through MethodDispatchIndex.
        for var_name, type_raw_name, assign_line in _walk_assignments(source, body):
            batch.local_var_bindings.append(LocalVarBinding(
                repo=repo.name,
                tenant_id=repo.tenant_id,
                file_path=rel_path,
                enclosing_scope_kind="function",
                enclosing_line_start=node.start_point[0] + 1,
                enclosing_line_end=node.end_point[0] + 1,
                var_name=var_name,
                type_raw_name=type_raw_name,
                line=assign_line,
            ))

        # Sprint 14g.2 — class-field typeBindings. `self.x = SomeClass(...)`
        # in __init__ (or anywhere in a method) gets stored on the
        # ENCLOSING CLASS scope, not this function's scope. Lookups from
        # other methods walk parent_of(method_scope) → class_scope and
        # find the field there; that mirrors GitNexus's class-field
        # storage (typeBindings on the class scope itself).
        if is_method and parent_class_line_start > 0:
            for field_name, type_raw_name, assign_line in _walk_self_field_assignments(source, body):
                batch.local_var_bindings.append(LocalVarBinding(
                    repo=repo.name,
                    tenant_id=repo.tenant_id,
                    file_path=rel_path,
                    enclosing_scope_kind="class",
                    enclosing_line_start=parent_class_line_start,
                    enclosing_line_end=parent_class_line_end,
                    var_name=field_name,
                    type_raw_name=type_raw_name,
                    line=assign_line,
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
        # Inheritance — superclasses are in the argument_list child. Recorded
        # as observed; resolver decides post-pass whether the parent maps
        # to an in-repo Class or stays as an external Symbol.
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for sub in superclasses.children:
                if sub.type in ("identifier", "attribute"):
                    parent_dotted = _node_text(source, sub)
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name, child_qn=qn, parent_qn=parent_dotted,
                    ))
        # Methods.
        class_line_start = node.start_point[0] + 1
        class_line_end = node.end_point[0] + 1
        if body is not None:
            for child in body.children:
                if child.type == "function_definition":
                    _emit_function(
                        child,
                        parent_class_qn=qn,
                        parent_class_line_start=class_line_start,
                        parent_class_line_end=class_line_end,
                    )

    for child in root.children:
        if child.type == "function_definition":
            _emit_function(child)
        elif child.type == "class_definition":
            _emit_class(child)

    # Sprint 14i — module-level typeBindings. `app = FastAPI()` at the
    # top of the file becomes a binding on the module scope so any
    # function in this file (or class methods) can resolve `app.get(...)`
    # via the scope-chain walk. Sentinel encoding: enclosing_line_end =
    # MODULE_SCOPE_END - 1 because the adapter's `_range_for` adds 1 for
    # half-open semantics; we want the resulting Range to match
    # `to_scopes`'s module ScopeId range exactly.
    if module_qn:
        from .scope_resolution._adapter import MODULE_SCOPE_END
        for var_name, type_raw_name, assign_line in _walk_module_level_assignments(source, root):
            batch.local_var_bindings.append(LocalVarBinding(
                repo=repo.name,
                tenant_id=repo.tenant_id,
                file_path=rel_path,
                enclosing_scope_kind="module",
                enclosing_line_start=0,
                enclosing_line_end=MODULE_SCOPE_END - 1,
                var_name=var_name,
                type_raw_name=type_raw_name,
                line=assign_line,
            ))

    # Imports — file-level edges.
    for target_qn, local_name, kind in _walk_imports(source, root):
        batch.imports.append(ImportEdge(
            repo=repo.name,
            file_path=rel_path,
            target_qn=target_qn,
            local_name=local_name,
            kind=kind,
        ))

    return batch
