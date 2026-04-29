"""Tree-sitter TypeScript / TSX / JavaScript extractor.

Mirrors extractor_python.py but for the TS family. Same IndexBatch shape
emitted, so the resolver and loader are language-agnostic.

Module-name convention:
    File path with the source extension stripped, slashes -> dots.
    `app/api/items/route.ts`     -> `app.api.items.route`
    `lib/prisma.ts`              -> `lib.prisma`
    `lib/index.ts`               -> `lib`         (JS convention — index files map to package)

Cross-file imports are resolved against `repo_files` (a set of relative
posix paths in the repo), not against tsconfig.json `paths` aliases. So:

    `import { X } from "./bar"`     -> resolved to a sibling .ts/.tsx file if present
    `import { X } from "react"`     -> external; target_qn stays as observed
    `import { X } from "@/lib/foo"` -> external (alias paths skipped in 10d v1)

What we emit per file:
    FileNode + ModuleNode + ClassNodes + FunctionNodes
    + ImportEdges (with target_qn already path-resolved when possible)
    + CallEdges (raw dotted callee names — resolver rewrites them)
    + InheritsEdges
    + (no Symbols — resolver decides)
"""
from __future__ import annotations

from typing import Any, Iterable

from .actions import (
    CallEdge,
    ClassNode,
    FileNode,
    FunctionNode,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    Language,
    LocalVarBinding,
    ModuleNode,
    RepoNode,
)

# Source-file extensions the resolver can map back to a module QN.
_SRC_EXTS = (".ts", ".tsx", ".js", ".jsx")


def module_qn_from_path(rel_path: str) -> str:
    """`app/api/items/route.ts` -> `app.api.items.route`. `lib/index.ts`
    collapses to `lib` (JS convention — index files are the package root)."""
    base = rel_path
    for ext in _SRC_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("/")
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _string_fragment(source: bytes, string_node: Any) -> str:
    """Pull the inner text of a `string` node (strips quotes via the
    string_fragment child). Empty string if shape is unexpected."""
    for c in string_node.children:
        if c.type == "string_fragment":
            return _node_text(source, c)
    return ""


def _resolve_relative_import(
    importing_file: str,
    specifier: str,
    repo_files: set[str],
) -> str | None:
    """Resolve `./bar` / `../shared/foo` against the importing file's dir,
    against the actual file set in the repo. Returns the resolved repo-
    relative path (with extension), or None if nothing matches.

    Strips an explicit `.ts` / `.tsx` / `.js` / `.jsx` suffix on the
    specifier before extension probing — TypeScript's `"moduleResolution":
    "node16"` and ESM-on-Node both REQUIRE explicit `.js` in imports
    even when the source is `.ts`, so `from '../foo.js'` must resolve to
    `../foo.ts` on disk. Without this strip we'd probe `foo.js.ts`,
    `foo.js.tsx`, etc. and miss the actual file.

    Tries: <stem>.ts, <stem>.tsx, <stem>.js, <stem>.jsx, <stem>/index.ts,
    <stem>/index.tsx, <stem>/index.js, <stem>/index.jsx — in that order.
    """
    if not specifier.startswith("."):
        return None
    # Compute the importing file's parent directory.
    parts = importing_file.split("/")
    parent = parts[:-1]
    # Apply the specifier path components.
    for seg in specifier.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parent:
                parent.pop()
            continue
        parent.append(seg)
    base = "/".join(parent)
    # Strip a source extension if the specifier already has one. TS-ESM
    # requires `.js` in imports even when the source is `.ts`; without
    # this strip we'd append another extension and never find the file.
    for ext in _SRC_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    # Try direct file extensions first, then index variants.
    for ext in _SRC_EXTS:
        candidate = base + ext
        if candidate in repo_files:
            return candidate
    for ext in _SRC_EXTS:
        candidate = f"{base}/index{ext}"
        if candidate in repo_files:
            return candidate
    return None


def _walk_imports(
    source: bytes,
    root: Any,
    rel_path: str,
    repo_files: set[str],
) -> list[tuple[str, str, str]]:
    """Return [(target_qn, local_name, kind)] for every import in the file.

    target_qn is the resolved module/symbol QN when the import is a
    relative path that resolves to an in-repo file. For bare specifiers
    (`react`, `node:path`) and unresolved relative paths we keep the
    original specifier so the resolver can still emit a Symbol for them.
    """
    out: list[tuple[str, str, str]] = []
    for child in root.children:
        if child.type != "import_statement":
            continue
        # Find the string specifier child.
        spec = ""
        for c in child.children:
            if c.type == "string":
                spec = _string_fragment(source, c)
                break
        if not spec:
            continue

        # Resolve the spec to a module QN (if relative + in-repo) OR keep
        # as-observed (external/alias). Resolved imports get the symbol QN
        # appended later (named imports).
        resolved_path = _resolve_relative_import(rel_path, spec, repo_files)
        target_module_qn = module_qn_from_path(resolved_path) if resolved_path else spec

        # Walk the import_clause children to figure out what's imported.
        clause = next((c for c in child.children if c.type == "import_clause"), None)
        if clause is None:
            # `import "./foo"` (side-effect only) — record the module link.
            out.append((target_module_qn, "", "module"))
            continue

        # Default import: `import foo from "..."` — clause has identifier child.
        for c in clause.children:
            if c.type == "identifier":
                out.append((target_module_qn, _node_text(source, c), "module"))

        # Namespace import: `import * as foo from "..."`.
        for c in clause.children:
            if c.type == "namespace_import":
                for sub in c.children:
                    if sub.type == "identifier":
                        out.append((target_module_qn, _node_text(source, sub), "module"))

        # Named imports: `import { Foo, Bar as Baz } from "..."`.
        for c in clause.children:
            if c.type != "named_imports":
                continue
            for spec_node in c.children:
                if spec_node.type != "import_specifier":
                    continue
                # name field = original; alias field = local binding (if `as` present).
                name_node = spec_node.child_by_field_name("name")
                alias_node = spec_node.child_by_field_name("alias")
                if name_node is None:
                    continue
                name = _node_text(source, name_node)
                local = _node_text(source, alias_node) if alias_node is not None else name
                # symbol-kind import: target_qn drills into the module.
                target_symbol_qn = (
                    f"{target_module_qn}.{name}" if target_module_qn else name
                )
                out.append((target_symbol_qn, local, "symbol"))
    return out


def _formal_params(source: bytes, params_node: Any) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Extract param names + their type annotations from a formal_parameters node.

    Returns (names, types). types is a tuple of (param_name, type_text) pairs;
    only includes params whose declaration has a `type_annotation` child.
    """
    if params_node is None:
        return (), ()
    names: list[str] = []
    types: list[tuple[str, str]] = []
    for child in params_node.children:
        if child.type not in ("required_parameter", "optional_parameter"):
            continue
        # Param name lives in the `pattern` field for both parameter kinds.
        pat = child.child_by_field_name("pattern")
        if pat is None:
            continue
        if pat.type == "identifier":
            param_name = _node_text(source, pat)
        else:
            # Destructuring patterns ({a, b}, [x, y], etc.) — skip; they
            # don't introduce a single bindable name we can match calls against.
            continue
        names.append(param_name)
        type_node = child.child_by_field_name("type")
        if type_node is not None:
            type_text = _node_text(source, type_node).lstrip(": ").strip()
            if type_text:
                types.append((param_name, type_text))
    return tuple(names), tuple(types)


def _walk_calls(source: bytes, body_node: Any) -> list[tuple[str, int]]:
    """Best-effort: every `call_expression` whose function is an identifier
    or member_expression chain. Returns [(dotted_name, line_1_indexed)]."""
    found: list[tuple[str, int]] = []

    def _flatten(n: Any) -> str | None:
        if n.type == "identifier":
            return _node_text(source, n)
        if n.type == "member_expression":
            obj = n.child_by_field_name("object")
            prop = n.child_by_field_name("property")
            base = _flatten(obj) if obj is not None else None
            if base is None or prop is None:
                return None
            return f"{base}.{_node_text(source, prop)}"
        return None

    def _visit(n: Any) -> None:
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                dotted = _flatten(fn)
                if dotted:
                    found.append((dotted, n.start_point[0] + 1))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _normalize_ts_type(text: str) -> str:
    """Sprint 14j — normalize a TS type annotation.

    Strips:
        Promise<X>      → X
        Array<X>        → X
        ReadonlyArray<X> → X
        X | null        → X
        X | undefined   → X
        readonly X[]    → X
        X[]             → X

    Multi-arg generics (`Map<K, V>`, `Record<K, V>`) keep the wrapper
    name as a heuristic — `Map` / `Record` won't resolve as classes
    so the resolver falls through cleanly.
    """
    text = text.strip()
    if not text:
        return ""
    # Trailing `[]` array suffix.
    while text.endswith("[]"):
        text = text[:-2].strip()
    # `readonly X[]` prefix (the [] was stripped above; strip the keyword).
    if text.startswith("readonly "):
        text = text[len("readonly "):].strip()
    # `X | null` / `X | undefined` / `null | X` / `undefined | X`.
    if " | " in text:
        parts = [p.strip() for p in text.split(" | ")]
        non_nullish = [p for p in parts if p not in ("null", "undefined")]
        if len(non_nullish) == 1:
            text = non_nullish[0]
    # Generics `Promise<X>`, `Array<X>`, etc.
    if "<" in text and text.endswith(">"):
        bracket = text.index("<")
        wrapper = text[:bracket].strip()
        inner = text[bracket + 1:-1].strip()
        if wrapper in {"Promise", "Array", "ReadonlyArray", "Iterable",
                       "AsyncIterable", "Awaitable", "Observable"}:
            # Single-arg unwrap. Multi-arg stays as wrapper name.
            if "," not in inner:
                return _normalize_ts_type(inner)
    return text


def _ts_type_annotation_text(source: bytes, type_annotation_node: Any) -> str:
    """Extract the type expression from a `type_annotation` node and
    normalize it. The `type_annotation` wraps the actual type child."""
    if type_annotation_node is None:
        return ""
    # The first non-`:` named child is the type expression.
    for child in type_annotation_node.children:
        if child.type in ("type_identifier", "predefined_type", "generic_type",
                          "union_type", "array_type", "readonly_type",
                          "literal_type", "object_type", "tuple_type",
                          "function_type", "constructor_type", "intersection_type"):
            return _normalize_ts_type(_node_text(source, child))
    return ""


def _walk_ts_assignments(source: bytes, body_node: Any) -> list[tuple[str, str, int]]:
    """Sprint 14j — return [(var_name, type_raw_name, line)] for every
    `const/let/var x = new Foo()` or `const x: Foo = …` in `body_node`.

    Annotation > inference precedence (per GitNexus's `interpret.ts`
    ordering): if both `type:` and `value: (new_expression)` are present,
    the annotation wins.

    Skips:
      - Destructuring (`const {a, b} = …`, `const [a] = …`)
      - Untyped non-constructor RHS (`const x = computed`)
    """
    found: list[tuple[str, str, int]] = []

    def _visit(n: Any) -> None:
        if n.type in ("lexical_declaration", "variable_declaration"):
            for declarator in n.children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                if name_node is None or name_node.type != "identifier":
                    continue
                # Annotation precedence first.
                type_annotation = declarator.child_by_field_name("type")
                if type_annotation is not None:
                    type_raw = _ts_type_annotation_text(source, type_annotation)
                    if type_raw:
                        found.append((
                            _node_text(source, name_node),
                            type_raw,
                            n.start_point[0] + 1,
                        ))
                        continue
                # Constructor inference: `new Foo()`.
                value_node = declarator.child_by_field_name("value")
                if value_node is not None and value_node.type == "new_expression":
                    ctor = value_node.child_by_field_name("constructor")
                    if ctor is not None and ctor.type == "identifier":
                        found.append((
                            _node_text(source, name_node),
                            _node_text(source, ctor),
                            n.start_point[0] + 1,
                        ))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _walk_ts_class_fields(source: bytes, class_body: Any) -> list[tuple[str, str, int]]:
    """Sprint 14j — return [(field_name, type_raw_name, line)] for every
    annotated/initialized class field declaration.

    TS shapes covered:
        client: Client                       — annotation only
        client: Client = new Client()        — annotation wins over init
        client = new Client()                — initializer-typed (no annotation)

    Tree-sitter-typescript represents class fields as `public_field_definition`
    nodes inside the class body.
    """
    found: list[tuple[str, str, int]] = []
    if class_body is None:
        return found
    for child in class_body.children:
        if child.type != "public_field_definition":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        if name_node.type not in ("property_identifier", "private_property_identifier"):
            continue
        field_name = _node_text(source, name_node)
        # Annotation first.
        type_annotation = child.child_by_field_name("type")
        if type_annotation is not None:
            type_raw = _ts_type_annotation_text(source, type_annotation)
            if type_raw:
                found.append((field_name, type_raw, child.start_point[0] + 1))
                continue
        # Initializer-only: `client = new Client()`.
        value_node = child.child_by_field_name("value")
        if value_node is not None and value_node.type == "new_expression":
            ctor = value_node.child_by_field_name("constructor")
            if ctor is not None and ctor.type == "identifier":
                found.append((
                    field_name,
                    _node_text(source, ctor),
                    child.start_point[0] + 1,
                ))
    return found


def _walk_ts_this_assignments(source: bytes, body_node: Any) -> list[tuple[str, str, int]]:
    """Sprint 14j — return [(field_name, type_raw_name, line)] for every
    `this.<field> = new <Class>()` assignment in `body_node` (typically
    a constructor body).

    Mirrors Python's `_walk_self_field_assignments`. The bound name is
    the FIELD name only — NOT "this.field" — so the resolver's
    `find(class_scope, "field", tree)` lookup works for `this.field.x()`
    chains via the same scope-chain walk.
    """
    found: list[tuple[str, str, int]] = []

    def _visit(n: Any) -> None:
        if n.type == "assignment_expression":
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if (left is not None and left.type == "member_expression"
                    and right is not None and right.type == "new_expression"):
                obj = left.child_by_field_name("object")
                attr = left.child_by_field_name("property")
                ctor = right.child_by_field_name("constructor")
                if (obj is not None and obj.type == "this"
                        and attr is not None
                        and attr.type == "property_identifier"
                        and ctor is not None and ctor.type == "identifier"):
                    found.append((
                        _node_text(source, attr),
                        _node_text(source, ctor),
                        n.start_point[0] + 1,
                    ))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _is_async_function(node: Any) -> bool:
    return any(c.type == "async" for c in node.children)


def extract_typescript_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
    repo_files: set[str],
    language: Language = "typescript",
    extract_routes: bool = False,
) -> IndexBatch:
    """Parse one .ts/.tsx/.js/.jsx file and return its IndexBatch fragment.

    `extract_routes` (Sprint 15a.2) opts in to HTTP route extraction —
    Express/Hono `app.get("/x", handler)` calls and Next.js App Router
    `app/.../route.ts` exported verbs become RouteNodes. Default off.
    """
    batch = IndexBatch(repo=repo)

    module_qn = module_qn_from_path(rel_path)
    batch.files.append(FileNode(repo=repo.name, path=rel_path, language=language, sha=sha))
    if module_qn:
        batch.modules.append(ModuleNode(repo=repo.name, qualified_name=module_qn, file_path=rel_path))

    tree = parser.parse(source)
    root = tree.root_node

    # Track names defined at the top level — same trick as the Python pass.
    local_names: set[str] = set()

    def _emit_function(
        node: Any, name: str,
        parent_class_qn: str = "",
        body_node: Any = None,
        parent_class_line_start: int = 0,
        parent_class_line_end: int = 0,
    ) -> None:
        is_async = _is_async_function(node)
        is_method = bool(parent_class_qn)
        if is_method:
            qn = f"{parent_class_qn}.{name}"
        else:
            qn = f"{module_qn}.{name}" if module_qn else name
            local_names.add(name)
        params = node.child_by_field_name("parameters")
        body = body_node if body_node is not None else node.child_by_field_name("body")
        param_names, param_types = _formal_params(source, params)
        # Sprint 14j — return type annotation on `function name(): ReturnType`
        # or `name(): ReturnType { … }`. Tree-sitter-typescript exposes
        # the annotation under field name `return_type` (a `type_annotation`
        # node), same as Python.
        return_type_raw = ""
        return_type_node = node.child_by_field_name("return_type")
        if return_type_node is not None:
            return_type_raw = _ts_type_annotation_text(source, return_type_node)
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
            return_type_raw=return_type_raw,
        ))
        for callee_dotted, line in _walk_calls(source, body):
            head = callee_dotted.split(".", 1)[0]
            if head in local_names and "." not in callee_dotted:
                callee_qn = f"{module_qn}.{callee_dotted}" if module_qn else callee_dotted
            else:
                callee_qn = callee_dotted
            batch.calls.append(CallEdge(
                repo=repo.name, caller_qn=qn, callee_qn=callee_qn, line=line,
            ))

        # Sprint 14j — local var typeBindings inside the function body.
        # Constructor inference + annotation precedence handled by
        # _walk_ts_assignments.
        for var_name, type_raw_name, assign_line in _walk_ts_assignments(source, body):
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

        # Sprint 14j — `this.x = new Y()` in a method body. Stored on
        # the CLASS scope (not the method's function scope) — mirrors
        # Python's `self.x = X()` handling. Hoists so other methods on
        # the same class can resolve `this.x.method()` chains.
        if is_method and parent_class_line_start > 0:
            for field_name, type_raw_name, assign_line in _walk_ts_this_assignments(source, body):
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

        # Sprint 14j — return-type binding on the function's enclosing
        # scope. Methods → class scope; free functions → module scope.
        # Auto-hoist semantics, mirrors Python's 14h.
        if return_type_raw:
            if is_method and parent_class_line_start > 0:
                batch.local_var_bindings.append(LocalVarBinding(
                    repo=repo.name,
                    tenant_id=repo.tenant_id,
                    file_path=rel_path,
                    enclosing_scope_kind="class",
                    enclosing_line_start=parent_class_line_start,
                    enclosing_line_end=parent_class_line_end,
                    var_name=name,
                    type_raw_name=return_type_raw,
                    line=node.start_point[0] + 1,
                ))
            elif not is_method and module_qn:
                from .scope_resolution._adapter import MODULE_SCOPE_END
                batch.local_var_bindings.append(LocalVarBinding(
                    repo=repo.name,
                    tenant_id=repo.tenant_id,
                    file_path=rel_path,
                    enclosing_scope_kind="module",
                    enclosing_line_start=0,
                    enclosing_line_end=MODULE_SCOPE_END - 1,
                    var_name=name,
                    type_raw_name=return_type_raw,
                    line=node.start_point[0] + 1,
                ))

    def _emit_arrow_const(declarator: Any) -> None:
        """`const foo = (...) => {...}` becomes a top-level Function."""
        name_node = declarator.child_by_field_name("name")
        value_node = declarator.child_by_field_name("value")
        if name_node is None or value_node is None or name_node.type != "identifier":
            return
        if value_node.type != "arrow_function":
            return
        name = _node_text(source, name_node)
        # arrow_function carries its own parameters/body fields.
        _emit_function(value_node, name, body_node=value_node.child_by_field_name("body"))

    def _emit_class(node: Any) -> None:
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node is None:
            return
        name = _node_text(source, name_node)
        qn = f"{module_qn}.{name}" if module_qn else name
        local_names.add(name)
        class_line_start = node.start_point[0] + 1
        class_line_end = node.end_point[0] + 1
        batch.classes.append(ClassNode(
            repo=repo.name,
            qualified_name=qn,
            name=name,
            file_path=rel_path,
            line_start=class_line_start,
            line_end=class_line_end,
        ))
        # Heritage — `extends Foo` or `extends Foo<T>` (we ignore generics).
        heritage = node.child_by_field_name("heritage") or next(
            (c for c in node.children if c.type == "class_heritage"), None
        )
        if heritage is not None:
            for sub in heritage.children:
                if sub.type == "extends_clause":
                    for inner in sub.children:
                        if inner.type in ("identifier", "type_identifier", "member_expression"):
                            parent_dotted = _node_text(source, inner)
                            batch.inherits.append(InheritsEdge(
                                repo=repo.name, child_qn=qn, parent_qn=parent_dotted,
                            ))
        # Sprint 14j — class field bindings (annotation or constructor-
        # initialized). Stored on the class scope so methods later doing
        # `this.field.method()` find them via the scope-chain walk.
        for field_name, type_raw_name, field_line in _walk_ts_class_fields(source, body):
            batch.local_var_bindings.append(LocalVarBinding(
                repo=repo.name,
                tenant_id=repo.tenant_id,
                file_path=rel_path,
                enclosing_scope_kind="class",
                enclosing_line_start=class_line_start,
                enclosing_line_end=class_line_end,
                var_name=field_name,
                type_raw_name=type_raw_name,
                line=field_line,
            ))
        # Methods. Pass the class line range so `this.x = new Y()` in
        # the constructor body (or any method) gets emitted on the class
        # scope, not the method's function scope.
        if body is not None:
            for child in body.children:
                if child.type == "method_definition":
                    name_field = child.child_by_field_name("name")
                    if name_field is not None:
                        method_name = _node_text(source, name_field)
                        _emit_function(
                            child, method_name,
                            parent_class_qn=qn,
                            parent_class_line_start=class_line_start,
                            parent_class_line_end=class_line_end,
                        )

    def _walk_top_level(nodes: Iterable[Any]) -> None:
        for child in nodes:
            # Look through `export_statement` wrappers for the actual decl.
            if child.type == "export_statement":
                _walk_top_level(child.children)
                continue
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    _emit_function(child, _node_text(source, name_node))
            elif child.type == "class_declaration":
                _emit_class(child)
            elif child.type == "lexical_declaration":
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        _emit_arrow_const(sub)

    _walk_top_level(root.children)

    # Sprint 15a.2 — HTTP routes. Two paths run when --with-routes is on:
    #   1. Walk every CallExpression in the file looking for
    #      Express/Hono `app.get("/x", handler)` patterns
    #   2. If this file is an App Router `route.ts`, parse the path from
    #      its filename and emit one Route per exported verb function
    if extract_routes:
        from .domain_extractors.routes_typescript import (
            extract_routes_from_call,
            extract_routes_nextjs_app_router,
            is_nextjs_route_file,
        )

        # (1) Express/Hono call-pattern walk over the whole tree.
        def _walk_for_routes(n: Any) -> None:
            if n.type == "call_expression":
                for route_node, route_edge in extract_routes_from_call(
                    source, n, "", rel_path, repo.name, repo.tenant_id,
                ):
                    batch.routes.append(route_node)
                    batch.route_edges.append(route_edge)
            for child in n.children:
                _walk_for_routes(child)
        _walk_for_routes(root)

        # (2) Next.js App Router file-based detection.
        if is_nextjs_route_file(rel_path):
            for route_node, route_edge in extract_routes_nextjs_app_router(
                source, root, rel_path, repo.name, repo.tenant_id, module_qn,
            ):
                batch.routes.append(route_node)
                batch.route_edges.append(route_edge)

    # Sprint 14j — module-level lexical declarations with constructor
    # values or annotations. `const app = new FastAPI()` at file top
    # becomes a module-scope binding.
    if module_qn:
        from .scope_resolution._adapter import MODULE_SCOPE_END
        for child in root.children:
            inner = child
            if inner.type == "export_statement":
                # Unwrap one level of `export const x = ...`.
                for sub in inner.children:
                    if sub.type in ("lexical_declaration", "variable_declaration"):
                        inner = sub
                        break
            if inner.type not in ("lexical_declaration", "variable_declaration"):
                continue
            for declarator in inner.children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                if name_node is None or name_node.type != "identifier":
                    continue
                # Skip arrow-function declarators (handled by _emit_arrow_const).
                value_node = declarator.child_by_field_name("value")
                if value_node is not None and value_node.type == "arrow_function":
                    continue
                # Annotation precedence first.
                type_raw_name: str | None = None
                type_annotation = declarator.child_by_field_name("type")
                if type_annotation is not None:
                    annot = _ts_type_annotation_text(source, type_annotation)
                    if annot:
                        type_raw_name = annot
                if type_raw_name is None and value_node is not None and value_node.type == "new_expression":
                    ctor = value_node.child_by_field_name("constructor")
                    if ctor is not None and ctor.type == "identifier":
                        type_raw_name = _node_text(source, ctor)
                if type_raw_name is None:
                    continue
                batch.local_var_bindings.append(LocalVarBinding(
                    repo=repo.name,
                    tenant_id=repo.tenant_id,
                    file_path=rel_path,
                    enclosing_scope_kind="module",
                    enclosing_line_start=0,
                    enclosing_line_end=MODULE_SCOPE_END - 1,
                    var_name=_node_text(source, name_node),
                    type_raw_name=type_raw_name,
                    line=inner.start_point[0] + 1,
                ))

    # Imports — file-level edges, with relative paths already resolved
    # against the repo file set.
    for target_qn, local_name, kind in _walk_imports(source, root, rel_path, repo_files):
        batch.imports.append(ImportEdge(
            repo=repo.name,
            file_path=rel_path,
            target_qn=target_qn,
            local_name=local_name,
            kind=kind,
        ))

    return batch
