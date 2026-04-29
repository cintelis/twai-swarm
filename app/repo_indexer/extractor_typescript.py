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
) -> IndexBatch:
    """Parse one .ts/.tsx/.js/.jsx file and return its IndexBatch fragment."""
    batch = IndexBatch(repo=repo)

    module_qn = module_qn_from_path(rel_path)
    batch.files.append(FileNode(repo=repo.name, path=rel_path, language=language, sha=sha))
    if module_qn:
        batch.modules.append(ModuleNode(repo=repo.name, qualified_name=module_qn, file_path=rel_path))

    tree = parser.parse(source)
    root = tree.root_node

    # Track names defined at the top level — same trick as the Python pass.
    local_names: set[str] = set()

    def _emit_function(node: Any, name: str, parent_class_qn: str = "", body_node: Any = None) -> None:
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
        batch.classes.append(ClassNode(
            repo=repo.name,
            qualified_name=qn,
            name=name,
            file_path=rel_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
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
        # Methods.
        if body is not None:
            for child in body.children:
                if child.type == "method_definition":
                    name_field = child.child_by_field_name("name")
                    if name_field is not None:
                        method_name = _node_text(source, name_field)
                        _emit_function(child, method_name, parent_class_qn=qn)

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
