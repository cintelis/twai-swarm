"""Tree-sitter C++ extractor — Sprint 16.

Walks the AST of one C/C++ source or header file and emits an
IndexBatch fragment. The shape mirrors `extractor_python.py` and
`extractor_typescript.py`; cross-language conventions documented
in `repo-indexer.md`.

Resolution model:
    - Functions / classes / structs DEFINED in this file → FunctionNode
      / ClassNode keyed on `<module>.<namespace>::<Class>.<method>`.
    - Free functions inside a `namespace foo::bar` get
      `<module>.foo::bar.<func>`.
    - Calls and inheritance to in-file names resolve directly; cross-file
      names are emitted as raw dotted/scoped strings and the resolver
      handles them via `#include` ImportEdges (wildcard-transitive,
      mirroring Python's `from x import *` semantics).

Module-qn convention (no Sprint 14e analog for C++):
    - Path-only, slashes → dots, extension stripped.
    - `src/audio/sound_engine.cpp` → `src.audio.sound_engine`
    - `include/twai/runtime/loop.h` → `include.twai.runtime.loop`

Const-method overloads:
    - `void foo();` and `void foo() const;` are different functions —
      MERGE-on-qn would collapse them. We append `:const` to the qn
      when `is_const=True` AND set the FunctionNode field for filters.

Out-of-line method definitions (Sprint 16f):
    - `void Foo::bar() { … }` at module-level emits a FunctionNode with
      `parent_class_qn=<module>.Foo`. Loader's MERGE-on-qn collapses
      with the in-class declaration; the impl row's body presence
      provides the line range.
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
)


def _module_qn_from_path(rel_path: str) -> str:
    """`src/audio/engine.cpp` → `src.audio.engine`. Header/source pairs
    map to DIFFERENT module qns (`engine.h` → `src.audio.engine_h`?
    No — we strip just the extension, so `engine.h` and `engine.cpp`
    both → `src.audio.engine`). That matches GitNexus's behaviour.
    """
    parts = rel_path.split("/")
    if not parts:
        return ""
    last = parts[-1]
    # Strip extension. Common cpp extensions handled here; `.h`-style
    # dotted multi-extensions (`.tar.gz`-pattern) don't occur in cpp.
    for ext in (".cpp", ".cc", ".cxx", ".c", ".hpp", ".hxx", ".hh", ".h"):
        if last.endswith(ext):
            last = last[: -len(ext)]
            break
    parts = parts[:-1] + [last] if last else parts[:-1]
    return ".".join(parts)


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _flatten_qualified_name(source: bytes, node: Any) -> str | None:
    """Flatten any cpp identifier-shape node to a dotted/scoped string.

    Handles:
        identifier              → "foo"
        qualified_identifier    → "Foo::bar" / "std::vector"
        template_function       → "make_shared" (drops the <T> args)
        template_type           → "Vec" (template head only)
        field_expression        → "obj.bar" / "ptr->bar"
        type_identifier         → "MyType"
        primitive_type          → "int" / "void" (returned as-is)
        operator_name           → "operator+" (best-effort)
        destructor_name         → "~Foo"

    Returns None when the node shape isn't recognised (e.g.
    parenthesized expressions, calls used as receivers).
    """
    t = node.type
    if t in ("identifier", "type_identifier", "field_identifier",
             "primitive_type", "namespace_identifier"):
        return _node_text(source, node)
    if t == "qualified_identifier":
        # Children: [scope :: name]. tree-sitter-cpp gives us
        # `scope` and `name` fields for nested `A::B::C` chains.
        scope = node.child_by_field_name("scope")
        name = node.child_by_field_name("name")
        scope_text = _flatten_qualified_name(source, scope) if scope is not None else None
        name_text = _flatten_qualified_name(source, name) if name is not None else None
        if name_text is None:
            # Fall back to raw text; covers `::foo` (global-scoped)
            return _node_text(source, node).replace(" ", "")
        if scope_text is None or scope_text == "":
            return name_text
        return f"{scope_text}::{name_text}"
    if t == "template_function":
        # `make_shared<X>`. Just the head name; template args dropped.
        head = node.child_by_field_name("name")
        if head is not None:
            return _flatten_qualified_name(source, head)
        return None
    if t == "template_type":
        head = node.child_by_field_name("name")
        if head is not None:
            return _flatten_qualified_name(source, head)
        return None
    if t == "field_expression":
        # `obj.bar` or `ptr->bar`. Use `.` as separator (we throw away
        # the `->` distinction; the resolver doesn't care).
        argument = node.child_by_field_name("argument")
        field = node.child_by_field_name("field")
        argt = _flatten_qualified_name(source, argument) if argument is not None else None
        ft = _flatten_qualified_name(source, field) if field is not None else None
        if argt is None or ft is None:
            return None
        return f"{argt}.{ft}"
    if t == "destructor_name":
        # `~Foo` — children: [`~`, identifier]
        for c in node.children:
            if c.type == "identifier":
                return f"~{_node_text(source, c)}"
        return None
    if t == "operator_name":
        return _node_text(source, node).replace(" ", "")
    if t == "pointer_declarator":
        # `*foo` — recurse into the declarator
        d = node.child_by_field_name("declarator")
        if d is not None:
            return _flatten_qualified_name(source, d)
    if t == "reference_declarator":
        for c in node.children:
            if c.type not in ("&", "&&"):
                hit = _flatten_qualified_name(source, c)
                if hit is not None:
                    return hit
    return None


def _extract_function_name_from_declarator(
    source: bytes, declarator: Any,
) -> tuple[str | None, bool, Any]:
    """Walk a `function_declarator`'s child chain to find the bare
    function name, whether the method is const, and the inner-most
    qualified-identifier node (for namespace-class detection).

    Returns (name, is_const, qid_node).

    `function_declarator` shape:
        function_declarator
          ├── declarator: identifier | qualified_identifier | field_identifier | template_function | destructor_name | operator_name | pointer_declarator | reference_declarator
          └── parameters: parameter_list
          └── (optional) type_qualifier: "const"
          └── (optional) virtual_specifier: "override" | "final"
    """
    name: str | None = None
    is_const = False
    qid_node: Any = None

    # Walk children to find the declarator field + type_qualifier siblings.
    inner = declarator.child_by_field_name("declarator")
    if inner is not None:
        if inner.type == "qualified_identifier":
            qid_node = inner
        name = _flatten_qualified_name(source, inner)

    # `const` and `override`/`final` appear as direct children, not fields,
    # of function_declarator. Check by node type.
    for c in declarator.children:
        if c.type == "type_qualifier":
            if _node_text(source, c).strip() == "const":
                is_const = True
        elif c.type == "virtual_specifier":
            # `override` / `final` — caller will set is_virtual.
            pass

    return name, is_const, qid_node


def _has_virtual_or_override(source: bytes, decl_node: Any) -> bool:
    """Inspect a declaration node (field_declaration or function_definition)
    for `virtual` keyword OR `override`/`final` virtual_specifier.

    `override` and `final` imply virtual without the keyword. Mirrors
    GitNexus's enforcement.
    """
    # `virtual` keyword surfaces in three shapes depending on grammar
    # version: as a bare `virtual` token, as a `virtual_function_specifier`
    # node, or wrapped in a `storage_class_specifier`. Walk all
    # descendants conservatively.
    for c in decl_node.children:
        if c.type in ("virtual", "virtual_specifier", "virtual_function_specifier"):
            return True
        if c.type == "storage_class_specifier" and _node_text(source, c).strip() == "virtual":
            return True

    # virtual_specifier nested inside the function_declarator
    declarator = None
    for c in decl_node.children:
        if c.type == "function_declarator":
            declarator = c
            break
        # nested through pointer/reference_declarator
        if c.type in ("pointer_declarator", "reference_declarator"):
            for sub in c.children:
                if sub.type == "function_declarator":
                    declarator = sub
                    break
            if declarator is not None:
                break
    if declarator is not None:
        for c in declarator.children:
            if c.type in ("virtual_specifier", "virtual_function_specifier"):
                return True
    return False


def _is_deleted_or_defaulted(source: bytes, decl_node: Any) -> bool:
    """`= delete` / `= default` — suppress these from emission to match
    GitNexus (method-extractors/configs/c-cpp.ts:312-318).

    The `default_method_clause` node holds the `= default` form;
    `delete_method_clause` holds `= delete`. Both appear as direct
    children of `function_definition` or `field_declaration`.
    """
    for c in decl_node.children:
        if c.type in ("default_method_clause", "delete_method_clause"):
            return True
    return False


def _function_params(source: bytes, params_node: Any) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return (param_names, param_types) for a `parameter_list`.

    `parameter_declaration` shape:
        parameter_declaration
          ├── type: <type expression>
          └── declarator: identifier | pointer_declarator | reference_declarator
    """
    if params_node is None:
        return (), ()
    names: list[str] = []
    types: list[tuple[str, str]] = []
    for child in params_node.children:
        if child.type != "parameter_declaration":
            continue
        type_node = child.child_by_field_name("type")
        decl_node = child.child_by_field_name("declarator")
        type_text = _node_text(source, type_node).strip() if type_node is not None else ""
        name = _flatten_qualified_name(source, decl_node) if decl_node is not None else None
        if name is None:
            # Anonymous parameter (e.g. `void f(int)`) — skip
            continue
        # Strip leading `*`/`&` decorations from the bare name
        bare = name.lstrip("*& ")
        names.append(bare)
        if type_text:
            types.append((bare, type_text))
    return tuple(names), tuple(types)


def _docstring_for(source: bytes, node: Any) -> str:
    """Best-effort docstring from a leading `comment` sibling.

    Walks backward from `node` looking for a `comment` whose end-line
    is contiguous with `node`'s start-line. C++ has no formal docstring
    convention; this captures the standard pattern of a `///`-prefixed
    block immediately before a function definition.
    """
    parent = node.parent
    if parent is None:
        return ""
    target_start = node.start_point[0]
    last_comment: Any = None
    for c in parent.children:
        if c.start_byte >= node.start_byte:
            break
        if c.type == "comment":
            last_comment = c
        else:
            # Reset — only contiguous comments count
            last_comment = None
    if last_comment is None:
        return ""
    if last_comment.end_point[0] + 1 < target_start:
        # Not contiguous — gap of more than 1 blank line
        return ""
    text = _node_text(source, last_comment)
    # Strip `//`, `///`, `/** */`, leading `*` per-line
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        for pfx in ("///", "//!", "//", "/**", "*/", "/*"):
            if ln.startswith(pfx):
                ln = ln[len(pfx):].strip()
                break
        if ln.startswith("* "):
            ln = ln[2:]
        elif ln == "*":
            ln = ""
        if ln:
            lines.append(ln)
    if not lines:
        return ""
    return lines[0][:200]


def _walk_calls(source: bytes, body_node: Any) -> list[tuple[str, int]]:
    """Return [(callee_dotted_name, line)] for every call site in a body.

    Handles four call flavours from GitNexus's SCM:
        - bare identifier           foo()
        - field_expression          obj.bar() / ptr->bar()
        - qualified_identifier      Foo::bar() / std::make_shared
        - template_function         make_shared<X>()
    Plus `new_expression` for constructor calls (`new Engine(...)` →
    callee="Engine").
    """
    found: list[tuple[str, int]] = []

    def _visit(n: Any) -> None:
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                dotted = _flatten_qualified_name(source, fn)
                if dotted:
                    found.append((dotted, n.start_point[0] + 1))
        elif n.type == "new_expression":
            # `new Foo(...)` — the type field carries the class name.
            type_node = n.child_by_field_name("type")
            if type_node is not None:
                dotted = _flatten_qualified_name(source, type_node)
                if dotted:
                    found.append((dotted, n.start_point[0] + 1))
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _namespace_path_text(source: bytes, ns_node: Any) -> str:
    """For `namespace twai::audio { … }` return "twai::audio".
    For `namespace foo { namespace bar { … } }` the caller stacks them.
    Anonymous namespace returns "" (caller substitutes a synthetic name).

    tree-sitter-cpp grammar variants:
        namespace_definition
          ├── name: namespace_identifier  (single segment)
          OR
          ├── name: nested_namespace_specifier (twai::audio chain)
          OR no name field (anonymous)
    """
    name = ns_node.child_by_field_name("name")
    if name is None:
        return ""
    if name.type == "namespace_identifier":
        return _node_text(source, name)
    # nested_namespace_specifier: walk children for namespace_identifier
    # nodes joined by ::
    if name.type == "nested_namespace_specifier":
        parts = [
            _node_text(source, c)
            for c in name.children
            if c.type == "namespace_identifier"
        ]
        return "::".join(parts)
    return _node_text(source, name).strip()


def _walk_includes(source: bytes, root: Any) -> list[tuple[str, bool]]:
    """Return [(include_path, is_quoted)] for every `#include` directive.

    `is_quoted=True` for `#include "foo.h"`, False for `#include <foo.h>`.
    Quoted includes get suffix-matched against the repo file set;
    angle-bracket includes are left unresolved (system headers).
    """
    found: list[tuple[str, bool]] = []
    for c in root.children:
        if c.type != "preproc_include":
            continue
        path_node = c.child_by_field_name("path")
        if path_node is None:
            continue
        text = _node_text(source, path_node).strip()
        if not text:
            continue
        is_quoted = text.startswith('"')
        # Strip surrounding quotes / angle brackets
        inner = text.strip('"').strip("<>").strip()
        if inner:
            found.append((inner, is_quoted))
    return found


def _resolve_quoted_include(
    include_path: str, importing_file: str, repo_files: set[str],
) -> str | None:
    """Suffix-match `include_path` against the repo file set.

    `#include "foo/bar.h"` matches any repo file whose tail equals
    `foo/bar.h`. Multiple matches → prefer one in the same directory
    as the importing file; fall back to the first match.

    Returns the resolved repo-relative path, or None.
    """
    target = include_path.replace("\\", "/").lstrip("./")
    candidates = [
        f for f in repo_files
        if f == target or f.endswith("/" + target)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer same-dir or nearest-ancestor
    importing_dir = importing_file.rsplit("/", 1)[0] if "/" in importing_file else ""
    best = None
    best_depth = -1
    for c in candidates:
        c_dir = c.rsplit("/", 1)[0] if "/" in c else ""
        # Common prefix depth
        a = importing_dir.split("/")
        b = c_dir.split("/")
        depth = 0
        for x, y in zip(a, b):
            if x == y:
                depth += 1
            else:
                break
        if depth > best_depth:
            best = c
            best_depth = depth
    return best or candidates[0]


def extract_cpp_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
    repo_files: set[str] | None = None,
) -> IndexBatch:
    """Parse one .cpp/.h file and return its IndexBatch fragment.

    `repo_files` is the set of repo-relative posix paths used by the
    quoted-include suffix matcher. Defaults to empty set; angle-bracket
    includes never need it.
    """
    if repo_files is None:
        repo_files = set()

    batch = IndexBatch(repo=repo)
    module_qn = _module_qn_from_path(rel_path)

    batch.files.append(FileNode(repo=repo.name, path=rel_path, language="cpp", sha=sha))
    if module_qn:
        batch.modules.append(ModuleNode(
            repo=repo.name, qualified_name=module_qn, file_path=rel_path,
        ))

    tree = parser.parse(source)
    root = tree.root_node

    # Track names defined in this file for same-file call resolution.
    local_names: set[str] = set()

    # Anonymous namespace counter — assign synthetic __anon_<line> names
    # so multiple anonymous namespaces in one TU don't collapse.
    def _ns_label(ns_node: Any) -> str:
        text = _namespace_path_text(source, ns_node)
        if text:
            return text
        return f"__anon_{ns_node.start_point[0] + 1}"

    def _emit_function(
        node: Any,
        ns_stack: list[str],
        parent_class_qn: str = "",
        parent_class_line_start: int = 0,
        parent_class_line_end: int = 0,
        is_method_decl: bool = False,
    ) -> None:
        """Emit a FunctionNode + derived call edges for a
        function_definition or in-class field_declaration that contains
        a function_declarator.

        `is_method_decl=True` for in-class declarations (header decl —
        body may or may not be present). Out-of-line definitions
        (`void Foo::bar()`) come through with `is_method_decl=False`
        but with the qid_node detection below pulling in parent_class_qn.
        """
        if _is_deleted_or_defaulted(source, node):
            return

        # Find the function_declarator. May be wrapped in pointer/
        # reference_declarator for return-type-modified shapes.
        declarator = node.child_by_field_name("declarator")
        # Unwrap pointer/reference wrappers
        depth = 0
        while declarator is not None and declarator.type in (
            "pointer_declarator", "reference_declarator",
        ) and depth < 3:
            inner = declarator.child_by_field_name("declarator")
            if inner is None:
                break
            declarator = inner
            depth += 1
        if declarator is None or declarator.type != "function_declarator":
            return

        name, is_const, qid_node = _extract_function_name_from_declarator(
            source, declarator,
        )
        if name is None:
            return

        # Skip ALL_CAPS_NAMES — tree-sitter-cpp parses macro invocations
        # like `ABSL_LOCKS_EXCLUDED(mu_) { ... }` as fake function
        # definitions when they appear after a real method's signature.
        # Without this filter, the fake "ABSL_LOCKS_EXCLUDED" function
        # overlaps line-range-wise with the real method's forward decl,
        # tripping the resolver's scope-tree invariant. C++ macro names
        # are conventionally ALL_CAPS_WITH_UNDERSCORES; real functions
        # virtually never are. Mirrors GitNexus's filter.
        bare_leaf = name.rsplit("::", 1)[-1].lstrip("~")
        if (
            len(bare_leaf) >= 3
            and bare_leaf.replace("_", "").isalnum()
            and bare_leaf.replace("_", "").isupper()
        ):
            return

        is_virtual = _has_virtual_or_override(source, node)

        # Sprint 16f — out-of-line method linking.
        # When `qid_node` is present (e.g. `void Foo::bar()`), split off
        # the trailing segment as the method name and use the prefix as
        # the parent_class_qn (anchored to this module).
        effective_parent_class_qn = parent_class_qn
        if qid_node is not None and "::" in name:
            scope, leaf = name.rsplit("::", 1)
            name = leaf
            # Anchor the scope to the current module's namespace stack.
            # Free-function OOL (rare) would have empty parent_class_qn.
            if module_qn:
                if ns_stack:
                    effective_parent_class_qn = (
                        f"{module_qn}.{'::'.join(ns_stack)}::{scope}"
                    )
                else:
                    effective_parent_class_qn = f"{module_qn}.{scope}"
            else:
                effective_parent_class_qn = scope

        # Build the qn. Free function inside namespace: `<module>.<ns>.fn`.
        # In-class method: `<parent_class_qn>.<method>`. OOL method:
        # `<effective_parent_class_qn>.<method>` (computed above).
        is_method = bool(effective_parent_class_qn) or is_method_decl
        if effective_parent_class_qn:
            qn_base = f"{effective_parent_class_qn}.{name}"
        else:
            ns_prefix = "::".join(ns_stack)
            if module_qn and ns_prefix:
                qn_base = f"{module_qn}.{ns_prefix}.{name}"
            elif module_qn:
                qn_base = f"{module_qn}.{name}"
            else:
                qn_base = name

        # Const-overload disambiguation: append `:const` to the qn so
        # MERGE-on-qn doesn't collapse `begin()` and `begin() const`.
        qn = f"{qn_base}:const" if is_const else qn_base

        local_names.add(name)

        params_node = declarator.child_by_field_name("parameters")
        param_names, param_types = _function_params(source, params_node)

        body = node.child_by_field_name("body")
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        batch.functions.append(FunctionNode(
            repo=repo.name,
            qualified_name=qn,
            name=name,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            is_async=False,
            is_method=is_method,
            parent_class_qn=effective_parent_class_qn,
            params=param_names,
            param_types=param_types,
            return_type_raw="",
            docstring=_docstring_for(source, node),
            is_const=is_const,
            is_virtual=is_virtual,
        ))

        if body is not None:
            for callee_dotted, line in _walk_calls(source, body):
                head = callee_dotted.split(".", 1)[0].split("::", 1)[0]
                if head in local_names and "." not in callee_dotted and "::" not in callee_dotted:
                    callee_qn = f"{module_qn}.{callee_dotted}" if module_qn else callee_dotted
                else:
                    callee_qn = callee_dotted
                batch.calls.append(CallEdge(
                    repo=repo.name, caller_qn=qn,
                    callee_qn=callee_qn, line=line,
                ))

    def _emit_class(
        node: Any,
        ns_stack: list[str],
    ) -> None:
        """Emit ClassNode for `class_specifier` / `struct_specifier` and
        recurse into the body for methods.
        """
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node is None:
            return
        # Class name might be a template_type — use the head identifier
        bare_name = _flatten_qualified_name(source, name_node)
        if bare_name is None:
            return

        ns_prefix = "::".join(ns_stack)
        if module_qn and ns_prefix:
            qn = f"{module_qn}.{ns_prefix}::{bare_name}"
        elif module_qn:
            qn = f"{module_qn}.{bare_name}"
        else:
            qn = bare_name
        local_names.add(bare_name)

        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        batch.classes.append(ClassNode(
            repo=repo.name, qualified_name=qn, name=bare_name,
            file_path=rel_path, line_start=line_start, line_end=line_end,
            docstring="",
        ))

        # Inheritance — base_class_clause child of class_specifier
        for c in node.children:
            if c.type != "base_class_clause":
                continue
            for sub in c.children:
                # Skip access specifiers, commas, virtual keyword
                if sub.type in (",", ":", "public", "private", "protected", "virtual",
                                "access_specifier"):
                    continue
                parent_text = _flatten_qualified_name(source, sub)
                if parent_text:
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name, child_qn=qn, parent_qn=parent_text,
                    ))

        # Walk body for in-class function decls
        if body is not None:
            _walk_class_body(body, ns_stack, qn, line_start, line_end)

    def _walk_class_body(
        body: Any, ns_stack: list[str], class_qn: str,
        class_line_start: int, class_line_end: int,
    ) -> None:
        for child in body.children:
            # `field_declaration` covers methods with a return type
            # (`void play();`); `declaration` covers ctor/dtor (no return
            # type) and `using` declarations; `function_definition`
            # covers in-line method bodies.
            if child.type in ("field_declaration", "declaration"):
                # Detect a function_declarator inside the declarator
                # chain (may be wrapped in pointer/reference declarators).
                has_fn_decl = False
                d = child.child_by_field_name("declarator")
                walked = 0
                while d is not None and walked < 4:
                    if d.type == "function_declarator":
                        has_fn_decl = True
                        break
                    if d.type in ("pointer_declarator", "reference_declarator"):
                        d = d.child_by_field_name("declarator")
                        walked += 1
                        continue
                    break
                if has_fn_decl:
                    _emit_function(
                        child, ns_stack,
                        parent_class_qn=class_qn,
                        parent_class_line_start=class_line_start,
                        parent_class_line_end=class_line_end,
                        is_method_decl=True,
                    )
            elif child.type == "function_definition":
                # Inline definition inside the class body
                _emit_function(
                    child, ns_stack,
                    parent_class_qn=class_qn,
                    parent_class_line_start=class_line_start,
                    parent_class_line_end=class_line_end,
                    is_method_decl=True,
                )
            elif child.type in ("class_specifier", "struct_specifier"):
                # Nested class
                _emit_class(child, ns_stack)
            elif child.type == "access_specifier":
                continue

    def _walk(node: Any, ns_stack: list[str]) -> None:
        for child in node.children:
            t = child.type
            if t == "namespace_definition":
                label = _ns_label(child)
                body = child.child_by_field_name("body")
                if body is not None:
                    _walk(body, ns_stack + [label])
            elif t in ("class_specifier", "struct_specifier"):
                _emit_class(child, ns_stack)
            elif t == "function_definition":
                _emit_function(child, ns_stack)
            elif t == "template_declaration":
                # `template<...> class Foo` / `template<...> void f()` —
                # wrapper around the actual entity. Recurse to find the
                # class/function inside.
                _walk(child, ns_stack)
            elif t in ("declaration", "linkage_specification"):
                # `extern "C" { ... }` — recurse into body
                _walk(child, ns_stack)
            elif t == "preproc_include":
                # Handled separately at module scope
                pass

    _walk(root, [])

    # Sprint 16d — `#include` resolution. Quoted forms suffix-match
    # against the repo file set; angle-bracket forms emit ImportEdge
    # targeting the bare path (resolver leaves them unresolved → become
    # Symbol nodes, which is correct for system headers).
    for include_path, is_quoted in _walk_includes(source, root):
        if is_quoted:
            resolved_file = _resolve_quoted_include(include_path, rel_path, repo_files)
            if resolved_file is not None:
                target_module_qn = _module_qn_from_path(resolved_file)
                if target_module_qn:
                    batch.imports.append(ImportEdge(
                        repo=repo.name,
                        file_path=rel_path,
                        target_qn=target_module_qn,
                        local_name="*",
                        kind="module",
                    ))
                    continue
        # Unresolved — emit a non-resolving ImportEdge so the include
        # is at least visible in queries. The resolver will leave it
        # alone (target_qn won't match any module).
        batch.imports.append(ImportEdge(
            repo=repo.name,
            file_path=rel_path,
            target_qn=include_path,
            local_name="*",
            kind="module",
        ))

    return batch


__all__ = ["extract_cpp_file"]
