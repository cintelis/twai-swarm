"""Tree-sitter Java extractor — Sprint 17a.

Walks the AST of one Java source file and emits an IndexBatch fragment.
Mirrors the shape of `extractor_cpp.py` and `extractor_python.py`;
cross-language conventions documented in `repo-indexer.md` and the
Sprint 17 plan.

Module-qn convention (NEW for Java — different from python/cpp):
    - Java's `package com.foo.bar;` declaration is authoritative. The
      filesystem path is incidental.
    - `package com.bench;` + file `CalculatorApp.java` →
      `ModuleNode.qualified_name = "com.bench.CalculatorApp"`
    - File at `src/main/java/com/right/Foo.java` declaring
      `package com.right;` → module qn = `"com.right.Foo"`, NOT
      anything path-derived.
    - Default package (no `package` declaration — rare): falls back to
      filename-stem only. `Foo.java` → module qn = `"Foo"`.
    - `package-info.java` and `module-info.java` are special Java files;
      we just emit File + Module nodes for them and skip class extraction
      naturally (they have no class_declaration children).
    - Inner classes embed in the qn: `Outer.Inner.foo` for a method on
      a nested class.

Annotation capture (17a contribution):
    - Class + Function annotations stored as raw source strings,
      including the `@`. e.g. `'@RestController'`,
      `'@RequestMapping("/api")'`, `'@Override'`, `'@GetMapping("/x")'`.
    - No argument parsing — Sprint 17f's Spring routes domain extractor
      will parse arguments out of these strings at consumption time.

Inheritance capture (17b contribution):
    - `class Foo extends Bar` → InheritsEdge(child_qn=qn(Foo),
      parent_qn="Bar")
    - `class Foo implements A, B` → one InheritsEdge per interface
    - `interface Foo extends A, B` → multiple InheritsEdges
    - Generic parents have args stripped: `extends Base<T>` →
      parent_qn="Base"
    - Scoped parents preserve dots: `extends pkg.Base` →
      parent_qn="pkg.Base"
    - Implicit `java.lang.Object` / `java.lang.Record` /
      `java.lang.Enum` / `java.lang.annotation.Annotation` parents
      are NOT modelled (matches the no-implicit-`java.lang.*` rule).

Out of 17a/17b scope (LATER sub-sprints — explicitly NOT handled here):
    - Call edges (bare / field_access / scoped / new) — Sprint 17c.
    - Imports + same-package implicit visibility + java.lang.* —
      Sprint 17d.
    - Method-overload qn disambiguation via `:type1,type2` suffix —
      Sprint 17e. Two `add(int,int)` and `add(double,double)` methods
      currently emit two FunctionNodes with the same qn; the loader's
      MERGE-on-(repo, qn) will collapse them. Acceptable for 17a; 17e
      will fix it.
    - Anonymous classes + lambdas (synthetic `__anon_<line>` /
      `__lambda_<line>` names) — Sprint 17e.
    - Spring routes domain extraction — Sprint 17f.

Known V1 limitations (carried forward — see Sprint 17 plan §pitfalls):
    - Lombok `@Data` and similar annotation processors generate
      methods at compile time that we never see in source. Documented;
      no synthesis.
    - Java records have implicit accessor methods (`x()`, `y()` for
      `record Point(int x, int y)`) that don't appear in source.
      We capture the compact constructor when present but not the
      synthetic accessors.
    - `module-info.java` declarations are treated as ordinary files
      with no extracted classes.
"""
from __future__ import annotations

from typing import Any

from .actions import (
    ClassNode,
    FileNode,
    FunctionNode,
    IndexBatch,
    InheritsEdge,
    ModuleNode,
    RepoNode,
)


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _filename_stem(rel_path: str) -> str:
    """`src/main/java/com/foo/Bar.java` → `Bar`. Strip directory and
    `.java` extension. Returns the bare name unchanged for files in
    the repo root.
    """
    last = rel_path.rsplit("/", 1)[-1]
    if last.endswith(".java"):
        last = last[: -len(".java")]
    return last


def _extract_package(source: bytes, root: Any) -> str:
    """Return the package name from `package com.foo.bar;` at the file
    top, or `""` for the default package.

    `package_declaration` shape:
        package_declaration
          ├── `package` keyword
          ├── scoped_identifier  (or identifier for top-level packages)
          └── `;`

    `scoped_identifier` flattens cleanly via `node.text` — tree-sitter
    preserves the dotted form verbatim.
    """
    for child in root.children:
        if child.type != "package_declaration":
            continue
        # The named children are the keyword + the dotted name + `;`.
        # We want the identifier-shaped child.
        for sub in child.children:
            if sub.type in ("scoped_identifier", "identifier"):
                return _node_text(source, sub).strip()
    return ""


def _module_qn_from_package(package: str, rel_path: str) -> str:
    """Build the ModuleNode.qualified_name from the package + filename.

    Rules:
        package="com.bench", file="…/CalculatorApp.java"  → "com.bench.CalculatorApp"
        package="",          file="Foo.java"               → "Foo"
        package="com.x",     file="package-info.java"      → "com.x.package-info"
        package="",          file="module-info.java"       → "module-info"

    The `package-info` / `module-info` cases are rare special files; we
    use the filename stem as-is and let the loader treat them like any
    other module. Their class_declaration child set is typically empty.
    """
    stem = _filename_stem(rel_path)
    if package:
        return f"{package}.{stem}" if stem else package
    return stem


def _extract_annotations(
    source: bytes, modifiers_node: Any,
) -> tuple[str, ...]:
    """Return the raw text of every annotation in a `modifiers` node.

    `modifiers` is a flat container that holds annotations + access /
    modifier keywords as peer children. Annotations come in two shapes:
      - `marker_annotation`  → `@Override`
      - `annotation`         → `@RequestMapping("/api")`

    Returned strings are the verbatim source spans, including the `@`
    and (for `annotation`) the parenthesised argument list. Order
    matches source order. Empty tuple when `modifiers_node` is None or
    holds no annotations.
    """
    if modifiers_node is None:
        return ()
    out: list[str] = []
    for child in modifiers_node.children:
        if child.type in ("marker_annotation", "annotation"):
            text = _node_text(source, child).strip()
            if text:
                out.append(text)
    return tuple(out)


def _modifiers_node(node: Any) -> Any:
    """Locate the `modifiers` child of a class / interface / enum /
    record / annotation_type / method / constructor declaration.

    Tree-sitter-java doesn't expose `modifiers` via `child_by_field_name`
    — it's a positional named child appearing as the first non-comment
    sibling. Walk the children list and return the first
    `modifiers`-typed child, or None if absent.
    """
    for child in node.children:
        if child.type == "modifiers":
            return child
    return None


def _extract_param_names_and_types(
    source: bytes, params_node: Any,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return (param_names, param_types) for a `formal_parameters` node.

    `formal_parameter` shape:
        formal_parameter
          ├── (optional) modifiers   (e.g. `final`)
          ├── type:  <type expression>
          └── name:  identifier

    `spread_parameter` (varargs `String... args`) shape:
        spread_parameter
          ├── <type node>            (no field name)
          ├── `...`
          └── variable_declarator
                └── name: identifier

    For varargs we keep the raw form like `String...` in the type text
    so consumers can distinguish vararg signatures from arrays.
    """
    if params_node is None:
        return (), ()
    names: list[str] = []
    types: list[tuple[str, str]] = []

    for child in params_node.children:
        if child.type == "formal_parameter":
            type_node = child.child_by_field_name("type")
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(source, name_node)
            names.append(name)
            if type_node is not None:
                type_text = _node_text(source, type_node).strip()
                if type_text:
                    types.append((name, type_text))
        elif child.type == "spread_parameter":
            # Positional children: type node (idx 0), `...`, variable_declarator.
            # No field names, so walk children manually.
            type_text = ""
            name = ""
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name_node = sub.child_by_field_name("name")
                    if name_node is None:
                        # Some grammar versions: identifier as a direct child.
                        for vc in sub.children:
                            if vc.type == "identifier":
                                name_node = vc
                                break
                    if name_node is not None:
                        name = _node_text(source, name_node)
                elif sub.type not in ("...",):
                    # First non-`...` child is the type expression.
                    if not type_text:
                        type_text = _node_text(source, sub).strip()
            if not name:
                continue
            names.append(name)
            if type_text:
                # Mark vararg shape so consumers can tell it apart from
                # an array param. `String... args` → `"String..."`.
                types.append((name, f"{type_text}..."))
    return tuple(names), tuple(types)


# Node types treated as type-declaring containers — ClassNode is emitted
# for each, regardless of whether they're class / interface / enum /
# record / annotation_type. The graph schema doesn't currently
# distinguish among them; future sprints could add a `kind` field.
_TYPE_DECL_TYPES = frozenset({
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "annotation_type_declaration",
})

# Node types that emit a FunctionNode. `compact_constructor_declaration`
# (records' inner `Foo { … }`) is included; its parameters are inherited
# from the enclosing record's header (we walk to the parent record_decl
# to fetch them at emit time).
_FUNCTION_DECL_TYPES = frozenset({
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
})


def _extract_parent_name(source: bytes, type_node: Any) -> str:
    """Return the dotted parent qn for a type node appearing in an
    `extends` or `implements` clause.

    Three node shapes the tree-sitter-java grammar produces in this
    position (confirmed against grammar 0.23.x):

      - `type_identifier`  →  bare name (`Animal`)
      - `scoped_type_identifier`  →  dotted form (`pkg.Base`,
        `com.foo.Bar.Inner`); `node.text` already preserves it verbatim
      - `generic_type`  →  `[type_identifier_or_scoped, type_arguments]`
        positionally; we recurse into child 0 and drop the args. So
        `Base<T>` → `"Base"` and `pkg.Base<T,U>` → `"pkg.Base"`.

    For any other node type we return `""` defensively — the caller
    skips the edge rather than emitting a malformed parent_qn. This
    matches the spec: the resolver maps these strings to ClassNodes
    later; emitting garbage here would create dangling Symbol nodes.
    """
    if type_node is None:
        return ""
    t = type_node.type
    if t == "type_identifier":
        return _node_text(source, type_node).strip()
    if t == "scoped_type_identifier":
        # `node.text` flattens the dotted form (e.g. `pkg.Base`) directly.
        return _node_text(source, type_node).strip()
    if t == "generic_type":
        # Positional children: [head_type, type_arguments]. Head is the
        # first named child; recurse so we handle `pkg.Base<T>` (head =
        # scoped_type_identifier) the same way as `Base<T>` (head =
        # type_identifier).
        for sub in type_node.children:
            if sub.type in (
                "type_identifier",
                "scoped_type_identifier",
                "generic_type",
            ):
                return _extract_parent_name(source, sub)
        return ""
    return ""


def _iter_type_list_children(type_list_node: Any) -> list[Any]:
    """Return the type-node children of a `type_list` node (the kind
    that appears under `super_interfaces` and `extends_interfaces`).

    Filters out punctuation (`,`) and other non-type children. The
    interesting type nodes are `type_identifier`,
    `scoped_type_identifier`, and `generic_type` — same set as
    `_extract_parent_name` handles.
    """
    if type_list_node is None:
        return []
    out: list[Any] = []
    for child in type_list_node.children:
        if child.type in (
            "type_identifier",
            "scoped_type_identifier",
            "generic_type",
        ):
            out.append(child)
    return out


def _find_named_child(node: Any, child_type: str) -> Any:
    """Return the first named child of `node` whose type matches
    `child_type`, or None. Used for grammar nodes that don't expose
    their interesting children via `child_by_field_name` — notably
    `extends_interfaces` on `interface_declaration`, which has no
    field name attached.
    """
    if node is None:
        return None
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def _emit_inherits_for_type_decl(
    repo: RepoNode,
    source: bytes,
    node: Any,
    class_qn: str,
    batch: IndexBatch,
) -> None:
    """Emit InheritsEdges for one type declaration (class / interface /
    enum / record). Annotation-type declarations are skipped — they
    implicitly extend `java.lang.annotation.Annotation`, which we don't
    model (matches the no-implicit-`java.lang.*` rule from the plan).

    Edge sources by node kind:
      - class_declaration:   `superclass=` field (single) +
                             `interfaces=` field (multiple)
      - record_declaration:  `interfaces=` field only (records can't
                             `extends`; their implicit `java.lang.Record`
                             parent is intentionally not modelled)
      - enum_declaration:    `interfaces=` field only (enums can't
                             `extends`; implicit `java.lang.Enum` not
                             modelled)
      - interface_declaration: walks named children for the
                               `extends_interfaces` node — that node has
                               NO field name, hence the manual scan.
                               Interfaces support multiple parents.

    `interfaces=` returns a `super_interfaces` node containing one
    `type_list` child; iterate the type-list's typed children.
    `extends_interfaces` (on interfaces) wraps a `type_list` directly.
    """
    t = node.type

    if t == "class_declaration":
        # extends: single parent.
        superclass_node = node.child_by_field_name("superclass")
        if superclass_node is not None:
            for sub in superclass_node.children:
                parent = _extract_parent_name(source, sub)
                if parent:
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name,
                        child_qn=class_qn,
                        parent_qn=parent,
                    ))
                    break  # extends takes exactly one type

        # implements: zero-or-more parents.
        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node is not None:
            type_list = _find_named_child(interfaces_node, "type_list")
            for type_child in _iter_type_list_children(type_list):
                parent = _extract_parent_name(source, type_child)
                if parent:
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name,
                        child_qn=class_qn,
                        parent_qn=parent,
                    ))

    elif t in ("record_declaration", "enum_declaration"):
        # Records and enums can `implements` but not `extends` — their
        # implicit `java.lang.Record` / `java.lang.Enum` parents are
        # intentionally NOT modelled (per plan §"java.lang.* allowlist").
        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node is not None:
            type_list = _find_named_child(interfaces_node, "type_list")
            for type_child in _iter_type_list_children(type_list):
                parent = _extract_parent_name(source, type_child)
                if parent:
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name,
                        child_qn=class_qn,
                        parent_qn=parent,
                    ))

    elif t == "interface_declaration":
        # `extends_interfaces` is a named child with NO field name in
        # tree-sitter-java 0.23.x — `child_by_field_name` returns None.
        # Walk children manually.
        extends_node = _find_named_child(node, "extends_interfaces")
        if extends_node is not None:
            type_list = _find_named_child(extends_node, "type_list")
            for type_child in _iter_type_list_children(type_list):
                parent = _extract_parent_name(source, type_child)
                if parent:
                    batch.inherits.append(InheritsEdge(
                        repo=repo.name,
                        child_qn=class_qn,
                        parent_qn=parent,
                    ))

    # annotation_type_declaration: intentionally skipped — implicit
    # `java.lang.annotation.Annotation` parent is not modelled.


def _body_node(node: Any) -> Any:
    """Locate the body of a type-declaring container.

    Field name varies by grammar branch:
      class_declaration            → body=class_body
      record_declaration           → body=class_body
      interface_declaration        → body=interface_body
      enum_declaration             → body=enum_body
      annotation_type_declaration  → body=annotation_type_body

    All are exposed via `child_by_field_name("body")` in tree-sitter-java
    0.23.x. Falls back to scanning children for the known body node types
    if the field lookup misses (defensive against grammar drift).
    """
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    for child in node.children:
        if child.type in (
            "class_body", "interface_body", "enum_body",
            "annotation_type_body",
        ):
            return child
    return None


def extract_java_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
    repo_files: set[str] | None = None,
) -> IndexBatch:
    """Parse one .java file and return its IndexBatch fragment.

    `repo_files` is currently unused (no import resolution in 17a) but
    accepted for signature parity with `extract_cpp_file` and forward-
    compatibility with Sprint 17d's import resolver.
    """
    del repo_files  # 17d will use this for cross-file import resolution.

    batch = IndexBatch(repo=repo)
    tree = parser.parse(source)
    root = tree.root_node

    package_qn = _extract_package(source, root)
    module_qn = _module_qn_from_package(package_qn, rel_path)

    # Every Java file gets a FileNode + ModuleNode, even
    # `package-info.java` / `module-info.java` (which carry no class).
    batch.files.append(FileNode(
        repo=repo.name, path=rel_path, language="java", sha=sha,
        package=package_qn,
    ))
    if module_qn:
        batch.modules.append(ModuleNode(
            repo=repo.name, qualified_name=module_qn, file_path=rel_path,
        ))

    def _emit_class(node: Any, parent_class_qn_stack: list[str]) -> None:
        """Emit ClassNode for one type declaration (class / interface /
        enum / record / annotation_type) and recurse into its body for
        nested types + methods.

        `parent_class_qn_stack` holds the qns of enclosing types — empty
        at file scope, single-element for one level of nesting, etc.
        Inner-class qns concatenate: parent qn + "." + name.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(source, name_node)
        annotations = _extract_annotations(source, _modifiers_node(node))

        if parent_class_qn_stack:
            class_qn = f"{parent_class_qn_stack[-1]}.{name}"
        elif package_qn:
            class_qn = f"{package_qn}.{name}"
        else:
            class_qn = name

        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        batch.classes.append(ClassNode(
            repo=repo.name,
            qualified_name=class_qn,
            name=name,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            docstring="",
            annotations=annotations,
        ))

        # 17b: emit InheritsEdges for `extends` + `implements` clauses.
        # Implicit `java.lang.Object` (classes), `java.lang.Record`
        # (records), `java.lang.Enum` (enums), and
        # `java.lang.annotation.Annotation` (annotation types) are all
        # intentionally NOT modelled — see plan §"java.lang.* allowlist".
        _emit_inherits_for_type_decl(repo, source, node, class_qn, batch)

        body = _body_node(node)
        if body is None:
            return
        # Walk the body. For enums the methods live inside an
        # `enum_body_declarations` wrapper child; we recurse through it
        # naturally via `_walk`. Annotation types' element declarations
        # (`String value();`) are NOT emitted as FunctionNodes — they're
        # not really methods; out of 17a scope.
        _walk(body, parent_class_qn_stack + [class_qn])

    def _emit_function(node: Any, parent_class_qn_stack: list[str]) -> None:
        """Emit FunctionNode for one method / constructor / compact-
        constructor declaration. Methods at file-scope (impossible in
        valid Java but defensive) get module-qn-prefixed qns; methods
        inside a type get parent-class-qn-prefixed qns.

        Constructor naming convention: we use the bare class name (the
        identifier from the constructor_declaration), which matches how
        the source reads. So `public Calculator() {}` inside `class
        Calculator` produces FunctionNode qn `<…>.Calculator.Calculator`.
        Mirrors the Java source view of constructors-as-named-methods.

        Compact constructors (record-only; `record Foo(...) { Foo { … } }`)
        have a `name` identifier child but no `formal_parameters` —
        their parameters are inherited from the enclosing record header.
        We walk to the parent record_declaration to fetch them.

        TODO 17e: When >1 method on the same class shares a name +
        differs only by parameter types, we currently emit identical
        qns and the loader's MERGE collapses them. Sprint 17e will add
        the `:type1,type2` qn suffix for overload disambiguation.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(source, name_node)
        annotations = _extract_annotations(source, _modifiers_node(node))

        # Compact constructors don't have a formal_parameters child;
        # walk to the parent record_declaration for params.
        params_node = node.child_by_field_name("parameters")
        if params_node is None and node.type == "compact_constructor_declaration":
            # parent → class_body, parent.parent → record_declaration
            grand = node.parent.parent if node.parent is not None else None
            if grand is not None and grand.type == "record_declaration":
                params_node = grand.child_by_field_name("parameters")
        param_names, param_types = _extract_param_names_and_types(source, params_node)

        is_method = bool(parent_class_qn_stack)
        parent_class_qn = parent_class_qn_stack[-1] if is_method else ""

        if is_method:
            fn_qn = f"{parent_class_qn}.{name}"
        elif package_qn:
            fn_qn = f"{package_qn}.{name}"
        else:
            fn_qn = name

        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        batch.functions.append(FunctionNode(
            repo=repo.name,
            qualified_name=fn_qn,
            name=name,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            is_async=False,
            is_method=is_method,
            parent_class_qn=parent_class_qn,
            params=param_names,
            param_types=param_types,
            return_type_raw="",
            docstring="",
            annotations=annotations,
        ))

    def _walk(node: Any, parent_class_qn_stack: list[str]) -> None:
        """Recurse into `node`'s children. Type declarations recurse via
        `_emit_class` (which calls `_walk` on the body); methods /
        constructors are emitted in place. Other nodes recurse so we find
        nested types inside method bodies (legal in Java — local
        classes — though rare).
        """
        for child in node.children:
            t = child.type
            if t in _TYPE_DECL_TYPES:
                _emit_class(child, parent_class_qn_stack)
            elif t in _FUNCTION_DECL_TYPES:
                _emit_function(child, parent_class_qn_stack)
            else:
                # Recurse into anything else so we find nested
                # type_declarations inside method bodies / enum body
                # declarations / etc. The named-only iteration below
                # would also work, but `children` matches what the cpp
                # extractor does and is simpler to reason about.
                _walk(child, parent_class_qn_stack)

    _walk(root, [])

    return batch


__all__ = ["extract_java_file"]
