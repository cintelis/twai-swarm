"""Tree-sitter Java extractor — Sprint 17a / 17b / 17c / 17d.

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

Call edges (17c contribution):
    - Bare `foo()` (`method_invocation`, no `object` field) →
      callee_qn = `foo`. If `foo` is in same-file local_names, prefix
      with `module_qn + "."` so same-file calls resolve to a
      qualified target. (Heuristic — works for the typical "calling
      a sibling method on the file's primary class" case.)
    - Field-access / scoped / static `obj.bar()` / `Foo.staticBar()` /
      `obj.list.add(x)` (`method_invocation` with `object=` field) →
      callee_qn = `<flattened-receiver>.<name>`. No syntactic
      distinction between instance and static at the AST level; the
      resolver disambiguates later. Chained field access flattens
      via `.` recursion — `obj.list.add` stays joined with periods.
    - Constructor `new Foo()` / `new Foo<T>()` / `new pkg.Bar()`
      (`object_creation_expression`) → callee_qn = head identifier
      (`Foo` / `pkg.Bar`); generics stripped via the same head-recurse
      logic as 17b's parent-name extractor.
    - `super(x)` / `this(x)` (`explicit_constructor_invocation`) →
      callee_qn = literal `"super"` / `"this"`. Resolver maps these
      to the parent / same class's constructor.
    - Method references (`Foo::bar`, `this::run`, `String::length`)
      (`method_reference`) → low-priority CallEdge with
      callee_qn = `<receiver>.<method>`. `Foo::new` (constructor
      reference) is captured as `Foo.new` — the resolver can map
      `.new` later or treat it as unresolved. If the shape is
      uncertain, we skip rather than crash.

Call-edge limitations (documented for 17e/17f follow-up):
    - Calls inside lambda bodies attribute to the ENCLOSING method
      — lambdas don't yet have synthetic FunctionNodes (Sprint 17e).
      Same for anonymous-class method bodies.
    - Static / instance initializer blocks (`static { … }` / `{ … }`)
      and field initializer expressions (`= new Foo()`) have no
      enclosing FunctionNode, so calls inside them are SKIPPED for
      17c. Sprint 17e will model these as synthetic `__cinit__` /
      `__init__`-style nodes.
    - No type-binding: `User u = new User(); u.foo()` emits
      callee_qn=`u.foo`, not `User.foo`. Receiver-type inference
      is a future-sprint concern.
    - Annotation arguments that look like calls
      (`@SuppressWarnings(value = "x")`) are NOT `method_invocation`
      nodes — no special handling needed.

Imports + java.lang.* visibility (17d contribution):
    - Every `import` declaration emits ONE ImportEdge. Four shapes:
        `import com.foo.Bar;`         → target_qn="com.foo.Bar",
                                         local_name="Bar",  kind="symbol"
        `import com.foo.*;`           → target_qn="com.foo",
                                         local_name="*",    kind="module"
        `import static com.foo.B.x;`  → target_qn="com.foo.B.x",
                                         local_name="x",    kind="symbol"
        `import static com.foo.B.*;`  → target_qn="com.foo.B",
                                         local_name="*",    kind="symbol"
    - Same-package implicit visibility (`com.bench.Foo` and
      `com.bench.Bar` see each other without an import) is NOT emitted
      as synthetic edges — the resolver consults `FileNode.package`
      (added in 17a) at query time. Emitting an ImportEdge per file
      pair would explode edge count on multi-file packages.
    - `java.lang.*` allowlist exposed as the module-level constant
      `JAVA_LANG_IMPLICIT`. The resolver consults it to decide whether
      a bare-name reference (`String`, `Override`, `RuntimeException`,
      etc.) should resolve to an external Symbol or be treated as
      unknown. The extractor itself does NOT special-case java.lang —
      it just publishes the allowlist for downstream consumers.
    - Build-system classpath discovery (Maven `pom.xml`, Gradle
      `build.gradle`) is explicitly out of scope.
    - `package-info.java` annotation imports use the same machinery —
      no special-casing required.
    - `module-info.java` `requires` / `exports` clauses are NOT
      modelled (consistent with 17a's choice to skip module-info).

Out of 17a/17b/17c/17d scope (LATER sub-sprints — explicitly NOT handled):
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


# ─── 17d: java.lang.* implicit-visibility allowlist ───────────────────────
#
# The Java Language Specification §7.5.5 declares all top-level types in
# `java.lang.*` to be implicitly imported into every compilation unit. This
# constant lists the package-level subset most likely to appear bare in
# user code — basic types, common errors / exceptions, threading, class
# metadata, and the standard built-in annotations. It is INTENTIONALLY
# not exhaustive: the full `java.lang` surface (reflection types in
# `java.lang.reflect`, instrument / management subpackages, etc.) is
# huge and rarely used unqualified.
#
# Resolver convention: when the resolver encounters a bare-name reference
# matching one of these names AND no in-file or imported binding shadows
# it, treat it as a known external Symbol and don't waste cycles trying
# to resolve. Source: JLS §7.5.5, Java 21 java.lang summary.
JAVA_LANG_IMPLICIT: frozenset[str] = frozenset({
    "ArithmeticException",
    "ArrayIndexOutOfBoundsException",
    "Boolean",
    "Byte",
    "Character",
    "CharSequence",
    "Class",
    "ClassCastException",
    "ClassLoader",
    "Comparable",
    "Deprecated",
    "Double",
    "Enum",
    "Error",
    "Exception",
    "Float",
    "FunctionalInterface",
    "IllegalArgumentException",
    "IllegalStateException",
    "IndexOutOfBoundsException",
    "Integer",
    "Iterable",
    "Long",
    "Math",
    "Number",
    "NullPointerException",
    "NumberFormatException",
    "Object",
    "Override",
    "Record",
    "Runnable",
    "RuntimeException",
    "SafeVarargs",
    "Short",
    "String",
    "StringBuffer",
    "StringBuilder",
    "SuppressWarnings",
    "System",
    "Thread",
    "ThreadLocal",
    "Throwable",
    "UnsupportedOperationException",
    "Void",
})


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


# ─── 17c: call-edge extraction helpers ─────────────────────────────────────


def _flatten_receiver_text(source: bytes, node: Any) -> str:
    """Flatten a method_invocation `object` (or a field_access `object`)
    into a dotted form suitable for use as a CallEdge callee prefix.

    Recognised shapes:
      - `identifier`          → its raw text (`"obj"`)
      - `this`                → `"this"`
      - `super`               → `"super"`
      - `field_access`        → recurse on `object` + `.` + `field.text`
                                (e.g. `obj.list` for `obj.list.add(x)`'s
                                `object`)
      - `method_invocation`   → best-effort raw text of the call expr
                                (so `getX().y` flattens to `"getX().y"`).
                                The resolver currently can't resolve
                                method-chained receivers; recorded
                                verbatim for future enhancement.
      - `parenthesized_expression` → recurse into the inner expression
      - anything else         → fall back to raw `node.text` (best-effort).

    Notes for 17e/17f follow-up: shapes we handle by raw-text fallback
    include `array_access`, `cast_expression`, and `object_creation_
    expression` (chained methods on a `new Foo().bar()`). They produce
    callee strings the resolver won't bind today, but the edge is at
    least visible for queries.
    """
    if node is None:
        return ""
    t = node.type
    if t == "identifier":
        return _node_text(source, node)
    if t == "this":
        return "this"
    if t == "super":
        return "super"
    if t == "field_access":
        obj = node.child_by_field_name("object")
        field = node.child_by_field_name("field")
        obj_text = _flatten_receiver_text(source, obj) if obj is not None else ""
        field_text = _node_text(source, field) if field is not None else ""
        if obj_text and field_text:
            return f"{obj_text}.{field_text}"
        if field_text:
            return field_text
        if obj_text:
            return obj_text
        # Fallback to raw text — preserves dotted shape verbatim.
        return _node_text(source, node).strip()
    if t == "parenthesized_expression":
        # Recurse into the first non-punctuation named child.
        for sub in node.children:
            if sub.type not in ("(", ")"):
                inner = _flatten_receiver_text(source, sub)
                if inner:
                    return inner
        return _node_text(source, node).strip()
    # Fallback: raw text. This covers method_invocation receivers,
    # cast_expression, array_access, object_creation_expression, etc.
    # Documented limitation; resolver won't bind, but edge is visible.
    return _node_text(source, node).strip()


def _strip_generics_to_head(source: bytes, type_node: Any) -> str:
    """Return the dotted head identifier of a constructor type expression.

    Reuses the same logic as 17b's `_extract_parent_name` — kept as a
    separate function for clarity at call sites (the 17b helper is
    written for `extends` / `implements` clauses which have a slightly
    different node-context contract).

    Handles:
      - `type_identifier`          → bare name (`"Foo"`)
      - `scoped_type_identifier`   → dotted form (`"com.foo.Bar"`)
      - `generic_type`             → recurse into head identifier,
                                     dropping `type_arguments`
                                     (`"ArrayList"` for `ArrayList<T>`)

    Returns `""` for unrecognised shapes — caller skips the edge.
    """
    return _extract_parent_name(source, type_node)


def _walk_calls(
    source: bytes,
    body_node: Any,
    local_names: set[str],
    module_qn: str,
) -> list[tuple[str, int]]:
    """Return [(callee_qn, line)] for every call site inside `body_node`.

    Dispatches by node.type:
      - `method_invocation`             → bare or member call
      - `object_creation_expression`    → constructor (`new Foo()`)
      - `explicit_constructor_invocation` → `super(…)` / `this(…)`
      - `method_reference`              → `Foo::bar` / `this::run`
                                          (low-priority; skip if shape
                                          is uncertain rather than crash)

    Walks ALL descendants (including lambda bodies and anonymous-class
    bodies) so calls inside them attribute to the enclosing method.
    Documented limitation; Sprint 17e introduces synthetic enclosing
    nodes for lambdas / anon classes.

    Local-name resolution: for bare identifier calls (no receiver),
    if the callee name is in `local_names` we prefix with
    `module_qn + "."`. This is a heuristic — works for the common
    case of "method on file's primary class calling a sibling method"
    (where module_qn is `pkg.PrimaryClass`). For nested-class methods
    calling siblings on their own (different) class, the resolver
    handles the disambiguation; the extractor stays conservative.
    """
    found: list[tuple[str, int]] = []

    def _visit(n: Any) -> None:
        t = n.type
        if t == "method_invocation":
            obj = n.child_by_field_name("object")
            name_node = n.child_by_field_name("name")
            if name_node is not None:
                name = _node_text(source, name_node)
                if obj is None:
                    # Bare call: `foo()`.
                    if name in local_names and module_qn:
                        callee = f"{module_qn}.{name}"
                    else:
                        callee = name
                    found.append((callee, n.start_point[0] + 1))
                else:
                    # Member call: `obj.bar()`, `this.bar()`,
                    # `Foo.staticBar()`, `obj.list.add(x)`.
                    receiver = _flatten_receiver_text(source, obj)
                    if receiver:
                        callee = f"{receiver}.{name}"
                    else:
                        callee = name
                    found.append((callee, n.start_point[0] + 1))
            # Recurse into arguments — they may contain nested calls
            # (`foo(bar(), baz())` → 3 edges).
            for child in n.children:
                _visit(child)
            return
        if t == "object_creation_expression":
            # `new Foo()` / `new Foo<T>(arg)` / `new com.foo.Bar()`.
            type_node = n.child_by_field_name("type")
            if type_node is not None:
                head = _strip_generics_to_head(source, type_node)
                if head:
                    found.append((head, n.start_point[0] + 1))
                else:
                    # Unrecognised type shape — fall back to raw text
                    # with generics naively stripped. Documented quirk.
                    raw = _node_text(source, type_node).strip()
                    if raw:
                        # Best-effort generics strip on a raw fallback.
                        if "<" in raw:
                            raw = raw.split("<", 1)[0]
                        found.append((raw, n.start_point[0] + 1))
            for child in n.children:
                _visit(child)
            return
        if t == "explicit_constructor_invocation":
            # `super(…)` / `this(…)` (NOT `super.foo(…)` — that's a
            # regular method_invocation with `object=super`).
            # Grammar shape: `constructor=super` / `constructor=this`
            # is the field name; in some grammar versions the keyword
            # appears as a positional child instead. Handle both.
            ctor_node = n.child_by_field_name("constructor")
            if ctor_node is not None:
                ctor_text = _node_text(source, ctor_node).strip()
            else:
                ctor_text = ""
                for sub in n.children:
                    if sub.type in ("super", "this"):
                        ctor_text = sub.type
                        break
            if ctor_text in ("super", "this"):
                found.append((ctor_text, n.start_point[0] + 1))
            for child in n.children:
                _visit(child)
            return
        if t == "method_reference":
            # `Foo::bar`, `this::run`, `String::length`, `Foo::new`.
            # NO field names — positional children:
            #   [receiver_or_type, "::", method_or_"new"]
            # Filter out punctuation; first non-`::` named child is the
            # receiver, last non-`::` named child is the method name.
            named_children = [
                c for c in n.children if c.type != "::"
            ]
            if len(named_children) >= 2:
                receiver_node = named_children[0]
                method_node = named_children[-1]
                # Receiver may be identifier / type_identifier /
                # scoped_type_identifier / generic_type / `super` /
                # `this` / field_access / etc.
                receiver = _flatten_receiver_text(source, receiver_node)
                if not receiver:
                    receiver = _node_text(source, receiver_node).strip()
                # Method-name node is usually `identifier`, but for
                # `Foo::new` the second child is the literal `new`
                # keyword (which tree-sitter exposes as an unnamed
                # token). Read it with raw text.
                method_text = _node_text(source, method_node).strip()
                if receiver and method_text:
                    found.append(
                        (f"{receiver}.{method_text}", n.start_point[0] + 1),
                    )
            # Don't recurse — method_reference has no nested calls.
            return
        # Default: recurse into children. Importantly this walks INTO
        # lambda_expression bodies, anonymous class bodies, switch
        # expressions, etc., so calls inside them attribute to the
        # enclosing method. Documented as a 17c limitation.
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)
    return found


def _collect_local_function_names(node: Any, source: bytes, out: set[str]) -> None:
    """Pre-pass: collect all method / constructor / class / interface /
    enum / record names declared anywhere in this file. Used by
    `_walk_calls` to decide whether a bare-identifier call should be
    prefixed with `module_qn`.

    Includes both class-name local resolution targets (for `new Foo()`
    where `Foo` is in this file) and method-name targets (for `bar()`
    where `bar` is a sibling method).

    Walks recursively so nested-class members are also captured.
    """
    for child in node.children:
        t = child.type
        if t in _TYPE_DECL_TYPES or t in _FUNCTION_DECL_TYPES:
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                out.add(_node_text(source, name_node))
        # Recurse: nested types live inside class_body / interface_body /
        # enum_body, and methods live inside those too.
        _collect_local_function_names(child, source, out)


# ─── 17d: import-edge extraction ──────────────────────────────────────────


def _walk_imports(
    source: bytes, root: Any,
) -> list[tuple[str, str, str]]:
    """Return [(target_qn, local_name, kind)] for every top-level import.

    Java has four import shapes; the AST distinguishes them positionally
    (no field names on `import_declaration`'s children):

      `import com.foo.Bar;`
        children: [`import` kw, scoped_identifier]
        → ("com.foo.Bar", "Bar", "symbol")

      `import com.foo.*;`
        children: [`import` kw, scoped_identifier, asterisk]
        → ("com.foo", "*", "module")

      `import static com.foo.Bar.baz;`
        children: [`import` kw, `static` kw, scoped_identifier]
        → ("com.foo.Bar.baz", "baz", "symbol")

      `import static com.foo.Bar.*;`
        children: [`import` kw, `static` kw, scoped_identifier, asterisk]
        → ("com.foo.Bar", "*", "symbol")

    The `static` keyword is an UNNAMED token child of
    `import_declaration` — `child_by_field_name` returns None for it.
    Detect it by scanning raw `node.children` for `child.type == "static"`.

    The dotted target is captured from the first named identifier-shaped
    child (`scoped_identifier` for ≥2 segments, plain `identifier` for
    pathological single-segment imports like `import com;`). We read its
    raw text — tree-sitter preserves the dotted form verbatim, no manual
    reassembly required.

    Wildcards: presence of an `asterisk` named child tags the import as
    a wildcard. local_name = "*". For non-static wildcards, kind becomes
    "module" (the target is a package); for static wildcards, kind stays
    "symbol" (the target is a type whose static members are imported).

    Single-segment defensive case: `import com;` (invalid Java but lexes
    cleanly) → emit ("com", "com", "symbol") so we don't crash and any
    downstream consumer sees something rather than nothing.
    """
    out: list[tuple[str, str, str]] = []
    for child in root.children:
        if child.type != "import_declaration":
            continue

        is_static = False
        target_node: Any = None
        is_wildcard = False
        for sub in child.children:
            t = sub.type
            if t == "static":
                is_static = True
            elif t in ("scoped_identifier", "identifier") and target_node is None:
                target_node = sub
            elif t == "asterisk":
                is_wildcard = True

        if target_node is None:
            # Malformed — `import ;` or similar. Skip defensively.
            continue
        target = _node_text(source, target_node).strip()
        if not target:
            continue

        if is_wildcard:
            # `import com.foo.*;`            → module import of `com.foo`.
            # `import static com.foo.Bar.*;` → symbol import of all
            #                                  static members of Bar.
            kind = "symbol" if is_static else "module"
            out.append((target, "*", kind))
            continue

        # Named (non-wildcard) import. Local name is the LAST dotted
        # segment of the target — for `import com.foo.Bar` that's "Bar";
        # for `import static com.foo.Bar.baz` it's "baz"; for the
        # pathological `import com;` (no dot at all) it's "com".
        local_name = target.rsplit(".", 1)[-1]
        out.append((target, local_name, "symbol"))

    return out


def extract_java_file(
    repo: RepoNode,
    rel_path: str,
    source: bytes,
    sha: str,
    parser: Any,
    repo_files: set[str] | None = None,
) -> IndexBatch:
    """Parse one .java file and return its IndexBatch fragment.

    `repo_files` is accepted for signature parity with `extract_cpp_file`
    but not currently consumed — Java import targets are dotted
    qualified names (not file paths), so cross-file resolution lives in
    the loader / resolver, not the extractor.
    """
    del repo_files  # Reserved for future use; resolver handles xref now.

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

    # 17d: emit ImportEdges for every `import` declaration. Position
    # mirrors python/cpp — imports are recorded BEFORE the recursive
    # body walk, so the resolver sees the file's import set as a
    # complete fact before its first call/inheritance lookup.
    for target_qn, local_name, kind in _walk_imports(source, root):
        batch.imports.append(ImportEdge(
            repo=repo.name,
            file_path=rel_path,
            target_qn=target_qn,
            local_name=local_name,
            kind=kind,
        ))

    # 17c: pre-pass to collect every method / constructor / class /
    # interface / enum / record name declared in this file. Used by
    # `_walk_calls` to decide whether a bare-identifier call should
    # be prefixed with `module_qn` for same-file resolution.
    local_names: set[str] = set()
    _collect_local_function_names(root, source, local_names)

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

        # 17c: walk the method body for call sites and emit CallEdges.
        # Abstract / interface methods have no body — `body=None` →
        # _walk_calls returns []. Constructors and compact constructors
        # always have a body when present. Static / instance initializer
        # blocks are NOT method declarations and never reach here, so
        # their calls are silently skipped (documented limitation).
        body_node = node.child_by_field_name("body")
        if body_node is not None:
            for callee_qn, line in _walk_calls(
                source, body_node, local_names, module_qn,
            ):
                batch.calls.append(CallEdge(
                    repo=repo.name,
                    caller_qn=fn_qn,
                    callee_qn=callee_qn,
                    line=line,
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
