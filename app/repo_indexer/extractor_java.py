"""Tree-sitter Java extractor — Sprint 17a / 17b / 17c / 17d / 17e.

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

Call-edge limitations (documented for 17f follow-up):
    - Static / instance initializer blocks (`static { … }` / `{ … }`)
      and field initializer expressions (`= new Foo()`) have no
      enclosing FunctionNode, so calls inside them are SKIPPED for
      17c/17e. Future sprint may model these as synthetic
      `__cinit__` / `__init__`-style nodes.
    - No type-binding: `User u = new User(); u.foo()` emits
      callee_qn=`u.foo`, not `User.foo`. Receiver-type inference
      is a future-sprint concern.
    - Annotation arguments that look like calls
      (`@SuppressWarnings(value = "x")`) are NOT `method_invocation`
      nodes — no special handling needed.

Overload disambiguation (17e contribution):
    - When ≥2 methods within the SAME class share a (parent_class_qn,
      bare-name), each gets a `:type1,type2,...` suffix appended to
      its qn — using SOURCE order of params (NOT alphabetical), since
      `add(int, String)` and `add(String, int)` are distinct Java
      methods. Singletons (only one method with that name in the
      class) keep the simple un-suffixed qn.
    - The suffix is derived from the OUTER bare type name only:
      `process(List<User>)` → `:List`, `process(Map<String, Foo>)` →
      `:Map`. Trailing `[]` array brackets are stripped. The full
      generic-bearing text stays in `param_types` (per the
      "captured as observed" contract).
    - Generics-in-overloads collision is ACCEPTED: `add(List<Integer>)`
      and `add(List<String>)` both reduce to `:List` and collapse —
      they're indistinguishable at JVM bytecode level (erasure)
      anyway. Documented limitation.
    - CallEdges emit raw target names (`"add"` or `"this.add"`) and
      do NOT carry the overload suffix; the resolver does fuzzy
      arity-matching at call-site lookup time. This means a sibling
      bare call to `add` in the same file matches the resolver's
      candidate set rather than a single suffixed FunctionNode qn.

Synthetic FunctionNodes / ClassNodes (17e contribution):
    - Anonymous classes (`new Runnable() { … }`) emit a synthetic
      ClassNode with name `__anon_<line>` (or `__anon_<line>_<col>`
      for the multiple-anon-on-one-line case). Anonymous classes
      attach to the enclosing TYPE (NOT the enclosing method) — so
      `class Outer { void run() { new R() {…}; } }` produces
      ClassNode qn `pkg.Outer.__anon_<N>`, not `pkg.Outer.run.__anon_<N>`.
      Methods declared inside the anonymous body emit FunctionNodes
      attached to the synthetic class.
    - Lambdas (`() -> doFoo()`, `x -> log(x)`) emit a synthetic
      FunctionNode with name `__lambda_<line>` (or `__lambda_<line>_<col>`
      for the multiple-lambdas-on-one-line case). Lambdas attach to
      the ENCLOSING METHOD (different from anonymous classes — they
      have no class parent in this model: `is_method=False`,
      `parent_class_qn=""`). A lambda's qn is
      `<enclosing_method_qn>.__lambda_<line>`.
    - Calls inside a lambda body re-attribute to the lambda's qn
      (NOT the enclosing method). Same for calls inside anonymous-
      class method bodies — they re-attribute to the anon-class
      method's qn. The walker re-enters at lambda / anonymous-class
      boundaries with a new caller_qn for descendants.
    - Lambda parameters are captured normally. Inferred-type forms
      (`(x, y) -> …`, `x -> …`) emit param_types entries with
      empty type-text strings — the position is preserved, the type
      is unknown.

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

Out of 17a/17b/17c/17d/17e scope (LATER sub-sprints — explicitly NOT handled):
    - Spring routes domain extraction — Sprint 17f.
    - Receiver-type binding (`User u = new User(); u.foo()` resolving
      to `User.foo`) — future sprint after typeBindings for Java.
    - Annotation-processor synthesis (Lombok `@Data` getters, JPA
      repo methods) — explicitly not modelled.
    - Generic erasure → typed call resolution — out per plan.
    - `permits` clause for sealed types — out per plan.

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

import dataclasses
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


def _strip_generics_for_overload_suffix(type_text: str) -> str:
    """Reduce a Java type expression to its outer bare-type identifier.

    Used by the overload-suffix computation (17e). Examples:
        "int"                    → "int"
        "String"                 → "String"
        "List<User>"             → "List"
        "Map<String, List<Foo>>" → "Map"
        "T"                      → "T"
        "String[]"               → "String"
        "int..."                 → "int"
        "List<User>[]"           → "List"
        "com.foo.Bar"            → "com.foo.Bar"
        "com.foo.Bar<T>"         → "com.foo.Bar"

    Algorithm: truncate at the first `<`, then strip trailing `[]`
    pairs and the vararg `...` marker. Whitespace stripped at the
    edges. Single-uppercase generic type-vars (`T`, `K`, `V`) pass
    through unchanged because they have no `<` and no `[]` to strip.
    """
    if not type_text:
        return ""
    s = type_text.strip()
    if "<" in s:
        s = s.split("<", 1)[0]
    s = s.strip()
    # Strip trailing `...` (varargs) once — `String...` → `String`.
    if s.endswith("..."):
        s = s[:-3].rstrip()
    # Strip any trailing `[]` array brackets — `String[][]` → `String`.
    while s.endswith("[]"):
        s = s[:-2].rstrip()
    return s


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


def _is_anonymous_class_creation(node: Any) -> Any:
    """If `node` is an `object_creation_expression` with an anonymous
    `class_body` child, return that class_body node. Otherwise None.

    Anonymous class shape (tree-sitter-java 0.23.x):
        object_creation_expression
          ├── type:      type expression (e.g. `Runnable`)
          ├── arguments: argument_list
          └── class_body  ← extra named child only present when anon

    The class_body has no field name — we scan named children.
    """
    if node is None or node.type != "object_creation_expression":
        return None
    for child in node.children:
        if child.type == "class_body":
            return child
    return None


def _extract_lambda_params(
    source: bytes, lambda_node: Any,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return (param_names, param_types) for a `lambda_expression` node.

    Three shapes for the `parameters=` field:
      - `inferred_parameters`  →  `(x, y) -> …`  — list of identifier
                                  children. type_text = "" (inferred).
      - `formal_parameters`    →  `(int x, int y) -> …` — same shape as
                                  a method's params; reuse
                                  `_extract_param_names_and_types`.
      - `identifier`           →  `x -> …` — single bare param. No type.

    Other (anomalous) shapes return empty tuples defensively — the
    lambda's FunctionNode is still emitted, just with no params.
    """
    params = lambda_node.child_by_field_name("parameters")
    if params is None:
        return (), ()
    t = params.type
    if t == "formal_parameters":
        return _extract_param_names_and_types(source, params)
    if t == "inferred_parameters":
        names: list[str] = []
        for c in params.children:
            if c.type == "identifier":
                names.append(_node_text(source, c))
        # No type info on inferred params; emit names with empty type.
        types = tuple((n, "") for n in names)
        return tuple(names), types
    if t == "identifier":
        n = _node_text(source, params)
        return (n,), ()
    return (), ()


def _walk_calls(
    source: bytes,
    body_node: Any,
    local_names: set[str],
    module_qn: str,
    on_call: Any,
    on_lambda: Any,
    on_anon_class: Any,
) -> None:
    """Walk `body_node` collecting call sites, INVOKING callbacks at
    lambda / anonymous-class boundaries so the orchestrator can attribute
    descendants under a fresh caller_qn.

    Callbacks:
      on_call(callee_qn: str, line: int)
        Invoked for every call site (method_invocation,
        object_creation_expression that's NOT anonymous,
        explicit_constructor_invocation, method_reference) under the
        current caller scope.

      on_lambda(lambda_node)
        Invoked when the walker encounters a `lambda_expression`. The
        callback is responsible for emitting the synthetic FunctionNode
        and re-entering the walk on the lambda's body with a new
        caller_qn. The walker DOES NOT recurse into the lambda's body
        itself — boundary respected.

      on_anon_class(object_creation_node, class_body_node)
        Invoked when the walker encounters an `object_creation_expression`
        with an anonymous class_body child. The callback emits the
        synthetic ClassNode, recurses into its method declarations
        (each gets its own FunctionNode with its own _walk_calls scope),
        and DOES NOT cross back into the outer call scope. Note: the
        constructor call itself (the `new Runnable()` part) is also
        recorded via on_call so the enclosing method's CALLS edge is
        preserved.

    The walker handles four call flavours:
      - `method_invocation`             → bare or member call
      - `object_creation_expression`    → constructor (`new Foo()`); skip
                                          recursion into arguments when
                                          the creation is anonymous (the
                                          on_anon_class handler walks the
                                          arguments separately if needed)
      - `explicit_constructor_invocation` → `super(…)` / `this(…)`
      - `method_reference`              → `Foo::bar` / `this::run`

    Local-name resolution: for bare identifier calls (no receiver), if
    the callee name is in `local_names` we prefix with `module_qn + "."`.
    """

    def _visit(n: Any) -> None:
        t = n.type
        if t == "lambda_expression":
            # Hand off to the lambda emitter. Walker boundary — do NOT
            # recurse into the lambda's body or parameters from here;
            # the on_lambda callback will re-enter with a new caller_qn.
            on_lambda(n)
            return
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
                    on_call(callee, n.start_point[0] + 1)
                else:
                    # Member call: `obj.bar()`, `this.bar()`,
                    # `Foo.staticBar()`, `obj.list.add(x)`.
                    receiver = _flatten_receiver_text(source, obj)
                    if receiver:
                        callee = f"{receiver}.{name}"
                    else:
                        callee = name
                    on_call(callee, n.start_point[0] + 1)
            # Recurse into arguments — they may contain nested calls
            # (`foo(bar(), baz())` → 3 edges) AND lambdas
            # (`list.forEach(x -> log(x))` — the lambda is an argument).
            for child in n.children:
                _visit(child)
            return
        if t == "object_creation_expression":
            # `new Foo()` / `new Foo<T>(arg)` / `new com.foo.Bar()`.
            # First, record the constructor call itself (regardless of
            # whether it's an anonymous-class creation or a plain `new`).
            type_node = n.child_by_field_name("type")
            if type_node is not None:
                head = _strip_generics_to_head(source, type_node)
                if head:
                    on_call(head, n.start_point[0] + 1)
                else:
                    # Unrecognised type shape — fall back to raw text
                    # with generics naively stripped. Documented quirk.
                    raw = _node_text(source, type_node).strip()
                    if raw:
                        if "<" in raw:
                            raw = raw.split("<", 1)[0]
                        on_call(raw, n.start_point[0] + 1)
            # Anonymous-class detection. If present, hand the body off
            # to the synthetic-class emitter and DO NOT recurse into
            # its members from here. We DO still recurse into the
            # constructor's argument_list (calls like
            # `new Foo(bar()) { … }` should record `bar` under the
            # outer caller).
            anon_body = _is_anonymous_class_creation(n)
            if anon_body is not None:
                on_anon_class(n, anon_body)
                # Recurse only into argument-list children; skip the
                # class_body to avoid double-attribution.
                for child in n.children:
                    if child is anon_body:
                        continue
                    _visit(child)
                return
            for child in n.children:
                _visit(child)
            return
        if t == "explicit_constructor_invocation":
            # `super(…)` / `this(…)` (NOT `super.foo(…)` — that's a
            # regular method_invocation with `object=super`).
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
                on_call(ctor_text, n.start_point[0] + 1)
            for child in n.children:
                _visit(child)
            return
        if t == "method_reference":
            # `Foo::bar`, `this::run`, `String::length`, `Foo::new`.
            named_children = [
                c for c in n.children if c.type != "::"
            ]
            if len(named_children) >= 2:
                receiver_node = named_children[0]
                method_node = named_children[-1]
                receiver = _flatten_receiver_text(source, receiver_node)
                if not receiver:
                    receiver = _node_text(source, receiver_node).strip()
                method_text = _node_text(source, method_node).strip()
                if receiver and method_text:
                    on_call(
                        f"{receiver}.{method_text}", n.start_point[0] + 1,
                    )
            # Don't recurse — method_reference has no nested calls.
            return
        # Default: recurse into children. Lambda / anonymous-class
        # boundaries are caught by the dedicated branches above; other
        # control-flow nodes (if/while/switch) fall through here.
        for child in n.children:
            _visit(child)

    if body_node is not None:
        _visit(body_node)


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


def _disambiguate_overloads(batch: IndexBatch) -> None:
    """Mutate `batch.functions` in place: append `:type1,type2,...` suffix
    to the qns of methods whose (parent_class_qn, name) group has more
    than one member.

    Param-type list uses SOURCE order (NOT alphabetical). `add(int, String)`
    and `add(String, int)` are distinct Java overloads that must produce
    distinct qns; sorting would collapse them.

    Singletons (only one method with the (parent_class_qn, name) key) are
    left untouched — their qn stays in its bare 17a form. This mirrors
    how the cpp extractor only adds `:const` when needed (Sprint 16a).

    Methods with empty `parent_class_qn` (synthetic lambdas, free
    functions in the default package — both rare) are NOT subjected to
    overload disambiguation. Lambdas already carry `__lambda_<line>`
    line-disambiguation in their name; free functions outside a class
    don't have meaningful overload semantics in this model.

    Implementation note: FunctionNode is `@dataclass(frozen=True)`,
    so we use `dataclasses.replace` to produce updated copies and
    overwrite the list.
    """
    # Group FunctionNode indices by (parent_class_qn, name). Skip
    # entries with empty parent_class_qn (lambdas, free functions).
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, fn in enumerate(batch.functions):
        if not fn.parent_class_qn:
            continue
        key = (fn.parent_class_qn, fn.name)
        groups.setdefault(key, []).append(idx)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        for idx in indices:
            fn = batch.functions[idx]
            # Build comma-joined outer-bare-type list in SOURCE order.
            # `param_types` carries (name, type_text) tuples in source
            # order; map each through _strip_generics_for_overload_suffix.
            type_parts: list[str] = []
            for _pname, type_text in fn.param_types:
                stripped = _strip_generics_for_overload_suffix(type_text)
                type_parts.append(stripped)
            # Methods with no params still need disambiguation if their
            # group has overloads — append an empty suffix `:` so all
            # zero-param overloads share one identical qn (which is
            # the correct behaviour: there can be at most ONE no-arg
            # method per name per class). The frozen-empty case is
            # rare (would require source-level error) but defensive.
            suffix = ",".join(type_parts)
            new_qn = f"{fn.qualified_name}:{suffix}"
            batch.functions[idx] = dataclasses.replace(
                fn, qualified_name=new_qn,
            )


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

    # 17e: synthetic-name collision tracker for lambdas + anon classes.
    # Multiple lambdas / anon classes on the same line need a `_<col>`
    # suffix to stay distinct. Key: ("lambda" | "anon", line). Value:
    # set of column offsets seen at that key — second-and-later entries
    # always get the col suffix; first entries also get the col suffix
    # if/when a collision is detected at emit-time. The simpler rule:
    # always include col when MORE THAN ONE entity sits at the same
    # (kind, line). To implement deterministically without two passes,
    # we count occurrences first then emit names.
    line_kind_counts: dict[tuple[str, int], int] = {}

    def _count_synthetic_names(n: Any) -> None:
        """Pre-pass: tally per-line counts of lambdas + anonymous-class
        creations so the emitter can decide whether to add `_<col>`
        disambiguation suffixes. Walks the entire root.
        """
        t = n.type
        if t == "lambda_expression":
            key = ("lambda", n.start_point[0] + 1)
            line_kind_counts[key] = line_kind_counts.get(key, 0) + 1
        elif t == "object_creation_expression":
            if _is_anonymous_class_creation(n) is not None:
                key = ("anon", n.start_point[0] + 1)
                line_kind_counts[key] = line_kind_counts.get(key, 0) + 1
        for child in n.children:
            _count_synthetic_names(child)

    _count_synthetic_names(root)

    def _synthetic_name(kind: str, line: int, col: int) -> str:
        """Build a synthetic name for a lambda / anon class.

        kind: "lambda" or "anon". line + col are 1-based.
        Returns `__lambda_<line>` / `__anon_<line>` for singletons,
        or `__lambda_<line>_<col>` / `__anon_<line>_<col>` when more
        than one entity of this kind sits on the same line.
        """
        prefix = "__lambda" if kind == "lambda" else "__anon"
        if line_kind_counts.get((kind, line), 0) > 1:
            return f"{prefix}_{line}_{col}"
        return f"{prefix}_{line}"

    def _emit_calls_for_caller(
        body: Any, caller_qn: str, enclosing_class_qn: str,
    ) -> None:
        """Walk `body` collecting CallEdges under `caller_qn`. Honors
        lambda / anonymous-class boundaries: descendants inside those
        attribute to fresh synthetic FunctionNode caller_qns.

        `enclosing_class_qn` is the class context for the CURRENT caller
        — used so that anonymous-class creations inside the caller's
        body anchor the synthetic anon ClassNode under the correct
        enclosing class. (Anonymous classes attach to the enclosing
        TYPE, not the enclosing method.)
        """
        if body is None:
            return

        def _on_call(callee_qn: str, line: int) -> None:
            batch.calls.append(CallEdge(
                repo=repo.name,
                caller_qn=caller_qn,
                callee_qn=callee_qn,
                line=line,
            ))

        def _on_lambda(lambda_node: Any) -> None:
            line = lambda_node.start_point[0] + 1
            col = lambda_node.start_point[1] + 1
            lname = _synthetic_name("lambda", line, col)
            # Lambdas attach to the enclosing METHOD, not the class.
            # qn = <enclosing_method_qn>.__lambda_<line>
            lqn = f"{caller_qn}.{lname}"
            param_names, param_types = _extract_lambda_params(source, lambda_node)
            line_end = lambda_node.end_point[0] + 1
            batch.functions.append(FunctionNode(
                repo=repo.name,
                qualified_name=lqn,
                name=lname,
                file_path=rel_path,
                line_start=line,
                line_end=line_end,
                is_async=False,
                is_method=False,
                parent_class_qn="",
                params=param_names,
                param_types=param_types,
                return_type_raw="",
                docstring="",
                annotations=(),
            ))
            # Re-enter the call walker on the lambda's body with the
            # lambda's qn as the new caller. The body= field can be an
            # expression or a block; both walk fine.
            lbody = lambda_node.child_by_field_name("body")
            _emit_calls_for_caller(lbody, lqn, enclosing_class_qn)

        def _on_anon_class(oce_node: Any, class_body: Any) -> None:
            line = oce_node.start_point[0] + 1
            col = oce_node.start_point[1] + 1
            aname = _synthetic_name("anon", line, col)
            # Anon classes attach to the enclosing CLASS (not method).
            # If we're at file scope (no enclosing class, e.g. a lambda
            # at top-level — impossible in valid Java but defensive),
            # fall back to module qn.
            if enclosing_class_qn:
                anon_qn = f"{enclosing_class_qn}.{aname}"
            elif module_qn:
                anon_qn = f"{module_qn}.{aname}"
            else:
                anon_qn = aname
            line_end = oce_node.end_point[0] + 1
            batch.classes.append(ClassNode(
                repo=repo.name,
                qualified_name=anon_qn,
                name=aname,
                file_path=rel_path,
                line_start=line,
                line_end=line_end,
                docstring="",
                annotations=(),
            ))
            # Walk the anonymous class body for method declarations.
            # Each method emits its own FunctionNode + walks its own
            # body for calls (with its own caller_qn). This re-uses
            # the standard _emit_function path with the anon class's
            # qn pushed onto the stack.
            _walk(class_body, [anon_qn])

        _walk_calls(
            source, body, local_names, module_qn,
            on_call=_on_call,
            on_lambda=_on_lambda,
            on_anon_class=_on_anon_class,
        )

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

        Overload disambiguation (17e) is applied as a post-pass over
        `batch.functions` once the whole file is walked — see
        `_disambiguate_overloads`.
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

        # 17c/17e: walk the method body for call sites and emit
        # CallEdges. Abstract / interface methods have no body —
        # `body=None` → _emit_calls_for_caller is a no-op. Constructors
        # and compact constructors always have a body when present.
        # Static / instance initializer blocks are NOT method
        # declarations and never reach here, so their calls are
        # silently skipped (documented limitation).
        body_node = node.child_by_field_name("body")
        _emit_calls_for_caller(body_node, fn_qn, parent_class_qn)

    def _walk(node: Any, parent_class_qn_stack: list[str]) -> None:
        """Recurse into `node`'s children. Type declarations recurse via
        `_emit_class` (which calls `_walk` on the body); methods /
        constructors are emitted in place. Other nodes recurse so we find
        nested types inside method bodies (legal in Java — local
        classes — though rare).

        17e: We DO NOT recurse into `lambda_expression` or anonymous
        `object_creation_expression` from the type/method walker; their
        contents are handled by `_emit_calls_for_caller` (which
        re-attributes calls + emits synthetic FunctionNodes / ClassNodes).
        Skipping them here prevents nested types declared inside a lambda
        body from being emitted twice.
        """
        for child in node.children:
            t = child.type
            if t in _TYPE_DECL_TYPES:
                _emit_class(child, parent_class_qn_stack)
            elif t in _FUNCTION_DECL_TYPES:
                _emit_function(child, parent_class_qn_stack)
            elif t == "lambda_expression":
                # Lambdas are handled by the caller-tracking call walker.
                continue
            elif t == "object_creation_expression" and (
                _is_anonymous_class_creation(child) is not None
            ):
                # Anonymous classes are handled by the call walker too.
                continue
            else:
                # Recurse into anything else so we find nested
                # type_declarations inside method bodies / enum body
                # declarations / etc. The named-only iteration below
                # would also work, but `children` matches what the cpp
                # extractor does and is simpler to reason about.
                _walk(child, parent_class_qn_stack)

    _walk(root, [])

    # 17e: post-pass — append `:type1,type2,...` qn suffix to overload
    # groups (≥2 methods on the same class with the same name).
    _disambiguate_overloads(batch)

    return batch


__all__ = ["extract_java_file"]
