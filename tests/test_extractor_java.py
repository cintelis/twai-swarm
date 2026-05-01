"""Sprint 17a — Java extractor tests.

Synthetic source-only tests; no real codebase scanned. Mirrors the
`tests/test_extractor_cpp.py` pattern.

Coverage targets (all 17a-scope; out-of-scope items deferred):
  - simple class + methods + constructor
  - package_declaration drives module qn (NOT path)
  - default-package fallback (no `package` line) → filename stem
  - nested classes embed in qn
  - annotation capture as raw strings on Class + Function
  - interface extraction
  - enum extraction (constants suppressed; methods captured)
  - record extraction (compact constructor as FunctionNode)
  - varargs param shape (`String...`)
  - inner-class isolation (qns don't collide across siblings)
"""
from __future__ import annotations

import pytest

try:
    import tree_sitter_java as _tsjava  # noqa: F401
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False


from app.repo_indexer.actions import RepoNode  # noqa: E402
from app.repo_indexer.extractor_java import (  # noqa: E402
    _extract_package,
    _filename_stem,
    _module_qn_from_package,
    extract_java_file,
)

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-java not installed")
    import tree_sitter_java as tsjava
    return Parser(Language(tsjava.language()))


# ─── helpers ────────────────────────────────────────────────────────────────

def _scan(parser, rel_path: str, src: bytes):
    return extract_java_file(REPO, rel_path, src, "sha", parser, repo_files=set())


# ─── module-qn pure-function tests ──────────────────────────────────────────

def test_filename_stem_strips_dirs_and_ext():
    assert _filename_stem("src/main/java/com/foo/Bar.java") == "Bar"
    assert _filename_stem("Foo.java") == "Foo"
    assert _filename_stem("package-info.java") == "package-info"


def test_module_qn_from_package_combines_or_falls_back():
    assert _module_qn_from_package("com.bench", "src/com/bench/Foo.java") == "com.bench.Foo"
    assert _module_qn_from_package("", "Foo.java") == "Foo"
    assert _module_qn_from_package("com.x", "package-info.java") == "com.x.package-info"


# ─── 1. simple class + methods + constructor ───────────────────────────────

def test_simple_class_and_methods(parser):
    """Constructor naming convention: bare class identifier (matches
    Java source view of constructors-as-named-methods). So
    `public Calculator()` → FunctionNode qn `…Calculator.Calculator`.
    """
    src = (
        b"package com.bench;\n"
        b"public class Calculator {\n"
        b"    public Calculator() {}\n"
        b"    public int add(int a, int b) { return a + b; }\n"
        b"    public int subtract(int a, int b) { return a - b; }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/com/bench/Calculator.java", src)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"com.bench.Calculator"}
    fn_qns = {f.qualified_name for f in batch.functions}
    assert "com.bench.Calculator.add" in fn_qns
    assert "com.bench.Calculator.subtract" in fn_qns
    assert "com.bench.Calculator.Calculator" in fn_qns


# ─── 2. package_declaration drives module qn (not path) ────────────────────

def test_package_declaration_drives_module_qn(parser):
    """Path can be misleading — the `package` declaration wins."""
    src = (
        b"package com.right;\n"
        b"public class Foo {}\n"
    )
    # File at a deliberately misleading path.
    batch = _scan(parser, "src/main/java/wrong/path/Foo.java", src)
    mod_qns = {m.qualified_name for m in batch.modules}
    assert mod_qns == {"com.right.Foo"}
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"com.right.Foo"}
    # File node has the package recorded for the import resolver (17d).
    assert batch.files[0].package == "com.right"


# ─── 3. default-package fallback (no `package` declaration) ────────────────

def test_no_package_declaration_falls_back_to_filename_stem(parser):
    src = b"public class Foo { public void hello() {} }\n"
    batch = _scan(parser, "Foo.java", src)
    mod_qns = {m.qualified_name for m in batch.modules}
    assert mod_qns == {"Foo"}
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"Foo"}
    # Method has no package prefix — bare class qn is the parent.
    fn_qns = {f.qualified_name for f in batch.functions}
    assert "Foo.hello" in fn_qns
    assert batch.files[0].package == ""


# ─── 4. nested classes embed in qn ─────────────────────────────────────────

def test_nested_classes_embed_in_qn(parser):
    src = (
        b"package pkg;\n"
        b"class Outer {\n"
        b"    class Inner { void foo() {} }\n"
        b"    static class Static { void bar() {} }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Outer.java", src)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"pkg.Outer", "pkg.Outer.Inner", "pkg.Outer.Static"}
    fn_qns = {f.qualified_name for f in batch.functions}
    assert "pkg.Outer.Inner.foo" in fn_qns
    assert "pkg.Outer.Static.bar" in fn_qns


# ─── 5. annotation capture on class + method ───────────────────────────────

def test_annotations_captured_on_class_and_method(parser):
    src = (
        b"package com.api;\n"
        b"@RestController\n"
        b"@RequestMapping(\"/api\")\n"
        b"public class Foo {\n"
        b"    @GetMapping(\"/x\")\n"
        b"    public String x() { return \"\"; }\n"
        b"    @Override\n"
        b"    public String toString() { return \"\"; }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/com/api/Foo.java", src)
    foo = next(c for c in batch.classes if c.name == "Foo")
    assert "@RestController" in foo.annotations
    assert '@RequestMapping("/api")' in foo.annotations

    x_fn = next(f for f in batch.functions if f.name == "x")
    assert '@GetMapping("/x")' in x_fn.annotations

    to_string = next(f for f in batch.functions if f.name == "toString")
    assert "@Override" in to_string.annotations


# ─── 6. interface extraction ───────────────────────────────────────────────

def test_interface_extraction(parser):
    """Interfaces emit ClassNodes (the schema doesn't separately model
    interfaces in 17a). Abstract methods (no body) still emit
    FunctionNodes — `is_abstract` isn't modelled here.
    """
    src = (
        b"package geo;\n"
        b"public interface Shape {\n"
        b"    double area();\n"
        b"}\n"
    )
    batch = _scan(parser, "src/geo/Shape.java", src)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"geo.Shape"}
    fn_qns = {f.qualified_name for f in batch.functions}
    assert "geo.Shape.area" in fn_qns


# ─── 7. enum extraction (constants suppressed; methods captured) ───────────

def test_enum_extraction(parser):
    """Enum constants (RED/GREEN/BLUE) are values, NOT FunctionNodes.
    Enum methods (e.g. `hex()`) are captured normally.
    """
    src = (
        b"package paint;\n"
        b"public enum Color {\n"
        b"    RED, GREEN, BLUE;\n"
        b"    public String hex() { return \"\"; }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/paint/Color.java", src)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"paint.Color"}
    fn_names = {f.name for f in batch.functions}
    assert "hex" in fn_names
    # Enum constants must NOT become FunctionNodes.
    assert "RED" not in fn_names
    assert "GREEN" not in fn_names
    assert "BLUE" not in fn_names


# ─── 8. record extraction (compact constructor as FunctionNode) ───────────

def test_record_extraction(parser):
    """Records emit a ClassNode for `Point`, plus a FunctionNode for
    the compact constructor (`public Point { … }`). The compact
    constructor has no `formal_parameters` of its own — params are
    inherited from the record header (`(int x, int y)`).

    Implicit accessor methods (`x()`, `y()`) are NOT captured —
    documented limitation; the JVM synthesises them, source has none.
    """
    src = (
        b"package geo;\n"
        b"public record Point(int x, int y) {\n"
        b"    public Point { }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/geo/Point.java", src)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert cls_qns == {"geo.Point"}
    point_fns = [f for f in batch.functions if f.name == "Point"]
    assert len(point_fns) == 1
    ctor = point_fns[0]
    # Compact constructor inherits the record-header params.
    assert ctor.params == ("x", "y")
    # Documented limitation: no implicit accessors emitted.
    fn_names = {f.name for f in batch.functions}
    assert "x" not in fn_names
    assert "y" not in fn_names


# ─── 9. varargs method ─────────────────────────────────────────────────────

def test_varargs_method(parser):
    """`void log(String... args)` → params=('args',), type captured
    with the trailing `...` so consumers can distinguish varargs from
    arrays.
    """
    src = (
        b"package log;\n"
        b"public class Logger {\n"
        b"    public void log(String... args) {}\n"
        b"}\n"
    )
    batch = _scan(parser, "src/log/Logger.java", src)
    log_fn = next(f for f in batch.functions if f.name == "log")
    assert log_fn.params == ("args",)
    type_map = dict(log_fn.param_types)
    # We document the varargs shape as `String...` (raw type + `...`
    # suffix) so consumers can distinguish from `String[]`.
    assert type_map["args"] == "String..."


# ─── 10. inner-class isolation (no qn collisions across siblings) ─────────

def test_inner_class_isolation(parser):
    """Two top-level classes in the same file each have a `foo()`
    method. Their qns must NOT collide — `pkg.A.foo` and `pkg.B.foo`.
    """
    src = (
        b"package pkg;\n"
        b"class A { void foo() {} }\n"
        b"class B { void foo() {} }\n"
    )
    batch = _scan(parser, "src/pkg/Multi.java", src)
    fn_qns = {f.qualified_name for f in batch.functions}
    assert "pkg.A.foo" in fn_qns
    assert "pkg.B.foo" in fn_qns
    assert len([f for f in batch.functions if f.name == "foo"]) == 2


# ─── 11. _extract_package edge cases ───────────────────────────────────────

def test_extract_package_with_scoped_identifier(parser):
    src = b"package com.foo.bar;\nclass X {}\n"
    tree = parser.parse(src)
    assert _extract_package(src, tree.root_node) == "com.foo.bar"


def test_extract_package_returns_empty_for_default_package(parser):
    src = b"class X {}\n"
    tree = parser.parse(src)
    assert _extract_package(src, tree.root_node) == ""


# ─── 12. walker integration (extension wiring) ────────────────────────────

def test_walker_routes_java_extension():
    from app.repo_indexer.walker import EXT_LANGUAGE
    assert EXT_LANGUAGE[".java"] == "java"


# ─── 13. package-info / module-info still emit File + Module ──────────────

def test_package_info_emits_module_with_no_classes(parser):
    """`package-info.java` is package-level annotations only — no class
    declaration. We still want a FileNode + ModuleNode so the file
    appears in repo browsing.
    """
    src = (
        b"@Deprecated\n"
        b"package com.legacy;\n"
    )
    batch = _scan(parser, "src/com/legacy/package-info.java", src)
    mod_qns = {m.qualified_name for m in batch.modules}
    assert mod_qns == {"com.legacy.package-info"}
    assert len(batch.classes) == 0
    assert batch.files[0].package == "com.legacy"


# ─── 17b: Inheritance edges (extends + implements) ────────────────────────


def test_class_extends_single(parser):
    """`class Cat extends Animal` → one InheritsEdge with
    child_qn=`zoo.Cat` and parent_qn=`Animal`. Parent is unresolved
    (no `Animal` class in this file); the resolver maps it later.
    """
    src = (
        b"package zoo;\n"
        b"public class Cat extends Animal {}\n"
    )
    batch = _scan(parser, "src/zoo/Cat.java", src)
    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "zoo.Cat"
    assert edge.parent_qn == "Animal"


def test_class_implements_multiple(parser):
    """`class Cat extends Animal implements Cuddly, Trainable` →
    1 extends + 2 implements = 3 InheritsEdges, one per parent.
    Source-order preserved (extends first, then interfaces in declared
    order).
    """
    src = (
        b"package zoo;\n"
        b"public class Cat extends Animal implements Cuddly, Trainable {}\n"
    )
    batch = _scan(parser, "src/zoo/Cat.java", src)
    parents = [(e.child_qn, e.parent_qn) for e in batch.inherits]
    assert ("zoo.Cat", "Animal") in parents
    assert ("zoo.Cat", "Cuddly") in parents
    assert ("zoo.Cat", "Trainable") in parents
    assert len(parents) == 3


def test_interface_extends_multiple(parser):
    """`interface Pet extends Cuddly, Trainable` — interfaces support
    multiple `extends` parents (unlike classes). Both edges emitted.
    Tree-sitter-java exposes these via `extends_interfaces` (no field
    name on that child — must walk named children manually).
    """
    src = (
        b"package zoo;\n"
        b"public interface Pet extends Cuddly, Trainable {}\n"
    )
    batch = _scan(parser, "src/zoo/Pet.java", src)
    parents = {(e.child_qn, e.parent_qn) for e in batch.inherits}
    assert parents == {("zoo.Pet", "Cuddly"), ("zoo.Pet", "Trainable")}


def test_generic_parent_strips_args(parser):
    """`class List extends ArrayList<String>` → parent_qn = `"ArrayList"`,
    NOT `"ArrayList<String>"`. The grammar wraps the parent in a
    `generic_type` node; we recurse to its head identifier and drop
    the `type_arguments`. Same rule for `implements Comparable<Foo>`.
    """
    src = (
        b"package col;\n"
        b"public class List extends ArrayList<String> implements Comparable<List> {}\n"
    )
    batch = _scan(parser, "src/col/List.java", src)
    parents = {(e.child_qn, e.parent_qn) for e in batch.inherits}
    assert ("col.List", "ArrayList") in parents
    assert ("col.List", "Comparable") in parents
    # Make sure we DIDN'T leak the generic args into the parent_qn.
    for _, parent in parents:
        assert "<" not in parent
        assert ">" not in parent


def test_scoped_parent_preserved(parser):
    """`class Foo extends com.bench.Bar` → parent_qn = `"com.bench.Bar"`
    (dotted form preserved verbatim). The resolver maps the dotted
    name to a ClassNode later; if it can't, the loader keeps it as a
    Symbol node — both behaviours are correct for graph queries.
    """
    src = (
        b"package app;\n"
        b"public class Foo extends com.bench.Bar {}\n"
    )
    batch = _scan(parser, "src/app/Foo.java", src)
    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "app.Foo"
    assert edge.parent_qn == "com.bench.Bar"


def test_diamond_inheritance(parser):
    """Diamond shape across interfaces and a class:
        interface A {}
        interface B extends A {}
        interface C extends A {}
        class D implements B, C {}
    Expected: 4 InheritsEdges (B→A, C→A, D→B, D→C). A has no parents
    (we don't model implicit `java.lang.Object`).
    """
    src = (
        b"package dia;\n"
        b"interface A {}\n"
        b"interface B extends A {}\n"
        b"interface C extends A {}\n"
        b"class D implements B, C {}\n"
    )
    batch = _scan(parser, "src/dia/Diamond.java", src)
    edges = {(e.child_qn, e.parent_qn) for e in batch.inherits}
    assert edges == {
        ("dia.B", "A"),
        ("dia.C", "A"),
        ("dia.D", "B"),
        ("dia.D", "C"),
    }


def test_no_implicit_object_parent(parser):
    """Bare `class Foo {}` (no `extends`, no `implements`) emits ZERO
    InheritsEdges. We do NOT model the implicit `java.lang.Object`
    parent — per plan §"java.lang.* allowlist", java.lang.* names
    stay outside the graph; modelling Object on every class would
    bloat the inheritance graph for zero query value.
    """
    src = (
        b"package app;\n"
        b"public class Foo {}\n"
    )
    batch = _scan(parser, "src/app/Foo.java", src)
    assert len(batch.inherits) == 0


def test_record_implements_serializable(parser):
    """`record Point(int x, int y) implements Serializable {}` →
    exactly ONE InheritsEdge (to `Serializable`). The implicit
    `java.lang.Record` parent is NOT modelled — same rule as the
    implicit `Object` parent on classes.
    """
    src = (
        b"package geo;\n"
        b"public record Point(int x, int y) implements Serializable {}\n"
    )
    batch = _scan(parser, "src/geo/Point.java", src)
    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "geo.Point"
    assert edge.parent_qn == "Serializable"


def test_enum_implements_interface(parser):
    """`enum Color implements Named { RED, GREEN; }` → 1 InheritsEdge
    to `Named`. The implicit `java.lang.Enum` parent is NOT modelled
    — same rule.
    """
    src = (
        b"package paint;\n"
        b"public enum Color implements Named { RED, GREEN; }\n"
    )
    batch = _scan(parser, "src/paint/Color.java", src)
    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "paint.Color"
    assert edge.parent_qn == "Named"


# ─── 17c: Call edges (4 flavours + super/this + method references) ────────


def test_bare_call_in_same_class_resolves_to_qn(parser):
    """`foo() { bar(); }` with `bar()` defined in the same file →
    CallEdge with callee_qn=`pkg.Cls.bar` (prefixed via local_names).
    The module qn for `Cls.java` in `pkg` is `pkg.Cls`, so prefixing
    bare `bar` produces `pkg.Cls.bar` — which is the actual qn the
    sibling method got. Same-file resolution win.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { bar(); }\n"
        b"    void bar() {}\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "pkg.Cls.bar") in edges


def test_bare_call_to_unknown_stays_bare(parser):
    """`unknownThing()` doesn't appear in this file's local_names →
    callee_qn stays `"unknownThing"` (no module prefix). The resolver
    will map it to a Symbol or to an imported function later (17d).
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { unknownThing(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "unknownThing") in edges


def test_field_access_call(parser):
    """`obj.bar()` → method_invocation with `object=identifier` →
    callee_qn=`"obj.bar"`. The receiver is captured verbatim; type
    binding (resolving `obj` → `User` so this becomes `User.bar`) is
    a future-sprint concern.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { obj.bar(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "obj.bar") in edges


def test_this_call(parser):
    """`this.bar()` → method_invocation with `object=this` →
    callee_qn=`"this.bar"`. The resolver maps `this.X` to `<class>.X`
    later — extractor stays mechanical.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { this.bar(); }\n"
        b"    void bar() {}\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "this.bar") in edges


def test_static_call(parser):
    """`Math.max(1, 2)` → method_invocation with `object=identifier`
    (the type name `Math`). No syntactic distinction from instance
    calls in the AST; we emit `"Math.max"` and let the resolver
    disambiguate later.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { Math.max(1, 2); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "Math.max") in edges


def test_chained_field_access_call(parser):
    """`obj.list.add(x)` → method_invocation with `object=field_access`.
    The field_access flattens recursively to `"obj.list"`, then we
    suffix `.add` → `"obj.list.add"`. Period-joined; documented form.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { obj.list.add(x); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    # Documented chain shape: receiver flattened with periods.
    assert ("pkg.Cls.foo", "obj.list.add") in edges


def test_constructor_call(parser):
    """`new User()` → object_creation_expression with `type=type_identifier`
    → callee_qn=`"User"`. No `User.User` form; the resolver maps
    constructor calls by class name.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { new User(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "User") in edges


def test_constructor_call_generic_strips_args(parser):
    """`new ArrayList<String>()` → generic_type wrapping the head
    type_identifier. Strip generics same as 17b's parent-name handling
    → callee_qn=`"ArrayList"`, NOT `"ArrayList<String>"`.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { new ArrayList<String>(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "ArrayList") in edges
    # Make sure the generic args didn't leak into any callee qn.
    for _, callee in edges:
        assert "<" not in callee
        assert ">" not in callee


def test_constructor_call_scoped(parser):
    """`new com.foo.Bar()` → scoped_type_identifier for the type field
    → callee_qn=`"com.foo.Bar"` (dotted form preserved verbatim).
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { new com.foo.Bar(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.foo", "com.foo.Bar") in edges


def test_super_constructor_invocation(parser):
    """`B(int x) { super(x); }` → explicit_constructor_invocation with
    `constructor=super`. Callee_qn is the literal string `"super"`;
    the resolver maps super-calls to the parent class's constructor.
    """
    src = (
        b"package pkg;\n"
        b"class B extends A {\n"
        b"    public B(int x) { super(x); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/B.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.B.B", "super") in edges


def test_this_constructor_invocation(parser):
    """`B() { this(0); }` → explicit_constructor_invocation with
    `constructor=this` → callee_qn=`"this"`. Resolver maps this-call
    to the same class's constructor (overload chain).
    """
    src = (
        b"package pkg;\n"
        b"class B {\n"
        b"    public B(int x) {}\n"
        b"    public B() { this(0); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/B.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.B.B", "this") in edges


def test_method_reference(parser):
    """`stream.map(this::transform)` produces TWO CallEdges:
      1. `stream.map`        (the map() invocation itself)
      2. `this.transform`    (the method-reference, low-priority but
                              recorded — we flatten `this::transform`
                              with `.` separator since the resolver
                              uses dotted callee qns uniformly)
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void foo() { stream.map(this::transform); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    # Primary edge — must always exist.
    assert ("pkg.Cls.foo", "stream.map") in edges
    # Method-ref edge — captured per spec.
    assert ("pkg.Cls.foo", "this.transform") in edges


def test_no_calls_in_abstract_method(parser):
    """`abstract void foo();` has `body=None` → _walk_calls returns
    empty → no CallEdges from this method. Same for unimplemented
    interface methods. Tests asserts no edge is attributed to
    `Cls.foo`.
    """
    src = (
        b"package pkg;\n"
        b"abstract class Cls {\n"
        b"    abstract void foo();\n"
        b"    void bar() { helper(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    callers = {c.caller_qn for c in batch.calls}
    # `foo` is abstract — no body, no edges from it.
    assert "pkg.Cls.foo" not in callers
    # `bar` has a body and a call — sanity-check the rest of the
    # extractor still works.
    assert "pkg.Cls.bar" in callers


def test_calls_inside_lambda_attribute_to_enclosing(parser):
    """`void run() { Runnable r = () -> doIt(); }` — the lambda body
    is walked as part of `run`'s body, so `doIt` attributes to
    `pkg.Cls.run`. Lambdas don't yet have synthetic enclosing
    FunctionNodes (Sprint 17e).
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void run() { Runnable r = () -> doIt(); }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.run", "doIt") in edges


def test_calls_inside_anonymous_class_attribute_to_enclosing(parser):
    """`void run() { new Runnable() { public void run() { doIt(); }}; }`
    — for 17c, calls inside the anonymous class body attribute to
    the ENCLOSING `run` (because `_walk_calls` recurses through
    object_creation_expression bodies). Documented imperfection;
    Sprint 17e introduces synthetic anon-class FunctionNodes.

    We accept either:
      - one edge from `pkg.Cls.run` to `doIt` (ideal case), or
      - the more verbose case where `Runnable` (the constructor)
        is also recorded — both are correct.
    """
    src = (
        b"package pkg;\n"
        b"class Cls {\n"
        b"    void run() {\n"
        b"        new Runnable() { public void run() { doIt(); } };\n"
        b"    }\n"
        b"}\n"
    )
    batch = _scan(parser, "src/pkg/Cls.java", src)
    edges = [(c.caller_qn, c.callee_qn) for c in batch.calls]
    assert ("pkg.Cls.run", "doIt") in edges
