"""Module-scope-index — exports filter + custom predicate."""
from __future__ import annotations

from app.repo_indexer.scope_resolution import (
    Declaration,
    Range,
    ScopeId,
    build_module_scope_index,
)


def r(file: str, start: int, end: int) -> Range:
    return Range(file_path=file, start_byte=start, end_byte=end)


def decl(
    qn: str,
    name: str,
    *,
    kind: str = "function",
    file: str = "a.py",
    scope: ScopeId | None = None,
) -> Declaration:
    return Declaration(
        qualified_name=qn,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        file_path=file,
        range=r(file, 0, 10),
        scope_id=scope,
    )


class TestDefaultPredicate:
    def test_excludes_underscore_names(self):
        decls = [
            decl("app.foo.public_fn", "public_fn"),
            decl("app.foo._private_fn", "_private_fn"),
            decl("app.foo.__dunder__", "__dunder__"),
            decl("app.foo.PublicClass", "PublicClass", kind="class"),
        ]
        idx = build_module_scope_index(decls)
        names = {d.name for d in idx.exports_of("app.foo")}
        assert names == {"public_fn", "PublicClass"}

    def test_skips_nested_declarations(self):
        # Methods inside a class have non-None scope_id; they aren't
        # module-level exports — the *class* is.
        class_scope = ScopeId(file_path="a.py", range=r("a.py", 0, 100), kind="class")
        decls = [
            decl("app.foo.MyClass", "MyClass", kind="class"),
            decl("app.foo.MyClass.method", "method", kind="method", scope=class_scope),
        ]
        idx = build_module_scope_index(decls)
        names = {d.name for d in idx.exports_of("app.foo")}
        assert names == {"MyClass"}

    def test_module_kind_indexes_under_self(self):
        # A module-as-declaration belongs to its own QN, not its parent's.
        decls = [
            decl("app.foo", "foo", kind="module"),
            decl("app.foo.bar", "bar"),
        ]
        idx = build_module_scope_index(decls)
        # The module itself is exported under "app.foo"
        assert any(d.kind == "module" for d in idx.exports_of("app.foo"))
        # And its function is also under "app.foo"
        assert any(d.name == "bar" for d in idx.exports_of("app.foo"))

    def test_module_qns_set(self):
        decls = [
            decl("app.foo.x", "x"),
            decl("app.bar.y", "y"),
        ]
        idx = build_module_scope_index(decls)
        assert idx.module_qns() == {"app.foo", "app.bar"}

    def test_unknown_module_returns_empty_list(self):
        idx = build_module_scope_index([decl("a.b.c", "c")])
        assert idx.exports_of("nonexistent.module") == []

    def test_top_level_no_dot_qn(self):
        # Edge case: a QN with no dots indexes under the empty-string module.
        idx = build_module_scope_index([decl("toplevel", "toplevel")])
        assert any(d.name == "toplevel" for d in idx.exports_of(""))


class TestCustomPredicate:
    def test_typescript_style_predicate(self):
        """TypeScript convention: single-underscore is OK, double-underscore private.

        Demonstrates that the override hook works — same input, different
        export set under different rules.
        """
        ts_private = lambda d: d.name.startswith("__")  # noqa: E731

        decls = [
            decl("pkg.fn", "fn"),
            decl("pkg._fn", "_fn"),     # OK under TS rule
            decl("pkg.__fn", "__fn"),   # private under TS rule
        ]
        idx = build_module_scope_index(decls, private_predicate=ts_private)
        names = {d.name for d in idx.exports_of("pkg")}
        assert names == {"fn", "_fn"}

    def test_predicate_that_exports_everything(self):
        """A predicate that always returns False should expose every decl."""
        decls = [
            decl("pkg.public", "public"),
            decl("pkg._private", "_private"),
            decl("pkg.__dunder__", "__dunder__"),
        ]
        idx = build_module_scope_index(decls, private_predicate=lambda _: False)
        names = {d.name for d in idx.exports_of("pkg")}
        assert names == {"public", "_private", "__dunder__"}


class TestExportsListIsCopy:
    def test_mutating_returned_list_does_not_affect_index(self):
        idx = build_module_scope_index([decl("pkg.fn", "fn")])
        first = idx.exports_of("pkg")
        first.clear()
        # A second call should still return the original entry.
        assert len(idx.exports_of("pkg")) == 1
