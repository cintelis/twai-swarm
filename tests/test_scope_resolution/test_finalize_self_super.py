"""Sprint 12c — `self.method()` / `super().method()` resolution through
finalize.

Tests build IndexBatches by hand to exercise the dispatch-index branches
in `finalize.py`. No tree-sitter, no Neo4j. Skips if rustworkx (the 12b
runtime dep) isn't installed.

Coverage:
  - `self.foo()` resolves to own-class method.
  - `self.foo()` resolves to inherited method (B inherits A; A has foo).
  - `super().foo()` resolves to parent's method.
  - `super().foo()` with no parent -> SymbolNode.
  - param-type method through inheritance — local Type.method missing,
    ancestor's wins.
  - method genuinely missing -> SymbolNode.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rustworkx")

from app.repo_indexer.actions import (  # noqa: E402
    CallEdge,
    ClassNode,
    FunctionNode,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    ModuleNode,
    RepoNode,
)
from app.repo_indexer.scope_resolution.finalize import finalize_batch  # noqa: E402


REPO = RepoNode(name="r", url="", commit_sha="")


def _fn(qn, file_path, **kwargs):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=1, line_end=2,
        **kwargs,
    )


def _cls(qn, file_path, line_start=1, line_end=2):
    return ClassNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
    )


def _mod(qn, file_path):
    return ModuleNode(repo="r", qualified_name=qn, file_path=file_path)


def _imp(file_path, target_qn, local_name, kind="module"):
    return ImportEdge(
        repo="r", file_path=file_path, target_qn=target_qn,
        local_name=local_name, kind=kind,
    )


# ---------------------------------------------------------------------------
# 1. `self.foo()` resolves to own-class method
# ---------------------------------------------------------------------------

def test_self_method_call_resolves_to_own_class():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [_cls("app.m.C", "app/m.py", line_start=1, line_end=20)]
    batch.functions = [
        _fn("app.m.C.foo", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    # Caller does `self.foo()`.
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="self.foo", line=10),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.m.C.foo"
    assert batch.symbols == []


# ---------------------------------------------------------------------------
# 2. `self.foo()` resolves to inherited method
# ---------------------------------------------------------------------------

def test_self_method_call_resolves_through_inheritance():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [
        _cls("app.m.B", "app/m.py", line_start=1, line_end=10),
        _cls("app.m.C", "app/m.py", line_start=12, line_end=20),
    ]
    batch.functions = [
        _fn("app.m.B.foo", "app/m.py", is_method=True, parent_class_qn="app.m.B"),
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    # Resolved-form InheritsEdge so the dispatch index sees it.
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.m.C", parent_qn="app.m.B")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="self.foo", line=15),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.m.B.foo"
    assert batch.symbols == []


def test_self_method_local_overrides_inherited():
    """C overrides B's foo; `self.foo()` from inside C calls C's, not B's."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [
        _cls("app.m.B", "app/m.py", line_start=1, line_end=10),
        _cls("app.m.C", "app/m.py", line_start=12, line_end=30),
    ]
    batch.functions = [
        _fn("app.m.B.foo", "app/m.py", is_method=True, parent_class_qn="app.m.B"),
        _fn("app.m.C.foo", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.m.C", parent_qn="app.m.B")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="self.foo", line=15),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.m.C.foo"


# ---------------------------------------------------------------------------
# 3. `super().foo()` resolves to parent's method
# ---------------------------------------------------------------------------

def test_super_method_call_resolves_to_parent():
    """The extractor doesn't emit `super().X` strings today (its
    `_flatten_attribute` returns None on call-typed object children),
    so we hand-construct the synthetic `super().foo` shape here. When
    extractor support lands, the same finalize branch starts firing on
    real source.
    """
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [
        _cls("app.m.B", "app/m.py", line_start=1, line_end=10),
        _cls("app.m.C", "app/m.py", line_start=12, line_end=30),
    ]
    batch.functions = [
        _fn("app.m.B.foo", "app/m.py", is_method=True, parent_class_qn="app.m.B"),
        _fn("app.m.C.foo", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.m.C", parent_qn="app.m.B")]
    # Inside C.foo, `super().foo()` should target B.foo (parent's method),
    # not C.foo itself.
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.foo",
                 callee_qn="super().foo", line=15),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.m.B.foo"


def test_super_method_call_works_with_bare_super_prefix():
    """`super.foo` (no parens, no real Python construct) — finalize
    accepts both `super` and `super()` as the leading token in case
    different extractor shapes emit different forms."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [
        _cls("app.m.B", "app/m.py", line_start=1, line_end=10),
        _cls("app.m.C", "app/m.py", line_start=12, line_end=30),
    ]
    batch.functions = [
        _fn("app.m.B.foo", "app/m.py", is_method=True, parent_class_qn="app.m.B"),
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.m.C", parent_qn="app.m.B")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="super.foo", line=15),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.m.B.foo"


# ---------------------------------------------------------------------------
# 4. `super().foo()` with no parent -> Symbol
# ---------------------------------------------------------------------------

def test_super_with_no_parent_falls_through_to_symbol():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [_cls("app.m.C", "app/m.py", line_start=1, line_end=20)]
    batch.functions = [
        _fn("app.m.C.foo", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    # No InheritsEdge for C.
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.foo",
                 callee_qn="super().bar", line=5),
    ]

    finalize_batch(batch)

    # Couldn't resolve — left intact, SymbolNode emitted.
    assert batch.calls[0].callee_qn == "super().bar"
    assert any(s.qualified_name == "super().bar" for s in batch.symbols)


# ---------------------------------------------------------------------------
# 5. `param: T; param.foo()` through inheritance
# ---------------------------------------------------------------------------

def test_param_type_method_resolves_through_inheritance():
    """Foo doesn't define `bar`; Foo's parent does. The 12b path resolves
    `Foo.bar` only if it exists on Foo directly; 12c walks ancestors."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.classes = [
        _cls("app.b.Base", "app/b.py", line_start=1, line_end=10),
        _cls("app.b.Foo", "app/b.py", line_start=12, line_end=20),
    ]
    batch.functions = [
        _fn("app.b.Base.bar", "app/b.py", is_method=True,
            parent_class_qn="app.b.Base"),
        # Note: Foo has NO local `bar` — must walk to Base.
        _fn("app.a.f", "app/a.py", param_types=(("x", "Foo"),)),
    ]
    batch.inherits = [
        InheritsEdge(repo="r", child_qn="app.b.Foo", parent_qn="app.b.Base"),
    ]
    batch.imports = [_imp("app/a.py", "app.b.Foo", "Foo", kind="symbol")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.a.f", callee_qn="x.bar", line=20),
    ]

    finalize_batch(batch)

    # Resolved through the dispatch index walk to Base.bar.
    assert batch.calls[0].callee_qn == "app.b.Base.bar"


def test_param_type_local_method_still_wins():
    """12b parity: when `Type.method` IS local on Type, that's what
    resolves — the dispatch fallback only fires after the local lookup
    misses. This protects every legacy resolution from being silently
    rerouted."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.classes = [_cls("app.b.Foo", "app/b.py", line_start=1, line_end=10)]
    batch.functions = [
        _fn("app.b.Foo.bar", "app/b.py", is_method=True,
            parent_class_qn="app.b.Foo"),
        _fn("app.a.f", "app/a.py", param_types=(("x", "Foo"),)),
    ]
    batch.imports = [_imp("app/a.py", "app.b.Foo", "Foo", kind="symbol")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.a.f", callee_qn="x.bar", line=20),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.b.Foo.bar"


# ---------------------------------------------------------------------------
# 6. Method genuinely missing -> Symbol
# ---------------------------------------------------------------------------

def test_self_method_genuinely_missing_emits_symbol():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [_cls("app.m.C", "app/m.py", line_start=1, line_end=20)]
    batch.functions = [
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="self.does_not_exist", line=10),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "self.does_not_exist"
    assert any(s.qualified_name == "self.does_not_exist" for s in batch.symbols)


# ---------------------------------------------------------------------------
# 7. Caller isn't a method — `self.foo` from a free function falls through
# ---------------------------------------------------------------------------

def test_self_call_from_non_method_caller_falls_through():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.functions = [
        _fn("app.m.free_fn", "app/m.py"),  # not a method
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.free_fn",
                 callee_qn="self.anything", line=1),
    ]

    finalize_batch(batch)

    # No class context -> Symbol.
    assert batch.calls[0].callee_qn == "self.anything"
    assert any(s.qualified_name == "self.anything" for s in batch.symbols)


# ---------------------------------------------------------------------------
# 8. Chained self attribute (`self.attr.foo()`) is out of scope
# ---------------------------------------------------------------------------

def test_chained_self_attribute_call_left_unresolved():
    """`self._tree.parent_of()` requires knowing the type of `self._tree`
    (an instance attribute) — that's local-variable type inference,
    deferred to Sprint 13+. 12c only resolves direct `self.method()`."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.m", "app/m.py")]
    batch.classes = [_cls("app.m.C", "app/m.py", line_start=1, line_end=20)]
    batch.functions = [
        _fn("app.m.C.caller", "app/m.py", is_method=True, parent_class_qn="app.m.C"),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.m.C.caller",
                 callee_qn="self.attr.foo", line=10),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "self.attr.foo"
    assert any(s.qualified_name == "self.attr.foo" for s in batch.symbols)
