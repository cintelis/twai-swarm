"""Resolver — cross-file Function/Class resolution against an IndexBatch.

Tests build IndexBatches by hand rather than parsing source so we control
the exact graph shape. Extractor → resolver is exercised end-to-end in
test_repo_indexer_e2e_python — those tests pin actual swarm-shaped code.
"""
from __future__ import annotations

import pytest

from app.repo_indexer.actions import (
    CallEdge,
    ClassNode,
    FunctionNode,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    RepoNode,
)
from app.repo_indexer.resolver import resolve_batch


REPO = RepoNode(name="r", url="", commit_sha="")


def _fn(qn, file_path, **kwargs):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=1, line_end=2,
        **kwargs,
    )


def _cls(qn, file_path):
    return ClassNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=1, line_end=2,
    )


def _imp(file_path, target_qn, local_name, kind="module"):
    return ImportEdge(
        repo="r", file_path=file_path, target_qn=target_qn,
        local_name=local_name, kind=kind,
    )


# ─── Bare-name resolution via `from x import y` ─────────────────────────────

def test_bare_name_resolves_via_from_import():
    """`bar()` resolves to `app.foo.bar` when the caller's file has
    `from app.foo import bar`."""
    batch = IndexBatch(repo=REPO)
    batch.functions = [
        _fn("app.foo.bar", "app/foo.py"),
        _fn("app.consumer.use_it", "app/consumer.py"),
    ]
    batch.imports = [_imp("app/consumer.py", "app.foo.bar", "bar", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.consumer.use_it",
                            callee_qn="bar", line=10)]

    resolve_batch(batch)

    assert batch.calls[0].callee_qn == "app.foo.bar"
    assert batch.symbols == []


# ─── `imported_module.func` resolution via `import x.y` ────────────────────

def test_module_dot_func_resolves_via_import():
    """`foo.bar()` resolves to `app.foo.bar` when the caller's file has
    `import app.foo as foo`."""
    batch = IndexBatch(repo=REPO)
    batch.functions = [
        _fn("app.foo.bar", "app/foo.py"),
        _fn("app.consumer.use_it", "app/consumer.py"),
    ]
    batch.imports = [_imp("app/consumer.py", "app.foo", "foo", kind="module")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.consumer.use_it",
                            callee_qn="foo.bar", line=10)]

    resolve_batch(batch)

    assert batch.calls[0].callee_qn == "app.foo.bar"


# ─── Param-type-driven method resolution ────────────────────────────────────

def test_param_method_resolves_via_type_annotation():
    """`box.run_bash()` resolves to `Sandbox.run_bash` when the caller's
    param `box: Sandbox` is annotated."""
    batch = IndexBatch(repo=REPO)
    batch.classes = [_cls("app.sandbox.Sandbox", "app/sandbox.py")]
    batch.functions = [
        _fn("app.sandbox.Sandbox.run_bash", "app/sandbox.py", is_method=True,
            parent_class_qn="app.sandbox.Sandbox"),
        _fn("app.tools.use_it", "app/tools.py",
            param_types=(("box", "Sandbox"),)),
    ]
    batch.imports = [_imp("app/tools.py", "app.sandbox.Sandbox", "Sandbox", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.tools.use_it",
                            callee_qn="box.run_bash", line=20)]

    resolve_batch(batch)

    assert batch.calls[0].callee_qn == "app.sandbox.Sandbox.run_bash"


def test_param_type_with_optional_resolves():
    """`Optional[Sandbox]` annotation should still resolve to `Sandbox`."""
    batch = IndexBatch(repo=REPO)
    batch.classes = [_cls("app.sandbox.Sandbox", "app/sandbox.py")]
    batch.functions = [
        _fn("app.sandbox.Sandbox.run_bash", "app/sandbox.py", is_method=True,
            parent_class_qn="app.sandbox.Sandbox"),
        _fn("app.tools.use_it", "app/tools.py",
            param_types=(("box", "Optional[Sandbox]"),)),
    ]
    batch.imports = [_imp("app/tools.py", "app.sandbox.Sandbox", "Sandbox", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.tools.use_it",
                            callee_qn="box.run_bash", line=20)]

    resolve_batch(batch)

    assert batch.calls[0].callee_qn == "app.sandbox.Sandbox.run_bash"


def test_param_type_with_union_resolves_leftmost():
    """`Sandbox | None` resolves on the leftmost type."""
    batch = IndexBatch(repo=REPO)
    batch.classes = [_cls("app.sandbox.Sandbox", "app/sandbox.py")]
    batch.functions = [
        _fn("app.sandbox.Sandbox.run_bash", "app/sandbox.py", is_method=True,
            parent_class_qn="app.sandbox.Sandbox"),
        _fn("app.tools.use_it", "app/tools.py",
            param_types=(("box", "Sandbox | None"),)),
    ]
    batch.imports = [_imp("app/tools.py", "app.sandbox.Sandbox", "Sandbox", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.tools.use_it",
                            callee_qn="box.run_bash", line=20)]

    resolve_batch(batch)

    assert batch.calls[0].callee_qn == "app.sandbox.Sandbox.run_bash"


# ─── Inheritance resolution ─────────────────────────────────────────────────

def test_inheritance_resolves_via_from_import():
    """`class Sub(Base):` where `from app.base import Base` should rewrite
    parent_qn to the qualified Class."""
    batch = IndexBatch(repo=REPO)
    batch.classes = [
        _cls("app.base.Base", "app/base.py"),
        _cls("app.derived.Sub", "app/derived.py"),
    ]
    batch.imports = [_imp("app/derived.py", "app.base.Base", "Base", kind="symbol")]
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.derived.Sub",
                                   parent_qn="Base")]

    resolve_batch(batch)

    assert batch.inherits[0].parent_qn == "app.base.Base"
    assert batch.symbols == []


# ─── Unresolved cases produce Symbol nodes ──────────────────────────────────

def test_unresolved_external_call_emits_symbol():
    batch = IndexBatch(repo=REPO)
    batch.functions = [_fn("app.x.use_it", "app/x.py")]
    batch.imports = [_imp("app/x.py", "json", "json", kind="module")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.x.use_it",
                            callee_qn="json.loads", line=5)]

    resolve_batch(batch)

    # `json.loads` doesn't resolve to any in-repo Function.
    # CallEdge keeps the original qn; resolver emits one Symbol for it.
    assert batch.calls[0].callee_qn == "json.loads"
    assert any(s.qualified_name == "json.loads" for s in batch.symbols)


def test_symbols_dedupe_across_calls():
    """Two callers referencing the same external symbol should produce
    exactly one SymbolNode."""
    batch = IndexBatch(repo=REPO)
    batch.functions = [
        _fn("app.x.f1", "app/x.py"),
        _fn("app.x.f2", "app/x.py"),
    ]
    batch.imports = [_imp("app/x.py", "json", "json", kind="module")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.x.f1", callee_qn="json.loads", line=1),
        CallEdge(repo="r", caller_qn="app.x.f2", callee_qn="json.loads", line=2),
    ]

    resolve_batch(batch)

    matches = [s for s in batch.symbols if s.qualified_name == "json.loads"]
    assert len(matches) == 1


# ─── Idempotence ────────────────────────────────────────────────────────────

def test_resolver_is_idempotent():
    """Running the resolver twice produces the same final state."""
    batch = IndexBatch(repo=REPO)
    batch.functions = [
        _fn("app.foo.bar", "app/foo.py"),
        _fn("app.consumer.use_it", "app/consumer.py"),
    ]
    batch.imports = [_imp("app/consumer.py", "app.foo.bar", "bar", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.consumer.use_it",
                            callee_qn="bar", line=10)]

    resolve_batch(batch)
    after_first = batch.calls[0].callee_qn
    resolve_batch(batch)
    after_second = batch.calls[0].callee_qn

    assert after_first == after_second == "app.foo.bar"
    # No duplicate symbols accumulated.
    assert len(batch.symbols) == 0
