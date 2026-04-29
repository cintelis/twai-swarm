"""Sprint 14g.2 — compound-receiver resolution end-to-end tests.

`obj.attr.method()` and `self.attr.method()` chains. Builds synthetic
IndexBatches with class-scoped LocalVarBindings (representing
`self.x = SomeClass(...)` patterns from `__init__`).
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
    LocalVarBinding,
    ModuleNode,
    RepoNode,
)
from app.repo_indexer.scope_resolution.finalize import finalize_batch  # noqa: E402


REPO = RepoNode(name="r", url="", commit_sha="")


def _fn(qn, file_path, line_start=1, line_end=20, **kwargs):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
        **kwargs,
    )


def _method(qn, file_path, parent_class_qn, line_start=2, line_end=4):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
        is_method=True, parent_class_qn=parent_class_qn,
    )


def _cls(qn, file_path, line_start=1, line_end=20):
    return ClassNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
    )


def _mod(qn, file_path):
    return ModuleNode(repo="r", qualified_name=qn, file_path=file_path)


def _imp(file_path, target_qn, local_name, kind="symbol"):
    return ImportEdge(
        repo="r", file_path=file_path, target_qn=target_qn,
        local_name=local_name, kind=kind,
    )


def _local_binding(file_path, fn_line_start, fn_line_end, var_name, type_raw_name, line):
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="function",
        enclosing_line_start=fn_line_start,
        enclosing_line_end=fn_line_end,
        var_name=var_name, type_raw_name=type_raw_name, line=line,
    )


def _field_binding(file_path, cls_line_start, cls_line_end, field_name, type_raw_name, line):
    """Class-field binding: enclosing scope is the CLASS, not a function."""
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="class",
        enclosing_line_start=cls_line_start,
        enclosing_line_end=cls_line_end,
        var_name=field_name, type_raw_name=type_raw_name, line=line,
    )


# ─── self.attr.method() — class-field via self ──────────────────────────────

def test_self_field_method_resolves():
    """In a method of class C, `self.x.method()` where C has a field
    `x: SomeClass` should resolve to SomeClass.method.

    Models the canonical pattern:

        class Service:
            def __init__(self):
                self.client = ApiClient()
            def fetch(self):
                return self.client.get(...)
    """
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    # ApiClient class with a `get` method.
    batch.classes = [
        _cls("a.ApiClient", "a.py", line_start=1, line_end=10),
        _cls("a.Service", "a.py", line_start=20, line_end=40),
    ]
    batch.functions = [
        _method("a.ApiClient.get", "a.py", parent_class_qn="a.ApiClient",
                line_start=5, line_end=7),
        # Service.fetch — the caller. line_start=30..35.
        _method("a.Service.fetch", "a.py", parent_class_qn="a.Service",
                line_start=30, line_end=35),
    ]
    # `self.client = ApiClient()` in Service's __init__ — stored on the
    # class scope (lines 20..40).
    batch.local_var_bindings = [
        _field_binding("a.py", 20, 40, "client", "ApiClient", line=22),
    ]
    # The call: inside `fetch`, `self.client.get(...)`.
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.Service.fetch",
                 callee_qn="self.client.get", line=33),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.ApiClient.get"


# ─── local_var.field.method() — local var → class field → method ───────────

def test_local_var_field_method_resolves():
    """`s = Service(); s.client.get(...)` chain — local-var binding
    leads to a class whose field's type carries the next method."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [
        _cls("a.ApiClient", "a.py", line_start=1, line_end=10),
        _cls("a.Service", "a.py", line_start=20, line_end=40),
    ]
    batch.functions = [
        _method("a.ApiClient.get", "a.py", parent_class_qn="a.ApiClient",
                line_start=5, line_end=7),
        _fn("a.run", "a.py", line_start=50, line_end=60),
    ]
    # Service has field `client: ApiClient`.
    # Caller `run` has local `s = Service()`.
    batch.local_var_bindings = [
        _field_binding("a.py", 20, 40, "client", "ApiClient", line=22),
        _local_binding("a.py", 50, 60, "s", "Service", line=51),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="s.client.get", line=52),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.ApiClient.get"


# ─── self method without compound — must not over-match ────────────────────

def test_self_dot_method_does_not_trigger_compound_path():
    """`self.method()` is a 2-segment dotted shape; the compound resolver
    requires ≥3 segments. Single-attribute self-calls go through the
    self/super resolver (12c)."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [_cls("a.MyClass", "a.py", line_start=1, line_end=20)]
    batch.functions = [
        _method("a.MyClass.greet", "a.py", parent_class_qn="a.MyClass"),
        _method("a.MyClass.use", "a.py", parent_class_qn="a.MyClass",
                line_start=10, line_end=15),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.MyClass.use",
                 callee_qn="self.greet", line=11),
    ]

    finalize_batch(batch)

    # Resolved by the self/super branch, not the compound resolver.
    assert batch.calls[0].callee_qn == "a.MyClass.greet"


# ─── depth cap ───────────────────────────────────────────────────────────────

def test_compound_receiver_caps_depth_at_3_hops():
    """`a.b.c.d.method()` is 5 segments — beyond the 4-segment depth cap.
    Stays unresolved (Symbol) rather than producing a wrong answer."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [_cls("a.X", "a.py")]
    batch.functions = [
        _method("a.X.method", "a.py", parent_class_qn="a.X"),
        _fn("a.run", "a.py", line_start=50, line_end=60),
    ]
    batch.local_var_bindings = [
        _local_binding("a.py", 50, 60, "x", "X", line=51),
    ]
    batch.calls = [
        # 5 segments: a.b.c.d.method()
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="x.b.c.d.method", line=52),
    ]

    finalize_batch(batch)

    # Stays unresolved.
    assert batch.calls[0].callee_qn == "x.b.c.d.method"
    assert any(s.qualified_name == "x.b.c.d.method" for s in batch.symbols)


# ─── falls through cleanly when class field missing ────────────────────────

def test_compound_receiver_falls_through_when_field_unknown():
    """Caller's class doesn't have a field by that name — chain breaks
    at hop 1; call stays unresolved."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [_cls("a.MyClass", "a.py", line_start=1, line_end=20)]
    batch.functions = [
        _method("a.MyClass.use", "a.py", parent_class_qn="a.MyClass",
                line_start=10, line_end=15),
    ]
    # No field bindings — no `self.x = ...` ever happened.
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.MyClass.use",
                 callee_qn="self.unknown.method", line=11),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "self.unknown.method"


# ─── existing param-typed compound case still resolves (parity) ────────────

def test_compound_receiver_parity_with_existing_resolutions():
    """A compound receiver that the OLD code couldn't resolve, AND the
    new code can't either, is correctly Symbol-emitted. Parity check."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [
        _fn("a.run", "a.py", line_start=50, line_end=60),
    ]
    batch.calls = [
        # No bindings, no class — chain is unresolvable.
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="undefined_thing.attr.method", line=51),
    ]

    finalize_batch(batch)

    # Stays unresolved — both 14g.1 and 14g.2 fall through cleanly.
    assert batch.calls[0].callee_qn == "undefined_thing.attr.method"
