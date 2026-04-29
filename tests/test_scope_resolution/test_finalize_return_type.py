"""Sprint 14h — function return-type tracking end-to-end tests.

Closes the LangGraph-orchestration pattern: `g = builder.compile();
g.invoke()` resolves through the return-type binding for `compile`.
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
from app.repo_indexer.scope_resolution._adapter import MODULE_SCOPE_END  # noqa: E402
from app.repo_indexer.scope_resolution.finalize import finalize_batch  # noqa: E402


REPO = RepoNode(name="r", url="", commit_sha="")


def _fn(qn, file_path, line_start=1, line_end=20, return_type_raw="", **kwargs):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
        return_type_raw=return_type_raw,
        **kwargs,
    )


def _method(qn, file_path, parent_class_qn, line_start=2, line_end=4,
            return_type_raw=""):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
        is_method=True, parent_class_qn=parent_class_qn,
        return_type_raw=return_type_raw,
    )


def _cls(qn, file_path, line_start=1, line_end=20):
    return ClassNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
    )


def _mod(qn, file_path):
    return ModuleNode(repo="r", qualified_name=qn, file_path=file_path)


def _local_binding(file_path, fn_line_start, fn_line_end, var_name, type_raw_name, line):
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="function",
        enclosing_line_start=fn_line_start,
        enclosing_line_end=fn_line_end,
        var_name=var_name, type_raw_name=type_raw_name, line=line,
    )


def _module_return_binding(file_path, fn_name, return_type_raw, line):
    """Free function's return-type binding lives on the module scope."""
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="module",
        enclosing_line_start=0,
        enclosing_line_end=MODULE_SCOPE_END - 1,
        var_name=fn_name, type_raw_name=return_type_raw, line=line,
    )


def _method_return_binding(file_path, cls_line_start, cls_line_end, method_name, return_type_raw, line):
    """Method's return-type binding lives on the class scope."""
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="class",
        enclosing_line_start=cls_line_start,
        enclosing_line_end=cls_line_end,
        var_name=method_name, type_raw_name=return_type_raw, line=line,
    )


# ─── Same-file: `x = func(); x.method()` (free function) ────────────────────

def test_free_function_return_type_resolves_method_call():
    """`def make_user() -> User: ...; def use(): u = make_user(); u.save()`
    — same file, no imports involved."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [_cls("a.User", "a.py", line_start=1, line_end=10)]
    batch.functions = [
        _method("a.User.save", "a.py", parent_class_qn="a.User",
                line_start=5, line_end=7),
        _fn("a.make_user", "a.py", line_start=15, line_end=17,
            return_type_raw="User"),
        _fn("a.use", "a.py", line_start=20, line_end=25),
    ]
    batch.local_var_bindings = [
        # Return-type binding for make_user, on the module scope.
        _module_return_binding("a.py", "make_user", "User", line=15),
        # Local var inside `use`: `u = make_user()`.
        _local_binding("a.py", 20, 25, "u", "make_user", line=21),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use",
                 callee_qn="u.save", line=22),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.User.save"


# ─── Method return type: the LangGraph case ────────────────────────────────

def test_method_return_type_resolves_chained_call():
    """`g = builder.compile(); g.invoke()` — the canonical LangGraph
    pattern. `compile` has a return-type binding on StateGraph's class
    scope; `invoke` is on CompiledStateGraph."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py"), _mod("a", "a.py")]
    batch.classes = [
        _cls("g.StateGraph", "g.py", line_start=1, line_end=20),
        _cls("g.CompiledStateGraph", "g.py", line_start=30, line_end=50),
    ]
    batch.functions = [
        _method("g.StateGraph.compile", "g.py",
                parent_class_qn="g.StateGraph",
                line_start=10, line_end=12,
                return_type_raw="CompiledStateGraph"),
        _method("g.CompiledStateGraph.invoke", "g.py",
                parent_class_qn="g.CompiledStateGraph",
                line_start=35, line_end=37),
        _fn("a.run", "a.py", line_start=10, line_end=20),
    ]
    batch.imports = [
        ImportEdge(repo="r", file_path="a.py",
                   target_qn="g.StateGraph", local_name="StateGraph",
                   kind="symbol"),
    ]
    batch.local_var_bindings = [
        # Return-type binding for StateGraph.compile, on the StateGraph
        # class scope.
        _method_return_binding("g.py", 1, 20, "compile",
                               "CompiledStateGraph", line=10),
        # Local in `run`: `builder = StateGraph()`.
        _local_binding("a.py", 10, 20, "builder", "StateGraph", line=11),
        # Local in `run`: `g = builder.compile()`.
        _local_binding("a.py", 10, 20, "g", "builder.compile", line=12),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="g.invoke", line=13),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "g.CompiledStateGraph.invoke"


# ─── Unannotated function falls through cleanly ─────────────────────────────

def test_unannotated_function_falls_through():
    """`def helper(): pass; def use(): x = helper(); x.method()` — no
    return annotation, no return-type binding emitted, x's type can't
    be inferred. Stays unresolved without crashing."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [
        _fn("a.helper", "a.py", line_start=1, line_end=3),  # no return_type
        _fn("a.use", "a.py", line_start=10, line_end=15),
    ]
    batch.local_var_bindings = [
        _local_binding("a.py", 10, 15, "x", "helper", line=11),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use",
                 callee_qn="x.method", line=12),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "x.method"
    assert any(s.qualified_name == "x.method" for s in batch.symbols)


# ─── Cyclic alias safety: depth-cap prevents infinite recursion ────────────

def test_cyclic_alias_does_not_infinite_loop():
    """Synthetic edge case: a function whose return type is itself.
    Depth cap at 8 prevents infinite recursion; falls through cleanly."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [
        # `a.f` returns `f` (itself, somehow) — pathological but mustn't crash.
        _fn("a.f", "a.py", line_start=1, line_end=3, return_type_raw="f"),
        _fn("a.use", "a.py", line_start=10, line_end=15),
    ]
    batch.local_var_bindings = [
        _module_return_binding("a.py", "f", "f", line=1),
        _local_binding("a.py", 10, 15, "x", "f", line=11),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use",
                 callee_qn="x.method", line=12),
    ]

    finalize_batch(batch)

    # Stays unresolved; no infinite loop.
    assert batch.calls[0].callee_qn == "x.method"


# ─── Inheritance through return-typed method ───────────────────────────────

def test_return_type_with_inheritance_walks_dispatch_index():
    """`compile()` returns CompiledStateGraph; `invoke` is defined on a
    BASE class. Dispatch index should walk the inheritance chain."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py")]
    batch.classes = [
        _cls("g.StateGraph", "g.py", line_start=1, line_end=15),
        _cls("g.BaseRunnable", "g.py", line_start=20, line_end=30),
        _cls("g.CompiledStateGraph", "g.py", line_start=40, line_end=50),
    ]
    batch.functions = [
        _method("g.StateGraph.compile", "g.py",
                parent_class_qn="g.StateGraph",
                line_start=10, line_end=12,
                return_type_raw="CompiledStateGraph"),
        _method("g.BaseRunnable.invoke", "g.py",
                parent_class_qn="g.BaseRunnable",
                line_start=25, line_end=27),
        _fn("g.run", "g.py", line_start=60, line_end=70),
    ]
    from app.repo_indexer.actions import InheritsEdge
    batch.inherits = [InheritsEdge(
        repo="r", child_qn="g.CompiledStateGraph", parent_qn="g.BaseRunnable",
    )]
    batch.local_var_bindings = [
        _method_return_binding("g.py", 1, 15, "compile",
                               "CompiledStateGraph", line=10),
        _local_binding("g.py", 60, 70, "builder", "StateGraph", line=61),
        _local_binding("g.py", 60, 70, "g", "builder.compile", line=62),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="g.run",
                 callee_qn="g.invoke", line=63),
    ]

    finalize_batch(batch)

    # Resolves via dispatch index walking CompiledStateGraph → BaseRunnable.
    assert batch.calls[0].callee_qn == "g.BaseRunnable.invoke"


# ─── Double-chain: x = a.b.c() then x.method() (depth-2 method chain) ──────

def test_method_chain_through_field_then_method():
    """`x = service.client.connect(); x.send()` — service has field
    `client: ApiClient`, which has method `connect() -> Connection`."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [
        _cls("a.ApiClient", "a.py", line_start=1, line_end=15),
        _cls("a.Connection", "a.py", line_start=20, line_end=30),
        _cls("a.Service", "a.py", line_start=40, line_end=60),
    ]
    batch.functions = [
        _method("a.ApiClient.connect", "a.py",
                parent_class_qn="a.ApiClient",
                line_start=10, line_end=12,
                return_type_raw="Connection"),
        _method("a.Connection.send", "a.py",
                parent_class_qn="a.Connection",
                line_start=25, line_end=27),
        _fn("a.run", "a.py", line_start=70, line_end=80),
    ]
    batch.local_var_bindings = [
        # `connect`'s return-type binding on ApiClient.
        _method_return_binding("a.py", 1, 15, "connect", "Connection", line=10),
        # Service has a `client: ApiClient` field.
        LocalVarBinding(
            repo="r", tenant_id="default", file_path="a.py",
            enclosing_scope_kind="class",
            enclosing_line_start=40, enclosing_line_end=60,
            var_name="client", type_raw_name="ApiClient", line=41,
        ),
        # `s = Service()` then `x = s.client.connect()`.
        _local_binding("a.py", 70, 80, "s", "Service", line=71),
        # x's type is the dotted chain `s.client.connect`.
        _local_binding("a.py", 70, 80, "x", "s.client.connect", line=72),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="x.send", line=73),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.Connection.send"
