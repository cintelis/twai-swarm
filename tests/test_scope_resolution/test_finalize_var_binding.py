"""Sprint 14g.1 — variable-binding receiver resolution end-to-end tests.

Constructs synthetic IndexBatches that include LocalVarBindings, runs
finalize_batch, and asserts the resulting CallEdge targets resolve to
methods on the bound class. Mirrors the test_finalize.py style: no
tree-sitter, no Neo4j; rustworkx is a runtime dep gated by
importorskip.
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


def _cls(qn, file_path, line_start=1, line_end=10):
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


def _binding(file_path, fn_line_start, fn_line_end, var_name, type_raw_name, line):
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="function",
        enclosing_line_start=fn_line_start,
        enclosing_line_end=fn_line_end,
        var_name=var_name, type_raw_name=type_raw_name, line=line,
    )


# ─── Case 7: simple typeBinding via constructor assignment ───────────────────

def test_var_binding_resolves_method_call_to_bound_class():
    """`builder = StateGraph()` then `builder.add_node()` — the
    assignment binds `builder: StateGraph`; `add_node` is a method on
    StateGraph; the resulting call edge points at the method."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py"), _mod("a", "a.py")]
    # StateGraph defined in g.py with a method add_node.
    batch.classes = [_cls("g.StateGraph", "g.py", line_start=1, line_end=20)]
    batch.functions = [
        _method("g.StateGraph.add_node", "g.py",
                parent_class_qn="g.StateGraph",
                line_start=5, line_end=8),
        # Caller — top-level function in a.py that does the binding.
        _fn("a.use_it", "a.py", line_start=10, line_end=15),
    ]
    # `from g import StateGraph` so the receiver-type lookup can resolve.
    batch.imports = [_imp("a.py", "g.StateGraph", "StateGraph", kind="symbol")]
    # The binding: `builder = StateGraph(...)` at line 11 inside use_it.
    batch.local_var_bindings = [
        _binding("a.py", 10, 15, "builder", "StateGraph", line=11),
    ]
    # The call: `builder.add_node(...)` at line 12.
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use_it",
                 callee_qn="builder.add_node", line=12),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "g.StateGraph.add_node"
    assert all(s.qualified_name != "builder.add_node" for s in batch.symbols)


def test_var_binding_resolves_method_via_inheritance():
    """`x = ChildClass()` then `x.parent_method()` — the method comes
    from a parent class; dispatch index walks the chain."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py"), _mod("a", "a.py")]
    batch.classes = [
        _cls("g.Base", "g.py", line_start=1, line_end=10),
        _cls("g.Child", "g.py", line_start=20, line_end=30),
    ]
    batch.functions = [
        _method("g.Base.greet", "g.py", parent_class_qn="g.Base",
                line_start=5, line_end=7),
        _fn("a.use_it", "a.py", line_start=40, line_end=45),
    ]
    batch.imports = [_imp("a.py", "g.Child", "Child", kind="symbol")]
    batch.local_var_bindings = [
        _binding("a.py", 40, 45, "c", "Child", line=41),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use_it",
                 callee_qn="c.greet", line=42),
    ]
    # Inheritance edge wires Child to Base.
    from app.repo_indexer.actions import InheritsEdge
    batch.inherits = [InheritsEdge(repo="r", child_qn="g.Child", parent_qn="g.Base")]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "g.Base.greet"


def test_var_binding_falls_through_when_method_not_on_class():
    """`x = StateGraph()` then `x.unknown_method()` — no method by that
    name on StateGraph or its ancestors; the call stays unresolved
    (Symbol)."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py"), _mod("a", "a.py")]
    batch.classes = [_cls("g.StateGraph", "g.py")]
    batch.functions = [
        _fn("a.use_it", "a.py", line_start=10, line_end=15),
    ]
    batch.imports = [_imp("a.py", "g.StateGraph", "StateGraph", kind="symbol")]
    batch.local_var_bindings = [
        _binding("a.py", 10, 15, "x", "StateGraph", line=11),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use_it",
                 callee_qn="x.unknown_method", line=12),
    ]

    finalize_batch(batch)

    # Stays unresolved → Symbol emitted with the raw dotted name.
    assert batch.calls[0].callee_qn == "x.unknown_method"
    assert any(s.qualified_name == "x.unknown_method" for s in batch.symbols)


def test_var_binding_falls_through_when_type_not_imported():
    """`builder = StateGraph()` but `StateGraph` isn't in scope (no
    import) — the type can't be resolved to a real class qn, so the
    call stays unresolved."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [
        _fn("a.use_it", "a.py", line_start=10, line_end=15),
    ]
    batch.local_var_bindings = [
        _binding("a.py", 10, 15, "builder", "StateGraph", line=11),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.use_it",
                 callee_qn="builder.add_node", line=12),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "builder.add_node"
    assert any(s.qualified_name == "builder.add_node" for s in batch.symbols)


def test_var_binding_does_not_block_existing_resolutions():
    """Parity invariant: anything 12b/12c resolves still resolves
    after 14g. A call that hits the param-type branch (case d) should
    NOT be re-resolved by the new variable-binding branch."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("g", "g.py"), _mod("a", "a.py")]
    batch.classes = [_cls("g.User", "g.py")]
    batch.functions = [
        _method("g.User.greet", "g.py", parent_class_qn="g.User",
                line_start=5, line_end=7),
        # Caller has a parameter `u: User` — case (d) should resolve.
        FunctionNode(
            repo="r", qualified_name="a.handle", name="handle",
            file_path="a.py", line_start=10, line_end=15,
            params=("u",), param_types=(("u", "User"),),
        ),
    ]
    batch.imports = [_imp("a.py", "g.User", "User", kind="symbol")]
    # No LocalVarBinding for u — it's a parameter, not a local assignment.
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.handle",
                 callee_qn="u.greet", line=12),
    ]

    finalize_batch(batch)

    # Resolved via case (d), not via the new branch.
    assert batch.calls[0].callee_qn == "g.User.greet"


def test_var_binding_empty_batch_is_a_no_op():
    """When there are zero bindings (the pre-14g case), behaviour is
    byte-identical to before. No spurious resolutions, no crashes."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [_fn("a.f", "a.py")]
    finalize_batch(batch)
    assert batch.calls == []
    assert batch.symbols == []
