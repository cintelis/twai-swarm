"""Sprint 14i — module-scope typeBindings end-to-end tests.

Module-level `app = FastAPI()` followed by `app.get(...)` anywhere in
the file — function bodies, methods, free-standing functions —
resolves via the scope-chain walk hitting the module scope.
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


def _fn(qn, file_path, line_start=10, line_end=20, **kwargs):
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=line_start, line_end=line_end,
        **kwargs,
    )


def _method(qn, file_path, parent_class_qn, line_start=15, line_end=18):
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


def _module_binding(file_path, var_name, type_raw_name, line):
    """Module-scope binding — the encoding the extractor uses for
    top-level `var = SomeClass(...)` assignments. Matches the sentinel
    range in `_adapter.to_scopes` so the resulting ScopeId structurally
    equals the Module ScopeId in the scope tree."""
    return LocalVarBinding(
        repo="r", tenant_id="default", file_path=file_path,
        enclosing_scope_kind="module",
        enclosing_line_start=0,
        enclosing_line_end=MODULE_SCOPE_END - 1,
        var_name=var_name, type_raw_name=type_raw_name, line=line,
    )


# ─── basic: module-level binding visible to function in same file ───────────

def test_module_binding_visible_to_top_level_function():
    """`app = FastAPI(); def handler(): app.get(...)` — the function
    walks scope chain → module scope → finds `app` typeBinding."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [_cls("a.FastAPI", "a.py", line_start=1, line_end=10)]
    batch.functions = [
        _method("a.FastAPI.get", "a.py", parent_class_qn="a.FastAPI",
                line_start=5, line_end=7),
        _fn("a.handler", "a.py", line_start=20, line_end=30),
    ]
    batch.local_var_bindings = [
        _module_binding("a.py", "app", "FastAPI", line=15),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.handler",
                 callee_qn="app.get", line=22),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.FastAPI.get"


# ─── module binding visible to method in same file ─────────────────────────

def test_module_binding_visible_inside_method():
    """`app = FastAPI()` at top, then `class Service: def handle(self):
    app.get(...)`. The method walks: function scope → class scope →
    module scope → finds `app`."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [
        _cls("a.FastAPI", "a.py", line_start=1, line_end=10),
        _cls("a.Service", "a.py", line_start=20, line_end=40),
    ]
    batch.functions = [
        _method("a.FastAPI.get", "a.py", parent_class_qn="a.FastAPI",
                line_start=5, line_end=7),
        _method("a.Service.handle", "a.py", parent_class_qn="a.Service",
                line_start=30, line_end=35),
    ]
    batch.local_var_bindings = [
        _module_binding("a.py", "app", "FastAPI", line=15),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.Service.handle",
                 callee_qn="app.get", line=32),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.FastAPI.get"


# ─── shadowing: function-local binding overrides module-level ──────────────

def test_function_binding_shadows_module_binding():
    """If a function has a local `app = OtherThing()`, that shadows the
    module-level one when resolving `app.method()` inside that function."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [
        _cls("a.FastAPI", "a.py", line_start=1, line_end=5),
        _cls("a.OtherThing", "a.py", line_start=6, line_end=10),
    ]
    batch.functions = [
        _method("a.FastAPI.get", "a.py", parent_class_qn="a.FastAPI",
                line_start=2, line_end=3),
        _method("a.OtherThing.get", "a.py", parent_class_qn="a.OtherThing",
                line_start=7, line_end=8),
        _fn("a.handler", "a.py", line_start=20, line_end=30),
    ]
    batch.local_var_bindings = [
        _module_binding("a.py", "app", "FastAPI", line=15),
        # Function-local rebinding inside `a.handler` (lines 20..30).
        LocalVarBinding(
            repo="r", tenant_id="default", file_path="a.py",
            enclosing_scope_kind="function",
            enclosing_line_start=20, enclosing_line_end=30,
            var_name="app", type_raw_name="OtherThing", line=21,
        ),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.handler",
                 callee_qn="app.get", line=22),
    ]

    finalize_batch(batch)

    # Innermost binding wins — OtherThing, not FastAPI.
    assert batch.calls[0].callee_qn == "a.OtherThing.get"


# ─── module bindings don't cross files ─────────────────────────────────────

def test_module_binding_does_not_leak_across_files():
    """Module bindings are file-scoped — `app = FastAPI()` in a.py is
    invisible to functions in b.py (different module → different scope
    tree root)."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py"), _mod("b", "b.py")]
    batch.classes = [_cls("a.FastAPI", "a.py", line_start=1, line_end=10)]
    batch.functions = [
        _method("a.FastAPI.get", "a.py", parent_class_qn="a.FastAPI",
                line_start=5, line_end=7),
        _fn("b.handler", "b.py", line_start=10, line_end=20),
    ]
    batch.local_var_bindings = [
        _module_binding("a.py", "app", "FastAPI", line=15),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="b.handler",
                 callee_qn="app.get", line=12),
    ]

    finalize_batch(batch)

    # b.handler can't see a.py's module binding → unresolved.
    assert batch.calls[0].callee_qn == "app.get"


# ─── compound chain through a module-level binding ─────────────────────────

def test_module_binding_supports_compound_receiver():
    """`engine = create_engine_class(); engine.dialect.fetch()` — the
    14g.2 compound resolver walks engine's class fields. The module-
    level binding feeds into the same compound machinery."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.classes = [
        _cls("a.Dialect", "a.py", line_start=1, line_end=8),
        _cls("a.Engine", "a.py", line_start=10, line_end=25),
    ]
    batch.functions = [
        _method("a.Dialect.fetch", "a.py", parent_class_qn="a.Dialect",
                line_start=5, line_end=7),
        _method("a.Engine.__init__", "a.py", parent_class_qn="a.Engine",
                line_start=15, line_end=18),
        _fn("a.run", "a.py", line_start=40, line_end=45),
    ]
    # Engine has a `dialect: Dialect` field (set in __init__).
    batch.local_var_bindings = [
        LocalVarBinding(
            repo="r", tenant_id="default", file_path="a.py",
            enclosing_scope_kind="class",
            enclosing_line_start=10, enclosing_line_end=25,
            var_name="dialect", type_raw_name="Dialect", line=16,
        ),
        # Module-level: engine = Engine().
        _module_binding("a.py", "engine", "Engine", line=30),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="a.run",
                 callee_qn="engine.dialect.fetch", line=41),
    ]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "a.Dialect.fetch"


# ─── empty-module-bindings parity ──────────────────────────────────────────

def test_no_module_bindings_is_a_no_op():
    """A batch with no module-level bindings (the common pre-14i case)
    behaves byte-identically. The scope tree gains a Module root but
    that doesn't introduce any new resolutions."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("a", "a.py")]
    batch.functions = [_fn("a.f", "a.py")]
    finalize_batch(batch)
    assert batch.calls == []
    assert batch.symbols == []
