"""Sprint 12b finalize — cross-file resolution against synthetic IndexBatches.

Tests build IndexBatches by hand rather than parsing source so we control
the exact graph shape. No tree-sitter, no Neo4j. Skips if rustworkx (the
12b runtime dep) isn't installed.

Parity note: scenarios 2–4 also exercise the legacy resolver to confirm
the new path matches the legacy on cases the legacy can already resolve.
The new path adds re-export-closure and wildcard handling on top.
"""
from __future__ import annotations

import pytest

# Skip the whole module if rustworkx isn't installed — same lazy-import
# pattern as test_repo_indexer_sha_skip.py uses for its tree-sitter dep.
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
from app.repo_indexer.resolver import resolve_batch as legacy_resolve_batch  # noqa: E402
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
# 1. Empty batch is a no-op
# ---------------------------------------------------------------------------

def test_finalize_empty_batch():
    batch = IndexBatch(repo=REPO)
    finalize_batch(batch)
    assert batch.calls == []
    assert batch.inherits == []
    assert batch.symbols == []
    assert batch.functions == []


# ---------------------------------------------------------------------------
# 2. Bare-name `from x import y; y()` — legacy parity
# ---------------------------------------------------------------------------

def test_finalize_resolves_bare_name_import():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.functions = [
        _fn("app.b.foo", "app/b.py"),
        _fn("app.a.use_it", "app/a.py"),
    ]
    batch.imports = [_imp("app/a.py", "app.b.foo", "foo", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.use_it",
                            callee_qn="foo", line=10)]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.b.foo"
    assert batch.symbols == []


# ---------------------------------------------------------------------------
# 3. Module-prefix import — legacy parity
# ---------------------------------------------------------------------------

def test_finalize_resolves_module_prefix_import():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.functions = [
        _fn("app.b.foo", "app/b.py"),
        _fn("app.a.use_it", "app/a.py"),
    ]
    batch.imports = [_imp("app/a.py", "app.b", "app", kind="module")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.use_it",
                            callee_qn="app.b.foo", line=10)]

    finalize_batch(batch)
    # Direct qn lookup (path "a" of the resolution order) wins here.
    assert batch.calls[0].callee_qn == "app.b.foo"


def test_finalize_resolves_module_prefix_with_alias():
    """`import app.b as bb` + `bb.foo()` — head goes through file_imports."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.functions = [
        _fn("app.b.foo", "app/b.py"),
        _fn("app.a.use_it", "app/a.py"),
    ]
    batch.imports = [_imp("app/a.py", "app.b", "bb", kind="module")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.use_it",
                            callee_qn="bb.foo", line=10)]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.b.foo"


# ---------------------------------------------------------------------------
# 4. Param-type method dispatch — legacy parity
# ---------------------------------------------------------------------------

def test_finalize_resolves_param_type_method():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.classes = [_cls("app.b.Bar", "app/b.py", line_start=1, line_end=10)]
    batch.functions = [
        _fn("app.b.Bar.method", "app/b.py", is_method=True,
            parent_class_qn="app.b.Bar"),
        _fn("app.a.f", "app/a.py", param_types=(("x", "Bar"),)),
    ]
    batch.imports = [_imp("app/a.py", "app.b.Bar", "Bar", kind="symbol")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.f",
                            callee_qn="x.method", line=20)]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.b.Bar.method"


# ---------------------------------------------------------------------------
# 5. Re-export closure (Tarjan SCC over import graph)
# ---------------------------------------------------------------------------

def test_finalize_resolves_reexport_cycle():
    """A defined in C, re-exported via cycle A → B → C → A; D imports
    from A and calls the symbol. Without SCC handling, the symbol
    appears to live in A but the qn is `app.c.foo`, so `_is_function`
    on `app.a.foo` fails. Closure-aware lookup finds it via app.c."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [
        _mod("app.a", "app/a.py"),
        _mod("app.b", "app/b.py"),
        _mod("app.c", "app/c.py"),
        _mod("app.d", "app/d.py"),
    ]
    batch.functions = [
        _fn("app.c.foo", "app/c.py"),
        _fn("app.d.use_it", "app/d.py"),
    ]
    # Cycle A → B → C → A. Each has a from-import that pretends to
    # re-export the others (the from-import edges themselves go through
    # the import-graph; the SCC will catch all three as one group).
    batch.imports = [
        _imp("app/a.py", "app.b", "app", kind="module"),
        _imp("app/b.py", "app.c", "app", kind="module"),
        _imp("app/c.py", "app.a", "app", kind="module"),
        # D imports `foo` from A — A is in the SCC with C, where foo is defined.
        _imp("app/d.py", "app.a.foo", "foo", kind="symbol"),
    ]
    batch.calls = [CallEdge(repo="r", caller_qn="app.d.use_it",
                            callee_qn="foo", line=5)]

    finalize_batch(batch)

    # Closure-aware lookup found foo via the SCC peer.
    assert batch.calls[0].callee_qn == "app.c.foo"


# ---------------------------------------------------------------------------
# 6. Wildcard `from x import *` — synthetic ImportEdge (extractor doesn't
#    emit wildcard edges today; finalize handles them when it does).
# ---------------------------------------------------------------------------

def test_finalize_resolves_wildcard_import():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.b", "app/b.py"), _mod("app.a", "app/a.py")]
    batch.functions = [
        _fn("app.b.foo", "app/b.py"),
        _fn("app.a.use_it", "app/a.py"),
    ]
    # Synthetic wildcard shape: kind="module", local_name="*", target = exporting module qn.
    batch.imports = [_imp("app/a.py", "app.b", "*", kind="module")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.use_it",
                            callee_qn="foo", line=10)]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "app.b.foo"
    assert batch.symbols == []


# ---------------------------------------------------------------------------
# 7. Unresolved external → SymbolNode
# ---------------------------------------------------------------------------

def test_finalize_emits_symbol_for_unresolved():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.a", "app/a.py")]
    batch.functions = [_fn("app.a.use_it", "app/a.py")]
    batch.calls = [CallEdge(repo="r", caller_qn="app.a.use_it",
                            callee_qn="unknown_external_lib", line=1)]

    finalize_batch(batch)

    assert batch.calls[0].callee_qn == "unknown_external_lib"
    assert any(s.qualified_name == "unknown_external_lib" for s in batch.symbols)


def test_finalize_dedupes_symbols():
    batch = IndexBatch(repo=REPO)
    batch.modules = [_mod("app.a", "app/a.py")]
    batch.functions = [
        _fn("app.a.f1", "app/a.py"),
        _fn("app.a.f2", "app/a.py"),
    ]
    batch.imports = [_imp("app/a.py", "json", "json", kind="module")]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.a.f1", callee_qn="json.loads", line=1),
        CallEdge(repo="r", caller_qn="app.a.f2", callee_qn="json.loads", line=2),
    ]

    finalize_batch(batch)

    matches = [s for s in batch.symbols if s.qualified_name == "json.loads"]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# 8. Inheritance via `from x import Cls`
# ---------------------------------------------------------------------------

def test_finalize_resolves_inheritance():
    batch = IndexBatch(repo=REPO)
    batch.modules = [
        _mod("app.base", "app/base.py"),
        _mod("app.derived", "app/derived.py"),
    ]
    batch.classes = [
        _cls("app.base.Base", "app/base.py", line_start=1, line_end=10),
        _cls("app.derived.Sub", "app/derived.py", line_start=1, line_end=10),
    ]
    batch.imports = [_imp("app/derived.py", "app.base.Base", "Base", kind="symbol")]
    batch.inherits = [InheritsEdge(repo="r", child_qn="app.derived.Sub",
                                   parent_qn="Base")]

    finalize_batch(batch)

    assert batch.inherits[0].parent_qn == "app.base.Base"
    assert batch.symbols == []


# ---------------------------------------------------------------------------
# 9. Parity vs legacy on cases legacy can resolve
# ---------------------------------------------------------------------------

def _build_simple_repo() -> IndexBatch:
    """Synthetic batch where legacy and finalize both resolve the same edges."""
    batch = IndexBatch(repo=REPO)
    batch.modules = [
        _mod("app.b", "app/b.py"),
        _mod("app.sand", "app/sand.py"),
        _mod("app.a", "app/a.py"),
    ]
    batch.classes = [
        _cls("app.sand.Sandbox", "app/sand.py", line_start=1, line_end=20),
    ]
    batch.functions = [
        _fn("app.b.bar", "app/b.py"),
        _fn("app.sand.Sandbox.run", "app/sand.py", is_method=True,
            parent_class_qn="app.sand.Sandbox"),
        # Caller has bare-name import + module-prefix-style + param-type usage.
        _fn("app.a.use1", "app/a.py"),
        _fn("app.a.use2", "app/a.py"),
        _fn("app.a.use3", "app/a.py", param_types=(("box", "Sandbox"),)),
    ]
    batch.imports = [
        _imp("app/a.py", "app.b.bar", "bar", kind="symbol"),
        _imp("app/a.py", "app.b", "b", kind="module"),
        _imp("app/a.py", "app.sand.Sandbox", "Sandbox", kind="symbol"),
    ]
    batch.calls = [
        CallEdge(repo="r", caller_qn="app.a.use1", callee_qn="bar", line=1),
        CallEdge(repo="r", caller_qn="app.a.use2", callee_qn="b.bar", line=2),
        CallEdge(repo="r", caller_qn="app.a.use3", callee_qn="box.run", line=3),
        # Unresolvable external — both paths should emit a Symbol.
        CallEdge(repo="r", caller_qn="app.a.use1", callee_qn="external.thing",
                 line=4),
    ]
    return batch


def test_finalize_parity_with_legacy_on_simple_repo():
    legacy_batch = _build_simple_repo()
    new_batch = _build_simple_repo()

    legacy_resolve_batch(legacy_batch)
    finalize_batch(new_batch)

    # Resolved-edge counts: callees rewritten to in-repo Function qns.
    in_repo_qns = {f.qualified_name for f in new_batch.functions}

    legacy_resolved = sum(1 for c in legacy_batch.calls if c.callee_qn in in_repo_qns)
    new_resolved = sum(1 for c in new_batch.calls if c.callee_qn in in_repo_qns)

    # Finalize must resolve at least as many as legacy.
    assert new_resolved >= legacy_resolved
    # On this fixture they should match — there are no closure / wildcard
    # cases hidden in here.
    assert new_resolved == legacy_resolved == 3

    # Pairwise edge equivalence (same caller + line + resolved callee).
    legacy_resolved_set = {
        (c.caller_qn, c.callee_qn, c.line)
        for c in legacy_batch.calls if c.callee_qn in in_repo_qns
    }
    new_resolved_set = {
        (c.caller_qn, c.callee_qn, c.line)
        for c in new_batch.calls if c.callee_qn in in_repo_qns
    }
    assert legacy_resolved_set == new_resolved_set
