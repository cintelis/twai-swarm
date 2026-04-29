"""Sprint 14g.1 — LocalVarTypeIndex unit tests.

Pure stdlib. No tree-sitter, no Neo4j. The index is a dict-of-dicts with
a scope-chain walk; tests construct synthetic ScopeTree fixtures directly
rather than going through the extractor.
"""
from __future__ import annotations

from app.repo_indexer.scope_resolution.local_var_type_index import LocalVarTypeIndex
from app.repo_indexer.scope_resolution.scope_tree import build_scope_tree
from app.repo_indexer.scope_resolution.types import Range, ScopeId, TypeRef


def _scope(file_path: str, start: int, end: int, kind: str = "function") -> ScopeId:
    return ScopeId(
        file_path=file_path,
        range=Range(file_path=file_path, start_byte=start, end_byte=end),
        kind=kind,  # type: ignore[arg-type]
    )


def _ref(name: str, declared_at: ScopeId, source: str = "constructor") -> TypeRef:
    return TypeRef(
        raw_name=name,
        declared_at_scope=declared_at,
        source=source,  # type: ignore[arg-type]
    )


# ─── basic add / contains / find at exact scope ─────────────────────────────

def test_empty_index_finds_nothing():
    idx = LocalVarTypeIndex()
    tree = build_scope_tree([_scope("a.py", 1, 10)])
    assert idx.find(_scope("a.py", 1, 10), "x", tree) is None
    assert len(idx) == 0


def test_add_and_find_at_same_scope():
    s = _scope("a.py", 1, 10)
    idx = LocalVarTypeIndex()
    idx.add(s, "builder", _ref("StateGraph", s))
    tree = build_scope_tree([s])

    found = idx.find(s, "builder", tree)
    assert found is not None
    assert found.raw_name == "StateGraph"
    assert found.source == "constructor"
    assert (s, "builder") in idx


def test_find_returns_none_for_missing_name():
    s = _scope("a.py", 1, 10)
    idx = LocalVarTypeIndex()
    idx.add(s, "builder", _ref("StateGraph", s))
    tree = build_scope_tree([s])
    assert idx.find(s, "graph", tree) is None


# ─── scope-chain walk ───────────────────────────────────────────────────────

def test_inner_scope_finds_outer_binding():
    """A nested function should see typeBindings from its enclosing scope.

    This is the lexical-scope contract — typeBindings declared in an
    outer function are visible in inner closures.
    """
    outer = _scope("a.py", 1, 100)
    inner = _scope("a.py", 10, 50)

    idx = LocalVarTypeIndex()
    idx.add(outer, "client", _ref("OpenAI", outer))
    tree = build_scope_tree([outer, inner])

    found = idx.find(inner, "client", tree)
    assert found is not None
    assert found.raw_name == "OpenAI"


def test_inner_scope_shadows_outer():
    """When the same name exists at multiple levels, innermost wins."""
    outer = _scope("a.py", 1, 100)
    inner = _scope("a.py", 10, 50)

    idx = LocalVarTypeIndex()
    idx.add(outer, "x", _ref("OuterClass", outer))
    idx.add(inner, "x", _ref("InnerClass", inner))
    tree = build_scope_tree([outer, inner])

    inner_hit = idx.find(inner, "x", tree)
    outer_hit = idx.find(outer, "x", tree)
    assert inner_hit is not None and inner_hit.raw_name == "InnerClass"
    assert outer_hit is not None and outer_hit.raw_name == "OuterClass"


def test_sibling_scopes_dont_see_each_other():
    """Two functions at the same level (no parent-child relationship)
    don't share typeBindings."""
    parent = _scope("a.py", 1, 100, kind="class")
    sibling_a = _scope("a.py", 10, 30)
    sibling_b = _scope("a.py", 40, 60)

    idx = LocalVarTypeIndex()
    idx.add(sibling_a, "a_only", _ref("A", sibling_a))
    idx.add(sibling_b, "b_only", _ref("B", sibling_b))
    tree = build_scope_tree([parent, sibling_a, sibling_b])

    assert idx.find(sibling_a, "b_only", tree) is None
    assert idx.find(sibling_b, "a_only", tree) is None


def test_walk_stops_at_root():
    """A scope with no parent + no binding returns None cleanly."""
    s = _scope("a.py", 1, 10)
    idx = LocalVarTypeIndex()  # empty
    tree = build_scope_tree([s])
    assert idx.find(s, "anything", tree) is None


# ─── last-write-wins on collision ───────────────────────────────────────────

def test_last_write_wins_at_same_scope():
    """Reassignment in the same scope replaces the type. Documented as
    the V1 approximation in the index docstring (flow-sensitive
    narrowing is explicitly deferred)."""
    s = _scope("a.py", 1, 10)
    idx = LocalVarTypeIndex()
    idx.add(s, "x", _ref("First", s))
    idx.add(s, "x", _ref("Second", s))
    tree = build_scope_tree([s])

    found = idx.find(s, "x", tree)
    assert found is not None
    assert found.raw_name == "Second"


# ─── len + contains hooks for diagnostic use ────────────────────────────────

def test_len_counts_total_bindings_across_scopes():
    s1 = _scope("a.py", 1, 10)
    s2 = _scope("a.py", 20, 30)
    idx = LocalVarTypeIndex()
    idx.add(s1, "x", _ref("X", s1))
    idx.add(s1, "y", _ref("Y", s1))
    idx.add(s2, "z", _ref("Z", s2))
    assert len(idx) == 3


def test_contains_does_not_walk_scope_chain():
    """`in` is a direct hit-check at exactly that scope. Tests rely on
    this to assert presence at a specific scope without conflating with
    the parent walk."""
    outer = _scope("a.py", 1, 100)
    inner = _scope("a.py", 10, 50)
    idx = LocalVarTypeIndex()
    idx.add(outer, "x", _ref("X", outer))
    # find() walks; __contains__ does not.
    tree = build_scope_tree([outer, inner])
    assert idx.find(inner, "x", tree) is not None
    assert (outer, "x") in idx
    assert (inner, "x") not in idx
