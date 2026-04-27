"""Sprint 12c — `MethodDispatchIndex` against synthetic Declarations.

Tests build Declarations + parent_relation by hand. No IndexBatch, no
extractor, no Neo4j — the dispatch index is pure structural data over
the 12a `Declaration` type.

Coverage:
  - own methods only (no inheritance)
  - linear chain inheritance (C -> B -> A)
  - override semantics (subclass method wins over ancestor's)
  - diamond inheritance (one parent's method wins; documented)
  - cycle protection (A -> B -> A doesn't infinite-loop)
  - unknown class lookup
  - parent_of for `super()` resolution
"""
from __future__ import annotations

from app.repo_indexer.scope_resolution.method_dispatch_index import (
    MethodDispatchIndex,
    build_method_dispatch_index,
)
from app.repo_indexer.scope_resolution.types import Declaration, Range


def _method(class_qn: str, method_name: str, file_path: str = "f.py") -> Declaration:
    return Declaration(
        qualified_name=f"{class_qn}.{method_name}",
        name=method_name,
        kind="method",
        file_path=file_path,
        range=Range(file_path=file_path, start_byte=0, end_byte=1),
        scope_id=None,
    )


# ---------------------------------------------------------------------------
# 1. Single class — own methods only
# ---------------------------------------------------------------------------

def test_dispatch_single_class_own_methods():
    methods = [
        _method("app.m.A", "foo"),
        _method("app.m.A", "bar"),
    ]
    idx = build_method_dispatch_index(methods, parent_relation={})

    own = idx.methods_for("app.m.A")
    assert set(own.keys()) == {"foo", "bar"}
    assert idx.resolve("app.m.A", "foo").qualified_name == "app.m.A.foo"
    assert idx.resolve("app.m.A", "missing") is None


# ---------------------------------------------------------------------------
# 2. Linear chain — C -> B -> A
# ---------------------------------------------------------------------------

def test_dispatch_linear_chain_inheritance():
    methods = [
        _method("app.m.A", "from_a"),
        _method("app.m.B", "from_b"),
        _method("app.m.C", "from_c"),
    ]
    parent_relation = {
        "app.m.B": ["app.m.A"],
        "app.m.C": ["app.m.B"],
    }
    idx = build_method_dispatch_index(methods, parent_relation)

    c_methods = idx.methods_for("app.m.C")
    assert set(c_methods.keys()) == {"from_a", "from_b", "from_c"}
    # Each method points at its declaring class.
    assert idx.resolve("app.m.C", "from_a").qualified_name == "app.m.A.from_a"
    assert idx.resolve("app.m.C", "from_b").qualified_name == "app.m.B.from_b"
    assert idx.resolve("app.m.C", "from_c").qualified_name == "app.m.C.from_c"


def test_dispatch_inherited_method_when_local_missing():
    """`C inherits B inherits A`. A defines `foo`; B and C don't.
    Resolving `C.foo` walks all the way to A.
    """
    methods = [_method("app.m.A", "foo")]
    parent_relation = {
        "app.m.B": ["app.m.A"],
        "app.m.C": ["app.m.B"],
    }
    idx = build_method_dispatch_index(methods, parent_relation)

    assert idx.resolve("app.m.C", "foo").qualified_name == "app.m.A.foo"
    assert idx.resolve("app.m.B", "foo").qualified_name == "app.m.A.foo"
    assert idx.resolve("app.m.A", "foo").qualified_name == "app.m.A.foo"


# ---------------------------------------------------------------------------
# 3. Override — subclass method wins
# ---------------------------------------------------------------------------

def test_dispatch_local_method_overrides_ancestor():
    methods = [
        _method("app.m.A", "foo"),
        _method("app.m.B", "foo"),
        _method("app.m.C", "foo"),
    ]
    parent_relation = {
        "app.m.B": ["app.m.A"],
        "app.m.C": ["app.m.B"],
    }
    idx = build_method_dispatch_index(methods, parent_relation)

    # C's own foo wins.
    assert idx.resolve("app.m.C", "foo").qualified_name == "app.m.C.foo"
    # B's own foo wins over A's when looking up from B.
    assert idx.resolve("app.m.B", "foo").qualified_name == "app.m.B.foo"


# ---------------------------------------------------------------------------
# 4. Diamond — C inherits A, B; both define foo. First-listed parent wins.
# ---------------------------------------------------------------------------

def test_dispatch_diamond_first_parent_wins():
    """Documented limitation: 12c picks the FIRST parent in the
    InheritsEdge-derived list. Real Python's MRO would pick by C3
    linearization which can differ; revisit when MRO matters.
    """
    methods = [
        _method("app.m.A", "foo"),
        _method("app.m.B", "foo"),
    ]
    # C inherits A first, then B.
    parent_relation = {"app.m.C": ["app.m.A", "app.m.B"]}
    idx = build_method_dispatch_index(methods, parent_relation)

    # First-listed parent (A) wins.
    assert idx.resolve("app.m.C", "foo").qualified_name == "app.m.A.foo"

    # Reverse order — B should win.
    parent_relation_2 = {"app.m.C": ["app.m.B", "app.m.A"]}
    idx2 = build_method_dispatch_index(methods, parent_relation_2)
    assert idx2.resolve("app.m.C", "foo").qualified_name == "app.m.B.foo"


# ---------------------------------------------------------------------------
# 5. Cycle — A -> B -> A. Must terminate.
# ---------------------------------------------------------------------------

def test_dispatch_cycle_does_not_infinite_loop():
    """Pathological inheritance cycle. Real Python rejects this (TypeError
    at class creation), but malformed extracted data could contain it.
    Visited-set guards in `_compute_merged` ensure termination.
    """
    methods = [
        _method("app.m.A", "from_a"),
        _method("app.m.B", "from_b"),
    ]
    parent_relation = {
        "app.m.A": ["app.m.B"],
        "app.m.B": ["app.m.A"],
    }
    idx = build_method_dispatch_index(methods, parent_relation)

    # Doesn't hang; both methods are visible from each side.
    a_methods = idx.methods_for("app.m.A")
    b_methods = idx.methods_for("app.m.B")
    assert "from_a" in a_methods
    assert "from_b" in a_methods
    assert "from_a" in b_methods
    assert "from_b" in b_methods


def test_dispatch_self_loop_terminates():
    """Class lists itself as parent. Pathological but must terminate."""
    methods = [_method("app.m.A", "foo")]
    parent_relation = {"app.m.A": ["app.m.A"]}
    idx = build_method_dispatch_index(methods, parent_relation)

    # Reaches own method without recursing forever.
    assert idx.resolve("app.m.A", "foo").qualified_name == "app.m.A.foo"


# ---------------------------------------------------------------------------
# 6. Unknown class — empty dict / None lookup
# ---------------------------------------------------------------------------

def test_dispatch_unknown_class_returns_empty():
    idx = build_method_dispatch_index([], parent_relation={})

    assert idx.methods_for("app.m.Nope") == {}
    assert idx.resolve("app.m.Nope", "foo") is None
    assert idx.parent_of("app.m.Nope") is None


# ---------------------------------------------------------------------------
# 7. parent_of — `super()` support
# ---------------------------------------------------------------------------

def test_dispatch_parent_of_returns_first_parent():
    methods: list[Declaration] = []
    parent_relation = {
        "app.m.C": ["app.m.B", "app.m.A"],
        "app.m.B": ["app.m.A"],
    }
    idx = build_method_dispatch_index(methods, parent_relation)

    # First-listed parent.
    assert idx.parent_of("app.m.C") == "app.m.B"
    assert idx.parent_of("app.m.B") == "app.m.A"
    # No parent.
    assert idx.parent_of("app.m.A") is None


def test_dispatch_parent_of_unknown_class():
    idx = build_method_dispatch_index([], parent_relation={})
    assert idx.parent_of("nothing") is None


# ---------------------------------------------------------------------------
# 8. Build-time API
# ---------------------------------------------------------------------------

def test_dispatch_skips_non_method_declarations():
    """Defensive: builder accepts only kind=='method' decls."""
    method = _method("app.m.A", "foo")
    not_a_method = Declaration(
        qualified_name="app.m.A.attr",
        name="attr",
        kind="function",  # not method
        file_path="f.py",
        range=Range(file_path="f.py", start_byte=0, end_byte=1),
        scope_id=None,
    )
    idx = build_method_dispatch_index([method, not_a_method], parent_relation={})

    a = idx.methods_for("app.m.A")
    assert "foo" in a
    assert "attr" not in a


def test_dispatch_methods_for_is_memoized():
    """Repeated lookups reuse the cached merged dict."""
    methods = [_method("app.m.A", "foo")]
    idx = build_method_dispatch_index(
        methods,
        parent_relation={"app.m.B": ["app.m.A"]},
    )

    first = idx.methods_for("app.m.B")
    second = idx.methods_for("app.m.B")
    # Same object back — memoized.
    assert first is second


def test_dispatch_index_class_can_construct_directly():
    """Sanity — `MethodDispatchIndex` can also be instantiated empty."""
    idx = MethodDispatchIndex()
    assert idx.methods_for("anything") == {}
