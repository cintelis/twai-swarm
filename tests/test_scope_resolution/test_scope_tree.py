"""Scope tree builder + predicates.

Synthetic 5-file fixture with nested scopes (module > class > methods >
blocks). No tree-sitter, no filesystem.
"""
from __future__ import annotations

import pytest

from app.repo_indexer.scope_resolution import (
    Range,
    ScopeId,
    ScopeTree,
    ScopeTreeInvariantError,
    build_scope_tree,
    format_range,
    range_strictly_contains,
    ranges_overlap,
    start_is_at_or_before,
)


# ---------------------------------------------------------------------------
# Helpers — keep tests readable
# ---------------------------------------------------------------------------

def r(file: str, start: int, end: int) -> Range:
    return Range(file_path=file, start_byte=start, end_byte=end)


def s(file: str, start: int, end: int, kind: str = "function") -> ScopeId:
    return ScopeId(file_path=file, range=r(file, start, end), kind=kind)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Range predicates
# ---------------------------------------------------------------------------

class TestRangePredicates:
    def test_overlap_same_file(self):
        assert ranges_overlap(r("a.py", 0, 10), r("a.py", 5, 15))
        assert ranges_overlap(r("a.py", 0, 10), r("a.py", 0, 10))
        assert not ranges_overlap(r("a.py", 0, 10), r("a.py", 10, 20))  # half-open
        assert not ranges_overlap(r("a.py", 0, 5), r("a.py", 6, 10))

    def test_overlap_cross_file_is_false(self):
        assert not ranges_overlap(r("a.py", 0, 100), r("b.py", 0, 100))

    def test_strict_contains(self):
        assert range_strictly_contains(r("a.py", 0, 100), r("a.py", 10, 20))
        assert range_strictly_contains(r("a.py", 0, 100), r("a.py", 0, 50))
        assert range_strictly_contains(r("a.py", 0, 100), r("a.py", 50, 100))
        # equal ranges: not strict
        assert not range_strictly_contains(r("a.py", 0, 100), r("a.py", 0, 100))
        # partial overlap: not contained
        assert not range_strictly_contains(r("a.py", 0, 50), r("a.py", 25, 75))
        # cross-file: never contains
        assert not range_strictly_contains(r("a.py", 0, 100), r("b.py", 10, 20))

    def test_start_is_at_or_before(self):
        assert start_is_at_or_before(r("a.py", 0, 10), r("a.py", 0, 5))
        assert start_is_at_or_before(r("a.py", 5, 10), r("a.py", 5, 10))
        assert start_is_at_or_before(r("a.py", 5, 10), r("a.py", 100, 200))
        assert not start_is_at_or_before(r("a.py", 100, 200), r("a.py", 5, 10))
        assert not start_is_at_or_before(r("a.py", 0, 10), r("b.py", 0, 5))

    def test_format_range(self):
        assert format_range(r("a/b.py", 10, 25)) == "a/b.py[10..25)"


# ---------------------------------------------------------------------------
# ScopeTree builder
# ---------------------------------------------------------------------------

class TestScopeTreeBuilder:
    def _five_file_fixture(self) -> tuple[ScopeTree, dict[str, ScopeId]]:
        """5-file repo with nested scopes (module > class > methods > blocks).

        Returns the tree and a name -> ScopeId dict for assertion ergonomics.
        """
        scopes = {
            # File a: module > Class > 2 methods, one with a nested block
            "a_mod": s("a.py", 0, 1000, "module"),
            "a_class": s("a.py", 50, 800, "class"),
            "a_m1": s("a.py", 100, 300, "function"),
            "a_m1_block": s("a.py", 200, 290, "block"),
            "a_m2": s("a.py", 400, 700, "function"),
            # File b: module with a top-level function only
            "b_mod": s("b.py", 0, 500, "module"),
            "b_fn": s("b.py", 100, 200, "function"),
            # File c: empty module (no children)
            "c_mod": s("c.py", 0, 100, "module"),
            # File d: deeply nested function chain
            "d_mod": s("d.py", 0, 500, "module"),
            "d_outer": s("d.py", 50, 450, "function"),
            "d_mid": s("d.py", 100, 400, "function"),
            "d_inner": s("d.py", 150, 350, "function"),
            # File e: module > 2 sibling classes
            "e_mod": s("e.py", 0, 600, "module"),
            "e_c1": s("e.py", 50, 250, "class"),
            "e_c2": s("e.py", 300, 550, "class"),
        }
        tree = build_scope_tree(scopes.values())
        return tree, scopes

    def test_parent_of_module_root(self):
        tree, sc = self._five_file_fixture()
        assert tree.parent_of(sc["a_mod"]) is None
        assert tree.parent_of(sc["b_mod"]) is None

    def test_parent_of_nested(self):
        tree, sc = self._five_file_fixture()
        assert tree.parent_of(sc["a_class"]) == sc["a_mod"]
        assert tree.parent_of(sc["a_m1"]) == sc["a_class"]
        assert tree.parent_of(sc["a_m1_block"]) == sc["a_m1"]
        assert tree.parent_of(sc["a_m2"]) == sc["a_class"]

    def test_children_in_document_order(self):
        tree, sc = self._five_file_fixture()
        # The class has two methods; m1 starts at 100, m2 at 400 — order matters.
        assert tree.children_of(sc["a_class"]) == (sc["a_m1"], sc["a_m2"])

    def test_children_of_leaf_is_empty(self):
        tree, sc = self._five_file_fixture()
        assert tree.children_of(sc["a_m1_block"]) == ()
        assert tree.children_of(sc["c_mod"]) == ()

    def test_deep_nesting(self):
        tree, sc = self._five_file_fixture()
        assert tree.parent_of(sc["d_outer"]) == sc["d_mod"]
        assert tree.parent_of(sc["d_mid"]) == sc["d_outer"]
        assert tree.parent_of(sc["d_inner"]) == sc["d_mid"]
        assert tree.children_of(sc["d_inner"]) == ()

    def test_sibling_classes(self):
        tree, sc = self._five_file_fixture()
        assert tree.children_of(sc["e_mod"]) == (sc["e_c1"], sc["e_c2"])
        assert tree.parent_of(sc["e_c1"]) == sc["e_mod"]
        assert tree.parent_of(sc["e_c2"]) == sc["e_mod"]

    def test_roots_per_file(self):
        tree, sc = self._five_file_fixture()
        assert tree.roots_for("a.py") == (sc["a_mod"],)
        assert tree.roots_for("b.py") == (sc["b_mod"],)
        assert tree.roots_for("c.py") == (sc["c_mod"],)
        assert tree.roots_for("nonexistent.py") == ()

    def test_contains_predicate(self):
        tree, sc = self._five_file_fixture()
        # Strict: a_class contains a_m1 (smaller range)
        assert tree.contains(sc["a_class"], sc["a_m1"])
        # And the module contains the class
        assert tree.contains(sc["a_mod"], sc["a_class"])
        # Cross-file: never contains
        assert not tree.contains(sc["a_mod"], sc["b_fn"])

    def test_membership(self):
        tree, sc = self._five_file_fixture()
        assert sc["a_class"] in tree
        assert s("z.py", 0, 1) not in tree

    def test_unknown_scope_raises_keyerror(self):
        tree, _ = self._five_file_fixture()
        with pytest.raises(KeyError):
            tree.parent_of(s("unknown.py", 0, 10))

    def test_idempotent_on_duplicates(self):
        scope = s("a.py", 0, 100, "module")
        tree = build_scope_tree([scope, scope, scope])
        assert tree.parent_of(scope) is None
        assert len(tree.all_scopes()) == 1


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

class TestScopeTreeInvariants:
    def test_overlap_without_contain_raises(self):
        # Two scopes with partial overlap — illegal.
        bad = [
            s("a.py", 0, 50, "function"),
            s("a.py", 25, 75, "function"),
        ]
        with pytest.raises(ScopeTreeInvariantError) as exc:
            build_scope_tree(bad)
        msg = str(exc.value)
        assert "a.py" in msg
        assert "overlap" in msg.lower()

    def test_identical_ranges_raise(self):
        # Two scopes with the exact same range, different kinds — ambiguous.
        bad = [
            s("a.py", 0, 100, "module"),
            s("a.py", 0, 100, "function"),
        ]
        with pytest.raises(ScopeTreeInvariantError) as exc:
            build_scope_tree(bad)
        assert "identical range" in str(exc.value)

    def test_disjoint_ranges_are_fine(self):
        # Two completely separate scopes in the same file — totally valid
        # (e.g. two top-level functions in a module, but no module scope here).
        ok = [
            s("a.py", 0, 50, "function"),
            s("a.py", 100, 150, "function"),
        ]
        tree = build_scope_tree(ok)
        assert tree.parent_of(ok[0]) is None
        assert tree.parent_of(ok[1]) is None
