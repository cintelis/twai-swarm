"""Position-index lookup — innermost-scope-at-offset, with a perf smoke test."""
from __future__ import annotations

import time

from app.repo_indexer.scope_resolution import (
    Position,
    Range,
    ScopeId,
    build_position_index,
    build_scope_tree,
)


def r(file: str, start: int, end: int) -> Range:
    return Range(file_path=file, start_byte=start, end_byte=end)


def s(file: str, start: int, end: int, kind: str = "function") -> ScopeId:
    return ScopeId(file_path=file, range=r(file, start, end), kind=kind)  # type: ignore[arg-type]


def p(file: str, offset: int) -> Position:
    return Position(file_path=file, byte_offset=offset)


class TestInnermostLookup:
    def _fixture(self):
        # module(0..1000) > class(50..800) > [m1(100..300) > block(200..290),
        #                                     m2(400..700)]
        scopes = {
            "mod": s("a.py", 0, 1000, "module"),
            "cls": s("a.py", 50, 800, "class"),
            "m1": s("a.py", 100, 300, "function"),
            "blk": s("a.py", 200, 290, "block"),
            "m2": s("a.py", 400, 700, "function"),
        }
        tree = build_scope_tree(scopes.values())
        return build_position_index(tree), scopes

    def test_inside_innermost_block(self):
        idx, sc = self._fixture()
        assert idx.scope_at(p("a.py", 250)) == sc["blk"]

    def test_inside_method_but_outside_block(self):
        idx, sc = self._fixture()
        # 150 is inside m1 but before the block (200..290).
        assert idx.scope_at(p("a.py", 150)) == sc["m1"]

    def test_inside_class_but_between_methods(self):
        idx, sc = self._fixture()
        # 350 is inside cls but between m1 and m2.
        assert idx.scope_at(p("a.py", 350)) == sc["cls"]

    def test_inside_module_but_outside_class(self):
        idx, sc = self._fixture()
        # 25 is inside the module but before the class starts (50..800).
        assert idx.scope_at(p("a.py", 25)) == sc["mod"]
        # 900 is inside the module but after the class ends.
        assert idx.scope_at(p("a.py", 900)) == sc["mod"]

    def test_outside_any_scope_returns_none(self):
        idx, _ = self._fixture()
        # past end of module
        assert idx.scope_at(p("a.py", 5000)) is None

    def test_unknown_file_returns_none(self):
        idx, _ = self._fixture()
        assert idx.scope_at(p("does-not-exist.py", 50)) is None

    def test_boundary_offsets(self):
        """Half-open: end_byte is exclusive."""
        idx, sc = self._fixture()
        # Offset 0 is start of module — should be inside.
        assert idx.scope_at(p("a.py", 0)) == sc["mod"]
        # Offset 999 is last byte inside module (1000 is exclusive).
        assert idx.scope_at(p("a.py", 999)) == sc["mod"]
        # Offset 1000 is just past the end.
        assert idx.scope_at(p("a.py", 1000)) is None
        # Offset 200 is start of block — block is innermost.
        assert idx.scope_at(p("a.py", 200)) == sc["blk"]
        # Offset 290 is past the block; m1 (which extends to 300) wins.
        assert idx.scope_at(p("a.py", 290)) == sc["m1"]


class TestPerformance:
    """Loose smoke test: 1000 lookups over 100 scopes complete in <100ms.

    Catches accidental O(n²) regressions if someone replaces the bisect
    with a linear scan. Bound is generous enough to survive a slow CI box.
    """

    def test_lookup_perf(self):
        # 100 nested scopes in a single file, each 100 bytes deep. Build:
        #   module(0..10000) > s1(50..9950) > s2(100..9900) > ... etc.
        # Plus a fan of 50 sibling functions inside the deepest scope.
        scopes: list[ScopeId] = []
        # 50 nested wrappers
        for i in range(50):
            scopes.append(s("perf.py", i * 10, 10000 - i * 10, "function"))
        # 50 siblings inside the deepest wrapper, at offsets 500, 600, ...
        deepest_start = 50 * 10  # 500
        for i in range(50):
            base = deepest_start + 50 + i * 100
            scopes.append(s("perf.py", base, base + 50, "block"))

        tree = build_scope_tree(scopes)
        idx = build_position_index(tree)

        # 1000 lookups at varying offsets.
        offsets = [(i * 9) % 10000 for i in range(1000)]
        start = time.perf_counter()
        for off in offsets:
            idx.scope_at(p("perf.py", off))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"1000 lookups took {elapsed:.3f}s (>100ms)"
