"""O(log n) lookup: at this byte position, what scope contains me?

Built once from a `ScopeTree`. The hot path is binary-search to find
the candidate root, then a linear walk down children — both bounded
by tree depth (typically <10 in real code). Mirrors GitNexus's
`position-index.ts`.

Used at finalize time (Sprint 12b) to map every reference site
("here's a CALL at byte 1234") to the scope that contains it; from
there the resolver walks up the tree looking for a binding.
"""
from __future__ import annotations

from bisect import bisect_right

from .scope_tree import ScopeTree, range_strictly_contains
from .types import Position, ScopeId


class PositionIndex:
    """Per-file sorted scope list + parent tree, queryable by position.

    For each file, we keep:
    - `_starts[file]`: list of `start_byte` values for that file's
      scopes, sorted ascending. `bisect_right` over this finds the
      rightmost scope whose start is <= the query position.
    - `_by_start[file]`: parallel list of `ScopeId`s (same order).

    Lookup picks the candidate, walks up its ancestors until one
    contains the position (handles the case where the bisect lands
    on a sibling that ends before the query), then walks *down* the
    tree refining to the innermost child that still contains the
    position.
    """

    def __init__(self, tree: ScopeTree) -> None:
        self._tree = tree
        self._starts: dict[str, list[int]] = {}
        self._by_start: dict[str, list[ScopeId]] = {}

    # ---- read API ----------------------------------------------------------

    def scope_at(self, position: Position) -> ScopeId | None:
        """Innermost scope containing `position`, or None if outside any scope.

        Returns None if the file has no indexed scopes, or if the
        position is past the end of every scope in the file.
        """
        starts = self._starts.get(position.file_path)
        if not starts:
            return None
        by_start = self._by_start[position.file_path]

        # bisect_right: index of first start strictly greater than offset.
        # The candidate is at index-1 (the rightmost start <= offset).
        idx = bisect_right(starts, position.byte_offset) - 1
        if idx < 0:
            return None

        candidate = by_start[idx]
        # The candidate's start is <= offset, but its end may be < offset
        # (we landed on a sibling that closes before the query). Walk up
        # parents until we find one whose end is past the offset, or we
        # run out of ancestors.
        cur: ScopeId | None = candidate
        while cur is not None:
            if cur.range.start_byte <= position.byte_offset < cur.range.end_byte:
                # Found a containing scope — now refine downward.
                return self._refine_innermost(cur, position.byte_offset)
            cur = self._tree.parent_of(cur)
        return None

    # ---- internals ---------------------------------------------------------

    def _refine_innermost(self, scope: ScopeId, offset: int) -> ScopeId:
        """Walk down children, picking the one containing `offset`, until none do."""
        cur = scope
        while True:
            next_child: ScopeId | None = None
            for child in self._tree.children_of(cur):
                if child.range.start_byte <= offset < child.range.end_byte:
                    next_child = child
                    break  # children are non-overlapping (tree invariant)
            if next_child is None:
                return cur
            cur = next_child


def build_position_index(tree: ScopeTree) -> PositionIndex:
    """Construct a `PositionIndex` from a `ScopeTree`.

    Per file: collect all scopes, sort by `start_byte`, and store
    parallel `_starts` / `_by_start` lists for `bisect`. Time: O(n log n)
    where n is total scope count.
    """
    idx = PositionIndex(tree)

    per_file: dict[str, list[ScopeId]] = {}
    for scope in tree.all_scopes():
        per_file.setdefault(scope.file_path, []).append(scope)

    for file_path, scopes in per_file.items():
        scopes.sort(key=lambda s: s.range.start_byte)
        idx._starts[file_path] = [s.range.start_byte for s in scopes]
        idx._by_start[file_path] = scopes

    return idx


__all__ = [
    "PositionIndex",
    "build_position_index",
]
