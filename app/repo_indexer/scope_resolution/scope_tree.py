"""Per-file forest of nested lexical scopes.

A `ScopeTree` is the structural backbone the other indexes hang off:
- `position_index` walks down the tree to find the innermost scope at a byte offset.
- 12b's finalize algorithm will walk *up* the tree to resolve references.

This module is pure stdlib. It accepts a flat list of `ScopeId`s
(extractors will hand it one per scope they observe) and infers the
parent/child relationships from byte-range containment alone — no
tree-sitter coupling, no parser-specific shape knowledge.

Mirrors GitNexus's `scope-tree.ts` in shape:
- `ScopeTreeInvariantError` for corruption (overlap-without-contain).
- `ranges_overlap`, `range_strictly_contains`, `start_is_at_or_before`,
  `format_range` as module-level helpers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import Range, ScopeId


class ScopeTreeInvariantError(ValueError):
    """Raised when scope ranges break the strict-nesting invariant.

    A sane AST gives us scopes that either nest (one fully contains the
    other) or are disjoint (no overlap). Any other relationship —
    partial overlap — is a corruption signal: either the extractor is
    handing us bad ranges, or two unrelated scopes were given the same
    byte span. Either way, we'd rather fail loudly here than build a
    quietly-wrong index.

    Also raised by `qualified_name_index` on QN collision; both shapes
    of corruption are "the input data is inconsistent."
    """


# ---------------------------------------------------------------------------
# Range predicates (module-level helpers, mirror of GitNexus naming)
# ---------------------------------------------------------------------------

def ranges_overlap(a: Range, b: Range) -> bool:
    """True if `a` and `b` share at least one byte (and same file).

    Different files never overlap. Half-open intervals: end is exclusive.
    """
    if a.file_path != b.file_path:
        return False
    return a.start_byte < b.end_byte and b.start_byte < a.end_byte


def range_strictly_contains(outer: Range, inner: Range) -> bool:
    """True if `inner` is fully inside `outer` AND they aren't identical.

    Strict in the GitNexus sense: equal ranges don't contain each other.
    Used to infer parent/child during tree construction — the parent's
    range must strictly contain the child's.
    """
    if outer.file_path != inner.file_path:
        return False
    if outer.start_byte == inner.start_byte and outer.end_byte == inner.end_byte:
        return False
    return outer.start_byte <= inner.start_byte and inner.end_byte <= outer.end_byte


def start_is_at_or_before(a: Range, b: Range) -> bool:
    """True if `a` starts at or before `b` in the same file.

    Sort key for placing scopes in document order. Different-file
    comparisons are false (caller should never compare cross-file).
    """
    if a.file_path != b.file_path:
        return False
    return a.start_byte <= b.start_byte


def format_range(r: Range) -> str:
    """Human-readable rendering for error messages."""
    return f"{r.file_path}[{r.start_byte}..{r.end_byte})"


# ---------------------------------------------------------------------------
# ScopeTree
# ---------------------------------------------------------------------------

class ScopeTree:
    """Forest of nested scopes, one tree per file.

    Built once via `build_scope_tree`, queried many times. Internal
    state is mutable so the builder can populate it efficiently; once
    handed to callers it should be treated as read-only (no public
    mutation API).
    """

    def __init__(self) -> None:
        # parent_of[child] = parent_scope_id (None for roots)
        self._parent: dict[ScopeId, ScopeId | None] = {}
        # children_of[parent] = list of children, in document order
        self._children: dict[ScopeId, list[ScopeId]] = defaultdict(list)
        # roots per file, in document order
        self._roots: dict[str, list[ScopeId]] = defaultdict(list)
        # all scopes (so we can answer "do you know about this id?")
        self._all: set[ScopeId] = set()

    # ---- read API ----------------------------------------------------------

    def parent_of(self, scope_id: ScopeId) -> ScopeId | None:
        """The immediate parent scope, or None if `scope_id` is a root.

        Raises KeyError if `scope_id` was never added to this tree —
        callers should check `scope_id in tree` first if they're not sure.
        """
        if scope_id not in self._all:
            raise KeyError(f"unknown scope: {format_range(scope_id.range)}")
        return self._parent.get(scope_id)

    def children_of(self, scope_id: ScopeId) -> tuple[ScopeId, ...]:
        """Direct children, in document order. Empty tuple for leaves."""
        if scope_id not in self._all:
            raise KeyError(f"unknown scope: {format_range(scope_id.range)}")
        return tuple(self._children.get(scope_id, ()))

    def contains(self, parent: ScopeId, child: ScopeId) -> bool:
        """Strict containment by range — same predicate the builder uses.

        Note: this is the *range* predicate, not the *tree-edge* predicate.
        A scope's grandparent also "contains" it, even though the tree
        edge skips a level. Use `parent_of` / `children_of` for tree edges.
        """
        return range_strictly_contains(parent.range, child.range)

    def roots_for(self, file_path: str) -> tuple[ScopeId, ...]:
        """Top-level scopes in `file_path`, in document order.

        For a well-formed Python or TypeScript file, this is normally a
        single module-level scope — but we don't enforce that, since
        some extractor configurations might emit multiple roots per file
        (e.g. a TypeScript namespace file).
        """
        return tuple(self._roots.get(file_path, ()))

    def all_scopes(self) -> tuple[ScopeId, ...]:
        """Every scope in the tree. Order is unspecified; callers that
        need document order should walk roots + children.
        """
        return tuple(self._all)

    def __contains__(self, scope_id: ScopeId) -> bool:
        return scope_id in self._all


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_scope_tree(scopes: Iterable[ScopeId]) -> ScopeTree:
    """Build a `ScopeTree` from a flat list of scopes.

    Algorithm: sort scopes per-file by `(start_byte, -end_byte)` — that
    ordering puts outer scopes before any inner scope they contain, and
    among same-start scopes the wider one first. Then walk the sorted
    list maintaining a stack of "open" scopes; for each new scope, pop
    until the stack-top strictly contains it; the top is then its parent
    (or it's a root if the stack is empty).

    Validates as it goes: any scope that overlaps the stack-top without
    being strictly contained is a corruption signal — raises
    `ScopeTreeInvariantError`. Two scopes with identical ranges in the
    same file are also rejected (no clean way to assign parent/child).

    Time: O(n log n) for the sort, O(n) for the walk.
    """
    tree = ScopeTree()

    # Group scopes by file so each file's traversal is independent.
    per_file: dict[str, list[ScopeId]] = defaultdict(list)
    for s in scopes:
        if s in tree._all:
            # Caller passed a duplicate. Idempotent — silently dedupe.
            continue
        tree._all.add(s)
        per_file[s.file_path].append(s)

    for file_path, file_scopes in per_file.items():
        # Outer-first ordering: smaller start, then larger end (wider).
        file_scopes.sort(key=lambda s: (s.range.start_byte, -s.range.end_byte))

        # Reject identical-range duplicates in the same file: they have
        # the same byte span but different `kind`, which is genuinely
        # ambiguous — we can't decide which is the parent.
        for i in range(1, len(file_scopes)):
            prev = file_scopes[i - 1].range
            cur = file_scopes[i].range
            if prev.start_byte == cur.start_byte and prev.end_byte == cur.end_byte:
                raise ScopeTreeInvariantError(
                    f"two scopes share an identical range in {file_path}: "
                    f"{format_range(prev)} (kinds: "
                    f"{file_scopes[i - 1].kind!r} vs {file_scopes[i].kind!r})"
                )

        stack: list[ScopeId] = []
        for s in file_scopes:
            # Pop any open scopes that don't strictly contain this one.
            while stack and not range_strictly_contains(stack[-1].range, s.range):
                top = stack.pop()
                # If the top OVERLAPS but doesn't contain, that's corruption.
                if ranges_overlap(top.range, s.range):
                    raise ScopeTreeInvariantError(
                        f"scopes overlap without containment in {file_path}: "
                        f"{format_range(top.range)} and {format_range(s.range)}"
                    )

            if stack:
                parent = stack[-1]
                tree._parent[s] = parent
                tree._children[parent].append(s)
            else:
                tree._parent[s] = None
                tree._roots[file_path].append(s)

            stack.append(s)

    return tree


__all__ = [
    "ScopeTree",
    "ScopeTreeInvariantError",
    "build_scope_tree",
    "ranges_overlap",
    "range_strictly_contains",
    "start_is_at_or_before",
    "format_range",
]
