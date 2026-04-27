"""Method dispatch — resolve method names through a class's inheritance chain.

Sprint 12c addition. Pure structural index, like the other 12a indexes:
no tree-sitter, no Neo4j, no `app.repo_indexer.*` runtime dependency.
Inputs are a flat list of method `Declaration`s + a parent-of relation
derived from `InheritsEdge`s; output is a class -> {method_name -> Declaration}
lookup that walks ancestors.

Override semantics: a class's own method always wins over any ancestor's
declaration of the same name. This matches Python's resolution order for
direct inheritance.

Multiple inheritance: 12c does NOT implement C3 / proper MRO. If a method
exists on multiple ancestors and isn't redefined locally, the ancestor
listed first in the `InheritsEdge`-derived `parent_relation` wins. Same
result as Python's MRO for the no-diamond cases that dominate real
codebases; differs from Python only for true diamond inheritance, which
is rare. Documented limitation; revisit when MRO becomes important
(Sprint 13+).

Cycle protection: a `visited` set guards the recursive walk so a
pathological inheritance cycle (which shouldn't occur in real Python but
could in malformed extracted data) doesn't infinite-loop.

This module is consumed by `finalize.py` to add three resolution paths
on top of the 12b pipeline:
    - `self.method()` -> `EnclosingClass.method`, walking ancestors.
    - `super().method()` -> `EnclosingClass`'s parent's `method`, one
      class up (no MRO).
    - `param: T; param.method()` where `T.method` doesn't exist locally
      on `T` but is inherited from `T`'s ancestor.
"""
from __future__ import annotations

from typing import Iterable

from .types import Declaration


class MethodDispatchIndex:
    """For each class, the resolved method set including inherited methods.

    Built once per `finalize_batch` run from the batch's method
    Declarations + the `InheritsEdge`-derived parent relation. Read-only
    after construction; safe to share across resolver helpers in finalize.
    """

    def __init__(self) -> None:
        # Local methods declared on each class — populated at build time
        # from the input Declarations grouped by `parent_class_qn` (which
        # the adapter encodes via Declaration.qualified_name's dotted prefix).
        self._local_methods: dict[str, dict[str, Declaration]] = {}
        # child_qn -> ordered list of resolved parent_qns. Resolved
        # against the QualifiedNameIndex by the caller, so external bases
        # (Symbol nodes — third-party / stdlib) don't appear here. They
        # have no methods we can index, and walking through them would
        # produce no resolutions anyway.
        self._parents: dict[str, list[str]] = {}
        # Memoized full method set per class — built lazily on first
        # `methods_for(C)` call so we don't pay for classes we never look
        # up. Ancestors share method dicts (no copy on merge), so the
        # memory cost is O(unique-methods) not O(classes * methods).
        self._merged_cache: dict[str, dict[str, Declaration]] = {}

    # ---- read API ----------------------------------------------------------

    def methods_for(self, class_qn: str) -> dict[str, Declaration]:
        """Return `{method_name: Declaration}` merged across this class
        and all reachable ancestors. Empty dict if `class_qn` is unknown.

        Local methods override ancestor methods (`dict.update` semantics
        applied bottom-up). Result is memoized per class_qn — repeated
        lookups are O(1).
        """
        cached = self._merged_cache.get(class_qn)
        if cached is not None:
            return cached
        merged = self._compute_merged(class_qn, visited=set())
        self._merged_cache[class_qn] = merged
        return merged

    def resolve(self, class_qn: str, method_name: str) -> Declaration | None:
        """Method declaration for `class_qn.method_name`, walking
        ancestors. None if neither the class nor any ancestor defines it.
        """
        return self.methods_for(class_qn).get(method_name)

    def parent_of(self, class_qn: str) -> str | None:
        """Immediate (first-listed) parent's qn, or None if no parent.

        Used by `super().method()` resolution — Sprint 12c picks the
        first parent unconditionally; proper MRO is deferred. For a class
        with no `InheritsEdge`s or whose only parents are unresolved
        externals, this returns None and the caller falls back to
        emitting a SymbolNode.
        """
        parents = self._parents.get(class_qn)
        if not parents:
            return None
        return parents[0]

    # ---- internals ---------------------------------------------------------

    def _compute_merged(
        self,
        class_qn: str,
        *,
        visited: set[str],
    ) -> dict[str, Declaration]:
        """Walk parents depth-first, accumulating method dicts.

        The `visited` set short-circuits cycles (A inherits B inherits A —
        which Python rejects but extracted data could contain). Order:
        ancestors first (so their methods land in the dict), then locals
        last (overriding). For multiple parents, earlier parents are
        visited later (so their methods take priority among ancestors)
        — `dict.update` lets later `update` calls win.
        """
        if class_qn in visited:
            return {}
        visited.add(class_qn)

        merged: dict[str, Declaration] = {}
        # Walk parents in REVERSE order, so the first parent's update
        # runs LAST among ancestors and its methods win conflicts. (Local
        # methods still override every ancestor below.)
        parents = self._parents.get(class_qn, ())
        for parent_qn in reversed(parents):
            ancestor_methods = self._compute_merged(parent_qn, visited=visited)
            merged.update(ancestor_methods)

        # Local methods last — override any inherited entries with the
        # same name.
        merged.update(self._local_methods.get(class_qn, {}))
        return merged


def build_method_dispatch_index(
    method_declarations: Iterable[Declaration],
    parent_relation: dict[str, list[str]],
) -> MethodDispatchIndex:
    """Build a `MethodDispatchIndex` from method Declarations + parent map.

    Args:
        method_declarations: Declarations whose `kind == "method"`. Each
            decl's `qualified_name` is `<class_qn>.<method_name>`; the
            class_qn is recovered by stripping the last dotted component.
            Method declarations whose qn lacks a dot are dropped (they
            can't be associated with a class).
        parent_relation: `child_qn -> [parent_qn, ...]` where every
            parent_qn already resolves to a known class (the caller —
            `_adapter.to_parent_relation` — filters externals out via
            the QualifiedNameIndex). InheritsEdge order is preserved so
            "first parent wins" semantics are deterministic across runs.

    Returns:
        A populated `MethodDispatchIndex` ready for `methods_for` /
        `resolve` / `parent_of` queries.
    """
    idx = MethodDispatchIndex()

    for decl in method_declarations:
        if decl.kind != "method":
            # Defensive — caller already filters, but spec is explicit.
            continue
        qn = decl.qualified_name
        if "." not in qn:
            continue
        class_qn = qn.rsplit(".", 1)[0]
        idx._local_methods.setdefault(class_qn, {})[decl.name] = decl

    # Copy parent_relation so post-construction mutation by the caller
    # doesn't bleed into the index. List values are kept by-reference but
    # finalize never mutates them.
    for child_qn, parents in parent_relation.items():
        if parents:
            idx._parents[child_qn] = list(parents)

    return idx


__all__ = [
    "MethodDispatchIndex",
    "build_method_dispatch_index",
]
