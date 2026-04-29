"""Sprint 14g — typeBindings index for receiver-type resolution.

Closes the variable-method-call gap that Sprint 12c deferred. Today's
`MethodDispatchIndex` only resolves when the receiver carries a
parameter type annotation (`def f(x: User)`). It misses:

    builder = StateGraph(MyState)   # ← receiver bound by assignment
    builder.add_node("start", fn)   # ← we want this to resolve

This index records `{(scope, name) -> TypeRef}`, walks the scope chain
upward at lookup time. Modeled directly after GitNexus's
`scope.typeBindings` map + `findReceiverTypeBinding` walker
(`gitnexus/src/core/ingestion/scope-resolution/scope/walkers.ts`).

Reused for case 0 (compound receivers): class-field types like
`self.x = User()` are stored on the class scope, so the same lookup
walker resolves `obj.attr` chains by descending from `obj`'s class
scope.

Pure stdlib. No tree-sitter, no Neo4j. The extractor populates the
index via `LocalVarBinding` records during the parse phase; the
resolver consults it during finalize.
"""
from __future__ import annotations

from .scope_tree import ScopeTree
from .types import ScopeId, TypeRef


class LocalVarTypeIndex:
    """Map `(scope, name) -> TypeRef` with scope-chain lookup.

    Storage is a dict-of-dicts so the per-scope inner map is sparse
    (most scopes bind zero variables — only the ones with assignments
    have entries). Lookup walks parent scopes via `ScopeTree.parent_of`
    and stops at the first hit, mirroring lexical-scope semantics.
    """

    def __init__(self) -> None:
        # bindings[scope_id][var_name] = TypeRef
        self._bindings: dict[ScopeId, dict[str, TypeRef]] = {}

    def add(self, scope_id: ScopeId, var_name: str, type_ref: TypeRef) -> None:
        """Register a typeBinding at `scope_id`.

        Last-write-wins on collision: a function that reassigns `x` to a
        different type ends up with the LAST binding. Flow-sensitive
        narrowing is explicitly deferred — for now we accept the
        last-assignment-wins approximation. In practice, code that
        re-binds a variable to incompatible types is rare in agent /
        framework code; the dominant pattern is one assignment per name.
        """
        scope_map = self._bindings.get(scope_id)
        if scope_map is None:
            scope_map = {}
            self._bindings[scope_id] = scope_map
        scope_map[var_name] = type_ref

    def find(
        self, start_scope: ScopeId, var_name: str, tree: ScopeTree,
    ) -> TypeRef | None:
        """Walk up the scope chain looking for `var_name`. Returns the
        first TypeRef found (innermost wins) or None.

        `start_scope` must be a scope that exists in `tree`; KeyError
        propagates if not. Cycle detection isn't needed because
        `ScopeTree`'s build invariants prevent cycles in the parent
        relation.
        """
        current: ScopeId | None = start_scope
        while current is not None:
            scope_map = self._bindings.get(current)
            if scope_map is not None:
                ref = scope_map.get(var_name)
                if ref is not None:
                    return ref
            current = tree.parent_of(current)
        return None

    def __len__(self) -> int:
        """Number of (scope, name) bindings registered. Diagnostic helper."""
        return sum(len(m) for m in self._bindings.values())

    def __contains__(self, key: tuple[ScopeId, str]) -> bool:
        """Direct hit-check at exactly `scope_id` (NO scope-chain walk).
        Use `find` for the lookup walker; this helper exists for tests
        that want to assert presence at a specific scope."""
        scope_id, var_name = key
        scope_map = self._bindings.get(scope_id)
        return scope_map is not None and var_name in scope_map


__all__ = ["LocalVarTypeIndex"]
