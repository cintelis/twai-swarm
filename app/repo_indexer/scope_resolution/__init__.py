"""Scope resolution — pure data structures + indexes for cross-file resolution.

Sprint 12a foundation. Standalone package: zero tree-sitter, zero Neo4j,
zero `app.repo_indexer.*` runtime dependency. Built to be consumed by
Sprint 12b's `finalize.py` (the actual resolver) and 12c's
`method_dispatch_index.py`.

Mirrors the shape of GitNexus's `gitnexus-shared/src/scope-resolution/`
package; see `repo-indexer-future-state.md` §2.2.
"""
from .module_scope_index import ModuleScopeIndex, build_module_scope_index
from .position_index import PositionIndex, build_position_index
from .qualified_name_index import QualifiedNameIndex, build_qualified_name_index
from .scope_tree import (
    ScopeTree,
    ScopeTreeInvariantError,
    build_scope_tree,
    format_range,
    range_strictly_contains,
    ranges_overlap,
    start_is_at_or_before,
)
from .types import Declaration, Position, Range, ScopeId

__all__ = [
    # types
    "Position",
    "Range",
    "ScopeId",
    "Declaration",
    # scope tree + helpers
    "ScopeTree",
    "ScopeTreeInvariantError",
    "build_scope_tree",
    "ranges_overlap",
    "range_strictly_contains",
    "start_is_at_or_before",
    "format_range",
    # position index
    "PositionIndex",
    "build_position_index",
    # module scope index
    "ModuleScopeIndex",
    "build_module_scope_index",
    # qualified name index
    "QualifiedNameIndex",
    "build_qualified_name_index",
]
