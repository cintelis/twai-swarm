"""Adapter — bridge `IndexBatch` shape to `scope_resolution` types.

The four indexes from Sprint 12a (`scope_tree`, `position_index`,
`qualified_name_index`, `module_scope_index`) are pure structural
algorithms over `Declaration` / `ScopeId` / `Range`. This module is the
one place that knows how to convert the indexer's `IndexBatch` records
(FunctionNode, ClassNode, ModuleNode) into those types.

Kept private (`_adapter`) so 12c's method-dispatch index — which will
also need the same conversion — has a single import target rather than
re-deriving the mapping inline. Public entry points: `to_declarations`,
`to_scopes`.

Range encoding: we use line numbers as the int values for `start_byte`
and `end_byte`. The 12a predicates only need monotonic int ordering for
strict containment — they don't care whether the ints represent bytes
or lines. The extractor emits 1-based inclusive line numbers; we feed
those through unchanged. `range_strictly_contains` rejects equal ranges,
so two scopes that happen to share the same line span will fail the
ScopeTree invariant — same shape as a byte-range collision.
"""
from __future__ import annotations

from ..actions import IndexBatch
from .local_var_type_index import LocalVarTypeIndex
from .qualified_name_index import QualifiedNameIndex
from .types import Declaration, Range, ScopeId, TypeRef


def _range_for(file_path: str, line_start: int, line_end: int) -> Range:
    # Line numbers used as ints in start_byte/end_byte. The 12a predicates
    # only need monotonic int ordering for containment — bytes vs lines is
    # immaterial to them. end_byte is treated half-open (matches the
    # ScopeTree convention), so we add 1 to the inclusive end line.
    return Range(file_path=file_path, start_byte=line_start, end_byte=line_end + 1)


def to_declarations(batch: IndexBatch) -> list[Declaration]:
    """Convert FunctionNodes / ClassNodes / ModuleNodes to Declarations.

    Method declarations carry a `scope_id` pointing at their enclosing
    class scope (looked up by `parent_class_qn`); top-level functions
    and classes get `scope_id=None`. ModuleNodes also produce
    Declarations (kind="module") so the qualified-name index can answer
    "is this qn a known module?" queries.
    """
    # Pre-build a map of class_qn -> ScopeId so methods can name their
    # enclosing scope without a second pass.
    class_scope_by_qn: dict[str, ScopeId] = {}
    for c in batch.classes:
        class_scope_by_qn[c.qualified_name] = ScopeId(
            file_path=c.file_path,
            range=_range_for(c.file_path, c.line_start, c.line_end),
            kind="class",
        )

    out: list[Declaration] = []

    for m in batch.modules:
        # Modules don't have line ranges in our extractor output. Encode a
        # zero-width range at line 0 — it's outside any function/class
        # range and won't collide. Module declarations are excluded from
        # ScopeTree (see to_scopes), so the range only matters for the
        # qualified-name index, which doesn't touch ranges.
        out.append(Declaration(
            qualified_name=m.qualified_name,
            name=m.qualified_name.rsplit(".", 1)[-1] if "." in m.qualified_name else m.qualified_name,
            kind="module",
            file_path=m.file_path,
            range=Range(file_path=m.file_path, start_byte=0, end_byte=0),
            scope_id=None,
        ))

    for c in batch.classes:
        out.append(Declaration(
            qualified_name=c.qualified_name,
            name=c.name,
            kind="class",
            file_path=c.file_path,
            range=_range_for(c.file_path, c.line_start, c.line_end),
            scope_id=None,
        ))

    for f in batch.functions:
        scope_id = None
        if f.is_method and f.parent_class_qn:
            scope_id = class_scope_by_qn.get(f.parent_class_qn)
        out.append(Declaration(
            qualified_name=f.qualified_name,
            name=f.name,
            kind="method" if f.is_method else "function",
            file_path=f.file_path,
            range=_range_for(f.file_path, f.line_start, f.line_end),
            scope_id=scope_id,
        ))

    return out


# Sprint 14i — Module scopes use a sentinel range that's wider than any
# real function/class scope, so they always strictly contain their
# children in the scope tree. The half-open end (10**9) is well past any
# realistic file's line count and matches the encoding `_range_for`
# produces from `enclosing_line_end = MODULE_SCOPE_END_LINE - 1`.
MODULE_SCOPE_END = 10**9


def to_scopes(batch: IndexBatch) -> list[ScopeId]:
    """Flat list of ScopeIds — one per ModuleNode, one per ClassNode,
    one per FunctionNode.

    Sprint 14i: ModuleNodes now produce Module scopes (kind="module")
    with a sentinel `(0, MODULE_SCOPE_END)` range so they strictly
    contain every class/function in the file. This gives the scope tree
    a root per file under which class+function scopes nest as children,
    and gives module-level typeBindings somewhere to live. Pre-14i,
    modules were absent from the scope tree.

    `build_scope_tree` consumes this list and rejects identical ranges
    via ScopeTreeInvariantError; let that propagate (same contract as
    the 12a indexes).
    """
    out: list[ScopeId] = []
    seen_module_files: set[str] = set()
    for m in batch.modules:
        # Defensive: a batch may legitimately have multiple ModuleNodes
        # for the same file (re-scans, fragment merges); dedupe by path.
        if m.file_path in seen_module_files:
            continue
        seen_module_files.add(m.file_path)
        out.append(ScopeId(
            file_path=m.file_path,
            range=Range(
                file_path=m.file_path,
                start_byte=0,
                end_byte=MODULE_SCOPE_END,
            ),
            kind="module",
        ))
    for c in batch.classes:
        out.append(ScopeId(
            file_path=c.file_path,
            range=_range_for(c.file_path, c.line_start, c.line_end),
            kind="class",
        ))
    for f in batch.functions:
        out.append(ScopeId(
            file_path=f.file_path,
            range=_range_for(f.file_path, f.line_start, f.line_end),
            kind="function",
        ))
    return out


def to_parent_relation(
    batch: IndexBatch,
    qn_index: QualifiedNameIndex,
) -> dict[str, list[str]]:
    """Build `child_qn -> [parent_qn, ...]` from `batch.inherits`.

    Sprint 12c — input to `build_method_dispatch_index`. Edges whose
    `parent_qn` doesn't resolve to a known class in `qn_index` are
    skipped (those parents are external — third-party / stdlib / not yet
    resolved through imports — and have no methods we can index from
    repo data). Note this means parents that *would* resolve through
    finalize's import chain but haven't been rewritten yet are also
    skipped: dispatch resolution can't find inherited methods through
    those edges. That gap is the first thing to revisit if Sprint 13's
    MRO work moves the dispatch index after import-edge rewriting.

    InheritsEdge order is preserved within the per-child list — first
    `inherits` entry stays first — so "first parent wins" semantics in
    the dispatch index are deterministic.
    """
    parent_relation: dict[str, list[str]] = {}
    for edge in batch.inherits:
        decl = qn_index.lookup(edge.parent_qn)
        if decl is None or decl.kind != "class":
            continue
        parent_relation.setdefault(edge.child_qn, []).append(edge.parent_qn)
    return parent_relation


def build_local_var_type_index(batch: IndexBatch) -> LocalVarTypeIndex:
    """Sprint 14g — convert `batch.local_var_bindings` into a populated
    `LocalVarTypeIndex`.

    The binding's `enclosing_line_start`/`enclosing_line_end` reconstructs
    the enclosing scope's `ScopeId` (line numbers as monotonic ints —
    same encoding `to_scopes` uses). The TypeRef's `declared_at_scope`
    is THIS scope (where the assignment was written) so the resolver
    later anchors the type-name resolution against the right file's
    imports.
    """
    index = LocalVarTypeIndex()
    for b in batch.local_var_bindings:
        scope_id = ScopeId(
            file_path=b.file_path,
            range=_range_for(b.file_path, b.enclosing_line_start, b.enclosing_line_end),
            kind=b.enclosing_scope_kind,
        )
        # `class_field` only when the binding was hoisted to a class
        # scope (from `self.x = X()` in __init__). Function-body and
        # module-level bindings are both "constructor-inferred" — the
        # source describes WHY we know the type, not WHERE the binding
        # lives in the scope tree.
        source = "class_field" if b.enclosing_scope_kind == "class" else "constructor"
        type_ref = TypeRef(
            raw_name=b.type_raw_name,
            declared_at_scope=scope_id,
            source=source,
        )
        index.add(scope_id, b.var_name, type_ref)
    return index


__all__ = [
    "build_local_var_type_index",
    "to_declarations",
    "to_parent_relation",
    "to_scopes",
]
