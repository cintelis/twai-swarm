"""Cross-file resolution — Sprint 12b's port of GitNexus's `finalize-algorithm.ts`.

Public API: `finalize_batch(batch)`. Same signature + contract as the
legacy `resolver.resolve_batch` — mutates the batch in place,
rewriting CallEdges and InheritsEdges that resolve to in-repo
Functions/Classes and emitting SymbolNodes for everything that doesn't.

The pipeline:

  1. Adapter — convert FunctionNode / ClassNode / ModuleNode records to
     `Declaration` + `ScopeId` (12a types).
  2. Indexes — build qualified_name_index, module_scope_index,
     scope_tree, position_index from the conversions above.
  3. Import graph — `rustworkx.PyDiGraph` over module qualified-names,
     with directed edges importing-module → imported-module.
  4. SCCs — `rustworkx.strongly_connected_components` over the graph.
     Non-trivial SCCs (size >= 2) capture circular re-export chains.
  5. Re-export closure — for each SCC, every member logically exposes
     the union of its members' module-level exports.
  6. Wildcard expansion — `from x import *` materialises a logical
     binding from the importing file to every name `x` exports
     (closure-aware: if `x` is in a re-export SCC, every closure
     member's exports count). NOTE: today's Python extractor emits
     nothing for `from x import *`. The wildcard branch fires only on
     synthetic ImportEdges that follow the shape (kind="module" with
     local_name in ("", "*")). When extractor support lands, this code
     starts handling real wildcard imports without further changes.
  7. Resolution — for every CallEdge / InheritsEdge:
       a. direct qn lookup
       b. import-based bare-name lookup (`from x import y` + `y()`)
       c. module-prefix lookup (`import x.y` + `x.y.foo()`)
       d. param-type method lookup (`def f(x: T): x.method()`)
       e. wildcard expansion (`from x import *` + `bar()`)
       f. closure-aware variant of (b)–(e) — if the resolution lands
          on a module qn inside a re-export SCC, search every closure
          member's exports for the symbol.
  8. SymbolNode emission — anything that didn't resolve produces one
     SymbolNode per unique unresolved qn.

Parity guarantee: steps (a)–(d) replicate the legacy resolver, so the
new path's resolved-edge count is >= legacy's on the same input. Steps
(e)–(f) only add resolutions; they never remove them.
"""
from __future__ import annotations

from typing import Any

from ..actions import (
    CallEdge,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    SymbolNode,
)
from . import _adapter
from .module_scope_index import build_module_scope_index
from .position_index import build_position_index
from .qualified_name_index import build_qualified_name_index
from .scope_tree import build_scope_tree


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finalize_batch(batch: IndexBatch) -> None:
    """Mutate `batch` in place: rewrite calls/inheritance to point at
    in-repo Function/Class nodes when possible; emit SymbolNodes for
    targets that genuinely don't resolve.

    Same signature as the legacy `resolver.resolve_batch` so the
    `phases.resolve` switch is one-line.
    """
    if not batch.functions and not batch.classes and not batch.calls and not batch.inherits:
        # Nothing to do — empty batch (or pure ScanPhase output).
        return

    decls = _adapter.to_declarations(batch)
    scopes = _adapter.to_scopes(batch)

    # build_scope_tree raises on overlap-without-containment; let that
    # propagate (mirrors 12a contract).
    scope_tree = build_scope_tree(scopes)
    _position_index = build_position_index(scope_tree)  # noqa: F841 — held for 12c

    # qualified_name_index would raise on QN collision. The IndexBatch can
    # legitimately contain the same module qn twice (one ModuleNode per
    # file, but also entries from re-scans). Defensive: dedupe by qn here.
    seen_qns: set[str] = set()
    deduped_decls = []
    for d in decls:
        if d.qualified_name in seen_qns:
            continue
        seen_qns.add(d.qualified_name)
        deduped_decls.append(d)
    qn_index = build_qualified_name_index(deduped_decls)
    module_index = build_module_scope_index(deduped_decls)

    # Build import graph + closure map.
    import_closure = _build_reexport_closure(batch, module_index)

    # Build per-file imports (legacy shape: file_path -> {local_name: (target_qn, kind)})
    imports_by_file: dict[str, dict[str, tuple[str, str]]] = {}
    wildcard_targets_by_file: dict[str, list[str]] = {}
    for imp in batch.imports:
        if _is_wildcard(imp):
            wildcard_targets_by_file.setdefault(imp.file_path, []).append(imp.target_qn)
            continue
        if not imp.local_name:
            # Older / partial extractor output — no usable binding.
            continue
        imports_by_file.setdefault(imp.file_path, {})[imp.local_name] = (imp.target_qn, imp.kind)

    # Caller-side metadata (same shape the legacy resolver uses).
    func_param_types: dict[str, dict[str, str]] = {
        f.qualified_name: dict(f.param_types) for f in batch.functions if f.param_types
    }
    func_file_path: dict[str, str] = {f.qualified_name: f.file_path for f in batch.functions}
    class_file_path: dict[str, str] = {c.qualified_name: c.file_path for c in batch.classes}

    # Lookup closures over the indexes.
    def _is_function(qn: str) -> bool:
        d = qn_index.lookup(qn)
        return d is not None and d.kind in ("function", "method")

    def _is_class(qn: str) -> bool:
        d = qn_index.lookup(qn)
        return d is not None and d.kind == "class"

    def _resolve_via_closure(module_qn: str, name: str, predicate) -> str | None:
        # If `module_qn` is in a re-export SCC, search every closure member's
        # exports for `name`. Predicate decides which kind to accept
        # (function vs class).
        members = import_closure.get(module_qn)
        if members is None:
            members = {module_qn}
        for member in members:
            for exp in module_index.exports_of(member):
                if exp.name == name and predicate(exp.qualified_name):
                    return exp.qualified_name
        return None

    def _resolve_type_name(file_path: str, type_text: str) -> str | None:
        """Map a type annotation string to a qualified Class name.

        Same heuristic as the legacy resolver — try every candidate
        identifier in order, first one that resolves wins. Covers
        Optional[T] / Union[T, U] / list[T] / T | None / dict[K, V]
        without language-specific special cases.
        """
        import re
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", type_text)
        if not candidates:
            return None
        file_imports = imports_by_file.get(file_path, {})
        for head in candidates:
            if _is_class(head):
                return head
            if head in file_imports:
                target_qn, _kind = file_imports[head]
                if _is_class(target_qn):
                    return target_qn
                # Closure-aware: if `target_qn` is itself a module reachable
                # in an SCC, look for the class as one of the closure exports.
                hit = _resolve_via_closure(target_qn, head, _is_class)
                if hit is not None:
                    return hit
        # Wildcard expansion fallback.
        for wc_target in wildcard_targets_by_file.get(file_path, ()):
            for cand in candidates:
                hit = _resolve_via_closure(wc_target, cand, _is_class)
                if hit is not None:
                    return hit
        return None

    def _resolve_callee(caller_qn: str, callee_dotted: str) -> str | None:
        # (a) direct
        if _is_function(callee_dotted):
            return callee_dotted

        file_path = func_file_path.get(caller_qn)
        if file_path is None:
            return None
        file_imports = imports_by_file.get(file_path, {})
        wildcard_targets = wildcard_targets_by_file.get(file_path, ())

        if "." not in callee_dotted:
            # (b) bare name via `from x import y`
            if callee_dotted in file_imports:
                target_qn, kind = file_imports[callee_dotted]
                if kind == "symbol" and _is_function(target_qn):
                    return target_qn
                # (f) closure-aware: target_qn might be inside an SCC.
                if kind == "symbol":
                    parent = target_qn.rsplit(".", 1)[0] if "." in target_qn else ""
                    hit = _resolve_via_closure(parent, callee_dotted, _is_function)
                    if hit is not None:
                        return hit
            # (e) wildcard expansion — does any wildcard-imported module
            # export a function called `callee_dotted`?
            for wc_target in wildcard_targets:
                hit = _resolve_via_closure(wc_target, callee_dotted, _is_function)
                if hit is not None:
                    return hit
            return None

        head, rest = callee_dotted.split(".", 1)

        # (d) param.method — resolve param's type via imports, then look up Type.method.
        param_types = func_param_types.get(caller_qn, {})
        if head in param_types:
            type_qn = _resolve_type_name(file_path, param_types[head])
            if type_qn is not None:
                candidate = f"{type_qn}.{rest}"
                if _is_function(candidate):
                    return candidate

        # (c) imported_module.func — resolve module via imports, then look up func.
        if head in file_imports:
            target_qn, kind = file_imports[head]
            if kind == "module":
                candidate = f"{target_qn}.{rest}"
                if _is_function(candidate):
                    return candidate
                # (f) closure-aware: target_qn might re-export `rest` from an SCC peer.
                hit = _resolve_via_closure(target_qn, rest, _is_function)
                if hit is not None:
                    return hit

        # (e) wildcard expansion for `head.rest` — head might be a class
        # imported via `from x import *` and we then call Class.method.
        for wc_target in wildcard_targets:
            class_hit = _resolve_via_closure(wc_target, head, _is_class)
            if class_hit is not None:
                candidate = f"{class_hit}.{rest}"
                if _is_function(candidate):
                    return candidate

        return None

    def _resolve_parent(child_qn: str, parent_dotted: str) -> str | None:
        if _is_class(parent_dotted):
            return parent_dotted

        child_file = class_file_path.get(child_qn)
        if child_file is None:
            return None
        file_imports = imports_by_file.get(child_file, {})
        wildcard_targets = wildcard_targets_by_file.get(child_file, ())

        if "." not in parent_dotted:
            if parent_dotted in file_imports:
                target_qn, kind = file_imports[parent_dotted]
                if kind == "symbol" and _is_class(target_qn):
                    return target_qn
                if kind == "symbol":
                    parent_mod = target_qn.rsplit(".", 1)[0] if "." in target_qn else ""
                    hit = _resolve_via_closure(parent_mod, parent_dotted, _is_class)
                    if hit is not None:
                        return hit
            for wc_target in wildcard_targets:
                hit = _resolve_via_closure(wc_target, parent_dotted, _is_class)
                if hit is not None:
                    return hit
            return None

        head, rest = parent_dotted.split(".", 1)
        if head in file_imports:
            target_qn, kind = file_imports[head]
            if kind == "module":
                candidate = f"{target_qn}.{rest}"
                if _is_class(candidate):
                    return candidate
                hit = _resolve_via_closure(target_qn, rest, _is_class)
                if hit is not None:
                    return hit
        return None

    needs_symbol: set[str] = set()

    # Rewrite calls.
    new_calls = []
    for edge in batch.calls:
        resolved = _resolve_callee(edge.caller_qn, edge.callee_qn)
        if resolved is not None:
            new_calls.append(CallEdge(
                repo=edge.repo,
                caller_qn=edge.caller_qn,
                callee_qn=resolved,
                line=edge.line,
            ))
        else:
            needs_symbol.add(edge.callee_qn)
            new_calls.append(edge)
    batch.calls.clear()
    batch.calls.extend(new_calls)

    # Rewrite inheritance.
    new_inherits = []
    for edge in batch.inherits:
        resolved = _resolve_parent(edge.child_qn, edge.parent_qn)
        if resolved is not None:
            new_inherits.append(InheritsEdge(
                repo=edge.repo,
                child_qn=edge.child_qn,
                parent_qn=resolved,
            ))
        else:
            needs_symbol.add(edge.parent_qn)
            new_inherits.append(edge)
    batch.inherits.clear()
    batch.inherits.extend(new_inherits)

    # Materialise Symbol nodes (deduped against any pre-existing Symbols).
    existing = {s.qualified_name for s in batch.symbols}
    for qn in needs_symbol:
        if qn in existing:
            continue
        batch.symbols.append(SymbolNode(
            repo=batch.repo.name,
            qualified_name=qn,
            name=qn.rsplit(".", 1)[-1],
        ))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _is_wildcard(imp: ImportEdge) -> bool:
    """True if `imp` is a `from x import *` shape.

    Today's Python extractor emits nothing for wildcard imports (it only
    handles `dotted_name` and `aliased_import` siblings of an
    `import_from_statement` — see `extractor_python._walk_imports`). So
    this branch only fires when callers construct a synthetic wildcard
    edge. When extractor support lands, the convention will be either
    `local_name="*"` or `local_name=""` with `kind="module"`.
    """
    return imp.local_name in ("", "*") and imp.kind == "module" and imp.target_qn != imp.local_name


def _build_reexport_closure(
    batch: IndexBatch,
    module_index: Any,
) -> dict[str, set[str]]:
    """Build {module_qn: set[closure_member_qns]} for re-export SCCs.

    Uses rustworkx for Tarjan SCC + topological iteration over the
    condensed DAG. The condensation isn't directly exposed by rustworkx
    (per the Sprint 12 spike); we build it manually from the SCC list.

    Trivial SCCs (singletons) map to `{self}`. Non-trivial SCCs (>= 2
    members) map every member to the same set — every closure member.
    Callers use this to turn `target_qn -> exports` into
    `closure(target_qn) -> exports`.
    """
    import rustworkx as rx

    g = rx.PyDiGraph()
    qn_to_node: dict[str, int] = {}

    def _node_for(qn: str) -> int:
        idx = qn_to_node.get(qn)
        if idx is None:
            idx = g.add_node(qn)
            qn_to_node[qn] = idx
        return idx

    # Seed nodes for every module we know about (so isolated modules get
    # their own singleton closure).
    for m in batch.modules:
        _node_for(m.qualified_name)

    # Edges: for every ImportEdge, edge from the importing file's module-qn
    # to the target's module-qn. For "symbol" kind imports the edge points
    # at the *module* not the symbol — that's where the symbol lives.
    file_to_module: dict[str, str] = {m.file_path: m.qualified_name for m in batch.modules}
    for imp in batch.imports:
        src_module = file_to_module.get(imp.file_path)
        if src_module is None:
            continue
        if _is_wildcard(imp):
            target_module = imp.target_qn
        elif imp.kind == "symbol":
            target_module = imp.target_qn.rsplit(".", 1)[0] if "." in imp.target_qn else imp.target_qn
        else:
            target_module = imp.target_qn
        if not target_module:
            continue
        s = _node_for(src_module)
        t = _node_for(target_module)
        # PyDiGraph allows parallel edges; that's fine for SCC.
        g.add_edge(s, t, None)

    sccs = rx.strongly_connected_components(g)
    closure: dict[str, set[str]] = {}
    for scc in sccs:
        members = {g[node_idx] for node_idx in scc}
        if len(members) <= 1:
            for qn in members:
                closure[qn] = {qn}
        else:
            for qn in members:
                closure[qn] = members
    return closure


__all__ = ["finalize_batch"]
