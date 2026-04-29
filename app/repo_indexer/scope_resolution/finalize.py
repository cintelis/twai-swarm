"""Cross-file resolution — Sprint 12b's port of GitNexus's `finalize-algorithm.ts`,
extended in Sprint 12c with a method-dispatch index for `self.method()` /
`super().method()` / inherited param-type methods.

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
  7. Inheritance resolution — rewrite each `InheritsEdge.parent_qn`
     (textual) into a resolved class qn via the same import chain used
     for callees. Inherits land FIRST so the parent_relation feeding
     the dispatch index reflects rewritten qns.
  8. Method dispatch index (Sprint 12c) — `MethodDispatchIndex` keyed
     on resolved class qns. For each class, an O(ancestors) walk
     produces the merged `{method_name -> Declaration}` set.
  9. Call resolution — for every CallEdge:
       a. direct qn lookup
       b. import-based bare-name lookup (`from x import y` + `y()`)
       c. module-prefix lookup (`import x.y` + `x.y.foo()`)
       d. param-type method lookup (`def f(x: T): x.method()`) —
          tries `T.method` first, then walks T's ancestors via the
          dispatch index (Sprint 12c).
       e. wildcard expansion (`from x import *` + `bar()`)
       f. closure-aware variant of (b)–(e) — if the resolution lands
          on a module qn inside a re-export SCC, search every closure
          member's exports for the symbol.
       g. (Sprint 12c) `self.method()` — caller is a method of class
          C; dispatch index resolves `method` walking C + ancestors.
       h. (Sprint 12c) `super().method()` — caller is a method of C;
          dispatch index walks `C`'s first parent + that parent's
          ancestors. NOTE: today's extractor's `_flatten_attribute`
          returns None on `super().X` chains (the inner `super()` is
          a `call` node, not `identifier`/`attribute`), so no
          CallEdge with `super().X` shape currently arrives here.
          The branch fires defensively for synthetic input + future
          extractor changes that emit `super().method` strings.
 10. SymbolNode emission — anything that didn't resolve produces one
     SymbolNode per unique unresolved qn.

Parity guarantee: steps (a)–(d), (f) replicate the 12b resolver — every
edge that resolved before 12c still resolves after. Steps (g) and (h)
only add resolutions; they never remove them. Step (d)'s ancestor
fallback only activates after the local lookup misses, so cases where
the legacy path resolved keep their answer.
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
from .method_dispatch_index import build_method_dispatch_index
from .module_scope_index import build_module_scope_index
from .position_index import build_position_index
from .qualified_name_index import build_qualified_name_index
from .scope_tree import build_scope_tree
from .types import Range, ScopeId


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

    # Sprint 14g — typeBinding index for variable-method-call resolution.
    # Empty when batch.local_var_bindings is empty (pre-14g batches stay
    # byte-identical in resolution behaviour).
    local_var_index = _adapter.build_local_var_type_index(batch)

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
    # Sprint 14g — file -> module qn. Lets `_resolve_type_name` fall back
    # to same-module class lookup when a bare type name isn't imported
    # (e.g., `class A: pass; b = A()` in the same file). Pre-14g this
    # didn't matter because param-type annotations almost always come
    # from an imported type; with constructor inference, intra-module
    # types are common.
    module_qn_by_file: dict[str, str] = {m.file_path: m.qualified_name for m in batch.modules}
    # Sprint 14g — caller_qn -> ScopeId of the caller's enclosing function.
    # Used by `_resolve_var_binding` to look up local-var typeBindings via
    # the LocalVarTypeIndex's scope-chain walk.
    func_scope_id: dict[str, ScopeId] = {
        f.qualified_name: ScopeId(
            file_path=f.file_path,
            range=Range(
                file_path=f.file_path,
                start_byte=f.line_start,
                end_byte=f.line_end + 1,
            ),
            kind="function",
        )
        for f in batch.functions
    }
    # Sprint 12c — caller_qn -> enclosing class qn (only set for methods).
    # Used by `self.method()` / `super().method()` resolution.
    caller_class_qn: dict[str, str] = {
        f.qualified_name: f.parent_class_qn
        for f in batch.functions
        if f.is_method and f.parent_class_qn
    }

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
        same_module_qn = module_qn_by_file.get(file_path)
        for head in candidates:
            if _is_class(head):
                return head
            # Sprint 14g — same-module class lookup. `class A: pass; b = A()`
            # in the same file: the bare type name `"A"` resolves via
            # `<this-module>.A` without any import edge being involved.
            if same_module_qn is not None:
                candidate = f"{same_module_qn}.{head}"
                if _is_class(candidate):
                    return candidate
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
                # `rest` may itself be dotted (e.g. `param.attr.method()`);
                # the dispatch index only resolves direct method names,
                # so split off the leaf and check whether the leading
                # segment is a method on `type_qn`. Most calls in the
                # wild are single-segment.
                candidate = f"{type_qn}.{rest}"
                if _is_function(candidate):
                    return candidate
                # Sprint 12c — `Type.method` not local; walk Type's
                # ancestors via the dispatch index. Only fires for
                # single-segment `rest` (`param.foo()`); chained
                # `param.foo.bar()` falls through unchanged.
                if "." not in rest:
                    inherited = dispatch_index.resolve(type_qn, rest)
                    if inherited is not None:
                        return inherited.qualified_name

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

    # Rewrite inheritance FIRST (Sprint 12c reordering). The dispatch
    # index needs resolved parent qns so it can chase ancestor methods;
    # leaving inherits as raw textual qns would feed external/unresolved
    # bases into `to_parent_relation` and break the walk.
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

    # Sprint 12c — method dispatch index. Built AFTER inheritance rewrite
    # so `to_parent_relation` sees resolved parent qns and only includes
    # edges whose parents the qn_index actually knows about (externals
    # are filtered there).
    parent_relation = _adapter.to_parent_relation(batch, qn_index)
    method_decls = [d for d in deduped_decls if d.kind == "method"]
    dispatch_index = build_method_dispatch_index(method_decls, parent_relation)

    def _resolve_type_chain(
        raw_name: str,
        declared_at_scope: ScopeId,
        depth: int = 0,
    ) -> str | None:
        """Sprint 14h — resolve a possibly-dotted type expression to a
        class qn, following typeBinding chains as needed.

        Handles three cases unified through the same recursion:

        1. Bare name resolves directly as a class (`StateGraph`).
        2. Bare name is a function whose return type is a typeBinding
           on the function's enclosing scope (`make_user` → `User`).
           Recurses with the bound type.
        3. Dotted name is a method-call chain (`builder.compile`):
           resolve head's class, then look up the method's return-type
           binding on the class scope. Recurses for each segment.

        Depth-capped at 8 to mirror GitNexus's `RECHAIN_MAX_DEPTH`. The
        cap exists for cyclic alias safety (`a = b()` and `b = a()`
        synthetic edge cases) — production fixtures stay well below.
        """
        if depth > 8:
            return None

        parts = raw_name.split(".")
        if len(parts) == 1:
            # Bare name. Try direct class resolution first.
            type_qn = _resolve_type_name(declared_at_scope.file_path, raw_name)
            if type_qn is not None:
                return type_qn
            # Try as a typeBinding chain — `raw_name` might be a function
            # whose return-type binding lives on declared_at_scope's chain.
            chained_ref = local_var_index.find(declared_at_scope, raw_name, scope_tree)
            if chained_ref is not None:
                return _resolve_type_chain(
                    chained_ref.raw_name, chained_ref.declared_at_scope, depth + 1,
                )
            return None

        # Dotted: head.middle…tail. Resolve head's class, walk middle,
        # treat tail as a method whose return type we want.
        head = parts[0]
        middle = parts[1:-1]
        tail = parts[-1]

        # Resolve head — try as a typeBinding first, then as a class name.
        head_class_qn: str | None = None
        head_ref = local_var_index.find(declared_at_scope, head, scope_tree)
        if head_ref is not None:
            head_class_qn = _resolve_type_chain(
                head_ref.raw_name, head_ref.declared_at_scope, depth + 1,
            )
        if head_class_qn is None:
            head_class_qn = _resolve_type_name(declared_at_scope.file_path, head)
        if head_class_qn is None:
            return None

        # Walk middle attributes as class-field bindings.
        current = head_class_qn
        for mid in middle:
            cls_scope = class_scope_id.get(current)
            if cls_scope is None:
                return None
            mid_ref = local_var_index.find(cls_scope, mid, scope_tree)
            if mid_ref is None:
                return None
            current = _resolve_type_chain(
                mid_ref.raw_name, mid_ref.declared_at_scope, depth + 1,
            )
            if current is None:
                return None

        # Tail is a method on `current`'s class — its typeBinding holds
        # the return type. Same lookup as the field walk; the binding
        # was emitted at extraction time by _emit_function for any
        # method that has a `-> X:` annotation.
        cls_scope = class_scope_id.get(current)
        if cls_scope is None:
            return None
        tail_ref = local_var_index.find(cls_scope, tail, scope_tree)
        if tail_ref is None:
            return None
        return _resolve_type_chain(
            tail_ref.raw_name, tail_ref.declared_at_scope, depth + 1,
        )

    def _resolve_var_binding(
        caller_qn: str,
        callee_dotted: str,
    ) -> str | None:
        """Sprint 14g — resolve `var.method()` where `var` was bound by an
        in-function assignment (`var = SomeClass(...)`).

        Walks the LocalVarTypeIndex from the caller's scope upward; if a
        TypeRef is found, resolves `raw_name` through the binding's
        declared-at-scope file imports, then dispatches `method` via
        MethodDispatchIndex (same machinery `_resolve_callee` case (d)
        uses for parameter-typed receivers).

        Only fires when the leaf is a single segment (`var.method`); chained
        attribute access (`var.attr.method`) is case 0 territory and goes
        through the compound-receiver resolver in 14g.2.
        """
        if "." not in callee_dotted:
            return None
        head, rest = callee_dotted.split(".", 1)
        if "." in rest:
            return None  # chained — defer to compound resolver

        caller_scope = func_scope_id.get(caller_qn)
        if caller_scope is None:
            return None
        type_ref = local_var_index.find(caller_scope, head, scope_tree)
        if type_ref is None:
            return None
        # Sprint 14h — chain-resolve handles dotted raw_names
        # (`g = builder.compile()` → "builder.compile" → CompiledStateGraph)
        # and bare-name function references (`x = make_user()` → "make_user"
        # → User via the return-type binding on the function's module
        # scope). Pre-14h this was just `_resolve_type_name`.
        type_qn = _resolve_type_chain(
            type_ref.raw_name, type_ref.declared_at_scope,
        )
        if type_qn is None:
            return None

        candidate = f"{type_qn}.{rest}"
        if _is_function(candidate):
            return candidate
        # Walk ancestors via dispatch index (same as case (d)).
        inherited = dispatch_index.resolve(type_qn, rest)
        if inherited is not None:
            return inherited.qualified_name
        return None

    # Map class qn -> ScopeId of the class body. Used by the compound-
    # receiver resolver to walk a class's typeBindings for field types.
    class_scope_id: dict[str, ScopeId] = {
        c.qualified_name: ScopeId(
            file_path=c.file_path,
            range=Range(
                file_path=c.file_path,
                start_byte=c.line_start,
                end_byte=c.line_end + 1,
            ),
            kind="class",
        )
        for c in batch.classes
    }

    def _resolve_compound_receiver(
        caller_qn: str,
        callee_dotted: str,
    ) -> str | None:
        """Sprint 14g.2 — Case 0 from GitNexus's dispatcher. Resolve
        `obj.attr.method()` and `self.attr.method()` chains by:
            1. Finding `obj`'s class via local-var typeBindings (or
               caller's `self` if obj=='self')
            2. Walking `.attr1.attr2…` segments by looking up each
               attribute as a field on the current class's scope
               (class fields are stored as typeBindings on the class
               scope; same LocalVarTypeIndex that holds local vars)
            3. Resolving the leaf method on the final class via dispatch
               index

        Bounded to depth 3 (two attribute hops + the method); deeper
        chains fall through to Symbol. Mirrors GitNexus's
        COMPOUND_RECEIVER_MAX_DEPTH=4 with one fewer hop because we
        don't yet handle return-type bindings from method calls in the
        chain (`f().g().h()` style).
        """
        parts = callee_dotted.split(".")
        if len(parts) < 3:
            return None  # not compound — handled by case (i)
        if len(parts) > 4:
            return None  # depth-cap; deferred

        head = parts[0]
        attrs = parts[1:-1]   # intermediate attribute hops
        method_name = parts[-1]

        # Step 1: find head's class.
        # Sprint 14j — accept `this` alongside `self` so the same
        # compound-receiver dispatcher serves both Python and TypeScript.
        if head in ("self", "this"):
            current_class_qn = caller_class_qn.get(caller_qn)
        else:
            caller_scope = func_scope_id.get(caller_qn)
            if caller_scope is None:
                return None
            type_ref = local_var_index.find(caller_scope, head, scope_tree)
            if type_ref is None:
                return None
            # Sprint 14h — chain-resolve for dotted raw_names + return
            # types. Same swap as in `_resolve_var_binding`.
            current_class_qn = _resolve_type_chain(
                type_ref.raw_name, type_ref.declared_at_scope,
            )

        if current_class_qn is None:
            return None

        # Step 2: walk attribute hops via class-scope field bindings.
        for attr in attrs:
            cls_scope = class_scope_id.get(current_class_qn)
            if cls_scope is None:
                return None
            field_type_ref = local_var_index.find(cls_scope, attr, scope_tree)
            if field_type_ref is None:
                return None
            # Sprint 14h — chain-resolve handles fields whose declared
            # types are themselves dotted (`self.x: pkg.User`).
            current_class_qn = _resolve_type_chain(
                field_type_ref.raw_name, field_type_ref.declared_at_scope,
            )
            if current_class_qn is None:
                return None

        # Step 3: resolve the method on the final class.
        candidate = f"{current_class_qn}.{method_name}"
        if _is_function(candidate):
            return candidate
        inherited = dispatch_index.resolve(current_class_qn, method_name)
        if inherited is not None:
            return inherited.qualified_name
        return None

    def _resolve_self_or_super(
        caller_qn: str,
        callee_dotted: str,
    ) -> str | None:
        """Sprint 12c — resolve `self.method` / `super().method` via the
        dispatch index. Only the FIRST dotted segment is examined; chained
        attribute calls like `self._tree.parent_of` are out of scope
        (would need local-variable type inference).

        Returns the resolved method qn, or None if either:
          - the caller isn't a method, or
          - the receiver isn't `self` / `super()`, or
          - the method name doesn't exist on the class or any ancestor, or
          - (super) the class has no resolvable parent.
        """
        class_qn = caller_class_qn.get(caller_qn)
        if class_qn is None:
            return None

        if "." not in callee_dotted:
            return None
        head, rest = callee_dotted.split(".", 1)
        # Only resolve the leaf method — `self.foo.bar` is a method on
        # `self.foo`'s type, not on `self`'s class.
        if "." in rest:
            return None

        # Sprint 14j — `this.method()` (TypeScript) goes through the
        # same dispatch as Python's `self.method()`.
        if head in ("self", "this"):
            decl = dispatch_index.resolve(class_qn, rest)
            return decl.qualified_name if decl is not None else None

        # `super()` case — the extractor's `_flatten_attribute` returns
        # None on `super().X` chains today (the inner `super()` is a
        # tree-sitter `call` node, breaking the identifier-or-attribute
        # recursion in extractor_python._flatten_attribute), so this
        # branch is unreachable from real Python source as of 12c. It
        # fires for synthetic `super().method` / `super.method` shapes
        # that callers (tests, future extractor changes) might pass.
        if head in ("super()", "super"):
            parent_qn = dispatch_index.parent_of(class_qn)
            if parent_qn is None:
                return None
            decl = dispatch_index.resolve(parent_qn, rest)
            return decl.qualified_name if decl is not None else None

        return None

    # Rewrite calls.
    new_calls = []
    for edge in batch.calls:
        resolved = _resolve_callee(edge.caller_qn, edge.callee_qn)
        if resolved is None:
            # Sprint 12c — self/super fallback after the import-based
            # chain has tried and failed. Keeps 12b parity (anything the
            # 12b chain resolves still resolves identically) while
            # filling in the previously-Symbol self.method/super().method
            # cases.
            resolved = _resolve_self_or_super(edge.caller_qn, edge.callee_qn)
        if resolved is None:
            # Sprint 14g — variable-binding fallback. Receiver is a bare
            # local name (not a parameter, not self/super, not an import)
            # whose type was inferred from a constructor assignment
            # (`x = SomeClass(...)`). Last resort before SymbolNode
            # emission so existing 12b/12c resolutions stay untouched.
            resolved = _resolve_var_binding(edge.caller_qn, edge.callee_qn)
        if resolved is None:
            # Sprint 14g.2 — compound receiver chains. `obj.attr.method()`
            # and `self.attr.method()` resolved via class-field
            # typeBindings stored on the class scope.
            resolved = _resolve_compound_receiver(edge.caller_qn, edge.callee_qn)
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
