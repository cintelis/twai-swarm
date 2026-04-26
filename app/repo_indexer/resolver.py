"""Cross-file call/inheritance resolution.

The extractor records calls + inheritance with the dotted name as observed
in source (e.g. `sandbox.run_bash`, `Parent`). After all files are parsed
the resolver runs once over the assembled IndexBatch and tries to rewrite
each edge so its `callee_qn` / `parent_qn` points at a real Function or
Class node defined elsewhere in the repo.

What we resolve (in order of precedence):

    1. Same-file definitions — the extractor already handles top-level
       calls within a file; this layer adds nothing.

    2. Bare names that match an `import` in the file:
         from app.foo import bar  ->  bar() resolves to app.foo.bar
         import json              ->  json.dumps() resolves to json.dumps
                                       (still external, becomes a Symbol —
                                       BUT with the right qn so duplicate
                                       calls collapse)

    3. `param.method(...)` where `param` has a type annotation in the
       enclosing function: look up `Type.method` after resolving `Type`
       through the file's imports.

What we DO NOT resolve in 10b (deferred to a future pass):

    - `self.attr` calls — would need full class analysis
    - Local variable assignments — would need flow analysis
    - String-literal forward references in annotations
    - Generic types (we strip generics naively when present)

When a target resolves: the CallEdge / InheritsEdge keeps its shape but
its qn is rewritten to the canonical Function/Class qn. When it doesn't:
we emit a SymbolNode for it so the downstream Cypher MERGE has something
to point at.
"""
from __future__ import annotations

from .actions import IndexBatch, SymbolNode


def resolve_batch(batch: IndexBatch) -> None:
    """Mutate `batch` in place: rewrite calls/inheritance to point at
    in-repo Function/Class nodes when possible; emit SymbolNodes for
    targets that genuinely don't resolve.

    Idempotent across the same input — running twice is a no-op past
    the first pass (rewritten edges already point at qns that match
    Functions/Classes, so the resolution path returns the same answer).
    """
    # ─── Lookup tables built once per batch ──────────────────────────────
    functions_by_qn: dict[str, str] = {f.qualified_name: f.qualified_name for f in batch.functions}
    classes_by_qn: dict[str, str] = {c.qualified_name: c.qualified_name for c in batch.classes}

    # File → caller's param-type map (only for callers whose container is a Function).
    # function_qn → {param_name: type_text}
    func_param_types: dict[str, dict[str, str]] = {
        f.qualified_name: dict(f.param_types) for f in batch.functions if f.param_types
    }
    func_file_path: dict[str, str] = {f.qualified_name: f.file_path for f in batch.functions}

    # File-level imports: file_path → {local_name: (target_qn, kind)}
    imports_by_file: dict[str, dict[str, tuple[str, str]]] = {}
    for imp in batch.imports:
        if not imp.local_name:
            # Older / partial extractor output — skip (resolver can't use it).
            continue
        imports_by_file.setdefault(imp.file_path, {})[imp.local_name] = (imp.target_qn, imp.kind)

    # ─── Symbol bookkeeping ──────────────────────────────────────────────
    # Collect every qn that needs a Symbol node at the end. Using a set
    # here prevents duplicate SymbolNodes for the same target.
    needs_symbol: set[str] = set()

    # ─── Helpers ─────────────────────────────────────────────────────────
    def _resolve_type_name(file_path: str, type_text: str) -> str | None:
        """Map an as-observed type annotation to a qualified Class name.

        Tries every candidate identifier in the annotation in order and
        returns the first one that resolves. This covers the common
        wrapper shapes — Optional[T], Union[T, U], list[T], T | None,
        dict[K, V] — without language-specific special cases.

            "Sandbox"               -> ["Sandbox"]
            "Optional[Sandbox]"     -> ["Optional", "Sandbox"]
            "list[Sandbox]"         -> ["list", "Sandbox"]
            "Sandbox | None"        -> ["Sandbox", "None"]
            "dict[str, Sandbox]"    -> ["dict", "str", "Sandbox"]

        First resolvable wins; for `Optional[Sandbox]` that's `Sandbox`
        because `Optional` isn't a Class node in our repo.
        """
        import re
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", type_text)
        if not candidates:
            return None
        file_imports = imports_by_file.get(file_path, {})
        for head in candidates:
            if head in classes_by_qn:
                return head
            if head in file_imports:
                target_qn, _ = file_imports[head]
                if target_qn in classes_by_qn:
                    return target_qn
        return None

    def _resolve_callee(caller_qn: str, callee_dotted: str) -> str | None:
        """Try to map `callee_dotted` to a Function qn. Returns None if
        it doesn't land on something we own in this repo."""
        # Already a Function we know about (extractor's same-file case).
        if callee_dotted in functions_by_qn:
            return callee_dotted

        file_path = func_file_path.get(caller_qn)
        if file_path is None:
            return None
        file_imports = imports_by_file.get(file_path, {})

        if "." not in callee_dotted:
            # Bare name — check imports for a `from x import callee` binding.
            if callee_dotted in file_imports:
                target_qn, kind = file_imports[callee_dotted]
                # `from app.foo import bar` → bar() === app.foo.bar()
                if kind == "symbol" and target_qn in functions_by_qn:
                    return target_qn
            return None

        head, rest = callee_dotted.split(".", 1)

        # `param.method` — resolve param's type annotation.
        param_types = func_param_types.get(caller_qn, {})
        if head in param_types:
            type_qn = _resolve_type_name(file_path, param_types[head])
            if type_qn is not None:
                # Method qn = "<type_qn>.<method_chain>"
                candidate = f"{type_qn}.{rest}"
                if candidate in functions_by_qn:
                    return candidate

        # `imported_module.func` — resolve module via imports, then look up the func.
        if head in file_imports:
            target_qn, kind = file_imports[head]
            if kind == "module":
                candidate = f"{target_qn}.{rest}"
                if candidate in functions_by_qn:
                    return candidate

        return None

    def _resolve_parent(child_qn: str, parent_dotted: str) -> str | None:
        """Map an InheritsEdge's parent_qn (as-observed dotted name) to a
        Class qn in this repo, or None if external."""
        if parent_dotted in classes_by_qn:
            return parent_dotted
        # The child class's defining file is where we look up imports.
        # (We stored child_qn as Module.ClassName; find the Class to get its file.)
        child_file = next(
            (c.file_path for c in batch.classes if c.qualified_name == child_qn),
            None,
        )
        if child_file is None:
            return None
        file_imports = imports_by_file.get(child_file, {})

        if "." not in parent_dotted:
            if parent_dotted in file_imports:
                target_qn, kind = file_imports[parent_dotted]
                if kind == "symbol" and target_qn in classes_by_qn:
                    return target_qn
            return None

        head, rest = parent_dotted.split(".", 1)
        if head in file_imports:
            target_qn, kind = file_imports[head]
            if kind == "module":
                candidate = f"{target_qn}.{rest}"
                if candidate in classes_by_qn:
                    return candidate
        return None

    # ─── Rewrite calls ───────────────────────────────────────────────────
    new_calls = []
    for edge in batch.calls:
        resolved = _resolve_callee(edge.caller_qn, edge.callee_qn)
        if resolved is not None:
            new_calls.append(type(edge)(
                repo=edge.repo,
                caller_qn=edge.caller_qn,
                callee_qn=resolved,
                line=edge.line,
            ))
        else:
            # External / unresolved — keep the original qn and emit a Symbol.
            needs_symbol.add(edge.callee_qn)
            new_calls.append(edge)
    batch.calls.clear()
    batch.calls.extend(new_calls)

    # ─── Rewrite inheritance ─────────────────────────────────────────────
    new_inherits = []
    for edge in batch.inherits:
        resolved = _resolve_parent(edge.child_qn, edge.parent_qn)
        if resolved is not None:
            new_inherits.append(type(edge)(
                repo=edge.repo,
                child_qn=edge.child_qn,
                parent_qn=resolved,
            ))
        else:
            needs_symbol.add(edge.parent_qn)
            new_inherits.append(edge)
    batch.inherits.clear()
    batch.inherits.extend(new_inherits)

    # ─── Materialise Symbol nodes ────────────────────────────────────────
    # Anything still in needs_symbol is genuinely external — emit one
    # SymbolNode per unique qn. Loader's MERGE collapses duplicates anyway,
    # but pre-deduping here keeps the batch shape predictable.
    existing = {s.qualified_name for s in batch.symbols}
    for qn in needs_symbol:
        if qn in existing:
            continue
        batch.symbols.append(SymbolNode(
            repo=batch.repo.name,
            qualified_name=qn,
            name=qn.rsplit(".", 1)[-1],
        ))
