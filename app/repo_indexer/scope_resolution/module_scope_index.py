"""What does each module export?

Mirrors GitNexus's `module-scope-index.ts`. The cross-file resolver
(Sprint 12b) hits this index to decide whether a `from foo import bar`
binds to a real declaration or should fall through to a Symbol node.

Python-flavoured default: anything whose bare name starts with `_` is
considered private and excluded from exports. Callers can override via
`private_predicate` — TypeScript needs different rules (everything
top-level is exported unless the source uses `export` keywords, which
is captured at extraction time, so for TS the predicate may always
return False).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .types import Declaration


def _default_private_predicate(decl: Declaration) -> bool:
    """Python convention: leading underscore = private.

    Module-level `_foo` and class-level `_bar` both excluded from
    exports. Dunder names (`__init__`, `__all__`) are also private by
    this rule, which matches Python's `from x import *` behavior.
    """
    return decl.name.startswith("_")


class ModuleScopeIndex:
    """`module_qualified_name -> [exported declarations]`.

    Only module-level declarations are eligible for export. A method
    on a class is part of the *class*'s scope, not the module's, even
    though its qualified name is dotted under the module — the exports
    list contains the class itself, and the resolver walks down from
    there for `from foo import Bar; Bar.baz()`.

    This index is lossy by design: we don't keep private declarations.
    Callers who need every declaration in a module should hit
    `qualified_name_index` instead.
    """

    def __init__(
        self,
        *,
        private_predicate: Callable[[Declaration], bool] | None = None,
    ) -> None:
        self._exports: dict[str, list[Declaration]] = defaultdict(list)
        self._private = private_predicate or _default_private_predicate

    # ---- read API ----------------------------------------------------------

    def exports_of(self, module_qn: str) -> list[Declaration]:
        """Public declarations for `module_qn`. Empty list if unknown.

        Returns a fresh list each call (callers may safely mutate the
        returned object without affecting the index).
        """
        return list(self._exports.get(module_qn, ()))

    def module_qns(self) -> set[str]:
        """All module qualified names this index has heard of."""
        return set(self._exports.keys())

    def __contains__(self, module_qn: str) -> bool:
        return module_qn in self._exports


def build_module_scope_index(
    declarations: Iterable[Declaration],
    *,
    private_predicate: Callable[[Declaration], bool] | None = None,
) -> ModuleScopeIndex:
    """Group public, module-level declarations by their owning module.

    A declaration is module-level when its `scope_id` is None (per the
    `Declaration` contract — "None for module-level declarations").
    Anything nested (a method, a closure) is skipped here; methods are
    discovered via the class's own membership at finalize time.

    The owning module's qualified name is derived from the declaration's
    own QN by dropping the last dotted component. For a module
    declaration itself (`kind="module"`, QN `app.foo`), the owning
    module is `app.foo` — the module-as-its-own-export edge case.
    """
    idx = ModuleScopeIndex(private_predicate=private_predicate)

    for decl in declarations:
        # Skip nested declarations.
        if decl.scope_id is not None:
            continue
        if idx._private(decl):
            continue

        if decl.kind == "module":
            module_qn = decl.qualified_name
        else:
            # "app.foo.Bar" -> "app.foo". Top-level names with no dot
            # become "" — we still index them under the empty-string
            # module (a degenerate case, but consistent).
            if "." in decl.qualified_name:
                module_qn = decl.qualified_name.rsplit(".", 1)[0]
            else:
                module_qn = ""

        idx._exports[module_qn].append(decl)

    return idx


__all__ = [
    "ModuleScopeIndex",
    "build_module_scope_index",
]
