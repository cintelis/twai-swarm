"""`qualified_name -> Declaration` — every named site in the repo.

Mirrors GitNexus's `qualified-name-index.ts`. This is the "phone book"
the resolver hits to turn a fully-qualified reference into a
declaration site.

Two declarations with the same QN are an extractor bug or a real
duplicate-symbol error in the source — either way, finalize can't pick
between them. We raise `ScopeTreeInvariantError` (same family used by
`scope_tree`) at build time with both file paths so the diagnostic
points at the conflict.
"""
from __future__ import annotations

from typing import Iterable

from .scope_tree import ScopeTreeInvariantError
from .types import Declaration


class QualifiedNameIndex:
    """`qualified_name -> Declaration`. Built once, queried many times."""

    def __init__(self) -> None:
        self._by_qn: dict[str, Declaration] = {}

    # ---- read API ----------------------------------------------------------

    def lookup(self, qn: str) -> Declaration | None:
        """Declaration for `qn`, or None if no such name is known."""
        return self._by_qn.get(qn)

    def __contains__(self, qn: str) -> bool:
        return qn in self._by_qn

    def __len__(self) -> int:
        return len(self._by_qn)


def build_qualified_name_index(
    declarations: Iterable[Declaration],
) -> QualifiedNameIndex:
    """Build a QN -> Declaration map; raise on collision.

    A "collision" is two declarations sharing the same `qualified_name`.
    Same-file duplicates (which would be a syntax error in real code)
    and cross-file duplicates (e.g. two modules both defining `app.foo.Bar`)
    are both caught.
    """
    idx = QualifiedNameIndex()

    for decl in declarations:
        existing = idx._by_qn.get(decl.qualified_name)
        if existing is not None:
            raise ScopeTreeInvariantError(
                f"qualified-name collision for {decl.qualified_name!r}: "
                f"declared in {existing.file_path!r} and {decl.file_path!r}"
            )
        idx._by_qn[decl.qualified_name] = decl

    return idx


__all__ = [
    "QualifiedNameIndex",
    "build_qualified_name_index",
]
