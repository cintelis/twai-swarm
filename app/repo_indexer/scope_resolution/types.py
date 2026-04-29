"""Frozen value types shared across the scope_resolution package.

Pure stdlib. No tree-sitter, no Neo4j. The four indexes built on top of
these types are themselves stdlib-only — the package is import-safe in
test environments without the parser stack.

These mirror (in shape, not in code) GitNexus's `gitnexus-shared/src/scope-resolution/`
types. We use frozen dataclasses so they're hashable and safe to use as
dict keys / in sets — the indexes lean heavily on that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ScopeKind = Literal["module", "class", "function", "block"]
DeclarationKind = Literal["function", "class", "module", "method", "import"]


@dataclass(frozen=True)
class Position:
    """A point in source: (file, byte-offset).

    Byte offsets (not line/col) so the math is unambiguous and matches
    what tree-sitter hands us as `start_byte` / `end_byte`.
    """
    file_path: str    # repo-relative posix path
    byte_offset: int  # absolute byte offset within the file


@dataclass(frozen=True)
class Range:
    """A half-open byte range [start_byte, end_byte) within one file.

    `end_byte` is exclusive — same convention tree-sitter uses. Two
    ranges with the same `start_byte`/`end_byte` in the same file are
    considered identical (no separate identity).
    """
    file_path: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class ScopeId:
    """Identity of a lexical scope.

    The (file_path, range, kind) triple is the scope's identity. We
    deliberately don't carry a synthetic id here: the range is unique
    per scope (no two scopes in a sane AST share an exact byte range),
    so the dataclass's structural equality is sufficient.

    The one edge case: nested function definitions whose ranges happen
    to coincide (zero-byte body, etc.) — see `build_scope_tree`'s
    invariant check; that case raises `ScopeTreeInvariantError`.
    """
    file_path: str
    range: Range
    kind: ScopeKind


@dataclass(frozen=True)
class Declaration:
    """A name declared at a site.

    `scope_id` is None for module-level declarations (the file *is* the
    enclosing scope; there's no parent to point at). For nested
    declarations (a function inside a function, a method inside a
    class), `scope_id` points at the immediately enclosing scope.

    `qualified_name` is the dotted name we'd look up by — `app.foo.Bar.baz`.
    `name` is the bare name — `baz`. The module-scope index uses `name`
    plus the underscore-prefix convention to decide what's exported.
    """
    qualified_name: str
    name: str
    kind: DeclarationKind
    file_path: str
    range: Range
    scope_id: Optional[ScopeId] = None


@dataclass(frozen=True)
class TypeRef:
    """Sprint 14g — a receiver-type binding, anchored to its declaration scope.

    Mirrors GitNexus's `TypeRef` (gitnexus-shared/src/scope-resolution/types.ts).
    The key fields:

    * `raw_name` — the type name as it was written in source. NOT a
      qualified name. `"StateGraph"` for `x = StateGraph(...)`,
      `"models.User"` for `x: models.User`. Resolved to a real qn at
      lookup time via the importing file's import chain.

    * `declared_at_scope` — the scope where the binding was declared.
      Anchors `raw_name`'s resolution: `"User"` resolves through THIS
      scope's imports, not the call-site's. Matters when the call site
      is a different file.

    * `source` — provenance. Determines edge cases:
        - "param"               — parameter annotation `def f(x: User)`
        - "self"                — methods' implicit `self` typing
        - "constructor"         — `x = User()` (case 7's bread and butter)
        - "return"              — `x = func()` where func has a return
                                  annotation (deferred — Sprint 14h)
        - "class_field"         — `self.attr = User()` in __init__ (14g.2)

    `type_args` is reserved for V2 (generic type arguments). V1 ignores;
    twai-swarm doesn't currently need them.
    """
    raw_name: str
    declared_at_scope: ScopeId
    source: Literal["param", "self", "constructor", "return", "class_field"]


__all__ = [
    "Position",
    "Range",
    "ScopeId",
    "Declaration",
    "TypeRef",
    "ScopeKind",
    "DeclarationKind",
]
