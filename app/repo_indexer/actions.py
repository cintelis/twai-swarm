"""Typed records the extractor emits and the loader writes.

Keeping them as plain dataclasses (not Cypher fragments) means the
extractor stays storage-agnostic and unit-testable — a JSON dump of an
IndexBatch is enough to verify the AST traversal without spinning up
Neo4j.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Language = Literal["python", "typescript", "javascript"]


@dataclass(frozen=True)
class RepoNode:
    name: str           # e.g. "twai-swarm"
    url: str            # canonical https URL or "" for local
    commit_sha: str     # 40-char hex; "" for ad-hoc local scans
    tenant_id: str = "default"


@dataclass(frozen=True)
class FileNode:
    repo: str           # repo name (foreign key)
    path: str           # relative posix path from repo root
    language: Language
    sha: str            # blob sha256, used for diff-skip on re-scan


@dataclass(frozen=True)
class ModuleNode:
    repo: str
    qualified_name: str  # e.g. "app.repo_indexer.actions"
    file_path: str       # canonical defining file (1:1 in Python; 1:N in TS)


@dataclass(frozen=True)
class ClassNode:
    repo: str
    qualified_name: str  # "<module>.<ClassName>"
    name: str            # bare class name
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""


@dataclass(frozen=True)
class FunctionNode:
    repo: str
    qualified_name: str  # "<module>.<func>" or "<module>.<Class>.<method>"
    name: str
    file_path: str
    line_start: int
    line_end: int
    is_async: bool = False
    is_method: bool = False
    parent_class_qn: str = ""    # populated iff is_method
    params: tuple[str, ...] = field(default_factory=tuple)
    docstring: str = ""


@dataclass(frozen=True)
class SymbolNode:
    """A name we saw a reference to but don't own a definition for.

    Could be a stdlib function, a third-party library symbol, a dynamic
    attribute access — anything we can't resolve to a Function/Class node
    in this repo. Tracked separately so unresolved edges don't pollute
    the resolved graph.
    """
    repo: str
    qualified_name: str  # best-effort dotted name we observed
    name: str            # bare name


@dataclass(frozen=True)
class InheritsEdge:
    repo: str
    child_qn: str
    parent_qn: str       # may be a SymbolNode if external


@dataclass(frozen=True)
class CallEdge:
    repo: str
    caller_qn: str       # always a FunctionNode in this repo
    callee_qn: str       # FunctionNode (resolved) or SymbolNode (external)
    line: int


@dataclass(frozen=True)
class ImportEdge:
    repo: str
    file_path: str       # importing file
    target_qn: str       # imported module's qualified_name (or symbol)


@dataclass
class IndexBatch:
    """Mutable accumulator the extractor populates per file.

    The loader takes one IndexBatch at a time and runs ~1 round-trip per
    node-type (UNWIND $rows MERGE …) — keeps Neo4j writes batchy without
    needing the extractor to know about Cypher.
    """
    repo: RepoNode
    files: list[FileNode] = field(default_factory=list)
    modules: list[ModuleNode] = field(default_factory=list)
    classes: list[ClassNode] = field(default_factory=list)
    functions: list[FunctionNode] = field(default_factory=list)
    symbols: list[SymbolNode] = field(default_factory=list)
    inherits: list[InheritsEdge] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)

    def extend(self, other: IndexBatch) -> None:
        """Merge `other` into self. Repos must match."""
        if other.repo != self.repo:
            raise ValueError(f"can't merge batches from different repos: {self.repo.name} vs {other.repo.name}")
        self.files.extend(other.files)
        self.modules.extend(other.modules)
        self.classes.extend(other.classes)
        self.functions.extend(other.functions)
        self.symbols.extend(other.symbols)
        self.inherits.extend(other.inherits)
        self.calls.extend(other.calls)
        self.imports.extend(other.imports)

    def counts(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            "modules": len(self.modules),
            "classes": len(self.classes),
            "functions": len(self.functions),
            "symbols": len(self.symbols),
            "inherits_edges": len(self.inherits),
            "call_edges": len(self.calls),
            "import_edges": len(self.imports),
        }
