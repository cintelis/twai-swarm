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
    # Per-param type annotation strings, e.g. (("sandbox", "Sandbox"), ...).
    # The resolver uses these to turn `param.method(...)` calls into edges
    # that point at the actual Function instead of an external Symbol.
    # Captured as observed (no normalisation); resolution maps the bare
    # type name through the file's imports.
    param_types: tuple[tuple[str, str], ...] = field(default_factory=tuple)
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
    # Local name the import binds to. For `import a.b` -> "a"; for
    # `import a.b as foo` -> "foo"; for `from x import y` -> "y";
    # for `from x import y as foo` -> "foo". Used by the resolver to
    # map bare-name references in this file back to a qualified target.
    local_name: str = ""
    # "module" if the import binds a module/package; "symbol" if it
    # binds a single name (function, class, constant) from inside a module.
    # `import x` / `import x.y` -> module. `from x import y` -> symbol.
    kind: str = "module"


# ─── Sprint 13a — community detection ───────────────────────────────────────

@dataclass(frozen=True)
class CommunityNode:
    """A graph community detected by Louvain over CALLS / IMPORTS / INHERITS_FROM.

    Derived data — re-computable from the base graph. `label` is heuristic
    (top-frequency token across member names) and unique within (repo, tenant_id).
    `cohesion` is the intra-community edge ratio (1.0 = fully internal,
    0.0 = singleton or fully disconnected).
    """
    repo: str
    tenant_id: str            # MANDATORY — see Cross-cutting invariants
    label: str                # heuristic, deterministic, unique within (repo, tenant_id)
    cohesion: float           # 0.0..1.0 — intra-community edge ratio
    size: int                 # member count


@dataclass(frozen=True)
class MemberOfEdge:
    """Edge from a Function or Class to its Community."""
    repo: str
    tenant_id: str
    member_qn: str            # qualified_name of the Function or Class
    community_label: str      # FK to CommunityNode.label


# ─── Sprint 13b — process (execution flow) extraction ───────────────────────

@dataclass(frozen=True)
class ProcessNode:
    """A chain of CALLS edges that crosses community boundaries.

    Derived data — recomputable from the resolved graph + community
    assignments. `name` is `<first.short_name> -> <last.short_name>` plus
    a `#N` suffix on collision; unique within (repo, tenant_id). `summary`
    is a comma-separated list of the first few step short-names, truncated.
    """
    repo: str
    tenant_id: str            # MANDATORY — see Cross-cutting invariants
    name: str                 # e.g. "RepoTaskWorkflow.run -> resolve_batch"
    summary: str              # comma-separated short names, ~200 chars max


@dataclass(frozen=True)
class StepInProcessEdge:
    """Edge from a Process to a Function, with its position in the chain."""
    repo: str
    tenant_id: str
    process_name: str         # FK to ProcessNode.name
    member_qn: str            # qualified_name of the Function in this step
    step: int                 # 0-indexed position in the chain


# ─── Sprint 14a — embeddings bridge ─────────────────────────────────────────

@dataclass(frozen=True)
class EmbeddingUpdate:
    """Pending embedding write for the loader. Pairs a node's qualified
    name with its vector. The loader matches the node by (repo, qn) and
    SETs the `embedding` property.

    `target_kind` lets the loader pick the right label (Function vs Class)
    when SET-ing — same shape as the resolver's Function/Symbol fan-out
    pattern in `loader.write_batch` for INHERITS_FROM and CALLS edges.
    """
    repo: str
    tenant_id: str            # MANDATORY — see Cross-cutting invariants
    target_kind: Literal["function", "class"]
    qualified_name: str
    embedding: tuple[float, ...]   # fixed-length, dim sourced from app.embeddings


@dataclass(frozen=True)
class LocalVarBinding:
    """Sprint 14g — receiver-type binding for a local variable.

    Emitted by the extractor when it sees `x = SomeClass(...)` inside a
    function body (case 7: simple typeBinding, "constructor-inferred")
    or `self.x = SomeClass(...)` in `__init__` (case 0: class-field
    binding). The resolver builds a `LocalVarTypeIndex` from these and
    consults it when a method call's receiver is a bare local-name with
    no parameter annotation.

    `enclosing_scope_kind` is "function" or "class" — methods of a class
    that bind `self.x` produce class-scoped bindings (kind="class") so
    every method on the class sees them via the scope-chain walk.
    `type_raw_name` is the constructor name as written ("StateGraph"),
    NOT a resolved qn — finalize resolves it through the file's import
    chain at lookup time.
    """
    repo: str
    tenant_id: str
    file_path: str
    enclosing_scope_kind: Literal["function", "class"]
    enclosing_line_start: int       # 1-based inclusive line of the enclosing scope
    enclosing_line_end: int         # 1-based inclusive line of the enclosing scope
    var_name: str                   # local var name, or "self.x" for class-field bindings
    type_raw_name: str              # constructor / RHS type name, as-written
    line: int                       # 1-based line of the assignment


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
    # Sprint 13a — derived community structure. Populated by
    # `phases.community_detect.CommunityDetectPhase` after resolution; the
    # loader writes them to the graph via MEMBER_OF edges.
    communities: list[CommunityNode] = field(default_factory=list)
    member_of: list[MemberOfEdge] = field(default_factory=list)
    # Sprint 13b — derived processes (execution flows). Populated by
    # `phases.process_extract.ProcessExtractPhase`; the loader writes them
    # via STEP_IN_PROCESS edges. No-op when 13a's community phase didn't run.
    processes: list[ProcessNode] = field(default_factory=list)
    step_in_process: list[StepInProcessEdge] = field(default_factory=list)
    # Sprint 14a — per-symbol embeddings. Populated by `phases.embed.EmbedPhase`
    # (opt-in via --with-embeddings; not in DEFAULT_PHASES). The loader writes
    # them as a `LIST<FLOAT>` property on the corresponding Function or Class
    # node. Empty list ⇒ loader writes nothing (the embed query is a no-op).
    embeddings: list[EmbeddingUpdate] = field(default_factory=list)
    # Sprint 14g — local variable + class-field type bindings. Populated by
    # the extractor for `x = SomeClass(...)` patterns; consumed by the
    # resolver's LocalVarTypeIndex. Not persisted to Neo4j (resolution-only
    # state). Empty list ⇒ no extra typeBinding-driven resolutions happen.
    local_var_bindings: list[LocalVarBinding] = field(default_factory=list)

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
        self.communities.extend(other.communities)
        self.member_of.extend(other.member_of)
        self.processes.extend(other.processes)
        self.step_in_process.extend(other.step_in_process)
        self.embeddings.extend(other.embeddings)
        self.local_var_bindings.extend(other.local_var_bindings)

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
            "communities": len(self.communities),
            "member_of_edges": len(self.member_of),
            "processes": len(self.processes),
            "step_in_process_edges": len(self.step_in_process),
            "embedding_updates": len(self.embeddings),
            "local_var_bindings": len(self.local_var_bindings),
        }
