"""Repo indexer — turn a source tree into a Neo4j call graph.

Sprint 10a: Python only. Sprint 10b adds TypeScript.

Pipeline:
    walker → parser → extractor → loader

    walker     — yields (rel_path, source_bytes, language) for every
                 source file in a repo, respecting .gitignore + a hardcoded
                 denylist (node_modules, .venv, etc.)
    parser     — tree-sitter wrapper; turns source bytes into an AST
    extractor  — language-specific AST → IndexAction (typed records)
    loader     — batches IndexActions and MERGEs them into Neo4j

Schema (see deploy/terraform/neo4j.tf header for the rationale):
    (Repo {name, url, commit_sha, scanned_at, tenant_id})
      -[:CONTAINS]-> (File {path, language, sha})
                       -[:DEFINES]-> (Module {qualified_name})
                                       -[:DEFINES]-> (Class | Function)
    (Class)    -[:INHERITS_FROM]-> (Class)
    (Class)    -[:DEFINES]->        (Function)        // methods
    (Function) -[:CALLS {line}]->   (Function | Symbol)
    (File)     -[:IMPORTS]->        (Module | File)
    (Function) -[:USES_TYPE]->      (Class)

Idempotent: re-scanning the same commit produces identical state. Stale
nodes from a previous scan (different commit_sha) are pruned at the end
of each run.
"""
from .actions import (
    ClassNode,
    FileNode,
    FunctionNode,
    ImportEdge,
    IndexBatch,
    InheritsEdge,
    ModuleNode,
    RepoNode,
    SymbolNode,
    CallEdge,
)
from .scope_resolution.finalize import finalize_batch

__all__ = [
    "ClassNode",
    "FileNode",
    "FunctionNode",
    "ImportEdge",
    "IndexBatch",
    "InheritsEdge",
    "ModuleNode",
    "RepoNode",
    "SymbolNode",
    "CallEdge",
    "finalize_batch",
]
