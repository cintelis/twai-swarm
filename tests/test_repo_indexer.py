"""Repo-indexer unit tests — walker + extractor.

The Neo4j loader is exercised in test_repo_indexer_neo4j.py (integration;
skipped when NEO4J_URL isn't set). Here we verify the AST traversal
produces the expected IndexBatch shape from a known Python source.
"""
from __future__ import annotations

import pytest

from app.repo_indexer.actions import IndexBatch, RepoNode
from app.repo_indexer.walker import walk_repo

# Tree-sitter is required for the extractor tests; skip if unavailable
# rather than failing the whole module import.
try:
    import tree_sitter_python  # noqa: F401
    from app.repo_indexer.extractor_python import extract_python_file
    from app.repo_indexer.__main__ import _parser_for_python
    HAS_TS = True
except ImportError:
    HAS_TS = False


REPO = RepoNode(name="testrepo", url="", commit_sha="abc123")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter / tree-sitter-python not installed")
    return _parser_for_python()


# ─── walker ─────────────────────────────────────────────────────────────────

def test_walker_yields_python_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def x(): pass", encoding="utf-8")
    (tmp_path / "src" / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("def y(): pass", encoding="utf-8")

    found = list(walk_repo(tmp_path))
    rel_paths = [r[0] for r in found]
    assert "src/main.py" in rel_paths
    assert "src/data.json" not in rel_paths       # wrong extension
    assert "node_modules/ignored.py" not in rel_paths  # SKIP_DIRS pruned


def test_walker_respects_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.generated.py\nbuilt/\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("def x(): pass", encoding="utf-8")
    (tmp_path / "thing.generated.py").write_text("def y(): pass", encoding="utf-8")
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "z.py").write_text("def z(): pass", encoding="utf-8")

    rel_paths = [r[0] for r in walk_repo(tmp_path)]
    assert "real.py" in rel_paths
    assert "thing.generated.py" not in rel_paths
    assert all(not p.startswith("built/") for p in rel_paths)


def test_walker_emits_sha(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    found = list(walk_repo(tmp_path))
    assert len(found) == 1
    rel_path, source, lang, sha = found[0]
    assert lang == "python"
    assert len(sha) == 64  # sha-256 hex


# ─── extractor ──────────────────────────────────────────────────────────────

def test_extractor_top_level_function(parser):
    src = b'def add(a, b):\n    """Sum of two numbers."""\n    return a + b\n'
    batch = extract_python_file(REPO, "math.py", src, "sha-1", parser)

    assert len(batch.functions) == 1
    fn = batch.functions[0]
    assert fn.qualified_name == "math.add"
    assert fn.name == "add"
    assert fn.is_async is False
    assert fn.is_method is False
    assert fn.parent_class_qn == ""
    assert fn.params == ("a", "b")
    assert fn.docstring == "Sum of two numbers."
    assert fn.line_start == 1


def test_extractor_async_function(parser):
    src = b"async def fetch(url):\n    return await get(url)\n"
    batch = extract_python_file(REPO, "io.py", src, "sha-2", parser)

    fn = batch.functions[0]
    assert fn.qualified_name == "io.fetch"
    assert fn.is_async is True


def test_extractor_class_with_methods(parser):
    src = b"""class Greeter:
    def __init__(self, name):
        self.name = name
    async def greet(self):
        return f"hi {self.name}"
"""
    batch = extract_python_file(REPO, "greet.py", src, "sha-3", parser)

    assert len(batch.classes) == 1
    cls = batch.classes[0]
    assert cls.qualified_name == "greet.Greeter"
    assert cls.name == "Greeter"

    # Two methods, both with parent_class_qn set.
    methods = [f for f in batch.functions if f.is_method]
    assert len(methods) == 2
    qns = {m.qualified_name for m in methods}
    assert qns == {"greet.Greeter.__init__", "greet.Greeter.greet"}
    greet_method = next(m for m in methods if m.name == "greet")
    assert greet_method.is_async is True
    assert greet_method.parent_class_qn == "greet.Greeter"


def test_extractor_inheritance(parser):
    src = b"class Child(Parent):\n    pass\n"
    batch = extract_python_file(REPO, "x.py", src, "sha-4", parser)

    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "x.Child"
    # Parent is unresolved here (would need cross-file pass), so it goes
    # in as the bare dotted name and a Symbol node is emitted.
    assert edge.parent_qn == "Parent"
    assert any(s.qualified_name == "Parent" for s in batch.symbols)


def test_extractor_intrafile_call_resolves(parser):
    src = b"""def helper():
    return 1
def main():
    return helper()
"""
    batch = extract_python_file(REPO, "main.py", src, "sha-5", parser)

    calls = batch.calls
    assert len(calls) == 1
    edge = calls[0]
    assert edge.caller_qn == "main.main"
    # Same-file top-level call resolves to a Function QN, not a Symbol.
    assert edge.callee_qn == "main.helper"


def test_extractor_external_call_emits_symbol(parser):
    src = b"""import json
def parse(data):
    return json.loads(data)
"""
    batch = extract_python_file(REPO, "x.py", src, "sha-6", parser)

    # The json.loads call is recorded as an edge to a SymbolNode.
    edges = batch.calls
    assert any(e.callee_qn == "json.loads" for e in edges)
    assert any(s.qualified_name == "json.loads" for s in batch.symbols)


def test_extractor_imports(parser):
    src = b"""import os
import json as j
from pathlib import Path
"""
    batch = extract_python_file(REPO, "x.py", src, "sha-7", parser)

    targets = sorted({i.target_qn for i in batch.imports})
    assert "os" in targets
    assert "json" in targets
    assert "pathlib" in targets


def test_extractor_init_module_qn(parser):
    """`pkg/__init__.py` should map to module qn `pkg`, not `pkg.__init__`."""
    src = b"def setup(): pass\n"
    batch = extract_python_file(REPO, "pkg/__init__.py", src, "sha-8", parser)
    assert any(m.qualified_name == "pkg" for m in batch.modules)
    assert batch.functions[0].qualified_name == "pkg.setup"


# ─── batch helpers ──────────────────────────────────────────────────────────

def test_batch_extend_merges():
    a = IndexBatch(repo=REPO)
    b = IndexBatch(repo=REPO)
    from app.repo_indexer.actions import FileNode
    a.files.append(FileNode(repo="testrepo", path="a.py", language="python", sha="x"))
    b.files.append(FileNode(repo="testrepo", path="b.py", language="python", sha="y"))
    a.extend(b)
    assert len(a.files) == 2


def test_batch_extend_rejects_repo_mismatch():
    a = IndexBatch(repo=REPO)
    other = RepoNode(name="other", url="", commit_sha="")
    b = IndexBatch(repo=other)
    with pytest.raises(ValueError, match="different repos"):
        a.extend(b)


def test_batch_counts():
    batch = IndexBatch(repo=REPO)
    counts = batch.counts()
    assert all(v == 0 for v in counts.values())
    assert set(counts.keys()) == {
        "files", "modules", "classes", "functions", "symbols",
        "inherits_edges", "call_edges", "import_edges",
    }
