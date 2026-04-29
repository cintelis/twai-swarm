"""Repo-indexer unit tests — walker + extractor.

The Neo4j loader is exercised in test_repo_indexer_neo4j.py (integration;
skipped when NEO4J_URL isn't set). Here we verify the AST traversal
produces the expected IndexBatch shape from a known Python source.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from app.repo_indexer.actions import IndexBatch, RepoNode
from app.repo_indexer.walker import walk_paths, walk_repo

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


# ─── Sprint 10g — path-only walker ──────────────────────────────────────────

def test_walk_paths_returns_paths_only(tmp_path):
    """walk_paths must NOT read file bytes — it just returns (path, language)."""
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    (tmp_path / "b.ts").write_text("export const x = 1;", encoding="utf-8")
    (tmp_path / "skip.md").write_text("# not a source file", encoding="utf-8")

    found = list(walk_paths(tmp_path))
    rel_paths = sorted(p for p, _lang in found)
    languages = {p: lang for p, lang in found}

    assert rel_paths == ["a.py", "b.ts"]
    assert languages["a.py"] == "python"
    assert languages["b.ts"] == "typescript"


def test_walk_paths_does_not_open_files(tmp_path, monkeypatch):
    """Critical 10g invariant: walk_paths must not call read_bytes — that
    was the I/O sink we paid 2x on TS+JS scans."""
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    (tmp_path / "b.ts").write_text("export const x = 1;", encoding="utf-8")

    opens = []
    real_read_bytes = Path.read_bytes

    def spy_read_bytes(self):
        opens.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    list(walk_paths(tmp_path))
    # walk_paths reads .gitignore at most; never the source files themselves.
    assert all(name == ".gitignore" for name in opens), f"unexpected reads: {opens}"


def test_walk_paths_respects_gitignore_and_skip_dirs(tmp_path):
    (tmp_path / ".gitignore").write_text("*.generated.py\nbuilt/\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("def x(): pass", encoding="utf-8")
    (tmp_path / "thing.generated.py").write_text("def y(): pass", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.ts").write_text("export const x = 1;", encoding="utf-8")
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "z.ts").write_text("export const z = 1;", encoding="utf-8")

    rel_paths = sorted(p for p, _lang in walk_paths(tmp_path))
    assert rel_paths == ["real.py"]


def test_chunked_write_splits_rows():
    """_chunked_write breaks a big row list into multiple session.run calls
    so a single Cypher payload never exceeds the chunk size."""
    from app.repo_indexer.loader import _chunked_write, WRITE_CHUNK_SIZE

    captured: list[int] = []

    class FakeSession:
        def run(self, query: str, **kwargs):
            captured.append(len(kwargs["rows"]))

    rows = [{"i": i} for i in range(2500)]
    _chunked_write(FakeSession(), "UNWIND $rows AS r RETURN r", rows, chunk_size=1000)
    assert captured == [1000, 1000, 500]
    assert WRITE_CHUNK_SIZE == 1000  # invariant we rely on for production scans


def test_chunked_write_skips_empty_list():
    from app.repo_indexer.loader import _chunked_write

    class FakeSession:
        def run(self, *a, **kw):
            raise AssertionError("should not call session.run on empty rows")
    _chunked_write(FakeSession(), "UNWIND $rows AS r RETURN r", [])


def test_walker_reads_each_file_exactly_once(tmp_path, monkeypatch):
    """10g invariant: walk_repo opens each source file ONCE (not twice
    like the old version that did a streaming SHA on a second open)."""
    (tmp_path / "a.py").write_text("def a(): pass", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b(): pass", encoding="utf-8")

    opens: list[str] = []
    real_read_bytes = Path.read_bytes

    def spy_read_bytes(self):
        opens.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    list(walk_repo(tmp_path))

    # .gitignore may be read once. Each source file exactly once.
    source_opens = [n for n in opens if n.endswith(".py")]
    assert sorted(source_opens) == ["a.py", "b.py"], source_opens


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
    # Extractor records the bare dotted name as observed; the resolver
    # decides post-pass whether it lands on a Class or a Symbol.
    assert edge.parent_qn == "Parent"
    # Sprint 10b: extractor no longer emits Symbol nodes eagerly — that's
    # the resolver's job.
    assert batch.symbols == []


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


def test_extractor_flattens_super_call_chain(parser):
    """`super().method()` should produce a CallEdge with callee_qn = `super.method`
    so finalize.py's super() resolution branch can pick it up. Pre-fix,
    _flatten_attribute returned None on the inner `call` node, swallowing
    the `.method` suffix entirely.
    """
    src = b"""class Child(Parent):
    def __init__(self, name):
        super().__init__(name)
        super().validate()
"""
    batch = extract_python_file(REPO, "child.py", src, "sha-super", parser)

    calls_by_callee = {edge.callee_qn for edge in batch.calls}
    assert "super.__init__" in calls_by_callee
    assert "super.validate" in calls_by_callee
    # No bare-`super` callees — those would mean we lost the method name.
    assert "super" not in calls_by_callee


def test_extractor_emits_wildcard_import(parser):
    """`from x import *` should produce an ImportEdge with local_name='*'
    and kind='module' so finalize.py's _is_wildcard branch picks it up.
    Pre-fix the wildcard_import tree-sitter node was silently skipped.
    """
    src = b"""from app.helpers import *
from app.utils import named_helper

def consumer():
    return named_helper()
"""
    batch = extract_python_file(REPO, "consumer.py", src, "sha-wild", parser)

    by_target = {(imp.target_qn, imp.local_name, imp.kind) for imp in batch.imports}
    assert ("app.helpers", "*", "module") in by_target
    # The non-wildcard import should still be there with the regular shape.
    assert ("app.utils.named_helper", "named_helper", "symbol") in by_target


def test_extractor_external_call_no_symbol_until_resolver(parser):
    """Extractor records the dotted name verbatim; Symbols come from the
    resolver, not the extractor."""
    src = b"""import json
def parse(data):
    return json.loads(data)
"""
    batch = extract_python_file(REPO, "x.py", src, "sha-6", parser)

    edges = batch.calls
    assert any(e.callee_qn == "json.loads" for e in edges)
    assert batch.symbols == []   # extractor doesn't emit them


def test_extractor_imports_capture_local_name_and_kind(parser):
    src = b"""import os
import json as j
from pathlib import Path
from app.foo import bar as baz
"""
    batch = extract_python_file(REPO, "x.py", src, "sha-7", parser)

    by_local = {i.local_name: i for i in batch.imports}
    assert by_local["os"].target_qn == "os"
    assert by_local["os"].kind == "module"
    assert by_local["j"].target_qn == "json"
    assert by_local["j"].kind == "module"
    assert by_local["Path"].target_qn == "pathlib.Path"
    assert by_local["Path"].kind == "symbol"
    assert by_local["baz"].target_qn == "app.foo.bar"
    assert by_local["baz"].kind == "symbol"


def test_extractor_captures_param_types(parser):
    src = b"""def handle(box: Sandbox, n: int = 5):
    return box.run(n)
"""
    batch = extract_python_file(REPO, "x.py", src, "sha-pt", parser)

    fn = batch.functions[0]
    types = dict(fn.param_types)
    assert types == {"box": "Sandbox", "n": "int"}


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
        # Sprint 13a — community detection.
        "communities", "member_of_edges",
        # Sprint 13b — process extraction.
        "processes", "step_in_process_edges",
        # Sprint 14a — embeddings bridge.
        "embedding_updates",
        # Sprint 14g — local variable type bindings (resolution-only state).
        "local_var_bindings",
        # Sprint 15a — HTTP route definitions and HANDLED_BY edges.
        "routes",
        "route_edges",
    }


# ─── Sprint 14g — local var typeBinding extraction ──────────────────────────

def test_extractor_emits_local_var_binding_for_constructor(parser):
    """`x = SomeClass(...)` inside a function emits a LocalVarBinding
    pointing at the enclosing function's line range."""
    src = b"def use_it():\n    builder = StateGraph(state)\n    builder.add_node(x)\n"
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)

    bindings = batch.local_var_bindings
    assert len(bindings) == 1
    b = bindings[0]
    assert b.var_name == "builder"
    assert b.type_raw_name == "StateGraph"
    assert b.enclosing_scope_kind == "function"
    assert b.enclosing_line_start == 1
    assert b.enclosing_line_end == 3
    assert b.line == 2  # the assignment line


def test_extractor_emits_dotted_callee_assignment(parser):
    """Sprint 14h — `u = models.User(...)` and `g = builder.compile()`
    both produce bindings with a DOTTED `type_raw_name`. The resolver
    interprets the dotted form: case-5 namespace prefix (`models.User`)
    or method-call chain (`builder.compile`'s return type)."""
    src = b"def use_it():\n    u = models.User()\n"
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)
    bindings = batch.local_var_bindings
    assert len(bindings) == 1
    assert bindings[0].var_name == "u"
    assert bindings[0].type_raw_name == "models.User"


def test_extractor_skips_function_call_assignment(parser):
    """`x = func()` is return-type tracking (Sprint 14h territory).
    14g.1 doesn't differentiate function-call from constructor-call at
    extraction time, so this currently DOES emit a binding with the
    function's name as the type. The resolver fails to find a class by
    that name and falls through cleanly. Acceptance test: doesn't crash,
    finalize doesn't mistakenly resolve."""
    src = b"def helper():\n    pass\n\ndef use_it():\n    x = helper()\n"
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)
    # We DO emit the binding — extractor doesn't know `helper` is a
    # function vs a class. The resolver's `_resolve_type_name` won't
    # find a class named `helper`, so the binding is harmless.
    bindings = batch.local_var_bindings
    assert len(bindings) == 1
    assert bindings[0].type_raw_name == "helper"


def test_extractor_skips_multi_target_assignment(parser):
    """`x, y = func()` — multi-target. Out of 14g.1 scope."""
    src = b"def use_it():\n    a, b = make_pair()\n"
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)
    assert batch.local_var_bindings == []


def test_extractor_skips_augmented_assignment(parser):
    """`x += 1` — augmented; LHS already typed, not a new binding."""
    src = b"def use_it():\n    x = 0\n    x += 1\n"
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)
    # Only the literal `x = 0` matches the assignment shape. Its RHS
    # is a number, not a Call, so no binding emitted.
    assert batch.local_var_bindings == []


def test_extractor_emits_per_function_independently(parser):
    """Two functions, each with one assignment — both bindings are
    emitted with the right enclosing-scope ranges."""
    src = (
        b"def f():\n    a = ClassA()\n\n"
        b"def g():\n    b = ClassB()\n"
    )
    batch = extract_python_file(REPO, "use.py", src, "sha", parser)
    bindings = sorted(batch.local_var_bindings, key=lambda b: b.var_name)
    assert len(bindings) == 2
    assert bindings[0].var_name == "a"
    assert bindings[0].type_raw_name == "ClassA"
    assert bindings[1].var_name == "b"
    assert bindings[1].type_raw_name == "ClassB"
    # Per-function enclosing ranges should differ.
    assert bindings[0].enclosing_line_start != bindings[1].enclosing_line_start


# ─── Sprint 14g.2 — class-field typeBinding extraction (self.x = ...) ────────

def test_extractor_emits_class_field_binding_from_init(parser):
    """`self.x = SomeClass()` in __init__ produces a class-scoped
    binding so other methods can resolve `self.x.method()`."""
    src = (
        b"class Service:\n"
        b"    def __init__(self):\n"
        b"        self.client = ApiClient()\n"
        b"        self.cache = Cache()\n"
    )
    batch = extract_python_file(REPO, "svc.py", src, "sha", parser)

    # Field bindings stored on the CLASS scope, not __init__'s function scope.
    field_bindings = [b for b in batch.local_var_bindings
                      if b.enclosing_scope_kind == "class"]
    assert len(field_bindings) == 2
    by_name = {b.var_name: b for b in field_bindings}
    assert by_name["client"].type_raw_name == "ApiClient"
    assert by_name["cache"].type_raw_name == "Cache"
    # Both share the class's line range (1..4 inclusive → 1..4).
    assert by_name["client"].enclosing_line_start == 1
    assert by_name["client"].enclosing_line_end == 4


def test_extractor_class_field_binding_has_field_name_only(parser):
    """`self.client = ApiClient()` produces var_name="client", NOT
    "self.client". The receiver-resolver does `find(class_scope,
    "client", tree)` for `self.client.method()` lookups."""
    src = (
        b"class S:\n"
        b"    def __init__(self):\n"
        b"        self.thing = Thing()\n"
    )
    batch = extract_python_file(REPO, "s.py", src, "sha", parser)
    field_bindings = [b for b in batch.local_var_bindings
                      if b.enclosing_scope_kind == "class"]
    assert len(field_bindings) == 1
    assert field_bindings[0].var_name == "thing"  # NOT "self.thing"


def test_extractor_skips_class_field_outside_method(parser):
    """Top-level `def f(): self.x = X()` is meaningless (no enclosing
    class) and shouldn't emit a class-scoped binding. Should pass since
    the extraction only runs when is_method=True."""
    src = b"def f(self):\n    self.x = X()\n"
    batch = extract_python_file(REPO, "f.py", src, "sha", parser)
    # No class-scoped bindings; only the function-scope ones (none for
    # this shape since LHS is `self.x`, not a bare identifier).
    field_bindings = [b for b in batch.local_var_bindings
                      if b.enclosing_scope_kind == "class"]
    assert field_bindings == []


def test_extractor_emits_both_local_and_field_bindings(parser):
    """A method can have BOTH local-var bindings and class-field
    bindings. They live at different scopes."""
    src = (
        b"class S:\n"
        b"    def setup(self):\n"
        b"        self.client = ApiClient()\n"   # class-scope binding
        b"        helper = Helper()\n"           # function-scope binding
    )
    batch = extract_python_file(REPO, "s.py", src, "sha", parser)

    fn_bindings = [b for b in batch.local_var_bindings
                   if b.enclosing_scope_kind == "function"]
    cls_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "class"]

    assert len(fn_bindings) == 1
    assert fn_bindings[0].var_name == "helper"
    assert fn_bindings[0].type_raw_name == "Helper"

    assert len(cls_bindings) == 1
    assert cls_bindings[0].var_name == "client"
    assert cls_bindings[0].type_raw_name == "ApiClient"


# ─── Sprint 14i — module-level typeBinding extraction ──────────────────────

def test_extractor_emits_module_level_binding(parser):
    """`app = FastAPI()` at module top emits a module-scoped binding."""
    src = b"app = FastAPI()\n\ndef handler():\n    pass\n"
    batch = extract_python_file(REPO, "main.py", src, "sha", parser)

    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    assert len(mod_bindings) == 1
    b = mod_bindings[0]
    assert b.var_name == "app"
    assert b.type_raw_name == "FastAPI"
    assert b.line == 1


def test_extractor_does_not_double_walk_into_function_bodies(parser):
    """The module-level walker must NOT recurse into function bodies —
    those are visited separately by `_walk_assignments`. If both fired,
    we'd get duplicate bindings (one at module scope, one at function
    scope) for the same source line."""
    src = (
        b"app = FastAPI()\n"
        b"\n"
        b"def handler():\n"
        b"    helper = Helper()\n"
    )
    batch = extract_python_file(REPO, "main.py", src, "sha", parser)

    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    fn_bindings = [b for b in batch.local_var_bindings
                   if b.enclosing_scope_kind == "function"]

    # Each should have exactly one binding; no overlap.
    assert len(mod_bindings) == 1
    assert mod_bindings[0].var_name == "app"
    assert len(fn_bindings) == 1
    assert fn_bindings[0].var_name == "helper"


def test_extractor_does_not_double_walk_into_class_bodies(parser):
    """Class-body assignments at the class level (not inside methods)
    are NOT module-level. They're class-scope, but our extractor today
    doesn't capture class-level annotated/initialized fields (only
    `self.x = ...` in __init__). Verify the walker doesn't mistakenly
    emit them as module-level."""
    src = (
        b"app = FastAPI()\n"
        b"\n"
        b"class Settings:\n"
        b"    db = Database()\n"   # class-level — currently not captured
    )
    batch = extract_python_file(REPO, "main.py", src, "sha", parser)

    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    # Only `app = FastAPI()` is module-level. `db = Database()` is inside
    # a class body — not module-level. (And we don't currently support
    # class-level field initialization without `self.` — that's a 14j
    # follow-up if needed.)
    assert [b.var_name for b in mod_bindings] == ["app"]


def test_extractor_skips_module_level_dotted_constructor(parser):
    """`x = models.User()` at module level is case-5 territory (namespace
    prefix) — same as the function-body skip rule."""
    src = b"u = models.User()\n"
    batch = extract_python_file(REPO, "main.py", src, "sha", parser)
    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    assert mod_bindings == []


def test_extractor_module_binding_has_sentinel_range(parser):
    """Module bindings use a sentinel (0, MODULE_SCOPE_END-1) line range
    so the adapter's _range_for produces a Range that exactly matches
    the Module ScopeId in `to_scopes` — same shape, structural equality
    in the LocalVarTypeIndex."""
    from app.repo_indexer.scope_resolution._adapter import MODULE_SCOPE_END
    src = b"app = FastAPI()\n"
    batch = extract_python_file(REPO, "main.py", src, "sha", parser)
    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    assert len(mod_bindings) == 1
    assert mod_bindings[0].enclosing_line_start == 0
    assert mod_bindings[0].enclosing_line_end == MODULE_SCOPE_END - 1


# ─── Sprint 14h — return-type extraction ────────────────────────────────────

def test_extractor_captures_function_return_annotation(parser):
    """`def make_user() -> User:` populates FunctionNode.return_type_raw."""
    src = b"def make_user() -> User:\n    return User()\n"
    batch = extract_python_file(REPO, "a.py", src, "sha", parser)
    assert len(batch.functions) == 1
    assert batch.functions[0].return_type_raw == "User"


def test_extractor_normalizes_optional_return():
    """`Optional[X]` → `X` (mirrors GitNexus's strip)."""
    from app.repo_indexer.extractor_python import _normalize_return_type
    assert _normalize_return_type("Optional[User]") == "User"
    assert _normalize_return_type("User | None") == "User"
    assert _normalize_return_type("None | User") == "User"


def test_extractor_normalizes_list_return():
    """Single-arg generic wrappers strip to the inner type."""
    from app.repo_indexer.extractor_python import _normalize_return_type
    assert _normalize_return_type("list[User]") == "User"
    assert _normalize_return_type("List[User]") == "User"
    assert _normalize_return_type("Iterable[User]") == "User"


def test_extractor_normalizes_quoted_forward_ref():
    """`def f() -> "User":` → `User`."""
    from app.repo_indexer.extractor_python import _normalize_return_type
    assert _normalize_return_type('"User"') == "User"
    assert _normalize_return_type("'User'") == "User"


def test_extractor_emits_module_return_binding_for_free_function(parser):
    """A free function with a return annotation emits a binding on the
    module scope keyed by the function name."""
    src = b"def make_user() -> User:\n    return User()\n"
    batch = extract_python_file(REPO, "a.py", src, "sha", parser)
    mod_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "module"]
    assert len(mod_bindings) == 1
    assert mod_bindings[0].var_name == "make_user"
    assert mod_bindings[0].type_raw_name == "User"


def test_extractor_emits_class_return_binding_for_method(parser):
    """A method with a return annotation emits a binding on its CLASS
    scope keyed by the method name (auto-hoist semantics — mirrors
    GitNexus's `pass4CollectTypeBindings` parent-hoist for return
    types)."""
    src = (
        b"class StateGraph:\n"
        b"    def compile(self) -> CompiledStateGraph:\n"
        b"        return CompiledStateGraph()\n"
    )
    batch = extract_python_file(REPO, "g.py", src, "sha", parser)
    cls_bindings = [b for b in batch.local_var_bindings
                    if b.enclosing_scope_kind == "class"]
    assert len(cls_bindings) == 1
    assert cls_bindings[0].var_name == "compile"
    assert cls_bindings[0].type_raw_name == "CompiledStateGraph"
    # The class scope is StateGraph's range, NOT compile's function range.
    assert cls_bindings[0].enclosing_line_start == 1


def test_extractor_dotted_callee_emits_chained_binding(parser):
    """`g = builder.compile()` produces a binding with dotted
    `type_raw_name = "builder.compile"` — the resolver interprets the
    dotted form via the chain-resolve helper."""
    src = (
        b"def use():\n"
        b"    builder = StateGraph()\n"
        b"    g = builder.compile()\n"
    )
    batch = extract_python_file(REPO, "u.py", src, "sha", parser)
    fn_bindings = [b for b in batch.local_var_bindings
                   if b.enclosing_scope_kind == "function"]
    by_name = {b.var_name: b for b in fn_bindings}
    assert by_name["builder"].type_raw_name == "StateGraph"
    assert by_name["g"].type_raw_name == "builder.compile"  # dotted!
