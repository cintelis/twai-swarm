"""TypeScript / TSX / JavaScript extractor — Sprint 10d.

Mirrors the Python extractor tests but exercises the TS family. Skip
gracefully if tree-sitter-typescript isn't installed (matches how the
Python tests handle their grammar).
"""
from __future__ import annotations

import pytest

from app.repo_indexer.actions import RepoNode

try:
    import tree_sitter_typescript  # noqa: F401
    from tree_sitter import Language, Parser
    from app.repo_indexer.extractor_typescript import (
        extract_typescript_file,
        module_qn_from_path,
        _resolve_relative_import,
    )
    HAS_TS = True
except ImportError:
    HAS_TS = False


REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def ts_parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-typescript not installed")
    import tree_sitter_typescript as tsts
    return Parser(Language(tsts.language_typescript()))


@pytest.fixture
def tsx_parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-typescript not installed")
    import tree_sitter_typescript as tsts
    return Parser(Language(tsts.language_tsx()))


# ─── module_qn_from_path ────────────────────────────────────────────────────

def test_module_qn_strips_extension():
    assert module_qn_from_path("app/api/route.ts") == "app.api.route"
    assert module_qn_from_path("app/api/route.tsx") == "app.api.route"
    assert module_qn_from_path("lib/util.js") == "lib.util"
    assert module_qn_from_path("Component.jsx") == "Component"


def test_module_qn_collapses_index():
    """`lib/index.ts` is the package root in JS convention."""
    assert module_qn_from_path("lib/index.ts") == "lib"
    assert module_qn_from_path("app/sandbox/index.tsx") == "app.sandbox"
    assert module_qn_from_path("index.ts") == ""


# ─── _resolve_relative_import ───────────────────────────────────────────────

def test_resolve_relative_sibling():
    files = {"app/foo.ts", "app/bar.ts"}
    assert _resolve_relative_import("app/foo.ts", "./bar", files) == "app/bar.ts"


def test_resolve_relative_parent():
    files = {"app/api/route.ts", "lib/shared.ts"}
    assert _resolve_relative_import("app/api/route.ts", "../../lib/shared", files) == "lib/shared.ts"


def test_resolve_relative_index():
    files = {"app/foo.ts", "app/sandbox/index.ts"}
    assert _resolve_relative_import("app/foo.ts", "./sandbox", files) == "app/sandbox/index.ts"


def test_resolve_relative_returns_none_for_external():
    """Bare specifiers (`react`, `node:path`) aren't in-repo paths."""
    assert _resolve_relative_import("app/foo.ts", "react", {"app/foo.ts"}) is None
    assert _resolve_relative_import("app/foo.ts", "@/lib/x", {"app/foo.ts"}) is None


def test_resolve_relative_returns_none_when_not_in_set():
    assert _resolve_relative_import("app/foo.ts", "./missing", {"app/foo.ts"}) is None


def test_resolve_relative_strips_js_extension_for_ts_esm():
    """TS-ESM requires explicit `.js` in imports even when the source is
    `.ts`. The resolver must strip the `.js` (or `.jsx`) and probe with
    the actual source extensions. Without this, GitNexus-style imports
    `from '../../lib/utils.js'` always fail to resolve."""
    files = {"app/foo.ts", "lib/utils.ts"}
    assert _resolve_relative_import("app/foo.ts", "../lib/utils.js", files) == "lib/utils.ts"


def test_resolve_relative_strips_jsx_extension():
    """Same for `.jsx` → `.tsx` substitution."""
    files = {"app/foo.tsx", "components/Button.tsx"}
    assert _resolve_relative_import(
        "app/foo.tsx", "../components/Button.jsx", files,
    ) == "components/Button.tsx"


def test_resolve_relative_explicit_ts_extension_works():
    """Explicit `.ts` in the specifier should also work (some configs use
    this; should round-trip cleanly)."""
    files = {"app/foo.ts", "lib/utils.ts"}
    assert _resolve_relative_import("app/foo.ts", "../lib/utils.ts", files) == "lib/utils.ts"


# ─── extractor — top-level shape ────────────────────────────────────────────

def test_extracts_top_level_function(ts_parser):
    src = b"""export function add(a: number, b: number): number {
  return a + b;
}
"""
    batch = extract_typescript_file(REPO, "math.ts", src, "sha1", ts_parser, repo_files=set())
    assert len(batch.functions) == 1
    fn = batch.functions[0]
    assert fn.qualified_name == "math.add"
    assert fn.name == "add"
    assert fn.is_async is False
    assert fn.is_method is False
    assert fn.params == ("a", "b")
    assert dict(fn.param_types) == {"a": "number", "b": "number"}


def test_extracts_async_function(ts_parser):
    src = b"export async function fetch(url: string): Promise<string> { return url; }\n"
    batch = extract_typescript_file(REPO, "io.ts", src, "sha2", ts_parser, repo_files=set())
    assert batch.functions[0].is_async is True


def test_extracts_arrow_const(ts_parser):
    src = b"export const greet = (name: string): string => `hi ${name}`;\n"
    batch = extract_typescript_file(REPO, "x.ts", src, "sha3", ts_parser, repo_files=set())
    fn = batch.functions[0]
    assert fn.qualified_name == "x.greet"
    assert fn.name == "greet"
    assert dict(fn.param_types) == {"name": "string"}


def test_extracts_class_with_methods(ts_parser):
    src = b"""export class Greeter {
  constructor(private name: string) {}
  async greet(): Promise<string> {
    return `hi ${this.name}`;
  }
}
"""
    batch = extract_typescript_file(REPO, "g.ts", src, "sha4", ts_parser, repo_files=set())
    assert len(batch.classes) == 1
    cls = batch.classes[0]
    assert cls.qualified_name == "g.Greeter"

    methods = [f for f in batch.functions if f.is_method]
    qns = {m.qualified_name for m in methods}
    assert "g.Greeter.greet" in qns


def test_extracts_inheritance(ts_parser):
    src = b"export class Sub extends Base {}\n"
    batch = extract_typescript_file(REPO, "x.ts", src, "sha5", ts_parser, repo_files=set())
    assert len(batch.inherits) == 1
    edge = batch.inherits[0]
    assert edge.child_qn == "x.Sub"
    assert edge.parent_qn == "Base"


def test_extracts_intrafile_call(ts_parser):
    src = b"""function helper() { return 1; }
function main() { return helper(); }
"""
    batch = extract_typescript_file(REPO, "main.ts", src, "sha6", ts_parser, repo_files=set())
    calls = batch.calls
    assert any(c.callee_qn == "main.helper" and c.caller_qn == "main.main" for c in calls)


def test_extracts_member_call(ts_parser):
    """`obj.method()` — extractor should record the dotted name as observed."""
    src = b"""function use(box: Sandbox) { return box.run("ls"); }
"""
    batch = extract_typescript_file(REPO, "x.ts", src, "sha7", ts_parser, repo_files=set())
    calls = batch.calls
    assert any(c.callee_qn == "box.run" for c in calls)


# ─── extractor — imports ────────────────────────────────────────────────────

def test_extracts_named_import_with_path_resolution(ts_parser):
    src = b'import { Sandbox } from "./sandbox";\nfunction f() { return new Sandbox(); }\n'
    files = {"app/sandbox.ts", "app/route.ts"}
    batch = extract_typescript_file(REPO, "app/route.ts", src, "sha8", ts_parser, repo_files=files)
    imports = {(i.local_name, i.target_qn, i.kind) for i in batch.imports}
    # Path resolved + symbol-kind binding for the named import.
    assert ("Sandbox", "app.sandbox.Sandbox", "symbol") in imports


def test_extracts_default_import(ts_parser):
    src = b'import path from "node:path";\n'
    batch = extract_typescript_file(REPO, "x.ts", src, "sha9", ts_parser, repo_files=set())
    imports = {(i.local_name, i.target_qn, i.kind) for i in batch.imports}
    # Bare specifier — kept as-observed.
    assert ("path", "node:path", "module") in imports


def test_extracts_namespace_import(ts_parser):
    src = b'import * as fs from "node:fs";\n'
    batch = extract_typescript_file(REPO, "x.ts", src, "sha10", ts_parser, repo_files=set())
    imports = {(i.local_name, i.target_qn, i.kind) for i in batch.imports}
    assert ("fs", "node:fs", "module") in imports


def test_extracts_aliased_named_import(ts_parser):
    src = b'import { Foo as MyFoo } from "./bar";\n'
    files = {"app/bar.ts", "app/x.ts"}
    batch = extract_typescript_file(REPO, "app/x.ts", src, "sha11", ts_parser, repo_files=files)
    imports = {(i.local_name, i.target_qn, i.kind) for i in batch.imports}
    assert ("MyFoo", "app.bar.Foo", "symbol") in imports


def test_extracts_unresolved_relative_falls_back(ts_parser):
    """Relative import that doesn't match any known file — keep raw spec."""
    src = b'import { X } from "./missing";\n'
    batch = extract_typescript_file(REPO, "app/x.ts", src, "sha12", ts_parser, repo_files={"app/x.ts"})
    imports = {(i.local_name, i.target_qn, i.kind) for i in batch.imports}
    assert ("X", "./missing.X", "symbol") in imports


# ─── TSX-specific (JSX literal handling) ────────────────────────────────────

def test_tsx_parser_handles_jsx_literal(tsx_parser):
    """The TSX grammar must parse JSX without erroring; regular TS would not."""
    src = b"""import React from "react";
export function Hello({ name }: { name: string }) {
  return <div>hi {name}</div>;
}
"""
    batch = extract_typescript_file(REPO, "Hello.tsx", src, "sha13", tsx_parser, repo_files=set())
    fn_names = {f.name for f in batch.functions}
    assert "Hello" in fn_names
