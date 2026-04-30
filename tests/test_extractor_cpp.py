"""Sprint 16 — C++ extractor tests.

Synthetic source-only tests; no real codebase scanned.

Coverage by sub-sprint:
  16a  - simple, namespace_chain, const_overloads, pure_virtual
  16b  - inheritance_diamond
  16c  - call edges (bare / field / qualified / template / new)
  16d  - includes_chain (quoted vs angle, suffix matching)
  16e  - namespace + qualified-name in qns
  16f  - header_impl_pairing (out-of-line method linking via MERGE-on-qn)
"""
from __future__ import annotations

import pytest

try:
    import tree_sitter_cpp as _tscpp  # noqa: F401
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False


from app.repo_indexer.actions import RepoNode  # noqa: E402
from app.repo_indexer.extractor_cpp import (  # noqa: E402
    _module_qn_from_path,
    _resolve_quoted_include,
    extract_cpp_file,
)

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-cpp not installed")
    import tree_sitter_cpp as tscpp
    return Parser(Language(tscpp.language()))


# ─── helpers ────────────────────────────────────────────────────────────────

def _scan(parser, files: dict[str, bytes], target_path: str):
    """Run extract_cpp_file for `target_path` with all other files in
    repo_files (so quoted-include resolution works)."""
    repo_files = set(files.keys())
    return extract_cpp_file(
        REPO, target_path, files[target_path], "sha", parser, repo_files,
    )


# ─── 16a: module qn ────────────────────────────────────────────────────────

def test_module_qn_strips_extensions():
    assert _module_qn_from_path("src/audio/engine.cpp") == "src.audio.engine"
    assert _module_qn_from_path("include/twai/runtime/loop.h") == "include.twai.runtime.loop"
    assert _module_qn_from_path("foo.hpp") == "foo"
    assert _module_qn_from_path("foo.cc") == "foo"
    assert _module_qn_from_path("a/b/c.cxx") == "a.b.c"


# ─── 16a: simple ───────────────────────────────────────────────────────────

def test_simple_class_and_methods(parser):
    src = (
        b"class Engine {\n"
        b"public:\n"
        b"    void play();\n"
        b"    int volume() const;\n"
        b"};\n"
        b"int free_function(int x) { return x; }\n"
    )
    batch = extract_cpp_file(REPO, "engine.cpp", src, "sha", parser)
    assert {c.qualified_name for c in batch.classes} == {"engine.Engine"}
    fn_qns = {fn.qualified_name for fn in batch.functions}
    assert "engine.Engine.play" in fn_qns
    assert "engine.Engine.volume:const" in fn_qns
    assert "engine.free_function" in fn_qns


def test_method_const_disambiguation(parser):
    """`begin()` and `begin() const` must be DIFFERENT FunctionNodes."""
    src = (
        b"class It {\n"
        b"public:\n"
        b"    int begin();\n"
        b"    int begin() const;\n"
        b"};\n"
    )
    batch = extract_cpp_file(REPO, "it.cpp", src, "sha", parser)
    qns = {fn.qualified_name for fn in batch.functions}
    assert "it.It.begin" in qns
    assert "it.It.begin:const" in qns


# ─── 16a: namespace_chain ──────────────────────────────────────────────────

def test_namespace_in_qn(parser):
    src = (
        b"namespace twai::audio {\n"
        b"class Engine {\n"
        b"public:\n"
        b"    void play();\n"
        b"};\n"
        b"}\n"
    )
    batch = extract_cpp_file(REPO, "audio.cpp", src, "sha", parser)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert "audio.twai::audio::Engine" in cls_qns
    fn_qns = {fn.qualified_name for fn in batch.functions}
    assert "audio.twai::audio::Engine.play" in fn_qns


def test_anonymous_namespace_synthetic_name(parser):
    src = (
        b"namespace {\n"
        b"void helper() {}\n"
        b"}\n"
    )
    batch = extract_cpp_file(REPO, "u.cpp", src, "sha", parser)
    qns = [fn.qualified_name for fn in batch.functions]
    assert any("__anon_" in q and q.endswith(".helper") for q in qns), \
        f"expected anon helper, got {qns}"


def test_nested_namespaces_stack(parser):
    src = (
        b"namespace a {\n"
        b"namespace b {\n"
        b"void f() {}\n"
        b"}\n"
        b"}\n"
    )
    batch = extract_cpp_file(REPO, "n.cpp", src, "sha", parser)
    qns = {fn.qualified_name for fn in batch.functions}
    assert "n.a::b.f" in qns


# ─── 16b: inheritance ──────────────────────────────────────────────────────

def test_single_inheritance(parser):
    src = (
        b"class Base {};\n"
        b"class Derived : public Base {};\n"
    )
    batch = extract_cpp_file(REPO, "h.cpp", src, "sha", parser)
    edges = [(e.child_qn, e.parent_qn) for e in batch.inherits]
    assert ("h.Derived", "Base") in edges


def test_diamond_inheritance(parser):
    """Crib of GitNexus's cpp-diamond fixture."""
    src = (
        b"class Animal { public: virtual void speak(); };\n"
        b"class Bird : public Animal {};\n"
        b"class Swimmer : public Animal {};\n"
        b"class Duck : public Bird, public Swimmer {};\n"
    )
    batch = extract_cpp_file(REPO, "d.cpp", src, "sha", parser)
    edges = {(e.child_qn, e.parent_qn) for e in batch.inherits}
    assert edges == {
        ("d.Bird", "Animal"),
        ("d.Swimmer", "Animal"),
        ("d.Duck", "Bird"),
        ("d.Duck", "Swimmer"),
    }


def test_template_inheritance_uses_head(parser):
    """`class Foo : public Base<T>` → parent qn is `Base` (head only)."""
    src = (
        b"template<typename T> class Foo : public Base<T> {};\n"
    )
    batch = extract_cpp_file(REPO, "f.cpp", src, "sha", parser)
    parents = {e.parent_qn for e in batch.inherits}
    assert "Base" in parents


# ─── 16c: call edges ───────────────────────────────────────────────────────

def test_call_flavours(parser):
    src = (
        b"#include <memory>\n"
        b"void caller() {\n"
        b"    foo();\n"
        b"    obj.bar();\n"
        b"    ptr->baz();\n"
        b"    Foo::quux();\n"
        b"    std::make_shared<Engine>();\n"
        b"    auto e = new Engine();\n"
        b"}\n"
    )
    batch = extract_cpp_file(REPO, "c.cpp", src, "sha", parser)
    callees = {c.callee_qn for c in batch.calls}
    assert "foo" in callees
    assert "obj.bar" in callees
    assert "ptr.baz" in callees   # `->` flattened to `.`
    assert "Foo::quux" in callees
    assert "std::make_shared" in callees
    assert "Engine" in callees    # new_expression


# ─── 16d: includes ─────────────────────────────────────────────────────────

def test_quoted_include_resolves_via_suffix(parser):
    """`#include "foo.h"` finds `src/foo.h` in the repo set."""
    src = b"#include \"foo.h\"\n"
    batch = extract_cpp_file(
        REPO, "src/main.cpp", src, "sha", parser,
        repo_files={"src/main.cpp", "src/foo.h"},
    )
    targets = {imp.target_qn for imp in batch.imports}
    assert "src.foo" in targets
    edge = next(imp for imp in batch.imports if imp.target_qn == "src.foo")
    assert edge.kind == "module"
    assert edge.local_name == "*"


def test_angle_include_left_unresolved(parser):
    src = b"#include <iostream>\n"
    batch = extract_cpp_file(REPO, "x.cpp", src, "sha", parser)
    targets = {imp.target_qn for imp in batch.imports}
    assert "iostream" in targets


def test_resolve_quoted_include_suffix_match():
    rf = {"src/audio/engine.h", "src/main.cpp", "include/util.h"}
    assert _resolve_quoted_include("audio/engine.h", "src/main.cpp", rf) == "src/audio/engine.h"
    assert _resolve_quoted_include("util.h", "src/main.cpp", rf) == "include/util.h"
    assert _resolve_quoted_include("missing.h", "src/main.cpp", rf) is None


def test_resolve_quoted_include_prefers_same_dir():
    rf = {"a/foo.h", "b/foo.h"}
    assert _resolve_quoted_include("foo.h", "a/main.cpp", rf) == "a/foo.h"
    assert _resolve_quoted_include("foo.h", "b/main.cpp", rf) == "b/foo.h"


# ─── 16f: out-of-line method linking ───────────────────────────────────────

def test_oof_method_qn_matches_in_class(parser):
    """Header decl + cpp impl produce IDENTICAL FunctionNode qns so
    MERGE-on-qn collapses them in the loader."""
    src = (
        b"class Engine {\n"
        b"public:\n"
        b"    void play();\n"
        b"};\n"
        b"void Engine::play() { /* impl */ }\n"
    )
    batch = extract_cpp_file(REPO, "e.cpp", src, "sha", parser)
    play_fns = [fn for fn in batch.functions if fn.name == "play"]
    assert len(play_fns) == 2  # both header decl AND OOL definition emitted
    qns = {fn.qualified_name for fn in play_fns}
    assert qns == {"e.Engine.play"}, \
        f"expected identical qns for header+impl, got {qns}"
    parents = {fn.parent_class_qn for fn in play_fns}
    assert parents == {"e.Engine"}


def test_ofl_const_method_preserves_const_in_qn(parser):
    src = (
        b"class Iter {\npublic:\n    int begin() const;\n};\n"
        b"int Iter::begin() const { return 0; }\n"
    )
    batch = extract_cpp_file(REPO, "i.cpp", src, "sha", parser)
    begin_fns = [fn for fn in batch.functions if fn.name == "begin"]
    assert len(begin_fns) == 2
    qns = {fn.qualified_name for fn in begin_fns}
    assert qns == {"i.Iter.begin:const"}


# ─── modifiers (virtual / override / final / =delete) ──────────────────────

def test_virtual_method_flag(parser):
    src = (
        b"class Base { public: virtual void foo(); };\n"
    )
    batch = extract_cpp_file(REPO, "b.cpp", src, "sha", parser)
    foo = next(fn for fn in batch.functions if fn.name == "foo")
    assert foo.is_virtual is True


def test_override_method_flag(parser):
    src = (
        b"class Derived : public Base {\n"
        b"public:\n"
        b"    void foo() override;\n"
        b"};\n"
    )
    batch = extract_cpp_file(REPO, "d.cpp", src, "sha", parser)
    foo = next(fn for fn in batch.functions if fn.name == "foo")
    assert foo.is_virtual is True


def test_deleted_method_skipped(parser):
    src = (
        b"class NoCopy {\n"
        b"public:\n"
        b"    NoCopy() = default;\n"
        b"    NoCopy(const NoCopy&) = delete;\n"
        b"    void foo();\n"
        b"};\n"
    )
    batch = extract_cpp_file(REPO, "n.cpp", src, "sha", parser)
    names = {fn.name for fn in batch.functions}
    assert "foo" in names      # normal method present
    # = default and = delete suppressed (mirror GitNexus)
    assert names == {"foo"}


# ─── struct ────────────────────────────────────────────────────────────────

def test_struct_emitted_as_class(parser):
    src = b"struct Point { int x; int y; void translate(int dx); };\n"
    batch = extract_cpp_file(REPO, "p.cpp", src, "sha", parser)
    cls_qns = {c.qualified_name for c in batch.classes}
    assert "p.Point" in cls_qns
    fn_qns = {fn.qualified_name for fn in batch.functions}
    assert "p.Point.translate" in fn_qns


# ─── walker integration ───────────────────────────────────────────────────

def test_walker_routes_cpp_extensions():
    from app.repo_indexer.walker import EXT_LANGUAGE
    assert EXT_LANGUAGE[".cpp"] == "cpp"
    assert EXT_LANGUAGE[".cc"] == "cpp"
    assert EXT_LANGUAGE[".cxx"] == "cpp"
    assert EXT_LANGUAGE[".c"] == "cpp"
    assert EXT_LANGUAGE[".h"] == "cpp"
    assert EXT_LANGUAGE[".hpp"] == "cpp"
    assert EXT_LANGUAGE[".hxx"] == "cpp"
    assert EXT_LANGUAGE[".hh"] == "cpp"
