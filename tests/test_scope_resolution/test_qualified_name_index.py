"""Qualified-name index — lookup, missing, collision."""
from __future__ import annotations

import pytest

from app.repo_indexer.scope_resolution import (
    Declaration,
    Range,
    ScopeTreeInvariantError,
    build_qualified_name_index,
)


def decl(qn: str, name: str, file: str = "a.py") -> Declaration:
    return Declaration(
        qualified_name=qn,
        name=name,
        kind="function",
        file_path=file,
        range=Range(file_path=file, start_byte=0, end_byte=10),
        scope_id=None,
    )


class TestLookup:
    def test_lookup_returns_declaration(self):
        d = decl("app.foo.bar", "bar")
        idx = build_qualified_name_index([d])
        assert idx.lookup("app.foo.bar") == d

    def test_missing_returns_none(self):
        idx = build_qualified_name_index([decl("app.foo", "foo")])
        assert idx.lookup("app.nonexistent") is None

    def test_membership(self):
        idx = build_qualified_name_index([decl("a.b", "b")])
        assert "a.b" in idx
        assert "a.c" not in idx

    def test_len(self):
        idx = build_qualified_name_index([
            decl("a.x", "x"),
            decl("a.y", "y"),
            decl("a.z", "z"),
        ])
        assert len(idx) == 3


class TestCollision:
    def test_collision_raises_with_both_paths(self):
        d1 = decl("app.foo.bar", "bar", file="src/a.py")
        d2 = decl("app.foo.bar", "bar", file="src/b.py")
        with pytest.raises(ScopeTreeInvariantError) as exc:
            build_qualified_name_index([d1, d2])
        msg = str(exc.value)
        assert "app.foo.bar" in msg
        assert "src/a.py" in msg
        assert "src/b.py" in msg

    def test_same_file_collision_also_raises(self):
        # Two declarations of the same QN in the same file (would be a
        # syntax error in real Python but might appear in TS namespaces
        # if extraction is buggy).
        d1 = decl("app.foo.bar", "bar", file="a.py")
        d2 = decl("app.foo.bar", "bar", file="a.py")
        with pytest.raises(ScopeTreeInvariantError):
            build_qualified_name_index([d1, d2])
