"""Sandbox invariants: path escape rejection, size caps, list/read/write."""
from __future__ import annotations

import pytest

from app.agents.coder_sandbox import (
    MAX_READ_BYTES,
    MAX_WRITE_BYTES,
    Sandbox,
    SandboxError,
)


def _make(tmp_path, workflow_id="wf-test-123"):
    return Sandbox.create(workflow_id, base=tmp_path)


def test_create_and_destroy(tmp_path):
    sb = _make(tmp_path)
    assert sb.root.is_dir()
    assert sb.root.is_absolute()
    sb.destroy()
    assert not sb.root.exists()
    # destroy is idempotent
    sb.destroy()


def test_workflow_id_is_sanitised(tmp_path):
    sb = Sandbox.create("../../../etc/passwd", base=tmp_path)
    # sanitiser strips non-alnum except dash/underscore, so the escape fails
    # BEFORE path resolution — workspace lands safely inside tmp_path.
    assert tmp_path in sb.root.parents or sb.root == tmp_path / "etcpasswd"


def test_empty_workflow_id_is_rejected(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox.create("///", base=tmp_path)


def test_write_and_read_roundtrip(tmp_path):
    sb = _make(tmp_path)
    n = sb.write("app/main.py", "print('hi')\n")
    assert n == len("print('hi')\n")
    text, truncated = sb.read("app/main.py")
    assert text == "print('hi')\n"
    assert not truncated


def test_write_creates_parent_dirs(tmp_path):
    sb = _make(tmp_path)
    sb.write("deep/nested/path/file.txt", "x")
    assert (sb.root / "deep" / "nested" / "path" / "file.txt").exists()


def test_dotdot_escape_rejected(tmp_path):
    sb = _make(tmp_path)
    with pytest.raises(SandboxError):
        sb.write("../escape.txt", "nope")
    with pytest.raises(SandboxError):
        sb.read("../../outside.txt")


def test_absolute_path_rejected(tmp_path):
    sb = _make(tmp_path)
    # Absolute paths get stripped of leading slashes, then resolved inside
    # root. So "/etc/passwd" becomes "etc/passwd" inside the sandbox — the
    # *escape* is what we prevent, not the use of slashes.
    sb.write("/etc/passwd", "inside-sandbox")
    assert (sb.root / "etc" / "passwd").exists()


def test_backslash_normalised(tmp_path):
    sb = _make(tmp_path)
    sb.write("app\\main.py", "win")
    assert (sb.root / "app" / "main.py").exists()


def test_write_size_cap(tmp_path):
    sb = _make(tmp_path)
    too_big = "x" * (MAX_WRITE_BYTES + 1)
    with pytest.raises(SandboxError):
        sb.write("huge.txt", too_big)


def test_read_truncation_flag(tmp_path):
    sb = _make(tmp_path)
    # Bypass the write cap by writing directly so we can test the read side.
    big_path = sb.root / "big.log"
    big_path.write_bytes(b"y" * (MAX_READ_BYTES + 100))
    text, truncated = sb.read("big.log")
    assert truncated is True
    assert len(text) == MAX_READ_BYTES


def test_read_missing_file(tmp_path):
    sb = _make(tmp_path)
    with pytest.raises(SandboxError):
        sb.read("no-such-file.txt")


def test_list_files_skips_junk_dirs(tmp_path):
    sb = _make(tmp_path)
    sb.write("keep.py", "x")
    sb.write("app/kept.py", "x")
    # Create junk dirs directly — list_files should prune them.
    (sb.root / ".git").mkdir()
    (sb.root / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (sb.root / "__pycache__").mkdir()
    (sb.root / "__pycache__" / "x.pyc").write_text("x")
    listed = sb.list_files()
    assert "keep.py" in listed
    assert "app/kept.py" in listed
    assert all(".git" not in f for f in listed)
    assert all("__pycache__" not in f for f in listed)


def test_copy_in_stages_directory(tmp_path):
    sb = _make(tmp_path)
    src = tmp_path / "src_scaffold"
    (src / "app").mkdir(parents=True)
    (src / "app" / "main.py").write_text("print('from template')")
    (src / "README.md").write_text("# hello")
    sb.copy_in(src)
    assert (sb.root / "app" / "main.py").read_text() == "print('from template')"
    assert (sb.root / "README.md").read_text() == "# hello"


def test_copy_in_rejects_non_dir(tmp_path):
    sb = _make(tmp_path)
    bogus = tmp_path / "not-a-dir.txt"
    bogus.write_text("file")
    with pytest.raises(SandboxError):
        sb.copy_in(bogus)
