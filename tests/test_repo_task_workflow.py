"""RepoTaskWorkflow activities — Sprint 10e.

Real subprocess for clone (against a tiny in-memory git repo), real
indexer for the index activity. The Coder activity needs Anthropic +
Neo4j + tree-sitter and is exercised end-to-end in deploy, not in unit
tests — we only check that the activity is wired into the worker.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from app.agents.coder_sandbox import Sandbox, SandboxError


# ─── Sandbox.wrap ───────────────────────────────────────────────────────────

def test_sandbox_wrap_uses_existing_dir(tmp_path):
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    sb = Sandbox.wrap(tmp_path)
    assert sb.root == tmp_path.resolve()
    text, truncated = sb.read("hello.txt")
    assert text == "hi"
    assert truncated is False


def test_sandbox_wrap_rejects_missing_dir(tmp_path):
    with pytest.raises(SandboxError, match="not a directory"):
        Sandbox.wrap(tmp_path / "nope")


# ─── clone_repo_activity ────────────────────────────────────────────────────
# Build a real tiny git repo on disk, then point the activity at it via
# `file://` URL — works with git's standard transport and avoids needing
# any network or auth in CI.

def _make_local_repo(path: Path) -> str:
    """Init a git repo with one commit; return absolute file:// URL."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    # file:// URL format works on every platform git supports.
    return path.resolve().as_uri()


def _clone_fn():
    """Resolve the underlying coroutine inside @activity.defn — temporalio
    keeps it on `.__wrapped__`, but fall back to the wrapper itself."""
    from app import activities as acts
    return getattr(acts.clone_repo_activity, "__wrapped__", None) or acts.clone_repo_activity


@pytest.mark.asyncio
async def test_clone_repo_activity_clones_and_returns_sha(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    import uuid

    src = tmp_path / "src-repo"
    repo_url = _make_local_repo(src)

    # Unique per-run workflow_id avoids cross-run state on the hardcoded
    # /tmp/repo-tasks/ prefix. The activity rmtree's any leftover before
    # cloning, but Windows file locks can defeat that — uuid is the belt+
    # braces fix.
    workflow_id = f"test-clone-{uuid.uuid4().hex[:8]}"
    expected_dest = Path("/tmp/repo-tasks") / workflow_id
    from temporalio import activity as ta
    monkeypatch.setattr(ta, "heartbeat", lambda *a, **kw: None)
    try:
        result = await _clone_fn()(repo_url, "main", workflow_id)
        assert "path" in result and "commit_sha" in result
        assert Path(result["path"]).is_dir()
        assert (Path(result["path"]) / "README.md").is_file()
        assert len(result["commit_sha"]) == 40
    finally:
        if expected_dest.exists():
            shutil.rmtree(expected_dest, ignore_errors=True)


@pytest.mark.asyncio
async def test_clone_repo_activity_fails_on_bad_url(monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    import uuid

    from temporalio import activity as ta
    monkeypatch.setattr(ta, "heartbeat", lambda *a, **kw: None)
    workflow_id = f"test-clone-bad-{uuid.uuid4().hex[:8]}"
    expected_dest = Path("/tmp/repo-tasks") / workflow_id
    try:
        with pytest.raises(RuntimeError, match="git clone failed"):
            await _clone_fn()("file:///nonexistent/path/to/nowhere.git", "main", workflow_id)
    finally:
        if expected_dest.exists():
            shutil.rmtree(expected_dest, ignore_errors=True)


# ─── Workflow + activity registration ───────────────────────────────────────
# These don't run the workflow; they just confirm the wiring is intact so
# a refactor can't silently drop a workflow / activity from the worker.

def test_repo_task_workflow_registered_in_worker():
    from app import worker
    from app.workflows import RepoTaskWorkflow
    assert RepoTaskWorkflow in worker.WORKFLOWS


def test_repo_task_activities_registered_in_worker():
    from app import worker
    from app.activities import (
        clone_repo_activity, index_repo_activity, run_repo_coder_activity,
    )
    assert clone_repo_activity in worker.ACTIVITIES
    assert index_repo_activity in worker.ACTIVITIES
    assert run_repo_coder_activity in worker.ACTIVITIES


def test_repo_task_input_dataclass_shape():
    from app.workflows import RepoTaskInput
    inp = RepoTaskInput(repo_url="https://x/y.git", branch="main", brief="do thing")
    assert inp.tenant_id == "default"
    assert inp.repo_name == ""


# ─── _capture_diff: untracked-file regression ───────────────────────────────
# A real bug we shipped: the Coder created a brand-new file (e.g. RateLimitFilter.java)
# alongside an edit to an existing file, but `git diff HEAD` only reports the
# tracked edit — untracked paths are invisible — so the PR pushed only the
# wiring change and CI broke on a missing import. The fix promotes untracked
# files to intent-to-add before diffing.

def test_capture_diff_includes_untracked_new_files(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    from app.agents.coder_repo import _capture_diff

    repo = tmp_path / "repo"
    _make_local_repo(repo)

    # Modify a tracked file and create an untracked one — exactly the shape
    # of a "wire up + add new class" Coder output.
    (repo / "README.md").write_text("hi\nedited\n", encoding="utf-8")
    (repo / "NEW_FILE.txt").write_text("brand new\n", encoding="utf-8")

    diff, files = _capture_diff(repo)
    assert "README.md" in files
    assert "NEW_FILE.txt" in files
    assert "brand new" in diff  # the new file's content shows up as +lines


def test_capture_diff_clean_repo_returns_empty(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    from app.agents.coder_repo import _capture_diff

    repo = tmp_path / "repo"
    _make_local_repo(repo)

    diff, files = _capture_diff(repo)
    assert diff == ""
    assert files == []
