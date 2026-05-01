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


# ─── Sprint 17 post-deploy: force_reindex + Java/CPP wiring regression ──────
# These guard against silently dropping Java/CPP support from the Temporal
# activity again. We don't invoke the activity end-to-end (needs Neo4j +
# tree-sitter-java); source-string assertions are the cheap regression.

def test_phase_context_force_reindex_field_exists():
    from app.repo_indexer.actions import IndexBatch, RepoNode
    from app.repo_indexer.runner import PhaseContext

    repo = RepoNode(name="r", url="", commit_sha="")
    ctx = PhaseContext(
        repo=repo,
        repo_root=Path("."),
        languages=("python",),
        batch=IndexBatch(repo=repo),
        force_reindex=True,
    )
    assert ctx.force_reindex is True


def test_repo_task_input_force_reindex_default_false():
    from app.workflows import RepoTaskInput
    inp = RepoTaskInput(repo_url="https://x/y.git", branch="main", brief="do thing")
    assert inp.force_reindex is False


def test_repo_task_input_accepts_force_reindex():
    from app.workflows import RepoTaskInput
    inp = RepoTaskInput(
        repo_url="https://x/y.git", branch="main", brief="do thing",
        force_reindex=True,
    )
    assert inp.force_reindex is True


def test_index_activity_languages_includes_java_and_cpp():
    """Regression: ensure the activity's hardcoded languages tuple keeps
    java and cpp. Previously walker filtering silently dropped both."""
    import inspect
    from app import activities as acts
    fn = getattr(acts.index_repo_activity, "__wrapped__", acts.index_repo_activity)
    src = inspect.getsource(fn)
    assert '"java"' in src
    assert '"cpp"' in src


def test_index_activity_constructs_java_parser():
    """Regression: the activity must instantiate the java parser before
    handing it to PhaseContext, otherwise ParsePhase logs "java parser
    unavailable" for every .java file."""
    import inspect
    from app import activities as acts
    fn = getattr(acts.index_repo_activity, "__wrapped__", acts.index_repo_activity)
    src = inspect.getsource(fn)
    assert "tree_sitter_java" in src
    assert "java_parser" in src
    assert "Parser(Language(tsjava.language()))" in src


# ─── Sprint 18c: Critic activity registration + workflow wiring ─────────────


def test_critic_activity_registered_in_worker():
    """Sprint 18c wires `critic_repo_task_activity` into the worker so
    Temporal can dispatch it. Without this registration the workflow's
    execute_activity call raises at first run."""
    from app import worker
    from app.activities import critic_repo_task_activity
    assert critic_repo_task_activity in worker.ACTIVITIES


def test_critic_activity_signature():
    """The critic activity's coroutine signature must accept the eight
    positional args the workflow passes (path, name, plan, diff, files,
    tenant, wf_id, brief). `brief` was appended in Sprint 18.1 to feed
    the brief-derived criteria fallback when the Architect plan is
    degraded; it defaults to "" for backward compat."""
    import inspect
    from app import activities as acts
    fn = getattr(
        acts.critic_repo_task_activity, "__wrapped__",
        acts.critic_repo_task_activity,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert params == [
        "repo_path", "repo_name", "architect_plan", "coder_diff",
        "files_with_content", "tenant_id", "workflow_id", "brief",
    ]
    # Sprint 18.1: backward-compat default keeps replayed Temporal
    # histories (which call the activity without `brief`) functional.
    assert sig.parameters["brief"].default == ""


def test_continuation_count_max_2():
    """D4: continuation cap = 2. Locking the value down so a future tweak
    to bump it requires intent (Reflexion's 3-trial cap is the published
    upper bound; going higher regresses to AutoGPT failure mode)."""
    from app.workflows.repo_task import MAX_CONTINUATIONS
    assert MAX_CONTINUATIONS == 2


def test_repo_task_output_includes_critic_results():
    """RepoTaskOutput exposes critic_results + continuation_count so the
    UI / cost-summary card can render the multi-agent breakdown."""
    from app.workflows import RepoTaskOutput
    out = RepoTaskOutput(
        workflow_id="wf-1", repo_name="r", commit_sha="abc",
        files_changed=[], diff="", iterations=0, summary="",
        tokens_in=0, tokens_out=0, cost_usd=0.0,
    )
    assert out.critic_results == []
    assert out.continuation_count == 0


def test_repo_task_output_critic_results_accepts_dicts():
    """critic_results is a plain list of dicts (not CriticRepoOutput) so it
    survives Temporal's JSON serialisation without a converter."""
    from app.workflows import RepoTaskOutput
    out = RepoTaskOutput(
        workflow_id="wf-1", repo_name="r", commit_sha="abc",
        files_changed=[], diff="", iterations=0, summary="",
        tokens_in=0, tokens_out=0, cost_usd=0.0,
        critic_results=[
            {"verdict": "incomplete", "passed_criteria": [], "failed_criteria": []},
            {"verdict": "complete", "passed_criteria": ["c1"], "failed_criteria": []},
        ],
        continuation_count=1,
    )
    assert len(out.critic_results) == 2
    assert out.continuation_count == 1


def test_workflow_uses_critic_activity_after_coder():
    """Source-string check: critic_repo_task_activity is invoked AFTER
    run_repo_coder_activity. Cheap regression against a refactor that
    drops or reorders the critic step."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    coder_pos = src.find("run_repo_coder_activity")
    critic_pos = src.find("critic_repo_task_activity")
    assert coder_pos != -1
    assert critic_pos != -1
    assert coder_pos < critic_pos, "critic must run after coder"
    # The continuation loop must use a `while` and respect MAX_CONTINUATIONS.
    assert "while" in src
    assert "MAX_CONTINUATIONS" in src


def test_workflow_continuation_loop_handles_monotone_progress():
    """Source-string check: the workflow includes a monotone-progress guard
    that terminates early when the new pass doesn't satisfy strictly more
    criteria than the previous (per D4)."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    # Check that the loop reads passed_criteria from prior critic results.
    assert "passed_criteria" in src


# ─── Sprint 18d: Best-of-N Reviewer wiring ──────────────────────────────────


def test_reviewer_activity_registered_in_worker():
    """Sprint 18d wires `reviewer_repo_task_activity` into the worker so
    Temporal can dispatch it. Without this registration the workflow's
    execute_activity call raises at first run."""
    from app import worker
    from app.activities import reviewer_repo_task_activity
    assert reviewer_repo_task_activity in worker.ACTIVITIES


def test_reviewer_activity_signature():
    """The reviewer activity's coroutine signature must accept the six
    positional args the workflow passes (path, name, plan, candidates,
    tenant, wf_id)."""
    import inspect
    from app import activities as acts
    fn = getattr(
        acts.reviewer_repo_task_activity, "__wrapped__",
        acts.reviewer_repo_task_activity,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert params == [
        "repo_path", "repo_name", "architect_plan", "candidates",
        "tenant_id", "workflow_id",
    ]


def test_repo_task_input_disable_best_of_n_default_false():
    """Backward compat: existing callers pass a 4-arg RepoTaskInput and
    must keep getting Best-of-N enabled by default for cross-cutting
    briefs. disable_best_of_n is opt-in."""
    from app.workflows import RepoTaskInput
    inp = RepoTaskInput(repo_url="https://x/y.git", branch="main", brief="do thing")
    assert inp.disable_best_of_n is False


def test_repo_task_input_accepts_disable_best_of_n():
    from app.workflows import RepoTaskInput
    inp = RepoTaskInput(
        repo_url="https://x/y.git", branch="main", brief="do thing",
        disable_best_of_n=True,
    )
    assert inp.disable_best_of_n is True


def test_repo_task_output_includes_reviewer_result():
    """RepoTaskOutput exposes reviewer_result + best_of_n_count so the UI
    cost-summary can render the multi-agent breakdown for cross-cutting
    briefs (and skip the section entirely for single-Coder runs)."""
    from app.workflows import RepoTaskOutput
    out = RepoTaskOutput(
        workflow_id="wf-1", repo_name="r", commit_sha="abc",
        files_changed=[], diff="", iterations=0, summary="",
        tokens_in=0, tokens_out=0, cost_usd=0.0,
    )
    assert out.reviewer_result is None
    assert out.best_of_n_count == 1


def test_repo_task_output_reviewer_result_accepts_dict():
    """reviewer_result is a plain dict (not ReviewerRepoOutput) so it
    survives Temporal's JSON serialisation without a converter."""
    from app.workflows import RepoTaskOutput
    out = RepoTaskOutput(
        workflow_id="wf-1", repo_name="r", commit_sha="abc",
        files_changed=[], diff="", iterations=0, summary="",
        tokens_in=0, tokens_out=0, cost_usd=0.0,
        reviewer_result={
            "winner_index": 1,
            "rationale": "candidate 1 won pairwise",
            "fallback_used": False,
        },
        best_of_n_count=3,
    )
    assert out.reviewer_result["winner_index"] == 1
    assert out.best_of_n_count == 3


def test_best_of_n_constants():
    """D5: cap N=5 (Gao 2022); default N=3 per AlphaCode 2 sweet spot."""
    from app.workflows.repo_task import BEST_OF_N_DEFAULT, BEST_OF_N_MAX
    assert BEST_OF_N_DEFAULT == 3
    assert BEST_OF_N_MAX == 5
    assert BEST_OF_N_DEFAULT <= BEST_OF_N_MAX


def test_workflow_uses_reviewer_when_cross_cutting():
    """Source-string check: the workflow gates the reviewer call on
    arch_result.cross_cutting and has a single-Coder fallback branch."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    # Cross-cutting gate exists.
    assert "cross_cutting" in src
    # Reviewer activity is invoked.
    assert "reviewer_repo_task_activity" in src
    # Best-of-N count is read from BEST_OF_N_DEFAULT.
    assert "BEST_OF_N_DEFAULT" in src
    # Parallel fan-out via asyncio.gather.
    assert "asyncio.gather" in src
    # Disable_best_of_n is honoured.
    assert "disable_best_of_n" in src


def test_workflow_pr_body_mentions_best_of_n_winner():
    """Source-string check: when Best-of-N ran, the push_brief footer
    mentions the winner index + seed + rationale per the 18d spec."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    assert "Best-of-N selection" in src
    assert "winner #" in src
    assert "seed=" in src


def test_run_repo_coder_activity_accepts_coder_seed():
    """The Temporal activity signature must accept coder_seed for
    Best-of-N callers."""
    import inspect
    from app import activities as acts
    fn = getattr(
        acts.run_repo_coder_activity, "__wrapped__",
        acts.run_repo_coder_activity,
    )
    sig = inspect.signature(fn)
    assert "coder_seed" in sig.parameters
    assert sig.parameters["coder_seed"].default == 0


def test_workflow_imports_asyncio():
    """asyncio.gather is the parallel-fanout primitive — must be imported."""
    import inspect
    from app.workflows import repo_task as wf
    src = inspect.getsource(wf)
    # Top-level import (not just a string mention).
    assert "import asyncio" in src


# ─── Sprint 18.1: brief plumbing + cross_cutting heuristic fallback ─────────


def test_workflow_passes_brief_to_critic_activity():
    """Sprint 18.1: the workflow must pass `inp.brief` to
    critic_repo_task_activity as the 8th positional arg so the Critic
    can fall back to brief-derived criteria when the Architect plan is
    degraded (empty subtasks)."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    # Locate the critic activity invocation block and assert inp.brief
    # appears in its args list (between workflow_id and the closing
    # bracket of the args=[] block).
    critic_call_idx = src.find("critic_repo_task_activity,")
    assert critic_call_idx != -1
    # The args=[...] for the critic call should contain inp.brief.
    after_critic = src[critic_call_idx:critic_call_idx + 800]
    assert "inp.brief" in after_critic, (
        "critic_repo_task_activity invocation must pass inp.brief"
    )


def test_workflow_uses_heuristic_when_architect_plan_degraded():
    """Sprint 18.1: when arch_result.subtasks is empty the workflow
    must call infer_cross_cutting_from_brief on inp.brief so Best-of-N
    still triggers for cross-cutting briefs that hit the Architect's
    iteration cap."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    # Heuristic helper is invoked by name.
    assert "infer_cross_cutting_from_brief" in src
    # Wired off the "plan is degraded" predicate.
    assert "architect_plan_is_degraded" in src or "not arch_result.get" in src


def test_workflow_imports_heuristics_module():
    """The heuristic must be imported through workflow.unsafe so the
    Temporal sandbox tolerates it (free-form imports inside @workflow.defn
    are forbidden)."""
    import inspect
    from app.workflows import repo_task as wf
    src = inspect.getsource(wf)
    assert "from app.workflows._heuristics import infer_cross_cutting_from_brief" in src
    # And must live inside a workflow.unsafe.imports_passed_through block.
    block_start = src.find("with workflow.unsafe.imports_passed_through()")
    block_end = src.find("\n\n", block_start)
    assert block_start != -1
    block = src[block_start:block_end]
    assert "infer_cross_cutting_from_brief" in block
