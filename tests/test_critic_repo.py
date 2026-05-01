"""Tests for the repo Critic — Sprint 18c.

Covers the dataclasses, the deterministic-gate routing/parsing, the
LLM-judge scaffolding (with the actual API call mocked out), and the
continuation handoff doc shape. The end-to-end loop (Critic → continuation
Coder → Critic) needs Anthropic + Neo4j + Temporal and is exercised in
deploy; here we guard the pure-Python contracts.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from app.agents.critic_repo import (
    CRITIC_MODEL,
    CriticFailure,
    CriticRepoOutput,
    GateFailure,
    MAX_TOKENS_JUDGE,
    MVN_TIMEOUT_SECONDS,
    NPM_TIMEOUT_SECONDS,
    RUFF_TIMEOUT_SECONDS,
    _flatten_acceptance_criteria,
    _group_files_by_language,
    _parse_judge_response,
    build_continuation_handoff_doc,
    critic_output_to_dict,
    run_critic_repo,
    run_deterministic_gate,
)


# ─── Dataclass roundtrip ────────────────────────────────────────────────────
# Same Temporal-serialisation invariant as the Architect: every activity
# return must round-trip through json.dumps. Test instantiates each
# dataclass with non-default values to flush out any non-primitive field.


def test_critic_output_dataclass_roundtrip():
    """Instantiate → asdict → json.dumps → loads → reconstruct → equal."""
    sample = CriticRepoOutput(
        verdict="incomplete",
        passed_criteria=["Endpoint returns 200 on valid input"],
        failed_criteria=[
            CriticFailure(
                criterion="Endpoint returns 401 on expired token",
                evidence="No 401 path in handler",
                severity="block",
            ),
            CriticFailure(
                criterion="Audit log entry created on refresh",
                evidence="Missing call to audit.log",
                severity="warn",
            ),
        ],
        deterministic_gate_passed=False,
        gate_failures=[
            GateFailure(
                tool="ruff", file="app/auth/routes.py", line=42,
                message="F401: imported but unused",
            ),
        ],
        continuation_prompt="## Current state summary\n...",
        _model="claude-sonnet-4-6",
        _provider="anthropic",
        _tokens_in=1234,
        _tokens_out=567,
        _cost_usd=0.012,
    )
    raw = asdict(sample)
    encoded = json.dumps(raw)
    decoded = json.loads(encoded)

    rebuilt = CriticRepoOutput(
        verdict=decoded["verdict"],
        passed_criteria=decoded["passed_criteria"],
        failed_criteria=[CriticFailure(**cf) for cf in decoded["failed_criteria"]],
        deterministic_gate_passed=decoded["deterministic_gate_passed"],
        gate_failures=[GateFailure(**gf) for gf in decoded["gate_failures"]],
        continuation_prompt=decoded["continuation_prompt"],
        _model=decoded["_model"],
        _provider=decoded["_provider"],
        _tokens_in=decoded["_tokens_in"],
        _tokens_out=decoded["_tokens_out"],
        _cost_usd=decoded["_cost_usd"],
    )
    assert rebuilt == sample


def test_critic_failure_severity_default_block():
    """Continuation fires only on `block` failures — default must be block."""
    cf = CriticFailure(criterion="x", evidence="y")
    assert cf.severity == "block"


def test_gate_failure_line_optional():
    """Some diagnostics don't have a line (file-level mvn errors)."""
    gf = GateFailure(tool="mvn", file="pom.xml", line=None, message="bad config")
    assert gf.line is None


def test_critic_output_to_dict_helper():
    out = CriticRepoOutput(verdict="complete")
    d = critic_output_to_dict(out)
    assert isinstance(d, dict)
    assert d["verdict"] == "complete"
    assert d["passed_criteria"] == []
    # Round-trip via JSON.
    assert json.loads(json.dumps(d))["verdict"] == "complete"


# ─── _group_files_by_language ───────────────────────────────────────────────


def test_group_files_by_language_buckets_correctly():
    files = [
        "app/foo.py", "tests/test_foo.py",
        "src/Main.java", "frontend/App.tsx", "frontend/util.ts",
        "client/index.js", "client/comp.jsx",
        "lib/x.cpp", "lib/x.h",
        "README.md", "config.toml",
    ]
    g = _group_files_by_language(files)
    assert sorted(g["python"]) == ["app/foo.py", "tests/test_foo.py"]
    assert g["java"] == ["src/Main.java"]
    assert sorted(g["ts"]) == ["client/comp.jsx", "client/index.js", "frontend/App.tsx", "frontend/util.ts"]
    assert sorted(g["cpp"]) == ["lib/x.cpp", "lib/x.h"]
    assert sorted(g["other"]) == ["README.md", "config.toml"]


def test_group_files_by_language_empty():
    g = _group_files_by_language([])
    assert g == {"python": [], "java": [], "ts": [], "cpp": [], "other": []}


# ─── Deterministic gate: ruff ───────────────────────────────────────────────
# We write a real .py file with a known violation and run ruff against it.
# Skips if ruff isn't installed (CI without the dev extras). The ruff
# binary lives in the venv; module invocation also works.


def _has_ruff() -> bool:
    if shutil.which("ruff") is not None:
        return True
    try:
        import ruff  # noqa: F401
        return True
    except ImportError:
        return False


def test_deterministic_gate_ruff_failure_synthetic(tmp_path):
    """A Python file with an unused import should produce a ruff failure."""
    if not _has_ruff():
        pytest.skip("ruff not available")
    # F401 (unused import) is always-on in ruff's default ruleset.
    bad = tmp_path / "bad.py"
    bad.write_text("import os\nimport sys\n\nprint(sys.version)\n", encoding="utf-8")
    passed, failures = run_deterministic_gate(tmp_path, ["bad.py"])
    assert passed is False
    # At least one ruff failure for `bad.py`. Other rule codes may also fire
    # depending on the ruff version; just assert that ruff reported something.
    ruff_failures = [f for f in failures if f.tool == "ruff"]
    assert ruff_failures, f"expected ruff failures, got {failures}"
    # The path should be rendered repo-relative.
    assert all(not Path(f.file).is_absolute() for f in ruff_failures)


def test_deterministic_gate_clean_python_passes(tmp_path):
    """A clean Python file should pass ruff + compileall with zero failures."""
    if not _has_ruff():
        pytest.skip("ruff not available")
    good = tmp_path / "good.py"
    good.write_text('"""Valid."""\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n', encoding="utf-8")
    passed, failures = run_deterministic_gate(tmp_path, ["good.py"])
    assert passed is True
    assert failures == []


def test_deterministic_gate_compileall_catches_syntax_error(tmp_path):
    """A .py file with a SyntaxError should at minimum trip compileall.

    compileall is the safety net under ruff — even if the user's ruff
    config silences everything, compileall's SyntaxError still fires.
    """
    bad = tmp_path / "syntax_error.py"
    bad.write_text("def broken(\n    pass\n", encoding="utf-8")
    passed, failures = run_deterministic_gate(tmp_path, ["syntax_error.py"])
    assert passed is False
    # We don't pin the exact tool because ruff catches syntax errors too;
    # we just want SOMETHING to flag the file.
    assert any(
        "syntax_error.py" in f.file for f in failures
    ), f"expected failures referencing syntax_error.py, got {failures}"


def test_deterministic_gate_skips_when_tool_missing(tmp_path, monkeypatch):
    """If ruff isn't on PATH and not importable, gate skips silently.

    Patching shutil.which + importlib.util.find_spec to simulate a
    completely tool-less environment. The compileall gate still runs
    (it's stdlib), so a clean .py file should still pass.
    """
    good = tmp_path / "good.py"
    good.write_text('"""Valid."""\n', encoding="utf-8")

    # Force ruff lookups to fail.
    real_which = shutil.which
    def fake_which(name, *args, **kwargs):
        if name == "ruff":
            return None
        return real_which(name, *args, **kwargs)
    monkeypatch.setattr("app.agents.critic_repo.shutil.which", fake_which)

    # Also force the importable check to say "no ruff".
    def fake_has_module(name):
        return False
    monkeypatch.setattr("app.agents.critic_repo._has_module", fake_has_module)

    passed, failures = run_deterministic_gate(tmp_path, ["good.py"])
    # Clean file → compileall passes too → overall pass with no failures.
    assert passed is True
    assert failures == []


def test_deterministic_gate_groups_by_language(tmp_path, monkeypatch):
    """Mixed .py + .java input: only the relevant gates fire.

    Mocks subprocess so we don't actually invoke mvn (which may not be
    on the worker). We just assert that the python gate ran AND a mvn
    invocation was attempted (mocked).
    """
    if not _has_ruff():
        pytest.skip("ruff not available")
    good_py = tmp_path / "good.py"
    good_py.write_text('"""ok."""\n', encoding="utf-8")
    java = tmp_path / "Main.java"
    java.write_text("public class Main {}\n", encoding="utf-8")
    # Make mvn appear available so the mvn gate is considered.
    monkeypatch.setattr("app.agents.critic_repo.shutil.which", lambda n: "/fake/" + n)
    # Need a pom.xml for the mvn gate to actually run.
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # Return a successful CompletedProcess regardless.
        import subprocess
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.agents.critic_repo.subprocess.run", fake_run)

    passed, failures = run_deterministic_gate(tmp_path, ["good.py", "Main.java"])
    assert passed is True
    # We expect ruff + compileall + mvn to have been invoked (3 subprocess calls).
    # ruff invocation may go via "<python> -m ruff" or "ruff"; either way
    # the cmd line will include "ruff".
    assert any("ruff" in str(c) for c in calls), f"ruff not invoked: {calls}"
    assert any("compileall" in str(c) for c in calls), f"compileall not invoked: {calls}"
    assert any("mvn" in str(c) for c in calls), f"mvn not invoked: {calls}"
    # cpp / other groups are skipped — no `gcc` / `cc` calls.
    assert not any("gcc" in str(c) for c in calls)


def test_deterministic_gate_empty_input_passes():
    """No files changed → no gates needed → passes vacuously."""
    passed, failures = run_deterministic_gate(Path("/tmp"), [])
    assert passed is True
    assert failures == []


# ─── _flatten_acceptance_criteria ───────────────────────────────────────────


def test_flatten_acceptance_criteria_handles_missing_subtasks():
    assert _flatten_acceptance_criteria(None) == []
    assert _flatten_acceptance_criteria({}) == []
    assert _flatten_acceptance_criteria({"subtasks": []}) == []


def test_flatten_acceptance_criteria_assigns_indices():
    plan = {
        "subtasks": [
            {"id": "a", "acceptance_criteria": ["c1", "c2"]},
            {"id": "b", "acceptance_criteria": ["c3"]},
        ],
    }
    flat = _flatten_acceptance_criteria(plan)
    assert len(flat) == 3
    assert [f[1] for f in flat] == ["c1", "c2", "c3"]
    assert [f[2] for f in flat] == ["0", "1", "2"]


# ─── _parse_judge_response ──────────────────────────────────────────────────


def test_parse_judge_response_strips_markdown_fences():
    text = '```json\n{"0": {"status": "yes", "evidence": "ok"}}\n```'
    parsed = _parse_judge_response(text, [("a", "c1", "0")])
    assert parsed == {"0": {"status": "yes", "evidence": "ok"}}


def test_parse_judge_response_extracts_first_json_block():
    text = 'I think this:\n{"0": {"status": "no", "evidence": "missing"}}\nthat is all.'
    parsed = _parse_judge_response(text, [("a", "c1", "0")])
    assert parsed["0"]["status"] == "no"


def test_parse_judge_response_returns_empty_on_garbage():
    parsed = _parse_judge_response("not json at all", [])
    assert parsed == {}


# ─── build_continuation_handoff_doc ─────────────────────────────────────────


def test_continuation_handoff_doc_format():
    """The handoff doc must contain the four required structural sections.

    Per D7: structured handoff, not chat transcript. The Coder relies on
    finding "## Immediate next steps" to know what to do — locking the
    section names down here so a refactor doesn't drift them.
    """
    plan = {"subtasks": [{"id": "a", "acceptance_criteria": ["c1", "c2"]}]}
    doc = build_continuation_handoff_doc(
        architect_plan=plan,
        prior_diff="diff --git a/foo b/foo\n+x = 1\n",
        prior_files_changed=["foo.py"],
        passed_criteria=["c1"],
        failed_criteria=[
            CriticFailure(criterion="c2", evidence="not implemented"),
        ],
        gate_failures=[
            GateFailure(tool="ruff", file="foo.py", line=1, message="E501"),
        ],
    )
    # Required sections per D7.
    assert "## Current state summary" in doc
    assert "## Acceptance criteria status" in doc
    assert "## Immediate next steps" in doc
    assert "## Open questions" in doc
    # Constraint section keeps the Coder from rewriting the prior diff.
    assert "## Constraint" in doc
    # Body content sanity checks.
    assert "foo.py" in doc
    assert "c1" in doc and "c2" in doc
    assert "ruff" in doc
    # The prior pass's passed criterion shows up under "Already satisfied".
    assert "Already satisfied" in doc
    # The next-steps list is numbered — first failed criterion should be #1.
    assert "1. Address acceptance criterion: c2" in doc


def test_continuation_handoff_doc_handles_empty_inputs():
    """Even with everything empty, the doc is well-formed (sections present)."""
    doc = build_continuation_handoff_doc(
        architect_plan={"subtasks": []},
        prior_diff="",
        prior_files_changed=[],
        passed_criteria=[],
        failed_criteria=[],
        gate_failures=[],
    )
    assert "## Current state summary" in doc
    assert "## Acceptance criteria status" in doc
    assert "## Immediate next steps" in doc
    assert "## Open questions" in doc
    assert "## Constraint" in doc


def test_continuation_handoff_doc_truncates_long_diff():
    """Handoff doc mentions a 10-line diff cap so the next Coder isn't
    flooded with the previous pass's full diff (it's already on disk)."""
    long_diff = "\n".join(f"+ line {i}" for i in range(50))
    doc = build_continuation_handoff_doc(
        architect_plan={"subtasks": []},
        prior_diff=long_diff,
        prior_files_changed=["x.py"],
        passed_criteria=[], failed_criteria=[], gate_failures=[],
    )
    assert "more lines" in doc
    # Only the first 10 lines should appear.
    assert "+ line 9" in doc
    assert "+ line 25" not in doc


# ─── run_critic_repo coordinator ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_complete_when_all_criteria_pass(tmp_path, monkeypatch):
    """Judge says yes/yes → verdict=complete, no continuation prompt."""
    plan = {
        "subtasks": [{"id": "a", "acceptance_criteria": ["c1", "c2"]}],
    }

    async def fake_judge(architect_plan, coder_diff, files_with_content, gate_failures):
        return (["c1", "c2"], [], 100, 50)

    monkeypatch.setattr(
        "app.agents.critic_repo.run_llm_checklist_judge", fake_judge,
    )
    # No files changed → gate is vacuously passing.
    out = await run_critic_repo(
        architect_plan=plan,
        coder_diff="diff",
        files_with_content=[],
        repo_root=tmp_path,
    )
    assert out.verdict == "complete"
    assert out.continuation_prompt is None
    assert sorted(out.passed_criteria) == ["c1", "c2"]
    assert out.failed_criteria == []


@pytest.mark.asyncio
async def test_critic_incomplete_with_continuation_prompt(tmp_path, monkeypatch):
    """Judge says yes/no → verdict=incomplete + non-empty continuation_prompt."""
    plan = {
        "subtasks": [{"id": "a", "acceptance_criteria": ["c1", "c2"]}],
    }

    async def fake_judge(architect_plan, coder_diff, files_with_content, gate_failures):
        passed = ["c1"]
        failed = [CriticFailure(criterion="c2", evidence="not done", severity="block")]
        return (passed, failed, 200, 80)

    monkeypatch.setattr(
        "app.agents.critic_repo.run_llm_checklist_judge", fake_judge,
    )
    out = await run_critic_repo(
        architect_plan=plan,
        coder_diff="diff",
        files_with_content=[],
        repo_root=tmp_path,
    )
    assert out.verdict == "incomplete"
    assert out.continuation_prompt is not None
    assert "## Current state summary" in out.continuation_prompt
    assert len(out.passed_criteria) == 1
    assert len(out.failed_criteria) == 1
    assert out.failed_criteria[0].criterion == "c2"


@pytest.mark.asyncio
async def test_critic_handles_missing_architect_plan(tmp_path):
    """No plan → skip the LLM judge, return verdict=complete (don't break workflow).

    Per the docstring contract: when the Architect failed and shipped a
    degraded output, the Critic must NOT block — there's no checklist to
    grade against. The workflow proceeds with the Coder's diff as-is.
    """
    out = await run_critic_repo(
        architect_plan=None,
        coder_diff="diff",
        files_with_content=[],
        repo_root=tmp_path,
    )
    assert out.verdict == "complete"
    assert out.continuation_prompt is None
    assert out.passed_criteria == []
    assert out.failed_criteria == []


@pytest.mark.asyncio
async def test_critic_treats_empty_subtasks_as_no_plan(tmp_path):
    """Plan with no subtasks → skip judge, same as missing plan."""
    out = await run_critic_repo(
        architect_plan={"narrative": "x", "subtasks": []},
        coder_diff="diff",
        files_with_content=[],
        repo_root=tmp_path,
    )
    assert out.verdict == "complete"


@pytest.mark.asyncio
async def test_critic_incomplete_when_gate_fails_even_if_judge_passes(tmp_path, monkeypatch):
    """Deterministic-gate failure alone is enough to flip verdict.

    Per D3: gates are weighted heaviest. A diff that compiles cleanly
    but fails ruff is still "incomplete" — Critic surfaces the fix.
    """
    plan = {"subtasks": [{"id": "a", "acceptance_criteria": ["c1"]}]}

    async def fake_judge(architect_plan, coder_diff, files_with_content, gate_failures):
        return (["c1"], [], 50, 25)

    def fake_gate(repo_root, files_changed):
        return (False, [GateFailure(tool="ruff", file="x.py", line=1, message="E501")])

    monkeypatch.setattr("app.agents.critic_repo.run_llm_checklist_judge", fake_judge)
    monkeypatch.setattr("app.agents.critic_repo.run_deterministic_gate", fake_gate)

    out = await run_critic_repo(
        architect_plan=plan,
        coder_diff="diff",
        files_with_content=[{"path": "x.py", "content": "x"}],
        repo_root=tmp_path,
    )
    assert out.verdict == "incomplete"
    assert out.deterministic_gate_passed is False
    assert out.continuation_prompt is not None


# ─── Module-level constants ─────────────────────────────────────────────────


def test_critic_model_is_sonnet():
    """Per D8: judging the Coder uses a different model — Sonnet, not Haiku."""
    assert CRITIC_MODEL == "claude-sonnet-4-6"


def test_max_tokens_judge_modest():
    """Single batched judge call — 4K is plenty for ~50 criteria."""
    assert MAX_TOKENS_JUDGE == 4096


def test_mvn_timeout_is_capped():
    """Maven is the slowest gate; cap at 5 minutes so a stuck `mvn compile`
    can't blow past the activity's schedule_to_close_timeout."""
    assert MVN_TIMEOUT_SECONDS == 300


def test_npm_timeout_is_capped():
    assert NPM_TIMEOUT_SECONDS == 180


def test_ruff_timeout_is_modest():
    assert RUFF_TIMEOUT_SECONDS == 60
