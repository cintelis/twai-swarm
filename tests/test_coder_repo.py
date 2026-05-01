"""Tests for the repo-aware Coder helpers — Sprint 17 follow-up.

Focus is on the recon-block formatting and the user-message wiring.
The agentic loop itself is exercised through the mocked-runner test in
`test_coder_loop.py`; here we just guard the new prepended block.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.agents import coder_repo
from app.agents.coder_repo import (
    MAX_ITERATIONS,
    REPO_CODER_SYSTEM_PROMPT,
    _build_user_message,
    _format_recon_block,
)


def _module(label: str, size: int, samples: list[str]) -> SimpleNamespace:
    """Duck-typed stand-in for `repo_query.ModuleSummary`."""
    return SimpleNamespace(
        label=label,
        size=size,
        cohesion=0.42,
        sample_member_qns=tuple(samples),
    )


def _process(name: str, members: list[str]) -> SimpleNamespace:
    """Duck-typed stand-in for `repo_query.ProcessSummary`."""
    return SimpleNamespace(
        name=name,
        summary="",
        step_count=len(members),
        member_qns=tuple(members),
    )


def test_format_recon_block_with_modules_and_processes():
    modules = [
        _module("auth", 12, ["app.auth.login", "app.auth.token", "app.auth.user"]),
        _module("api", 8, ["app.api.routes", "app.api.deps"]),
    ]
    processes = [
        _process("login_flow", ["app.api.routes.login", "app.auth.login.authenticate", "app.auth.token.issue"]),
        _process("calc_request", ["app.api.routes.calc", "app.calc.engine.run"]),
    ]
    out = _format_recon_block(modules, processes)
    assert out.startswith("## Repo recon")
    # Module labels surface
    assert "`auth`" in out
    assert "`api`" in out
    assert "12 symbols" in out
    # Process names surface
    assert "`login_flow`" in out
    assert "`calc_request`" in out
    # The chain abbreviates correctly when >2 members
    assert "app.api.routes.login → … → app.auth.token.issue" in out
    # The hint footer is present
    assert "repo_find_modules" in out and "repo_find_callers" in out


def test_format_recon_block_empty_returns_empty_string():
    assert _format_recon_block([], []) == ""


def test_format_recon_block_caps_modules_at_15():
    modules = [_module(f"mod{i}", 5, [f"pkg.mod{i}.fn"]) for i in range(30)]
    out = _format_recon_block(modules, [])
    # Render exactly 15 module bullets (not the 30 we passed).
    bullet_lines = [
        ln for ln in out.splitlines()
        if ln.startswith("- `mod") and "symbols" in ln
    ]
    assert len(bullet_lines) == 15
    # Truncation marker for the remaining 15.
    assert "…and 15 more" in out


def test_build_user_message_includes_recon_when_provided():
    recon = "## Repo recon (auto-generated)\n\n### Modules (1)\n- `core` (3 symbols): a, b, c"
    msg = _build_user_message("Add rate limiting.", "myrepo", recon_block=recon)
    # Recon must come BEFORE the task brief so the model reads it as map-then-task.
    recon_pos = msg.find("## Repo recon")
    brief_pos = msg.find("## Task brief")
    assert recon_pos != -1
    assert brief_pos != -1
    assert recon_pos < brief_pos
    assert "Add rate limiting." in msg
    assert "myrepo" in msg


def test_build_user_message_omits_recon_when_empty():
    msg = _build_user_message("Add a thing.", "myrepo", recon_block="")
    assert "## Repo recon" not in msg
    # The downstream sections still render.
    assert "## Task brief" in msg
    assert "Add a thing." in msg
    assert "myrepo" in msg


def test_format_recon_block_processes_only():
    """Empty modules + non-empty processes still renders cleanly (no Modules header)."""
    processes = [_process("startup", ["main", "init"])]
    out = _format_recon_block([], processes)
    assert out.startswith("## Repo recon")
    assert "### Modules" not in out
    assert "### Top processes (1)" in out
    # Two-member chain uses the simple arrow form.
    assert "main → init" in out


# --- Sprint 18a: budget awareness + MAX_ITERATIONS bump ---------------------
#
# Mid-stream Coder-facing injection was investigated and PUNTED for 18a:
# the Anthropic SDK's `client.beta.messages.tool_runner(...)` manages its
# own message history end-to-end and exposes no documented per-turn hook
# for inserting a system-side note between iterations without forking the
# runner. Instead Sprint 18a:
#   (a) raises MAX_ITERATIONS 15 → 30,
#   (b) declares the budget in REPO_CODER_SYSTEM_PROMPT up front so the
#       Coder can pace itself, and
#   (c) emits a "completion mode" heartbeat at iteration ≥ 24 (80% of cap)
#       so operators have visibility — see run_agentic_repo_coder.
# When the SDK gains a per-turn injection hook, the heartbeat-only signal
# in (c) should be upgraded to an actual Coder-visible message.


def test_max_iterations_is_30():
    """Sprint 18a bumped the repo-Coder ceiling from 15 → 30."""
    assert MAX_ITERATIONS == 30
    # And the module-level constant matches what the function reads.
    assert coder_repo.MAX_ITERATIONS == 30


def test_system_prompt_mentions_budget():
    """The repo-Coder system prompt must declare its iteration budget."""
    prompt_lower = REPO_CODER_SYSTEM_PROMPT.lower()
    assert "budget" in prompt_lower
    assert "30" in REPO_CODER_SYSTEM_PROMPT
    assert "iteration" in prompt_lower


def test_system_prompt_warns_against_late_refactor():
    """The prompt must give a concrete late-iteration calibration warning.

    The exact wording is "A 5-step refactor at iteration 28 will not finish."
    — checking the "iteration 28" anchor keeps the test stable if the
    surrounding sentence is reworded.
    """
    assert "iteration 28" in REPO_CODER_SYSTEM_PROMPT


def test_system_prompt_mentions_completion_mode():
    """Completion-mode handoff is documented in the prompt for when (and if)
    operator-side heartbeats start being relayed back to the Coder."""
    assert "completion mode" in REPO_CODER_SYSTEM_PROMPT.lower()


def test_completion_mode_threshold_is_80_percent():
    """At iteration 24 (80% of 30) the heartbeat should switch to completion-mode
    wording. This is a literal-int check rather than a runtime trace because the
    full loop pulls in Anthropic SDK + Neo4j; the threshold logic itself is a
    one-liner in run_agentic_repo_coder and the constant is what we lock down."""
    assert int(MAX_ITERATIONS * 0.8) == 24
