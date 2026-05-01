"""Tests for the repo Architect — Sprint 18b.

Covers the dataclasses, the markdown rendering helpers, the user-message
builder, and the planning-only tool filter. The tool_runner loop itself
needs Anthropic + Neo4j and is exercised end-to-end in deploy; here we
guard the pure-Python scaffolding around it.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from app.agents.architect_repo import (
    ARCHITECT_MODEL,
    ARCHITECT_REPO_SYSTEM_PROMPT,
    MAX_ARCHITECT_ITERATIONS,
    ArchitectRepoOutput,
    Subtask,
    _build_architect_user_message,
    _coerce_subtasks,
    _planning_only_tools,
    _WRITE_TOOL_NAMES,
    architect_output_to_dict,
    render_architect_plan_section,
    render_risk_section,
)


# ─── Dataclass roundtrip ────────────────────────────────────────────────────
# Temporal serialises activity returns through JSON. If the dataclass has any
# non-primitive field (set, datetime, custom class), the activity raises at
# runtime — and only on the first real workflow run, not in unit tests that
# never invoke Temporal. The roundtrip below is the cheap insurance.


def test_architect_output_dataclass_roundtrip():
    """Instantiate → asdict → json.dumps → loads → reconstruct → equal."""
    sample = ArchitectRepoOutput(
        narrative="Investigate auth flow then add refresh endpoint.",
        subtasks=[
            Subtask(
                id="backend.refresh_endpoint",
                description="Add POST /auth/refresh that returns a new JWT.",
                files_to_touch=["app/auth/routes.py", "app/auth/jwt.py"],
                depends_on=[],
                independent=True,
                acceptance_criteria=[
                    "Endpoint returns 200 + JWT for valid refresh token",
                    "Endpoint returns 401 for expired refresh token",
                ],
            ),
            Subtask(
                id="frontend.refresh_call",
                description="Wire LoginPage.tsx to call /auth/refresh on 401.",
                files_to_touch=["frontend/LoginPage.tsx"],
                depends_on=["backend.refresh_endpoint"],
                independent=False,
                acceptance_criteria=[
                    "LoginPage retries failed request once after refresh",
                ],
            ),
        ],
        cross_cutting=True,
        risk_notes=[
            "auth.jwt.issue has 12 callers — don't change its signature",
        ],
        _model="claude-sonnet-4-6",
        _provider="anthropic",
        _tokens_in=1234,
        _tokens_out=567,
        _cost_usd=0.012,
    )
    raw = asdict(sample)
    encoded = json.dumps(raw)
    decoded = json.loads(encoded)

    # Round-trip back into dataclasses for equality. Subtasks need explicit
    # rehydration because asdict flattens them.
    rebuilt = ArchitectRepoOutput(
        narrative=decoded["narrative"],
        subtasks=[Subtask(**st) for st in decoded["subtasks"]],
        cross_cutting=decoded["cross_cutting"],
        risk_notes=decoded["risk_notes"],
        _model=decoded["_model"],
        _provider=decoded["_provider"],
        _tokens_in=decoded["_tokens_in"],
        _tokens_out=decoded["_tokens_out"],
        _cost_usd=decoded["_cost_usd"],
    )
    assert rebuilt == sample


def test_subtask_dataclass_validation():
    """Subtask shape: required fields + sensible defaults.

    `depends_on` referencing an existing or non-existent ID both serialise
    cleanly — DAG validity is the workflow's job (Sprint 18d), not the
    dataclass's. Documenting the no-validation contract here so a future
    refactor doesn't silently add validation.
    """
    # Required-only construction: id + description + files_to_touch.
    minimal = Subtask(
        id="minimal", description="do a thing", files_to_touch=[],
    )
    assert minimal.depends_on == []
    assert minimal.independent is True
    assert minimal.acceptance_criteria == []

    # Forward-reference (depends_on names a sibling that exists in the DAG).
    forward = Subtask(
        id="b", description="depends on a", files_to_touch=[],
        depends_on=["a"],
    )
    assert forward.depends_on == ["a"]

    # Dangling-reference (depends_on names an ID no other subtask has) does
    # NOT raise — by design. The dataclass is data, not a validator.
    dangling = Subtask(
        id="c", description="depends on ghost", files_to_touch=[],
        depends_on=["ghost-that-does-not-exist"],
    )
    asdict(dangling)  # asserts it serialises, no exception
    json.dumps(asdict(dangling))


def test_architect_output_cross_cutting_default():
    """Default cross_cutting is False — single-file briefs aren't scaled-up."""
    out = ArchitectRepoOutput(narrative="trivial fix")
    assert out.cross_cutting is False
    assert out.subtasks == []
    assert out.risk_notes == []


def test_architect_output_to_dict_helper():
    """The asdict helper produces a JSON-serialisable plain dict."""
    out = ArchitectRepoOutput(
        narrative="x",
        subtasks=[Subtask(id="s1", description="d", files_to_touch=["f"])],
    )
    d = architect_output_to_dict(out)
    assert isinstance(d, dict)
    assert d["narrative"] == "x"
    assert len(d["subtasks"]) == 1
    assert d["subtasks"][0]["id"] == "s1"
    # round-trip
    assert json.loads(json.dumps(d))["subtasks"][0]["files_to_touch"] == ["f"]


# ─── Markdown renderers (consumed by coder_repo._build_user_message) ────────


def test_render_architect_plan_section_full():
    plan = {
        "narrative": "Touch backend then frontend.",
        "subtasks": [
            {
                "id": "be.endpoint",
                "description": "Add the POST handler.",
                "files_to_touch": ["app/api/routes.py"],
                "acceptance_criteria": [
                    "Returns 200 on valid input",
                    "Returns 422 on missing field",
                ],
            },
            {
                "id": "fe.button",
                "description": "Wire the button.",
                "files_to_touch": [],
            },
        ],
        "cross_cutting": True,
        "risk_notes": ["routes.py is shared with /v1 routes"],
    }
    out = render_architect_plan_section(plan)
    assert "## Architect plan" in out
    assert "Touch backend then frontend." in out
    assert "**be.endpoint**" in out
    assert "`app/api/routes.py`" in out
    assert "Returns 200 on valid input" in out
    # Empty files_to_touch surfaces "uncertain"
    assert "(uncertain)" in out
    # cross_cutting line
    assert "cross_cutting: True" in out


def test_render_architect_plan_section_empty_returns_empty_string():
    assert render_architect_plan_section(None) == ""
    assert render_architect_plan_section({}) == ""


def test_render_risk_section():
    plan = {"risk_notes": ["don't change signature", "shared with v2"]}
    out = render_risk_section(plan)
    assert "## Risks" in out
    assert "don't change signature" in out
    assert "shared with v2" in out


def test_render_risk_section_empty_when_no_notes():
    assert render_risk_section({}) == ""
    assert render_risk_section({"risk_notes": []}) == ""
    assert render_risk_section(None) == ""


# ─── User-message builder ───────────────────────────────────────────────────


def test_build_architect_user_message_orders_recon_first():
    msg = _build_architect_user_message(
        brief="Add refresh tokens.",
        repo_name="myrepo",
        recon_block="## Repo recon (auto-generated)\n\n### Modules (1)\n- `auth` (5 symbols): app.auth.x",
    )
    recon_pos = msg.find("## Repo recon")
    brief_pos = msg.find("## Task brief")
    assert recon_pos != -1 and brief_pos != -1
    assert recon_pos < brief_pos
    assert "Add refresh tokens." in msg
    assert "myrepo" in msg
    # Architect's instructions reach the model.
    assert "emit_plan" in msg


def test_build_architect_user_message_omits_recon_when_empty():
    msg = _build_architect_user_message("brief", "myrepo", recon_block="")
    assert "## Repo recon" not in msg
    assert "## Task brief" in msg


# ─── Planning-only tool filter ──────────────────────────────────────────────


def test_planning_only_tools_drops_writes():
    """The Architect must NOT receive write_file / bash_exec / run_verify."""
    # Synthetic tool stand-ins: anthropic's @beta_async_tool sets `.name`
    # on the wrapped object; mimic that here.
    class FakeTool:
        def __init__(self, name): self.name = name

    tools = [
        FakeTool("list_files"),
        FakeTool("read_file"),
        FakeTool("write_file"),       # must be dropped
        FakeTool("bash_exec"),        # must be dropped
        FakeTool("run_verify"),       # must be dropped
        FakeTool("repo_search"),
        FakeTool("repo_find_callers"),
    ]
    kept = _planning_only_tools(tools, stats={})
    kept_names = [t.name for t in kept]
    assert "write_file" not in kept_names
    assert "bash_exec" not in kept_names
    assert "run_verify" not in kept_names
    # Read-only / planning tools survive.
    assert "list_files" in kept_names
    assert "read_file" in kept_names
    assert "repo_search" in kept_names
    assert "repo_find_callers" in kept_names


def test_write_tool_names_constant_locked():
    """Lock down the exact write-tool blocklist so a refactor can't silently
    grant the Architect write access (e.g. by renaming bash_exec)."""
    assert _WRITE_TOOL_NAMES == frozenset({
        "write_file", "bash_exec", "run_verify",
    })


# ─── Subtask coercion ───────────────────────────────────────────────────────


def test_coerce_subtasks_fills_defaults():
    raw = [
        # Minimal — only required fields present.
        {"id": "a", "description": "d", "files_to_touch": ["f1"]},
        # Full — every field present.
        {
            "id": "b", "description": "d2", "files_to_touch": ["f2"],
            "depends_on": ["a"], "independent": False,
            "acceptance_criteria": ["criterion1"],
        },
    ]
    out = _coerce_subtasks(raw)
    assert len(out) == 2
    assert out[0].depends_on == []
    assert out[0].independent is True
    assert out[0].acceptance_criteria == []
    assert out[1].depends_on == ["a"]
    assert out[1].independent is False
    assert out[1].acceptance_criteria == ["criterion1"]


def test_coerce_subtasks_skips_malformed():
    raw = [
        {"id": "ok", "description": "d", "files_to_touch": []},
        "not a dict",
        42,
        None,
    ]
    out = _coerce_subtasks(raw)
    assert len(out) == 1
    assert out[0].id == "ok"


def test_coerce_subtasks_handles_non_list():
    assert _coerce_subtasks(None) == []
    assert _coerce_subtasks({}) == []
    assert _coerce_subtasks("nope") == []


# ─── Module-level constants ─────────────────────────────────────────────────


def test_architect_model_is_sonnet():
    """Per D8: planning is reasoning-heavy → Sonnet 4.6, not Haiku."""
    assert ARCHITECT_MODEL == "claude-sonnet-4-6"


def test_max_iterations_is_15():
    """Sprint 18.1 bumped 8 -> 15 after run 019de315 telemetry showed 8
    was too tight for cross-cutting briefs (Architect kept investigating
    and never reached emit_plan, leaving the safety nets to vacuously
    pass against an empty subtask list). Locking the value down so a
    future tweak is intentional — Sonnet calls are 3x Haiku and the cost
    compounds across N=3 Best-of-N runs in 18d if the cap drifts up."""
    assert MAX_ARCHITECT_ITERATIONS == 15


def test_system_prompt_mentions_commit_mode():
    """Sprint 18.1 calibration note: prompt must tell the model to
    transition to 'commit mode' by iteration 12, even with a rough
    plan. Without this hint the model investigates to the bitter end."""
    p = ARCHITECT_REPO_SYSTEM_PROMPT.lower()
    assert "iteration 12" in p
    assert "commit mode" in p


def test_system_prompt_forbids_writes():
    p = ARCHITECT_REPO_SYSTEM_PROMPT.lower()
    assert "must not edit" in p or "do not edit code" in p
    # The schema-tool instruction must be present so the model knows how
    # to emit its structured output.
    assert "emit_plan" in ARCHITECT_REPO_SYSTEM_PROMPT


def test_system_prompt_mentions_acceptance_criteria_contract():
    p = ARCHITECT_REPO_SYSTEM_PROMPT.lower()
    assert "acceptance_criteria" in p
    assert "contract" in p


# ─── Worker / activity registration ─────────────────────────────────────────


def test_architect_activity_registered_in_worker():
    """Sprint 18b wires `architect_repo_task_activity` into the worker so
    Temporal can dispatch it. Refactor that drops it would silently break
    the workflow at first deploy — guard it here."""
    from app import worker
    from app.activities import architect_repo_task_activity
    assert architect_repo_task_activity in worker.ACTIVITIES


def test_architect_activity_signature():
    """The architect activity's coroutine signature must accept the five
    positional args the workflow passes (path, name, brief, tenant, wf_id).
    """
    import inspect
    from app import activities as acts
    fn = getattr(
        acts.architect_repo_task_activity, "__wrapped__",
        acts.architect_repo_task_activity,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert params == [
        "repo_path", "repo_name", "brief", "tenant_id", "workflow_id",
    ]


# ─── Workflow / output wiring ───────────────────────────────────────────────


def test_repo_task_output_has_architect_plan_field():
    """RepoTaskOutput.architect_plan: dict | None defaults to None — the
    UI consumes it for rendering the plan alongside the diff."""
    from app.workflows import RepoTaskOutput
    out = RepoTaskOutput(
        workflow_id="wf-1", repo_name="r", commit_sha="abc", files_changed=[],
        diff="", iterations=0, summary="", tokens_in=0, tokens_out=0,
        cost_usd=0.0,
    )
    assert out.architect_plan is None


def test_run_repo_coder_activity_signature_accepts_architect_plan():
    """Sprint 18b extends `run_repo_coder_activity(...)` with a trailing
    `architect_plan: dict | None = None` parameter so the workflow can
    forward the architect output. Keeping the default = None preserves
    backward-compat with replayed Temporal histories that predate 18b."""
    import inspect
    from app import activities as acts
    fn = getattr(
        acts.run_repo_coder_activity, "__wrapped__", acts.run_repo_coder_activity,
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    # New parameter is appended (positional-or-keyword with default None).
    assert "architect_plan" in params
    assert sig.parameters["architect_plan"].default is None


def test_run_agentic_repo_coder_signature_accepts_architect_plan():
    """The underlying agentic Coder function must accept architect_plan
    too, since the activity threads it through."""
    import inspect
    from app.agents.coder_repo import run_agentic_repo_coder
    sig = inspect.signature(run_agentic_repo_coder)
    assert "architect_plan" in sig.parameters
    assert sig.parameters["architect_plan"].default is None


def test_workflow_uses_architect_activity():
    """Source-string check that the workflow calls architect_repo_task_activity
    BEFORE run_repo_coder_activity. Cheap regression against a refactor that
    drops the architect step."""
    import inspect
    from app.workflows.repo_task import RepoTaskWorkflow
    src = inspect.getsource(RepoTaskWorkflow)
    arch_pos = src.find("architect_repo_task_activity")
    coder_pos = src.find("run_repo_coder_activity")
    assert arch_pos != -1, "workflow no longer mentions architect activity"
    assert coder_pos != -1
    assert arch_pos < coder_pos, "architect must run before coder"
    # And the coder call must include the arch_result in its args.
    assert "arch_result" in src
