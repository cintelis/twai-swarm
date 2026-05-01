"""Sprint 19 — assert each repo-task activity wraps its body in workflow_trace
+ agent_span with the right metadata.

DESIGN CHOICE: source-string assertions via `inspect.getsource(activity)`
rather than full mocking of Anthropic + Neo4j + sandbox setup. The plan
explicitly authorises this fallback ("If wiring tests prove too fragile,
fall back to source-string assertions ... Document the choice").

Rationale:
- The activity bodies are wrapped statically — there's no dynamic dispatch
  or runtime config that could move the wrap. A source-string check is
  exactly as strong as a mocked-call check for "is the wrap present and
  using the right kwargs", at a fraction of the test fragility.
- Mocking the full activity stack (Anthropic AsyncAnthropic, Neo4j
  driver factory, repo_query.find_modules / find_processes, the sandbox
  Path object, GitHub App push, etc.) for each of seven activities
  would be ~600 LOC of fixture scaffolding. Per the plan's LOC budget
  (~200 lines for ALL Sprint 19 tests), source-string checks are the
  pragmatic choice.
- The runtime correctness of workflow_trace / agent_span themselves is
  covered by tests/test_observability_workflow_trace.py.
"""
from __future__ import annotations

import inspect

from app import activities


def _src(fn):
    """Return the underlying function source for either an @activity.defn
    or a plain function."""
    target = getattr(fn, "__wrapped__", fn)
    return inspect.getsource(target)


# ─── workflow_trace + agent_span wrapping per activity ──────────────────────


def test_clone_activity_wraps_in_workflow_trace():
    src = _src(activities.clone_repo_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="clone"' in src
    assert "observability.flush()" in src


def test_index_activity_wraps_in_workflow_trace():
    src = _src(activities.index_repo_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="index"' in src
    assert "observability.flush()" in src


def test_architect_activity_wraps_in_workflow_trace_and_agent_span():
    src = _src(activities.architect_repo_task_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="architect"' in src
    assert "observability.agent_span(" in src
    assert 'name="architect"' in src
    assert 'agent_role="architect"' in src
    assert "brief_hash" in src
    assert "observability.flush()" in src


def test_coder_activity_wraps_in_workflow_trace_and_agent_span():
    src = _src(activities.run_repo_coder_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="coder"' in src
    assert "observability.agent_span(" in src
    assert 'agent_role="coder"' in src
    assert "coder_seed" in src
    assert "observability.flush()" in src


def test_critic_activity_wraps_in_workflow_trace_and_agent_span():
    src = _src(activities.critic_repo_task_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="critic"' in src
    assert "observability.agent_span(" in src
    assert 'agent_role="critic"' in src
    assert "verdict_source" in src
    assert "observability.flush()" in src


def test_reviewer_activity_wraps_in_workflow_trace_and_agent_span():
    src = _src(activities.reviewer_repo_task_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="reviewer"' in src
    assert "observability.agent_span(" in src
    assert 'agent_role="reviewer"' in src
    assert "n_candidates" in src
    assert "observability.flush()" in src


def test_push_activity_wraps_in_workflow_trace():
    src = _src(activities.push_repo_changes_activity)
    assert "observability.workflow_trace(" in src
    assert 'phase="push"' in src
    assert "observability.flush()" in src


# ─── generation() wraps each agent's LLM call sites ────────────────────────


def test_architect_tool_runner_wrapped_in_generation():
    """architect_repo.run_architect_repo wraps tool_runner in generation()."""
    from app.agents import architect_repo
    src = inspect.getsource(architect_repo.run_architect_repo)
    assert "observability.generation(" in src
    assert ".tool_runner" in src
    assert "client.beta.messages.tool_runner" in src


def test_architect_force_emit_wrapped_in_generation():
    """architect_repo._force_emit_plan wraps the messages.create in generation()."""
    from app.agents import architect_repo
    src = inspect.getsource(architect_repo._force_emit_plan)
    assert "observability.generation(" in src
    assert ".force_emit" in src


def test_coder_tool_runner_wrapped_in_generation():
    from app.agents import coder_repo
    src = inspect.getsource(coder_repo.run_agentic_repo_coder)
    assert "observability.generation(" in src
    assert ".tool_runner" in src
    assert "client.beta.messages.tool_runner" in src


def test_critic_judge_wrapped_in_generation():
    """critic_repo.run_llm_checklist_judge wraps messages.create in generation()."""
    from app.agents import critic_repo
    src = inspect.getsource(critic_repo.run_llm_checklist_judge)
    assert "observability.generation(" in src
    assert ".judge" in src


def test_critic_brief_criteria_wrapped_in_generation():
    """critic_repo._extract_criteria_from_brief wraps messages.create in generation()."""
    from app.agents import critic_repo
    src = inspect.getsource(critic_repo._extract_criteria_from_brief)
    assert "observability.generation(" in src
    assert ".brief_criteria" in src


def test_reviewer_pairwise_wrapped_in_generation():
    """reviewer_repo._call_judge wraps messages.create in generation()."""
    from app.agents import reviewer_repo
    src = inspect.getsource(reviewer_repo._call_judge)
    assert "observability.generation(" in src
    assert ".pairwise" in src


def test_greenfield_coder_tool_runner_wrapped_in_generation():
    """coder_agentic.run_agentic_coder wraps tool_runner in generation()."""
    from app.agents import coder_agentic
    src = inspect.getsource(coder_agentic.run_agentic_coder)
    assert "observability.generation(" in src
    assert ".tool_runner" in src


# ─── Smoke test: the activities are still importable + decorated correctly ──


def test_repo_task_activities_still_have_activity_defn():
    """Sanity check: the @activity.defn decorator survived the body rewrite."""
    for fn in [
        activities.clone_repo_activity,
        activities.index_repo_activity,
        activities.architect_repo_task_activity,
        activities.run_repo_coder_activity,
        activities.critic_repo_task_activity,
        activities.reviewer_repo_task_activity,
        activities.push_repo_changes_activity,
    ]:
        # @activity.defn attaches __temporal_activity_definition.
        assert hasattr(fn, "__temporal_activity_definition") or hasattr(fn, "__wrapped__"), (
            f"{fn.__name__} no longer looks like a Temporal activity"
        )
