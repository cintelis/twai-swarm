"""Sprint 19 — Tier 1 workflow tracing tests.

Covers the new ContextVars (`_current_workflow_trace_id`,
`_current_agent_span_id`), the four new helpers (`workflow_trace`,
`agent_span`, `score_workflow`, `shutdown`), and the migrated
`generation()` cm that now passes `trace_id` + `parent_observation_id`
to Langfuse.

Patches `_get_client` (no-op cases) or `_client` (mocked-call cases)
per the plan's test isolation strategy.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_module(monkeypatch):
    """Fresh module state between tests — no client cache leaks."""
    from app import observability
    observability._client = None
    observability._initialised = False
    yield


def _enable_langfuse(monkeypatch, observability, mock_client):
    """Helper: pretend Langfuse is configured + inject a mock client."""
    monkeypatch.setattr("app.config.LANGFUSE_HOST", "https://lf.example.com")
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(observability, "_client", mock_client)
    monkeypatch.setattr(observability, "_initialised", True)


# ─── ContextVar getters + workflow_trace ───────────────────────────────────


def test_workflow_trace_sets_contextvar(monkeypatch):
    """Inside the cm: ContextVar is set; outside: reset to None."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    assert observability.current_workflow_trace_id() is None
    with observability.workflow_trace(
        workflow_id="wf-abc-123",
        name="repo-task-wf-abc-123",
        phase="clone",
    ):
        assert observability.current_workflow_trace_id() == "wf-abc-123"
    assert observability.current_workflow_trace_id() is None


def test_workflow_trace_noop_when_langfuse_unset(monkeypatch):
    """Patch _get_client → None: workflow_trace yields None and ContextVar stays None."""
    from app import observability
    monkeypatch.setattr(observability, "_get_client", lambda: None)

    with observability.workflow_trace(workflow_id="wf-1", name="t") as wid:
        assert wid is None
        # Per the spec: when Langfuse is disabled, the ContextVar is NOT
        # set (avoids confusing a downstream generation() call into
        # thinking it should attach to a non-existent trace).
        assert observability.current_workflow_trace_id() is None


def test_workflow_trace_idempotent_across_calls(monkeypatch):
    """Two enters with the same workflow_id → client.trace called twice (Pattern 1)."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.workflow_trace(workflow_id="wf-x", name="t", phase="a"):
        pass
    with observability.workflow_trace(workflow_id="wf-x", name="t", phase="b"):
        pass

    # Both calls fire client.trace; the SERVER dedups on id (Pattern 1).
    assert mock_client.trace.call_count == 2
    # Both calls used the same id.
    for call in mock_client.trace.call_args_list:
        assert call.kwargs["id"] == "wf-x"


def test_workflow_trace_swallows_langfuse_errors(monkeypatch):
    """If client.trace raises, the wrapped block still runs to completion."""
    from app import observability
    mock_client = MagicMock()
    mock_client.trace.side_effect = RuntimeError("langfuse server down")
    _enable_langfuse(monkeypatch, observability, mock_client)

    ran = False
    with observability.workflow_trace(workflow_id="wf-9", name="t"):
        ran = True
        # ContextVar IS still set even when the upsert failed (so nested
        # spans/generations attempt their own attach).
        assert observability.current_workflow_trace_id() == "wf-9"
    assert ran, "workflow_trace block should still run when client.trace raises"


# ─── agent_span ────────────────────────────────────────────────────────────


def test_agent_span_nests_under_workflow_trace(monkeypatch):
    """agent_span called inside workflow_trace → client.span gets trace_id=workflow_id, parent_observation_id=None."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.workflow_trace(workflow_id="wf-42", name="t"):
        with observability.agent_span("architect", agent_role="architect"):
            pass

    span_call = mock_client.span.call_args
    assert span_call.kwargs["trace_id"] == "wf-42"
    assert span_call.kwargs["parent_observation_id"] is None
    assert span_call.kwargs["name"] == "architect"
    assert span_call.kwargs["metadata"]["agent_role"] == "architect"


def test_agent_span_nests_under_parent_agent_span(monkeypatch):
    """span B inside span A → B's parent_observation_id == A's id."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.workflow_trace(workflow_id="wf-7", name="t"):
        with observability.agent_span("outer", agent_role="critic") as a_id:
            with observability.agent_span("inner", agent_role="critic"):
                pass

    # Two span calls; the second one's parent_observation_id is the first's id.
    assert mock_client.span.call_count == 2
    calls = mock_client.span.call_args_list
    outer_id = calls[0].kwargs["id"]
    assert outer_id == a_id
    assert calls[1].kwargs["parent_observation_id"] == outer_id


def test_agent_span_noop_outside_workflow_trace(monkeypatch):
    """agent_span called without an active workflow_trace → yields None, no client call."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.agent_span("orphan", agent_role="x") as sid:
        assert sid is None

    mock_client.span.assert_not_called()


def test_agent_span_marks_error_on_exception(monkeypatch):
    """Raise inside agent_span → span.end called with level=ERROR + exception re-raised."""
    from app import observability
    mock_span = MagicMock()
    mock_client = MagicMock()
    mock_client.span.return_value = mock_span
    _enable_langfuse(monkeypatch, observability, mock_client)

    with pytest.raises(RuntimeError, match="boom"):
        with observability.workflow_trace(workflow_id="wf-err", name="t"):
            with observability.agent_span("coder", agent_role="coder"):
                raise RuntimeError("boom")

    # span.end was called with level=ERROR.
    mock_span.end.assert_called_once()
    end_kwargs = mock_span.end.call_args.kwargs
    assert end_kwargs["level"] == "ERROR"
    assert "boom" in end_kwargs["status_message"]


# ─── score_workflow ────────────────────────────────────────────────────────


def test_score_workflow_calls_client_score(monkeypatch):
    """score_workflow forwards to client.score with the right args."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    observability.score_workflow(
        workflow_id="wf-score",
        name="acceptance_criteria_pct",
        value=0.83,
        comment="6/7 passed",
    )
    mock_client.score.assert_called_once()
    kw = mock_client.score.call_args.kwargs
    assert kw["trace_id"] == "wf-score"
    assert kw["name"] == "acceptance_criteria_pct"
    assert kw["value"] == 0.83
    assert kw["data_type"] == "NUMERIC"
    assert kw["comment"] == "6/7 passed"


def test_score_workflow_noop_when_langfuse_unset(monkeypatch):
    """No-op silently when client is None — no exception, no client call."""
    from app import observability
    monkeypatch.setattr(observability, "_get_client", lambda: None)
    # Should not raise.
    observability.score_workflow(workflow_id="wf-x", name="any", value=1.0)


# ─── shutdown ──────────────────────────────────────────────────────────────


def test_shutdown_calls_client_shutdown(monkeypatch):
    """shutdown() forwards to client.shutdown()."""
    from app import observability
    mock_client = MagicMock()
    _enable_langfuse(monkeypatch, observability, mock_client)

    observability.shutdown()
    mock_client.shutdown.assert_called_once()


# ─── generation linkage to ContextVars ─────────────────────────────────────


def test_generation_passes_trace_and_span_ids_when_set(monkeypatch):
    """Inside workflow_trace + agent_span, generation() must pass trace_id +
    parent_observation_id to client.generation."""
    from app import observability
    mock_gen = MagicMock()
    mock_client = MagicMock()
    mock_client.generation.return_value = mock_gen
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.workflow_trace(workflow_id="wf-gen-99", name="t"):
        with observability.agent_span("architect", agent_role="architect") as span_id:
            with observability.generation(
                name="anthropic.x", model="x", provider="anthropic",
                system="s", user="u",
            ) as gen:
                gen.end(output="ok", usage={"input": 5, "output": 7})

    call_kwargs = mock_client.generation.call_args.kwargs
    assert call_kwargs["trace_id"] == "wf-gen-99"
    assert call_kwargs["parent_observation_id"] == span_id

    # Verify usage_details migration (Sprint 19 D3) — the wrapper now
    # forwards `usage` as `usage_details` to client.generation.end().
    end_kwargs = mock_gen.end.call_args.kwargs
    assert end_kwargs.get("usage_details") == {"input": 5, "output": 7}
    assert "usage" not in end_kwargs


def test_generation_orphan_when_no_workflow_trace(monkeypatch):
    """Outside workflow_trace, generation() works but doesn't pass trace_id."""
    from app import observability
    mock_gen = MagicMock()
    mock_client = MagicMock()
    mock_client.generation.return_value = mock_gen
    _enable_langfuse(monkeypatch, observability, mock_client)

    with observability.generation(
        name="orphan", model="x", provider="anthropic", system="s", user="u",
    ) as gen:
        gen.end(output="ok", usage={"input": 1, "output": 2})

    call_kwargs = mock_client.generation.call_args.kwargs
    # trace_id and parent_observation_id are absent (or None) when no
    # workflow_trace is active; the SDK creates an orphan generation.
    assert "trace_id" not in call_kwargs
    assert "parent_observation_id" not in call_kwargs
