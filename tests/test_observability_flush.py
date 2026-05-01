"""Sprint 19 — per-activity flush + worker shutdown hook tests."""
from __future__ import annotations

import inspect

import pytest


@pytest.fixture(autouse=True)
def reset_module(monkeypatch):
    from app import observability
    observability._client = None
    observability._initialised = False
    yield


def test_activity_calls_flush_in_finally(monkeypatch):
    """If an activity body raises, observability.flush() still fires.

    Mirrors the per-activity flush pattern: every repo-task activity
    wraps its body in `try: ... finally: observability.flush()`. We
    verify the pattern works with a stand-in async function shaped
    like an activity body.
    """
    from app import observability

    flush_calls = []
    monkeypatch.setattr(observability, "flush", lambda: flush_calls.append(1))

    async def _fake_activity():
        try:
            with observability.workflow_trace(workflow_id="wf-flush", name="t"):
                raise RuntimeError("body failed mid-flight")
        finally:
            observability.flush()

    import asyncio
    with pytest.raises(RuntimeError, match="mid-flight"):
        asyncio.run(_fake_activity())

    # flush was called despite the exception.
    assert flush_calls == [1]


def test_worker_module_registers_shutdown_hook():
    """app/worker.py must register observability.shutdown via atexit AND
    call it in the outer finally of worker.main()."""
    from app import worker
    src = inspect.getsource(worker)
    assert "atexit.register(observability.shutdown)" in src, (
        "Worker must register shutdown hook via atexit (Sprint 19 D4)"
    )
    assert "observability.shutdown()" in src, (
        "Worker must call observability.shutdown() somewhere in main "
        "(Sprint 19 D4 graceful path)"
    )


def test_observability_flush_safe_when_disabled(monkeypatch):
    """flush() is a cheap no-op when Langfuse is disabled — no exception."""
    from app import observability
    monkeypatch.setattr(observability, "_get_client", lambda: None)
    # Should not raise.
    observability.flush()


def test_observability_shutdown_safe_when_disabled(monkeypatch):
    """shutdown() is a cheap no-op when Langfuse is disabled — no exception."""
    from app import observability
    monkeypatch.setattr(observability, "_get_client", lambda: None)
    # Should not raise.
    observability.shutdown()
