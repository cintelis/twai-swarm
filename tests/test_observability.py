"""Langfuse observability wrapper plumbing — no real Langfuse calls."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_module(monkeypatch):
    """Fresh module state between tests — no client cache leaks."""
    from app import observability
    observability._client = None
    observability._initialised = False
    yield


def test_get_client_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr("app.config.LANGFUSE_HOST", None)
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", None)
    from app import observability
    assert observability._get_client() is None


def test_get_client_returns_none_when_unset_placeholder(monkeypatch):
    """Terraform writes 'UNSET' as the placeholder before first-run config."""
    monkeypatch.setattr("app.config.LANGFUSE_HOST", "https://langfuse.example.com")
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", "UNSET")
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", "UNSET")
    from app import observability
    assert observability._get_client() is None


def test_generation_yields_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.LANGFUSE_HOST", None)
    from app import observability
    with observability.generation(
        name="test", model="m", provider="p", system="s", user="u",
    ) as gen:
        # No-op object must accept .end() with any kwargs without error
        gen.end(output="ok", usage={"input": 1, "output": 2})
        gen.end(level="ERROR", status_message="bang")
    # No exceptions expected


def test_generation_wraps_sdk_when_configured(monkeypatch):
    """When configured, generation() constructs a Langfuse generation
    and forwards end() to it."""
    mock_gen = MagicMock()
    mock_client = MagicMock()
    mock_client.generation.return_value = mock_gen

    from app import observability
    monkeypatch.setattr("app.config.LANGFUSE_HOST", "https://langfuse.example.com")
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(observability, "_client", mock_client)
    monkeypatch.setattr(observability, "_initialised", True)

    with observability.generation(
        name="anthropic.opus", model="opus", provider="anthropic",
        system="you are helpful", user="hi", tools=None, tenant_id="acme",
    ) as gen:
        gen.end(output="hello", usage={"input": 10, "output": 5})

    # generation() was called with the expected kwargs
    mock_client.generation.assert_called_once()
    call_kwargs = mock_client.generation.call_args.kwargs
    assert call_kwargs["name"] == "anthropic.opus"
    assert call_kwargs["model"] == "opus"
    assert call_kwargs["metadata"]["tenant_id"] == "acme"

    # end() was forwarded with output + usage
    mock_gen.end.assert_called_once()
    end_kwargs = mock_gen.end.call_args.kwargs
    assert end_kwargs["output"] == "hello"
    assert end_kwargs["usage"] == {"input": 10, "output": 5}


def test_generation_marks_error_on_exception(monkeypatch):
    """Exception inside the with block → generation ended with level=ERROR + re-raised."""
    mock_gen = MagicMock()
    mock_client = MagicMock()
    mock_client.generation.return_value = mock_gen

    from app import observability
    monkeypatch.setattr("app.config.LANGFUSE_HOST", "https://langfuse.example.com")
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(observability, "_client", mock_client)
    monkeypatch.setattr(observability, "_initialised", True)

    with pytest.raises(RuntimeError, match="boom"):
        with observability.generation(
            name="n", model="m", provider="p", system="s", user="u",
        ):
            raise RuntimeError("boom")

    # Generation was ended with ERROR level
    mock_gen.end.assert_called_once()
    end_kwargs = mock_gen.end.call_args.kwargs
    assert end_kwargs["level"] == "ERROR"
    assert "boom" in end_kwargs["status_message"]


def test_sdk_failure_falls_back_to_noop(monkeypatch):
    """If the Langfuse SDK construction fails, we log and continue."""
    from app import observability

    monkeypatch.setattr("app.config.LANGFUSE_HOST", "https://langfuse.example.com")
    monkeypatch.setattr("app.config.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr("app.config.LANGFUSE_SECRET_KEY", "sk-lf-test")

    # Force the Langfuse constructor to raise
    def _bad_construct(*a, **kw):
        raise RuntimeError("langfuse init failed")

    # Patch the import inside _get_client by patching sys.modules
    fake_module = SimpleNamespace(Langfuse=_bad_construct)
    monkeypatch.setitem(__import__("sys").modules, "langfuse", fake_module)

    # _get_client should return None after catching the exception
    assert observability._get_client() is None

    # And generation should still no-op cleanly
    with observability.generation(
        name="n", model="m", provider="p", system="s", user="u",
    ) as gen:
        gen.end(output="x")
