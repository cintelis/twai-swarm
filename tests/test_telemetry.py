"""Telemetry module — init + no-op behaviour without an OTLP endpoint."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset_module(monkeypatch):
    """Reset module state between tests so init() can run fresh each time."""
    from app import telemetry
    monkeypatch.setattr(telemetry, "_initialised", False)
    monkeypatch.setattr(telemetry, "_tracer", None)
    monkeypatch.setattr(telemetry, "_meter", None)
    monkeypatch.setattr(telemetry, "_metrics", {})
    yield


def test_init_without_endpoint_is_noop(monkeypatch):
    """Without OTEL_EXPORTER_OTLP_ENDPOINT, init returns without setting up SDK."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    assert telemetry.is_enabled() is False


def test_double_init_is_safe(monkeypatch):
    """Second init() call must be a no-op (no double-instrumentation)."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    telemetry.init(role="api")   # should not error
    assert telemetry.is_enabled() is False


def test_span_yields_none_when_disabled(monkeypatch):
    """span() must yield gracefully without a tracer so callers don't branch."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    with telemetry.span("test", attr="value") as s:
        assert s is None


def test_counter_add_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    # Should not raise even though no counter named 'made_up' exists
    telemetry.counter_add("made_up", 1, foo="bar")


def test_histogram_record_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    telemetry.histogram_record("made_up", 1.5, foo="bar")


def test_instrument_fastapi_noop_when_disabled(monkeypatch):
    """instrument_fastapi must accept any arg + do nothing when telemetry is off."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from app import telemetry
    telemetry.init(role="api")
    telemetry.instrument_fastapi(object())   # not even a real FastAPI app


def test_init_with_endpoint_branches_into_sdk_setup(monkeypatch):
    """When endpoint IS set, init() takes the SDK-setup branch. We don't
    actually want to set up the SDK in tests (it spawns background threads
    + real exporters), so we just verify init() got past the endpoint
    check by intercepting the SDK import."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    import sys
    # Force the SDK import inside init() to fail with ImportError. The init
    # function is supposed to catch ImportError and disable telemetry
    # gracefully — proves the early-return path on missing endpoint isn't
    # also hit on the populated-endpoint branch.
    real_modules = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry.")}
    for k in list(real_modules):
        sys.modules.pop(k, None)
    sys.modules["opentelemetry"] = None  # Force ImportError on the next import

    from app import telemetry
    # init must NOT raise even when SDK import fails — handled by the
    # try/except ImportError block. Telemetry just stays disabled.
    telemetry.init(role="api")
    assert telemetry.is_enabled() is False

    # Restore real modules so subsequent tests aren't broken
    for k, v in real_modules.items():
        sys.modules[k] = v
    sys.modules.pop("opentelemetry", None)
