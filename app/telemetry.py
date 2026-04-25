"""
OpenTelemetry — application-layer traces + metrics.

Three signals captured (when configured):
- Auto traces from FastAPI requests, asyncpg queries, httpx outbound calls.
- Manual spans on swarm-specific paths (agent activities, Coder loop,
  embeddings, LLM provider calls).
- Custom metrics (counters + histograms) for workflow / LLM / fallback
  health that don't fit naturally into spans.

`init(role)` is the entry point. Called once per process from api.py
(role="api") and worker.py (role="worker"). Idempotent — second call
is a no-op so reloads / tests don't double-instrument.

Sends nothing when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset — the SDK
isn't even initialised, span/meter calls become cheap no-ops via the
default global providers. Greenfield will point this at the EKS OTel
collector; dev today runs without the export overhead.

Tenant propagation: spans created under a `observability.tenant_scope`
auto-attach `tenant_id` as a span attribute via `_attach_tenant()`.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Module-level state — set by init(); used by helpers
_initialised = False
_tracer: Optional[Any] = None
_meter: Optional[Any] = None
_metrics: dict[str, Any] = {}
_role: str = "unknown"


def is_enabled() -> bool:
    """True when telemetry export is configured and SDK initialised."""
    return _initialised


def init(role: str) -> None:
    """Initialise tracer + meter providers and instrument shared libraries.

    `role` is "api" / "worker" / "bootstrap" — used in the service.name
    resource attribute so the same trace can show which process produced it.
    Safe to call once per process. Subsequent calls log + return without
    re-instrumenting.
    """
    global _initialised, _tracer, _meter, _role
    if _initialised:
        logger.debug("telemetry.init() called more than once; ignoring")
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info(
            "telemetry disabled: OTEL_EXPORTER_OTLP_ENDPOINT not set "
            "(spans + metrics become cheap no-ops)"
        )
        # Still set _role so attribute helpers work in tests + dev.
        _role = role
        return

    try:
        from opentelemetry import metrics as ot_metrics
        from opentelemetry import trace as ot_trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.warning("OTel SDK not installed (%s); telemetry disabled", e)
        _role = role
        return

    service_name = os.getenv("OTEL_SERVICE_NAME") or f"twai-swarm-{role}"
    resource = Resource(attributes={
        "service.name": service_name,
        "service.namespace": "twai-swarm",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "dev"),
        "swarm.role": role,
    })

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    ot_trace.set_tracer_provider(tracer_provider)
    _tracer = ot_trace.get_tracer("twai_swarm")

    # Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
        export_interval_millis=15_000,   # flush every 15s
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    ot_metrics.set_meter_provider(meter_provider)
    _meter = ot_metrics.get_meter("twai_swarm")

    _build_metrics()
    _instrument_libraries()

    _initialised = True
    _role = role
    logger.info("telemetry initialised: role=%s endpoint=%s", role, endpoint)


def _build_metrics() -> None:
    """Define the swarm-specific counters + histograms once at init."""
    if _meter is None:
        return
    _metrics["workflow_starts"] = _meter.create_counter(
        name="twai_swarm.workflow.starts",
        description="Workflows scheduled (including auto-approved + manual)",
        unit="1",
    )
    _metrics["llm_calls"] = _meter.create_counter(
        name="twai_swarm.llm.calls",
        description="LLM API calls completed (any provider)",
        unit="1",
    )
    _metrics["llm_fallback_fired"] = _meter.create_counter(
        name="twai_swarm.llm.fallback_fired",
        description="Times the OpenAI fallback was triggered by a primary's transient error",
        unit="1",
    )
    _metrics["llm_latency"] = _meter.create_histogram(
        name="twai_swarm.llm.latency_seconds",
        description="End-to-end LLM call wall time",
        unit="s",
    )
    _metrics["agent_activity_duration"] = _meter.create_histogram(
        name="twai_swarm.activity.duration_seconds",
        description="Wall time for run_agent_activity / run_coder_activity",
        unit="s",
    )
    _metrics["coder_iterations"] = _meter.create_histogram(
        name="twai_swarm.coder.iterations",
        description="Iterations the agentic Coder ran before halting",
        unit="1",
    )


def _instrument_libraries() -> None:
    """Activate auto-instrumentation for libraries we use heavily."""
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        AsyncPGInstrumentor().instrument()
    except Exception as e:
        logger.warning("asyncpg auto-instrumentation failed: %s", e)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except Exception as e:
        logger.warning("httpx auto-instrumentation failed: %s", e)


def instrument_fastapi(app: Any) -> None:
    """Wrap a FastAPI app with HTTP-server-side auto-instrumentation.

    Called from api.py AFTER the FastAPI app is constructed. Safe when
    telemetry is disabled — the instrumentor itself just no-ops.
    """
    if not _initialised:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logger.warning("FastAPI auto-instrumentation failed: %s", e)


def _attach_tenant(span: Any) -> None:
    """Tag the current tenant_id on the given span.

    Reads from observability.current_tenant() so it picks up the
    `tenant_scope` set by the runner / activity wrapper.
    """
    if span is None:
        return
    try:
        from app import observability
        span.set_attribute("swarm.tenant_id", observability.current_tenant())
    except Exception:
        pass   # observability import failure shouldn't kill spans


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager that opens a manual span with given attributes.

    No-op when telemetry isn't initialised — yields None and the body runs
    unchanged. When initialised: yields the active span so callers can
    add late-discovered attributes via `s.set_attribute(...)`.

    `tenant_id` is auto-attached from the observability contextvar.
    """
    if not _initialised or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            try:
                s.set_attribute(k, v)
            except Exception:
                pass
        _attach_tenant(s)
        try:
            yield s
        except Exception as e:
            try:
                from opentelemetry import trace as ot_trace
                s.set_status(ot_trace.Status(ot_trace.StatusCode.ERROR, str(e)[:200]))
                s.record_exception(e)
            except Exception:
                pass
            raise


def counter_add(name: str, value: int = 1, **attributes: Any) -> None:
    """Increment a named counter. No-op if not initialised."""
    if not _initialised:
        return
    counter = _metrics.get(name)
    if counter is None:
        return
    try:
        counter.add(value, attributes=attributes)
    except Exception:
        pass


def histogram_record(name: str, value: float, **attributes: Any) -> None:
    """Record a histogram observation. No-op if not initialised."""
    if not _initialised:
        return
    histogram = _metrics.get(name)
    if histogram is None:
        return
    try:
        histogram.record(value, attributes=attributes)
    except Exception:
        pass
