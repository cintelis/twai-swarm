"""
Langfuse integration — LLM call tracing.

Every provider call is wrapped with a `generation` context so Langfuse
captures the prompt, response, tokens, model, and cost. Same SDK code
works against Langfuse Cloud and our self-hosted instance — only the
`host=` value differs.

Graceful no-op path: if any of LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY is missing or equals the "UNSET" placeholder
Terraform writes before the first project is created, every function
here becomes a cheap no-op. The provider code doesn't need to check —
it calls `start_generation()` and `end_generation()` unconditionally.

This keeps Langfuse strictly additive: local dev, CI, and first-apply
production all work without Langfuse being fully configured.

Sprint 19 — Tier 1 workflow tracing. Adds two new ContextVars
(`_current_workflow_trace_id`, `_current_agent_span_id`) plus four
helpers (`workflow_trace`, `agent_span`, `score_workflow`, `shutdown`)
so each repo-task activity can emit a Langfuse trace + span pair that
the existing `generation()` cm auto-attaches to. Pattern 1 from
langfuse-integration.md (deterministic trace IDs = workflow_id; the
Langfuse server upserts on `trace.id`, so each activity's call to
`client.trace(id=workflow_id, ...)` merges into the same row).
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from app import config

logger = logging.getLogger(__name__)

_client: Optional[Any] = None
_initialised = False

# ContextVar so callers up the stack (runner, activities) can set the
# current tenant once and have every subsequent LLM-call trace auto-tag
# with it — without plumbing tenant_id through every provider signature.
# Defaults to "default" so un-scoped code paths still work.
_current_tenant: ContextVar[str] = ContextVar("swarm_tenant_id", default="default")

# Sprint 19 — workflow + agent ContextVars. Set by `workflow_trace()` /
# `agent_span()` context managers; read by `generation()` so that an
# LLM call inside a wrapped activity automatically nests under the right
# trace + parent span without the provider code needing to know about it.
# Both default to None, which is interpreted as "orphan generation"
# (still tracked, just not nested) by the Langfuse SDK.
_current_workflow_trace_id: ContextVar[Optional[str]] = ContextVar(
    "swarm_workflow_trace_id", default=None
)
_current_agent_span_id: ContextVar[Optional[str]] = ContextVar(
    "swarm_agent_span_id", default=None
)


@contextmanager
def tenant_scope(tenant_id: str):
    """Set the current tenant for all nested observability.generation() calls.

    Usage:
        with observability.tenant_scope(tenant_id):
            result = await provider.complete(...)

    On exit, restores the previous tenant (or "default"). ContextVar is
    asyncio-safe: concurrent activities running in the same worker for
    different tenants each get their own scope.
    """
    token = _current_tenant.set(tenant_id)
    try:
        yield
    finally:
        _current_tenant.reset(token)


def current_tenant() -> str:
    """Read the tenant_id set by the innermost `tenant_scope` on the stack."""
    return _current_tenant.get()


def current_workflow_trace_id() -> Optional[str]:
    """Read the workflow trace id set by the innermost `workflow_trace`.

    Returns None when called outside any `workflow_trace` block. Used by
    `generation()` to attach LLM-call observations to the right trace.
    """
    return _current_workflow_trace_id.get()


def current_agent_span_id() -> Optional[str]:
    """Read the agent span id set by the innermost `agent_span`.

    Returns None when called outside any `agent_span` block. Used by
    `generation()` to nest LLM-call observations under the right parent
    span (so the trace tree shows agent -> generation rather than a flat
    list of generations under the workflow trace).
    """
    return _current_agent_span_id.get()


def _get_client() -> Optional[Any]:
    """Lazily construct the Langfuse client, or return None if not configured.

    Returns None (and logs a one-line note) when any required credential is
    missing or still holds the UNSET placeholder. Callers treat None as
    "tracing disabled" and skip.
    """
    global _client, _initialised
    if _initialised:
        return _client

    _initialised = True

    required = (config.LANGFUSE_HOST, config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY)
    if not all(required) or any(v == "UNSET" for v in required):
        logger.info("Langfuse tracing disabled (missing host or placeholder keys)")
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            host=config.LANGFUSE_HOST,
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
        )
        logger.info("Langfuse tracing enabled → %s", config.LANGFUSE_HOST)
    except Exception as e:
        # If the SDK import fails or the client construction errors, log and
        # disable — don't let Langfuse setup failure break the swarm.
        logger.warning("Langfuse client init failed (tracing off): %s", e)
        _client = None
    return _client


@contextmanager
def workflow_trace(
    workflow_id: str,
    name: str,
    brief: str = "",
    tenant_id: str = "default",
    **metadata: Any,
):
    """One Langfuse trace per workflow. Idempotent across activities.

    Calls `client.trace(id=workflow_id, ...)` — Langfuse server upserts on
    id ("Traces are upserted on id" — verbatim docstring), so each
    activity in a workflow that wraps its body in `workflow_trace`
    contributes metadata + spans to the SAME trace row. Tags + metadata
    are merged server-side; later writes win on conflicting keys.

    Sets `_current_workflow_trace_id` so nested `agent_span()` and
    `generation()` calls auto-attach. Restores previous value on exit.

    Yields the deterministic trace_id (== workflow_id) so callers can
    pass it to other systems (e.g. include in PR body for cross-link).
    Yields None when Langfuse is disabled.

    Marks the trace `level=ERROR` if the wrapped block raises; the
    exception is re-raised — tracing must NEVER swallow workflow errors.
    """
    client = _get_client()
    if client is None:
        # No-op path. Don't set the ContextVar; downstream generation()
        # calls will see None and skip parent_observation_id wiring.
        yield None
        return

    token = _current_workflow_trace_id.set(workflow_id)
    try:
        try:
            client.trace(
                id=workflow_id,
                name=name,
                user_id=tenant_id,
                input={"brief": brief} if brief else None,
                metadata={"tenant_id": tenant_id, **metadata},
                tags=[f"phase:{metadata['phase']}"] if metadata.get("phase") else None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Langfuse workflow_trace upsert failed: %s; "
                "ContextVar still set so nested calls will attach", e,
            )
        try:
            yield workflow_id
        except Exception as e:
            # Mark the trace as errored, then re-raise. Per D5: tracing
            # must NEVER be load-bearing — we don't swallow.
            try:
                client.trace(
                    id=workflow_id,
                    metadata={"error": f"{type(e).__name__}: {str(e)[:500]}"},
                    tags=["error"],
                )
            except Exception:
                pass
            raise
    finally:
        _current_workflow_trace_id.reset(token)


@contextmanager
def agent_span(
    name: str,
    agent_role: str,
    **metadata: Any,
):
    """One span per agent invocation. Nests under current workflow_trace.

    Reads `_current_workflow_trace_id` to wire the span under the right
    trace. If no workflow_trace is active (or Langfuse is disabled), this
    is a no-op — the agent runs but produces no span.

    Generates a fresh span id (uuid4) on each entry; sets
    `_current_agent_span_id` so any nested `generation()` calls inside
    this block use it as `parent_observation_id`. Restores prior value
    on exit so sibling spans nest correctly.

    Yields the span id (or None when no-op). Marks span `level=ERROR`
    + re-raises if the wrapped block raises.
    """
    client = _get_client()
    trace_id = _current_workflow_trace_id.get()
    if client is None or trace_id is None:
        yield None
        return

    span_id = str(uuid.uuid4())
    start_time = dt.datetime.now(dt.timezone.utc)
    span_obj: Any = None
    try:
        span_obj = client.span(
            id=span_id,
            trace_id=trace_id,
            parent_observation_id=_current_agent_span_id.get(),
            name=name,
            start_time=start_time,
            metadata={"agent_role": agent_role, **metadata},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Langfuse agent_span start failed: %s; "
            "proceeding without span (ContextVar still set)", e,
        )

    token = _current_agent_span_id.set(span_id)
    try:
        try:
            yield span_id
        except Exception as e:
            # Mark errored + end + re-raise.
            if span_obj is not None:
                try:
                    span_obj.end(
                        end_time=dt.datetime.now(dt.timezone.utc),
                        level="ERROR",
                        status_message=f"{type(e).__name__}: {str(e)[:500]}",
                    )
                except Exception:
                    pass
            raise
        else:
            # Normal end.
            if span_obj is not None:
                try:
                    span_obj.end(end_time=dt.datetime.now(dt.timezone.utc))
                except Exception as e:  # noqa: BLE001
                    logger.warning("Langfuse agent_span end failed: %s", e)
    finally:
        _current_agent_span_id.reset(token)


def score_workflow(
    workflow_id: str,
    name: str,
    value: Any,
    data_type: str = "NUMERIC",
    comment: str = "",
) -> None:
    """Attach a score to a (possibly already-completed) workflow trace.

    Tier 1 helper for Tier 2's scoring pipeline (acceptance_criteria_pct,
    cost_usd_total, etc.). No-op when Langfuse is disabled. Errors are
    swallowed and logged — scoring is additive, never load-bearing.

    `data_type`: "NUMERIC" (default), "CATEGORICAL", "BOOLEAN" — passed
    verbatim to the SDK; see langfuse.client.Langfuse.score docstring.
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.score(
            trace_id=workflow_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment or None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse score_workflow failed: %s", e)


def shutdown() -> None:
    """Graceful flush + thread join. Call from worker shutdown path.

    The v2 SDK's `shutdown()` flushes the pending event queue AND joins
    the consumer thread, which is what we want at process exit so the
    last batch of events makes it out before the process dies.

    No-op when Langfuse is disabled. Errors swallowed (logged).
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse shutdown failed: %s", e)


@contextmanager
def generation(
    name: str,
    model: str,
    provider: str,
    system: str,
    user: str,
    tools: list[dict] | None = None,
    tenant_id: str | None = None,
    metadata: dict | None = None,
):
    """Context manager that opens a Langfuse generation span.

    Yields an object with `.end(output, usage, cost, level)` — callers pass
    their result back in. If Langfuse isn't configured, yields a no-op object
    with the same interface so provider code is uniform.

    Sprint 19: when called inside a `workflow_trace` + `agent_span` block,
    auto-wires `trace_id` and `parent_observation_id` so the generation
    nests under the right span in the Langfuse UI tree. Outside any
    workflow_trace, both default to None which means "orphan generation"
    — still tracked, just not nested.

    Typical use from a provider adapter:

        with observability.generation(
            name=f"anthropic.{model}", model=model, provider="anthropic",
            system=system, user=user, tools=tools,
        ) as gen:
            resp = await client.messages.create(...)
            gen.end(output=text, usage={"input": ..., "output": ...})
    """
    client = _get_client()
    if client is None:
        yield _NoopGeneration()
        return

    # Resolve tenant: explicit param > ContextVar > default. Providers don't
    # pass tenant_id explicitly — they inherit from the tenant_scope the
    # runner sets. An explicit param still wins if a caller wants to override.
    effective_tenant = tenant_id if tenant_id is not None else _current_tenant.get()

    # Sprint 19: read the workflow + agent ContextVars so the generation
    # auto-nests under the active workflow_trace + agent_span. Both None
    # is the "orphan generation" path (greenfield providers running
    # outside any workflow_trace block keep working unchanged).
    trace_id = _current_workflow_trace_id.get()
    parent_observation_id = _current_agent_span_id.get()

    try:
        gen_kwargs: dict = dict(
            name=name,
            model=model,
            input={"system": system, "user": user, "tools": tools},
            metadata={
                "provider": provider,
                "tenant_id": effective_tenant,
                **(metadata or {}),
            },
        )
        if trace_id is not None:
            gen_kwargs["trace_id"] = trace_id
        if parent_observation_id is not None:
            gen_kwargs["parent_observation_id"] = parent_observation_id
        lf_gen = client.generation(**gen_kwargs)
    except Exception as e:
        logger.warning("Langfuse generation start failed: %s; proceeding without trace", e)
        yield _NoopGeneration()
        return

    wrapper = _LangfuseGeneration(lf_gen)
    try:
        yield wrapper
    except Exception as e:
        # Don't swallow the exception — just mark the trace and re-raise.
        try:
            wrapper.end(level="ERROR", status_message=str(e)[:500])
        except Exception:
            pass
        raise


class _LangfuseGeneration:
    """Thin wrapper so our callers have a stable interface regardless of
    which Langfuse SDK version is installed."""

    def __init__(self, underlying: Any):
        self._gen = underlying
        self._ended = False

    def end(
        self,
        output: Any = None,
        usage: dict | None = None,
        cost: float | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            kwargs: dict = {}
            if output is not None:
                kwargs["output"] = output
            # Sprint 19: rename usage→usage_details (v2.60.9 marks `usage`
            # DEPRECATED in favor of `usage_details` with the same dict
            # shape). Provider call sites still pass `usage=...` to THIS
            # wrapper; only the kwarg name passed downstream to
            # client.generation() changes.
            if usage is not None:
                kwargs["usage_details"] = usage
            # Sprint 19: cost → cost_details so the Langfuse UI renders it
            # natively in the per-trace cost column. Same dict shape as
            # before; only the wrapper kwarg name changes.
            if cost is not None:
                kwargs["cost_details"] = {"input": cost}
            if level is not None:
                kwargs["level"] = level
            if status_message is not None:
                kwargs["status_message"] = status_message
            self._gen.end(**kwargs)
        except Exception as e:
            logger.warning("Langfuse generation end failed: %s", e)


class _NoopGeneration:
    """Same interface as _LangfuseGeneration when tracing is disabled."""

    def end(self, *args: Any, **kwargs: Any) -> None:
        return None


def flush() -> None:
    """Force-flush pending traces. Call before process exit to avoid drops.

    Called from each repo-task activity's `finally` block (per-activity
    flush — Sprint 19 D4) and from the worker's shutdown handler. Safe
    when tracing is off.
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
