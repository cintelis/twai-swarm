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
"""
from __future__ import annotations

import logging
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

    try:
        lf_gen = client.generation(
            name=name,
            model=model,
            input={"system": system, "user": user, "tools": tools},
            metadata={
                "provider": provider,
                "tenant_id": effective_tenant,
                **(metadata or {}),
            },
        )
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
            if usage is not None:
                kwargs["usage"] = usage
            if cost is not None:
                kwargs["metadata"] = {"cost_usd": cost}
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

    Called from the worker's shutdown handler. Safe when tracing is off.
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
