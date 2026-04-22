"""Fallback chain in app.agents.runner: primary transient failure → OpenAI wins."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents import runner
from app.providers import ProviderResult


def _result(text="OK", tokens_in=10, tokens_out=5):
    return ProviderResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out)


class _TransientError(Exception):
    """Looks like a 503."""
    def __init__(self):
        super().__init__("simulated 503")
        self.status_code = 503


class _AuthError(Exception):
    """Looks like a 401 — must NOT trigger fallback."""
    def __init__(self):
        super().__init__("simulated 401")
        self.status_code = 401


@pytest.mark.asyncio
async def test_transient_primary_falls_back_to_openai(monkeypatch):
    anthropic_fn = AsyncMock(side_effect=_TransientError())
    openai_fn = AsyncMock(return_value=_result(text="fallback wins", tokens_in=42, tokens_out=11))
    monkeypatch.setitem(runner._PROVIDERS, "anthropic", anthropic_fn)
    monkeypatch.setitem(runner._PROVIDERS, "openai", openai_fn)

    decision = runner.router.RouteDecision(
        key="opus",
        spec=runner.router.MODELS["opus"],
        reason="test",
    )
    result, effective = await runner._complete_with_fallback(
        decision=decision,
        system="sys",
        user="u",
        max_tokens=100,
        tools=None,
    )
    assert result.text == "fallback wins"
    assert effective.provider == "openai"
    assert effective.key == "gpt54"
    assert "fallback from opus" in effective.reason
    anthropic_fn.assert_awaited_once()
    openai_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_error_does_not_trigger_fallback(monkeypatch):
    anthropic_fn = AsyncMock(side_effect=_AuthError())
    openai_fn = AsyncMock(return_value=_result())
    monkeypatch.setitem(runner._PROVIDERS, "anthropic", anthropic_fn)
    monkeypatch.setitem(runner._PROVIDERS, "openai", openai_fn)

    decision = runner.router.RouteDecision(
        key="opus",
        spec=runner.router.MODELS["opus"],
        reason="test",
    )
    with pytest.raises(_AuthError):
        await runner._complete_with_fallback(
            decision=decision,
            system="sys",
            user="u",
            max_tokens=100,
            tools=None,
        )
    openai_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_happy_path_no_fallback(monkeypatch):
    anthropic_fn = AsyncMock(return_value=_result(text="primary wins", tokens_in=7, tokens_out=3))
    openai_fn = AsyncMock(return_value=_result(text="should not be called"))
    monkeypatch.setitem(runner._PROVIDERS, "anthropic", anthropic_fn)
    monkeypatch.setitem(runner._PROVIDERS, "openai", openai_fn)

    decision = runner.router.RouteDecision(
        key="opus",
        spec=runner.router.MODELS["opus"],
        reason="test",
    )
    result, effective = await runner._complete_with_fallback(
        decision=decision,
        system="sys", user="u", max_tokens=100, tools=None,
    )
    assert result.text == "primary wins"
    assert effective.provider == "anthropic"
    assert effective.key == "opus"
    openai_fn.assert_not_awaited()


def test_is_transient_status_codes():
    # 5xx via .status_code attribute
    assert runner._is_transient(_TransientError())
    # 4xx is not transient
    assert not runner._is_transient(_AuthError())
    # Generic ValueError isn't transient
    assert not runner._is_transient(ValueError("bad input"))


def test_translate_tools_maps_web_search():
    # Anthropic-style
    assert runner.openai_provider._translate_tools(
        [{"type": "web_search_20260209", "name": "web_search"}]
    ) == [{"type": "web_search"}]
    # xAI sends both — dedupe into one
    assert runner.openai_provider._translate_tools(
        [{"type": "web_search"}, {"type": "x_search"}]
    ) == [{"type": "web_search"}]
    # Unknown tool dropped silently
    assert runner.openai_provider._translate_tools([{"type": "some_custom_tool"}]) is None
    # Empty/None passthrough
    assert runner.openai_provider._translate_tools(None) is None
    assert runner.openai_provider._translate_tools([]) is None
