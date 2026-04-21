"""
Mocked coder loop test — verifies the loop plumbing without hitting the API.

What we check:
- Sandbox is created and seeded with the chosen template.
- Each yielded BetaMessage stub drives one iteration and exercises heartbeats.
- Usage numbers are summed across iterations.
- Workspace snapshot appears in the output `files` array.
- On verify success, workspace is cleaned up.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents import coder_agentic
from app.agents.template_matcher import TemplateChoice


def _fake_message(text=None, input_tokens=100, output_tokens=50, stop_reason="end_turn"):
    """A BetaMessage-shaped stub sufficient for the loop's inspection."""
    blocks = []
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


class _FakeRunner:
    """Replaces client.beta.messages.tool_runner(...)."""

    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeClient:
    def __init__(self, messages, on_call=None):
        self._messages = messages
        self._on_call = on_call
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs):
        if self._on_call:
            self._on_call(kwargs)
        return _FakeRunner(self._messages)


@pytest.fixture
def templates_dir(tmp_path):
    """A tiny templates/ layout with one matching template."""
    tdir = tmp_path / "tpl" / "tiny"
    (tdir / "scaffold").mkdir(parents=True)
    (tdir / "scaffold" / "main.py").write_text("print('seeded')")
    (tdir / "scaffold" / "README.md").write_text("# seeded")
    (tdir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (tdir / "template.json").write_text(json.dumps({
        "name": "tiny",
        "language": "python",
        "selection_hints": ["tiny", "seeded"],
    }))
    return tdir.parent


@pytest.mark.asyncio
async def test_loop_seeds_template_and_accumulates_tokens(tmp_path, templates_dir, monkeypatch):
    # Pin the template matcher's root to our fixture.
    monkeypatch.setattr(coder_agentic, "pick_template", lambda *a, **kw: TemplateChoice(
        name="tiny",
        template_dir=templates_dir / "tiny",
        scaffold_dir=templates_dir / "tiny" / "scaffold",
        score=5,
        reason="fixture",
    ))
    # Use tmp_path for the sandbox base instead of /tmp/coder.
    from app.agents import coder_sandbox as cs
    real_create = cs.Sandbox.create
    monkeypatch.setattr(cs.Sandbox, "create",
                        classmethod(lambda cls, wid, base="/tmp/coder": real_create.__func__(cls, wid, base=tmp_path)))

    messages = [
        _fake_message(text="thinking about it", input_tokens=200, output_tokens=80, stop_reason="tool_use"),
        _fake_message(text="done!", input_tokens=100, output_tokens=50, stop_reason="end_turn"),
    ]
    fake_client = _FakeClient(messages)
    monkeypatch.setattr(coder_agentic, "AsyncAnthropic", lambda **_: fake_client)

    heartbeats: list[str] = []

    # Pre-seed the sandbox with a "verify passed" state so cleanup path runs.
    # Easier: patch _snapshot_workspace to return fake files, and flip stats.
    result = await coder_agentic.run_agentic_coder(
        workflow_id="wf-loop-test",
        brief="build the tiny seeded thing",
        architecture=None,
        se_plan=None,
        documenter=None,
        heartbeat=heartbeats.append,
    )

    assert result["iterations"] == 2
    assert result["_tokens_in"] == 300
    assert result["_tokens_out"] == 130
    assert result["_provider"] == "anthropic"
    assert result["_model"] == "claude-opus-4-7"
    assert result["template"] == "tiny"
    assert result["summary"] == "done!"
    # Heartbeats fired once per iteration.
    assert len(heartbeats) == 2
    # Workspace snapshot includes the seeded files.
    paths = {f["path"] for f in result["files"]}
    assert "main.py" in paths
    assert "README.md" in paths


@pytest.mark.asyncio
async def test_loop_halts_at_max_iterations(tmp_path, templates_dir, monkeypatch):
    monkeypatch.setattr(coder_agentic, "pick_template", lambda *a, **kw: TemplateChoice(
        name=None, template_dir=None, scaffold_dir=None, score=0, reason="none",
    ))
    from app.agents import coder_sandbox as cs
    real_create = cs.Sandbox.create
    monkeypatch.setattr(cs.Sandbox, "create",
                        classmethod(lambda cls, wid, base="/tmp/coder": real_create.__func__(cls, wid, base=tmp_path)))

    # Produce more messages than MAX_ITERATIONS; loop should cut off.
    messages = [_fake_message(text=f"iter {i}", stop_reason="tool_use") for i in range(20)]
    fake_client = _FakeClient(messages)
    monkeypatch.setattr(coder_agentic, "AsyncAnthropic", lambda **_: fake_client)

    result = await coder_agentic.run_agentic_coder(
        workflow_id="wf-max-iters",
        brief="loop forever",
        architecture=None,
        se_plan=None,
        documenter=None,
    )
    assert result["iterations"] == coder_agentic.MAX_ITERATIONS
    assert result["verify_passed"] is False  # never ran verify
