"""Tests for the repo Documenter — Sprint 20a.

Covers the dataclass round-trip, the title cap, the fallback helpers,
and the xAI-failure → fallback path. The xAI call itself is mocked so
the suite stays offline; end-to-end runs (xAI key + Temporal worker)
are exercised in deploy.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

from app.agents.documenter_repo import (
    DOCUMENTER_MODEL,
    DocumenterRepoOutput,
    MAX_TITLE_CHARS,
    _fallback_body,
    _fallback_title,
    _parse_json_response,
    _truncate_diff,
    run_documenter_repo,
)


# ─── Dataclass roundtrip ────────────────────────────────────────────────────


def test_documenter_output_dataclass_roundtrip():
    """Instantiate → asdict → json.dumps → loads → reconstruct → equal.

    Same Temporal-serialisation invariant guarded for the Architect/Critic
    outputs — every activity return must round-trip cleanly through JSON.
    """
    sample = DocumenterRepoOutput(
        pr_title="Auth: add per-user rate limiting on /refresh",
        pr_body="## Summary\nAdds Redis-backed rate limiting...\n",
        _model=DOCUMENTER_MODEL,
        _provider="xai",
        _tokens_in=4321,
        _tokens_out=890,
        _cost_usd=0.001234,
    )
    raw = asdict(sample)
    encoded = json.dumps(raw)
    decoded = json.loads(encoded)
    rebuilt = DocumenterRepoOutput(**decoded)
    assert rebuilt == sample


def test_documenter_output_pr_title_under_70_chars():
    """The Documenter's own clamp keeps the title under MAX_TITLE_CHARS.

    Asserted via the truncation behaviour of `run_documenter_repo` itself
    rather than the dataclass (the dataclass holds whatever caller built;
    the activity is the enforcement boundary).
    """
    long_title = "X" * 200
    fake_response = {
        "pr_title": long_title,
        "pr_body": "## Summary\nbody\n",
    }
    fake_text = "```json\n" + json.dumps(fake_response) + "\n```"

    class _FakeResult:
        text = fake_text
        tokens_in = 100
        tokens_out = 50

    with patch(
        "app.agents.documenter_repo.xai_provider.complete",
        new=AsyncMock(return_value=_FakeResult()),
    ):
        import asyncio
        out = asyncio.run(run_documenter_repo(
            brief="do thing", architect_plan=None, coder_diff="",
            files_changed=[], critic_result=None,
        ))
    assert len(out.pr_title) <= MAX_TITLE_CHARS


# ─── Fallbacks ──────────────────────────────────────────────────────────────


def test_fallback_title_from_brief():
    """Multi-line brief → first non-empty line, prefixed `Swarm: `, ≤70 chars.

    The legacy push-activity title shape used the joined-on-whitespace
    version of the brief; the fallback preserves that so a Documenter
    failure produces an indistinguishable PR title.
    """
    multi_line = "Add rate limiting to the refresh endpoint\n\nMore detail follows."
    title = _fallback_title(multi_line)
    assert title.startswith("Swarm: ")
    assert len(title) <= MAX_TITLE_CHARS
    # Empty brief → still returns a non-empty default.
    assert _fallback_title("") == "Swarm: automated change"
    # Very long brief gets clamped with an ellipsis.
    long_brief = "x " * 200
    long_title = _fallback_title(long_brief)
    assert len(long_title) <= MAX_TITLE_CHARS
    assert long_title.endswith("…")


def test_fallback_body_includes_brief_and_files():
    """Fallback PR body has the brief verbatim AND the file list.

    Identical to the pre-20a push-activity body shape — the operator
    looking at the PR can't distinguish a degraded Documenter from the
    pre-Documenter behaviour.
    """
    brief = "Add a /healthz endpoint returning 200 OK"
    files = ["app/routes.py", "tests/test_health.py"]
    body = _fallback_body(brief, files)
    assert brief in body
    for f in files:
        assert f"`{f}`" in body
    # Header markers from the legacy template are preserved.
    assert "## Brief" in body
    assert "## Files changed (2)" in body
    # Empty file list still produces a body, doesn't raise.
    body_empty = _fallback_body(brief, [])
    assert "(no files reported)" in body_empty


def test_documenter_returns_fallback_on_xai_error():
    """Mock xai_provider to raise — assert DocumenterRepoOutput uses fallback.

    The activity should NEVER propagate an xAI error to the workflow
    (per the docstring contract): a Documenter outage degrades to the
    legacy brief-derived PR shape, not a workflow failure.
    """
    with patch(
        "app.agents.documenter_repo.xai_provider.complete",
        new=AsyncMock(side_effect=RuntimeError("xAI down")),
    ):
        import asyncio
        brief = "Refactor the auth middleware"
        files = ["app/auth.py"]
        out = asyncio.run(run_documenter_repo(
            brief=brief, architect_plan=None, coder_diff="",
            files_changed=files, critic_result=None,
        ))
    assert isinstance(out, DocumenterRepoOutput)
    assert out.pr_title == _fallback_title(brief)
    assert brief in out.pr_body
    assert "`app/auth.py`" in out.pr_body
    # Cost stays zero on degraded path — no successful call to bill.
    assert out._cost_usd == 0.0
    assert out._tokens_in == 0
    assert out._tokens_out == 0


# ─── Helper: parser tolerance ───────────────────────────────────────────────


def test_parse_json_response_handles_fenced_block():
    """The Grok response can be wrapped in a ```json fence; parser strips it."""
    text = """Here's the PR description:

```json
{"pr_title": "API: add /healthz", "pr_body": "## Summary\\nReturns 200."}
```
"""
    parsed = _parse_json_response(text)
    assert parsed["pr_title"] == "API: add /healthz"
    assert "## Summary" in parsed["pr_body"]


def test_parse_json_response_handles_bare_object():
    """Falls back to first {...} when no fence is present."""
    text = '{"pr_title": "T", "pr_body": "B"}'
    parsed = _parse_json_response(text)
    assert parsed == {"pr_title": "T", "pr_body": "B"}


def test_parse_json_response_returns_empty_on_garbage():
    parsed = _parse_json_response("nothing parseable here")
    assert parsed == {}
    parsed = _parse_json_response("")
    assert parsed == {}


def test_truncate_diff_clips_long_diff():
    """Diff > budget gets head + tail; diff <= budget is untouched."""
    short = "+a\n-b\n"
    assert _truncate_diff(short) == short
    long_diff = "x" * 20000
    out = _truncate_diff(long_diff, budget=1000)
    assert "diff truncated" in out
    assert len(out) < len(long_diff)
