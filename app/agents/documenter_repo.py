"""Repo-aware Documenter — Sprint 20a.

Post-step that runs AFTER the Critic continuation loop, BEFORE the auto-PR
push. Generates a proper PR title + markdown body from the workflow's
artifacts (brief + Architect plan + Coder diff + Critic verdict) so the
PR body the human reviewer sees is a structured summary instead of the
raw brief verbatim (the pre-20a behaviour).

Provider choice: xAI Grok via the existing `app.providers.xai_provider`.
The router's `documenter` role default is `grok-fast` (which maps to
`grok-4-1-fast-reasoning` per app/router.py) — speed > nuance for write-
ups, and explicitly diversifies away from Anthropic per the architecture
audit. The Anthropic-tier critic right above us already cost a Sonnet
call; this one is sub-cent.

Output shape: `DocumenterRepoOutput` dataclass that round-trips through
`dataclasses.asdict` → `json.dumps` (Temporal serialisation invariant —
same convention as the Architect/Critic/Reviewer outputs).

Failure mode: if the xAI call raises for any reason, we degrade to the
legacy fallback — title derived from the brief's first line, body
containing the brief + a flat file list. The downstream
`push_repo_changes_activity` then produces the same PR shape it did
pre-20a, so a Documenter outage NEVER breaks the workflow.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.providers import xai_provider

logger = logging.getLogger(__name__)

# Per architecture-audit: documenter is a write-up role, speed > nuance.
# `grok-fast` is the router key; the actual API model string is
# "grok-4-1-fast-reasoning" — the -reasoning suffix routes to the
# standard-API endpoint (the plain alias is Enterprise-only). See
# app/router.py and feedback_xai_model_tiers memory note.
DOCUMENTER_MODEL = "grok-4-1-fast-reasoning"

# Single-shot, ~5K input + ~1K output, no streaming. 4K output gives
# headroom for ~80 changed files in the bullet list before truncation.
MAX_TOKENS = 4096

# Diff truncation budget — anything past this gets head+tail clipped so
# we don't blow the context budget on giant changesets.
MAX_DIFF_CHARS = 8000

# PR title hard cap — GitHub truncates at 256, but the CLAUDE.md style
# guide says ≤70. We enforce both at output time AND on the fallback.
MAX_TITLE_CHARS = 70


@dataclass
class DocumenterRepoOutput:
    """Documenter's deliverable for one repo-task workflow.

    Provenance fields (`_model`, `_provider`, `_tokens_in`, `_tokens_out`,
    `_cost_usd`) match the Architect/Critic convention so the cost-summary
    card aggregates without a special case.
    """
    pr_title: str                      # ≤70 chars (enforced at output)
    pr_body: str                       # markdown
    _model: str = DOCUMENTER_MODEL
    _provider: str = "xai"
    _tokens_in: int = 0
    _tokens_out: int = 0
    _cost_usd: float = 0.0


DOCUMENTER_REPO_SYSTEM_PROMPT = """You are a Documenter writing PR descriptions for an agentic coding pipeline. You will be shown:

- The original task brief
- The Architect's plan (subtasks + acceptance criteria)
- The actual diff the Coder shipped
- The list of files changed
- The Critic's verdict (which acceptance criteria passed/failed)

Your job: produce a PR title and markdown body that lets a human reviewer understand the change in 30 seconds.

PR title rules:
- <= 70 characters (will be truncated otherwise)
- Imperative mood ("Add per-user rate limiting" not "Added rate limiting")
- Format: `<area>: <change>` where area is the primary subsystem touched (e.g. "Auth", "API", "Frontend")
- No emojis

PR body structure (use this exact template):

```
## Summary
<1-2 sentence overview - what this PR does and why>

## What changed
- `path/to/file.java` - <one-line summary of the change>
- `path/to/another.tsx` - <one-line summary>
(group by file; one bullet per touched file; <= 8 words per bullet)

## Tests
- <list test files added or modified, with brief>
- (or "No tests in this PR - flagged in Risks below")

## Risks the Critic flagged
- <each failed_criterion verbatim, or "None - Critic verdict: complete">

## Breaking changes
- <API surface changes, schema migrations, removed exports>
- (or "None")

## Acceptance criteria status
- [x] <each passed criterion>
- [ ] <each failed criterion>
```

CRITICAL constraints:
- Do NOT invent acceptance criteria - only use what's in the Architect plan or Critic verdict
- Do NOT speculate about behavior the diff doesn't show
- Use file path basenames in bullets when paths are long (e.g. `LoginPage.tsx` not the full path)
- Be terse - reviewers skim
- Output a single JSON object with EXACTLY two keys: `pr_title` (string) and `pr_body` (string).
  Wrap the JSON in a ```json fenced code block. Do not output anything outside the code block.
"""


# Tool schema kept for future migration to a tool-call API path. The
# current xai_provider Responses-API path is reserved for server-side
# tools (web_search/x_search) — see xai_provider.complete docstring. We
# instead force structured output via the prompt's "wrap JSON in ```json
# fence" instruction and parse post-hoc, identical in spirit to the
# critic_repo judge response parser.
_EMIT_PR_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_pr_description",
    "description": "Emit the PR title + body for the just-shipped change",
    "input_schema": {
        "type": "object",
        "properties": {
            "pr_title": {
                "type": "string",
                "maxLength": MAX_TITLE_CHARS,
                "description": "PR title, imperative mood, <= 70 chars",
            },
            "pr_body": {
                "type": "string",
                "description": "Full markdown PR body following the template",
            },
        },
        "required": ["pr_title", "pr_body"],
    },
}


def _truncate_diff(diff: str, budget: int = MAX_DIFF_CHARS) -> str:
    """Head+tail clip for diffs that blow the prompt budget.

    Mirrors `_build_judge_prompt`'s 30K cap but tighter — the Documenter
    only needs enough diff context to write a one-line summary per file,
    not regrade every line.
    """
    if len(diff) <= budget:
        return diff
    head = diff[: budget * 2 // 3]
    tail = diff[-(budget // 3):]
    return f"{head}\n\n... (diff truncated for prompt budget) ...\n\n{tail}"


def _build_user_message(
    brief: str,
    architect_plan: dict | None,
    coder_diff: str,
    files_changed: list[str],
    critic_result: dict | None,
) -> str:
    """Assemble the single user message the Documenter sees.

    Sections are ordered to match the PR-body template — easier for the
    model to copy-shape from than a wall of unstructured context.
    """
    parts: list[str] = []

    parts.append("## Original task brief")
    parts.append(brief.strip() or "(empty brief)")
    parts.append("")

    if architect_plan and isinstance(architect_plan, dict):
        narrative = str(architect_plan.get("narrative", "")).strip()
        if narrative:
            parts.append("## Architect narrative")
            parts.append(narrative)
            parts.append("")
        subtasks = architect_plan.get("subtasks") or []
        if subtasks:
            parts.append("## Architect subtasks + acceptance criteria")
            for st in subtasks[:20]:
                if not isinstance(st, dict):
                    continue
                sid = st.get("id", "")
                desc = st.get("description", "")
                parts.append(f"- **{sid}**: {desc}")
                for c in (st.get("acceptance_criteria") or [])[:10]:
                    parts.append(f"  - {c}")
            if len(subtasks) > 20:
                parts.append(f"- ...and {len(subtasks) - 20} more subtasks")
            parts.append("")

    parts.append(f"## Files changed ({len(files_changed)})")
    for f in files_changed[:50]:
        parts.append(f"- `{f}`")
    if len(files_changed) > 50:
        parts.append(f"- ...and {len(files_changed) - 50} more")
    parts.append("")

    parts.append("## Coder diff")
    parts.append("```diff")
    parts.append(_truncate_diff(coder_diff or "(empty diff)"))
    parts.append("```")
    parts.append("")

    if critic_result and isinstance(critic_result, dict):
        verdict = critic_result.get("verdict", "(unknown)")
        passed = critic_result.get("passed_criteria") or []
        failed = critic_result.get("failed_criteria") or []
        parts.append(f"## Critic verdict: {verdict}")
        if passed:
            parts.append("### Passed criteria")
            for c in passed[:30]:
                parts.append(f"- {c}")
            if len(passed) > 30:
                parts.append(f"- ...and {len(passed) - 30} more")
        if failed:
            parts.append("### Failed criteria")
            for cf in failed[:30]:
                if isinstance(cf, dict):
                    text = cf.get("criterion", "")
                    evidence = cf.get("evidence", "")
                    parts.append(f"- {text} - {evidence}")
                else:
                    parts.append(f"- {cf}")
            if len(failed) > 30:
                parts.append(f"- ...and {len(failed) - 30} more")
        parts.append("")

    parts.append(
        "Now produce the PR title + body per the template in the system "
        "prompt. Return a single JSON object with keys `pr_title` and "
        "`pr_body`, wrapped in a ```json fenced code block."
    )
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract the {pr_title, pr_body} JSON from a Grok response.

    Tolerant of markdown fences (```json ... ```), bare objects, and
    leading/trailing chatter. Returns {} on unparseable input — caller
    treats that as "Documenter degraded" and falls back to the legacy
    body. Same shape as critic_repo._parse_judge_response.
    """
    if not text:
        return {}
    stripped = text.strip()
    # Prefer an explicit ```json fence — Grok follows the prompt instruction
    # most of the time but occasionally emits a bare object.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        # Fall back to first {...} balanced-ish span.
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first == -1 or last <= first:
            return {}
        candidate = stripped[first : last + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        logger.warning("documenter response not parseable as JSON")
    return {}


def _fallback_title(brief: str) -> str:
    """Legacy pre-20a title shape: first non-empty line, capped at 70 chars.

    The push activity used to format `Swarm: <one-line>...` here; we keep
    that prefix so the fallback PR title still reads like a swarm-authored
    PR (operators filter PRs by this prefix in some dashboards).
    """
    if not brief:
        return "Swarm: automated change"
    one_line = " ".join(brief.split())
    title = f"Swarm: {one_line}"
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 1].rstrip() + "…"
    return title


def _fallback_body(brief: str, files_changed: list[str]) -> str:
    """Legacy pre-20a body shape: brief verbatim + flat file list.

    Identical structure to what `push_repo_changes_activity` emitted
    before this sprint, so a Documenter failure is invisible to anyone
    watching PRs day-to-day.
    """
    files = files_changed or []
    body = (
        "Automated change from twai-swarm (Documenter degraded; see logs).\n\n"
        f"## Brief\n\n{brief}\n\n"
        f"## Files changed ({len(files)})\n\n"
    )
    if files:
        body += "\n".join(f"- `{f}`" for f in files)
    else:
        body += "(no files reported)"
    return body


async def run_documenter_repo(
    brief: str,
    architect_plan: dict | None,
    coder_diff: str,
    files_changed: list[str],
    critic_result: dict | None,
) -> DocumenterRepoOutput:
    """Run xAI Grok to generate PR title + body from the workflow artifacts.

    Cheap call: ~5K input tokens (brief + plan + truncated diff + critic) +
    ~1K output tokens (PR title + body). Estimated $0.001-0.005 per
    workflow at grok-fast pricing ($0.20/$0.50 per Mtok).

    Failure mode: degrades to a fallback PR body (just the brief + file
    list, like pre-20a) so the workflow ships with the same UX as before
    if Documenter fails. The activity caller writes the result to its
    return dict regardless, so the workflow always has SOMETHING to thread
    into push_repo_changes_activity.
    """
    user_message = _build_user_message(
        brief=brief,
        architect_plan=architect_plan,
        coder_diff=coder_diff,
        files_changed=files_changed,
        critic_result=critic_result,
    )

    try:
        # Plain Chat-Completions path — no server-side tools needed. The
        # xai_provider auto-wraps in observability.generation() so this
        # call nests under the documenter agent_span set up by the
        # activity wrapper.
        result = await xai_provider.complete(
            model=DOCUMENTER_MODEL,
            system=DOCUMENTER_REPO_SYSTEM_PROMPT,
            user=user_message,
            max_tokens=MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "documenter_repo xAI call failed: %s; using fallback PR body", e,
        )
        return DocumenterRepoOutput(
            pr_title=_fallback_title(brief),
            pr_body=_fallback_body(brief, files_changed),
        )

    parsed = _parse_json_response(result.text)
    pr_title_raw = parsed.get("pr_title") or ""
    pr_body_raw = parsed.get("pr_body") or ""
    if not pr_title_raw or not pr_body_raw:
        logger.warning(
            "documenter_repo response missing pr_title/pr_body "
            "(title_len=%d body_len=%d); falling back",
            len(pr_title_raw), len(pr_body_raw),
        )
        return DocumenterRepoOutput(
            pr_title=_fallback_title(brief),
            pr_body=_fallback_body(brief, files_changed),
            _tokens_in=int(getattr(result, "tokens_in", 0) or 0),
            _tokens_out=int(getattr(result, "tokens_out", 0) or 0),
        )

    # grok-fast pricing per 1M tokens (matches router.MODELS["grok-fast"]).
    tokens_in = int(getattr(result, "tokens_in", 0) or 0)
    tokens_out = int(getattr(result, "tokens_out", 0) or 0)
    cost_usd = round(
        tokens_in * 0.20 / 1_000_000 + tokens_out * 0.50 / 1_000_000,
        6,
    )

    title = str(pr_title_raw).strip()
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 1].rstrip() + "…"

    return DocumenterRepoOutput(
        pr_title=title,
        pr_body=str(pr_body_raw),
        _tokens_in=tokens_in,
        _tokens_out=tokens_out,
        _cost_usd=cost_usd,
    )


__all__ = [
    "DOCUMENTER_MODEL",
    "DocumenterRepoOutput",
    "DOCUMENTER_REPO_SYSTEM_PROMPT",
    "MAX_DIFF_CHARS",
    "MAX_TITLE_CHARS",
    "MAX_TOKENS",
    "_EMIT_PR_TOOL_SCHEMA",
    "_build_user_message",
    "_fallback_body",
    "_fallback_title",
    "_parse_json_response",
    "_truncate_diff",
    "run_documenter_repo",
]
# Silence unused-import lint for the typing field import (no field-level
# defaults needed yet but keeping the import keeps the dataclass extensible
# without re-importing later).
_ = field
