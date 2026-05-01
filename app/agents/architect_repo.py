"""Repo-aware Architect activity — Sprint 18b.

Pre-step before the agentic Coder. Mirrors the role-split that the
greenfield ProjectWorkflow already does (BA → Architect → SE → Coder)
into the repo-task pipeline: a *planning-only* agent that runs against
an indexed repo and produces a structured plan + acceptance-criteria
checklist for the downstream Coder to consume.

Design (per `sprint-18-plan.md` §"Architectural decisions"):
  - D1 — Architect emits free-form prose (the "how I'd solve this"
    narrative) AND a structured `subtasks: list[Subtask]` DAG. The
    prose is load-bearing (Aider-style); the DAG is what the Critic
    later checks against.
  - D6 — Architect tags the brief as `cross_cutting: bool` based on a
    coarse heuristic (≥4 subtasks OR ≥4 files OR multiple subsystems).
    Sprint 18d's Best-of-N branch reads this flag to decide whether to
    run K=3 parallel Coders.
  - D7 — Output is a structured handoff document (not a chat
    transcript), persisted via the Temporal activity result so Coder
    runs N+1 (Sprint 18c continuation loop) can consume it cleanly.
  - D8 — Architect uses Sonnet 4.6, NOT Haiku. Planning is reasoning-
    heavy; the Coder stays on Haiku for cost discipline.
  - D9 — All agent outputs persist via the same JSONB pattern as the
    greenfield Tasks table. (For repo-task v1 the Architect output flows
    via the workflow result; full Tasks-row persistence lands in 18d.)

The Architect has the SAME graph tools as the Coder (`repo_search`,
`repo_find_callers`, etc.) but **NO write tools** (no `write_file`, no
`bash_exec`, no `run_verify`). It is a pure planner.

Returns an `ArchitectRepoOutput` dataclass that round-trips through
`dataclasses.asdict` → `json.dumps` (Temporal serialization invariant).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from app import config
from .coder_sandbox import Sandbox
from .coder_tools import build_tools

logger = logging.getLogger(__name__)

# Architect runs many fewer turns than the Coder — it's investigating,
# not editing. Originally 8 (enough to call repo_find_modules +
# repo_find_processes + a handful of repo_search / repo_find_callers
# drill-ins, then emit the plan). Sprint 18.1 telemetry showed 8 was
# too tight for cross-cutting briefs (Refresh Tokens A/B run 019de315):
# the model kept investigating and never reached emit_plan, leaving the
# Critic + Best-of-N safety nets to vacuously pass against an empty
# subtask list. Bumped to 15 — still well under the Coder's 30 — and
# paired with a force-emit-plan fallback below to harden the worst case.
MAX_ARCHITECT_ITERATIONS = 15
MAX_TOKENS_PER_TURN = 16384

# Per D8: Architect uses Sonnet 4.6, distinct from the Coder's Haiku.
# Hardcoded (not via the router) because the Architect is the only call
# site for Sonnet inside the repo-task pipeline; threading routing for
# a single role is scope-creep. When 18c's Critic / 18d's Reviewer also
# use Sonnet we can revisit and unify through the router.
ARCHITECT_MODEL = "claude-sonnet-4-6"

# Tool names that perform writes / side effects. The Architect is a pure
# planner per D1 — strip these from the Coder tool surface before passing
# the list to the tool_runner. Enumerated explicitly (rather than "everything
# graph-prefixed") so a future graph-side write tool can't sneak in.
_WRITE_TOOL_NAMES = frozenset({
    "write_file", "bash_exec", "run_verify",
})


ARCHITECT_REPO_SYSTEM_PROMPT = """You are an Architect for an existing-code repo-task workflow. Your role is PURE PLANNING — you must NOT edit code. You will be replaced by a Coder agent who reads your plan and edits files.

Your inputs:
- A task brief (what the user wants done).
- A "Repo recon" block with high-value graph signals (modules, processes) about the repo.
- A set of graph tools (repo_search, repo_find_callers, repo_find_definition, repo_find_processes, repo_semantic_search) for repo investigation.

Your output is a STRUCTURED PLAN with two parts:

1. A free-form narrative (prose, ~300-1500 words). Reason out loud about how you'd solve the brief. Identify the affected files, the existing patterns you'll mirror, the risks. This is the most important part — the Coder reads it as authoritative scope.

2. A structured subtask DAG. Each subtask has:
   - id (e.g. "backend.refresh_endpoint")
   - description (1-3 sentences)
   - files_to_touch (repo-relative paths; empty list if uncertain)
   - depends_on (subtask IDs that must complete first; usually empty)
   - independent (can it run in parallel with other independents — defaults true)
   - acceptance_criteria (testable statements; the Critic will verify each)

Plus:
- cross_cutting (bool): true if ≥4 subtasks OR ≥4 files OR spans ≥2 subsystems (backend+frontend, prod+test)
- risk_notes: list of things the Coder should be careful about (e.g. "this method has 8 callers — don't change its signature")

CRITICAL constraints:
- Do NOT edit code. You have no write tools.
- Do NOT speculate about file contents you haven't read. Use repo_find_definition / repo_search to inspect.
- Be calibrated: a 1-line typo fix is 1 subtask, not 4. A cross-cutting refactor with backend + frontend + tests is typically 5-10 subtasks.
- acceptance_criteria are CONTRACTS: the Coder will be checked against them. Make them concrete and testable.
- BEFORE proposing a subtask that touches an existing function, use repo_find_callers to understand the blast radius. Mention it in risk_notes.
- Output JSON via the structured tool-use format the SDK enforces. The narrative goes in the `narrative` field; subtasks go in `subtasks`.

When you have enough information to write the plan, call the `emit_plan` tool with the full structured output. That call ends your turn — do not continue investigating after emitting.

CALIBRATION — iteration budget:
You have 15 iterations of investigation. By iteration 12, you should be in 'commit mode' — call emit_plan even if your plan is rough. An imperfect plan is better than no plan, because the Critic and Reviewer downstream depend on your subtasks + acceptance_criteria. Investigating to iteration 15 without emitting leaves the safety-net pipeline with nothing to grade against.
"""


@dataclass
class Subtask:
    """One unit of work in the Architect's plan.

    Fields are all primitives or lists of primitives so `dataclasses.asdict`
    produces a JSON-serializable dict (Temporal activity invariant).
    """
    id: str
    description: str
    files_to_touch: list[str]
    depends_on: list[str] = field(default_factory=list)
    independent: bool = True
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class ArchitectRepoOutput:
    """Architect's deliverable for one repo-task brief.

    Mirrors the provenance fields used by the greenfield agents
    (`_model`, `_provider`, `_tokens_in`, `_tokens_out`, `_cost_usd`)
    so the cost-summary card aggregates without a special case.
    """
    narrative: str
    subtasks: list[Subtask] = field(default_factory=list)
    cross_cutting: bool = False
    risk_notes: list[str] = field(default_factory=list)
    # Provenance — leading underscore matches the convention in
    # coder_repo.run_agentic_repo_coder's return dict.
    _model: str = ARCHITECT_MODEL
    _provider: str = "anthropic"
    _tokens_in: int = 0
    _tokens_out: int = 0
    _cost_usd: float = 0.0


def _planning_only_tools(tools: list, stats: dict) -> list:
    """Strip write/exec tools from the Coder tool surface.

    The Coder's `build_tools` returns ~14 tools (graph + sandbox). The
    Architect needs the graph tools but NOTHING that mutates the disk
    or runs commands — per D1 the Architect is a pure planner.
    """
    kept = []
    for t in tools:
        # `@beta_async_tool` wraps the function but keeps `.name` on the
        # tool object. Fall back to `__name__` on the underlying callable
        # in case the SDK's internals change.
        name = getattr(t, "name", None) or getattr(t, "__name__", "")
        if name in _WRITE_TOOL_NAMES:
            continue
        kept.append(t)
    return kept


def _build_architect_user_message(
    brief: str, repo_name: str, recon_block: str = "",
) -> str:
    """Render the user-side message the Architect sees.

    Recon-block-first ordering mirrors `coder_repo._build_user_message`
    so the model sees the map before the task — important for repos it
    has no prior context on.
    """
    parts: list[str] = []
    if recon_block:
        parts.append(recon_block)
    parts.append(f"## Task brief\n{brief.strip()}")
    parts.append(
        f"## Repo\nThe repository `{repo_name}` is already cloned and indexed. Investigate it via the graph tools (repo_find_modules / repo_find_processes / repo_search / repo_find_definition / repo_find_callers / repo_semantic_search). Do NOT attempt to edit files — you have no write tools.\n\n"
        f"When you have enough information, call the `emit_plan` tool with your structured plan."
    )
    return "\n\n".join(parts)


# JSON schema for the `emit_plan` tool. Anthropic's tool-use enforces this
# shape via the `input_schema` field, which is how we get a typed JSON
# output from the model without parsing free-form text. Mirrors the
# `ArchitectRepoOutput` dataclass + nested `Subtask` shape.
_EMIT_PLAN_TOOL_SCHEMA = {
    "name": "emit_plan",
    "description": (
        "Emit the final structured architecture plan. Call this exactly "
        "ONCE when you have finished investigating the repo and are ready "
        "to hand off to the Coder. After calling this tool, your turn ends."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": (
                    "Free-form prose (~300-1500 words) explaining how you'd "
                    "solve the brief. The Coder reads this as authoritative scope."
                ),
            },
            "subtasks": {
                "type": "array",
                "description": "Structured subtask DAG.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "files_to_touch": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "depends_on": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "independent": {"type": "boolean"},
                        "acceptance_criteria": {
                            "type": "array", "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "description", "files_to_touch"],
                },
            },
            "cross_cutting": {
                "type": "boolean",
                "description": (
                    "True if ≥4 subtasks OR ≥4 files OR spans ≥2 subsystems "
                    "(backend+frontend, prod+test). Used by Sprint 18d's "
                    "Best-of-N gating."
                ),
            },
            "risk_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Things the Coder should be careful about (e.g. "
                    "\"this method has 8 callers — don't change its signature\")."
                ),
            },
        },
        "required": ["narrative", "subtasks", "cross_cutting"],
    },
}


def _coerce_subtasks(raw: Any) -> list[Subtask]:
    """Defensive: coerce the model-emitted subtasks list into Subtask dataclasses.

    The tool schema enforces the shape, but the SDK delivers it as plain
    dicts. This adds default values for optional fields (`depends_on`,
    `independent`, `acceptance_criteria`) when the model omits them.
    """
    out: list[Subtask] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(Subtask(
                id=str(item.get("id", "")),
                description=str(item.get("description", "")),
                files_to_touch=[str(p) for p in (item.get("files_to_touch") or [])],
                depends_on=[str(d) for d in (item.get("depends_on") or [])],
                independent=bool(item.get("independent", True)),
                acceptance_criteria=[
                    str(c) for c in (item.get("acceptance_criteria") or [])
                ],
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping malformed subtask %r: %s", item, e)
    return out


async def _force_emit_plan(
    client: AsyncAnthropic,
    original_user_message: str,
    last_text: str,
) -> dict | None:
    """Force the Architect to emit a plan when the iteration cap was hit.

    Sprint 18.1 fallback. The main runner ran out of investigation
    iterations without calling emit_plan (Refresh Tokens A/B run
    019de315 failure mode). Rather than hand the workflow an empty
    subtask list and let the Critic/Best-of-N silently no-op, we make
    one more Sonnet call with `tool_choice` pinned to emit_plan — the
    model MUST return the structured payload, even if the plan is
    rough. An imperfect plan is still better than no plan because the
    downstream safety nets need acceptance_criteria to grade against.

    Returns the emit_plan tool_use input dict (with extra
    `_forced_tokens_in` / `_forced_tokens_out` keys for accounting), or
    None if the forced call also fails — in which case the caller
    proceeds to the existing degraded path (empty subtasks, narrative =
    last_text).
    """
    deadline_message = (
        "Time's up. You have used your investigation budget without "
        "calling emit_plan. You must emit your plan now with whatever "
        "you have learned. Do not investigate further. Call emit_plan "
        "with your best draft of subtasks (even if incomplete) and set "
        "cross_cutting based on the brief — if the brief touches "
        "multiple subsystems or asks for >=4 distinct items, set "
        "cross_cutting=True. Use your prior reasoning where possible:\n\n"
        f"{(last_text or '(no prior reasoning captured)')[:4000]}"
    )
    try:
        forced_response = await client.messages.create(
            model=ARCHITECT_MODEL,
            max_tokens=MAX_TOKENS_PER_TURN,
            system=ARCHITECT_REPO_SYSTEM_PROMPT,
            tools=[_EMIT_PLAN_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "emit_plan"},
            messages=[
                {"role": "user", "content": original_user_message},
                {"role": "user", "content": deadline_message},
            ],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("architect force-emit-plan fallback failed: %s", e)
        return None

    forced_in = int(getattr(forced_response.usage, "input_tokens", 0) or 0)
    forced_out = int(getattr(forced_response.usage, "output_tokens", 0) or 0)
    for block in (forced_response.content or []):
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "emit_plan"
        ):
            payload = dict(getattr(block, "input", None) or {})
            payload["_forced_tokens_in"] = forced_in
            payload["_forced_tokens_out"] = forced_out
            logger.info(
                "architect force-emit-plan succeeded: %d subtasks, "
                "cross_cutting=%s",
                len(payload.get("subtasks") or []),
                bool(payload.get("cross_cutting", False)),
            )
            return payload
    logger.warning(
        "architect force-emit-plan returned no emit_plan tool_use block",
    )
    return None


async def run_architect_repo(
    workflow_id: str,
    repo_root: Path,
    repo_name: str,
    brief: str,
    neo4j_driver: Any,
    recon_block: str = "",
    heartbeat: Any = None,
    tenant_id: str = "default",
) -> ArchitectRepoOutput:
    """Run the Architect agent over an indexed repo + brief.

    Returns an `ArchitectRepoOutput` (dataclass; caller serializes via
    `dataclasses.asdict` for Temporal-friendly hand-off to the Coder).

    On any tool-runner failure (transient API hiccup, malformed model
    output, MAX_ITERATIONS hit without an emit_plan call), returns a
    degraded ArchitectRepoOutput with the model's last text as narrative
    and an empty subtask list — the workflow then falls through to the
    existing "no plan" Coder path. Failure is non-fatal.
    """
    from app import observability

    sandbox = Sandbox.wrap(repo_root)
    full_tools, stats = build_tools(
        sandbox, neo4j_driver=neo4j_driver, repo_name=repo_name,
    )
    tools = _planning_only_tools(full_tools, stats)
    # Append the structured-output tool. Forcing tool_choice on it would
    # block investigation turns (the SDK requires the chosen tool be
    # called immediately); leaving tool_choice unset lets the model use
    # graph tools first, then call emit_plan when ready.
    tools_with_emit = list(tools) + [_EMIT_PLAN_TOOL_SCHEMA]

    user_message = _build_architect_user_message(
        brief, repo_name, recon_block=recon_block,
    )

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
    initial_user_message = {"role": "user", "content": user_message}
    runner_kwargs: dict = dict(
        model=ARCHITECT_MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=ARCHITECT_REPO_SYSTEM_PROMPT,
        tools=tools_with_emit,
        messages=[initial_user_message],
    )
    runner = client.beta.messages.tool_runner(**runner_kwargs)

    iterations = 0
    total_input_tokens = 0
    total_output_tokens = 0
    last_text = ""
    plan_payload: dict | None = None

    tenant_ctx = observability.tenant_scope(tenant_id)
    tenant_ctx.__enter__()
    try:
        async for message in runner:
            iterations += 1
            if heartbeat is not None:
                try:
                    heartbeat(
                        f"architect iteration {iterations}/"
                        f"{MAX_ARCHITECT_ITERATIONS}"
                    )
                except Exception:
                    pass
            if getattr(message, "usage", None) is not None:
                total_input_tokens += int(
                    getattr(message.usage, "input_tokens", 0) or 0
                )
                total_output_tokens += int(
                    getattr(message.usage, "output_tokens", 0) or 0
                )
            for block in (message.content or []):
                btype = getattr(block, "type", None)
                if btype == "text":
                    t = getattr(block, "text", "") or ""
                    if t.strip():
                        last_text = t
                elif btype == "tool_use" and getattr(block, "name", None) == "emit_plan":
                    plan_payload = getattr(block, "input", None) or {}
            # Once the model emits its plan we have everything we need;
            # break out of the runner so we don't burn another turn on a
            # courtesy ack message.
            if plan_payload is not None:
                break
            if iterations >= MAX_ARCHITECT_ITERATIONS:
                logger.warning(
                    "architect hit MAX_ARCHITECT_ITERATIONS=%d without "
                    "emit_plan; falling back to last_text",
                    MAX_ARCHITECT_ITERATIONS,
                )
                break
    except Exception as e:  # noqa: BLE001
        logger.error(
            "architect tool_runner failed at iteration %d: %s",
            iterations, e,
        )
    finally:
        tenant_ctx.__exit__(None, None, None)

    # Sprint 18.1 force-emit-plan fallback. If the runner terminated
    # without the model ever calling emit_plan (iteration cap hit, or
    # transient API failure mid-investigation), fire one final stateless
    # call to Sonnet with `tool_choice` pinned to emit_plan. The SDK's
    # tool_runner doesn't expose mid-stream tool_choice manipulation
    # cleanly, and replaying the runner's private message history is
    # brittle — so we send a fresh, minimal conversation: the original
    # brief + a deadline instruction. The model has no escape hatch
    # because tool_choice="emit_plan" forces the structured response.
    # This guarantees the Critic + Best-of-N safety nets downstream
    # always have a plan to grade against, even on a degraded run.
    if plan_payload is None:
        plan_payload = await _force_emit_plan(
            client, user_message, last_text,
        )
        if plan_payload is not None:
            # Token accounting for the forced call lives inside the helper
            # and bumps these counters via the returned payload's
            # `_forced_tokens_in` / `_forced_tokens_out` keys (popped before
            # the payload is treated as plan content).
            total_input_tokens += int(plan_payload.pop("_forced_tokens_in", 0))
            total_output_tokens += int(plan_payload.pop("_forced_tokens_out", 0))

    # Sonnet 4.6 pricing per 1M tokens (matches router.MODELS["sonnet"]).
    input_cost = total_input_tokens * 3.0 / 1_000_000
    output_cost = total_output_tokens * 15.0 / 1_000_000

    if plan_payload is not None:
        narrative = str(plan_payload.get("narrative") or last_text or "")
        subtasks = _coerce_subtasks(plan_payload.get("subtasks"))
        cross_cutting = bool(plan_payload.get("cross_cutting", False))
        risk_notes = [str(r) for r in (plan_payload.get("risk_notes") or [])]
    else:
        # Degraded path — model didn't emit_plan in time. Hand back
        # whatever prose it managed; downstream Coder still gets the
        # original brief verbatim.
        narrative = (last_text or "(architect produced no plan)").strip()
        subtasks = []
        cross_cutting = False
        risk_notes = []

    return ArchitectRepoOutput(
        narrative=narrative,
        subtasks=subtasks,
        cross_cutting=cross_cutting,
        risk_notes=risk_notes,
        _model=ARCHITECT_MODEL,
        _provider="anthropic",
        _tokens_in=total_input_tokens,
        _tokens_out=total_output_tokens,
        _cost_usd=round(input_cost + output_cost, 6),
    )


def architect_output_to_dict(out: ArchitectRepoOutput) -> dict:
    """Helper: convert ArchitectRepoOutput to a plain dict.

    Equivalent to `dataclasses.asdict(out)` but called out as a helper so
    callers (the activity, tests) reach for the same canonical conversion.
    """
    return asdict(out)


def render_architect_plan_section(plan: dict) -> str:
    """Render the Architect's plan as a markdown section for the Coder.

    Pulled out as a top-level helper so both the Coder's user-message
    builder AND the 18c continuation-loop handoff doc share one
    canonical format. Returns "" when `plan` is None / empty so callers
    can `if section: prepend(...)` without a branch.
    """
    if not plan or not isinstance(plan, dict):
        return ""
    lines: list[str] = ["## Architect plan"]
    narrative = (plan.get("narrative") or "").strip()
    if narrative:
        lines.append("")
        lines.append(narrative)
    subtasks = plan.get("subtasks") or []
    if subtasks:
        lines.append("")
        lines.append(f"### Subtasks ({len(subtasks)})")
        for st in subtasks:
            if not isinstance(st, dict):
                continue
            sid = st.get("id", "")
            desc = (st.get("description") or "").strip()
            files = st.get("files_to_touch") or []
            files_str = ", ".join(f"`{f}`" for f in files) if files else "(uncertain)"
            lines.append(f"- **{sid}** — {desc}")
            lines.append(f"  - files: {files_str}")
            criteria = st.get("acceptance_criteria") or []
            if criteria:
                lines.append("  - acceptance:")
                for c in criteria:
                    lines.append(f"    - {c}")
    cross = plan.get("cross_cutting")
    if cross is not None:
        lines.append("")
        lines.append(f"_cross_cutting: {bool(cross)}_")
    return "\n".join(lines)


def render_risk_section(plan: dict) -> str:
    """Render the Architect's risk_notes as a separate markdown section.

    Empty plan / no risk_notes returns "" so callers can skip injection.
    """
    if not plan or not isinstance(plan, dict):
        return ""
    risks = plan.get("risk_notes") or []
    if not risks:
        return ""
    lines: list[str] = ["## Risks"]
    for r in risks:
        lines.append(f"- {r}")
    return "\n".join(lines)


# Re-export json so callers that want to round-trip the dataclass through
# Temporal's JSON-only payload don't need to import it themselves.
__all__ = [
    "ARCHITECT_MODEL",
    "ARCHITECT_REPO_SYSTEM_PROMPT",
    "ArchitectRepoOutput",
    "MAX_ARCHITECT_ITERATIONS",
    "Subtask",
    "architect_output_to_dict",
    "render_architect_plan_section",
    "render_risk_section",
    "run_architect_repo",
]
_ = json  # silence unused-import lint until a caller needs it
