"""Repo-aware Reviewer activity — Sprint 18d.

Two-stage Best-of-N selection. Runs AFTER K parallel Coders have all
finished against the SAME Architect plan (different seeds/temperatures).
Picks one winner; the workflow then sends only the winner through the
Critic loop and the auto-PR step.

Design (per `sprint-18-plan.md` §"Architectural decisions"):
  - D5 — Two-stage selection (AlphaCode 2 pattern):
      Stage 1: deterministic gate filter — drops candidates that fail
        ruff/compileall/mvn/typecheck. Implemented by overlaying each
        candidate's `files_with_content` onto a temp scratch dir copy
        of the cloned repo, then re-using `critic_repo.run_deterministic_gate`.
      Stage 2: pairwise LLM-as-judge with position-swap. For each pair
        of survivors, the judge is asked TWICE (A-first, then B-first);
        agreement under swap is recorded as `position_swap_consistent=True`.
        Disagreement falls back to a deterministic tie-break (lower index
        wins) so two reproducible runs of the same data pick the same
        winner — important for replay debugging.
  - D6 — Best-of-N is gated on the Architect's `cross_cutting: bool`.
    The Reviewer itself doesn't enforce this; the workflow decides
    whether to call Reviewer at all. This module is gate-agnostic.
  - D8 — Reviewer uses Sonnet 4.6, NOT Haiku — different model than the
    Coder, mitigates the self-enhancement bias documented in the
    LLM-as-judge survey (arxiv 2411.15594).

Performance note (V1 simplicity over efficiency):
  Each candidate triggers a full `shutil.copytree` of the cloned repo
  (excluding `.git/`) into a per-candidate scratch dir before its
  deterministic gate runs. For a 10K-file repo that's a meaningful per-
  candidate cost (~seconds of disk I/O × 3 candidates). The simpler
  alternative — apply candidate to the original cloned repo, run gate,
  revert — risks corrupting the workspace if any gate process holds a
  file handle when we revert; full copytree is correct by construction.
  See sprint-18-plan.md "Out of scope" — caching the scratch tree across
  candidates (rsync-style overlay) is a Sprint 19 candidate if profiling
  shows this is the bottleneck. For now: correct first, fast later.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anthropic import AsyncAnthropic

from app import config

from .critic_repo import GateFailure, run_deterministic_gate

logger = logging.getLogger(__name__)


# Per D8: Reviewer uses Sonnet 4.6, distinct from the Coder's Haiku.
# Hardcoded for the same reason as ARCHITECT_MODEL / CRITIC_MODEL — this
# is a single call site; threading the router for one role is overkill.
REVIEWER_MODEL = "claude-sonnet-4-6"

# Pairwise judge call is non-streaming and bounded (~1 turn per pair-side).
# 1K tokens of output is plenty for the {"winner": "A", "rationale": "..."}
# shape; we cap modestly so a runaway judge can't blow the budget.
MAX_TOKENS_JUDGE = 1024

# Cap on how many characters of each candidate diff we paste into the
# pairwise judge prompt. 8K chars per side keeps the prompt under ~20K
# tokens total (plus the architect plan + acceptance criteria) — a safe
# headroom for Sonnet's input window. Bigger diffs get truncated with a
# marker line so the judge knows it's seeing a fragment.
DIFF_PROMPT_CAP_CHARS = 8000


# Verbatim — same prompt is shipped on every pairwise call. Position-swap
# protection is achieved by swapping which diff is labelled A vs B in the
# user message, NOT by changing this system prompt between calls.
JUDGE_SYSTEM_PROMPT = """You are a code-diff judge. You will be shown two diffs (A and B) that both attempt to satisfy the same brief.

The brief includes acceptance criteria from an Architect plan. Your job is to pick the diff that better satisfies the brief, with these rubric items in this priority order:

1. Brief satisfaction: which diff addresses MORE of the acceptance criteria?
2. Scope discipline: which diff stays within the planned files? (a diff that touches files not mentioned in the plan is suspect)
3. Regression risk: which diff is less likely to break existing functionality? (look for changes to public APIs, removed code, suspicious deletions)
4. Code style consistency: which diff better matches the existing codebase patterns?

CRITICAL: do NOT favor longer diffs. Verbosity is not quality. A surgical 20-line diff that addresses 5 criteria is better than a 200-line diff that addresses 3.

Output your answer as JSON:
{
  "winner": "A" | "B",
  "rationale": "<one-line explanation>"
}
"""


@dataclass
class CandidateAssessment:
    """Per-candidate assessment after Stage 1 (deterministic gate).

    Fields are all primitives or lists of primitives so `dataclasses.asdict`
    produces a JSON-serializable dict (Temporal activity invariant).
    """
    candidate_index: int
    seed: int
    deterministic_gate_passed: bool
    # GateFailure dicts (CriticRepoOutput-compatible — same shape so the
    # UI cost-summary can render them through the existing critic
    # template if it cares).
    gate_failures: list[dict] = field(default_factory=list)
    files_changed_count: int = 0
    diff_size_bytes: int = 0


@dataclass
class PairwiseResult:
    """One pairwise judge result (after both A-first and B-first calls)."""
    candidate_a: int
    candidate_b: int
    winner: int                       # which candidate index won this pair
    rationale: str
    # True if both A-first and B-first calls picked the same winner.
    # False signals the judge is uncertain on this pair; we still pick a
    # winner via deterministic tie-break (lower index) so the run is
    # reproducible, but the consistency flag surfaces in the rationale.
    position_swap_consistent: bool = True


@dataclass
class ReviewerRepoOutput:
    """Reviewer's deliverable for one Best-of-N selection.

    Mirrors the provenance fields used by the other repo-task agents
    (`_model`, `_provider`, `_tokens_in`, `_tokens_out`, `_cost_usd`)
    so the cost-summary card aggregates without a special case.
    """
    # Index into the input candidates list. None when all candidates
    # failed the deterministic gate AND the fallback was disabled (we
    # never disable in practice — see fallback_used).
    winner_index: int | None
    candidate_assessments: list[CandidateAssessment] = field(default_factory=list)
    pairwise_results: list[PairwiseResult] = field(default_factory=list)
    rationale: str = ""
    # True if every candidate failed the deterministic gate; the winner
    # was picked as the one with the FEWEST gate failures ("least bad").
    # The downstream Critic loop will still flag the gate failures and
    # likely fire continuation passes — fallback_used just means the
    # selection wasn't grounded in a clean baseline.
    fallback_used: bool = False
    # Provenance — leading underscore matches the Architect/Critic convention.
    _model: str = REVIEWER_MODEL
    _provider: str = "anthropic"
    _tokens_in: int = 0
    _tokens_out: int = 0
    _cost_usd: float = 0.0


# ─── Stage 1: per-candidate scratch dir + gate ──────────────────────────────


def _apply_candidate_to_scratch_dir(
    repo_root: Path, candidate_files: list[dict],
) -> Path:
    """Copy `repo_root` into a tmp scratch dir + overlay candidate files.

    Returns the absolute path to the scratch dir. Caller is responsible
    for `shutil.rmtree`-ing it in a `finally` block.

    `.git/` is excluded from the copytree because we don't need git
    metadata for the deterministic gate (gates run on file contents),
    and copying it on a large repo is the dominant cost.

    Per the module-level "Performance note": this is full-tree copy per
    candidate. Correct by construction; expensive on large repos. Cache
    optimisation deferred to a future sprint.
    """
    scratch = Path(tempfile.mkdtemp(prefix="reviewer-cand-"))
    # `dirs_exist_ok=False` (the default) is correct here — mkdtemp made
    # the dir; copytree wants to recreate it, so we need to remove first.
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.copytree(
        repo_root, scratch,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    # Overlay each candidate file.
    for f in candidate_files or []:
        path = f.get("path") if isinstance(f, dict) else None
        content = f.get("content") if isinstance(f, dict) else None
        if not path or content is None:
            continue
        target = scratch / path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except (OSError, UnicodeError) as e:
            logger.warning(
                "skipping overlay for %s in scratch dir: %s", path, e,
            )
    return scratch


def _assess_candidate(
    repo_root: Path, candidate: dict, candidate_index: int, seed: int,
) -> CandidateAssessment:
    """Run the deterministic gate on a single candidate's files.

    Builds a scratch dir, overlays the candidate's `files_with_content`,
    runs the gate, computes the assessment, then ALWAYS removes the
    scratch dir (success or failure) so we don't leak temp space.
    """
    files_with_content = candidate.get("files_with_content") or []
    files_changed = [
        str(f.get("path", ""))
        for f in files_with_content if isinstance(f, dict) and f.get("path")
    ]
    diff = candidate.get("diff") or ""
    diff_size = len(diff.encode("utf-8")) if isinstance(diff, str) else 0

    scratch: Path | None = None
    try:
        scratch = _apply_candidate_to_scratch_dir(repo_root, files_with_content)
        gate_passed, gate_failures = run_deterministic_gate(scratch, files_changed)
    except Exception as e:  # noqa: BLE001 — gate is best-effort by design.
        logger.warning(
            "candidate %d: deterministic gate threw (%s); treating as failed",
            candidate_index, e,
        )
        gate_passed = False
        gate_failures = [
            GateFailure(
                tool="(unknown)", file="(scratch-dir)", line=None,
                message=f"reviewer scratch-dir gate exception: {type(e).__name__}",
            ),
        ]
    finally:
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    return CandidateAssessment(
        candidate_index=candidate_index,
        seed=seed,
        deterministic_gate_passed=gate_passed,
        gate_failures=[asdict(gf) for gf in gate_failures],
        files_changed_count=len(files_changed),
        diff_size_bytes=diff_size,
    )


# ─── Stage 2: pairwise LLM-as-judge ─────────────────────────────────────────


def _truncate_diff(diff: str, cap: int = DIFF_PROMPT_CAP_CHARS) -> str:
    """Cap diff length so the prompt stays under Sonnet's input budget.

    Truncated with a marker line so the judge knows it's seeing a fragment
    (not the full diff). Keeps the head — the diff's structure is more
    informative early than late.
    """
    if not isinstance(diff, str):
        return ""
    if len(diff) <= cap:
        return diff
    truncated = diff[:cap]
    extra_lines = diff[cap:].count("\n") + 1
    return f"{truncated}\n... [diff continues, {extra_lines} more lines]"


def _build_judge_user_message(
    architect_plan: dict, diff_a: str, diff_b: str,
) -> str:
    """Render the user-side message for one pairwise judge call.

    The system prompt is verbatim across calls (see JUDGE_SYSTEM_PROMPT);
    position-swap protection comes from swapping which candidate's diff
    is labelled A vs B in this user message.
    """
    parts: list[str] = []
    narrative = (architect_plan.get("narrative") or "").strip() if isinstance(architect_plan, dict) else ""
    if narrative:
        parts.append("## Architect plan narrative")
        parts.append(narrative)
        parts.append("")
    # Pull every acceptance criterion from every subtask and present as
    # a flat numbered list — same shape the Critic uses, so the judge
    # sees the same checklist.
    criteria: list[str] = []
    if isinstance(architect_plan, dict):
        for subtask in (architect_plan.get("subtasks") or []):
            if not isinstance(subtask, dict):
                continue
            for c in (subtask.get("acceptance_criteria") or []):
                if c:
                    criteria.append(str(c))
    if criteria:
        parts.append("## Acceptance criteria")
        for i, c in enumerate(criteria):
            parts.append(f"{i + 1}. {c}")
        parts.append("")

    parts.append("## Diff A")
    parts.append("```diff")
    parts.append(_truncate_diff(diff_a) or "(empty diff)")
    parts.append("```")
    parts.append("")
    parts.append("## Diff B")
    parts.append("```diff")
    parts.append(_truncate_diff(diff_b) or "(empty diff)")
    parts.append("```")
    parts.append("")
    parts.append(
        "Pick A or B per the rubric. Return JSON: "
        '{"winner": "A" | "B", "rationale": "<one-line>"}'
    )
    return "\n".join(parts)


def _parse_judge_winner(text: str) -> tuple[str | None, str]:
    """Parse the judge's JSON response to (winner_letter, rationale).

    Returns (None, error_msg) if the response can't be salvaged. Caller
    treats parse failure as "judge uncertain on this pair" and falls back
    to the deterministic tie-break.
    """
    if not text:
        return (None, "(empty judge response)")
    stripped = text.strip()
    # Strip markdown code fences if present.
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Last-ditch: extract first {...} block.
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last > first:
            try:
                parsed = json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                return (None, "judge response not parseable as JSON")
        else:
            return (None, "judge response had no JSON object")

    if not isinstance(parsed, dict):
        return (None, "judge JSON was not an object")
    winner = str(parsed.get("winner", "")).strip().upper()
    rationale = str(parsed.get("rationale", "")).strip() or "(no rationale)"
    if winner not in {"A", "B"}:
        return (None, f"unrecognised winner value {winner!r}")
    return (winner, rationale)


async def _call_judge(
    client: AsyncAnthropic, architect_plan: dict, diff_a: str, diff_b: str,
) -> tuple[str | None, str, int, int]:
    """One pairwise judge API call. Returns (winner_letter, rationale, t_in, t_out).

    On API failure returns (None, "<error>", 0, 0); caller treats as
    pair uncertainty and falls back to deterministic tie-break.
    """
    user_message = _build_judge_user_message(architect_plan, diff_a, diff_b)
    try:
        resp = await client.messages.create(
            model=REVIEWER_MODEL,
            max_tokens=MAX_TOKENS_JUDGE,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("reviewer judge call failed: %s", e)
        return (None, f"judge api error: {type(e).__name__}", 0, 0)

    text_parts: list[str] = []
    for block in (resp.content or []):
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
    winner, rationale = _parse_judge_winner("".join(text_parts))
    tokens_in = int(getattr(resp.usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(resp.usage, "output_tokens", 0) or 0)
    return (winner, rationale, tokens_in, tokens_out)


async def _pairwise_compare(
    client: AsyncAnthropic,
    architect_plan: dict,
    candidates: list[dict],
    idx_a: int, idx_b: int,
) -> tuple[PairwiseResult, int, int]:
    """Compare one pair of candidates with position-swap.

    Calls the judge twice: first with `idx_a` as A, then with `idx_b` as A.
    Agreement → position_swap_consistent=True. Disagreement → tie-break:
    pick the LOWER candidate index (deterministic for replay).

    Returns (PairwiseResult, total_tokens_in, total_tokens_out).
    """
    diff_a = candidates[idx_a].get("diff") or ""
    diff_b = candidates[idx_b].get("diff") or ""

    # First call: A=candidate idx_a, B=candidate idx_b
    winner1, rationale1, t_in1, t_out1 = await _call_judge(
        client, architect_plan, diff_a, diff_b,
    )
    # Second call: positions swapped — A=candidate idx_b, B=candidate idx_a
    winner2, rationale2, t_in2, t_out2 = await _call_judge(
        client, architect_plan, diff_b, diff_a,
    )

    # Translate each call's "A" or "B" letter back to a candidate index.
    pick1 = None
    if winner1 == "A":
        pick1 = idx_a
    elif winner1 == "B":
        pick1 = idx_b
    pick2 = None
    if winner2 == "A":
        pick2 = idx_b  # positions were swapped
    elif winner2 == "B":
        pick2 = idx_a

    if pick1 is not None and pick2 is not None and pick1 == pick2:
        # Both calls agreed — high-confidence pair result.
        return (
            PairwiseResult(
                candidate_a=idx_a,
                candidate_b=idx_b,
                winner=pick1,
                rationale=rationale1,
                position_swap_consistent=True,
            ),
            t_in1 + t_in2,
            t_out1 + t_out2,
        )

    # Disagreement (or one/both calls failed). Deterministic tie-break:
    # the lower-indexed candidate wins. Reproducibility > judge confidence
    # in this scenario — operators replaying the same data should get the
    # same winner.
    chosen = min(idx_a, idx_b)
    rationale = (
        f"position-swap disagreed (A-first picked {pick1}, B-first picked {pick2}); "
        f"deterministic tie-break to lower index ({chosen}). "
        f"A-first rationale: {rationale1}; B-first rationale: {rationale2}"
    )
    return (
        PairwiseResult(
            candidate_a=idx_a,
            candidate_b=idx_b,
            winner=chosen,
            rationale=rationale,
            position_swap_consistent=False,
        ),
        t_in1 + t_in2,
        t_out1 + t_out2,
    )


# ─── Selection logic ────────────────────────────────────────────────────────


def _pick_winner_from_pairwise(
    survivors: list[int],
    pairwise_results: list[PairwiseResult],
) -> int:
    """Tally pairwise wins across `survivors`; return the winning index.

    Tie-break: most position-swap-consistent wins (more confident wins
    prevail). Final tie-break: lowest survivor index (deterministic).
    """
    win_count: dict[int, int] = {idx: 0 for idx in survivors}
    consistent_win_count: dict[int, int] = {idx: 0 for idx in survivors}
    for pr in pairwise_results:
        if pr.winner in win_count:
            win_count[pr.winner] += 1
            if pr.position_swap_consistent:
                consistent_win_count[pr.winner] += 1
    # Sort by (-wins, -consistent_wins, +index) so max win count wins,
    # ties broken by consistent wins, then by lowest index.
    ranked = sorted(
        survivors,
        key=lambda idx: (
            -win_count[idx],
            -consistent_win_count[idx],
            idx,
        ),
    )
    return ranked[0]


# ─── Coordinator ────────────────────────────────────────────────────────────


async def run_reviewer_repo(
    architect_plan: dict | None,
    candidates: list[dict],
    repo_root: Path,
) -> ReviewerRepoOutput:
    """Two-stage Best-of-N selection over `candidates`.

    `candidates` is a list of Coder result dicts (the shape produced by
    `run_agentic_repo_coder` — has `diff`, `files_changed`,
    `files_with_content`, `_tokens_in`, etc).

    Returns a `ReviewerRepoOutput` (dataclass; caller serializes via
    `dataclasses.asdict` for Temporal-friendly hand-off). On any
    catastrophic failure (no candidates, every candidate malformed) the
    output's `winner_index` is None — the workflow then falls back to
    candidate 0 (the most-deterministic seed) so we still ship something.
    """
    if not candidates:
        return ReviewerRepoOutput(
            winner_index=None,
            rationale="no candidates supplied to reviewer",
            fallback_used=False,
        )

    # ─── Stage 1: deterministic gate on every candidate ──────────────────
    assessments: list[CandidateAssessment] = []
    for i, cand in enumerate(candidates):
        # The Coder's result dict doesn't carry the seed it was run with
        # (the seed is a workflow-side parameter). We accept it via an
        # explicit `_coder_seed` key the workflow tags onto each result
        # before calling the Reviewer; fall back to the candidate index
        # so older callers / replay still get a reasonable seed value.
        seed = int(cand.get("_coder_seed", i)) if isinstance(cand, dict) else i
        assessments.append(_assess_candidate(repo_root, cand, i, seed))

    survivors = [
        a.candidate_index for a in assessments if a.deterministic_gate_passed
    ]

    # All candidates failed gate — fall back to "least bad" (fewest failures).
    if not survivors:
        # Pick by ascending (gate_failure_count, candidate_index) so the
        # cleanest run wins; ties broken by lower index for reproducibility.
        ranked = sorted(
            assessments,
            key=lambda a: (len(a.gate_failures), a.candidate_index),
        )
        winner = ranked[0]
        rationale = (
            f"All {len(candidates)} candidates failed deterministic gate; "
            f"selected candidate {winner.candidate_index} with fewest "
            f"failures ({len(winner.gate_failures)}) as least-bad option."
        )
        return ReviewerRepoOutput(
            winner_index=winner.candidate_index,
            candidate_assessments=assessments,
            pairwise_results=[],
            rationale=rationale,
            fallback_used=True,
        )

    # Exactly 1 survives — no pairwise needed.
    if len(survivors) == 1:
        winner_idx = survivors[0]
        return ReviewerRepoOutput(
            winner_index=winner_idx,
            candidate_assessments=assessments,
            pairwise_results=[],
            rationale=f"Only candidate {winner_idx} passed deterministic gate.",
            fallback_used=False,
        )

    # ─── Stage 2: pairwise judge on every pair of survivors ──────────────
    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
    architect_plan_dict = architect_plan if isinstance(architect_plan, dict) else {}

    pairwise_results: list[PairwiseResult] = []
    total_tokens_in = 0
    total_tokens_out = 0
    # Iterate pairs in sorted (i, j) order with i < j — deterministic.
    for i_pos in range(len(survivors)):
        for j_pos in range(i_pos + 1, len(survivors)):
            idx_a = survivors[i_pos]
            idx_b = survivors[j_pos]
            pr, t_in, t_out = await _pairwise_compare(
                client, architect_plan_dict, candidates, idx_a, idx_b,
            )
            pairwise_results.append(pr)
            total_tokens_in += t_in
            total_tokens_out += t_out

    winner_idx = _pick_winner_from_pairwise(survivors, pairwise_results)
    win_count = sum(1 for pr in pairwise_results if pr.winner == winner_idx)
    rationale = (
        f"Selected candidate {winner_idx} as Best-of-N winner: "
        f"{win_count} pairwise wins out of {len(pairwise_results)} pairs "
        f"among {len(survivors)} survivors."
    )

    # Sonnet 4.6 pricing per 1M tokens (matches router.MODELS["sonnet"]).
    input_cost = total_tokens_in * 3.0 / 1_000_000
    output_cost = total_tokens_out * 15.0 / 1_000_000

    return ReviewerRepoOutput(
        winner_index=winner_idx,
        candidate_assessments=assessments,
        pairwise_results=pairwise_results,
        rationale=rationale,
        fallback_used=False,
        _model=REVIEWER_MODEL,
        _provider="anthropic",
        _tokens_in=total_tokens_in,
        _tokens_out=total_tokens_out,
        _cost_usd=round(input_cost + output_cost, 6),
    )


def reviewer_output_to_dict(out: ReviewerRepoOutput) -> dict:
    """Helper: convert ReviewerRepoOutput to a plain dict.

    Equivalent to `dataclasses.asdict(out)`. Wraps it as a helper so
    callers (the activity, tests) reach for the same canonical conversion
    (mirrors `architect_output_to_dict` / `critic_output_to_dict`).
    """
    return asdict(out)


__all__ = [
    "CandidateAssessment",
    "DIFF_PROMPT_CAP_CHARS",
    "JUDGE_SYSTEM_PROMPT",
    "MAX_TOKENS_JUDGE",
    "PairwiseResult",
    "REVIEWER_MODEL",
    "ReviewerRepoOutput",
    "_apply_candidate_to_scratch_dir",
    "_assess_candidate",
    "_build_judge_user_message",
    "_call_judge",
    "_pairwise_compare",
    "_parse_judge_winner",
    "_pick_winner_from_pairwise",
    "_truncate_diff",
    "reviewer_output_to_dict",
    "run_reviewer_repo",
]
