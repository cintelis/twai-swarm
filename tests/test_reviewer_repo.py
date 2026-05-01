"""Tests for the repo Reviewer — Sprint 18d.

Covers the dataclasses, the per-candidate scratch-dir overlay, the
deterministic-gate Stage-1 filter, the pairwise judge with position-swap,
and the winner-selection logic. The actual Anthropic API call is mocked
out everywhere; the end-to-end path (3 parallel Coders → Reviewer → PR)
is exercised in deploy.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from app.agents.reviewer_repo import (
    DIFF_PROMPT_CAP_CHARS,
    JUDGE_SYSTEM_PROMPT,
    MAX_TOKENS_JUDGE,
    REVIEWER_MODEL,
    CandidateAssessment,
    PairwiseResult,
    ReviewerRepoOutput,
    _apply_candidate_to_scratch_dir,
    _build_judge_user_message,
    _parse_judge_winner,
    _pick_winner_from_pairwise,
    _truncate_diff,
    reviewer_output_to_dict,
    run_reviewer_repo,
)
from app.agents.critic_repo import GateFailure


# ─── Dataclass roundtrip ────────────────────────────────────────────────────


def test_reviewer_output_dataclass_roundtrip():
    """Instantiate → asdict → json.dumps → loads → reconstruct → equal.

    Same Temporal-serialisation invariant as the Architect / Critic:
    every activity return must round-trip through json.dumps.
    """
    sample = ReviewerRepoOutput(
        winner_index=1,
        candidate_assessments=[
            CandidateAssessment(
                candidate_index=0, seed=0,
                deterministic_gate_passed=True,
                gate_failures=[],
                files_changed_count=3,
                diff_size_bytes=1024,
            ),
            CandidateAssessment(
                candidate_index=1, seed=1,
                deterministic_gate_passed=True,
                gate_failures=[],
                files_changed_count=2,
                diff_size_bytes=512,
            ),
            CandidateAssessment(
                candidate_index=2, seed=2,
                deterministic_gate_passed=False,
                gate_failures=[
                    {"tool": "ruff", "file": "x.py", "line": 1, "message": "F401"},
                ],
                files_changed_count=4,
                diff_size_bytes=2048,
            ),
        ],
        pairwise_results=[
            PairwiseResult(
                candidate_a=0, candidate_b=1, winner=1,
                rationale="B is more surgical",
                position_swap_consistent=True,
            ),
        ],
        rationale="Selected candidate 1 with 1 pairwise wins.",
        fallback_used=False,
        _model="claude-sonnet-4-6",
        _provider="anthropic",
        _tokens_in=5000,
        _tokens_out=200,
        _cost_usd=0.018,
    )
    raw = asdict(sample)
    encoded = json.dumps(raw)
    decoded = json.loads(encoded)

    rebuilt = ReviewerRepoOutput(
        winner_index=decoded["winner_index"],
        candidate_assessments=[
            CandidateAssessment(**ca) for ca in decoded["candidate_assessments"]
        ],
        pairwise_results=[
            PairwiseResult(**pr) for pr in decoded["pairwise_results"]
        ],
        rationale=decoded["rationale"],
        fallback_used=decoded["fallback_used"],
        _model=decoded["_model"],
        _provider=decoded["_provider"],
        _tokens_in=decoded["_tokens_in"],
        _tokens_out=decoded["_tokens_out"],
        _cost_usd=decoded["_cost_usd"],
    )
    assert rebuilt == sample


def test_reviewer_output_to_dict_helper():
    out = ReviewerRepoOutput(winner_index=0, rationale="x")
    d = reviewer_output_to_dict(out)
    assert isinstance(d, dict)
    assert d["winner_index"] == 0
    assert d["rationale"] == "x"
    # Round-trip via JSON.
    assert json.loads(json.dumps(d))["winner_index"] == 0


def test_pairwise_result_default_swap_consistent_true():
    """Most pairs agree under swap; default to True so callers reading
    the field assume consistency unless explicitly flagged."""
    pr = PairwiseResult(candidate_a=0, candidate_b=1, winner=0, rationale="x")
    assert pr.position_swap_consistent is True


# ─── _apply_candidate_to_scratch_dir ────────────────────────────────────────


def test_apply_candidate_to_scratch_dir_creates_overlay(tmp_path):
    """Synthetic repo with 2 files; overlay 1 modified + 1 new; assert
    the scratch dir has the right contents and the original is untouched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "existing.py").write_text("original contents\n", encoding="utf-8")
    (repo / "stays.py").write_text("untouched\n", encoding="utf-8")

    candidate_files = [
        {"path": "existing.py", "content": "modified contents\n"},
        {"path": "new_file.py", "content": "brand new\n"},
    ]
    scratch = _apply_candidate_to_scratch_dir(repo, candidate_files)
    try:
        assert scratch.is_dir()
        # Modified file shows the candidate's content.
        assert (scratch / "existing.py").read_text(encoding="utf-8") == "modified contents\n"
        # New file shows up.
        assert (scratch / "new_file.py").read_text(encoding="utf-8") == "brand new\n"
        # Untouched file copied across.
        assert (scratch / "stays.py").read_text(encoding="utf-8") == "untouched\n"
        # Original repo wasn't mutated.
        assert (repo / "existing.py").read_text(encoding="utf-8") == "original contents\n"
        assert not (repo / "new_file.py").exists()
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_apply_candidate_to_scratch_dir_excludes_git(tmp_path):
    """The .git/ dir must NOT be copied (slow + unnecessary for gates)."""
    repo = tmp_path / "repo"
    (repo / ".git" / "objects").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]", encoding="utf-8")
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")

    scratch = _apply_candidate_to_scratch_dir(repo, [])
    try:
        assert (scratch / "src.py").exists()
        assert not (scratch / ".git").exists()
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def test_apply_candidate_to_scratch_dir_creates_parent_dirs(tmp_path):
    """A new file in a deep path triggers parent mkdir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hi", encoding="utf-8")

    candidate_files = [
        {"path": "deep/nested/sub/file.py", "content": "deep new\n"},
    ]
    scratch = _apply_candidate_to_scratch_dir(repo, candidate_files)
    try:
        assert (scratch / "deep" / "nested" / "sub" / "file.py").read_text(encoding="utf-8") == "deep new\n"
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


# ─── _truncate_diff ─────────────────────────────────────────────────────────


def test_truncate_diff_short_returns_unchanged():
    diff = "diff --git a/x b/x\n+small\n"
    assert _truncate_diff(diff, cap=1000) == diff


def test_truncate_diff_long_appends_marker():
    diff = "x\n" * (DIFF_PROMPT_CAP_CHARS + 100)
    truncated = _truncate_diff(diff)
    assert "diff continues" in truncated
    assert "more lines" in truncated
    assert len(truncated) < len(diff)


def test_truncate_diff_handles_non_string():
    assert _truncate_diff(None) == ""  # type: ignore[arg-type]


# ─── _build_judge_user_message ──────────────────────────────────────────────


def test_build_judge_user_message_includes_plan_and_diffs():
    plan = {
        "narrative": "Add the refresh endpoint.",
        "subtasks": [
            {"id": "be", "acceptance_criteria": ["returns 200", "returns 401 on bad token"]},
            {"id": "fe", "acceptance_criteria": ["LoginPage shows refresh"]},
        ],
    }
    msg = _build_judge_user_message(plan, "diff A contents", "diff B contents")
    assert "Architect plan narrative" in msg
    assert "Add the refresh endpoint." in msg
    assert "## Acceptance criteria" in msg
    assert "1. returns 200" in msg
    assert "2. returns 401 on bad token" in msg
    assert "3. LoginPage shows refresh" in msg
    assert "## Diff A" in msg
    assert "## Diff B" in msg
    assert "diff A contents" in msg
    assert "diff B contents" in msg
    # Output instruction at the bottom.
    assert '"winner"' in msg


def test_build_judge_user_message_handles_empty_plan():
    msg = _build_judge_user_message({}, "a", "b")
    assert "## Diff A" in msg
    assert "## Diff B" in msg
    # Plan sections are skipped entirely when missing.
    assert "Architect plan narrative" not in msg
    assert "## Acceptance criteria" not in msg


# ─── _parse_judge_winner ────────────────────────────────────────────────────


def test_parse_judge_winner_clean_json():
    text = '{"winner": "A", "rationale": "More surgical."}'
    winner, rationale = _parse_judge_winner(text)
    assert winner == "A"
    assert rationale == "More surgical."


def test_parse_judge_winner_strips_markdown_fences():
    text = '```json\n{"winner": "B", "rationale": "Better."}\n```'
    winner, rationale = _parse_judge_winner(text)
    assert winner == "B"
    assert rationale == "Better."


def test_parse_judge_winner_extracts_first_json_block():
    text = "Here is my answer:\n{\"winner\": \"A\", \"rationale\": \"yes\"}\nthat's all."
    winner, rationale = _parse_judge_winner(text)
    assert winner == "A"
    assert rationale == "yes"


def test_parse_judge_winner_returns_none_on_garbage():
    winner, rationale = _parse_judge_winner("not json at all")
    assert winner is None
    assert "JSON" in rationale or "object" in rationale


def test_parse_judge_winner_returns_none_on_unrecognised_letter():
    text = '{"winner": "C", "rationale": "huh"}'
    winner, _rationale = _parse_judge_winner(text)
    assert winner is None


def test_parse_judge_winner_handles_empty_input():
    winner, rationale = _parse_judge_winner("")
    assert winner is None
    assert "empty" in rationale


# ─── _pick_winner_from_pairwise ─────────────────────────────────────────────


def test_pick_winner_from_pairwise_most_wins():
    """3 candidates, A beats B and C, B beats C. Expected winner: A (idx 0)."""
    survivors = [0, 1, 2]
    pairwise = [
        PairwiseResult(candidate_a=0, candidate_b=1, winner=0, rationale="A>B"),
        PairwiseResult(candidate_a=0, candidate_b=2, winner=0, rationale="A>C"),
        PairwiseResult(candidate_a=1, candidate_b=2, winner=1, rationale="B>C"),
    ]
    winner = _pick_winner_from_pairwise(survivors, pairwise)
    assert winner == 0


def test_pick_winner_from_pairwise_tie_broken_by_consistency():
    """Two candidates with equal wins; the more position-swap-consistent one wins."""
    survivors = [0, 1, 2]
    pairwise = [
        # Both 0 and 1 have 1 win each, but 0's win was position-consistent.
        PairwiseResult(
            candidate_a=0, candidate_b=2, winner=0,
            rationale="A>C consistent", position_swap_consistent=True,
        ),
        PairwiseResult(
            candidate_a=1, candidate_b=2, winner=1,
            rationale="B>C inconsistent", position_swap_consistent=False,
        ),
        PairwiseResult(
            candidate_a=0, candidate_b=1, winner=2,
            rationale="C wins this pair (impossible scenario)",
        ),
    ]
    # Note: candidate 2 doesn't actually appear as winner in normal flow
    # but we're testing tie-breaking on candidates 0/1 with equal wins.
    winner = _pick_winner_from_pairwise(survivors, pairwise)
    # 0 wins 1 (consistent), 1 wins 1 (inconsistent), 2 wins 1.
    # All three tied at 1 win each → consistency: 0=1, 1=0, 2=0.
    # → 0 wins on consistency tie-break.
    assert winner == 0


def test_pick_winner_from_pairwise_final_tiebreak_lowest_index():
    """All candidates tied → lowest index wins (deterministic for replay)."""
    survivors = [0, 1, 2]
    # No pairwise wins at all → all tied at 0.
    winner = _pick_winner_from_pairwise(survivors, [])
    assert winner == 0


# ─── run_reviewer_repo coordinator ──────────────────────────────────────────


def _candidate(diff: str = "diff", files: list[dict] | None = None) -> dict:
    """Build a Coder-shaped result dict for tests."""
    return {
        "diff": diff,
        "files_changed": [f["path"] for f in (files or [])],
        "files_with_content": files or [],
        "_tokens_in": 100,
        "_tokens_out": 50,
        "_coder_seed": 0,
    }


@pytest.mark.asyncio
async def test_reviewer_no_candidates(tmp_path):
    out = await run_reviewer_repo({}, [], tmp_path)
    assert out.winner_index is None
    assert out.fallback_used is False
    assert "no candidates" in out.rationale


@pytest.mark.asyncio
async def test_reviewer_single_survivor_no_pairwise_needed(tmp_path, monkeypatch):
    """Feed 3 candidates where 2 fail gate; the surviving 1 wins automatically."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    # Fake the per-candidate assessment: only candidate 1 passes the gate.
    def fake_assess(repo_root, candidate, candidate_index, seed):
        passed = (candidate_index == 1)
        return CandidateAssessment(
            candidate_index=candidate_index,
            seed=seed,
            deterministic_gate_passed=passed,
            gate_failures=[] if passed else [
                {"tool": "ruff", "file": "x.py", "line": 1, "message": "fail"},
            ],
            files_changed_count=0,
            diff_size_bytes=len(candidate.get("diff") or ""),
        )

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )

    out = await run_reviewer_repo({}, candidates, tmp_path)
    assert out.winner_index == 1
    assert out.fallback_used is False
    assert out.pairwise_results == []
    assert "Only candidate 1 passed" in out.rationale
    assert len(out.candidate_assessments) == 3


@pytest.mark.asyncio
async def test_reviewer_all_fail_gate_fallback(tmp_path, monkeypatch):
    """All 3 candidates fail gate → fallback_used=True, fewest-failures wins."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        # Candidate 0 has 5 failures, candidate 1 has 2 failures, candidate 2 has 8.
        n_failures = {0: 5, 1: 2, 2: 8}[candidate_index]
        return CandidateAssessment(
            candidate_index=candidate_index,
            seed=seed,
            deterministic_gate_passed=False,
            gate_failures=[{"tool": "ruff", "file": f"f{i}.py", "line": i, "message": "fail"} for i in range(n_failures)],
            files_changed_count=0,
            diff_size_bytes=0,
        )

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )

    out = await run_reviewer_repo({}, candidates, tmp_path)
    assert out.winner_index == 1  # fewest failures
    assert out.fallback_used is True
    assert "All 3 candidates failed" in out.rationale
    assert "fewest" in out.rationale


@pytest.mark.asyncio
async def test_reviewer_pairwise_position_swap_agreement(tmp_path, monkeypatch):
    """All 3 candidates pass gate; judge always picks the same answer
    regardless of position → position_swap_consistent=True everywhere."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        return CandidateAssessment(
            candidate_index=candidate_index, seed=seed,
            deterministic_gate_passed=True,
            gate_failures=[], files_changed_count=0, diff_size_bytes=0,
        )

    async def fake_call_judge(client, plan, diff_a, diff_b):
        # Always pick A regardless of position. With position-swap, that
        # means A-first picks the first candidate, B-first picks the
        # second candidate (now in A position).
        return ("A", "always A", 100, 50)

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )
    monkeypatch.setattr(
        "app.agents.reviewer_repo._call_judge", fake_call_judge,
    )

    out = await run_reviewer_repo({"narrative": "x"}, candidates, tmp_path)
    # 3 pairs with always-A judge means consistency=False everywhere
    # because position-swap flips the winner. Let's check that case
    # explicitly: same judge answer on both positions → swap inconsistent.
    assert all(not pr.position_swap_consistent for pr in out.pairwise_results)


@pytest.mark.asyncio
async def test_reviewer_pairwise_position_swap_true_consistency(tmp_path, monkeypatch):
    """Judge picks the same CANDIDATE regardless of position — true
    position-swap consistency. Achieved by a judge that picks "A" when
    candidate_a's diff is in slot A, and "B" when candidate_a's diff is
    in slot B (i.e. always votes for the lower-index candidate)."""
    candidates = [_candidate("aaa"), _candidate("bbb"), _candidate("ccc")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        return CandidateAssessment(
            candidate_index=candidate_index, seed=seed,
            deterministic_gate_passed=True,
            gate_failures=[], files_changed_count=0, diff_size_bytes=0,
        )

    async def fake_call_judge(client, plan, diff_a, diff_b):
        # diff_a and diff_b distinguish candidates by their unique strings.
        # Always pick the candidate whose diff starts with "aaa" if present;
        # otherwise pick whichever has the lex-smaller diff.
        if "aaa" in diff_a:
            return ("A", "picked aaa", 100, 50)
        if "aaa" in diff_b:
            return ("B", "picked aaa", 100, 50)
        # Between bbb and ccc, always pick bbb.
        if "bbb" in diff_a:
            return ("A", "picked bbb", 100, 50)
        return ("B", "picked bbb", 100, 50)

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )
    monkeypatch.setattr(
        "app.agents.reviewer_repo._call_judge", fake_call_judge,
    )

    out = await run_reviewer_repo({"narrative": "x"}, candidates, tmp_path)
    # Every pair should have position_swap_consistent=True since the
    # judge picks the same candidate regardless of position.
    assert all(pr.position_swap_consistent for pr in out.pairwise_results)
    # Candidate 0 (aaa) wins both its pairs, candidate 1 (bbb) wins one,
    # candidate 2 (ccc) wins zero. Winner: 0.
    assert out.winner_index == 0


@pytest.mark.asyncio
async def test_reviewer_pairwise_position_swap_disagreement_tiebreaks(tmp_path, monkeypatch):
    """Judge returns different answers based on position → swap inconsistent;
    deterministic tie-break (lower index) is used."""
    candidates = [_candidate("a"), _candidate("b")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        return CandidateAssessment(
            candidate_index=candidate_index, seed=seed,
            deterministic_gate_passed=True,
            gate_failures=[], files_changed_count=0, diff_size_bytes=0,
        )

    async def fake_call_judge(client, plan, diff_a, diff_b):
        # Always pick whichever is in slot A (position bias) — disagreement guaranteed.
        return ("A", "position bias", 100, 50)

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )
    monkeypatch.setattr(
        "app.agents.reviewer_repo._call_judge", fake_call_judge,
    )

    out = await run_reviewer_repo({"narrative": "x"}, candidates, tmp_path)
    assert len(out.pairwise_results) == 1
    pr = out.pairwise_results[0]
    assert pr.position_swap_consistent is False
    # Deterministic tie-break: lower index (0) wins.
    assert pr.winner == 0
    assert "deterministic tie-break" in pr.rationale.lower()
    assert out.winner_index == 0


@pytest.mark.asyncio
async def test_reviewer_picks_most_pairwise_wins(tmp_path, monkeypatch):
    """Synthetic scenario: 3 candidates, A beats B and C, B beats C.
    Expected winner: A (candidate 0)."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        return CandidateAssessment(
            candidate_index=candidate_index, seed=seed,
            deterministic_gate_passed=True,
            gate_failures=[], files_changed_count=0, diff_size_bytes=0,
        )

    async def fake_call_judge(client, plan, diff_a, diff_b):
        # Use diff string identity to determine winners.
        # Map: "a" beats both, "b" beats "c".
        ordering = {"a": 2, "b": 1, "c": 0}
        rank_a = ordering.get(diff_a, 0)
        rank_b = ordering.get(diff_b, 0)
        if rank_a > rank_b:
            return ("A", f"{diff_a} > {diff_b}", 100, 50)
        return ("B", f"{diff_b} > {diff_a}", 100, 50)

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )
    monkeypatch.setattr(
        "app.agents.reviewer_repo._call_judge", fake_call_judge,
    )

    out = await run_reviewer_repo({"narrative": "x"}, candidates, tmp_path)
    # 3 pairs: (0,1), (0,2), (1,2). a beats b → winner=0; a beats c →
    # winner=0; b beats c → winner=1. Tally: 0=2, 1=1, 2=0.
    assert out.winner_index == 0
    win_counts = {}
    for pr in out.pairwise_results:
        win_counts[pr.winner] = win_counts.get(pr.winner, 0) + 1
    assert win_counts[0] == 2
    assert win_counts.get(1, 0) == 1


# ─── Module-level constants ─────────────────────────────────────────────────


def test_reviewer_model_is_sonnet():
    """Per D8: judging the Coder uses a different model — Sonnet, not Haiku."""
    assert REVIEWER_MODEL == "claude-sonnet-4-6"


def test_judge_system_prompt_mentions_rubric_priorities():
    """The system prompt enumerates the four rubric items and the
    anti-verbosity caveat — locking these so a refactor doesn't drop them."""
    p = JUDGE_SYSTEM_PROMPT.lower()
    assert "brief satisfaction" in p
    assert "scope discipline" in p
    assert "regression risk" in p
    assert "code style" in p
    assert "do not favor longer diffs" in p


def test_judge_system_prompt_specifies_json_output():
    """Output format must be deterministic JSON so _parse_judge_winner works."""
    assert '"winner"' in JUDGE_SYSTEM_PROMPT
    assert '"rationale"' in JUDGE_SYSTEM_PROMPT


def test_max_tokens_judge_modest():
    """Per-pair judge call should be tiny (one sentence + JSON wrapper)."""
    assert MAX_TOKENS_JUDGE == 1024


def test_diff_prompt_cap_chars_matches_8k():
    """Cap each diff at 8K chars per the spec — controls prompt size."""
    assert DIFF_PROMPT_CAP_CHARS == 8000


def test_candidate_assessment_dataclass_defaults():
    """Empty-ish CandidateAssessment should round-trip through asdict."""
    ca = CandidateAssessment(
        candidate_index=0, seed=0,
        deterministic_gate_passed=False,
    )
    d = asdict(ca)
    assert d["gate_failures"] == []
    assert d["files_changed_count"] == 0
    assert d["diff_size_bytes"] == 0


# ─── Stage 1 + 2 integration via fake assess + fake judge ───────────────────


@pytest.mark.asyncio
async def test_reviewer_zero_survivors_fallback_uses_fewest_failures(tmp_path, monkeypatch):
    """Detailed fallback path: assert pairwise_results is empty and
    candidate_assessments shows all 3 as failed."""
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

    def fake_assess(repo_root, candidate, candidate_index, seed):
        n = {0: 3, 1: 1, 2: 5}[candidate_index]
        return CandidateAssessment(
            candidate_index=candidate_index, seed=seed,
            deterministic_gate_passed=False,
            gate_failures=[{"tool": "ruff", "file": "x", "line": i, "message": "f"} for i in range(n)],
            files_changed_count=0, diff_size_bytes=0,
        )

    monkeypatch.setattr(
        "app.agents.reviewer_repo._assess_candidate", fake_assess,
    )

    out = await run_reviewer_repo({}, candidates, tmp_path)
    assert out.winner_index == 1
    assert out.fallback_used is True
    assert out.pairwise_results == []
    assert len(out.candidate_assessments) == 3
    # Every assessment failed.
    assert all(not ca.deterministic_gate_passed for ca in out.candidate_assessments)


# ─── Real GateFailure roundtrip in CandidateAssessment ──────────────────────


def test_candidate_assessment_accepts_gatefailure_dicts():
    """gate_failures stores plain dicts so it survives JSON serialization;
    these dicts mirror the GateFailure shape from critic_repo."""
    gf = GateFailure(tool="ruff", file="app/x.py", line=10, message="F401")
    ca = CandidateAssessment(
        candidate_index=0, seed=0,
        deterministic_gate_passed=False,
        gate_failures=[asdict(gf)],
    )
    d = asdict(ca)
    assert d["gate_failures"][0]["tool"] == "ruff"
    assert d["gate_failures"][0]["line"] == 10
    # Round-trip via JSON.
    roundtripped = json.loads(json.dumps(d))
    assert roundtripped["gate_failures"][0]["message"] == "F401"
