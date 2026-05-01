"""Tests for the workflow heuristics module — Sprint 18.1.

`infer_cross_cutting_from_brief` is a pure function (no I/O, no clock,
no random) so it can run inside the Temporal sandbox. Tests cover the
threshold + each signal type independently, plus the Refresh Tokens
brief that motivated the Sprint 18.1 fix.
"""
from __future__ import annotations

from app.workflows._heuristics import infer_cross_cutting_from_brief


def test_empty_brief_is_not_cross_cutting():
    """Defensive: empty / whitespace-only briefs short-circuit to False
    so we never trigger Best-of-N on a no-op input."""
    assert infer_cross_cutting_from_brief("") is False
    assert infer_cross_cutting_from_brief(None) is False  # type: ignore[arg-type]


def test_single_numbered_item_is_not_cross_cutting():
    """One numbered ask isn't enough — threshold is >=4 signals."""
    brief = "1. Add a /healthz endpoint that returns 200."
    assert infer_cross_cutting_from_brief(brief) is False


def test_four_numbered_items_flips_cross_cutting():
    """Four numbered items alone clears the >=4 signal threshold."""
    brief = (
        "Please:\n"
        "1. Add a /healthz endpoint.\n"
        "2. Add a /readyz endpoint.\n"
        "3. Wire both into the FastAPI app.\n"
        "4. Add a smoke test for both.\n"
    )
    assert infer_cross_cutting_from_brief(brief) is True


def test_five_distinct_file_paths_flips_cross_cutting():
    """Mentioning >=4 distinct file paths is a strong cross-cutting signal."""
    brief = (
        "Update auth flow. Touch app/auth/routes.py, app/auth/jwt.py, "
        "app/auth/models.py, frontend/LoginPage.tsx and tests/test_auth.py."
    )
    assert infer_cross_cutting_from_brief(brief) is True


def test_repeated_file_path_counts_once():
    """File mentions are set-deduped — naming routes.py twice is 1 signal,
    not 2. Otherwise a chatty brief that re-mentions the same file would
    falsely trigger Best-of-N."""
    brief = (
        "Edit routes.py. Then edit routes.py again. Then once more "
        "edit routes.py. (Only one numbered item: 1. do the edit.)"
    )
    # 1 numbered + 1 distinct file = 2 signals < 4.
    assert infer_cross_cutting_from_brief(brief) is False


def test_three_integration_tests_plus_two_numbered_items_flips():
    """Mixed signals add up: 3 test asks + 2 numbered items = 5 signals."""
    brief = (
        "Add the following:\n"
        "1. POST /auth/refresh endpoint.\n"
        "2. JWT helper for refresh tokens.\n"
        "Plus an integration test for the happy path, "
        "an integration test for expired tokens, and "
        "an integration test for revoked tokens."
    )
    assert infer_cross_cutting_from_brief(brief) is True


def test_code_blocks_count_as_signals():
    """Each fenced code block (paired ``` markers) counts once."""
    brief = (
        "Implement these:\n"
        "```py\ndef a(): ...\n```\n"
        "```py\ndef b(): ...\n```\n"
        "```py\ndef c(): ...\n```\n"
        "```py\ndef d(): ...\n```\n"
    )
    # 4 fenced blocks = 4 signals = at threshold.
    assert infer_cross_cutting_from_brief(brief) is True


def test_short_typo_fix_is_not_cross_cutting():
    """The "1-line typo fix is 1 subtask" calibration: prose-only brief
    with no list, no files mentioned, no code blocks → False."""
    brief = "Fix the typo in the welcome message — 'Helo' should be 'Hello'."
    assert infer_cross_cutting_from_brief(brief) is False


def test_refresh_tokens_brief_flips_cross_cutting():
    """The actual Refresh Tokens brief from run 019de315 — the failure
    case that motivated Sprint 18.1. Must come back True so the
    workflow's heuristic fallback triggers Best-of-N."""
    brief = (
        "Add refresh token support to the calculator app's auth flow.\n"
        "\n"
        "1. Add a new POST /auth/refresh endpoint in app/auth/routes.py "
        "that accepts a refresh token and returns a fresh access JWT.\n"
        "2. Add a refresh-token issuer helper in app/auth/jwt.py "
        "(separate signing key from the access token).\n"
        "3. Update app/auth/models.py with a RefreshToken model so "
        "tokens can be revoked individually.\n"
        "4. Wire frontend/LoginPage.tsx to call /auth/refresh when an "
        "API call returns 401.\n"
        "\n"
        "Tests:\n"
        "- An integration test that exercises the happy refresh path.\n"
        "- An integration test that confirms expired refresh tokens "
        "are rejected with 401.\n"
        "- An integration test that confirms revoked tokens cannot be "
        "exchanged for new access JWTs.\n"
    )
    assert infer_cross_cutting_from_brief(brief) is True


def test_threshold_boundary_is_exclusive_below():
    """Exactly 3 signals → False; exactly 4 → True. Locks the threshold."""
    three_signal_brief = (
        "1. one\n2. two\n3. three\n"
    )
    assert infer_cross_cutting_from_brief(three_signal_brief) is False

    four_signal_brief = (
        "1. one\n2. two\n3. three\n4. four\n"
    )
    assert infer_cross_cutting_from_brief(four_signal_brief) is True
