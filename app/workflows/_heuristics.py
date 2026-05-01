"""Pure heuristics for repo-task workflow classification — Sprint 18.1.

Kept module-level (not inside the workflow class) because Temporal
workflows are sandboxed; helpers must be importable via
`workflow.unsafe.imports_passed_through()` so they can be invoked from
inside a `@workflow.defn` body without violating determinism rules.

These helpers must stay PURE (no I/O, no clock reads, no random numbers)
for the same reason — Temporal replays workflow code from the event
history and any non-determinism causes a NonDeterministicError.
"""
from __future__ import annotations

import re

# A line starting with "1.", "2)", "3:" etc. — the canonical "TODO list"
# shape humans use in briefs. Anchored to MULTILINE start-of-line to
# avoid matching numbers inside paragraphs.
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\):]\s+", re.MULTILINE)
# Triple-backticks counted as "this is one fenced section" via //2 below.
_CODE_BLOCK_RE = re.compile(r"```", re.MULTILINE)
# A path-like token followed by a known source-file extension. Required
# leading word-boundary so "fooSomething.py" still matches but ".py"
# alone doesn't. Languages enumerated explicitly so we don't pick up
# .md / .csv / .log / etc. as "files".
_FILE_MENTION_RE = re.compile(
    # Non-capturing group on the extension so re.findall returns the
    # whole "path.ext" match (capturing group would yield only the ext,
    # collapsing distinct paths down to the same key for set-dedup).
    r"\b[\w/]+\.(?:py|java|ts|tsx|js|jsx|cpp|cc|cxx|h|hpp|json|yaml|yml|toml)\b"
)

# Conservative threshold: signals must clear 4 to flip cross_cutting.
# Picked to match the Architect's own "cross_cutting iff >=4 subtasks
# OR >=4 files OR multiple subsystems" rule (see ARCHITECT_REPO_SYSTEM_PROMPT).
# Lower would over-trigger Best-of-N (3x cost) on standard briefs;
# higher would miss the failure mode this heuristic exists to catch.
_CROSS_CUTTING_THRESHOLD = 4


def infer_cross_cutting_from_brief(brief: str) -> bool:
    """Heuristic: count signals of multi-item work in the brief.

    Returns True if the brief looks like it asks for >=4 distinct
    changes. Used as a fallback when the Architect didn't set
    cross_cutting (typically because emit_plan wasn't called and the
    force-emit fallback also produced an empty cross_cutting=False
    payload).

    Signals (each contributes 1 to the counter):
    - Numbered list items (1. 2. 3.)
    - Distinct file paths mentioned (set-deduped)
    - Code blocks (each suggests a separate change; counted in pairs)
    - Test asks ("integration test" / " test that " phrasing)

    Conservative: errors on the side of NOT marking cross-cutting
    (avoids unnecessary 3x cost on borderline briefs). When in doubt,
    the workflow stays on the single-Coder path.
    """
    if not brief:
        return False
    numbered_items = len(_NUMBERED_ITEM_RE.findall(brief))
    file_mentions = len(set(_FILE_MENTION_RE.findall(brief)))
    # Each fenced code block is a pair of ``` markers; integer-divide
    # so an unmatched single ``` doesn't inflate the count.
    code_blocks = max(0, len(_CODE_BLOCK_RE.findall(brief)) // 2)

    # "Tests:" sub-list often signals multi-test ask. Lower-cased once
    # to keep the two substring counts cheap.
    lower = brief.lower()
    test_asks = lower.count("integration test") + lower.count(" test that ")

    signal_count = numbered_items + file_mentions + code_blocks + test_asks
    return signal_count >= _CROSS_CUTTING_THRESHOLD


__all__ = ["infer_cross_cutting_from_brief"]
