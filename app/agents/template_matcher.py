"""
Template matcher — picks the best template for a project before the Coder
starts editing.

Design: lightweight. Each template advertises `selection_hints` and
`anti_hints` in its template.json. We score templates by how well those
hints match the brief + architect's tech_choices + SE's files. Highest
non-negative score wins; if nothing scores positive, we return None and
the Coder starts from an empty workspace.

Why not ask an LLM to pick? We could, but:
- The hints are already plain language — grep-level matching gets ~90% of
  the value at zero cost and zero extra latency.
- Deterministic: same brief always picks the same template. Easier to
  reason about in tests and demos.

If hint matching stops being enough (more than ~10 templates, ambiguous
overlap), switch to `messages.parse()` with a schema here — the callers
just want a template name back.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Anchor discovery at the repo root. Runtime layout: /app/templates/ in the
# container image, C:\code\twai-swarm\templates\ in dev. Compute once at
# import time so we don't stat on every project.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATES_DIR = _REPO_ROOT / "templates"


@dataclass(frozen=True)
class TemplateChoice:
    """The matcher's decision for one brief."""
    name: str | None          # None = no template matched, start empty
    template_dir: Path | None # None iff name is None
    scaffold_dir: Path | None # scaffold/ subdir to copy into workspace
    score: int
    reason: str


def _tokenise(text: str) -> set[str]:
    """Lower-cased word tokens, 3+ chars, stripped of punctuation."""
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{2,}", (text or "").lower())}


def _hint_hits(hint: str, corpus: set[str]) -> bool:
    """Does the hint match anything in the corpus?

    We lowercase the hint and check if any token in the hint appears in the
    corpus. Cheap and good enough — hints are short human-readable
    phrases, and we want generous matching.
    """
    for tok in _tokenise(hint):
        if tok in corpus:
            return True
    return False


def _score_template(meta: dict, corpus: set[str]) -> tuple[int, list[str]]:
    """Score one template against the corpus. Returns (score, matched_reasons)."""
    hits: list[str] = []
    score = 0
    for hint in meta.get("selection_hints", []) or []:
        if _hint_hits(hint, corpus):
            score += 2
            hits.append(f"hint: {hint}")
    for anti in meta.get("anti_hints", []) or []:
        if _hint_hits(anti, corpus):
            score -= 3
            hits.append(f"anti: {anti}")
    # Language / framework direct match is a strong signal.
    for key in ("language", "framework"):
        val = (meta.get(key) or "").lower()
        if val and val in corpus:
            score += 3
            hits.append(f"{key}={val}")
    return score, hits


def _load_templates(root: Path) -> list[tuple[dict, Path]]:
    """Every template.json under `root` paired with its template dir."""
    templates: list[tuple[dict, Path]] = []
    if not root.is_dir():
        return templates
    for manifest in root.glob("*/template.json"):
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        templates.append((meta, manifest.parent))
    return templates


def pick_template(
    brief: str,
    architecture: dict | None = None,
    se_plan: dict | None = None,
    templates_dir: Path | None = None,
) -> TemplateChoice:
    """Pick the best template for this project.

    The corpus is the union of tokens from the brief, the architect's
    tech_choices, and the SE's file list — anywhere the user's intent has
    been expressed in words. We then score every template against that corpus.
    """
    root = templates_dir or DEFAULT_TEMPLATES_DIR
    templates = _load_templates(root)
    if not templates:
        return TemplateChoice(None, None, None, 0, "no templates available")

    corpus = _tokenise(brief)
    if architecture:
        corpus |= _tokenise(json.dumps(architecture))
    if se_plan:
        corpus |= _tokenise(json.dumps(se_plan))

    best: tuple[int, list[str], dict, Path] | None = None
    for meta, tdir in templates:
        score, reasons = _score_template(meta, corpus)
        if best is None or score > best[0]:
            best = (score, reasons, meta, tdir)

    if best is None:
        return TemplateChoice(None, None, None, 0, "no templates scored")

    score, reasons, meta, tdir = best
    if score <= 0:
        # Nothing hit positive — better an empty workspace than a template
        # the model has to fight against.
        return TemplateChoice(None, None, None, score, "no template scored above zero")

    scaffold = tdir / "scaffold"
    if not scaffold.is_dir():
        return TemplateChoice(None, None, None, score, f"{tdir.name} has no scaffold/ dir")

    return TemplateChoice(
        name=meta.get("name") or tdir.name,
        template_dir=tdir,
        scaffold_dir=scaffold,
        score=score,
        reason="; ".join(reasons) or "default pick",
    )
