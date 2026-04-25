"""Template matcher scoring + dispatch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.template_matcher import TemplateChoice, pick_template


def _write_template(root: Path, name: str, meta: dict, scaffold_files: dict[str, str] | None = None):
    tdir = root / name
    tdir.mkdir(parents=True)
    (tdir / "template.json").write_text(json.dumps(meta), encoding="utf-8")
    (tdir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    scaffold = tdir / "scaffold"
    scaffold.mkdir()
    for rel, content in (scaffold_files or {}).items():
        p = scaffold / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tdir


def test_no_templates_means_no_choice(tmp_path):
    choice = pick_template("build a TODO app", templates_dir=tmp_path)
    assert choice.name is None
    assert choice.scaffold_dir is None


def test_positive_score_picks_template(tmp_path):
    _write_template(tmp_path, "fastapi-pg", {
        "name": "fastapi-pg",
        "language": "python",
        "framework": "fastapi",
        "selection_hints": ["REST API", "Postgres", "FastAPI"],
        "anti_hints": ["frontend-only"],
    }, scaffold_files={"app/main.py": "x"})

    choice = pick_template(
        brief="Build a FastAPI REST API backed by Postgres for task tracking",
        templates_dir=tmp_path,
    )
    assert choice.name == "fastapi-pg"
    assert choice.scaffold_dir.is_dir()
    assert choice.score > 0


def test_anti_hints_push_score_negative(tmp_path):
    _write_template(tmp_path, "fastapi-pg", {
        "name": "fastapi-pg",
        "language": "python",
        "framework": "fastapi",
        "selection_hints": ["REST API"],
        "anti_hints": ["React", "TypeScript", "frontend-only"],
    }, scaffold_files={"app/main.py": "x"})

    choice = pick_template(
        brief="A React TypeScript frontend app with nothing else",
        templates_dir=tmp_path,
    )
    # Anti-hint multiplier should outweigh selection-hint matches here.
    assert choice.name is None, f"got {choice.name} with reason {choice.reason}"


def test_highest_scoring_template_wins(tmp_path):
    _write_template(tmp_path, "weaker", {
        "name": "weaker",
        "language": "python",
        "selection_hints": ["REST API"],
    }, scaffold_files={"a": "x"})
    _write_template(tmp_path, "stronger", {
        "name": "stronger",
        "language": "python",
        "framework": "fastapi",
        "selection_hints": ["REST API", "FastAPI", "Postgres", "async"],
    }, scaffold_files={"a": "x"})

    choice = pick_template(
        brief="Build a FastAPI REST API with Postgres and async endpoints",
        templates_dir=tmp_path,
    )
    assert choice.name == "stronger"


def test_missing_scaffold_dir_disqualifies(tmp_path):
    tdir = tmp_path / "broken"
    tdir.mkdir()
    (tdir / "template.json").write_text(json.dumps({
        "name": "broken",
        "selection_hints": ["REST API", "FastAPI"],
    }))
    # No scaffold/ subdir created.

    choice = pick_template("Build a FastAPI REST API", templates_dir=tmp_path)
    assert choice.name is None
    assert "scaffold" in choice.reason.lower()


def test_architecture_and_plan_contribute_to_corpus(tmp_path):
    _write_template(tmp_path, "fastapi", {
        "name": "fastapi",
        "language": "python",
        "framework": "fastapi",
        "selection_hints": ["FastAPI"],
    }, scaffold_files={"a": "x"})

    # Brief is intentionally generic — the fastapi signal has to come from
    # the architect's tech_choices, not the brief itself.
    choice = pick_template(
        brief="a backend service",
        architecture={"tech_choices": [{"name": "FastAPI", "why": "async ergonomics"}]},
        templates_dir=tmp_path,
    )
    assert choice.name == "fastapi"


# ─── Real templates dispatch ─────────────────────────────────────────────────
# These tests don't isolate templates_dir — they hit the real templates/
# tree. If a new template is added that matches one of these briefs better
# than the asserted one, update the assertion deliberately rather than the
# hint metadata. The point is that the catalogue stays coherent.

@pytest.mark.parametrize("brief, expected", [
    ("FastAPI books service with Postgres", "python-fastapi-postgres"),
    ("Async Python REST API for tracking inventory", "python-fastapi-postgres"),
    ("Next.js TypeScript app with Prisma to manage widgets", "nextjs-ts-prisma"),
    ("Full-stack Next.js app with database for a blog", "nextjs-ts-prisma"),
    ("React SPA with Tailwind for a kanban board", "vite-react-tailwind"),
    ("Vite React dashboard showing analytics charts", "vite-react-tailwind"),
])
def test_real_templates_dispatch(brief, expected):
    choice = pick_template(brief)
    assert choice.name == expected, (
        f"brief {brief!r} expected {expected!r} but matcher returned "
        f"{choice.name!r} (score={choice.score}, reason={choice.reason})"
    )
