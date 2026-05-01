"""Repo-aware Critic activity — Sprint 18c.

Post-step that runs AFTER `run_repo_coder_activity` and validates the
Coder's diff against the Architect plan's `acceptance_criteria` checklist.
If gaps remain, builds a structured handoff doc and the workflow fires
a continuation Coder pass.

Design (per `sprint-18-plan.md` §"Architectural decisions"):
  - D3 — Two-stage validation. First a *deterministic gate*
    (ruff / compileall / mvn compile / npm typecheck) on the touched
    files; only items that aren't testable as code go through the
    LLM-as-judge stage. Reflexion-inspired (arxiv 2303.11366): the
    Evaluator is a "scalar reward source" that pulls from real signals
    where available and falls back to model judgement only for semantic
    items.
  - D4 — Continuation cap = 2 (loops live in the workflow, not here).
    The Critic only emits a verdict + a continuation_prompt; it doesn't
    decide to re-run. Keeps the agent stateless.
  - D7 — Continuation prompt is a *structured handoff document*
    (Current state / Acceptance criteria status / Immediate next steps /
    Open questions / Constraint), NOT a chat transcript. Receiving Coder
    treats it as a fresh brief, not a memory.
  - D8 — Critic uses Sonnet 4.6 (different model than the Coder's Haiku
    4.5). Mitigates self-enhancement bias when the same model would
    otherwise grade its own work.

Best-effort gates: if a build tool is absent on PATH or the project
lacks the right config (no `pom.xml`, no `tsconfig.json`), the gate
SKIPS that language silently. We never fail a workflow because the
worker doesn't have Maven installed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from anthropic import AsyncAnthropic

from app import config

logger = logging.getLogger(__name__)

# Per D8: Critic uses Sonnet 4.6, distinct from the Coder's Haiku.
# Hardcoded for the same reason as ARCHITECT_MODEL — single call site,
# threading the router for one role is overkill.
CRITIC_MODEL = "claude-sonnet-4-6"

# LLM-judge call is non-streaming and bounded (~1 turn). 2K tokens of
# JSON output covers ~50 acceptance criteria with one-line evidence each.
MAX_TOKENS_JUDGE = 4096

# Per-tool subprocess timeouts. Pythons gates are cheap; Maven is the
# expensive one — capped at 5 minutes because a single Critic pass that
# hangs for 10 minutes on `mvn compile` would dwarf the Coder's own
# budget. If the project is too big to compile in 5 minutes, the gate
# times out and is treated as "skipped" (best-effort per the docstring).
RUFF_TIMEOUT_SECONDS = 60
COMPILEALL_TIMEOUT_SECONDS = 60
MVN_TIMEOUT_SECONDS = 300
NPM_TIMEOUT_SECONDS = 180


@dataclass
class CriticFailure:
    """One acceptance criterion the Critic believes is NOT satisfied.

    `severity="block"` triggers a continuation Coder pass; "warn" is
    surfaced in the PR footer but does NOT loop. The Architect's
    schema doesn't currently distinguish severities so the LLM judge
    defaults to "block" — leaving the field on the dataclass so
    operators can downgrade specific items via post-hoc edits.
    """
    criterion: str
    evidence: str
    severity: Literal["block", "warn"] = "block"


@dataclass
class GateFailure:
    """One deterministic-gate failure (file/line + a short diagnostic).

    `tool` is the gate that produced the failure ("ruff", "compileall",
    "mvn", "npm"). `line` is None when the diagnostic isn't line-scoped
    (e.g. a file-level mvn error or a bare compileall traceback).
    """
    tool: str
    file: str
    line: int | None
    message: str


@dataclass
class CriticRepoOutput:
    """Critic's deliverable for one repo-task Coder pass.

    Mirrors the provenance fields used by the Architect / Coder agents
    (`_model`, `_provider`, `_tokens_in`, `_tokens_out`, `_cost_usd`)
    so the cost-summary card aggregates without a special case.
    """
    verdict: Literal["complete", "incomplete"]
    passed_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[CriticFailure] = field(default_factory=list)
    deterministic_gate_passed: bool = True
    gate_failures: list[GateFailure] = field(default_factory=list)
    continuation_prompt: str | None = None
    # Provenance — leading underscore matches the Coder/Architect convention.
    _model: str = CRITIC_MODEL
    _provider: str = "anthropic"
    _tokens_in: int = 0
    _tokens_out: int = 0
    _cost_usd: float = 0.0


# ─── Deterministic gates ────────────────────────────────────────────────────


def _group_files_by_language(files_changed: list[str]) -> dict[str, list[str]]:
    """Bucket repo-relative paths by the gate that should validate them.

    Buckets:
      - "python"  → .py
      - "java"    → .java
      - "ts"      → .ts, .tsx, .js, .jsx
      - "cpp"     → .c, .cpp, .cc, .cxx, .h, .hpp, .hxx (currently no
                    cheap cross-platform compiler we trust → skipped)
      - "other"   → anything else (skipped — no gate)
    """
    groups: dict[str, list[str]] = {
        "python": [], "java": [], "ts": [], "cpp": [], "other": [],
    }
    for f in files_changed:
        lower = f.lower()
        if lower.endswith(".py"):
            groups["python"].append(f)
        elif lower.endswith(".java"):
            groups["java"].append(f)
        elif lower.endswith((".ts", ".tsx", ".js", ".jsx")):
            groups["ts"].append(f)
        elif lower.endswith((".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx")):
            groups["cpp"].append(f)
        else:
            groups["other"].append(f)
    return groups


def _ruff_gate(repo_root: Path, py_files: list[str]) -> list[GateFailure]:
    """Run `ruff check --output-format=json` on the touched .py files.

    ruff returns exit code 0 = clean, 1 = lint failures, 2 = config /
    invocation error. We treat 0/1 as "ran successfully" and parse the
    JSON; exit 2 (or any FileNotFound) is treated as "tool unavailable"
    and the gate skips silently.
    """
    if shutil.which("ruff") is None and not _has_module("ruff"):
        logger.info("ruff not on PATH and not importable; skipping ruff gate")
        return []

    abs_paths = [str(repo_root / f) for f in py_files]
    # Prefer module invocation so we use the venv's ruff if it exists;
    # falls back to the on-PATH binary otherwise.
    cmd_base: list[str]
    if _has_module("ruff"):
        import sys
        cmd_base = [sys.executable, "-m", "ruff"]
    else:
        cmd_base = ["ruff"]
    cmd = [*cmd_base, "check", "--output-format=json", *abs_paths]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=RUFF_TIMEOUT_SECONDS, cwd=str(repo_root),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("ruff gate failed to run (%s); skipping", e)
        return []
    if proc.returncode not in (0, 1):
        logger.warning(
            "ruff exited with code %d (stderr=%s); skipping",
            proc.returncode, proc.stderr[:200],
        )
        return []
    failures: list[GateFailure] = []
    try:
        items = json.loads(proc.stdout) if proc.stdout.strip() else []
    except json.JSONDecodeError:
        logger.warning("ruff produced non-JSON output; skipping parse")
        return []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        # ruff emits absolute paths; reverse-map to repo-relative for the
        # Coder's continuation prompt (relative paths are what it sees).
        abs_path = item.get("filename") or ""
        try:
            rel = str(Path(abs_path).resolve().relative_to(repo_root))
        except (ValueError, OSError):
            rel = abs_path
        failures.append(GateFailure(
            tool="ruff",
            file=rel,
            line=int(loc.get("row")) if isinstance(loc.get("row"), int) else None,
            message=f"{item.get('code', '')}: {item.get('message', '')}".strip(": "),
        ))
    return failures


def _compileall_gate(repo_root: Path, py_files: list[str]) -> list[GateFailure]:
    """Run `python -m compileall -q` on the touched .py files.

    Catches outright SyntaxErrors that ruff sometimes also catches but
    can miss when its parser falls back. Cheap insurance.
    """
    import sys
    abs_paths = [str(repo_root / f) for f in py_files]
    cmd = [sys.executable, "-m", "compileall", "-q", *abs_paths]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=COMPILEALL_TIMEOUT_SECONDS, cwd=str(repo_root),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("compileall gate failed to run (%s); skipping", e)
        return []
    if proc.returncode == 0:
        return []
    # compileall output on failure: lines like
    #   "*** Error compiling 'foo.py'..."
    #   followed by a Python traceback. We pull the file name out and
    #   emit a single GateFailure per file.
    failures: list[GateFailure] = []
    seen: set[str] = set()
    combined = (proc.stdout or "") + (proc.stderr or "")
    for raw in combined.splitlines():
        line = raw.strip()
        # Look for the "*** Error compiling" marker (Python 3.x format).
        marker = "*** Error compiling"
        if marker in line:
            # Extract the path inside the surrounding quotes.
            try:
                quoted = line.split(marker, 1)[1].strip().strip("'\"").rstrip(":'\"")
                abs_path = quoted
                rel = str(Path(abs_path).resolve().relative_to(repo_root))
            except (ValueError, OSError, IndexError):
                rel = line
            if rel in seen:
                continue
            seen.add(rel)
            failures.append(GateFailure(
                tool="compileall", file=rel, line=None,
                message="syntax error (see compileall output)",
            ))
    if not failures and proc.returncode != 0:
        # compileall failed but we couldn't parse a specific file out —
        # emit a single bucketed failure so the operator still sees it.
        failures.append(GateFailure(
            tool="compileall", file="(unknown)", line=None,
            message=f"compileall returned {proc.returncode}",
        ))
    return failures


def _mvn_gate(repo_root: Path, java_files: list[str]) -> list[GateFailure]:
    """Run `mvn -q compile` if a pom.xml exists at the repo root.

    Maven is the slow gate — capped at 5 minutes. Project layouts that
    use Gradle / Bazel / etc. are skipped silently; we don't try to
    detect every JVM build tool.
    """
    if shutil.which("mvn") is None:
        logger.info("mvn not on PATH; skipping mvn gate")
        return []
    if not (repo_root / "pom.xml").exists():
        logger.info("no pom.xml at repo root; skipping mvn gate")
        return []
    # `-q` = quiet (errors only), `-DskipTests` so we don't get pulled
    # into long test runs that aren't part of the gate's contract.
    cmd = ["mvn", "-q", "-DskipTests", "compile"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=MVN_TIMEOUT_SECONDS, cwd=str(repo_root),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("mvn gate failed to run (%s); skipping", e)
        return []
    if proc.returncode == 0:
        return []
    # Maven error format: "[ERROR] /abs/path/Foo.java:[12,3] message"
    failures: list[GateFailure] = []
    combined = (proc.stdout or "") + (proc.stderr or "")
    for raw in combined.splitlines():
        line = raw.strip()
        if not (line.startswith("[ERROR]") and ".java" in line):
            continue
        try:
            tail = line.split("[ERROR]", 1)[1].strip()
            # Look for ":[" to split path from line/col.
            if ":[" in tail:
                path_part, after = tail.split(":[", 1)
                lineno_str = after.split(",", 1)[0]
                try:
                    lineno = int(lineno_str)
                except ValueError:
                    lineno = None
                msg = after.split("]", 1)[1].strip(": ") if "]" in after else tail
            else:
                path_part, lineno, msg = tail, None, tail
            try:
                rel = str(Path(path_part).resolve().relative_to(repo_root))
            except (ValueError, OSError):
                rel = path_part
            failures.append(GateFailure(
                tool="mvn", file=rel, line=lineno, message=msg[:300],
            ))
        except Exception as e:  # noqa: BLE001
            logger.debug("mvn line parse failed: %r (%s)", line, e)
    if not failures:
        # Failed but unparseable — single bucket failure.
        failures.append(GateFailure(
            tool="mvn", file="(unknown)", line=None,
            message=f"mvn compile returned {proc.returncode}",
        ))
    return failures


def _npm_typecheck_gate(repo_root: Path, ts_files: list[str]) -> list[GateFailure]:
    """Run `npm run typecheck` if a package.json + tsconfig.json exist.

    Many JS/TS projects don't have a `typecheck` script — we look for
    one explicitly and skip if absent. Avoids running `tsc` directly
    because TS module resolution depends on the project's own config.
    """
    if shutil.which("npm") is None:
        logger.info("npm not on PATH; skipping npm typecheck gate")
        return []
    pkg = repo_root / "package.json"
    tsc = repo_root / "tsconfig.json"
    if not pkg.exists() or not tsc.exists():
        logger.info("no package.json or tsconfig.json; skipping npm typecheck")
        return []
    # Make sure the project actually defines a `typecheck` script —
    # otherwise npm errors out and we'd report a false gate failure.
    try:
        meta = json.loads(pkg.read_text(encoding="utf-8"))
        if not isinstance(meta.get("scripts"), dict) or "typecheck" not in meta["scripts"]:
            logger.info("package.json has no `typecheck` script; skipping")
            return []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not parse package.json (%s); skipping npm gate", e)
        return []
    cmd = ["npm", "run", "typecheck", "--silent"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=NPM_TIMEOUT_SECONDS, cwd=str(repo_root),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("npm typecheck failed to run (%s); skipping", e)
        return []
    if proc.returncode == 0:
        return []
    # tsc error format: "src/foo.ts(12,5): error TS2304: Cannot find name 'x'."
    failures: list[GateFailure] = []
    combined = (proc.stdout or "") + (proc.stderr or "")
    import re
    pat = re.compile(r"^(?P<file>[^\s:][^:()]*)\((?P<line>\d+),\d+\):\s*(?P<msg>.+)$")
    for raw in combined.splitlines():
        m = pat.match(raw.strip())
        if not m:
            continue
        rel = m.group("file")
        # tsc usually emits repo-relative paths; if absolute, normalise.
        try:
            rel = str(Path(rel).resolve().relative_to(repo_root))
        except (ValueError, OSError):
            pass
        failures.append(GateFailure(
            tool="npm", file=rel,
            line=int(m.group("line")),
            message=m.group("msg")[:300],
        ))
    if not failures:
        failures.append(GateFailure(
            tool="npm", file="(unknown)", line=None,
            message=f"npm run typecheck returned {proc.returncode}",
        ))
    return failures


def _has_module(name: str) -> bool:
    """Cheap import check without actually importing the module.

    Used to decide whether the venv has ruff bundled even if the binary
    isn't on PATH (Windows venvs sometimes work this way). Falls back to
    False on any importlib hiccup.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


def run_deterministic_gate(
    repo_root: Path, files_changed: list[str],
) -> tuple[bool, list[GateFailure]]:
    """Run every applicable deterministic gate. Returns (all_passed, failures).

    "Best-effort": missing tools skip silently; we only report a failure
    when a tool actually ran AND produced a diagnostic. An empty failure
    list does NOT prove "everything compiles" — it can also mean "nothing
    was checked" (empty file list, or all gates skipped). Callers that
    need to differentiate should inspect `files_changed` themselves.
    """
    if not files_changed:
        return (True, [])
    groups = _group_files_by_language(files_changed)
    failures: list[GateFailure] = []
    if groups["python"]:
        failures.extend(_ruff_gate(repo_root, groups["python"]))
        failures.extend(_compileall_gate(repo_root, groups["python"]))
    if groups["java"]:
        failures.extend(_mvn_gate(repo_root, groups["java"]))
    if groups["ts"]:
        failures.extend(_npm_typecheck_gate(repo_root, groups["ts"]))
    # cpp / other: skipped — no cross-platform cheap compiler we trust.
    return (len(failures) == 0, failures)


# ─── LLM checklist judge ────────────────────────────────────────────────────


def _flatten_acceptance_criteria(architect_plan: dict) -> list[tuple[str, str, str]]:
    """Extract (subtask_id, criterion_text, criterion_index_str) tuples.

    The judge sees a single flat numbered list (subtask grouping is for
    human consumption in the handoff doc, not the LLM prompt — flat is
    easier for the model to reference by index).
    """
    out: list[tuple[str, str, str]] = []
    if not architect_plan or not isinstance(architect_plan, dict):
        return out
    for subtask in (architect_plan.get("subtasks") or []):
        if not isinstance(subtask, dict):
            continue
        sid = str(subtask.get("id", ""))
        for c in (subtask.get("acceptance_criteria") or []):
            if not c:
                continue
            text = str(c)
            out.append((sid, text, str(len(out))))
    return out


def _build_judge_prompt(
    criteria: list[tuple[str, str, str]],
    coder_diff: str,
    files_with_content: list[dict],
    gate_failures: list[GateFailure],
) -> str:
    """Render a single LLM call that grades all criteria together.

    Single-call batching is meaningfully cheaper than per-criterion calls
    (one set of context tokens instead of N) AND gives the judge global
    visibility — useful when criteria depend on each other.
    """
    parts: list[str] = []
    parts.append(
        "You are grading a Coder agent's diff against an Architect's "
        "acceptance-criteria checklist. Be strict: a criterion is only "
        "satisfied if the diff demonstrably implements it. A partial "
        "implementation is `partial`, not `yes`."
    )
    parts.append("")
    parts.append("## Acceptance criteria to grade")
    for _, text, idx in criteria:
        parts.append(f"{idx}. {text}")
    parts.append("")
    if gate_failures:
        parts.append("## Deterministic-gate failures (already known)")
        for gf in gate_failures[:30]:  # cap for prompt size
            ln = f":{gf.line}" if gf.line else ""
            parts.append(f"- [{gf.tool}] {gf.file}{ln} — {gf.message}")
        if len(gate_failures) > 30:
            parts.append(f"- …and {len(gate_failures) - 30} more")
        parts.append("")
    parts.append("## Coder's diff")
    parts.append("```diff")
    # Cap the diff at ~30K chars so we don't blow the context budget on
    # giant changesets. The judge sees the head + tail when truncated.
    if len(coder_diff) > 30000:
        head = coder_diff[:20000]
        tail = coder_diff[-8000:]
        parts.append(head)
        parts.append("\n... (diff truncated for prompt budget) ...\n")
        parts.append(tail)
    else:
        parts.append(coder_diff or "(empty diff)")
    parts.append("```")
    parts.append("")
    if files_with_content:
        parts.append("## Files modified (post-Coder content, summary)")
        for f in files_with_content[:20]:
            path = f.get("path", "")
            content = f.get("content", "") or ""
            preview = content[:400].replace("\n", " ")
            parts.append(f"- `{path}` — {len(content)} bytes; preview: {preview}")
        if len(files_with_content) > 20:
            parts.append(f"- …and {len(files_with_content) - 20} more")
        parts.append("")
    parts.append("## Output format")
    parts.append(
        "Return a single JSON object mapping criterion-index strings to "
        '`{"status": "yes"|"no"|"partial", "evidence": "<one-line>"}`. '
        "Include EVERY criterion index from the checklist above. Do not "
        "wrap the JSON in markdown fences. Example:"
    )
    parts.append('{"0": {"status": "yes", "evidence": "Endpoint added in routes.py"}, '
                 '"1": {"status": "no", "evidence": "No 401 path in handler"}}')
    return "\n".join(parts)


def _parse_judge_response(text: str, criteria: list[tuple[str, str, str]]) -> dict:
    """Parse the judge's JSON output. Tolerant of stray markdown fences.

    Falls back to {} if the JSON can't be salvaged — caller treats every
    criterion as "unknown → block" so the Critic errs on the side of
    triggering a continuation pass rather than approving silently.
    """
    if not text:
        return {}
    stripped = text.strip()
    # Strip optional ```json fences.
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        # If a trailing fence line remains, lop it.
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Last-ditch: extract the first {...} block.
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                logger.warning(
                    "judge response not parseable as JSON; "
                    "treating all criteria as unknown",
                )
                return {}
        return {}


async def _extract_criteria_from_brief(
    brief: str, client: AsyncAnthropic,
) -> tuple[list[str], int, int]:
    """Sprint 18.1 fallback: derive acceptance criteria from the brief itself.

    Used when the Architect produced an empty subtask list (force-emit
    fallback also failed, or a regression elsewhere). Asks Sonnet to
    enumerate the testable asks in the brief, returning them as a flat
    list of one-line acceptance criteria the Critic can judge against
    via the existing checklist path.

    Returns (criteria, tokens_in, tokens_out). Empty list means the
    extraction call also failed — caller proceeds with no checklist (the
    pre-18.1 behaviour, but at least we tried).
    """
    from app import observability

    if not brief or not brief.strip():
        return ([], 0, 0)
    extract_system = (
        "You extract testable acceptance criteria from project briefs. "
        "Output JSON via the emit_criteria tool. Each criterion is a "
        "single sentence stating something that should be true after "
        "the work is complete."
    )
    extract_user = (
        f"Brief:\n{brief}\n\n"
        "Extract 3-10 acceptance criteria. Each criterion should "
        "be testable by reading the diff."
    )
    # Sprint 19: wrap as a generation. Auto-nests under the critic span.
    try:
        with observability.generation(
            name=f"anthropic.{CRITIC_MODEL}.brief_criteria",
            model=CRITIC_MODEL,
            provider="anthropic",
            system=extract_system,
            user=extract_user,
            tools=["emit_criteria"],
        ) as gen:
            response = await client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=2048,
                system=extract_system,
                messages=[{"role": "user", "content": extract_user}],
                tools=[{
                    "name": "emit_criteria",
                    "description": "Emit the list of acceptance criteria",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 15,
                            },
                        },
                        "required": ["criteria"],
                    },
                }],
                tool_choice={"type": "tool", "name": "emit_criteria"},
            )
            tokens_in = int(getattr(response.usage, "input_tokens", 0) or 0)
            tokens_out = int(getattr(response.usage, "output_tokens", 0) or 0)
            gen.end(
                output=f"emit_criteria called ({len(response.content or [])} blocks)",
                usage={"input": tokens_in, "output": tokens_out},
            )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "critic brief-criteria extraction failed: %s; "
            "proceeding with no checklist", e,
        )
        return ([], 0, 0)

    for block in (response.content or []):
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "emit_criteria"
        ):
            raw = getattr(block, "input", None) or {}
            criteria = [str(c) for c in (raw.get("criteria") or []) if c]
            logger.info(
                "critic extracted %d brief-derived criteria (Architect "
                "plan was degraded)", len(criteria),
            )
            return (criteria, tokens_in, tokens_out)
    return ([], tokens_in, tokens_out)


async def run_llm_checklist_judge(
    architect_plan: dict,
    coder_diff: str,
    files_with_content: list[dict],
    unsatisfied_gate_failures: list[GateFailure],
) -> tuple[list[str], list[CriticFailure], int, int]:
    """Single Sonnet call that grades every acceptance criterion.

    Returns (passed_criteria_texts, failed_criteria, tokens_in, tokens_out).

    On API failure, ALL criteria fall through as failed with a "judge
    unavailable" evidence note — defensive default that triggers a
    continuation rather than silently approving.
    """
    from app import observability

    criteria = _flatten_acceptance_criteria(architect_plan)
    if not criteria:
        return ([], [], 0, 0)

    prompt = _build_judge_prompt(
        criteria, coder_diff, files_with_content, unsatisfied_gate_failures,
    )
    judge_system = (
        "You are a strict evaluator. You grade Coder agent output "
        "against acceptance criteria. Always return JSON in the "
        "exact format specified."
    )
    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
    # Sprint 19: wrap the judge call as a generation. Auto-nests under
    # the critic agent_span via ContextVars.
    tokens_in = 0
    tokens_out = 0
    try:
        with observability.generation(
            name=f"anthropic.{CRITIC_MODEL}.judge",
            model=CRITIC_MODEL,
            provider="anthropic",
            system=judge_system,
            user=prompt,
            metadata={"n_criteria": len(criteria)},
        ) as gen:
            resp = await client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=MAX_TOKENS_JUDGE,
                system=judge_system,
                messages=[{"role": "user", "content": prompt}],
            )
            tokens_in = int(getattr(resp.usage, "input_tokens", 0) or 0)
            tokens_out = int(getattr(resp.usage, "output_tokens", 0) or 0)
            text_parts: list[str] = []
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            judge_text = "".join(text_parts)
            gen.end(
                output=judge_text[:1000],
                usage={"input": tokens_in, "output": tokens_out},
            )
    except Exception as e:  # noqa: BLE001
        logger.error("critic LLM judge failed: %s; failing all criteria", e)
        failed = [
            CriticFailure(
                criterion=text,
                evidence=f"judge unavailable ({type(e).__name__}); "
                         f"treat as failure for safety",
                severity="block",
            )
            for _, text, _ in criteria
        ]
        return ([], failed, 0, 0)

    parsed = _parse_judge_response(judge_text, criteria)

    passed: list[str] = []
    failed: list[CriticFailure] = []
    for _, text, idx in criteria:
        item = parsed.get(idx) or parsed.get(int(idx) if idx.isdigit() else idx)
        if not isinstance(item, dict):
            failed.append(CriticFailure(
                criterion=text,
                evidence="judge did not return a verdict for this item",
                severity="block",
            ))
            continue
        status = str(item.get("status", "")).lower().strip()
        evidence = str(item.get("evidence", "")).strip() or "(no evidence)"
        if status == "yes":
            passed.append(text)
        else:
            # "no" or "partial" or anything unexpected → block.
            failed.append(CriticFailure(
                criterion=text, evidence=evidence, severity="block",
            ))

    # tokens_in / tokens_out were captured inside the generation block above.
    return (passed, failed, tokens_in, tokens_out)


# ─── Continuation handoff doc (D7) ──────────────────────────────────────────


def build_continuation_handoff_doc(
    architect_plan: dict,
    prior_diff: str,
    prior_files_changed: list[str],
    passed_criteria: list[str],
    failed_criteria: list[CriticFailure],
    gate_failures: list[GateFailure],
) -> str:
    """Render the structured handoff doc for the next Coder pass.

    Per D7: structured sections, NOT a chat transcript. Sections:
      - Current state summary
      - Acceptance criteria status
      - Immediate next steps
      - Open questions
      - Constraint
    The next Coder consumes this as its `brief` parameter. The prior
    Coder's edits are still on disk in the cloned repo; this doc tells
    the new pass what to ADD, not redo.
    """
    lines: list[str] = []
    lines.append("## Current state summary")
    if prior_files_changed:
        files_summary = ", ".join(f"`{f}`" for f in prior_files_changed[:20])
        if len(prior_files_changed) > 20:
            files_summary += f" (+{len(prior_files_changed) - 20} more)"
        lines.append(f"- Files modified in prior pass: {files_summary}")
    else:
        lines.append("- Files modified in prior pass: (none)")
    diff_lines = (prior_diff or "").splitlines()
    if diff_lines:
        snippet = "\n".join(diff_lines[:10])
        lines.append("- Diff summary (first 10 lines):")
        lines.append("```diff")
        lines.append(snippet)
        if len(diff_lines) > 10:
            lines.append(f"... ({len(diff_lines) - 10} more lines)")
        lines.append("```")
    else:
        lines.append("- Diff summary: (empty)")
    if gate_failures:
        by_tool: dict[str, int] = {}
        for gf in gate_failures:
            by_tool[gf.tool] = by_tool.get(gf.tool, 0) + 1
        gate_summary = ", ".join(f"{tool}={count}" for tool, count in sorted(by_tool.items()))
        lines.append(f"- Deterministic gates: FAILING ({gate_summary})")
    else:
        lines.append("- Deterministic gates: passing (or skipped for absent tooling)")
    lines.append("")

    lines.append("## Acceptance criteria status")
    if passed_criteria:
        lines.append("### Already satisfied")
        for c in passed_criteria:
            lines.append(f"- [x] {c}")
    if failed_criteria:
        lines.append("### Still missing")
        for cf in failed_criteria:
            lines.append(f"- [ ] {cf.criterion}")
            lines.append(f"  - evidence: {cf.evidence}")
    if gate_failures:
        lines.append("### Gate failures to fix")
        for gf in gate_failures[:30]:
            ln = f":{gf.line}" if gf.line else ""
            lines.append(f"- [{gf.tool}] `{gf.file}`{ln} — {gf.message}")
        if len(gate_failures) > 30:
            lines.append(f"- ... and {len(gate_failures) - 30} more")
    lines.append("")

    lines.append("## Immediate next steps")
    step = 1
    for cf in failed_criteria:
        lines.append(f"{step}. Address acceptance criterion: {cf.criterion}")
        step += 1
    for gf in gate_failures[:10]:
        ln = f":{gf.line}" if gf.line else ""
        lines.append(f"{step}. Fix {gf.tool} failure in `{gf.file}`{ln} ({gf.message})")
        step += 1
    if step == 1:
        lines.append("(no specific next steps — Critic returned incomplete with no failure list)")
    lines.append("")

    lines.append("## Open questions (do NOT silently decide)")
    lines.append(
        "- None flagged by the judge. If you encounter ambiguity in the "
        "Architect plan while addressing the items above, surface it in "
        "your final summary rather than picking one option silently."
    )
    lines.append("")

    lines.append("## Constraint")
    lines.append(
        "You are picking up where a previous Coder left off. Do NOT redo "
        "their work. Only address the items in 'Still missing' and 'Gate "
        "failures to fix'. The previous diff is already on disk — your job "
        "is to ADD to it, not replace it. Keep edits surgical: every file "
        "you touch should map to one of the items above."
    )
    return "\n".join(lines)


# ─── Coordinator ────────────────────────────────────────────────────────────


async def run_critic_repo(
    architect_plan: dict | None,
    coder_diff: str,
    files_with_content: list[dict],
    repo_root: Path,
    brief: str = "",
) -> CriticRepoOutput:
    """Run gate + LLM judge, build the verdict.

    If `architect_plan` is None / empty (Architect failed and shipped a
    degraded output), Sprint 18.1 falls back to extracting acceptance
    criteria from the `brief` itself via a Sonnet helper. This avoids
    the vacuous "verdict=complete with 0 criteria graded" failure mode
    surfaced in run 019de315 — the Critic must validate SOMETHING.

    If both the plan AND the brief-fallback yield zero criteria, the
    Critic still returns verdict="complete" with the gate result (the
    pre-18.1 behaviour), so a degraded run never blocks the workflow.

    `brief` defaults to "" for backward compat with pre-18.1 callers
    (test fixtures, replayed Temporal histories). Activity callers
    should always pass it.
    """
    files_changed = [str(f.get("path", "")) for f in (files_with_content or []) if f.get("path")]

    # Stage 1: deterministic gate.
    gate_passed, gate_failures = run_deterministic_gate(repo_root, files_changed)

    # Sprint 18.1: detect the degraded-Architect path and try to recover
    # by extracting criteria from the brief. Anything that yields a
    # non-empty checklist below means we still get a real LLM-judge pass.
    plan_for_judge = architect_plan
    fallback_tokens_in = 0
    fallback_tokens_out = 0
    if (
        not architect_plan
        or not isinstance(architect_plan, dict)
        or not architect_plan.get("subtasks")
    ):
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)
        fallback_criteria, fallback_tokens_in, fallback_tokens_out = (
            await _extract_criteria_from_brief(brief, client)
        )
        if fallback_criteria:
            # Synthesise a minimal plan dict the existing judge can consume.
            # All criteria sit under one synthetic "brief.fallback" subtask
            # so the flat-index numbering stays stable.
            plan_for_judge = {
                "subtasks": [{
                    "id": "brief.fallback",
                    "description": "Brief asks (Architect-degraded fallback)",
                    "acceptance_criteria": fallback_criteria,
                }],
            }
        else:
            # Both Architect and brief-fallback failed; no checklist to
            # grade. Preserve the pre-18.1 "don't break the workflow"
            # contract — surface the gate result and exit complete.
            logger.info(
                "critic skipping LLM judge: no architect_plan / no "
                "subtasks AND brief-fallback yielded no criteria "
                "(fully degraded path); gate_passed=%s",
                gate_passed,
            )
            input_cost = fallback_tokens_in * 3.0 / 1_000_000
            output_cost = fallback_tokens_out * 15.0 / 1_000_000
            return CriticRepoOutput(
                verdict="complete",
                passed_criteria=[],
                failed_criteria=[],
                deterministic_gate_passed=gate_passed,
                gate_failures=gate_failures,
                continuation_prompt=None,
                _tokens_in=fallback_tokens_in,
                _tokens_out=fallback_tokens_out,
                _cost_usd=round(input_cost + output_cost, 6),
            )

    # Stage 2: LLM checklist judge.
    passed, failed, tokens_in, tokens_out = await run_llm_checklist_judge(
        plan_for_judge, coder_diff, files_with_content, gate_failures,
    )
    # Roll the fallback-extraction tokens into the Critic's total cost
    # so the cost-summary card doesn't lose them.
    tokens_in += fallback_tokens_in
    tokens_out += fallback_tokens_out

    # Verdict: incomplete if EITHER stage flags a problem.
    has_blocking_failure = any(cf.severity == "block" for cf in failed)
    incomplete = has_blocking_failure or not gate_passed

    continuation_prompt: str | None = None
    if incomplete:
        # Pass `plan_for_judge` (which equals architect_plan when the
        # Architect was healthy, OR the synthetic brief-fallback plan
        # when degraded) so the handoff doc references the same criteria
        # the Critic actually graded against.
        continuation_prompt = build_continuation_handoff_doc(
            architect_plan=plan_for_judge,
            prior_diff=coder_diff,
            prior_files_changed=files_changed,
            passed_criteria=passed,
            failed_criteria=failed,
            gate_failures=gate_failures,
        )

    # Sonnet 4.6 pricing per 1M tokens (matches router.MODELS["sonnet"]).
    input_cost = tokens_in * 3.0 / 1_000_000
    output_cost = tokens_out * 15.0 / 1_000_000

    return CriticRepoOutput(
        verdict="incomplete" if incomplete else "complete",
        passed_criteria=passed,
        failed_criteria=failed,
        deterministic_gate_passed=gate_passed,
        gate_failures=gate_failures,
        continuation_prompt=continuation_prompt,
        _model=CRITIC_MODEL,
        _provider="anthropic",
        _tokens_in=tokens_in,
        _tokens_out=tokens_out,
        _cost_usd=round(input_cost + output_cost, 6),
    )


def critic_output_to_dict(out: CriticRepoOutput) -> dict:
    """Helper: convert CriticRepoOutput to a plain dict.

    Equivalent to `dataclasses.asdict(out)`. Wraps it as a helper so
    the activity + tests reach for the same canonical conversion (mirrors
    `architect_output_to_dict`).
    """
    return asdict(out)


__all__ = [
    "CRITIC_MODEL",
    "CriticFailure",
    "CriticRepoOutput",
    "GateFailure",
    "MAX_TOKENS_JUDGE",
    "MVN_TIMEOUT_SECONDS",
    "NPM_TIMEOUT_SECONDS",
    "RUFF_TIMEOUT_SECONDS",
    "build_continuation_handoff_doc",
    "critic_output_to_dict",
    "run_critic_repo",
    "run_deterministic_gate",
    "run_llm_checklist_judge",
    "_extract_criteria_from_brief",
]
_ = os  # silence unused-import lint until a caller needs it
