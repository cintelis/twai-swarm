"""
Agentic Coder — tool-using loop over Claude Opus 4.7.

Flow:
  1. Pick a template (or start empty).
  2. Stage the template into a per-workflow sandbox dir.
  3. Prompt Claude with brief + architecture + SE plan + customisation_guide.
  4. Claude calls list_files/read_file/write_file/run_verify in a loop.
  5. Stop when run_verify returns exit 0, or we hit MAX_ITERATIONS.
  6. Emit {files: [{path, content}, ...], verify_result, iterations, ...} —
     same shape the download endpoint already understands.

Portability: the tool runner is Anthropic-specific, but the tool
*contract* (list/read/write/run_verify against a sandboxed dir) is plain
subprocess + fs I/O. Porting to Bedrock or xAI means replacing the loop
driver with their tool-use API; the sandbox + tools stay the same.
"""
from __future__ import annotations

import json
import logging
import shutil
from typing import Any

from anthropic import AsyncAnthropic

from app import config
from .coder_sandbox import Sandbox
from .coder_tools import build_tools
from .template_matcher import TemplateChoice, pick_template

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6          # hard cap — cheap escape valve
MAX_TOKENS_PER_TURN = 8000  # per model response; tool-runner sums these up
CODER_MODEL = "claude-opus-4-7"

CODER_SYSTEM_PROMPT = """You are an agentic Coder. Your job is to produce a runnable starter project that passes `verify.sh` with exit code 0.

You have these tools in a sandboxed workspace:
- list_files — see what's already there (the workspace may be pre-seeded from a template)
- read_file(path) — inspect any file
- write_file(path, content) — create or overwrite
- run_verify — run the scaffold's verify.sh; exit 0 means you're done

Workflow:
1. ALWAYS start by calling list_files to see what's there.
2. If the workspace was seeded from a template, read the template's README and the primary files called out in the customisation guide.
3. Customise the template to match the brief: rename the project, replace the example entities with the domain ones, update tests accordingly.
4. Call run_verify often — after each meaningful change. Read the stderr tail carefully; it tells you exactly what to fix.
5. When verify passes (exit_code 0), STOP calling tools and return a short final summary. Do not keep editing.

Hard rules:
- Preserve every file the template's customisation_guide.preserve list calls out unless you have a strong reason.
- Do NOT delete the verify.sh at the workspace root — it's how success is measured.
- Keep changes minimal — you're customising, not rewriting.
- If verify fails the same way twice in a row, try a different approach instead of retrying the same edit.
"""


def _build_user_message(
    brief: str,
    architecture: dict | None,
    se_plan: dict | None,
    documenter: dict | None,
    template: TemplateChoice,
) -> str:
    parts = [f"## Project brief\n{brief.strip()}\n"]
    if architecture:
        parts.append(f"## Architect's design\n{json.dumps(architecture, indent=2)}\n")
    if se_plan:
        parts.append(f"## SE implementation plan\n{json.dumps(se_plan, indent=2)}\n")
    if documenter:
        parts.append(f"## Draft README (from Documenter)\n{json.dumps(documenter, indent=2)}\n")

    if template.name and template.template_dir:
        manifest = template.template_dir / "template.json"
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            guide = meta.get("customisation_guide", {})
            parts.append(
                f"## Template seeded: `{template.name}`\n"
                f"The workspace has been pre-populated with this template's scaffold.\n"
                f"Run list_files first, then customise.\n\n"
                f"Customisation guide:\n{json.dumps(guide, indent=2)}\n"
            )
        except (OSError, json.JSONDecodeError):
            parts.append(f"## Template seeded: `{template.name}`\n")
    else:
        parts.append(
            "## No template matched\n"
            "The workspace is empty. You'll need to bootstrap everything — "
            "source files, pyproject.toml or package.json, a verify.sh that "
            "lints and runs one smoke test, README, .gitignore.\n"
        )

    parts.append(
        "\nStart by calling list_files. Your goal is `run_verify` returning exit_code 0."
    )
    return "\n".join(parts)


def _stage_template(sandbox: Sandbox, template: TemplateChoice) -> None:
    """Copy the template's scaffold + verify.sh into the workspace."""
    if template.scaffold_dir and template.scaffold_dir.is_dir():
        sandbox.copy_in(template.scaffold_dir)
    if template.template_dir:
        src_verify = template.template_dir / "verify.sh"
        if src_verify.exists():
            dst_verify = sandbox.root / "verify.sh"
            shutil.copy2(src_verify, dst_verify)
            try:
                dst_verify.chmod(0o755)
            except OSError:
                pass


def _snapshot_workspace(sandbox: Sandbox) -> list[dict[str, str]]:
    """Walk the workspace and emit [{path, content}, ...] for the download zip."""
    files: list[dict[str, str]] = []
    for rel in sandbox.list_files(max_entries=2000):
        full = sandbox.root / rel
        try:
            # Skip anything that's clearly binary by a quick peek.
            data = full.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                logger.debug("skipping binary file in snapshot: %s", rel)
                continue
            files.append({"path": rel, "content": text})
        except OSError:
            continue
    return files


async def run_agentic_coder(
    workflow_id: str,
    brief: str,
    architecture: dict | None,
    se_plan: dict | None,
    documenter: dict | None,
    heartbeat: Any = None,  # Optional[Callable[[str], None]] — temporalio.activity.heartbeat
    tenant_id: str = "default",
) -> dict:
    """Run the agentic Coder loop. Returns the same shape as the one-shot coder.

    `tenant_id` is set in the observability contextvar so every LLM call
    made by the tool-runner inherits it (tools themselves emit no traces).
    """
    from app import observability
    sandbox = Sandbox.create(workflow_id)
    template = pick_template(brief, architecture=architecture, se_plan=se_plan)
    _stage_template(sandbox, template)

    tools, stats = build_tools(sandbox)
    user_message = _build_user_message(brief, architecture, se_plan, documenter, template)

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0)

    runner = client.beta.messages.tool_runner(
        model=CODER_MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=CODER_SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": user_message}],
        # Adaptive thinking per Opus 4.7 guidance — it's the only on-mode.
        thinking={"type": "adaptive"},
    )

    iterations = 0
    total_input_tokens = 0
    total_output_tokens = 0
    stop_reason: str | None = None
    last_text = ""

    # Set the tenant scope around the whole tool-runner loop so when
    # Coder calls get Langfuse-instrumented (future work — the SDK's
    # tool_runner bypasses our provider adapter), tenant_id propagates
    # automatically via observability contextvar.
    tenant_ctx = observability.tenant_scope(tenant_id)
    tenant_ctx.__enter__()
    try:
        async for message in runner:
            iterations += 1
            if heartbeat is not None:
                try:
                    heartbeat(f"coder iteration {iterations}")
                except Exception:
                    # Heartbeat failures shouldn't kill the loop.
                    pass

            if getattr(message, "usage", None) is not None:
                total_input_tokens += int(getattr(message.usage, "input_tokens", 0) or 0)
                total_output_tokens += int(getattr(message.usage, "output_tokens", 0) or 0)

            # Capture the model's text so we have a summary if nothing else.
            for block in (message.content or []):
                if getattr(block, "type", None) == "text":
                    t = getattr(block, "text", "") or ""
                    if t.strip():
                        last_text = t

            stop_reason = getattr(message, "stop_reason", None)

            # Safety cap — if verify is green, the runner will usually stop
            # itself on the next turn, but if the model keeps going we cut it off.
            if iterations >= MAX_ITERATIONS:
                logger.warning("coder hit MAX_ITERATIONS=%d, halting", MAX_ITERATIONS)
                break
    finally:
        # We purposely do NOT destroy the sandbox on the happy path — the
        # snapshot has to happen first. Destroy only happens after a successful
        # run AND a successful snapshot (below), so failure cases leave the
        # workspace on disk for post-mortem inspection.
        tenant_ctx.__exit__(None, None, None)

    files = _snapshot_workspace(sandbox)
    verify_passed = stats.get("last_verify_exit") == 0

    # Per-1M pricing for Opus 4.7 — matches router's price table. Cheap
    # duplication; router.estimate_cost_usd requires a `ModelSpec` and we
    # don't route here (we hardcode the coder to Opus 4.7).
    input_cost = total_input_tokens * 5.0 / 1_000_000
    output_cost = total_output_tokens * 25.0 / 1_000_000

    result = {
        "language": "python",  # Filled in properly from the template meta below
        "template": template.name,
        "template_reason": template.reason,
        "summary": (last_text or "").strip()[:2000],
        "iterations": iterations,
        "stop_reason": stop_reason,
        "verify_passed": verify_passed,
        "verify_exit_code": stats.get("last_verify_exit"),
        "verify_stdout_tail": stats.get("last_verify_stdout", "")[-4000:],
        "verify_stderr_tail": stats.get("last_verify_stderr", "")[-4000:],
        "tool_calls": {
            "list_files": stats["list_files_calls"],
            "read_file": stats["read_file_calls"],
            "write_file": stats["write_file_calls"],
            "run_verify": stats["run_verify_calls"],
        },
        "files": files,
        "_tokens_in": total_input_tokens,
        "_tokens_out": total_output_tokens,
        "_cost_usd": round(input_cost + output_cost, 6),
        "_provider": "anthropic",
        "_model": CODER_MODEL,
    }

    # Enrich `language` from the template manifest if we have one.
    if template.template_dir:
        try:
            meta = json.loads((template.template_dir / "template.json").read_text(encoding="utf-8"))
            result["language"] = meta.get("language") or result["language"]
        except (OSError, json.JSONDecodeError):
            pass

    # Only clean up on a fully-clean run. On failure we keep the dir for
    # ops to inspect (cost is cheap; /tmp is ephemeral anyway).
    if verify_passed:
        sandbox.destroy()

    return result
