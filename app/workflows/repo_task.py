"""RepoTaskWorkflow — Sprint 10e.

The "work on existing code" capability. Sister to ProjectWorkflow (which
generates new projects from briefs); this one takes an existing git repo
+ a task brief, indexes it into Neo4j, and hands an agentic Coder both
the workspace tools and the graph-aware tools (Sprint 10c) to make a
surgical change.

Flow:
    1. clone_repo_activity  — `git clone --depth 1 --branch <b>` to /tmp/...
    2. index_repo_activity  — runs the Sprint 10a-d indexer over the clone
    3. run_repo_coder_activity — agentic Coder loop with graph tools enabled

Output is a unified `git diff` against the cloned commit + a list of files
the Coder modified. Pushing back to a branch / opening a PR is deferred
to Sprint 10f (it can reuse the existing GitHub App push code).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow

# Activities are imported through workflow.unsafe so the workflow module
# stays sandboxed (Temporal requirement — workflows can't directly call
# I/O-bound code at definition time).
with workflow.unsafe.imports_passed_through():
    from app.activities import (
        clone_repo_activity,
        index_repo_activity,
        run_repo_coder_activity,
        push_repo_changes_activity,
        # Sprint 18b — Architect pre-step.
        architect_repo_task_activity,
        # Sprint 18c — Critic post-step + continuation loop.
        critic_repo_task_activity,
    )


# Sprint 18c: hard cap on continuation Coder passes after the initial
# Coder run. Total Coder iterations across initial + N continuations
# stays bounded at MAX_ITERATIONS * (1 + MAX_CONTINUATIONS) = 30 * 3 = 90.
# Per D4 (Reflexion-inspired): more than 2 continuations regresses to
# AutoGPT's hallucination-loop failure mode. Plus a monotone-progress
# check catches "going backwards" continuations early.
MAX_CONTINUATIONS = 2


@dataclass
class RepoTaskInput:
    repo_url: str            # https git URL; must be reachable from worker without auth in v1
    branch: str              # branch / tag / commit-ish to clone
    brief: str               # what the Coder should do
    repo_name: str = ""      # name to register in Neo4j (defaults to URL's last path segment)
    tenant_id: str = "default"
    auto_pr: bool = True     # open a PR via the GitHub App when Coder finishes (default on)
    # Sprint 17 post-deploy fix: bypass the indexer's per-file SHA short-circuit
    # when an extractor-version bump means previously-cached files need to be
    # re-extracted (e.g. Java extractor + Spring routes added in 17). Defaults
    # off so routine repo-task runs stay incremental.
    force_reindex: bool = False


@dataclass
class RepoTaskOutput:
    workflow_id: str
    repo_name: str
    commit_sha: str          # HEAD of the branch we cloned
    files_changed: list[str]
    diff: str                # full unified diff vs commit_sha
    iterations: int
    summary: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    # Auto-PR step output. None means the push step didn't run (auto_pr=False
    # or no files changed) or it ran but failed gracefully (push_error set).
    pr_url: str | None = None
    pr_number: int | None = None
    branch_name: str | None = None
    push_error: str | None = None
    # Sprint 18b — Architect plan dict (asdict(ArchitectRepoOutput)) so the
    # UI can render the narrative + subtasks alongside the diff. None when
    # the Architect step ran but produced no output (degraded path) or
    # when a future opt-out flag is added.
    architect_plan: dict | None = None
    # Sprint 18c — Every Critic verdict observed during the run, in
    # order: index 0 is the initial Coder pass's Critic; indices 1..N
    # are the continuation passes' Critics. Empty list means the Critic
    # step never ran (degraded Architect path or pre-18c replay).
    critic_results: list[dict] = field(default_factory=list)
    # Sprint 18c — How many continuation Coder passes fired (0 if the
    # initial pass passed the Critic on first try). Bounded by
    # MAX_CONTINUATIONS = 2.
    continuation_count: int = 0


@workflow.defn
class RepoTaskWorkflow:
    """Three sequential activities; no branching, no signals (yet).

    Long timeouts: clone can take 30s on a big repo; index can take a
    minute on a large monorepo; the Coder loop can run 5+ minutes on
    a non-trivial change. Numbers below are conservative ceilings —
    workflow heartbeats keep things alive.
    """

    @workflow.run
    async def run(self, inp: RepoTaskInput) -> RepoTaskOutput:
        repo_name = inp.repo_name or inp.repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")

        clone_result = await workflow.execute_activity(
            clone_repo_activity,
            args=[inp.repo_url, inp.branch, workflow.info().workflow_id],
            schedule_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(minutes=2),
        )

        await workflow.execute_activity(
            index_repo_activity,
            args=[clone_result["path"], repo_name, clone_result["commit_sha"], inp.tenant_id, inp.force_reindex],
            schedule_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=2),
        )

        # Sprint 18b: Architect runs BEFORE the Coder. Produces a structured
        # plan (narrative + subtask DAG + acceptance_criteria + risk_notes)
        # that the Coder then consumes as authoritative scope. Architect uses
        # Sonnet 4.6 (planning is reasoning-heavy); Coder stays on Haiku.
        # Failure inside the architect activity surfaces a degraded plan
        # (empty subtasks, narrative = last_text) rather than blowing up the
        # workflow — the Coder will still run, just without the plan.
        arch_result = await workflow.execute_activity(
            architect_repo_task_activity,
            args=[
                clone_result["path"], repo_name, inp.brief, inp.tenant_id,
                workflow.info().workflow_id,
            ],
            schedule_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=2),
        )

        coder_result = await workflow.execute_activity(
            run_repo_coder_activity,
            args=[
                clone_result["path"], repo_name, inp.brief, inp.tenant_id,
                workflow.info().workflow_id, arch_result,
            ],
            schedule_to_close_timeout=timedelta(minutes=20),
            heartbeat_timeout=timedelta(minutes=3),
        )

        # Sprint 18c: Critic + continuation loop. After every Coder pass
        # (initial + up to MAX_CONTINUATIONS continuations), run the Critic
        # against the diff. If verdict="incomplete" AND budget remains AND
        # progress is monotone, fire another Coder pass with the Critic's
        # structured handoff doc as the brief. The previous Coder's edits
        # persist on disk in the cloned repo (Temporal heartbeats keep the
        # workspace pinned for the activity worker), so the new pass adds
        # incrementally rather than rewriting from scratch.
        critic_results: list[dict] = []
        continuation_count = 0
        while continuation_count <= MAX_CONTINUATIONS:
            critic_result = await workflow.execute_activity(
                critic_repo_task_activity,
                args=[
                    clone_result["path"], repo_name, arch_result,
                    coder_result["diff"], coder_result["files_with_content"],
                    inp.tenant_id, workflow.info().workflow_id,
                ],
                schedule_to_close_timeout=timedelta(minutes=10),
                heartbeat_timeout=timedelta(minutes=2),
            )
            critic_results.append(critic_result)

            if critic_result["verdict"] == "complete":
                break
            if continuation_count == MAX_CONTINUATIONS:
                # Cap reached — ship anyway. The PR footer will note the
                # known gaps for human triage (D4: bounded autonomy).
                break

            # Monotone-progress guard: continuation N+1 must satisfy
            # STRICTLY MORE checklist items than continuation N. If the
            # new pass regressed (or held steady), terminate early — we'd
            # otherwise burn another Sonnet judge call to learn nothing.
            if continuation_count > 0:
                prev = critic_results[-2]
                prev_passed = len(prev.get("passed_criteria") or [])
                curr_passed = len(critic_result.get("passed_criteria") or [])
                if curr_passed <= prev_passed:
                    break

            # Continuation Coder pass. Same activity — the Critic's
            # `continuation_prompt` IS the brief (it's a structured
            # markdown doc per D7). The Architect plan is threaded through
            # unchanged so the new pass still sees the original
            # acceptance_criteria as authoritative scope.
            coder_result = await workflow.execute_activity(
                run_repo_coder_activity,
                args=[
                    clone_result["path"], repo_name,
                    critic_result["continuation_prompt"], inp.tenant_id,
                    workflow.info().workflow_id, arch_result,
                ],
                schedule_to_close_timeout=timedelta(minutes=20),
                heartbeat_timeout=timedelta(minutes=3),
            )
            continuation_count += 1

        # Auto-PR step. Skipped when the request opted out, when nothing
        # actually changed, or when no installation has access — all
        # surfaced via push_error in the output rather than failing the
        # workflow, since the diff is still useful even if the push fails.
        pr_url: str | None = None
        pr_number: int | None = None
        branch_name: str | None = None
        push_error: str | None = None

        files_with_content = coder_result.get("files_with_content") or []

        # Sprint 18c: when continuation passes ran, the final Critic verdict
        # may still flag gaps (cap reached, or monotone-progress terminated).
        # Append a "Known gaps" footer to the brief so the PR body surfaces
        # them inline for human triage. push_repo_changes_activity renders
        # `brief` verbatim into the PR body, so this is the lightest-touch
        # injection point.
        push_brief = inp.brief
        if continuation_count > 0 and critic_results:
            final = critic_results[-1]
            failed = final.get("failed_criteria") or []
            gates = final.get("gate_failures") or []
            if final.get("verdict") == "incomplete" and (failed or gates):
                footer_lines = [
                    "",
                    "---",
                    f"## Known gaps (after {continuation_count} continuation pass"
                    f"{'es' if continuation_count != 1 else ''})",
                    "",
                    "The Critic flagged the following items as still missing "
                    "after the configured continuation budget was exhausted:",
                    "",
                ]
                for cf in failed[:20]:
                    footer_lines.append(f"- {cf.get('criterion', '')}: {cf.get('evidence', '')}")
                if len(failed) > 20:
                    footer_lines.append(f"- ...and {len(failed) - 20} more")
                if gates:
                    footer_lines.append("")
                    footer_lines.append("### Gate failures")
                    for gf in gates[:10]:
                        ln = f":{gf.get('line')}" if gf.get("line") else ""
                        footer_lines.append(
                            f"- [{gf.get('tool', '')}] `{gf.get('file', '')}`{ln}"
                            f" — {gf.get('message', '')}"
                        )
                    if len(gates) > 10:
                        footer_lines.append(f"- ...and {len(gates) - 10} more")
                push_brief = inp.brief + "\n" + "\n".join(footer_lines)

        if not inp.auto_pr:
            push_error = "auto_pr disabled"
        elif not files_with_content:
            push_error = "no files changed"
        else:
            push_result = await workflow.execute_activity(
                push_repo_changes_activity,
                args=[
                    inp.repo_url,
                    files_with_content,
                    workflow.info().workflow_id,
                    push_brief,
                    inp.tenant_id,
                ],
                schedule_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(minutes=1),
            )
            pr_url = push_result.get("pr_url")
            pr_number = push_result.get("pr_number")
            branch_name = push_result.get("branch_name")
            push_error = push_result.get("error")

        return RepoTaskOutput(
            workflow_id=workflow.info().workflow_id,
            repo_name=repo_name,
            commit_sha=clone_result["commit_sha"],
            files_changed=coder_result["files_changed"],
            diff=coder_result["diff"],
            iterations=coder_result["iterations"],
            summary=coder_result["summary"],
            tokens_in=coder_result["_tokens_in"],
            tokens_out=coder_result["_tokens_out"],
            cost_usd=coder_result["_cost_usd"],
            pr_url=pr_url,
            pr_number=pr_number,
            branch_name=branch_name,
            push_error=push_error,
            architect_plan=arch_result,
            critic_results=critic_results,
            continuation_count=continuation_count,
        )
