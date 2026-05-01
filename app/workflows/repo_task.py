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

from dataclasses import dataclass
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
    )


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

        coder_result = await workflow.execute_activity(
            run_repo_coder_activity,
            args=[clone_result["path"], repo_name, inp.brief, inp.tenant_id, workflow.info().workflow_id],
            schedule_to_close_timeout=timedelta(minutes=20),
            heartbeat_timeout=timedelta(minutes=3),
        )

        # Auto-PR step. Skipped when the request opted out, when nothing
        # actually changed, or when no installation has access — all
        # surfaced via push_error in the output rather than failing the
        # workflow, since the diff is still useful even if the push fails.
        pr_url: str | None = None
        pr_number: int | None = None
        branch_name: str | None = None
        push_error: str | None = None

        files_with_content = coder_result.get("files_with_content") or []
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
                    inp.brief,
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
        )
