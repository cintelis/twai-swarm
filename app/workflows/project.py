"""
ProjectWorkflow: the durable orchestrator.

Pipeline:
    BA + Researcher  (parallel)
        ↓
    Architect
        ↓
    [HUMAN APPROVAL GATE]   <-- new: workflow sleeps until POST /projects/{id}/approve
        ↓
    SE
        ↓
    Estimator               <-- new: effort + cost estimation
        ↓
    Reviewer
        ↓
    Documenter

The approval gate uses @workflow.signal. The workflow durably waits for the
signal with a 24h timeout -- no polling, no wasted compute. If you don't
approve in 24h the workflow auto-rejects.

Workflow determinism rules still apply: no direct I/O, all side effects
via activities, no datetime.now() or random.
"""
from dataclasses import dataclass
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.activities import (
        AgentTaskInput,
        AgentTaskResult,
        create_project_record,
        create_task_record,
        run_agent_activity,
    )
    from app import config

@dataclass
class ProjectInput:
    name: str
    brief: str
    # If True, skip the human approval gate (useful for unattended runs + tests).
    auto_approve: bool = False

@dataclass
class ProjectOutput:
    project_id: str
    final_docs: dict
    task_count: int
    approved: bool
    rejection_reason: str | None = None

@workflow.defn
class ProjectWorkflow:

    def __init__(self) -> None:
        self._approved: bool = False
        self._rejected: bool = False
        self._rejection_reason: str | None = None

    # ─── Signals (called from API via Temporal client) ──────────────────────

    @workflow.signal
    def approve(self) -> None:
        """Human signed off -- continue from the gate."""
        self._approved = True

    @workflow.signal
    def reject(self, reason: str = "") -> None:
        """Human rejected -- workflow short-circuits to done."""
        self._rejected = True
        self._rejection_reason = reason

    # ─── Queries (read-only inspection from API) ────────────────────────────

    @workflow.query
    def status(self) -> dict:
        return {
            "approved": self._approved,
            "rejected": self._rejected,
            "rejection_reason": self._rejection_reason,
        }

    # ─── Main ──────────────────────────────────────────────────────────────

    @workflow.run
    async def run(self, inp: ProjectInput) -> ProjectOutput:
        wf_id = workflow.info().workflow_id

        project_id = await workflow.execute_activity(
            create_project_record,
            args=[inp.name, inp.brief, wf_id],
            start_to_close_timeout=timedelta(seconds=10),
        )

        async def run_step(
            role: str,
            title: str,
            description: str,
            parent: str | None = None,
            complexity: int = 1,
        ) -> AgentTaskResult:
            task_input = AgentTaskInput(
                project_id=project_id,
                role=role,
                title=title,
                description=description,
                parent_task_id=parent,
                complexity_hint=complexity,
            )
            task_id = await workflow.execute_activity(
                create_task_record,
                task_input,
                task_queue=config.QUEUES[role],
                start_to_close_timeout=timedelta(seconds=10),
            )
            return await workflow.execute_activity(
                run_agent_activity,
                args=[task_id, task_input],
                task_queue=config.QUEUES[role],
                # Architect + coder can run 3-8 min with web_search / 16K
                # output; BA + researcher 1-3 min. Generous ceiling so real
                # work finishes; the background heartbeat in activities.py
                # keeps us honest about whether the worker is actually alive.
                start_to_close_timeout=timedelta(minutes=15),
                heartbeat_timeout=timedelta(minutes=2),
            )

        # Phase 1: discovery (parallel)
        import asyncio
        ba_future = run_step("ba", "Extract requirements", inp.brief)
        research_future = run_step("researcher", "Background research", inp.brief)
        ba_result, _research_result = await asyncio.gather(ba_future, research_future)

        # Phase 2: design
        arch_result = await run_step(
            "architect",
            "System design",
            f"Design a system for: {inp.brief}",
            parent=ba_result.task_id,
            complexity=2,
        )

        # ─── Approval gate ────────────────────────────────────────────────
        # Skip if caller opted out (e.g. tests, unattended batch jobs).
        if not inp.auto_approve:
            workflow.logger.info("Awaiting human approval after architect...")
            try:
                await workflow.wait_condition(
                    lambda: self._approved or self._rejected,
                    timeout=timedelta(hours=24),
                )
            except TimeoutError:
                return ProjectOutput(
                    project_id=project_id,
                    final_docs={},
                    task_count=3,
                    approved=False,
                    rejection_reason="approval timeout (24h)",
                )

            if self._rejected:
                return ProjectOutput(
                    project_id=project_id,
                    final_docs={},
                    task_count=3,
                    approved=False,
                    rejection_reason=self._rejection_reason or "rejected without reason",
                )

        # Phase 3: implementation planning + estimation
        se_result = await run_step(
            "se",
            "Implementation plan",
            "Produce an implementation plan for the architecture above.",
            parent=arch_result.task_id,
        )

        estimate_result = await run_step(
            "estimator",
            "Effort and cost estimates",
            "Estimate effort (hours) and costs for the implementation plan above. "
            "Include realistic risks and your assumptions.",
            parent=se_result.task_id,
        )

        # Phase 4: review + documentation
        review_result = await run_step(
            "reviewer",
            "Review plan + estimates",
            "Review both the implementation plan and the estimates. "
            "Flag gaps, risks, and any estimates that seem unrealistic.",
            parent=estimate_result.task_id,
        )

        doc_result = await run_step(
            "documenter",
            "Write README",
            "Produce a README summarising the project, design, plan, and estimates.",
            parent=review_result.task_id,
        )

        # Phase 5: code scaffold
        # Generates a runnable starter tree from the plan + docs. Output is
        # consumed by GET /projects/{id}/download (zip).
        await run_step(
            "coder",
            "Code scaffold",
            "Generate a runnable starter scaffold for the project based on the architecture, "
            "implementation plan, and README produced by the previous agents. Stub non-trivial "
            "logic with TODO comments — the goal is a green-build skeleton, not finished code.",
            parent=doc_result.task_id,
            complexity=2,
        )

        return ProjectOutput(
            project_id=project_id,
            final_docs=doc_result.output,
            task_count=8,
            approved=True,
        )
