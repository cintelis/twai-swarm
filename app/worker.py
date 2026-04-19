"""
Worker process.

Adds a minimal /health endpoint on port 8001 so ECS (regular Fargate service,
not Express Mode) can do liveness checks via TCP or HTTP health check.

For the lean setup, ONE container image registers for ALL role queues.
When any role's throughput becomes a bottleneck, run this image as a separate
ECS service with TEMPORAL_QUEUES env var narrowed to specific roles.
"""
import asyncio
import os
from aiohttp import web

from temporalio.client import Client
from temporalio.worker import Worker

from app import config
from app.workflows import ProjectWorkflow
from app.activities import (
    create_project_record,
    create_task_record,
    run_agent_activity,
)

ACTIVITIES = [create_project_record, create_task_record, run_agent_activity]
WORKFLOWS = [ProjectWorkflow]

_state = {"ready": False, "workers_running": 0}

async def health(request):
    if _state["ready"] and _state["workers_running"] > 0:
        return web.json_response({"status": "ok", "workers": _state["workers_running"]})
    return web.json_response({"status": "starting"}, status=503)

async def start_health_server():
    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8001)
    await site.start()
    print("[health] listening on :8001")
    return runner

def _connect_kwargs():
    """Temporal Cloud uses TLS + API key; local dev uses plaintext."""
    if not config.TEMPORAL_TLS:
        return {}
    return {"tls": True, "api_key": config.TEMPORAL_API_KEY}

async def main():
    config.validate_runtime()
    queues_env = os.getenv("TEMPORAL_QUEUES", "").strip()
    if queues_env:
        roles = [r.strip() for r in queues_env.split(",") if r.strip()]
        queues = {r: config.QUEUES[r] for r in roles if r in config.QUEUES}
    else:
        queues = dict(config.QUEUES)

    print(f"[worker] connecting to {config.TEMPORAL_HOST} namespace={config.TEMPORAL_NAMESPACE}")
    client = await Client.connect(
        config.TEMPORAL_HOST,
        namespace=config.TEMPORAL_NAMESPACE,
        **_connect_kwargs(),
    )

    workers = []
    for role, queue in queues.items():
        w = Worker(
            client,
            task_queue=queue,
            workflows=WORKFLOWS,
            activities=ACTIVITIES,
            max_concurrent_activities=5,
        )
        workers.append(w)
        print(f"[worker] registered on queue: {queue} (role={role})")

    default_worker = Worker(
        client,
        task_queue="project-workflows",
        workflows=WORKFLOWS,
        activities=ACTIVITIES,
    )
    workers.append(default_worker)
    print("[worker] registered on queue: project-workflows")

    _state["workers_running"] = len(workers)

    health_runner = await start_health_server()
    try:
        run_tasks = [asyncio.create_task(w.run()) for w in workers]
        # At this point every Worker has been instantiated AND its run() task
        # has been scheduled on the event loop and is actively polling its
        # queue. Only now is /health allowed to return "ok".
        _state["ready"] = True
        await asyncio.gather(*run_tasks)
    finally:
        await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
