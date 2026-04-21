#!/usr/bin/env python
"""
Manual smoke test — exercises the agentic Coder against real Claude Opus 4.7
using the python-fastapi-postgres template.

Not in CI (would burn API credits every push). Run locally to sanity-check
the loop end-to-end:

    python scripts/smoke_coder.py

Prints the final verify outcome, iteration count, and the list of files
the model produced. Exits 0 iff verify passed.
"""
from __future__ import annotations

import asyncio
import json
import sys

from app.agents import coder_agentic


async def main() -> int:
    brief = (
        "A FastAPI + Postgres REST API that exposes a `/tasks` resource with "
        "CRUD: list, create, get-by-id, delete. Each task has a title (required "
        "string), description (optional string), and a done boolean. Use "
        "SQLAlchemy 2.0 async, Alembic migrations, and pytest for a minimal "
        "smoke test. Keep the example Item model if it helps; otherwise replace it."
    )
    architecture = {
        "components": [
            {"name": "FastAPI app", "responsibility": "HTTP + routing"},
            {"name": "SQLAlchemy async engine", "responsibility": "DB access"},
            {"name": "Alembic", "responsibility": "migrations"},
        ],
        "tech_choices": [
            {"name": "FastAPI", "why": "async ergonomics + type-safe routing"},
            {"name": "PostgreSQL", "why": "relational + JSONB if we ever need it"},
        ],
    }

    result = await coder_agentic.run_agentic_coder(
        workflow_id="smoke-coder-manual",
        brief=brief,
        architecture=architecture,
        se_plan=None,
        documenter=None,
    )

    print("─" * 60)
    print(f"template: {result['template']} (reason: {result['template_reason']})")
    print(f"iterations: {result['iterations']}")
    print(f"stop_reason: {result['stop_reason']}")
    print(f"verify exit: {result['verify_exit_code']} (passed={result['verify_passed']})")
    print(f"tool calls: {json.dumps(result['tool_calls'])}")
    print(f"tokens in/out: {result['_tokens_in']}/{result['_tokens_out']}")
    print(f"cost USD: ${result['_cost_usd']:.4f}")
    print(f"files produced: {len(result['files'])}")
    for f in result["files"]:
        print(f"  - {f['path']} ({len(f['content'])} chars)")
    print("─" * 60)
    if not result["verify_passed"]:
        print("verify stderr tail:")
        print(result["verify_stderr_tail"][-2000:])
    return 0 if result["verify_passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
