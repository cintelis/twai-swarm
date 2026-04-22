#!/usr/bin/env python
"""
One-shot backfill: embed every existing task that has output but no
task_embeddings row.

Run from the repo root with PG_DSN + OPENAI_API_KEY set:

    python scripts/backfill_embeddings.py

Cost: ~$0.0002 per workflow (~7 task outputs × ~1500 tokens). For a few
hundred historical tasks: well under $1.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Match smoke_models.py — bypass app.config's Temporal startup check.
os.environ.setdefault("TEMPORAL_TLS", "false")
os.environ.setdefault("TEMPORAL_HOST", "localhost:7233")
os.environ.setdefault("TEMPORAL_NAMESPACE", "default")

import json  # noqa: E402

from app import db, embeddings  # noqa: E402


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 2
    if not os.getenv("PG_DSN"):
        print("PG_DSN is required", file=sys.stderr)
        return 2

    pool = await db.get_pool()
    rows = await pool.fetch(
        """
        SELECT t.id, t.role, t.title, t.output
        FROM tasks t
        LEFT JOIN task_embeddings te ON te.task_id = t.id
        WHERE t.output IS NOT NULL
          AND te.task_id IS NULL
        ORDER BY t.created_at ASC
        """,
    )
    print(f"[backfill] {len(rows)} tasks need embedding")
    if not rows:
        return 0

    ok = 0
    failed = 0
    for r in rows:
        tid = str(r["id"])
        try:
            output = json.loads(r["output"])
            content = embeddings.task_to_embedding_text(r["role"], r["title"], output)
            vec = await embeddings.embed_text(content)
            await db.upsert_task_embedding(tid, content, vec)
            ok += 1
            if ok % 25 == 0:
                print(f"[backfill] {ok}/{len(rows)} embedded")
        except Exception as e:
            failed += 1
            print(f"[backfill] FAIL task {tid}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[backfill] done · {ok} ok · {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
