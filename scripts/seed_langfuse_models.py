"""Seed Langfuse model definitions from app/router.py MODELS catalogue.

Why this script exists:
    Langfuse computes per-trace cost from `model` field + a registered model
    definition. Without registration, costs show $0.00 in the UI. We keep
    pricing canonically in app/router.py (it drives both routing decisions
    AND cost telemetry inside the swarm) and project it into Langfuse here
    rather than maintaining two sources.

When to run:
    - First-time Langfuse deploy
    - After editing pricing in app/router.py MODELS
    - After adding a new model to the catalogue

Idempotent. Re-runs upsert via Langfuse's /api/public/models endpoint —
matching by modelName + matchPattern.

Required env (already populated by terraform secrets when running on ECS,
or via .env locally):
    LANGFUSE_HOST          e.g. https://le-XXXX.ecs.ap-southeast-2.on.aws
    LANGFUSE_PUBLIC_KEY    pk-lf-...
    LANGFUSE_SECRET_KEY    sk-lf-...
"""
from __future__ import annotations

import os
import sys
from base64 import b64encode

import httpx

# Resolve the swarm package even when run from scripts/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.router import MODELS  # noqa: E402


def main() -> int:
    host = (os.getenv("LANGFUSE_HOST") or "").rstrip("/")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or ""

    if not (host and public_key and secret_key):
        print("ERROR: LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY must all be set",
              file=sys.stderr)
        return 1
    if public_key == "UNSET" or secret_key == "UNSET":
        print("ERROR: Langfuse keys still 'UNSET' — set them in tfvars + apply first",
              file=sys.stderr)
        return 1

    auth = "Basic " + b64encode(f"{public_key}:{secret_key}".encode()).decode()
    url = f"{host}/api/public/models"
    headers = {"Authorization": auth, "Content-Type": "application/json"}

    failures = 0
    with httpx.Client(timeout=20.0) as client:
        for key, spec in MODELS.items():
            # Trace `model` field in observability.py is "<provider>.<model>".
            # Anchor the regex so claude-opus-4-7 doesn't also match
            # claude-opus-4-7-something-else.
            trace_name = f"{spec.provider}.{spec.model}"
            payload = {
                "modelName": trace_name,
                "matchPattern": f"(?i)^{trace_name.replace('.', chr(92) + '.')}$",
                "unit": "TOKENS",
                "inputPrice": spec.input_usd_per_mtok / 1_000_000,
                "outputPrice": spec.output_usd_per_mtok / 1_000_000,
            }
            r = client.post(url, json=payload, headers=headers)
            if r.status_code in (200, 201):
                print(f"  OK   {trace_name}  in=${spec.input_usd_per_mtok}/M  out=${spec.output_usd_per_mtok}/M")
            elif r.status_code == 409:
                # Model already exists at this name — Langfuse versions
                # entries by startDate so the new POST creates a new active
                # version. 409 means a duplicate within the same minute,
                # not a real failure.
                print(f"  SKIP {trace_name}  (already current)")
            else:
                failures += 1
                print(f"  FAIL {trace_name}  HTTP {r.status_code}: {r.text[:200]}",
                      file=sys.stderr)

    if failures:
        print(f"\n{failures} model(s) failed to seed", file=sys.stderr)
        return 2
    print(f"\nSeeded {len(MODELS)} models into Langfuse at {host}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
