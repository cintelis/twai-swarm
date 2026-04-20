"""Ping every LLM model the router knows about with a 1-token request.

Catches:
- SDK / kwarg incompatibilities (anthropic 0.39 + httpx 0.28 surfaced here)
- Stale model name strings (grok-4.1-fast vs grok-4-1-fast)
- Provider auth misconfiguration (wrong key, expired key)
- Anything that breaks at import time

Run from repo root with the venv active:
    python scripts/smoke_models.py

Required env vars: ANTHROPIC_API_KEY, XAI_API_KEY.
Exits 0 on full pass, 1 on any failure (single line summary at end either way).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bypass app.config's web-search/Temporal startup checks; this script only
# exercises the LLM providers and doesn't talk to Temporal.
os.environ.setdefault("TEMPORAL_TLS", "false")
os.environ.setdefault("TEMPORAL_HOST", "localhost:7233")
os.environ.setdefault("TEMPORAL_NAMESPACE", "default")

from app.providers import anthropic_provider, xai_provider  # noqa: E402
from app.router import MODELS  # noqa: E402

PROVIDERS = {
    "anthropic": anthropic_provider.complete,
    "xai":       xai_provider.complete,
}


async def ping(key: str, spec) -> tuple[str, bool, str]:
    """Returns (model_key, ok, message)."""
    try:
        fn = PROVIDERS.get(spec.provider)
        if fn is None:
            return key, False, f"unknown provider {spec.provider!r}"
        result = await fn(
            model=spec.model,
            system="You answer with a single word.",
            user="Reply with the word OK and nothing else.",
            max_tokens=8,
        )
        text = (result.text or "").strip()
        ok = bool(text)
        return key, ok, f"{spec.model} → {text!r} ({result.tokens_in}→{result.tokens_out} tok)"
    except Exception as e:
        return key, False, f"{spec.model} → {type(e).__name__}: {e}"


async def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set", file=sys.stderr)
        return 2

    results = await asyncio.gather(*(ping(k, s) for k, s in MODELS.items()))
    failures = [(k, m) for k, ok, m in results if not ok]

    for k, ok, msg in results:
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}] {k:14s} {msg}")

    print()
    if failures:
        print(f"{len(failures)}/{len(results)} model ping(s) failed")
        return 1
    print(f"All {len(results)} models reachable")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
