"""
Embedding helper for context retrieval.

Uses OpenAI text-embedding-3-small (1536 dims, $0.02/1M tokens) — matches
the `vector(1536)` column shape in `task_embeddings` (see bootstrap_db.py).
We're already paying the cost of having an OpenAI key wired up for the
gpt-5.4 fallback provider; embeddings ride on the same client.

Cost shape: ~$0.0002 per workflow (7 task embeddings × ~1500 tokens each).
Negligible compared to the agent LLM calls themselves.
"""
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app import config

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Cap input text at ~2000 tokens worth (OpenAI's limit is 8191 for this
# model). 8000 chars is a safe-ish proxy for English-heavy JSON.
MAX_INPUT_CHARS = 8000

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — embeddings require it. "
                "Set the env var or skip the embedding path."
            )
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)
    return _client


async def embed_text(text: str) -> list[float]:
    """Return a 1536-dim embedding for `text`. Truncates over MAX_INPUT_CHARS."""
    if not text or not isinstance(text, str):
        raise ValueError("embed_text requires a non-empty string")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    client = _get_client()
    resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def task_to_embedding_text(role: str, title: str, output: object) -> str:
    """Turn a completed task into the text we embed.

    Skips noisy/large fields:
    - `files` (Coder's source-tree dump — too noisy for semantic match)
    - `_citations` (provenance, not signal)
    - `verify_stdout_tail` / `verify_stderr_tail` (Coder verify spam)

    Falls back to repr if `output` isn't a dict.
    """
    if isinstance(output, dict):
        payload = {k: v for k, v in output.items()
                   if k not in ("files", "_citations", "verify_stdout_tail", "verify_stderr_tail")}
        body = json.dumps(payload)[:6000]
    else:
        body = str(output)[:6000]
    return f"role={role}\ntitle={title}\noutput={body}"


def vector_literal(vec: list[float]) -> str:
    """Format a vector for pgvector's '[1.0,2.0,...]' input syntax."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
