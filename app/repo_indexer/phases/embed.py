"""Embeddings bridge phase — generate per-symbol vectors via app.embeddings.

Sprint 14a. Runs AFTER ResolvePhase (so qualified names are stable) and is
opt-in via `--with-embeddings`. NOT part of `DEFAULT_PHASES` — embedding
generation is network-bound and per-symbol, so it can be slow on large
repos. The default scan path stays fast.

Bridge contract
---------------
* Input: `ctx.batch.functions` and `ctx.batch.classes`.
* Per-symbol text: `embed_text.embedding_text_for_function` /
  `embedding_text_for_class`. Pure functions; deterministic across runs.
* Vectorisation: `app.embeddings.embed_text` (existing). It's an `async`
  function on the OpenAI client; we bridge to the sync phase pipeline
  via `asyncio.run` per batch (the embeddings client itself maintains an
  internal connection pool, so spinning up an event loop here is cheap
  relative to the network round-trip).
* Output: one `EmbeddingUpdate` per (kind, qn, vector) appended to
  `ctx.batch.embeddings`. The loader writes them via UNWIND-MATCH-SET
  on the existing Function / Class nodes.

Why not call OpenAI's batch API
-------------------------------
`app.embeddings.embed_text` is single-input, but OpenAI's embeddings
endpoint accepts a list and returns one vector per input. Sprint 14a
keeps the bridge dumb (one symbol, one call) so the only place that
talks to the provider is `app.embeddings`. If batch-throughput becomes
the bottleneck, the right fix is to add an `embed_batch` to
`app.embeddings.py`, not to fan out from here.

Defense in depth
----------------
The phase is opt-in via `ctx.embed_enabled`. The CLI only adds it to
the phase tuple when `--with-embeddings` is set, but if a future
caller plugs it into a custom phase list and forgets to flip the flag,
the phase short-circuits. Same pattern as the SHA short-circuit guard
in ParsePhase.
"""
from __future__ import annotations

import asyncio
import time

from ..actions import EmbeddingUpdate
from ..embed_text import embedding_text_for_class, embedding_text_for_function
from ..runner import PhaseContext


# Per-batch size for the synchronous async-bridge. Keep small enough that
# a network blip on one symbol doesn't lose 1000+ vectors of work but big
# enough to amortize the asyncio.run + client setup cost. 32 is roughly
# what the OpenAI client recommends for sub-second turnaround per batch
# at our typical input size (~1KB per symbol).
EMBED_BATCH_SIZE = 32

# Progress emission cadence — the phase prints after every batch so a
# 5000-symbol scan doesn't look hung.
PROGRESS_EVERY_N_BATCHES = 1


async def _embed_batch_async(texts: list[str]) -> list[list[float]]:
    """Vectorise a batch of texts via app.embeddings.embed_text.

    Today this just `gather`s a per-text call — `app.embeddings` exposes
    `embed_text(str)` only. When/if a true batch endpoint lands there,
    swap this to call it directly.
    """
    from app.embeddings import embed_text
    return await asyncio.gather(*(embed_text(t) for t in texts))


class EmbedPhase:
    """Generate embeddings for every Function + Class in the batch."""

    name = "embed"

    def run(self, ctx: PhaseContext) -> None:
        # Defense-in-depth: even if the phase is in the tuple, only run
        # when explicitly enabled. The CLI flag is the primary gate.
        if not getattr(ctx, "embed_enabled", False):
            return

        # Nothing to embed → fast no-op (also handles the empty-batch case
        # cleanly; tests rely on it).
        if not ctx.batch.functions and not ctx.batch.classes:
            return

        # All chunks run under one asyncio.run so the cached AsyncOpenAI
        # client's connection pool stays bound to a single event loop.
        # A previous version did asyncio.run per-chunk, which left the
        # cached client referencing closed-loop sockets on chunk N+1 →
        # "Event loop is closed" once SDK retries spilled across chunks.
        asyncio.run(self._run(ctx))

    async def _run(self, ctx: PhaseContext) -> None:
        start = time.monotonic()

        # Build parallel arrays of (kind, qn, text). Maintaining order
        # lets us pair response vectors back up by index — the embedder
        # preserves input order in its output.
        kinds: list[str] = []
        qns: list[str] = []
        texts: list[str] = []
        for fn in ctx.batch.functions:
            kinds.append("function")
            qns.append(fn.qualified_name)
            texts.append(embedding_text_for_function(fn))
        for cls in ctx.batch.classes:
            kinds.append("class")
            qns.append(cls.qualified_name)
            texts.append(embedding_text_for_class(cls))

        if not texts:
            return

        tenant_id = ctx.repo.tenant_id
        repo_name = ctx.repo.name
        total = len(texts)
        batch_count = 0
        embedded = 0

        for offset in range(0, total, EMBED_BATCH_SIZE):
            chunk_texts = texts[offset:offset + EMBED_BATCH_SIZE]
            chunk_kinds = kinds[offset:offset + EMBED_BATCH_SIZE]
            chunk_qns = qns[offset:offset + EMBED_BATCH_SIZE]

            vectors = await _embed_batch_async(chunk_texts)

            for kind, qn, vec in zip(chunk_kinds, chunk_qns, vectors):
                ctx.batch.embeddings.append(EmbeddingUpdate(
                    repo=repo_name,
                    tenant_id=tenant_id,
                    target_kind=kind,  # type: ignore[arg-type]
                    qualified_name=qn,
                    embedding=tuple(vec),
                ))

            batch_count += 1
            embedded += len(chunk_texts)

            if batch_count % PROGRESS_EVERY_N_BATCHES == 0:
                elapsed = time.monotonic() - start
                ctx.progress(
                    f"[indexer] embedded {embedded}/{total} symbols "
                    f"({batch_count} batches) in {elapsed:.2f}s"
                )

        elapsed = time.monotonic() - start
        ctx.progress(
            f"[indexer] embedded {embedded} symbols ({batch_count} batches) "
            f"in {elapsed:.2f}s"
        )
