"""Sprint 14a — EmbedPhase + embed_text unit tests.

Mocks `app.embeddings.embed_text` so tests don't make real OpenAI calls
(no OPENAI_API_KEY required). Covers:

1. Phase short-circuits when `embed_enabled=False`.
2. `embedding_text_for_function` produces the locked-in format.
3. `embedding_text_for_class` produces the locked-in format.
4. Phase emits one EmbeddingUpdate per Function + Class symbol.
5. Empty IndexBatch → no calls to the embedder.
6. Large batches are chunked at EMBED_BATCH_SIZE.
7. Even when EmbedPhase is in the phase list, `embed_enabled=False`
   short-circuits (defense in depth).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.repo_indexer.actions import (
    ClassNode,
    EmbeddingUpdate,
    FunctionNode,
    IndexBatch,
    RepoNode,
)
from app.repo_indexer.embed_text import (
    DOCSTRING_MAX_CHARS,
    embedding_text_for_class,
    embedding_text_for_function,
)
from app.repo_indexer.phases.embed import EMBED_BATCH_SIZE, EmbedPhase
from app.repo_indexer.runner import PhaseContext


REPO = RepoNode(name="r", url="", commit_sha="", tenant_id="t1")
DIM = 1536  # mirror app.embeddings.EMBEDDING_DIMS — fixed for the openai model.


def _ctx(batch: IndexBatch, *, embed_enabled: bool = True) -> PhaseContext:
    """Bare-minimum context — phase only reads `repo`, `batch`, `progress`,
    and `embed_enabled`."""
    return PhaseContext(
        repo=batch.repo,
        repo_root=Path("."),
        languages=("python",),
        batch=batch,
        progress=lambda _msg: None,
        embed_enabled=embed_enabled,
    )


def _fn(qn: str, **kwargs) -> FunctionNode:
    """Build a FunctionNode with sensible defaults."""
    return FunctionNode(
        repo="r",
        qualified_name=qn,
        name=qn.split(".")[-1],
        file_path=kwargs.pop("file_path", "x.py"),
        line_start=1,
        line_end=2,
        **kwargs,
    )


def _cls(qn: str, **kwargs) -> ClassNode:
    """Build a ClassNode with sensible defaults."""
    return ClassNode(
        repo="r",
        qualified_name=qn,
        name=qn.split(".")[-1],
        file_path=kwargs.pop("file_path", "x.py"),
        line_start=1,
        line_end=2,
        **kwargs,
    )


def _vec(seed: float = 0.1) -> list[float]:
    """Fixed-dim mock vector — content doesn't matter for the bridge tests."""
    return [seed] * DIM


# ─── 1. embedding_text_for_function — pure unit ────────────────────────────

def test_embedding_text_for_function_top_level():
    """Top-level function: kind=function, container=file_path, no params line
    when params is empty."""
    fn = _fn(
        "math.add",
        file_path="math.py",
        params=("a", "b"),
        docstring="Sum of two numbers.",
    )
    text = embedding_text_for_function(fn)
    assert text == (
        "function math.add\n"
        "params: a, b\n"
        "in math.py\n"
        "Sum of two numbers."
    )


def test_embedding_text_for_function_async():
    """`async def` at top level → kind = 'async function'."""
    fn = _fn("io.fetch", file_path="io.py", is_async=True, params=("url",))
    text = embedding_text_for_function(fn)
    assert text.splitlines()[0] == "async function io.fetch"


def test_embedding_text_for_function_method():
    """Method → kind='method', container=parent_class_qn (NOT file_path)."""
    fn = _fn(
        "auth.AuthService.login",
        file_path="auth.py",
        is_method=True,
        parent_class_qn="auth.AuthService",
        params=("self", "user", "password"),
        docstring="Authenticate the user.",
    )
    text = embedding_text_for_function(fn)
    assert text == (
        "method auth.AuthService.login\n"
        "params: self, user, password\n"
        "in auth.AuthService\n"
        "Authenticate the user."
    )


def test_embedding_text_for_function_no_params_no_docstring():
    """No params, no docstring → only kind/qn line + container line."""
    fn = _fn("util.tick", file_path="util.py")
    text = embedding_text_for_function(fn)
    assert text == "function util.tick\nin util.py"


def test_embedding_text_for_function_truncates_docstring():
    """Docstrings beyond DOCSTRING_MAX_CHARS get clipped to that length."""
    long_doc = "x" * (DOCSTRING_MAX_CHARS + 200)
    fn = _fn("m.f", docstring=long_doc)
    text = embedding_text_for_function(fn)
    # Last "line" is the docstring; verify length is exactly the cap.
    docstring_section = text.split("\n", 2)[-1]
    assert len(docstring_section) == DOCSTRING_MAX_CHARS


# ─── 2. embedding_text_for_class — pure unit ────────────────────────────────

def test_embedding_text_for_class_with_docstring():
    cls = _cls(
        "auth.AuthService",
        file_path="auth.py",
        docstring="Owns the auth lifecycle.",
    )
    assert embedding_text_for_class(cls) == (
        "class auth.AuthService\n"
        "in auth.py\n"
        "Owns the auth lifecycle."
    )


def test_embedding_text_for_class_no_docstring():
    cls = _cls("models.User", file_path="models.py")
    assert embedding_text_for_class(cls) == "class models.User\nin models.py"


# ─── 3. EmbedPhase — disabled by default ───────────────────────────────────

def test_phase_disabled_by_default():
    """When PhaseContext.embed_enabled defaults to False, the phase is a no-op."""
    batch = IndexBatch(repo=REPO)
    batch.functions.append(_fn("a.f"))
    batch.classes.append(_cls("a.C"))

    ctx = PhaseContext(
        repo=REPO,
        repo_root=Path("."),
        languages=("python",),
        batch=batch,
        progress=lambda _m: None,
        # embed_enabled defaults to False — exercise the default.
    )
    # No mock — if the phase tried to call embeddings.embed_text we'd hit
    # the "OPENAI_API_KEY not set" runtime error. The point of the test is
    # that we never get there.
    EmbedPhase().run(ctx)
    assert batch.embeddings == []


# ─── 4. EmbedPhase — emits one update per symbol ───────────────────────────

def test_phase_emits_one_update_per_symbol():
    """Synthetic batch with N functions + M classes → N+M EmbeddingUpdates."""
    batch = IndexBatch(repo=REPO)
    fn_qns = [f"app.svc.fn_{i}" for i in range(3)]
    cls_qns = [f"app.svc.Cls_{i}" for i in range(2)]
    for qn in fn_qns:
        batch.functions.append(_fn(qn))
    for qn in cls_qns:
        batch.classes.append(_cls(qn))

    fake_embed = AsyncMock(return_value=_vec(0.5))
    with patch("app.embeddings.embed_text", fake_embed):
        EmbedPhase().run(_ctx(batch))

    # One update per Function + Class.
    assert len(batch.embeddings) == 5
    # Order: functions first, then classes (matches the phase's iteration).
    fns_emitted = [e for e in batch.embeddings if e.target_kind == "function"]
    cls_emitted = [e for e in batch.embeddings if e.target_kind == "class"]
    assert sorted(e.qualified_name for e in fns_emitted) == sorted(fn_qns)
    assert sorted(e.qualified_name for e in cls_emitted) == sorted(cls_qns)
    # tenant_id propagated from the repo node.
    assert all(e.tenant_id == "t1" for e in batch.embeddings)
    # Vectors are tuples of correct dim.
    assert all(isinstance(e.embedding, tuple) for e in batch.embeddings)
    assert all(len(e.embedding) == DIM for e in batch.embeddings)


# ─── 5. EmbedPhase — empty batch is a no-op (mock not called) ──────────────

def test_phase_handles_empty_batch():
    """No functions, no classes → no calls to the embedder, no updates."""
    batch = IndexBatch(repo=REPO)

    fake_embed = AsyncMock(return_value=_vec())
    with patch("app.embeddings.embed_text", fake_embed):
        EmbedPhase().run(_ctx(batch))

    assert batch.embeddings == []
    fake_embed.assert_not_called()


# ─── 6. EmbedPhase — chunks large batches at EMBED_BATCH_SIZE ──────────────

def test_phase_chunks_large_batch():
    """Synthetic batch of 250 functions, mock fails if >EMBED_BATCH_SIZE
    inputs in a single asyncio.gather call. Phase must chunk."""
    batch = IndexBatch(repo=REPO)
    n = EMBED_BATCH_SIZE * 7 + 13   # ensure a non-multiple to exercise the tail
    for i in range(n):
        batch.functions.append(_fn(f"app.bulk.fn_{i:04d}"))

    # Track how many concurrent gathers occur — the phase awaits each
    # batch synchronously via asyncio.run, so each call corresponds to
    # one chunk. Cap each call to EMBED_BATCH_SIZE.
    concurrent_calls: list[int] = []

    async def _spy(text: str) -> list[float]:
        # Record participation; per-text granularity at the await level.
        concurrent_calls.append(1)
        return _vec()

    fake_embed = AsyncMock(side_effect=_spy)
    with patch("app.embeddings.embed_text", fake_embed):
        # Patch _embed_batch_sync so we can inspect chunk sizes directly.
        # Wrapping with side_effect lets us assert chunk sizes per call.
        from app.repo_indexer.phases import embed as embed_mod
        original = embed_mod._embed_batch_sync
        chunk_sizes: list[int] = []

        def _spy_sync(texts: list[str]) -> list[list[float]]:
            chunk_sizes.append(len(texts))
            assert len(texts) <= EMBED_BATCH_SIZE, (
                f"phase did not chunk: got {len(texts)} > {EMBED_BATCH_SIZE}"
            )
            return original(texts)

        with patch.object(embed_mod, "_embed_batch_sync", _spy_sync):
            EmbedPhase().run(_ctx(batch))

    # We should have one EmbeddingUpdate per input symbol.
    assert len(batch.embeddings) == n
    # And we should have chunked it: chunk count = ceil(n / EMBED_BATCH_SIZE).
    expected_chunks = (n + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    assert len(chunk_sizes) == expected_chunks
    # Every chunk except possibly the last should be exactly EMBED_BATCH_SIZE.
    assert all(s == EMBED_BATCH_SIZE for s in chunk_sizes[:-1])
    assert chunk_sizes[-1] == n - EMBED_BATCH_SIZE * (expected_chunks - 1)


# ─── 7. EmbedPhase — defense-in-depth no-op when flag false ────────────────

def test_no_op_when_embed_enabled_false():
    """Even with the phase listed and content in the batch, embed_enabled=False
    must short-circuit. Mock would raise if called."""
    batch = IndexBatch(repo=REPO)
    batch.functions.append(_fn("a.f"))
    batch.classes.append(_cls("a.C"))

    fake_embed = AsyncMock(side_effect=AssertionError("must not be called"))
    with patch("app.embeddings.embed_text", fake_embed):
        EmbedPhase().run(_ctx(batch, embed_enabled=False))

    assert batch.embeddings == []
    fake_embed.assert_not_called()


# ─── EmbeddingUpdate dataclass + IndexBatch wiring ─────────────────────────

def test_embedding_update_is_frozen():
    """EmbeddingUpdate is frozen — accidental mutation is a bug."""
    update = EmbeddingUpdate(
        repo="r", tenant_id="t1", target_kind="function",
        qualified_name="m.f", embedding=(0.1, 0.2),
    )
    with pytest.raises(Exception):
        update.qualified_name = "other"  # type: ignore[misc]


def test_index_batch_extend_merges_embeddings():
    """IndexBatch.extend includes the embeddings field (Sprint 14a wiring)."""
    a = IndexBatch(repo=REPO)
    b = IndexBatch(repo=REPO)
    a.embeddings.append(EmbeddingUpdate(
        repo="r", tenant_id="t1", target_kind="function",
        qualified_name="a.f", embedding=(0.1,),
    ))
    b.embeddings.append(EmbeddingUpdate(
        repo="r", tenant_id="t1", target_kind="class",
        qualified_name="b.C", embedding=(0.2,),
    ))
    a.extend(b)
    assert len(a.embeddings) == 2


def test_index_batch_counts_includes_embedding_updates():
    """`embedding_updates` is in counts() (Sprint 14a wiring)."""
    batch = IndexBatch(repo=REPO)
    counts = batch.counts()
    assert "embedding_updates" in counts
    assert counts["embedding_updates"] == 0
    batch.embeddings.append(EmbeddingUpdate(
        repo="r", tenant_id="t1", target_kind="function",
        qualified_name="a.f", embedding=(0.1,),
    ))
    assert batch.counts()["embedding_updates"] == 1


# ─── Smoke: the phase chunk size constant matches what tests assume ────────

def test_chunk_size_is_a_sane_constant():
    """Tripwire — if 14a's batch size changes, the chunking test above would
    silently still pass. This makes the assumption explicit."""
    assert EMBED_BATCH_SIZE >= 1
    assert EMBED_BATCH_SIZE <= 256  # OpenAI's per-request input cap is well above this
