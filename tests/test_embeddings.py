"""Embedding helper plumbing — no real OpenAI call."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import embeddings


def test_task_to_embedding_text_strips_noise():
    out = {
        "summary": "ok",
        "files": [{"path": "a.py", "content": "x"}],
        "_citations": [{"url": "x"}],
        "verify_stdout_tail": "blah",
        "verify_stderr_tail": "blah",
        "thing": "kept",
    }
    text = embeddings.task_to_embedding_text("coder", "Build it", out)
    assert "summary" in text
    assert "thing" in text
    assert "files" not in text
    assert "_citations" not in text
    assert "verify_stdout_tail" not in text
    assert text.startswith("role=coder\ntitle=Build it\noutput=")


def test_task_to_embedding_text_handles_non_dict_output():
    text = embeddings.task_to_embedding_text("ba", "Extract", "raw text output")
    assert "raw text output" in text


def test_vector_literal_format():
    assert embeddings.vector_literal([1.0, 2.5, -0.001]) == "[1.000000,2.500000,-0.001000]"


@pytest.mark.asyncio
async def test_embed_text_invokes_openai_with_truncation(monkeypatch):
    big_input = "x" * (embeddings.MAX_INPUT_CHARS + 100)
    fake_resp = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * embeddings.EMBEDDING_DIMS)])
    fake_create = AsyncMock(return_value=fake_resp)
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(embeddings, "_get_client", lambda: fake_client)

    vec = await embeddings.embed_text(big_input)
    assert len(vec) == embeddings.EMBEDDING_DIMS
    fake_create.assert_awaited_once()
    args, kwargs = fake_create.await_args
    assert kwargs["model"] == embeddings.EMBEDDING_MODEL
    assert len(kwargs["input"]) == embeddings.MAX_INPUT_CHARS  # truncated


@pytest.mark.asyncio
async def test_embed_text_rejects_empty():
    with pytest.raises(ValueError):
        await embeddings.embed_text("")
