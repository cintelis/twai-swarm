"""
xAI provider. The Grok API is OpenAI-chat-completions compatible, so we use
the official `openai` SDK pointed at api.x.ai. This avoids pulling in an
extra xai-sdk dependency.

Note on reasoning models: per xAI docs, Grok 4.x is a reasoning model and
does NOT accept presencePenalty/frequencyPenalty/stop params. Don't add them.
"""
from openai import AsyncOpenAI
from app import config
from . import ProviderResult

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            timeout=60.0,
        )
    return _client

async def complete(model: str, system: str, user: str, max_tokens: int = 2048) -> ProviderResult:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    choice = resp.choices[0]
    if choice.message is None or choice.message.content is None:
        raise RuntimeError(
            f"xAI returned empty content (model={model}, finish_reason={choice.finish_reason})"
        )
    return ProviderResult(
        text=choice.message.content,
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
        finish_reason=choice.finish_reason,
    )
