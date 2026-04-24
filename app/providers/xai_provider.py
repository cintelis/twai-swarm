"""
xAI provider. The Grok API is OpenAI-chat-completions compatible, so we use
the official `openai` SDK pointed at api.x.ai. This avoids pulling in an
extra xai-sdk dependency.

Note on reasoning models: per xAI docs, Grok 4.x is a reasoning model and
does NOT accept presencePenalty/frequencyPenalty/stop params. Don't add them.
"""
from openai import AsyncOpenAI
from app import config, observability
from . import ProviderResult

_client: AsyncOpenAI | None = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            # Mirrors the Anthropic client — Grok reasoning + web_search can
            # take 3-5 min on complex briefs.
            timeout=300.0,
        )
    return _client

async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
) -> ProviderResult:
    """
    Chat-Completions path when tools is None/empty (cheaper, simpler).
    Responses-API path when tools are passed — required for xAI's server-side
    tools (web_search, x_search). Live Search on Chat Completions is deprecated
    per https://docs.x.ai/docs/tools/web-search.
    """
    client = _get_client()

    with observability.generation(
        name=f"xai.{model}",
        model=model,
        provider="xai",
        system=system,
        user=user,
        tools=tools,
    ) as gen:
        if tools:
            resp = await client.responses.create(
                model=model,
                instructions=system,
                input=user,
                tools=tools,
                max_output_tokens=max_tokens,
            )
            text = (resp.output_text or "").strip()
            if not text:
                raise RuntimeError(
                    f"xAI Responses returned empty content (model={model}, status={getattr(resp, 'status', None)})"
                )
            # Citations (web_search) live on annotations of output text items.
            citations: list[dict] = []
            for item in getattr(resp, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    for ann in getattr(content, "annotations", []) or []:
                        url = getattr(ann, "url", None)
                        if url:
                            citations.append({"url": url, "title": getattr(ann, "title", None)})
            gen.end(
                output=text,
                usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
            )
            return ProviderResult(
                text=text,
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
                finish_reason=getattr(resp, "status", None),
                citations=citations or None,
            )

        resp = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        if choice.message is None or choice.message.content is None:
            raise RuntimeError(
                f"xAI returned empty content (model={model}, finish_reason={choice.finish_reason})"
            )
        gen.end(
            output=choice.message.content,
            usage={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
        )
        return ProviderResult(
            text=choice.message.content,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
            finish_reason=choice.finish_reason,
        )
