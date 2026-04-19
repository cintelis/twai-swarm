from anthropic import AsyncAnthropic
from app import config
from . import ProviderResult

_client: AsyncAnthropic | None = None

def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=60.0)
    return _client

async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
) -> ProviderResult:
    """
    Pass `tools` (e.g. [{"type": "web_search_20260209", "name": "web_search"}])
    to enable Anthropic server-side tools. Web search citations are extracted
    from each text block's `citations` field and bubbled up.
    """
    client = _get_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        kwargs["tools"] = tools

    resp = await client.messages.create(**kwargs)

    # Concatenate every text block; web_search interleaves Claude's narration,
    # server_tool_use blocks, web_search_tool_result blocks, and final cited
    # text. Only the text blocks carry the actual response prose.
    text_parts: list[str] = []
    citations: list[dict] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
            for c in (getattr(block, "citations", None) or []):
                if getattr(c, "type", None) == "web_search_result_location":
                    citations.append({
                        "url": getattr(c, "url", None),
                        "title": getattr(c, "title", None),
                        "cited_text": getattr(c, "cited_text", None),
                    })

    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError(
            f"Anthropic returned no text content (model={model}, stop_reason={resp.stop_reason})"
        )

    return ProviderResult(
        text=text,
        tokens_in=resp.usage.input_tokens,
        tokens_out=resp.usage.output_tokens,
        finish_reason=resp.stop_reason,
        citations=citations or None,
    )
