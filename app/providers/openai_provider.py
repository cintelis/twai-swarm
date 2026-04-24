"""
OpenAI provider — used as the fallback when Anthropic / xAI return transient
errors (5xx, 429, timeout, connection error). Default model is `gpt-5.4`,
called via the Responses API with reasoning effort high — matches OpenAI's
"Default model for most coding tasks" guidance.

Why Responses API (not Chat Completions):
- It's the API path OpenAI is investing in for reasoning + tools.
- Server-side `web_search` lives there; we use it when callers pass tools.
- xAI provider already uses Responses API for the tools path, so the
  citation-extraction shape is consistent across providers.
"""
from openai import AsyncOpenAI
from app import config, observability
from . import ProviderResult

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        # 300s mirrors Anthropic / xAI clients — reasoning + web_search on
        # gpt-5.4 can run minutes on complex briefs.
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=300.0)
    return _client


# Map provider-specific tool specs that callers pass through to the OpenAI
# Responses API equivalent. Anything we can't translate is dropped — the
# fallback degrades gracefully rather than crashing on a foreign tool name.
def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        ttype = (t.get("type") or "").lower()
        if ttype.startswith("web_search") or ttype == "x_search":
            # All web-style grounding maps to OpenAI's built-in web_search.
            # Dedupe — primary providers often pass two web_search variants
            # (e.g. xAI sends both web_search + x_search).
            if not any(o.get("type") == "web_search" for o in out):
                out.append({"type": "web_search"})
    return out or None


async def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
) -> ProviderResult:
    client = _get_client()
    translated = _translate_tools(tools)

    kwargs: dict = {
        "model": model,
        "instructions": system,
        "input": user,
        "max_output_tokens": max_tokens,
        # OpenAI's Responses API takes reasoning depth as a separate dial.
        # `high` matches the docs' default-coding example; bump to env-tunable
        # if we end up wanting per-role depth.
        "reasoning": {"effort": "high"},
    }
    if translated:
        kwargs["tools"] = translated

    with observability.generation(
        name=f"openai.{model}",
        model=model,
        provider="openai",
        system=system,
        user=user,
        tools=translated,
    ) as gen:
        resp = await client.responses.create(**kwargs)

        text = (resp.output_text or "").strip()
        if not text:
            raise RuntimeError(
                f"OpenAI Responses returned empty content (model={model}, status={getattr(resp, 'status', None)})"
            )

        # Citation extraction mirrors xai_provider — annotations on output text
        # items carry the URL when web_search fired.
        citations: list[dict] = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                for ann in getattr(content, "annotations", []) or []:
                    url = getattr(ann, "url", None)
                    if url:
                        citations.append({
                            "url": url,
                            "title": getattr(ann, "title", None),
                        })

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
