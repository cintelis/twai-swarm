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
    `tools` accepted for signature symmetry with xai_provider; not yet wired
    to Anthropic's native tool-use surface. When we enable web_search for
    Claude roles, branch on `tools` here and pass to messages.create(tools=...).
    """
    _ = tools  # explicit ignore until we wire Anthropic tool use
    client = _get_client()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return ProviderResult(
        text=resp.content[0].text,
        tokens_in=resp.usage.input_tokens,
        tokens_out=resp.usage.output_tokens,
        finish_reason=resp.stop_reason,
    )
