"""
Provider adapters. One thin module per provider with a uniform shape:

    async def complete(model, system, user, max_tokens) -> ProviderResult

Adding a provider = one file here + one MODELS entry in router.py.
"""
from dataclasses import dataclass

@dataclass
class ProviderResult:
    text: str
    tokens_in: int
    tokens_out: int
    finish_reason: str | None = None
