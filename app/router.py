"""
Model router -- provider-aware.

Shape: role + complexity -> RouteDecision (provider, model, pricing).

Keep the decision table explicit. When someone asks 'why did Grok get picked
for researcher?', you want to read one file, not trace through heuristics.

Cost data lives here (not in provider adapters) so routing decisions and
cost observability share a single source of truth.
"""
from dataclasses import dataclass
from typing import Literal

Role = Literal["ba", "architect", "se", "estimator", "reviewer", "researcher", "documenter", "coder"]
Provider = Literal["anthropic", "xai", "openai"]
Tier = Literal["fast", "mid", "flagship"]

@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str                    # exact API model string
    tier: Tier
    input_usd_per_mtok: float
    output_usd_per_mtok: float

# Catalogue. Add new models here; everything else picks them up.
# Pricing current as of Apr 2026. grok-4.20 output rate confirmed at
# https://docs.x.ai/docs/models ($2.00 input / $0.20 cached / $6.00 output per M tokens).
MODELS: dict[str, ModelSpec] = {
    # Anthropic
    "haiku":  ModelSpec("anthropic", "claude-haiku-4-5",  "fast",     1.00,  5.00),
    "sonnet": ModelSpec("anthropic", "claude-sonnet-4-6", "mid",      3.00, 15.00),
    "opus":   ModelSpec("anthropic", "claude-opus-4-7",   "flagship", 15.00, 75.00),

    # xAI -- grok-4.20 is their current flagship (2M context, reasoning + tools).
    # Using the base alias so we auto-track stable releases; pin to
    # "grok-4.20-<date>" (e.g. grok-4.20-0309-non-reasoning) for reproducibility.
    # Naming convention is all-dashes: grok-4-1-fast (NOT grok-4.1-fast).
    "grok":          ModelSpec("xai", "grok-4.20",           "flagship", 2.00, 6.00),
    "grok-fast":     ModelSpec("xai", "grok-4-1-fast",       "fast",     0.20, 0.50),
    # Reasoning variant required for server-side tools (web_search, x_search)
    # via the Responses API. Same per-token pricing as the flagship; the
    # extra cost is tool invocations ($5 / 1k calls).
    "grok-research": ModelSpec("xai", "grok-4.20-reasoning", "flagship", 2.00, 6.00),

    # OpenAI — fallback only. Per OpenAI's "Default model for most coding tasks"
    # guidance, gpt-5.4 is the right default for both general + coding work.
    # Pricing here is a placeholder until we confirm gpt-5.4's official rates;
    # cost telemetry stays roughly accurate as long as input+output orders of
    # magnitude are right. Update from platform.openai.com/docs/pricing.
    "gpt54": ModelSpec("openai", "gpt-5.4", "flagship", 5.00, 20.00),
}

# Fallback chain: when a primary provider raises a transient error, the
# runner walks this list to find a working substitute. Empty list = no
# fallback, raise upstream. Coder is intentionally NOT extended — the
# agentic loop uses Anthropic-SDK-specific helpers (beta_async_tool +
# tool_runner) and falls back to the in-process oneshot coder instead.
FALLBACK_CHAIN: dict[Provider, list[str]] = {
    "anthropic": ["gpt54"],
    "xai":       ["gpt54"],
    "openai":    [],   # already the fallback target; nowhere further to go
}

# Default model key per role. Mix providers freely.
ROLE_DEFAULTS: dict[Role, str] = {
    "ba":         "sonnet",     # requirements shaping; Sonnet tone + reasoning
    "architect":  "opus",       # design -- worth the spend
    "se":         "sonnet",     # implementation plans
    "estimator":  "grok",       # reasoning + cost-awareness; Grok's strength
    "reviewer":   "grok",       # second opinion from different family
    "researcher": "grok-research",  # web_search + x_search (see ROLE_TOOLS in runner.py)
    "documenter": "grok-fast",      # writing up; speed > nuance
    "coder":      "opus",           # code generation: precision > speed
}

# Escalation: complexity_hint=3 bumps one step along this chain.
ESCALATION: dict[str, str] = {
    "haiku":     "sonnet",
    "sonnet":    "opus",
    "grok-fast": "grok",
    "grok":      "opus",   # cross-provider ceiling for "need max reasoning"
}

@dataclass
class RouteDecision:
    key: str
    spec: ModelSpec
    reason: str

    @property
    def model(self) -> str:
        return self.spec.model

    @property
    def provider(self) -> Provider:
        return self.spec.provider

def route(role: Role, complexity_hint: int = 1) -> RouteDecision:
    """
    complexity_hint: 1 (simple) .. 3 (hard). Agents can raise this based on
    input length, dep count, or prior failure count.
    """
    key = ROLE_DEFAULTS[role]
    reason = f"default for {role}"

    if complexity_hint >= 3:
        escalated = ESCALATION.get(key)
        if escalated and escalated != key:
            reason = f"{role} escalated {key}->{escalated} (complexity=3)"
            key = escalated

    return RouteDecision(key=key, spec=MODELS[key], reason=reason)

def estimate_cost_usd(spec: ModelSpec, tokens_in: int, tokens_out: int) -> float:
    return (
        tokens_in  / 1_000_000 * spec.input_usd_per_mtok +
        tokens_out / 1_000_000 * spec.output_usd_per_mtok
    )
