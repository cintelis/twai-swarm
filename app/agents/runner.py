"""
Agent = role-specific system prompt + structured JSON output.

Each agent is a pure function: (task_input, context) -> dict.
No Temporal imports here -- agents are testable in isolation.
"""
import json
from app import router
from app.providers import anthropic_provider, xai_provider, ProviderResult

SYSTEM_PROMPTS = {
    "ba": """You are a Business Analyst agent. Given a project brief, produce a structured analysis that explicitly separates what the brief actually says from what you are inferring.

Process (think step by step before writing JSON):
1. facts_from_brief — statements directly supported by the brief's wording. Quote or paraphrase. Do NOT add detail the brief doesn't contain.
2. assumptions — anything you must assume to write requirements. Each must be falsifiable (a stakeholder could say "no, that's wrong"). Tag each with a short label like {{"assumption": "...", "id": "A1"}}.
3. requirements — concrete, testable. For each, list the fact/assumption ids it depends on under "depends_on": ["F2","A1"].
4. open_questions — blockers whose answer would meaningfully change the requirements. Phrase as questions a human could answer in one sentence.

Be honest: if the brief is too thin to ground a requirement, push it into open_questions instead of inventing assumptions to fill the gap.

Output as JSON:
{
  "facts_from_brief": [{"id": "F1", "text": "..."}, ...],
  "assumptions":      [{"id": "A1", "text": "..."}, ...],
  "requirements":     [{"text": "...", "depends_on": ["F1","A2"]}, ...],
  "open_questions":   ["...", ...]
}""",

    "architect": """You are a Software Architect. Given requirements, produce a system design:
- components: list of components with responsibility
- data_flow: how data moves between components
- tech_choices: key technology decisions with rationale
Output as JSON: {"components": [...], "data_flow": "...", "tech_choices": [...]}""",

    "se": """You are a Software Engineer. Given a design, produce an implementation plan:
- files: list of files to create/modify with a one-line purpose
- key_functions: signatures of the critical functions
Output as JSON: {"files": [...], "key_functions": [...]}""",

    "estimator": """You are an Estimator. Given an implementation plan, produce effort and cost estimates:
- items: list of {task, hours_low, hours_high, confidence: "low"|"med"|"high"}
- total_hours_range: [low_int, high_int]
- key_risks: factors that could blow the estimate (string list)
- token_cost_estimate_usd: rough LLM/API cost if this involves AI workloads, else 0
- assumptions: what you assumed about team skill, tooling, prior art
Be realistic. A solo engineer shipping to prod is slower than the plan suggests.
Output as JSON: {"items": [...], "total_hours_range": [l, h], "key_risks": [...], "token_cost_estimate_usd": 0, "assumptions": [...]}""",

    "reviewer": """You are a code/design Reviewer. Given prior work, produce:
- issues: list of concrete problems, each with severity (low|med|high)
- suggestions: constructive improvements
Output as JSON: {"issues": [...], "suggestions": [...]}""",

    "researcher": """You are a Researcher. Given a topic, produce:
- findings: bullet list of relevant facts
- sources_needed: what external info would strengthen this
Output as JSON: {"findings": [...], "sources_needed": [...]}""",

    "documenter": """You are a Documenter. Given prior work, produce user-facing docs:
- overview: 2-3 sentence summary
- sections: list of {heading, body} for a README
Output as JSON: {"overview": "...", "sections": [...]}""",
}

def _format_context(context: list[dict]) -> str:
    if not context:
        return "No prior context."
    parts = []
    for c in context:
        parts.append(f"## {c['role'].upper()} — {c['title']}\n{json.dumps(c['output'], indent=2)}")
    return "\n\n".join(parts)

# Provider dispatch. Add a new provider by adding one entry here.
_PROVIDERS = {
    "anthropic": anthropic_provider.complete,
    "xai":       xai_provider.complete,
}

# Per-role grounding tools. Empty/missing = no tools (cheaper path).
# xAI server-side tools live behind the Responses API; Anthropic tool wiring
# is a future step (see anthropic_provider.complete docstring).
ROLE_TOOLS: dict[str, list[dict]] = {
    "researcher": [{"type": "web_search"}, {"type": "x_search"}],
}

async def _complete(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
) -> ProviderResult:
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}")
    return await fn(model, system, user, max_tokens, tools)

async def run_agent(
    role: str,
    task_description: str,
    context: list[dict],
    complexity_hint: int = 1,
) -> dict:
    """Run one agent turn. Returns dict with output + usage + cost."""
    decision = router.route(role, complexity_hint)
    system = SYSTEM_PROMPTS[role]

    user_msg = f"""Task: {task_description}

Prior context from upstream agents:
{_format_context(context)}

Respond with ONLY the JSON object specified in your system prompt. No preamble, no markdown fences."""

    result = await _complete(
        provider=decision.provider,
        model=decision.model,
        system=system,
        user=user_msg,
        tools=ROLE_TOOLS.get(role),
    )

    # Defensive parse -- strip markdown fences if the model added them
    text = result.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        output = json.loads(text)
    except json.JSONDecodeError:
        output = {"raw_text": text, "parse_error": True}

    cost = router.estimate_cost_usd(decision.spec, result.tokens_in, result.tokens_out)

    return {
        "output": output,
        "provider": decision.provider,
        "model": decision.model,
        "model_key": decision.key,
        "route_reason": decision.reason,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cost_usd": round(cost, 6),
        "citations": result.citations or [],
    }
