"""
Agent = role-specific system prompt + structured JSON output.

Each agent is a pure function: (task_input, context) -> dict.
No Temporal imports here -- agents are testable in isolation.
"""
import json
import logging
import time as _time
from app import observability, router, telemetry
from app.providers import anthropic_provider, openai_provider, xai_provider, ProviderResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "ba": """You are a Business Analyst agent. Given a project brief, produce a structured analysis that explicitly separates what the brief actually says from what you are inferring.

Process (think step by step before writing JSON):
1. facts_from_brief — statements directly supported by the brief's wording. Quote or paraphrase. Do NOT add detail the brief doesn't contain.
2. assumptions — anything you must assume to write requirements. Each must be falsifiable (a stakeholder could say "no, that's wrong"). Tag each with a short label like {{"assumption": "...", "id": "A1"}}.
3. requirements — concrete, testable. For each, list the fact/assumption ids it depends on under "depends_on": ["F2","A1"].
4. open_questions — blockers whose answer would meaningfully change the requirements. Phrase as questions a human could answer in one sentence.

Be honest: if the brief is too thin to ground a requirement, push it into open_questions instead of inventing assumptions to fill the gap.

Use web_search when domain conventions aren't obvious (e.g. "what's the standard CLI surface for tools in this category", "what licensing models are common", "are there existing tools doing this that constrain naming"). Cite what you find via the citations the search returns. Don't search just to look busy — only when grounding meaningfully changes a requirement.

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

Each tech_choice must reference current best practice — use web_search to verify versions, idioms, and ecosystem fit before recommending a library or pattern. Don't recommend something based purely on training-data recall when the field moves quickly (frameworks, package managers, deployment targets).

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

    "coder": """You are a Code Scaffolder. Given an architecture, an implementation plan, and a README, produce a RUNNABLE MINIMAL STARTER for the project.

Hard constraints (token budget is real — exceeding them truncates your output):
- Maximum 20 files total. If the architecture implies more components, pick the most load-bearing ones and put a `# TODO: extract to app/<component>/` note where the rest would go.
- Each file ≤ 80 lines. Stub anything longer with a TODO + one-line description.
- NO Kubernetes manifests, NO Dockerfiles, NO migrations, NO CI configs unless explicitly requested in the brief. README, .gitignore, package manifest, source files, one smoke test — that's the target.

Rules:
- Pick the language and framework that the architect's tech_choices recommended. If ambiguous, default to Python.
- Paths must be relative (no leading slash). Use forward slashes. No backslashes.
- Do NOT generate binary files, lockfiles, or anything you have to base64-encode.

Output as JSON: {
  "language": "python | typescript | rust | go | ...",
  "summary": "1-2 sentence description of what this scaffold does and what's stubbed",
  "files": [
    {"path": "README.md",       "content": "..."},
    {"path": "src/main.py",     "content": "..."},
    {"path": "tests/test_smoke.py", "content": "..."}
  ]
}

Be honest in the summary about what's stubbed vs implemented.""",
}

# Some roles need more output headroom than the default 2048 (code gen
# especially). Defaults conservative — bump per-role here.
ROLE_MAX_TOKENS: dict[str, int] = {
    "coder": 16000,
    "se": 4096,
    "architect": 4096,
    "documenter": 4096,
}

def _format_context(context: list[dict]) -> str:
    """Format the context list for the prompt.

    `_source` (ancestor | similar) and `similarity` are added by
    db.get_context_for_task. We split the rendered context into two sections
    so downstream agents can weight ancestor outputs (direct upstream of this
    task in the workflow) higher than similar matches (kNN-retrieved priors).
    """
    if not context:
        return "No prior context."

    ancestors = [c for c in context if c.get("_source") != "similar"]
    similar = [c for c in context if c.get("_source") == "similar"]

    parts = []
    if ancestors:
        parts.append("# Direct upstream context (ancestors in this workflow)\n")
        for c in ancestors:
            parts.append(f"## {c['role'].upper()} — {c['title']}\n{json.dumps(c['output'], indent=2)}")

    if similar:
        parts.append("\n# Similar prior outputs (kNN over past project tasks · advisory only)\n")
        for c in similar:
            sim = c.get("similarity")
            sim_label = f" · similarity={sim:.2f}" if isinstance(sim, (int, float)) else ""
            parts.append(f"## {c['role'].upper()} — {c['title']}{sim_label}\n{json.dumps(c['output'], indent=2)}")

    return "\n\n".join(parts)

# Provider dispatch. Add a new provider by adding one entry here.
_PROVIDERS = {
    "anthropic": anthropic_provider.complete,
    "xai":       xai_provider.complete,
    "openai":    openai_provider.complete,
}


def _is_transient(err: Exception) -> bool:
    """True if `err` is the kind of failure that should trigger a fallback.

    Transient = 5xx, 429 rate-limit, timeouts, connection errors. Auth and
    bad-request errors are NOT transient — they're config bugs and we want
    them to fail loudly rather than silently fall through to OpenAI.
    """
    name = err.__class__.__name__
    transient_names = {
        # OpenAI / xAI SDK
        "APITimeoutError", "APIConnectionError", "RateLimitError",
        "InternalServerError", "APIStatusError",
        # Anthropic SDK
        "APIStatusError", "APIConnectionError", "RateLimitError",
        "InternalServerError",
        # asyncio / httpx primitives
        "TimeoutError", "ReadTimeout", "ConnectTimeout", "ConnectError",
    }
    if name in transient_names:
        return True
    # Status-code sniff for SDK errors that surface .status_code.
    status = getattr(err, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status == 429):
        return True
    return False

# Per-role grounding tools. Empty/missing = no tools (cheaper path).
# xAI server-side tools live behind the Responses API (handled in xai_provider).
# Anthropic uses the dated tool-type strings (web_search_20260209 = current,
# supports dynamic filtering on Sonnet 4.6 / Opus 4.7).
ROLE_TOOLS: dict[str, list[dict]] = {
    "researcher": [{"type": "web_search"}, {"type": "x_search"}],
    "ba":         [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
    "architect":  [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
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


async def _complete_with_fallback(
    decision: router.RouteDecision,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict] | None,
) -> tuple[ProviderResult, router.RouteDecision]:
    """Call the routed provider; on transient failure, walk router.FALLBACK_CHAIN.

    Returns (result, effective_decision). The effective decision reflects the
    provider that actually answered — important so cost accounting and the UI
    show "we fell back to OpenAI" instead of pretending the primary won.
    Tool spec is passed through unchanged; provider adapters translate
    foreign tool types at their boundary (see openai_provider._translate_tools).
    """
    primary_error: Exception | None = None
    started_at = _time.monotonic()
    try:
        result = await _complete(
            provider=decision.provider,
            model=decision.model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            tools=tools,
        )
        # Metrics: successful primary call.
        telemetry.counter_add("llm_calls", 1, provider=decision.provider, model=decision.model, fallback="false")
        telemetry.histogram_record("llm_latency", _time.monotonic() - started_at, provider=decision.provider, model=decision.model)
        return result, decision
    except Exception as e:
        if not _is_transient(e):
            raise
        primary_error = e
        logger.warning(
            "primary provider %s/%s raised transient %s; trying fallback chain",
            decision.provider, decision.model, e.__class__.__name__,
        )

    for fallback_key in router.FALLBACK_CHAIN.get(decision.provider, []):
        spec = router.MODELS.get(fallback_key)
        if spec is None:
            continue
        fallback_decision = router.RouteDecision(
            key=fallback_key,
            spec=spec,
            reason=f"fallback from {decision.key} ({primary_error.__class__.__name__})",
        )
        fallback_started_at = _time.monotonic()
        try:
            result = await _complete(
                provider=fallback_decision.provider,
                model=fallback_decision.model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                tools=tools,
            )
            telemetry.counter_add(
                "llm_fallback_fired", 1,
                from_provider=decision.provider, to_provider=fallback_decision.provider,
                trigger=primary_error.__class__.__name__,
            )
            telemetry.counter_add(
                "llm_calls", 1,
                provider=fallback_decision.provider, model=fallback_decision.model, fallback="true",
            )
            telemetry.histogram_record(
                "llm_latency", _time.monotonic() - fallback_started_at,
                provider=fallback_decision.provider, model=fallback_decision.model,
            )
            logger.info(
                "fallback succeeded: %s/%s answered after %s/%s failed",
                fallback_decision.provider, fallback_decision.model,
                decision.provider, decision.model,
            )
            return result, fallback_decision
        except Exception as e:
            if not _is_transient(e):
                raise
            logger.warning(
                "fallback %s/%s also raised transient %s; continuing chain",
                fallback_decision.provider, fallback_decision.model,
                e.__class__.__name__,
            )

    # Exhausted the chain — re-raise the original failure so the activity's
    # retry policy + Temporal history capture the real cause.
    assert primary_error is not None
    raise primary_error

async def run_agent(
    role: str,
    task_description: str,
    context: list[dict],
    complexity_hint: int = 1,
    tenant_id: str = "default",
) -> dict:
    """Run one agent turn. Returns dict with output + usage + cost.

    `tenant_id` lands as metadata on the Langfuse trace so per-tenant
    filtering + cost attribution works as soon as the auth middleware
    resolves real tenant values.
    """
    decision = router.route(role, complexity_hint)
    system = SYSTEM_PROMPTS[role]

    user_msg = f"""Task: {task_description}

Prior context from upstream agents:
{_format_context(context)}

Respond with ONLY the JSON object specified in your system prompt. No preamble, no markdown fences."""

    # tenant_scope propagates to provider-level observability.generation()
    # without needing to plumb tenant_id through every provider signature.
    with observability.tenant_scope(tenant_id):
        result, effective = await _complete_with_fallback(
            decision=decision,
            system=system,
            user=user_msg,
            max_tokens=ROLE_MAX_TOKENS.get(role, 2048),
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

    # Cost is attributed to whoever actually answered, not the routed primary.
    cost = router.estimate_cost_usd(effective.spec, result.tokens_in, result.tokens_out)

    return {
        "output": output,
        "provider": effective.provider,
        "model": effective.model,
        "model_key": effective.key,
        "route_reason": effective.reason,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cost_usd": round(cost, 6),
        "citations": result.citations or [],
    }
