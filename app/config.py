import os
from dotenv import load_dotenv

load_dotenv()

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")

# TLS is required for Temporal Cloud; plaintext for local docker-compose.
TEMPORAL_TLS = os.getenv("TEMPORAL_TLS", "false").lower() == "true"

# Temporal Cloud API key. Required when TEMPORAL_TLS=true; ignored otherwise.
TEMPORAL_API_KEY = os.getenv("TEMPORAL_API_KEY")

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/agentdb")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

def validate_runtime() -> None:
    """Fail fast if a runtime-required env var is missing.

    Call from API / worker entry points only — NOT at import time, so that
    CI smoke imports, the bootstrap container, and ad-hoc debug sessions
    can `from app import config` without supplying the LLM credentials.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — check .env or Secrets Manager binding")
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is not set — check .env or Secrets Manager binding")
    if TEMPORAL_TLS and not TEMPORAL_API_KEY:
        raise RuntimeError("TEMPORAL_API_KEY is required when TEMPORAL_TLS=true (Temporal Cloud)")

QUEUES = {
    "ba": "ba-tasks",
    "architect": "architect-tasks",
    "se": "se-tasks",
    "estimator": "estimator-tasks",
    "reviewer": "reviewer-tasks",
    "researcher": "researcher-tasks",
    "documenter": "documenter-tasks",
    "coder": "coder-tasks",
}
