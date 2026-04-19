import os
import base64
from dotenv import load_dotenv

load_dotenv()

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")

# TLS is required for Temporal Cloud; plaintext for local docker-compose.
TEMPORAL_TLS = os.getenv("TEMPORAL_TLS", "false").lower() == "true"

def _decode(v: str | None) -> str | None:
    """Accept either raw PEM or base64-encoded PEM (easier to stuff in env/SSM)."""
    if not v:
        return None
    if v.startswith("-----BEGIN"):
        return v
    try:
        return base64.b64decode(v).decode()
    except Exception:
        return v

TEMPORAL_CLIENT_CERT = _decode(os.getenv("TEMPORAL_CLIENT_CERT"))
TEMPORAL_CLIENT_KEY = _decode(os.getenv("TEMPORAL_CLIENT_KEY"))

PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/agentdb")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set — check .env or Secrets Manager binding")
if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY is not set — check .env or Secrets Manager binding")

QUEUES = {
    "ba": "ba-tasks",
    "architect": "architect-tasks",
    "se": "se-tasks",
    "estimator": "estimator-tasks",
    "reviewer": "reviewer-tasks",
    "researcher": "researcher-tasks",
    "documenter": "documenter-tasks",
}
