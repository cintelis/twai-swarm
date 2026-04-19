"""One-shot smoke test for Temporal Cloud connectivity.

Run from repo root with the venv active:
    python scripts/smoke_temporal.py
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("XAI_API_KEY", "x")

from app import config
from temporalio.client import Client
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest


async def main() -> None:
    print(f"host={config.TEMPORAL_HOST} ns={config.TEMPORAL_NAMESPACE} tls={config.TEMPORAL_TLS}")
    client = await Client.connect(
        config.TEMPORAL_HOST,
        namespace=config.TEMPORAL_NAMESPACE,
        tls=True,
        api_key=config.TEMPORAL_API_KEY,
    )
    info = await client.workflow_service.get_system_info(GetSystemInfoRequest())
    print(f"OK — server version: {info.server_version}")


if __name__ == "__main__":
    asyncio.run(main())
