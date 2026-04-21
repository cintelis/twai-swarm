"""Shared pytest fixtures. Keep this file lean — most tests don't need state."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The Sprint 6 coder tests import from `app.*` without installing the package.
# Ensure the repo root is on sys.path so `from app.agents.coder_sandbox ...`
# works in CI (`pytest`) and locally (`python -m pytest`).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These tests don't hit the network or the LLM — but app.config imports
# python-dotenv and reads env at import time. Set bland placeholders so
# `from app import config` works regardless of the dev shell.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("XAI_API_KEY", "test-xai-key")
