"""pytest fixtures.

Uses an in-memory SQLite DB for tests so pristine `verify.sh` passes without
requiring a Postgres container. Swap to a real Postgres in CI or dev if the
domain needs Postgres-specific features (e.g. JSONB operators).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import db as db_module
from app.db import get_db
from app.main import app
from app.models import Base


@pytest.fixture
async def test_engine():
    """In-memory SQLite engine, fresh schema per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def client(test_engine, monkeypatch) -> AsyncIterator[AsyncClient]:
    """FastAPI client wired to the test engine via dependency override."""
    TestSession = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    # Also override the module-level engine in case anything imports it directly.
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
