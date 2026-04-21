"""Example CRUD test against the Item routes.

Delete this file when you replace the Item model with your domain entities.
It serves as a pattern for how to test async routes with the DB fixture.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_and_list(client: AsyncClient):
    # List is empty to start
    r = await client.get("/items")
    assert r.status_code == 200
    assert r.json() == []

    # Create
    r = await client.post("/items", json={"name": "widget", "description": "test"})
    assert r.status_code == 201
    created = r.json()
    assert created["name"] == "widget"
    item_id = created["id"]

    # Read back
    r = await client.get(f"/items/{item_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "widget"

    # List has one
    r = await client.get("/items")
    assert len(r.json()) == 1

    # Delete
    r = await client.delete(f"/items/{item_id}")
    assert r.status_code == 204

    # 404 after delete
    r = await client.get(f"/items/{item_id}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_requires_name(client: AsyncClient):
    r = await client.post("/items", json={"description": "no name"})
    assert r.status_code == 422
