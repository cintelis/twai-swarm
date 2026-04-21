"""Example CRUD routes for the Item model.

**Replace this file with routes for your actual domain.** The patterns here
(pydantic request/response models, dependency-injected session, async SQLAlchemy
queries) are worth preserving even when the entity changes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Item

router = APIRouter(prefix="/items", tags=["items"])


# ─── Request / response schemas ────────────────────────────────────────────

class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class ItemOut(BaseModel):
    id: int
    name: str
    description: str | None

    model_config = {"from_attributes": True}


# ─── Routes ────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ItemOut])
async def list_items(db: AsyncSession = Depends(get_db)) -> list[Item]:
    result = await db.execute(select(Item).order_by(Item.id.desc()).limit(100))
    return list(result.scalars().all())


@router.get("/{item_id}", response_model=ItemOut)
async def get_item(item_id: int, db: AsyncSession = Depends(get_db)) -> Item:
    item = await db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item


@router.post("", response_model=ItemOut, status_code=201)
async def create_item(payload: ItemCreate, db: AsyncSession = Depends(get_db)) -> Item:
    item = Item(name=payload.name, description=payload.description)
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=204)
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)) -> None:
    item = await db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    await db.delete(item)
    await db.commit()
