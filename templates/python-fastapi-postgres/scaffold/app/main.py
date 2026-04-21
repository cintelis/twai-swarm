"""FastAPI entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.config import get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    log.info("startup env=%s", settings.env)
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="app", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(api_router)
    log.info("app initialised env=%s", settings.env)
    return app


app = create_app()
