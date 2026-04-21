# {{project_name}}

> Replace this README with a description of what the service actually does. The Coder customises this file during workflow execution.

A FastAPI service scaffold with:

- **FastAPI** async HTTP framework
- **SQLAlchemy 2.0** async ORM
- **Alembic** database migrations
- **Pydantic-settings** env-driven config
- **pytest + pytest-asyncio** with DB fixtures
- **PostgreSQL** via Docker Compose for local dev
- **GitHub Actions** CI (ruff + pytest)

## Quickstart

```bash
# Local dev — start Postgres
docker compose up -d db

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Copy env template and edit
cp .env.example .env

# Run migrations (once Alembic revisions exist)
alembic upgrade head

# Start the service
uvicorn app.main:app --reload --port 8000

# Smoke check
curl http://localhost:8000/health
```

## Tests

```bash
pytest -q
```

## Layout

```
app/
├── main.py           FastAPI entry + lifespan
├── config.py         Pydantic settings (env-driven)
├── db.py             Async engine + session factory
├── models.py         SQLAlchemy ORM models
└── api/
    ├── health.py     GET /health
    └── routes.py     Example /items CRUD — replace with your domain

tests/
├── conftest.py       pytest fixtures (DB, client)
├── test_smoke.py     /health returns 200
└── test_items.py     CRUD flow against example routes

migrations/           Alembic scaffold (autogen from app.models.Base)
docker-compose.yml    Local Postgres (db service)
Dockerfile            Multi-stage, non-root runtime
```

## Deploying

The Dockerfile is ready for any container runtime (ECS Fargate, Cloud Run, Kubernetes, Fly.io). Set `APP_DATABASE_URL` to a reachable Postgres. Run `alembic upgrade head` as a one-shot task before starting the HTTP service for the first time.
