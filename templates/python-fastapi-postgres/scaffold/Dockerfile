# Multi-stage build. Keep final image slim and run as non-root.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps first for layer caching. Copy only the metadata
# needed by pip for resolution; avoid invalidating the layer on source edits.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install .

COPY app ./app

# Non-root runtime user.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
