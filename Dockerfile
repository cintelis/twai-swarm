FROM python:3.12-slim

# Force unbuffered stdout/stderr so print() lands in CloudWatch immediately.
# Without this, Python block-buffers when stdout is a pipe and short messages
# never appear until ~4KB accumulates (or the process exits).
ENV PYTHONUNBUFFERED=1

# OS deps:
#   - ca-certificates / curl: TLS roots + curl for verify.sh health probes
#   - bash: required by the Coder's run_verify + bash_exec tools
#   - git: Coder may `git init` / `git diff` inside its workspace
#   - nodejs + npm (via NodeSource Node 20 LTS): toolchain for JS/TS templates
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bash git gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + project templates the Coder copies into its sandbox.
# Templates ship in the image so the worker doesn't need a runtime download.
COPY app ./app
COPY templates ./templates

# Non-root
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# The container runs EITHER the API or the worker based on CMD.
# ECS task definitions override CMD to pick a role.
# Default to API so `docker run image` does something useful.
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
