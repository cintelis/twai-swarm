"""
Bearer-token auth.

Opt-in via the API_AUTH_TOKEN env var:
- Unset / empty → auth disabled (every request passes). Useful for local dev
  and for backwards compatibility with deployments that haven't been updated.
- Set            → every protected route requires `Authorization: Bearer <token>`.

Apply to a route via FastAPI dependency:
    @app.post("/projects", dependencies=[Depends(require_auth)])
"""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# `auto_error=False` lets us return 401 ourselves with a body the UI can parse,
# instead of FastAPI's default {"detail": "Not authenticated"}.
_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    return bool((os.getenv("API_AUTH_TOKEN") or "").strip())


def require_auth(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Reject the request unless a matching bearer token was supplied."""
    expected = (os.getenv("API_AUTH_TOKEN") or "").strip()
    if not expected:
        return  # auth disabled — let everything through
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="auth_required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_token",
            headers={"WWW-Authenticate": "Bearer"},
        )
