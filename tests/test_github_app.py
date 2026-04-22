"""GitHub App service plumbing — JWT generation, token cache, no real API calls."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest


# Generate a throwaway RSA key once for the test session — PyJWT validates the
# key shape at sign time, so a real key (not a placeholder string) is required.
@pytest.fixture(scope="module")
def rsa_private_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("ascii")


@pytest.fixture(autouse=True)
def configure_app(monkeypatch, rsa_private_key):
    monkeypatch.setattr("app.config.GITHUB_APP_ID", "123456")
    monkeypatch.setattr("app.config.GITHUB_APP_PRIVATE_KEY", rsa_private_key)
    # Reset the module-level token cache between tests.
    from app import github_app
    github_app._token_cache.clear()
    yield


def test_generate_app_jwt_includes_app_id(rsa_private_key):
    import jwt as pyjwt
    from app import github_app
    token = github_app._generate_app_jwt()
    # Decode without verification to inspect claims (we just need the header/payload).
    payload = pyjwt.decode(token, options={"verify_signature": False})
    assert payload["iss"] == "123456"
    assert payload["exp"] > payload["iat"]
    assert payload["exp"] - payload["iat"] <= github_app.JWT_LIFETIME_SECONDS + 60


def test_generate_app_jwt_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr("app.config.GITHUB_APP_ID", None)
    from app import github_app
    with pytest.raises(github_app.GitHubAppError):
        github_app._generate_app_jwt()


@pytest.mark.asyncio
async def test_get_installation_token_caches(monkeypatch):
    """Two calls within the cache window should hit GitHub once."""
    from app import github_app
    from types import SimpleNamespace

    # Mock httpx.AsyncClient to return a fake token mint response.
    fake_resp = SimpleNamespace(
        status_code=201,
        json=lambda: {"token": "ghs_faketoken123", "expires_at": "2026-04-22T03:00:00Z"},
        text="ok",
    )

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        post = AsyncMock(return_value=fake_resp)

    fake_post = _FakeClient.post
    monkeypatch.setattr(github_app.httpx, "AsyncClient", _FakeClient)

    t1 = await github_app.get_installation_token(42)
    t2 = await github_app.get_installation_token(42)
    assert t1 == t2 == "ghs_faketoken123"
    assert fake_post.await_count == 1   # second call hit cache, not the API


@pytest.mark.asyncio
async def test_get_installation_token_raises_on_failure(monkeypatch):
    from app import github_app
    from types import SimpleNamespace

    fake_resp = SimpleNamespace(status_code=403, text="Bad credentials", json=lambda: {})

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        post = AsyncMock(return_value=fake_resp)

    monkeypatch.setattr(github_app.httpx, "AsyncClient", _FakeClient)

    with pytest.raises(github_app.GitHubAppError, match="403"):
        await github_app.get_installation_token(42)


@pytest.mark.asyncio
async def test_token_cache_expiry_re_mints(monkeypatch):
    """A token whose cached expiry is in the past should be re-minted."""
    from app import github_app
    from types import SimpleNamespace

    # Pre-seed cache with an "expired" entry.
    github_app._token_cache[99] = github_app._CachedToken(
        token="ghs_old", expires_at=time.time() - 10,
    )

    fake_resp = SimpleNamespace(
        status_code=201,
        json=lambda: {"token": "ghs_new", "expires_at": "2026-04-22T03:00:00Z"},
        text="ok",
    )

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        post = AsyncMock(return_value=fake_resp)

    monkeypatch.setattr(github_app.httpx, "AsyncClient", _FakeClient)

    t = await github_app.get_installation_token(99)
    assert t == "ghs_new"
