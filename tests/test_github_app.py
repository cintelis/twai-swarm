"""GitHub App service plumbing — JWT generation, token cache, 401 retry, no real API calls."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    payload = pyjwt.decode(token, options={"verify_signature": False})
    assert payload["iss"] == "123456"
    assert payload["exp"] > payload["iat"]
    assert payload["exp"] - payload["iat"] <= github_app.JWT_LIFETIME_SECONDS + 60


def test_generate_app_jwt_raises_when_unconfigured(monkeypatch):
    monkeypatch.setattr("app.config.GITHUB_APP_ID", None)
    from app import github_app
    with pytest.raises(github_app.GitHubAppError):
        github_app._generate_app_jwt()


# ─── Token mint / cache tests ─────────────────────────────
# These mock httpx.AsyncClient.post directly because the mint function uses
# .post() (not .request()) — keeps it simple, no _authed_request involvement.

def _patch_mint_only(monkeypatch, mint_resp):
    """Mock httpx.AsyncClient with a `.post` that returns the mint response.
    Used by tests that exercise get_installation_token in isolation."""
    from app import github_app

    class _C:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        post = AsyncMock(return_value=mint_resp)
    monkeypatch.setattr(github_app.httpx, "AsyncClient", _C)
    return _C


@pytest.mark.asyncio
async def test_get_installation_token_caches(monkeypatch):
    from app import github_app
    fake_resp = SimpleNamespace(
        status_code=201, json=lambda: {"token": "ghs_faketoken123"}, text="ok",
    )
    cls = _patch_mint_only(monkeypatch, fake_resp)
    t1 = await github_app.get_installation_token(42)
    t2 = await github_app.get_installation_token(42)
    assert t1 == t2 == "ghs_faketoken123"
    assert cls.post.await_count == 1   # second call hit cache


@pytest.mark.asyncio
async def test_get_installation_token_raises_on_failure(monkeypatch):
    from app import github_app
    _patch_mint_only(
        monkeypatch,
        SimpleNamespace(status_code=403, text="Bad credentials", json=lambda: {}),
    )
    with pytest.raises(github_app.GitHubAppError, match="403"):
        await github_app.get_installation_token(42)


@pytest.mark.asyncio
async def test_token_cache_expiry_re_mints(monkeypatch):
    from app import github_app
    github_app._token_cache[99] = github_app._CachedToken(
        token="ghs_old", expires_at=time.time() - 10,
    )
    _patch_mint_only(
        monkeypatch,
        SimpleNamespace(status_code=201, json=lambda: {"token": "ghs_new"}, text="ok"),
    )
    t = await github_app.get_installation_token(99)
    assert t == "ghs_new"


@pytest.mark.asyncio
async def test_get_installation_token_force_refresh_skips_cache(monkeypatch):
    """force_refresh=True should mint fresh even with a valid cached token."""
    from app import github_app
    github_app._token_cache[55] = github_app._CachedToken(
        token="ghs_cached", expires_at=time.time() + 600,
    )
    cls = _patch_mint_only(
        monkeypatch,
        SimpleNamespace(status_code=201, json=lambda: {"token": "ghs_forced"}, text="ok"),
    )
    t = await github_app.get_installation_token(55, force_refresh=True)
    assert t == "ghs_forced"
    assert cls.post.await_count == 1


# ─── _authed_request + caller tests ──────────────────────
# These mock httpx.AsyncClient.request (what _authed_request uses).
# Pre-seed the token cache so mint isn't part of the test path.

def _patch_request(monkeypatch, request_handler):
    """Replace httpx.AsyncClient with a fake whose `.request(...)` is the
    given handler. Pre-seeds installation 7777's token cache."""
    from app import github_app
    github_app._token_cache[7777] = github_app._CachedToken(
        token="ghs_test", expires_at=time.time() + 600,
    )

    class _C:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def request(self, method, url, **kw):
            return await request_handler(method, url, kw)
        # Some code paths (mint) use .post directly — give a default that mints OK.
        post = AsyncMock(return_value=SimpleNamespace(
            status_code=201, json=lambda: {"token": "ghs_remint"}, text="ok",
        ))
    monkeypatch.setattr(github_app.httpx, "AsyncClient", _C)
    return _C


@pytest.mark.asyncio
async def test_repo_exists_true_on_200(monkeypatch):
    from app import github_app

    async def handler(method, url, kw):
        return SimpleNamespace(status_code=200, text="", json=lambda: {})
    _patch_request(monkeypatch, handler)
    assert await github_app.repo_exists(7777, "Cintelis-Ai", "swarm-test") is True


@pytest.mark.asyncio
async def test_repo_exists_false_on_404(monkeypatch):
    from app import github_app

    async def handler(method, url, kw):
        return SimpleNamespace(status_code=404, text="not found", json=lambda: {})
    _patch_request(monkeypatch, handler)
    assert await github_app.repo_exists(7777, "Cintelis-Ai", "doesnt-exist") is False


@pytest.mark.asyncio
async def test_repo_exists_raises_on_other_status(monkeypatch):
    from app import github_app

    async def handler(method, url, kw):
        return SimpleNamespace(status_code=500, text="oops", json=lambda: {})
    _patch_request(monkeypatch, handler)
    with pytest.raises(github_app.GitHubAppError, match="500"):
        await github_app.repo_exists(7777, "Cintelis-Ai", "boom")


@pytest.mark.asyncio
async def test_create_org_repo_success(monkeypatch):
    from app import github_app
    fake_repo = {
        "name": "swarm-new",
        "full_name": "Cintelis-Ai/swarm-new",
        "default_branch": "main",
        "html_url": "https://github.com/Cintelis-Ai/swarm-new",
    }

    async def handler(method, url, kw):
        assert method == "POST"
        assert "/orgs/Cintelis-Ai/repos" in url
        return SimpleNamespace(status_code=201, json=lambda: fake_repo, text="ok")
    _patch_request(monkeypatch, handler)
    result = await github_app.create_org_repo(
        installation_id=7777, org="Cintelis-Ai", name="swarm-new",
        description="test", private=True,
    )
    assert result["full_name"] == "Cintelis-Ai/swarm-new"
    assert result["default_branch"] == "main"


@pytest.mark.asyncio
async def test_create_org_repo_failure(monkeypatch):
    from app import github_app

    async def handler(method, url, kw):
        return SimpleNamespace(
            status_code=403, json=lambda: {},
            text="Resource not accessible by integration",
        )
    _patch_request(monkeypatch, handler)
    with pytest.raises(github_app.GitHubAppError, match="403"):
        await github_app.create_org_repo(
            installation_id=7777, org="Cintelis-Ai", name="swarm-x",
        )


# ─── 401 retry tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_authed_request_retries_on_401_with_fresh_mint(monkeypatch):
    """Cached token returns 401 → cache evicted → fresh token minted → retry succeeds."""
    from app import github_app

    # Pre-seed cache with a token that GitHub will reject.
    github_app._token_cache[7777] = github_app._CachedToken(
        token="ghs_revoked", expires_at=time.time() + 600,
    )

    request_calls: list[tuple[str, str, str]] = []  # (method, url, auth_header)

    async def handler(method, url, kw):
        auth = kw.get("headers", {}).get("Authorization", "")
        request_calls.append((method, url, auth))
        if auth == "Bearer ghs_revoked":
            return SimpleNamespace(status_code=401, text="Bad credentials", json=lambda: {})
        # The fresh token (after re-mint) → success.
        return SimpleNamespace(status_code=200, text="", json=lambda: {})

    class _C:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def request(self, method, url, **kw):
            return await handler(method, url, kw)
        # Re-mint returns a fresh token.
        post = AsyncMock(return_value=SimpleNamespace(
            status_code=201, json=lambda: {"token": "ghs_fresh"}, text="ok",
        ))
    monkeypatch.setattr(github_app.httpx, "AsyncClient", _C)

    result = await github_app.repo_exists(7777, "owner", "name")

    assert result is True
    # Two requests: first with the revoked token (401), retry with fresh.
    assert len(request_calls) == 2
    assert request_calls[0][2] == "Bearer ghs_revoked"
    assert request_calls[1][2] == "Bearer ghs_fresh"
    # Cache now holds the fresh token.
    assert github_app._token_cache[7777].token == "ghs_fresh"


@pytest.mark.asyncio
async def test_authed_request_no_retry_when_first_call_succeeds(monkeypatch):
    """Happy path — single request, no re-mint."""
    from app import github_app

    request_count = {"n": 0}

    async def handler(method, url, kw):
        request_count["n"] += 1
        return SimpleNamespace(status_code=200, text="", json=lambda: {})

    cls = _patch_request(monkeypatch, handler)
    await github_app.repo_exists(7777, "owner", "name")

    assert request_count["n"] == 1
    assert cls.post.await_count == 0   # mint not called (cache hit)


@pytest.mark.asyncio
async def test_authed_request_does_not_retry_on_403(monkeypatch):
    """403 (permission denied) is permanent — bubble up, don't retry."""
    from app import github_app

    request_count = {"n": 0}

    async def handler(method, url, kw):
        request_count["n"] += 1
        return SimpleNamespace(
            status_code=403, json=lambda: {},
            text="Resource not accessible by integration",
        )
    _patch_request(monkeypatch, handler)
    # repo_exists treats 403 as a fail → raises GitHubAppError.
    with pytest.raises(github_app.GitHubAppError, match="403"):
        await github_app.repo_exists(7777, "owner", "name")
    assert request_count["n"] == 1   # only one attempt
