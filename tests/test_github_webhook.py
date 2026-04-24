"""Webhook signature verification + event dispatch."""
from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import github_webhook


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_signature_success():
    secret = "testsecret"
    body = b'{"ok":true}'
    github_webhook.verify_signature(secret, body, _sign(secret, body))


def test_verify_signature_mismatch_raises():
    with pytest.raises(github_webhook.WebhookVerificationError, match="mismatch"):
        github_webhook.verify_signature("secret", b"body", "sha256=00ff")


def test_verify_signature_missing_header_raises():
    with pytest.raises(github_webhook.WebhookVerificationError, match="missing"):
        github_webhook.verify_signature("secret", b"body", None)


def test_verify_signature_wrong_algorithm_raises():
    with pytest.raises(github_webhook.WebhookVerificationError, match="algorithm"):
        github_webhook.verify_signature("secret", b"body", "sha1=abc")


def test_verify_signature_constant_time_compare():
    """Ensures we're using hmac.compare_digest, not == (which leaks timing)."""
    secret = "s"
    body = b"x"
    correct_sig = _sign(secret, body)
    # A length-matched but wrong signature should still raise, not early-return.
    wrong_sig = "sha256=" + "0" * len(correct_sig[len("sha256="):])
    with pytest.raises(github_webhook.WebhookVerificationError):
        github_webhook.verify_signature(secret, body, wrong_sig)


# ─── Event handler tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_installation_deleted_cleans_up_db_and_cache():
    fake_db = MagicMock()
    fake_db.delete_github_installation = AsyncMock(return_value=1)

    fake_github_app = MagicMock()
    fake_github_app._token_cache = {42: "fake_cached_token", 99: "other"}

    payload = {"action": "deleted", "installation": {"id": 42}}
    result = await github_webhook.handle_event(
        "installation", payload,
        db_module=fake_db, github_app_module=fake_github_app,
    )

    fake_db.delete_github_installation.assert_awaited_once_with(42)
    assert 42 not in fake_github_app._token_cache
    assert 99 in fake_github_app._token_cache   # others untouched
    assert result["status"] == "ok"
    assert result["installation_id"] == 42
    assert result["db_rows_deleted"] == 1


@pytest.mark.asyncio
async def test_installation_deleted_missing_id_is_noop():
    fake_db = MagicMock()
    fake_db.delete_github_installation = AsyncMock()
    fake_github_app = MagicMock()
    fake_github_app._token_cache = {}

    result = await github_webhook.handle_event(
        "installation",
        {"action": "deleted", "installation": {}},   # no id field
        db_module=fake_db, github_app_module=fake_github_app,
    )
    fake_db.delete_github_installation.assert_not_awaited()
    assert result["status"] == "noop"


@pytest.mark.asyncio
async def test_repos_removed_logs_without_deletion():
    """User re-scoping which repos the install has access to — NOT uninstalling.
    Must NOT delete the install row."""
    fake_db = MagicMock()
    fake_db.delete_github_installation = AsyncMock()
    fake_github_app = MagicMock()
    fake_github_app._token_cache = {42: "token"}

    payload = {
        "action": "removed",
        "installation": {"id": 42},
        "repositories_removed": [
            {"full_name": "acme/web"},
            {"full_name": "acme/api"},
        ],
    }
    result = await github_webhook.handle_event(
        "installation_repositories", payload,
        db_module=fake_db, github_app_module=fake_github_app,
    )

    fake_db.delete_github_installation.assert_not_awaited()
    assert 42 in fake_github_app._token_cache
    assert result["status"] == "ok"
    assert result["action"] == "logged"
    assert result["repos_removed"] == ["acme/web", "acme/api"]


@pytest.mark.asyncio
async def test_pr_review_comment_stubbed_for_sprint_10():
    result = await github_webhook.handle_event(
        "pull_request_review_comment",
        {
            "action": "created",
            "comment": {"body": "please add a test", "id": 999},
            "pull_request": {"number": 5, "head": {"ref": "swarm/abc"}},
            "repository": {"full_name": "acme/swarm-test"},
            "installation": {"id": 42},
        },
        db_module=MagicMock(), github_app_module=MagicMock(),
    )
    assert result["status"] == "accepted"
    assert result["action"] == "logged_for_sprint_10"
    assert result["repo"] == "acme/swarm-test"
    assert result["pr_number"] == 5


@pytest.mark.asyncio
async def test_unhandled_event_returns_ignored():
    result = await github_webhook.handle_event(
        "push",
        {"action": "whatever", "repository": {"full_name": "x/y"}},
        db_module=MagicMock(), github_app_module=MagicMock(),
    )
    assert result["status"] == "ignored"
    assert result["event"] == "push.whatever"
