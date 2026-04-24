"""
GitHub App webhook handler.

GitHub POSTs events to /github/webhook. This module:
1. Verifies the HMAC-SHA256 signature against our configured webhook secret.
2. Dispatches events to per-type handlers.
3. Returns quickly so GitHub doesn't retry on slow handlers.

Events handled in v1:
- `installation.deleted` — user uninstalled the App on github.com. We
  auto-delete the DB row + clear the token cache so the next push doesn't
  see a stale install. Fixes the bug we hit manually during Langfuse deploy.
- `installation_repositories.removed` — user removed specific repos from
  the install without uninstalling. We log but DON'T delete the install row;
  user is just re-scoping.
- `pull_request_review_comment` — STUB for Sprint 10. Returns 202 + logs.
  When Sprint 10 lands, this triggers a continuation workflow that responds
  to the comment in the same PR branch.

Everything else is returned as "ignored" with 200 so GitHub stops retrying.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebhookVerificationError(Exception):
    """Signature missing or wrong. Caller returns HTTP 401."""


def verify_signature(secret: str, body: bytes, header: str | None) -> None:
    """Raise if the signature header doesn't match HMAC-SHA256(secret, body).

    GitHub sends `X-Hub-Signature-256: sha256=<hex>`. Constant-time compare
    via `hmac.compare_digest` to prevent timing attacks.
    """
    if not header:
        raise WebhookVerificationError("missing X-Hub-Signature-256 header")
    if not header.startswith("sha256="):
        raise WebhookVerificationError(
            f"unexpected signature algorithm: {header[:20]}..."
        )
    received = header[len("sha256="):]
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(received, expected):
        raise WebhookVerificationError("signature mismatch")


async def handle_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    db_module: Any,
    github_app_module: Any,
) -> dict[str, Any]:
    """Route the event to the right handler. Returns the response body dict.

    `db_module` and `github_app_module` are injected rather than imported
    so tests can swap them for fakes without patching sys.modules.
    """
    action = payload.get("action", "")
    key = f"{event_type}.{action}"

    if key == "installation.deleted":
        return await _handle_installation_deleted(payload, db_module, github_app_module)
    if key == "installation_repositories.removed":
        return _handle_repos_removed(payload)
    if event_type == "pull_request_review_comment":
        return _handle_pr_comment_stub(payload)

    # GitHub sends lots of events we haven't subscribed to or don't care about
    # yet. Return 200 OK so it doesn't waste retries on them.
    return {"status": "ignored", "event": key}


async def _handle_installation_deleted(
    payload: dict[str, Any],
    db_module: Any,
    github_app_module: Any,
) -> dict[str, Any]:
    """User uninstalled the App. Drop our DB row + evict cached token."""
    install_id = payload.get("installation", {}).get("id")
    if not install_id:
        logger.warning("installation.deleted payload missing installation.id")
        return {"status": "noop", "reason": "missing installation_id"}

    install_id = int(install_id)
    deleted = await db_module.delete_github_installation(install_id)
    github_app_module._token_cache.pop(install_id, None)
    logger.info(
        "installation.deleted webhook: installation_id=%d · DB rows removed=%d",
        install_id, deleted,
    )
    return {
        "status": "ok",
        "installation_id": install_id,
        "db_rows_deleted": deleted,
    }


def _handle_repos_removed(payload: dict[str, Any]) -> dict[str, Any]:
    """User removed specific repos from the install. Log only — they may
    add them back. Don't delete the install row; it's still alive."""
    install = payload.get("installation", {})
    repos = [r.get("full_name", "?") for r in payload.get("repositories_removed", [])]
    logger.info(
        "installation_repositories.removed: installation_id=%s · repos=%s",
        install.get("id"), repos,
    )
    return {
        "status": "ok",
        "action": "logged",
        "installation_id": install.get("id"),
        "repos_removed": repos,
    }


def _handle_pr_comment_stub(payload: dict[str, Any]) -> dict[str, Any]:
    """Sprint 10 will wire this to a continuation workflow that responds
    to the comment on the same branch. v1: log + return 202 accepted."""
    comment = payload.get("comment", {})
    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {}).get("full_name", "?")
    install_id = payload.get("installation", {}).get("id")
    logger.info(
        "pr_review_comment (Sprint 10 stub): repo=%s pr=%s comment_id=%s install=%s",
        repo, pr.get("number"), comment.get("id"), install_id,
    )
    return {
        "status": "accepted",
        "action": "logged_for_sprint_10",
        "repo": repo,
        "pr_number": pr.get("number"),
    }
