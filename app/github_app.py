"""
GitHub App authentication + push helper.

Two-step auth, both via PyJWT + GitHub REST:
1. Sign a short-lived JWT (10 min, RS256, key = the App's private key) — proves
   we're the App. JWT identity = the App ID.
2. POST /app/installations/{id}/access_tokens with the JWT to get an
   installation token (1-hour TTL) scoped to that installation's repos.
3. Use the installation token as `Authorization: Bearer <token>` on every
   GitHub API call.

We cache installation tokens for ~55 minutes to amortise the JWT mint + token
exchange cost. Cache key = installation_id.

Push approach: we use the Git Data API (blobs + tree + commit + ref) rather
than the Contents API. Contents API is one-file-per-call and chatty; the Git
Data API lets us push N files in a single tree + commit + ref-update sequence.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import jwt as pyjwt

from app import config

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
JWT_LIFETIME_SECONDS = 540          # 9 min — leaves slack vs GitHub's 10-min cap
INSTALLATION_TOKEN_TTL = 3300       # cache for 55 min — GitHub's tokens are 60 min
USER_AGENT = "cintelis-swarm/1.0"


class GitHubAppError(Exception):
    """Raised when something goes wrong on the GitHub side (auth, push, API)."""


@dataclass
class _CachedToken:
    token: str
    expires_at: float   # epoch seconds


# Process-wide cache. Keyed by installation_id.
_token_cache: dict[int, _CachedToken] = {}


def _generate_app_jwt() -> str:
    """Mint a JWT signed with the App's private key. Identity = the App ID.

    Used only to mint installation tokens — never call the GitHub API directly
    with this JWT for repo operations.
    """
    if not config.GITHUB_APP_ID or not config.GITHUB_APP_PRIVATE_KEY:
        raise GitHubAppError(
            "GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY must be set to use the GitHub App"
        )

    now = int(time.time())
    payload = {
        "iat": now - 30,                          # 30s clock skew tolerance
        "exp": now + JWT_LIFETIME_SECONDS,
        "iss": config.GITHUB_APP_ID,
    }
    return pyjwt.encode(payload, config.GITHUB_APP_PRIVATE_KEY, algorithm="RS256")


async def get_installation_token(installation_id: int) -> str:
    """Return a cached installation access token, minting if necessary."""
    cached = _token_cache.get(installation_id)
    if cached and cached.expires_at > time.time() + 60:    # 1-min safety margin
        return cached.token

    app_jwt = _generate_app_jwt()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
        )
    if resp.status_code != 201:
        raise GitHubAppError(
            f"installation token mint failed: {resp.status_code} {resp.text[:200]}"
        )
    body = resp.json()
    token = body["token"]
    _token_cache[installation_id] = _CachedToken(
        token=token,
        expires_at=time.time() + INSTALLATION_TOKEN_TTL,
    )
    return token


async def fetch_installation_metadata(installation_id: int) -> dict:
    """Read the install's account_login, account_type, and granted permissions.

    Used by the callback (to persist a friendly name alongside the install ID)
    and by the permission-preflight check (to surface a clear error before
    we hit GitHub with a call that'll 403).
    """
    app_jwt = _generate_app_jwt()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{GITHUB_API}/app/installations/{installation_id}",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
        )
    if resp.status_code != 200:
        raise GitHubAppError(
            f"installation metadata fetch failed: {resp.status_code} {resp.text[:200]}"
        )
    body = resp.json()
    return {
        "installation_id": int(body["id"]),
        "account_login": body["account"]["login"],
        "account_type": body["account"]["type"],   # 'Organization' or 'User'
        "permissions": body.get("permissions", {}), # {perm_name: "read" | "write"}
    }


# What we need for "Push to GitHub" with auto-create-repo to work end-to-end.
# Organization_administration:write is the one that commonly fails because
# the org owner hasn't accepted the new permission on the existing install.
REQUIRED_PERMISSIONS = {
    "contents": "write",
    "pull_requests": "write",
    "metadata": "read",
    "organization_administration": "write",
}

_LEVEL_RANK = {"read": 1, "write": 2, "admin": 3}


def missing_permissions(granted: dict) -> list[str]:
    """Return a human-readable list of permissions that don't satisfy the
    required set. Empty list = the install has everything we need."""
    missing: list[str] = []
    for perm, required in REQUIRED_PERMISSIONS.items():
        current = granted.get(perm)
        if current is None:
            missing.append(f"{perm}: need {required}, not granted")
            continue
        if _LEVEL_RANK.get(current, 0) < _LEVEL_RANK.get(required, 0):
            missing.append(f"{perm}: need {required}, have {current}")
    return missing


async def list_installation_repos(installation_id: int) -> list[dict]:
    """List repositories this installation can access."""
    token = await get_installation_token(installation_id)
    repos: list[dict] = []
    page = 1
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            resp = await client.get(
                f"{GITHUB_API}/installation/repositories",
                params={"per_page": 100, "page": page},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": USER_AGENT,
                },
            )
            if resp.status_code != 200:
                raise GitHubAppError(
                    f"list repos failed: {resp.status_code} {resp.text[:200]}"
                )
            body = resp.json()
            for r in body.get("repositories", []):
                repos.append({
                    "full_name": r["full_name"],
                    "owner": r["owner"]["login"],
                    "name": r["name"],
                    "default_branch": r["default_branch"],
                    "private": r["private"],
                })
            if len(body.get("repositories", [])) < 100:
                break
            page += 1
    return repos


async def repo_exists(installation_id: int, owner: str, name: str) -> bool:
    """True iff the repo is visible to the installation's token.

    Used to branch between "push to existing" and "create then push" paths.
    A 404 here means either the repo doesn't exist OR the App hasn't been
    granted access to it — both cases are handled the same way by the caller
    (either create it or the push step will fail loudly).
    """
    token = await get_installation_token(installation_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{name}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
        )
    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    raise GitHubAppError(
        f"repo existence check failed: {resp.status_code} {resp.text[:200]}"
    )


async def create_org_repo(
    installation_id: int,
    org: str,
    name: str,
    description: str = "",
    private: bool = True,
) -> dict:
    """Create a repo in the given org via the App. Requires the App's
    organization permission `Administration: Write`.

    `auto_init=true` creates an initial empty commit so the repo has a
    default branch — the push code path relies on HEAD already existing.
    Without it, `GET /git/ref/heads/<default>` would 404.

    Returns the full repo JSON from GitHub (owner, name, default_branch,
    clone_url, html_url, etc.).
    """
    token = await get_installation_token(installation_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GITHUB_API}/orgs/{org}/repos",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            },
            json={
                "name": name,
                "description": description or f"Scaffold from 365Soft Labs Swarm: {name}",
                "private": private,
                "auto_init": True,
                "has_issues": True,
                "has_projects": False,
                "has_wiki": False,
            },
        )
    if resp.status_code != 201:
        raise GitHubAppError(
            f"create repo failed: {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()


@dataclass
class PushResult:
    branch: str
    commit_sha: str
    pr_url: Optional[str]
    pr_number: Optional[int]
    files_pushed: int
    repo_created: bool = False   # True iff we created the repo on this push


async def push_files_as_branch(
    installation_id: int,
    repo_owner: str,
    repo_name: str,
    branch: str,
    files: list[dict],
    commit_message: str,
    open_pr: bool = True,
    pr_title: Optional[str] = None,
    pr_body: Optional[str] = None,
) -> PushResult:
    """Push `files` ([{path, content}]) as a new branch off the repo's default
    branch. Optionally open a PR.

    Steps (all via Git Data API):
      1. Resolve default branch + its head SHA.
      2. Create a blob per file.
      3. Build a tree referencing all blobs (base = default-branch tree).
      4. Create a commit with that tree + parent = default-branch head.
      5. Create the new branch ref pointing at the new commit.
      6. (Optional) open a PR from the new branch to the default branch.
    """
    if not files:
        raise GitHubAppError("no files to push")

    token = await get_installation_token(installation_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    base = f"{GITHUB_API}/repos/{repo_owner}/{repo_name}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Default branch + base commit
        repo_resp = await client.get(base, headers=headers)
        if repo_resp.status_code != 200:
            raise GitHubAppError(f"get repo failed: {repo_resp.status_code} {repo_resp.text[:200]}")
        default_branch = repo_resp.json()["default_branch"]

        ref_resp = await client.get(f"{base}/git/ref/heads/{default_branch}", headers=headers)
        if ref_resp.status_code != 200:
            raise GitHubAppError(f"get default ref failed: {ref_resp.status_code} {ref_resp.text[:200]}")
        base_sha = ref_resp.json()["object"]["sha"]

        commit_resp = await client.get(f"{base}/git/commits/{base_sha}", headers=headers)
        if commit_resp.status_code != 200:
            raise GitHubAppError(f"get base commit failed: {commit_resp.status_code} {commit_resp.text[:200]}")
        base_tree_sha = commit_resp.json()["tree"]["sha"]

        # 2. One blob per file (parallelised — typical scaffold is 10-20 files)
        async def _create_blob(path: str, content: str) -> dict:
            r = await client.post(
                f"{base}/git/blobs",
                headers=headers,
                json={
                    "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                    "encoding": "base64",
                },
            )
            if r.status_code != 201:
                raise GitHubAppError(
                    f"create blob {path} failed: {r.status_code} {r.text[:200]}"
                )
            return {"path": path, "sha": r.json()["sha"]}

        blob_results = await asyncio.gather(*(
            _create_blob(f["path"], f.get("content", "")) for f in files
        ))

        # 3. Tree
        tree_payload = {
            "base_tree": base_tree_sha,
            "tree": [
                {"path": b["path"], "mode": "100644", "type": "blob", "sha": b["sha"]}
                for b in blob_results
            ],
        }
        tree_resp = await client.post(f"{base}/git/trees", headers=headers, json=tree_payload)
        if tree_resp.status_code != 201:
            raise GitHubAppError(f"create tree failed: {tree_resp.status_code} {tree_resp.text[:200]}")
        tree_sha = tree_resp.json()["sha"]

        # 4. Commit
        commit_payload = {
            "message": commit_message,
            "tree": tree_sha,
            "parents": [base_sha],
        }
        c_resp = await client.post(f"{base}/git/commits", headers=headers, json=commit_payload)
        if c_resp.status_code != 201:
            raise GitHubAppError(f"create commit failed: {c_resp.status_code} {c_resp.text[:200]}")
        new_commit_sha = c_resp.json()["sha"]

        # 5. Branch ref
        ref_payload = {"ref": f"refs/heads/{branch}", "sha": new_commit_sha}
        r = await client.post(f"{base}/git/refs", headers=headers, json=ref_payload)
        if r.status_code not in (201, 422):
            raise GitHubAppError(f"create ref failed: {r.status_code} {r.text[:200]}")
        if r.status_code == 422:
            # Branch already exists — fast-forward it instead of failing.
            r = await client.patch(
                f"{base}/git/refs/heads/{branch}",
                headers=headers,
                json={"sha": new_commit_sha, "force": True},
            )
            if r.status_code != 200:
                raise GitHubAppError(f"update ref failed: {r.status_code} {r.text[:200]}")

        # 6. Optional PR
        pr_url: Optional[str] = None
        pr_number: Optional[int] = None
        if open_pr:
            pr_payload = {
                "title": pr_title or f"Scaffold from twai-swarm: {branch}",
                "head": branch,
                "base": default_branch,
                "body": pr_body or "Generated by the twai-swarm agentic Coder.",
            }
            pr_resp = await client.post(f"{base}/pulls", headers=headers, json=pr_payload)
            if pr_resp.status_code == 201:
                pj = pr_resp.json()
                pr_url = pj["html_url"]
                pr_number = int(pj["number"])
            else:
                # PR creation failure isn't fatal — branch is pushed; user can open
                # the PR manually. Log + continue.
                logger.warning(
                    "PR open failed (branch %s pushed OK): %s %s",
                    branch, pr_resp.status_code, pr_resp.text[:200],
                )

    return PushResult(
        branch=branch,
        commit_sha=new_commit_sha,
        pr_url=pr_url,
        pr_number=pr_number,
        files_pushed=len(files),
    )
