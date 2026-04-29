"""Sprint 15a — language-agnostic helpers for HTTP route extraction.

`normalize_path` and the HTTP_CLIENT_RECEIVERS guard are shared between
the Python and TypeScript route extractors. Keeping them here avoids
divergent implementations of "is this `app.get(...)` actually a route?".
"""
from __future__ import annotations

# Receiver names used for HTTP CLIENTS (not servers). When the
# extractor sees `<receiver>.get("/x", ...)`, it must check the receiver
# isn't one of these — `axios.get(...)` and `fetch.post(...)` look
# identical to Express's `app.get("/x", handler)` at the AST level.
# Mirror of GitNexus's `HTTP_CLIENT_RECEIVERS` (parse-worker.ts:874).
HTTP_CLIENT_RECEIVERS = frozenset({
    "axios", "fetch", "got", "ky", "request",
    "httpservice", "httpclient",  # NestJS HttpService etc.
    "http", "https",              # Node `http.get` / `https.get`
    "client", "api", "api_client",
    "session",                    # Python `requests.Session()` instance
    "requests",                   # Python `requests.get(...)`
    "urllib",                     # Python `urllib.request`
    "aiohttp",                    # async client
})


# HTTP methods we recognise on `app.<verb>(...)` patterns. Lowercase for
# matching against `attribute` text; we uppercase before storing.
HTTP_METHODS = frozenset({
    "get", "post", "put", "delete", "patch", "head", "options",
})


def normalize_path(raw: str) -> str:
    """Normalise a route path for the composite uniqueness key.

    Rules (mirror of GitNexus's `normalizeHttpPath`):
        - lower-case
        - strip leading/trailing whitespace
        - strip trailing slash unless the path IS just "/"
        - collapse repeated slashes to one
        - leave path-parameter syntax alone (FastAPI `{id}`,
          Express `:id`, Django `<int:id>`, Hono `:id`)

    Returns "/" for empty / unparseable input.
    """
    if not raw:
        return "/"
    s = raw.strip()
    if not s:
        return "/"
    # Strip surrounding quotes if the caller passed a quoted token.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.lower()
    while "//" in s:
        s = s.replace("//", "/")
    if len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    if not s.startswith("/"):
        s = "/" + s
    return s


__all__ = ["HTTP_CLIENT_RECEIVERS", "HTTP_METHODS", "normalize_path"]
