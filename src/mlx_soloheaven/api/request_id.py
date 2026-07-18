"""Per-request correlation ID (bultagi.com integration, Batch D — observability).

A lightweight ASGI middleware that assigns every HTTP request a request ID and
echoes it back in the ``X-Request-ID`` response header. An INCOMING
``X-Request-ID`` is honored when it is present AND safe (bounded length, a
conservative ``[A-Za-z0-9._-]`` charset — so a client cannot smuggle CRLF /
header-injection payloads or unbounded junk through it); otherwise a fresh
``uuid4`` hex is generated.

The active ID is stored in a ``ContextVar`` so handlers and logging can read it
for correlation (``get_request_id()``); the ID itself is NEVER used as a metric
label (unbounded cardinality — it belongs in logs/headers only, see
api/metrics.py).

This module imports NO mlx (only stdlib), so it is safe to load in the
process-mode parent. It is a PURE-ASGI middleware (not BaseHTTPMiddleware) so it
adds no per-request task/checkpoint overhead and works uniformly for JSON and
streaming (SSE) responses — the ``send`` wrapper stamps the header on the
``http.response.start`` message regardless of the body that follows.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

# Header name (lowercase, bytes) for ASGI byte-header matching / stamping.
_HEADER_LOWER = b"x-request-id"

# ASGI ``scope`` key the middleware stashes the id under (finding 6). Starlette's
# outer ServerErrorMiddleware generates uncaught 500s OUTSIDE this middleware's
# send-wrapper AND after our ContextVar has been reset, so the 500 handler reads
# the id from the (shared, still-live) scope instead of the ContextVar.
SCOPE_KEY = "soloheaven_request_id"


def request_id_from_scope(scope) -> str:
    """The request id stashed on an ASGI ``scope`` by ``RequestIDMiddleware``
    ('' when absent — e.g. a scope that never passed through the middleware)."""
    try:
        return scope.get(SCOPE_KEY, "") or ""
    except AttributeError:
        return ""

# A conservative, bounded allowlist for an incoming id. Anything else is
# rejected and replaced by a generated id — this is the guard against header
# injection (CR/LF), oversized values, and non-ASCII smuggling.
_SAFE_RE = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")

# The active request id for the current async context. Empty string == "no
# request in scope" (e.g. a background task or a direct unit-test call).
_request_id_var: ContextVar[str] = ContextVar("soloheaven_request_id", default="")


def get_request_id() -> str:
    """The request id bound to the current context ('' when none is set)."""
    return _request_id_var.get()

def set_request_id(request_id: str):
    """Bind ``request_id`` to the current context; returns the reset token."""
    return _request_id_var.set(request_id)


def new_request_id() -> str:
    """Generate a fresh, collision-resistant request id."""
    return uuid.uuid4().hex


def _sanitize_incoming(raw: bytes | None) -> str | None:
    """Return a SAFE incoming id, or None if absent/unsafe (caller generates)."""
    if not raw:
        return None
    try:
        value = raw.decode("latin-1")
    except Exception:  # noqa: BLE001 — undecodable => treat as absent
        return None
    return value if _SAFE_RE.match(value) else None


class RequestIDMiddleware:
    """Assign/propagate ``X-Request-ID`` and expose it via a ContextVar.

    Registered with ``app.add_middleware(RequestIDMiddleware)``. Non-HTTP scopes
    (lifespan, websocket) pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for key, value in scope.get("headers", ()):  # list[tuple[bytes, bytes]]
            if key == _HEADER_LOWER:
                incoming = value
                break
        request_id = _sanitize_incoming(incoming) or new_request_id()

        # Finding 6: stash on the (shared) scope so the outer 500 handler can
        # read it even after the ContextVar below has been reset by our finally.
        scope[SCOPE_KEY] = request_id

        rid_bytes = request_id.encode("latin-1")

        async def _send(message):
            if message["type"] == "http.response.start":
                # Drop any upstream copy, then append ours (echo exactly once).
                headers = [
                    (k, v)
                    for (k, v) in message.get("headers", [])
                    if k.lower() != _HEADER_LOWER
                ]
                headers.append((_HEADER_LOWER, rid_bytes))
                message["headers"] = headers
            await send(message)

        token = _request_id_var.set(request_id)
        try:
            await self.app(scope, receive, _send)
        finally:
            _request_id_var.reset(token)
