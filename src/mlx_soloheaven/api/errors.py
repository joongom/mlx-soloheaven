"""Canonical error-envelope helpers (bultagi.com integration, Batch B).

ONE envelope shape for every non-streaming error response and every in-band
SSE error frame:

    {"error": {"message": <str>, "type": <str>, "code": <str>}}

A single builder here keeps the shape from drifting across the error sites
(400 validation, 429 saturation, 503 not-ready, 500 corruption, 404 not-found,
and the streaming in-band frames). The response body NEVER carries raw
exception text (``str(exc)``) — Batch B drops the old ``detail: str(exc)``
leaks; exceptions are still logged server-side.

Status-code policy (downstream-confirmed):
- admission / pool saturation (queue full)  -> 429 queue_full      (+ Retry-After short)
- loading / reloading / shutting-down / dead-or-respawning child / admission
  stopped / engine-not-ready                 -> 503 engine_not_ready (+ Retry-After long)
- request validation / bad params           -> 400/422 invalid_request_error

This module imports NO mlx (only fastapi), so the process-mode parent stays
mlx-free.
"""

from typing import Optional

from fastapi.responses import JSONResponse

# --- canonical type / code vocabulary --------------------------------------
# 400 / 422 client errors keep OpenAI's ``invalid_request_error`` type (clients
# already branch on it) and add a stable machine ``code``.
TYPE_INVALID_REQUEST = "invalid_request_error"
CODE_INVALID_REQUEST = "invalid_request"

# 429 — the engine is UP but momentarily out of admission/pool capacity.
TYPE_QUEUE_FULL = "queue_full"
CODE_QUEUE_FULL = "queue_full"

# 503 — the engine is not accepting work (loading/reloading/shutting down, a
# dead/respawning process-mode child, admission stopped).
TYPE_ENGINE_NOT_READY = "engine_not_ready"
CODE_ENGINE_NOT_READY = "engine_not_ready"

# 503 — a bounded read could not be served because a generation holds the
# engine (local endpoint degrade; distinct from the not-ready lifecycle state).
TYPE_ENGINE_BUSY = "engine_busy"
CODE_ENGINE_BUSY = "engine_busy"

# 500 — internal/server error (e.g. fail-closed cache-corruption termination).
TYPE_SERVER_ERROR = "server_error"
CODE_SERVER_ERROR = "server_error"

# 404 — resource (session) not found.
TYPE_NOT_FOUND = "not_found"
CODE_NOT_FOUND = "not_found"

# 501 — an OPTIONAL capability (json_schema structured output) is unavailable
# because its server-side dependency (outlines_core) is not installed. This is
# a SERVER setup condition, NOT the client's fault: a perfectly valid request
# cannot be honored. It is distinct from 400 invalid_request (the request is
# fine) AND from 503 engine_not_ready (the engine IS ready; only this optional
# feature is missing, and retrying will not make the package appear — so no
# Retry-After). The body NEVER carries the underlying ImportError text.
TYPE_NOT_SUPPORTED = "not_supported"
CODE_STRUCTURED_OUTPUT_UNAVAILABLE = "structured_output_unavailable"

# --- Retry-After defaults (seconds, as header strings) ---------------------
RETRY_AFTER_QUEUE_FULL = "1"          # transient saturation: retry very soon
RETRY_AFTER_ENGINE_NOT_READY = "30"   # model load/reload/restart: retry later


def error_dict(message: str, type_: str, code: str) -> dict:
    """The canonical error envelope as a plain dict (for SSE frames + bodies)."""
    return {"error": {"message": message, "type": type_, "code": code}}


def error_response(
    status_code: int,
    message: str,
    type_: str,
    code: str,
    *,
    retry_after: Optional[str] = None,
) -> JSONResponse:
    """A JSONResponse carrying the canonical envelope, with an optional
    Retry-After header (429/503 advisories)."""
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    return JSONResponse(
        status_code=status_code,
        content=error_dict(message, type_, code),
        headers=headers,
    )


# --- ready-made responses for the common outcomes --------------------------


def invalid_request_response(message: str) -> JSONResponse:
    """400 invalid_request_error envelope (adds the stable ``code``)."""
    return error_response(
        400, message, TYPE_INVALID_REQUEST, CODE_INVALID_REQUEST
    )


def queue_full_response(
    message: str = "engine queue is full, retry shortly",
) -> JSONResponse:
    """429 queue_full envelope + short Retry-After (admission saturation)."""
    return error_response(
        429, message, TYPE_QUEUE_FULL, CODE_QUEUE_FULL,
        retry_after=RETRY_AFTER_QUEUE_FULL,
    )


def engine_not_ready_response(
    message: str = "engine not ready, retry shortly",
    *,
    retry_after: str = RETRY_AFTER_ENGINE_NOT_READY,
) -> JSONResponse:
    """503 engine_not_ready envelope + Retry-After (engine unavailable)."""
    return error_response(
        503, message, TYPE_ENGINE_NOT_READY, CODE_ENGINE_NOT_READY,
        retry_after=retry_after,
    )


def server_error_response(
    message: str = "internal server error",
) -> JSONResponse:
    """500 server_error envelope (Batch B canonical shape).

    Used for uncaught exceptions (finding 6: the outer 500 handler) — NEVER
    carries the underlying exception text (that is logged server-side only)."""
    return error_response(
        500, message, TYPE_SERVER_ERROR, CODE_SERVER_ERROR
    )


def structured_output_unavailable_response(
    message: str = (
        "json_schema structured output is not supported by this server"
    ),
) -> JSONResponse:
    """501 Not Implemented: the optional structured-output dependency
    (outlines_core) is not installed server-side.

    The client's request is VALID, so this is NOT a 400 invalid_request; and
    retrying will not make the missing package appear, so it is NOT a 503
    engine_not_ready (no Retry-After). NEVER carries the underlying ImportError
    text — the real exception is logged server-side by the caller."""
    return error_response(
        501, message, TYPE_NOT_SUPPORTED, CODE_STRUCTURED_OUTPUT_UNAVAILABLE
    )
