"""Additive server-info endpoints (bultagi.com integration, Batch A).

POST /tokenize — tokenize text with the loaded tokenizer.
GET  /ready    — model + tokenizer readiness probe (200 ready / 503 not ready).

Both are purely additive and read-only: they never touch the generation or
cache paths. They resolve the engine from the same registry the OpenAI-compat
router uses (set on startup) and reuse the project's standard error envelopes.
In process mode the engine is the ``EngineProcessProxy``; ``tokenize`` forwards
to the child via the generic RPC, while readiness is answered from the proxy's
own liveness state (no RPC, no GPU lock).
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mlx_soloheaven.api.errors import (
    engine_not_ready_response,
    invalid_request_response,
)
from mlx_soloheaven.executors import run_read
from mlx_soloheaven import inference_queue

if TYPE_CHECKING:
    # Type-only import: keeps mlx.core / mlx_vlm out of the FastAPI parent
    # process so `--engine-mode process` actually isolates MLX in the child.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Engine registry set by server.py on startup (mirrors openai_compat).
_engines: "dict[str, MLXEngine]" = {}
_default_engine: "MLXEngine" = None  # type: ignore


def set_engines(engines: "dict[str, MLXEngine]", default: "MLXEngine"):
    global _engines, _default_engine
    _engines = engines
    _default_engine = default


def _invalid_request(message: str) -> JSONResponse:
    """OpenAI-style 400 invalid_request_error envelope — shares the canonical
    {message,type,code} builder with openai_compat (batch B: adds ``code``)."""
    return invalid_request_response(message)


def _not_ready_response() -> JSONResponse:
    """503 engine_not_ready envelope for a not-yet-ready engine (model/
    tokenizer not loaded, or a process-mode child dead/respawning). Batch B:
    canonical {message,type,code} + Retry-After via the shared builder."""
    return engine_not_ready_response("engine not ready")


class TokenizeRequest(BaseModel):
    # content is intentionally OPTIONAL at the schema level so a missing/empty
    # value yields our standard 400 envelope (below) rather than FastAPI's 422.
    content: Optional[str] = None
    add_special: bool = False
    with_pieces: bool = False


@router.post("/tokenize")
async def tokenize_endpoint(body: TokenizeRequest):
    """Tokenize ``content`` with the default engine's tokenizer.

    Response: ``{"tokens": [int, ...], "count": int}``, plus
    ``"pieces": [{"id": int, "piece": str}, ...]`` when ``with_pieces`` is set.
    A missing/empty ``content`` -> 400 (standard envelope). A not-ready engine
    (tokenizer unavailable / process-mode child restarting) -> 503: the guard
    below catches the pre-load case, and an in-flight EngineRestartingError /
    EngineBusyError from the RPC path is mapped to 503 by the app-level engine
    error handlers.
    """
    if not body.content:
        return _invalid_request("content is required and must be non-empty")

    engine = _default_engine
    if engine is None or not engine.is_ready():
        return _not_ready_response()

    # Off the event loop on the reserved read pool (U14/F2): tokenize is a pure
    # CPU op in-process, and in process mode it round-trips to the child — both
    # must not block the loop. EngineRestartingError / EngineBusyError propagate
    # to the app-level handlers (-> 503).
    result = await run_read(
        engine.tokenize,
        body.content,
        add_special=bool(body.add_special),
        with_pieces=bool(body.with_pieces),
    )
    return result


@router.get("/ready")
async def ready_endpoint():
    """Readiness probe: 200 ONLY when the model AND tokenizer are loaded and
    able to serve a request; 503 otherwise.

    POLICY (Batch A): readiness keys off model+tokenizer readiness ONLY. A
    saturated inference queue / all-workers-busy / GPU-lock-held engine is
    STILL ready (200) — queue saturation is not a readiness failure. The probe
    never consults admission/pool state or acquires the GPU lock; it only reads
    already-published engine attributes (in process mode: the proxy's local
    liveness state) plus the parent-side gate counter.

    Batch C: ``queue_length`` is now the REAL inference-queue depth
    (waiting + running) from the parent-side FIFO admission gate — the same
    counter that bounds admission and answers 429 when saturated — rather than
    the Batch A in-flight probe. A saturated-but-ready queue still returns 200.
    """
    engine = _default_engine
    if engine is None or not engine.is_ready():
        return _not_ready_response()

    try:
        queue_length = int(inference_queue.current_queue_length())
    except Exception:  # noqa: BLE001 — a probe must never fail readiness
        queue_length = 0

    return {
        "status": "ready",
        "model": engine.model_id,
        "context_length": getattr(engine, "max_context_length", None),
        "queue_length": queue_length,
    }
