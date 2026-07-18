"""
OpenAI-compatible API endpoints.
POST /v1/chat/completions — with streaming SSE and tool calling
GET  /v1/models
"""

import asyncio
import json
import math
import time
import uuid
import logging
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChunkChoice,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    ResponseMessage,
    ToolCall,
    FunctionCall,
    UsageInfo,
)
from mlx_soloheaven.api.errors import (
    error_dict,
    error_response,
    invalid_request_response,
    structured_output_unavailable_response,
    CODE_ENGINE_BUSY,
    CODE_ENGINE_NOT_READY,
    CODE_NOT_FOUND,
    CODE_SERVER_ERROR,
    RETRY_AFTER_ENGINE_NOT_READY,
    TYPE_ENGINE_BUSY,
    TYPE_ENGINE_NOT_READY,
    TYPE_NOT_FOUND,
    TYPE_SERVER_ERROR,
)
from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.types import EngineBusyError
from mlx_soloheaven.executors import run_critical, run_long, run_read
from mlx_soloheaven import inference_queue
from mlx_soloheaven.api import metrics
from mlx_soloheaven.api.gate_stream import (
    SlotStreamingResponse,
    closed_completion_response,
    closed_stream_response,
)
from mlx_soloheaven.engine.tool_parser import (
    CHANNEL_REASONING,
    ThinkingRouter,
    _partial_marker_tail,
    generate_call_id,
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
    strip_thinking_tags,
    thinking_router_active,
)

if TYPE_CHECKING:
    # Type-only import: keeps mlx.core/mlx_vlm out of the FastAPI parent
    # process so `--engine-mode process` actually isolates MLX in the child.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Engine registry set by server.py on startup
_engines: dict[str, "MLXEngine"] = {}
_default_engine: "MLXEngine" = None  # type: ignore


def set_engines(engines: dict[str, "MLXEngine"], default: "MLXEngine"):
    global _engines, _default_engine
    _engines = engines
    _default_engine = default


def _get_engine(model: str) -> "MLXEngine":
    """Resolve model name to engine. Tries exact match, then substring match."""
    if model in _engines:
        return _engines[model]
    # Substring match: "qwen3.5-122b" matches "Qwen3.5-122B-A10B-8bit"
    model_lower = model.lower()
    for key, engine in _engines.items():
        if model_lower in key.lower() or model_lower in engine.model_id.lower():
            return engine
    return _default_engine


class _Gemma4ThinkingStripper:
    """Backward-compatible shim over the shared ``ThinkingRouter`` (FIX 3).

    The reasoning-channel feature replaced the old strip-and-drop stripper with
    ``tool_parser.ThinkingRouter``, which ROUTES the gemma4 thought channel to a
    reasoning stream instead of dropping it. This shim keeps the old
    ``feed(text) -> content_str`` / ``flush() -> content_str`` surface (returns
    ONLY the content-channel text, dropping reasoning) so prior callers/tests
    that just want the stripped answer still work. New code should use
    ``ThinkingRouter`` directly to obtain BOTH channels.
    """

    def __init__(self, active: bool):
        self.active = active
        self._router = ThinkingRouter(active=active, model_family="gemma4")

    def feed(self, text: str) -> str:
        if not self.active:
            return text
        return "".join(
            t for ch, t in self._router.feed(text) if ch != CHANNEL_REASONING
        )

    def flush(self) -> str:
        return "".join(
            t for ch, t in self._router.flush() if ch != CHANNEL_REASONING
        )


def _invalid_request(message: str) -> JSONResponse:
    """OpenAI-style 400 invalid_request_error envelope (U24/U20 shape).

    Batch B: delegates to the shared builder so every 400 carries the same
    {message, type, code} shape (code="invalid_request")."""
    return invalid_request_response(message)


def _compile_response_format_schema(schema: dict) -> None:
    """Compile a ``response_format`` json_schema up front to validate it.

    Isolated into a named module-level function for two reasons:
      1. The compile dependency stays a LOCAL import. Importing the mlx-backed
         ``engine.structured`` module here would pull ``mlx.core`` into the
         process-mode PARENT process, defeating the isolation the TYPE_CHECKING
         import guard preserves — so we call ``outlines_core`` directly, which
         needs no mlx and no tokenizer for the regex-compile validation.
      2. Tests can monkeypatch this single entry point to exercise the
         acceptance / dependency-missing / malformed-schema branches
         deterministically, without the optional ``outlines_core`` package.

    Distinguishable failure classes (the CALLER maps them to statuses):
      - ImportError / ModuleNotFoundError -> the optional ``outlines_core``
        dependency is absent: a SERVER setup condition, not a client error.
      - any other exception (ValueError, ...) -> the schema itself is invalid.
    """
    from outlines_core.json_schema import build_regex_from_schema
    build_regex_from_schema(json.dumps(schema))


def _effective_thinking(request: ChatCompletionRequest, default: bool) -> bool:
    """Resolve the ONE effective thinking flag from every input the request
    can carry (batch B item 3).

    Two inputs can drive thinking on/off:
      - the top-level ``thinking`` flag (this API's native switch), and
      - ``chat_template_kwargs.enable_thinking`` (the nested form spec clients
        such as bultagi send: {"chat_template_kwargs":{"enable_thinking":false}}).

    PRECEDENCE (documented): an explicit top-level ``thinking`` wins when BOTH
    are set (it is this API's first-class parameter); otherwise a boolean
    ``chat_template_kwargs.enable_thinking`` drives it; otherwise the server
    default. Both inputs are normalized to one bool HERE so every downstream
    consumer (prompt render, history strip, reasoning router) sees a single
    contract — the top-level flag and the nested flag behave identically."""
    if request.thinking is not None:
        return bool(request.thinking)
    ctk = request.chat_template_kwargs
    if ctk:
        et = ctk.get("enable_thinking")
        if isinstance(et, bool):
            return et
    return bool(default)


def _validate_sampling_params(request: ChatCompletionRequest) -> Optional[JSONResponse]:
    """U20: validate sampling parameters at the API boundary.

    Unvalidated values reached the engine unclamped: repetition_penalty=0
    divides logits by zero (inf/NaN cascade), top_k > vocab raised inside the
    compiled top-k filter mid-generation, negative temperature inverts the
    distribution. Reject with a 400 + OpenAI error envelope HERE (before any
    stream starts); the engine additionally keeps cheap defensive clamps
    (top_k capped to vocab in the eager sampler, non-positive
    repetition_penalty neutralized) for non-API callers.

    Returns a 400 JSONResponse on the first violation, else None. Ranges:
      temperature >= 0; top_p in (0, 1]; min_p in [0, 1]; top_k >= 0 (excess
      is capped to vocab engine-side, not an error); repetition_penalty > 0
      (multiplicative — sane range ~0.1-2.0, 1.0 disables);
      frequency/presence_penalty in [-2, 2] (OpenAI); max_tokens /
      max_completion_tokens >= 1.

    Round 2 (codex F6): every float param is FIRST required to be finite —
    Python's json parser (via Starlette) accepts the NaN/Infinity literals
    and pydantic floats admit them, and NaN sails through every comparison
    below (``nan < 0`` is False), reaching the engine as a poisoned
    temperature/penalty. Non-finite => 400.
    """
    for name, val in (
        ("temperature", request.temperature),
        ("top_p", request.top_p),
        ("min_p", request.min_p),
        ("repetition_penalty", request.repetition_penalty),
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
    ):
        if val is not None and not math.isfinite(val):
            return _invalid_request(
                f"'{name}': must be a finite number, got {val}"
            )
    if request.temperature is not None and request.temperature < 0:
        return _invalid_request(
            f"'temperature': must be >= 0, got {request.temperature}"
        )
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return _invalid_request(
            f"'top_p': must be in (0, 1], got {request.top_p}"
        )
    if request.min_p is not None and not (0 <= request.min_p <= 1):
        return _invalid_request(
            f"'min_p': must be in [0, 1], got {request.min_p}"
        )
    if request.top_k is not None and request.top_k < 0:
        return _invalid_request(
            f"'top_k': must be >= 0, got {request.top_k}"
        )
    if request.repetition_penalty is not None and request.repetition_penalty <= 0:
        return _invalid_request(
            f"'repetition_penalty': must be > 0 (multiplicative; 1.0 "
            f"disables, sane range 0.1-2.0), got {request.repetition_penalty}"
        )
    for name, val in (
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
    ):
        if val is not None and not (-2 <= val <= 2):
            return _invalid_request(
                f"'{name}': must be in [-2, 2], got {val}"
            )
    for name, val in (
        ("max_tokens", request.max_tokens),
        ("max_completion_tokens", request.max_completion_tokens),
    ):
        if val is not None and val < 1:
            return _invalid_request(
                f"'{name}': must be >= 1, got {val}"
            )
    return None


# --- POST /v1/chat/completions ---

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request = None
):
    engine = _get_engine(request.model)

    # Fail FAST while the process-mode child worker is dead/respawning:
    # raising HERE (before StreamingResponse is returned) lets the server's
    # EngineRestartingError handler answer a real HTTP 503 instead of a 200
    # with a dead stream. In-process engines have no ensure_available.
    ensure_available = getattr(engine, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    # U20: sampling parameters are validated up front (before any streaming
    # response starts) — see _validate_sampling_params for the ranges.
    sampling_error = _validate_sampling_params(request)
    if sampling_error is not None:
        return sampling_error

    # U24: validate the ``stop`` parameter early (OpenAI: a string or an
    # array of up to 4 non-empty strings). Anything else is a 400, matching
    # OpenAI's invalid_request_error behavior.
    if request.stop is not None:
        stops = [request.stop] if isinstance(request.stop, str) else list(request.stop)
        if len(stops) > 4:
            return _invalid_request("'stop': maximum of 4 stop sequences allowed")
        if any(not isinstance(s, str) or not s for s in stops):
            return _invalid_request(
                "'stop': stop sequences must be non-empty strings"
            )

    # Validate response_format up front (batch B item 4). ``type`` is now a
    # plain string in the schema (not a pydantic Literal), so an unsupported
    # value reaches here as data instead of tripping FastAPI's 422 — reject it
    # with an explicit 400 (spec: unsupported options must be 400, not a 422
    # nor a silent ignore). Also reject tools+response_format explicitly
    # instead of silently DROPPING response_format (the old behavior).
    rf = request.response_format
    if rf is not None:
        supported_types = ("text", "json_object", "json_schema")
        if rf.type not in supported_types:
            return _invalid_request(
                f"response_format.type {rf.type!r} is not supported "
                f"(supported: {', '.join(supported_types)})"
            )
        # tools + a JSON-constraining response_format are mutually exclusive:
        # a grammar FSM and free-form tool-call emission cannot both drive the
        # decoder. type="text" imposes no constraint, so it is harmless with
        # tools and stays allowed.
        if request.tools and rf.type in ("json_object", "json_schema"):
            return _invalid_request(
                "response_format and tools are mutually exclusive: send one "
                "or the other (response_format.type='text' is unconstrained "
                "and may be combined with tools)"
            )
        # json_schema requires a schema, and the schema must compile — 400 on
        # malformed input (matches OpenAI; avoids a silent fallback to
        # unconstrained generation).
        if rf.type == "json_schema":
            js = rf.json_schema
            if not js or not js.schema_:
                return _invalid_request(
                    "response_format.json_schema.schema is required when "
                    "type=json_schema"
                )
            try:
                _compile_response_format_schema(js.schema_)
            except (ModuleNotFoundError, ImportError):
                # DEPENDENCY / SETUP failure — NOT the client's fault. The
                # optional ``outlines_core`` schema compiler is not installed,
                # so a perfectly VALID json_schema request cannot be honored.
                # The old f-string returned a 400 that both (a) wrongly blamed
                # the client and (b) leaked "No module named 'outlines_core'"
                # into the body. 501 Not Implemented is the honest status: the
                # feature is unsupported on THIS server, and retrying will not
                # make the package appear (no Retry-After). It is deliberately
                # NOT 503 engine_not_ready — the engine IS ready; only this
                # optional capability is missing. The real ImportError is
                # logged server-side, never sent to the client.
                logger.warning(
                    "[Structured] json_schema requested but the outlines_core "
                    "schema compiler is unavailable — returning 501",
                    exc_info=True,
                )
                return structured_output_unavailable_response()
            except Exception:
                # MALFORMED CLIENT SCHEMA — the compiler (dependency present)
                # rejected an actually-invalid schema. 400 with a GENERIC
                # message; the raw compiler exception is logged server-side but
                # NEVER echoed into the response body (Batch B no-str(exc)
                # contract — the raw text can carry internal detail).
                logger.warning(
                    "[Structured] invalid json_schema in response_format",
                    exc_info=True,
                )
                return _invalid_request(
                    "Invalid JSON schema in response_format"
                )

    # Batch D log-leak scrub: log SAFE request metadata ONLY — never the
    # message/prompt CONTENT. Prompts can carry user PII/secrets and every log
    # record is broadcast to the admin log SSE, so we emit the role sequence +
    # per-message content LENGTHS (chars) + counts, with the text redacted.
    role_summary = []
    for m in request.messages[:8]:
        raw = m.content
        clen = len(str(raw)) if isinstance(raw, list) else len(raw or "")
        role_summary.append(f"{m.role}({clen})")
    roles_str = " ".join(role_summary)
    if len(request.messages) > 8:
        roles_str += f" ...+{len(request.messages) - 8} more"
    logger.info(
        f"[Request] user={request.user!r}, model={request.model} -> {engine.model_id}, "
        f"stream={request.stream}, thinking={request.thinking}, "
        f"max_tokens={request.max_tokens or request.max_completion_tokens}, "
        f"messages={len(request.messages)} | roles=[{roles_str}] (content redacted)"
    )
    if request.stream:
        # Batch C: FIFO + bounded inference-queue admission. Acquire the
        # single generation slot BEFORE returning StreamingResponse, so a
        # saturated queue answers a REAL 429 queue_full (once the SSE 200 has
        # started, a status can no longer be sent) and a shutting-down server
        # answers 503 — both raised immediately (no await) by the gate and
        # mapped by the app-level EngineBusyError handler. FIFO waiting blocks
        # here (the client waits for headers).
        #
        # Finding 2: a real queued HTTP disconnect does NOT cancel this
        # coroutine (uvicorn only wakes receive with http.disconnect), so we
        # RACE the acquire against a disconnect watcher on the ASGI receive
        # channel. If the client left while queued, drop the ticket and return
        # a closed stream — never enter generation.
        # Findings 3+4: the slot is owned by SlotStreamingResponse, which
        # closes the inner generation stream (full U13/C1 teardown) BEFORE
        # releasing the lease, on EVERY exit — including a pre-body cancel
        # (before the body generator starts) and a mid-send cancel.
        gate = inference_queue.get_inference_gate()
        receive = raw_request.receive if raw_request is not None else None
        # Batch D metrics: queue_wait == the gate acquire duration (time between
        # the acquire() request and the slot grant). Measured here, threaded into
        # the stream body which records the per-request metric at completion.
        _t_acquire = time.perf_counter()
        try:
            lease = await inference_queue.acquire_or_disconnect(gate, receive)
        except inference_queue.ClientDisconnected:
            logger.info(
                f"[Request] user={request.user!r} | client disconnected while "
                f"queued on the inference gate — not generating"
            )
            return closed_stream_response()
        queue_wait = time.perf_counter() - _t_acquire
        # Finding 3: ARM the deferred metric at generation START (right after the
        # slot is acquired), defaulting to a CANCEL terminal, so EVERY admitted
        # request records exactly one metric even if it dies before the body's
        # explicit terminal record (a client disconnect / mid-frame cancel /
        # EngineRestartingError). The body UPDATEs ttft/tokens/finish/cache/error
        # as data arrives; SlotStreamingResponse FIRES it via on_release, strictly
        # AFTER the lease is released (batch-C: never inside the lease window).
        metric_sink = metrics.DeferredRequestMetric()
        metric_sink.arm(
            model=getattr(engine, "model_id", None),
            queue_wait=queue_wait,
            finish_reason="cancel",
            error_code="cancel",
        )
        return SlotStreamingResponse(
            gate,
            lease,
            _stream_completion(
                request, engine, queue_wait=queue_wait, metric_sink=metric_sink,
            ),
            on_release=metric_sink.fire,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # U14: _sync_completion blocks on the engine for the WHOLE
        # generation (in-process: inline; process mode: a synchronous RPC
        # the child serves only between generations). Run it off the event
        # loop so concurrent SSE streams / other endpoints stay live.
        # F2: on the bounded LONG-ops executor (not to_thread's shared
        # default pool) so a burst of these can never exhaust the workers
        # bounded reads (/health, admin) need to even start.
        # Batch C: the same FIFO gate fronts the non-streaming path (acquire /
        # run / release all in this coroutine). Queue-full -> 429, shutdown ->
        # 503; run_long underneath is now uncontended on the API path.
        #
        # Batch C round 3, finding 3: make the non-streaming path
        # disconnect-aware too. uvicorn does NOT cancel the app task on socket
        # loss, so a bare inference_slot() would PROMOTE a dead queued request
        # and run the full _sync_completion / engine generation. Race the
        # acquire against the ASGI disconnect watcher (as the streaming path
        # does): a client that left while queued is dropped BEFORE
        # _sync_completion / engine entry, and the lease is released.
        gate = inference_queue.get_inference_gate()
        receive = raw_request.receive if raw_request is not None else None
        # Batch D metrics: time the gate acquire (queue_wait) and the generation.
        _t_acquire = time.perf_counter()
        try:
            lease = await inference_queue.acquire_or_disconnect(gate, receive)
        except inference_queue.ClientDisconnected:
            logger.info(
                f"[Request] user={request.user!r} | client disconnected while "
                f"queued (non-streaming) — not generating"
            )
            # The client is gone; the body is never read. Return a clean empty
            # response (the lease was already released inside
            # acquire_or_disconnect before ClientDisconnected was raised).
            return closed_completion_response()
        queue_wait = time.perf_counter() - _t_acquire
        # Finding 3: ARM the metric at generation START (post-admission),
        # defaulting to an ERROR terminal, so an exception AFTER admission
        # (EngineRestarting/Busy/any) still records exactly one metric. The
        # normal path UPDATEs it with the real result; the fire happens in the
        # finally, strictly AFTER the lease release (batch-C invariant).
        metric_sink = metrics.DeferredRequestMetric()
        metric_sink.arm(
            model=getattr(engine, "model_id", None),
            queue_wait=queue_wait,
            finish_reason="error",
            error_code="server_error",
        )
        response = None
        _gen_elapsed: Optional[float] = None
        _t_gen = time.perf_counter()
        try:
            response = await run_long(_sync_completion, request, engine)
            _gen_elapsed = time.perf_counter() - _t_gen
            _record_completion_metrics(
                engine, response,
                queue_wait=queue_wait,
                generation_time=_gen_elapsed,
                sink=metric_sink,
            )
            return response
        except Exception as exc:
            # Finding 3: an exception after admission still emits one metric —
            # error_code by class — fired after the lease release below.
            metric_sink.update(
                finish_reason="error",
                error_code=_error_code_for_exception(exc),
                generation_time=time.perf_counter() - _t_gen,
            )
            raise
        finally:
            # Finding 3(b): release the lease FIRST, then fire the per-request
            # metric — strictly OUTSIDE the lease window, so a metrics-lock stall
            # (e.g. a slow /metrics scrape) can never extend the inference lease.
            gate.release(lease)
            metric_sink.fire()


def _error_code_for_exception(exc: Exception) -> str:
    """Bounded ``error_code`` metric label for a post-admission exception
    (finding 3). EngineRestartingError -> engine_not_ready; EngineBusyError ->
    its reason (queue_full / engine_not_ready); anything else -> server_error.
    Kept to the SAME small vocabulary the HTTP error contract uses so the metric
    cardinality stays bounded."""
    if isinstance(exc, EngineRestartingError):
        return "engine_not_ready"
    if isinstance(exc, EngineBusyError):
        reason = getattr(exc, "reason", EngineBusyError.REASON_ENGINE_NOT_READY)
        return "queue_full" if reason == EngineBusyError.REASON_QUEUE_FULL else "engine_not_ready"
    return "server_error"


def _record_completion_metrics(
    engine, response, *, queue_wait=None, generation_time=None, sink=None
) -> None:
    """Record the Batch D per-request metric for a NON-STREAMING completion.

    ``response`` is a ChatCompletionResponse (normal) or a JSONResponse error
    object (fail-closed 500 corruption). TTFT is not observed here — a
    non-streaming request has no first-token event; queue_wait + generation_time
    are. Defensive: never raises (metrics must not break a response).

    Finding 3: when ``sink`` is a ``DeferredRequestMetric`` the record is MERGED
    into it (enriching the arm-at-start baseline) and FIRED later by the handler
    finally, AFTER the lease is released; without a sink it records immediately."""
    try:
        model = getattr(engine, "model_id", None)
        if isinstance(response, ChatCompletionResponse):
            usage = response.usage
            finish = response.choices[0].finish_reason if response.choices else None
            cache_result, reused = metrics.cache_result_from_info(
                getattr(usage, "cache_info", None)
            )
            kwargs = dict(
                model=model,
                finish_reason=finish,
                error_code="none",
                cache_result=cache_result,
                reused_tokens=reused,
                queue_wait=queue_wait,
                generation_time=generation_time,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        else:
            # A JSONResponse error object (e.g. 500 fail-closed cache corruption).
            kwargs = dict(
                model=model,
                finish_reason="error",
                error_code="server_error",
                queue_wait=queue_wait,
                generation_time=generation_time,
            )
        if sink is not None:
            sink.update(**kwargs)
        else:
            metrics.observe_request(**kwargs)
    except Exception:  # noqa: BLE001 — metrics must never break the response
        logger.debug("metrics: completion record failed (ignored)", exc_info=True)


def _record_stream_metrics(
    engine, *, finish_reason, error_code, cache_info, queue_wait,
    t_body_start, t_first_token, prompt_tokens, completion_tokens, sink=None,
) -> None:
    """Record the Batch D per-request metric for a STREAMING completion at its
    terminal frame. Finding 7: TTFT = FIRST engine token frame - body start
    (anchored the same way as the web-chat path); generation_time = completion -
    body start. Defensive: never raises.

    Finding 3(b): when ``sink`` is a ``DeferredRequestMetric`` the (lock-taking)
    record is ARMED here — computed cheaply and lock-free — and FIRED later by
    ``SlotStreamingResponse`` AFTER the inference lease is released, so metrics
    recording never sits inside the lease window. Without a sink (direct callers)
    it records immediately."""
    try:
        ttft = (t_first_token - t_body_start) if t_first_token else None
        generation_time = time.perf_counter() - t_body_start
        cache_result, reused = metrics.cache_result_from_info(cache_info)
        kwargs = dict(
            model=getattr(engine, "model_id", None),
            finish_reason=finish_reason,
            error_code=error_code,
            cache_result=cache_result,
            reused_tokens=reused,
            queue_wait=queue_wait,
            ttft=ttft,
            generation_time=generation_time,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )
        if sink is not None:
            # Finding 3: MERGE onto the early-armed baseline (arm-at-start) so a
            # terminal record enriches it; the fire happens after lease release.
            sink.update(**kwargs)
        else:
            metrics.observe_request(**kwargs)
    except Exception:  # noqa: BLE001 — metrics must never break the stream
        logger.debug("metrics: stream record failed (ignored)", exc_info=True)


def _sync_completion(request: ChatCompletionRequest, engine: "MLXEngine"):
    """Non-streaming completion. BLOCKING — callers on the event loop must
    run it off-loop via ``executors.run_long`` (U14/F2).

    Returns a ChatCompletionResponse, or a 500 JSONResponse error object
    when the engine terminated the stream fail-closed (U6/F1: 'error' is an
    engine-internal reason, never a valid OpenAI finish_reason)."""
    # Batch B item 3: normalize top-level `thinking` and the nested
    # chat_template_kwargs.enable_thinking to ONE effective bool.
    enable_thinking = _effective_thinking(request, engine.cfg.enable_thinking)
    tools = [t.model_dump() for t in request.tools] if request.tools else None

    response_format = request.response_format
    if response_format and tools:
        logger.warning(
            f"[Structured] response_format={response_format.type} ignored: "
            f"tools are present (OpenAI behavior)."
        )
        response_format = None

    # U9: structured output suppresses thinking for the request. The FSM
    # only permits JSON-grammar tokens, so a thinking prompt opener would
    # pair an unclosed <think> with a pure-JSON stream — the whole answer
    # would be routed to reasoning_content and content would stay empty.
    # The engine applies the same suppression (prompt rendered without the
    # thinking opener); this keeps the API-side split consistent with it.
    # Codex round 9, finding 3: resolved BEFORE the input-history strip —
    # enable_thinking below is the EFFECTIVE contract the whole request
    # (prompt render, router, history preprocessing) runs under.
    if response_format and response_format.type in ("json_schema", "json_object"):
        enable_thinking = False

    # FIX 3: pass model_family so Gemma 4 <|channel>thought...<channel|> spans
    # in the INPUT history are actually stripped (the default "chatml" left
    # them — and degenerate trailing reasoning — to replay raw into the prompt).
    # Codex round 7, finding 3: thread the request's thinking contract so the
    # AMBIGUOUS gemma4 bare ``thought\n`` opener is only stripped when
    # thinking is active (full markers strip regardless). Codex round 9,
    # finding 3: the contract is the EFFECTIVE one (structured suppression
    # applied above) — pre-fix the strip ran under the request flag while the
    # engine rendered under thinking=False, mutilating gemma4 bare-opener
    # history the thinking=False contract preserves.
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages],
        model_family=engine.model_family,
        thinking_active=enable_thinking,
    )

    # Map OpenAI frequency/presence_penalty to repetition_penalty if not explicitly set
    rep_penalty = request.repetition_penalty
    if rep_penalty is None and (request.frequency_penalty or request.presence_penalty):
        # Approximate: OpenAI penalties are additive [-2,2], repetition_penalty is multiplicative [0.1, 2.0]
        # U20: floor at 0.1 — fp+pp == -4 (both at the OpenAI minimum) would
        # map to exactly 0, which divides logits by zero in the penalty.
        fp = request.frequency_penalty or 0.0
        pp = request.presence_penalty or 0.0
        rep_penalty = max(0.1, 1.0 + (fp + pp) * 0.25)  # rough mapping

    result = engine.complete(
        messages,
        max_tokens=request.max_tokens or request.max_completion_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        min_p=request.min_p,
        top_k=request.top_k,
        repetition_penalty=rep_penalty,
        tools=tools,
        session_id=request.user,
        thinking=enable_thinking,
        thinking_budget=request.thinking_budget,
        response_format=response_format,
        stop=request.stop,
    )

    # U6/F1: corruption-terminated generation — return an error response,
    # never a completion (and never tool_calls parsed from truncated text).
    # Nothing is persisted: the engine already skipped its session save and
    # invalidated the cache, so the retry takes an honest MISS.
    if result.finish_reason == "error":
        logger.error(
            f"[Request] user={request.user!r} | generation terminated by "
            f"cache corruption — returning 500 error object"
        )
        return error_response(
            500,
            "generation terminated: session cache corruption detected; "
            "partial output is unreliable and was not persisted — retry the "
            "request",
            TYPE_SERVER_ERROR,
            CODE_SERVER_ERROR,
        )

    msg = ResponseMessage(content=result.content)
    # Expose the model's thinking as a SEPARATE reasoning channel (matches LM
    # Studio's reasoning_content). The engine already split it via
    # split_thinking_and_content; content stays the clean answer.
    if result.thinking:
        msg.reasoning_content = result.thinking
    if result.tool_calls:
        msg.tool_calls = [
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in result.tool_calls
        ]
        msg.content = result.content

    if request.user:
        assistant_msg: dict = {"role": "assistant", "content": result.content or ""}
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in result.tool_calls
            ]
        engine.update_session_messages(request.user, messages + [assistant_msg])

    return ChatCompletionResponse(
        model=request.model,
        choices=[Choice(message=msg, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            cache_info=result.cache_info,
        ),
    )


async def _stream_completion(
    request: ChatCompletionRequest,
    engine: "MLXEngine",
    *,
    queue_wait: Optional[float] = None,
    metric_sink=None,
) -> AsyncGenerator[str, None]:
    """Streaming SSE completion with tool call detection.

    Thin wrapper over ``_stream_completion_body``: once the SSE response has
    started, a 503 can no longer be sent — if the process-mode child worker
    dies mid-stream (EngineRestartingError), emit a proper in-band error
    frame and a terminating [DONE] so the client sees a clean, explained
    close instead of a silently-dead stream.

    Finding 1(a): the inner body generator is closed in a ``finally`` so an
    ``aclose()`` of this wrapper (client disconnect) CASCADES GeneratorExit into
    it — ``async for`` alone does NOT close a nested async generator, so without
    this the engine stream's C1 teardown would run only later (on GC).

    ``metric_sink`` (finding 3b): the deferred per-request metric the body arms
    at its terminal frame; SlotStreamingResponse fires it after the lease is
    released."""
    body = _stream_completion_body(
        request, engine, queue_wait=queue_wait, metric_sink=metric_sink,
    )
    try:
        async for chunk in body:
            yield chunk
    except EngineRestartingError as exc:
        logger.error(
            f"[Stream] user={request.user!r} | engine unavailable mid-stream: {exc}"
        )
        # Finding 3: the engine died mid-stream — the body's terminal record was
        # never reached, so UPDATE the early-armed metric to an error terminal
        # (fired once via on_release after the lease release).
        if metric_sink is not None:
            metric_sink.update(
                finish_reason="error", error_code="engine_not_ready",
            )
        # Batch B: canonical {message,type,code} envelope; type renamed to
        # engine_not_ready; NO detail:str(exc) leak (exc is logged above). The
        # in-band frame fires under the SAME condition as before — only the
        # shape changed.
        err = error_dict(
            "engine not ready, retry shortly",
            TYPE_ENGINE_NOT_READY,
            CODE_ENGINE_NOT_READY,
        )
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Cascade the close down to the engine stream so its C1 teardown runs.
        await body.aclose()


async def _stream_completion_body(
    request: ChatCompletionRequest,
    engine: "MLXEngine",
    *,
    queue_wait: Optional[float] = None,
    metric_sink=None,
) -> AsyncGenerator[str, None]:
    """Streaming SSE completion with tool call detection."""
    # Batch D metrics: mark the body start (generation start, post gate-acquire)
    # and the first-token time for TTFT; record the per-request metric at the
    # terminal points below. ``queue_wait`` is the gate acquire duration measured
    # by the handler.
    _t_body_start = time.perf_counter()
    _t_first_token: Optional[float] = None
    # Determine thinking mode (moved above the input-history strip — codex
    # round 7, finding 3: the strip needs the request's thinking contract).
    # Batch B item 3: top-level `thinking` and nested
    # chat_template_kwargs.enable_thinking normalize to ONE effective bool.
    enable_thinking = _effective_thinking(request, engine.cfg.enable_thinking)
    thinking_budget = request.thinking_budget

    tools = [t.model_dump() for t in request.tools] if request.tools else None
    has_tools = bool(tools)

    # Structured output (response_format): build constraint but skip if
    # tools are present (tools take priority per OpenAI semantics).
    # Resolved BEFORE the ThinkingRouter below — U9 needs the surviving
    # response_format to decide the router's active flag.
    response_format = request.response_format
    if response_format and has_tools:
        logger.warning(
            f"[Structured] response_format={response_format.type} ignored: "
            f"tools are present (OpenAI behavior)."
        )
        response_format = None

    # U9: structured output suppresses thinking for the request. The FSM
    # masks logits to the JSON grammar from the first token, so with the
    # chatml/glm <think> prompt opener the ENTIRE JSON answer streamed as
    # reasoning_content and content stayed empty (no </think> ever comes; a
    # budget-forced one would corrupt the JSON). The engine suppresses the
    # prompt-side opener the same way, so the router must start in content
    # mode to match the actual stream.
    # Codex round 9, finding 3: resolved BEFORE the input-history strip —
    # enable_thinking below is the EFFECTIVE contract the whole request
    # (prompt render, router, history preprocessing) runs under.
    if response_format and response_format.type in ("json_schema", "json_object"):
        if enable_thinking:
            logger.info(
                f"[Structured] thinking suppressed: response_format="
                f"{response_format.type} constrains output to JSON"
            )
        enable_thinking = False

    # FIX 3: pass model_family so Gemma 4 thinking channels (incl. degenerate
    # multi-cycle / trailing reasoning) are stripped from the INPUT history
    # rather than replayed raw into the prompt.
    # Codex round 7, finding 3: thread the thinking contract — the AMBIGUOUS
    # gemma4 bare ``thought\n`` opener is only stripped when thinking is
    # active (full markers strip regardless). Codex round 9, finding 3: the
    # contract is the EFFECTIVE one (structured suppression applied above) —
    # pre-fix the strip ran under the request flag while the engine rendered
    # under thinking=False, mutilating gemma4 bare-opener history the
    # thinking=False contract preserves.
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages],
        model_family=engine.model_family,
        thinking_active=enable_thinking,
    )

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model

    # First chunk: role
    first_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Map OpenAI frequency/presence_penalty to repetition_penalty if not explicitly set
    rep_penalty = request.repetition_penalty
    if rep_penalty is None and (request.frequency_penalty or request.presence_penalty):
        # U20: floor at 0.1 — fp+pp == -4 would map to a zero penalty (see
        # the identical clamp in _sync_completion).
        fp = request.frequency_penalty or 0.0
        pp = request.presence_penalty or 0.0
        rep_penalty = max(0.1, 1.0 + (fp + pp) * 0.25)

    model_family = engine.model_family

    # Reasoning-channel router (shared with the web chat path). Instead of
    # dropping the thought channel, route it: thinking-phase text becomes
    # ``delta.reasoning_content`` (LM-Studio shape), the post-thinking answer
    # becomes ``delta.content``. chatml/glm activate on the thinking flag
    # (their stream begins inside <think>, routed up to </think>; a
    # pass-through when thinking is disabled, so non-thinking output stays
    # byte-identical). Codex round 5, finding 3: gemma4 is MARKER-ACTIVE
    # regardless of the flag (thinking_router_active) — the model emits its
    # own channel markers, and an inactive passthrough let a thought-span
    # tool-call REHEARSAL reach the tool FSM as a real call while the batch
    # parse and the session store excluded it.
    # Codex round 7, finding 3: the effective thinking contract is threaded
    # in — the gemma4 router stays marker-active regardless (full markers
    # authoritative), but the AMBIGUOUS bare ``thought\n`` opener heuristic
    # inside it only fires when thinking is active.
    thinking_router = ThinkingRouter(
        active=thinking_router_active(model_family, enable_thinking),
        model_family=model_family,
        enable_thinking=enable_thinking,
    )
    # NOTE: we no longer inject an opening "<think>\n" content chunk for
    # non-gemma4 — reasoning is now routed to reasoning_content instead of being
    # wrapped in <think> tags inside content.

    # Generate and stream
    # PERF: append-to-list + join at consumption points avoids the O(N^2)
    # cost of repeated ``str += text`` across the streaming loop.
    acc_parts: list[str] = []
    final_prompt_tokens = 0
    final_completion_tokens = 0
    final_cache_info = None
    token_count = 0  # tracked for disconnect diagnostics
    TOOL_START, TOOL_END = get_tool_markers(model_family)
    holdback = ""

    # === Incremental tool_call emission state ===
    # When a <tool_call> block starts, we buffer the per-block text
    # (tc_block). Codex round 3, finding 5: NOTHING is emitted for the block
    # until it is STRUCTURALLY VALIDATED — the old shape emitted the id+name
    # delta as soon as try_extract_tool_name succeeded, so a block whose
    # body later failed to parse produced a PHANTOM tool_calls delta
    # followed by the raw block as content (the client saw a call that
    # never materialized, then finish_reason "stop"). The block closes →
    # parse → on success emit id+name, then arguments (order preserved); on
    # failure the raw block passes through as content ONLY. Tool blocks are
    # small, so full-block buffering costs nothing perceptible, and the
    # round-1 'dangling name chunk' edge (stream ends mid-block after the
    # name was emitted) is gone by construction — no name delta exists
    # before validation. Codex round 11, finding 2: the streamed tool_calls
    # index is assigned POST-validation (len(parsed_tool_calls) at emission
    # time) — the old per-start-marker counter also charged malformed blocks
    # that were passed through as content, leaving HOLES in the emitted
    # index sequence (clients reconstructing calls by index drop/reject).
    tc_active = False           # inside a tool_call block
    tc_block = ""               # buffered text after TOOL_START (excl. start tag itself)
    tc_id: Optional[str] = None
    parsed_tool_calls: list[dict] = []   # completed calls (for session persistence)

    # COALESCING: consume batches of GenerationResult.
    #  - has_tools False: concatenate the batch's content and emit ONE content
    #    delta chunk per batch (reuse the existing chunk builder).
    #  - has_tools True: feed the batch's concatenated text through the existing
    #    tool-call parser, restructured as a loop so an arbitrary chunk that may
    #    contain TOOL_START...TOOL_END (or partial / multiple blocks / trailing
    #    content) is processed correctly. The emitted JSON for both content and
    #    tool_calls is byte-identical to the prior per-token behavior; only the
    #    BATCHING of the content deltas changes.
    finished = False
    # U6/F1: the engine's terminal reason. "error" (fail-closed cache
    # corruption) diverts to the in-band error envelope below instead of a
    # normal completion frame.
    final_finish_reason: Optional[str] = None
    # Finding 1(a): hold the engine stream so its aclose can be cascaded in the
    # finally below — closing THIS body generator (from _stream_completion's
    # finally) must drive the engine generator's GeneratorExit rescue (C1).
    engine_stream = engine.generate_stream_batches_async(
        messages,
        max_tokens=request.max_tokens or request.max_completion_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        min_p=request.min_p,
        top_k=request.top_k,
        repetition_penalty=rep_penalty,
        session_id=request.user,
        tools=tools,
        thinking=enable_thinking,
        thinking_budget=thinking_budget,
        response_format=response_format,
        stop=request.stop,
    )
    try:
        async for batch in engine_stream:
            # Concatenate the batch's content text; capture finish separately.
            chunk_text_parts: list[str] = []
            for result in batch:
                if result.finish_reason is not None:
                    # U6/F1: keep the engine's terminal reason — an "error"
                    # (cache-corruption) terminal must not be dressed up as
                    # a normal stop/tool_calls completion below.
                    final_finish_reason = result.finish_reason
                    final_prompt_tokens = result.prompt_tokens
                    final_completion_tokens = result.completion_tokens
                    final_cache_info = result.cache_info
                    finished = True
                    break
                if result.status == "generating":
                    continue
                # Finding 4/7: anchor TTFT on the FIRST frame that ACTUALLY
                # produced a token (``token_produced``) — even when its detok
                # text is empty (one frame per token, MTP contract) — and NEVER
                # on a keepalive frame (``token_produced=False``, empty text
                # emitted during long prefill). ``token == 0`` / empty ``text``
                # can each be a real token OR a keepalive, so the explicit flag
                # is the only reliable discriminator. Matches the web-chat path.
                if result.token_produced and _t_first_token is None:
                    _t_first_token = time.perf_counter()
                if not result.text:
                    continue
                chunk_text_parts.append(result.text)

            chunk_text = "".join(chunk_text_parts)
            if not chunk_text:
                if finished:
                    break
                # Empty batch (keepalive / status-only) during prompt processing
                yield ": keepalive\n\n"
                continue

            # Keep acc_parts RAW (drives session-persistence split + disconnect
            # tail). Route the OUTBOUND text through the shared ThinkingRouter so
            # thinking-phase text is emitted as reasoning_content deltas and the
            # post-thinking answer as content deltas. For non-thinking output the
            # router is a pass-through (all content). Segments are processed IN
            # ORDER so reasoning/content interleaving (multi-cycle) is preserved.
            acc_parts.append(chunk_text)
            token_count += 1
            # Finding 7: TTFT is anchored on the first TOKEN frame above (incl.
            # empty-detok) — NOT here on the first non-empty visible-text batch.
            segments = thinking_router.feed(chunk_text)
            if not segments:
                # Reasoning-only held partial marker / nothing emittable yet.
                if finished:
                    break
                yield ": keepalive\n\n"
                continue

            # Process each routed segment in order. Reasoning segments emit a
            # reasoning_content delta; content segments go through the existing
            # content / tool-call path (the tool state machine consumes content
            # text only — tool XML never appears on the reasoning channel).
            for seg_channel, seg_text in segments:
                if not seg_text:
                    continue
                if seg_channel == CHANNEL_REASONING:
                    yield f"data: {_make_reasoning_chunk(chunk_id, created, model, seg_text)}\n\n"
                    continue

                if not has_tools:
                    # Plain content path — one delta chunk per content segment.
                    chunk = _make_content_chunk(chunk_id, created, model, seg_text)
                    yield f"data: {chunk}\n\n"
                    continue

                # has_tools True — drive the tool-call state machine over this
                # content segment. ``chunk`` is the remaining unconsumed text;
                # loop until empty or no further progress (partial block / start).
                chunk = seg_text
                while chunk:
                    if tc_active:
                        tc_block += chunk
                        chunk = ""

                        # Check for block close. Codex round 3, finding 5:
                        # the id/name delta is DEFERRED to this point — it is
                        # only emitted once the closed block actually parses,
                        # so a malformed block never produces a phantom
                        # tool_calls delta before its raw passthrough.
                        if TOOL_END in tc_block:
                            end_idx = tc_block.index(TOOL_END)
                            block_text = TOOL_START + tc_block[:end_idx] + TOOL_END
                            _, calls = parse_tool_calls(block_text, model_family=model_family)
                            if not calls:
                                # U11 fail-closed: the closed block did not
                                # parse (e.g. a malformed body) — pass the
                                # RAW block through as content instead of
                                # silently deleting it. No tool_calls delta
                                # was emitted for it (finding 5).
                                content_chunk = _make_content_chunk(
                                    chunk_id, created, model, block_text
                                )
                                yield f"data: {content_chunk}\n\n"
                            if calls:
                                tc = calls[0]
                                # Structurally validated: emit id+name first,
                                # then arguments (same chunk order clients
                                # always saw). Finding 2: the index is the
                                # count of calls parsed SO FAR — contiguous
                                # from 0 regardless of malformed passthrough
                                # blocks in between.
                                tc_index = len(parsed_tool_calls)
                                first = ChatCompletionChunk(
                                    id=chunk_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                        "index": tc_index,
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": "",
                                        },
                                    }]))],
                                )
                                yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
                                args_chunk = ChatCompletionChunk(
                                    id=chunk_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                        "index": tc_index,
                                        "function": {"arguments": tc["function"]["arguments"]},
                                    }]))],
                                )
                                yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n"
                                # Use the id we generated at block-start so session
                                # + SSE agree.
                                tc["id"] = tc_id
                                parsed_tool_calls.append(tc)

                            # Reset for next block; any trailing text after TOOL_END
                            # is re-processed through the holdback/content path.
                            trailing = tc_block[end_idx + len(TOOL_END):]
                            tc_active = False
                            tc_block = ""
                            tc_id = None
                            if trailing:
                                chunk = trailing
                        # else: block still open, wait for more text (chunk empty).
                        continue

                    holdback += chunk
                    chunk = ""

                    if TOOL_START in holdback:
                        idx = holdback.index(TOOL_START)
                        before = holdback[:idx]
                        if before:
                            content_chunk = _make_content_chunk(
                                chunk_id, created, model, before
                            )
                            yield f"data: {content_chunk}\n\n"
                        tc_active = True
                        tc_id = generate_call_id()
                        # Re-feed everything after TOOL_START into the active-block
                        # branch on the next loop turn (handles full block in chunk).
                        chunk = holdback[idx + len(TOOL_START):]
                        holdback = ""
                        continue

                    # FIX 3: a partial start marker may trail VISIBLE content
                    # (e.g. "before <too"). Hold back only the LONGEST SUFFIX of
                    # the buffer that is a prefix of TOOL_START, and emit the
                    # preceding text now. The old check only held when the WHOLE
                    # (stripped) buffer was a prefix, so it leaked the partial
                    # marker as content. ``keep`` covers the all-prefix case too
                    # (keep == len(holdback) -> emit nothing, hold everything).
                    keep = _partial_marker_tail(holdback, (TOOL_START,))
                    emit = holdback[: len(holdback) - keep]
                    holdback = holdback[len(holdback) - keep:]
                    if emit:
                        content_chunk = _make_content_chunk(
                            chunk_id, created, model, emit
                        )
                        yield f"data: {content_chunk}\n\n"

            if finished:
                break
    except (asyncio.CancelledError, GeneratorExit) as exc:
        # Batch D log-leak scrub: do NOT log the generated-output tail (it is
        # conversation content, broadcast to the admin log SSE). Log only the
        # safe token count + generated char length.
        gen_len = len("".join(acc_parts))
        logger.info(
            f"[Stream] user={request.user!r} | client disconnected "
            f"({type(exc).__name__}) after {token_count} tokens | "
            f"generated_chars={gen_len} (content redacted)"
        )
        # Finding 3: enrich the early-armed (cancel) metric with what streamed
        # before the disconnect. It stays finish_reason/error_code=cancel and
        # fires exactly once via on_release AFTER the lease release.
        if metric_sink is not None:
            metric_sink.update(
                finish_reason="cancel",
                error_code="cancel",
                queue_wait=queue_wait,
                ttft=(_t_first_token - _t_body_start) if _t_first_token else None,
                generation_time=time.perf_counter() - _t_body_start,
                completion_tokens=final_completion_tokens,
                prompt_tokens=final_prompt_tokens,
            )
        raise
    finally:
        # Finding 1(a): drive the engine generator's GeneratorExit rescue (C1
        # commit-or-invalidate) synchronously here. On normal completion the
        # engine stream is exhausted (no-op); on a disconnect (this except
        # re-raises) this aclose does not return until C1 has run in BOTH engine
        # modes, so the gate lease released above this chain is freed only after
        # the turn is wound down.
        await engine_stream.aclose()

    # U6/F1: corruption-terminated stream. "error" is NOT a valid OpenAI
    # finish_reason, and the already-streamed text is truncated at an
    # arbitrary point — synthesizing stop/tool_calls here (or best-effort
    # emitting a truncated tool call below) would hand the client corrupt
    # partial output as a successful completion. Emit the same in-band error
    # envelope shape the liveness work uses (data:{"error":{...}} then
    # [DONE]) and SUPPRESS tool-call parsing and session persistence of the
    # truncated text. The engine has already invalidated the session cache;
    # the client's retry takes an honest MISS.
    if final_finish_reason == "error":
        logger.error(
            f"[Stream] user={request.user!r} | generation terminated by "
            f"cache corruption after {token_count} tokens — emitting error "
            f"envelope (no tool_calls, nothing persisted)"
        )
        # Finding 3: record (UPDATE the armed sink) BEFORE the terminal frames,
        # so a disconnect between the error frame and [DONE] still fires the
        # ERROR metric, not the cancel baseline. The fire itself is still after
        # the lease release (on_release).
        _record_stream_metrics(
            engine, finish_reason="error", error_code="server_error",
            cache_info=final_cache_info, queue_wait=queue_wait,
            t_body_start=_t_body_start, t_first_token=_t_first_token,
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
            sink=metric_sink,
        )
        # Batch B: canonical {message,type,code} envelope (code now a string).
        # U6/F1 fail-closed semantics unchanged — same terminal condition,
        # same suppression of tool_calls/persistence; only the shape changed.
        err = error_dict(
            "generation terminated: session cache corruption detected "
            "mid-stream; partial output is unreliable and was not persisted "
            "— retry the request",
            TYPE_SERVER_ERROR,
            CODE_SERVER_ERROR,
        )
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Flush the router's held tail. A partial-opener prefix held back as a
    # possible split marker is real content/reasoning at stream end and must be
    # emitted (a marker that never completed is not a marker). Reasoning that
    # never closed its channel (degenerate) is dropped by the router itself.
    for seg_channel, seg_text in thinking_router.flush():
        if not seg_text:
            continue
        if seg_channel == CHANNEL_REASONING:
            yield f"data: {_make_reasoning_chunk(chunk_id, created, model, seg_text)}\n\n"
        elif not has_tools:
            chunk = _make_content_chunk(chunk_id, created, model, seg_text)
            yield f"data: {chunk}\n\n"
        else:
            # Route content through the post-loop holdback flush below (the held
            # tail cannot be a tool-start fragment — those are held in
            # ``holdback``, not the router — so appending is safe).
            holdback += seg_text

    # Flush remaining holdback (only content path; tool_call active is handled below)
    if holdback and not tc_active:
        chunk = _make_content_chunk(chunk_id, created, model, holdback)
        yield f"data: {chunk}\n\n"

    # If generation ended mid-block (no TOOL_END seen), try best-effort parse
    # so we don't silently drop a tool_call the model truncated. Round-1
    # behavior kept (codex round 3, finding 5): an unparseable truncated
    # block is a raw content passthrough — and since the name delta is now
    # deferred to validation, no dangling name chunk can precede it.
    # Codex round 5, finding 4: the branch runs whenever tc_active — a
    # stream ending EXACTLY at the start marker leaves tc_block EMPTY
    # (tc_active was set, the marker itself already consumed), and the old
    # ``tc_active and tc_block`` guard silently dropped the bare marker
    # while the batch parser preserves it byte-for-byte. An empty body
    # never parses, so it flows through the raw-content passthrough below,
    # emitting the bare marker as content (all families share this FSM —
    # TOOL_START is the family's own marker).
    if tc_active:
        block_text = TOOL_START + tc_block
        _, calls = parse_tool_calls(block_text, model_family=model_family)
        if not calls:
            # U11 fail-closed: the truncated block did not parse — flush the
            # raw text as content rather than silently dropping it.
            content_chunk = _make_content_chunk(
                chunk_id, created, model, block_text
            )
            yield f"data: {content_chunk}\n\n"
        if calls:
            tc = calls[0]
            # Finding 2: post-validation index here too — the end-of-stream
            # flush must stay contiguous with the mid-stream emissions.
            tc_index = len(parsed_tool_calls)
            first = ChatCompletionChunk(
                id=chunk_id, created=created, model=model,
                choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                    "index": tc_index,
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": "",
                    },
                }]))],
            )
            yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
            args_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=model,
                choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                    "index": tc_index,
                    "function": {"arguments": tc["function"]["arguments"]},
                }]))],
            )
            yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n"
            tc["id"] = tc_id
            parsed_tool_calls.append(tc)

    # U7: honor the engine's terminal reason. max_tokens exhaustion must
    # surface as OpenAI finish_reason="length" — the old unconditional
    # "stop" overwrote it. Tool-call turns keep reporting "tool_calls";
    # "error" was already diverted to the error envelope above; anything
    # unexpected falls back to "stop".
    if parsed_tool_calls:
        finish_reason = "tool_calls"
    elif final_finish_reason == "length":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    # Update session — persist tool_calls in assistant message so next turn's
    # chat template can render {% if m.tool_calls %} block (required for
    # multi-turn tool use with stateful clients like OpenClaw).
    if request.user:
        # FIX 1: the chatml/glm stream began inside <think> when thinking was
        # enabled (opener lives in the prompt suffix). Pass that so a degenerate
        # no-</think> turn is persisted as reasoning (content="") — matching what
        # the streaming router already emitted on the wire.
        # Codex round 3, finding 4: with thinking DISABLED the router was a
        # pass-through — the whole text is content and a literal </think> in
        # it is a quote, never a boundary (mirrors the engine's batch parse).
        if model_family != "gemma4" and not enable_thinking:
            thinking, content = None, "".join(acc_parts)
        else:
            # Codex round 7, finding 3: thinking_active threads the contract
            # into the gemma4 split so its bare-opener recognition matches
            # the router that streamed the text.
            thinking, content = split_thinking_and_content(
                "".join(acc_parts),
                model_family=model_family,
                started_in_thinking=enable_thinking and model_family != "gemma4",
                thinking_active=enable_thinking,
            )
        assistant_msg: dict = {"role": "assistant", "content": content or ""}
        if parsed_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in parsed_tool_calls
            ]
        # U14: synchronous engine RPC — keep it off the event loop.
        # Codex round 5, finding 1: this commit runs BETWEEN the streamed
        # tokens and the final chunk/[DONE] below — a saturated long pool
        # used to reject it (EngineBusyError escaped mid-stream, [DONE]
        # never emitted). Guaranteed critical lane + log-only failure: the
        # engine already marked the session dirty at install (finding 1a),
        # so the terminal chunks below must be unconditional.
        try:
            await run_critical(
                engine.update_session_messages, request.user,
                messages + [assistant_msg],
            )
        except Exception:  # noqa: BLE001 — [DONE] must still go out
            logger.exception(
                f"[OpenAI Stream] user={request.user} | post-stream session "
                f"commit failed (persistence already guaranteed engine-side)"
            )

    # Batch D metrics: record the completed streaming request (bounded labels).
    # Finding 3(b): armed here, fired after the lease is released.
    _record_stream_metrics(
        engine, finish_reason=finish_reason, error_code="none",
        cache_info=final_cache_info, queue_wait=queue_wait,
        t_body_start=_t_body_start, t_first_token=_t_first_token,
        prompt_tokens=final_prompt_tokens,
        completion_tokens=final_completion_tokens,
        sink=metric_sink,
    )

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
            total_tokens=final_prompt_tokens + final_completion_tokens,
            cache_info=final_cache_info,
        ),
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def _make_content_chunk(chunk_id: str, created: int, model: str, text: str) -> str:
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(content=text))],
    )
    return chunk.model_dump_json(exclude_none=True)


def _make_reasoning_chunk(chunk_id: str, created: int, model: str, text: str) -> str:
    """Reasoning-channel delta (LM-Studio shape): delta.reasoning_content."""
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(reasoning_content=text))],
    )
    return chunk.model_dump_json(exclude_none=True)


# --- GET /v1/models ---

@router.get("/v1/models")
async def list_models():
    return ModelListResponse(
        data=[
            ModelInfo(
                id=engine.model_id,
                owned_by="mlx-soloheaven",
            )
            for engine in _engines.values()
        ]
    )


# --- Session management ---

class CompactRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/v1/sessions/{session_id}/compact")
async def compact_session(session_id: str, request: CompactRequest):
    """Rebuild KV cache for a session with new (compressed) messages."""
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    # U14: heavy synchronous engine call — keep it off the event loop.
    # F2: mutating RPC -> long-ops executor.
    result = await run_long(
        _default_engine.compact_session, session_id, messages
    )
    return result


@router.get("/v1/sessions")
async def list_sessions():
    """List all active sessions with cache stats.

    U14: engine reads run off the event loop with a bounded wait; a busy
    engine (generation in flight) degrades to a ``{"busy": true}`` marker
    per model instead of hanging the endpoint. F2: on the reserved reads
    executor so the bounded wait actually starts even under long-op load."""
    result: dict = {"sessions": {}, "base_caches": {}}
    for model_id, engine in _engines.items():
        try:
            result["sessions"][model_id] = await run_read(
                engine.list_sessions
            )
            result["base_caches"][model_id] = await run_read(
                engine.base_cache_stats
            )
        except EngineBusyError:
            result["sessions"][model_id] = {"busy": True}
            result["base_caches"][model_id] = {"busy": True}
    return result


@router.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    try:
        # F2: bounded read -> reserved reads executor.
        info = await run_read(_default_engine.get_session, session_id)
    except EngineBusyError:
        # U14: generation in flight — honest 503 instead of a long hang. This
        # is a LOCAL degrade (the endpoint answers rather than propagating to
        # the app handler), so it stays 503 (not the 429 saturation contract).
        # Batch B: envelope the old bare-string body into {message,type,code}.
        return error_response(
            503,
            "engine busy (generation in progress), retry shortly",
            TYPE_ENGINE_BUSY,
            CODE_ENGINE_BUSY,
            retry_after=RETRY_AFTER_ENGINE_NOT_READY,
        )
    if not info:
        # Batch B: envelope the old bare-string body into {message,type,code}.
        return error_response(
            404, "session not found", TYPE_NOT_FOUND, CODE_NOT_FOUND
        )
    return info


@router.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its cache."""
    # U14: mutating op — waits for the engine (full RPC timeout) but off the
    # event loop, so other requests keep flowing while it queues.
    # F2: mutating RPC -> long-ops executor.
    await run_long(_default_engine.delete_session, session_id)
    return {"status": "ok", "session_id": session_id}
