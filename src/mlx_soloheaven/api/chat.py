"""
Chat session API for the web frontend.
Manages sessions, messages, and provides SSE streaming with stats.
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator, TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.types import EngineBusyError
from mlx_soloheaven.executors import (
    LongReservation,
    reserve_long_slot,
    run_critical,
    run_long,
    run_read,
)
from mlx_soloheaven.engine.tool_parser import (
    CHANNEL_REASONING,
    ThinkingRouter,
    split_thinking_and_content,
    thinking_router_active,
)
from mlx_soloheaven import inference_queue
from mlx_soloheaven.api import metrics
from mlx_soloheaven.api.gate_stream import (
    SlotStreamingResponse,
    closed_stream_response,
)
from mlx_soloheaven.storage import database as db
from mlx_soloheaven.api.compaction import build_post_compaction_messages

if TYPE_CHECKING:
    # Type-only import: avoids pulling mlx.core/mlx_vlm into the FastAPI parent
    # process. In `--engine-mode process`, MLX must only live in the child.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

engine: "MLXEngine" = None  # type: ignore
_engines: dict[str, "MLXEngine"] = {}


def set_engine(e: "MLXEngine"):
    global engine
    engine = e


def set_engines(engines: dict[str, "MLXEngine"], default: "MLXEngine"):
    global engine, _engines
    _engines = engines
    engine = default


def _get_engine(model: str | None) -> "MLXEngine":
    """Resolve model name to engine."""
    if not model or not _engines:
        return engine
    if model in _engines:
        return _engines[model]
    model_lower = model.lower()
    for key, eng in _engines.items():
        if model_lower in key.lower() or model_lower in eng.model_id.lower():
            return eng
    return engine


def _record_chat_metrics(
    eng, *, finish_reason, error_code, cache_info, queue_wait,
    ttft=None, generation_time=None, prompt_tokens=0, completion_tokens=0,
    sink=None,
) -> None:
    """Record the Batch D per-request metric for a web-chat generation.
    Defensive: never raises (metrics must not break a response).

    Finding 3(b): when ``sink`` is a ``DeferredRequestMetric`` the (lock-taking)
    record is ARMED here and FIRED later — after the inference lease is released
    (SlotStreamingResponse.on_release for streaming, the chat endpoint's finally
    for non-streaming) — so metrics recording never sits inside the lease
    window. Without a sink it records immediately."""
    try:
        cache_result, reused = metrics.cache_result_from_info(cache_info)
        kwargs = dict(
            model=getattr(eng, "model_id", None),
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
    except Exception:  # noqa: BLE001 — metrics must never break the response
        logger.debug("metrics: chat record failed (ignored)", exc_info=True)


# --- Request models ---

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    system_prompt: str = ""


class SendMessageRequest(BaseModel):
    content: str
    stream: bool = True
    model: str | None = None


class AddMemoryRequest(BaseModel):
    content: str
    category: str = "general"
    importance: int = 5


class BranchRequest(BaseModel):
    turn: int  # message index to branch from (0-based, inclusive)


# --- Session endpoints ---

@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    return await db.create_session(title=req.title, system_prompt=req.system_prompt)


@router.get("/sessions")
async def list_sessions():
    sessions = await db.list_sessions()
    # Enrich with per-session in-memory drafter stats when present.
    # Engine session IDs match DB session IDs; merge across all engines so
    # multi-model deployments still surface stats for the active engine.
    engine_stats: dict[str, dict] = {}
    for eng in (_engines.values() if _engines else [engine] if engine else []):
        # U14: engine read off the event loop, bounded — a busy engine
        # (generation in flight) just skips the drafter-stats enrichment.
        # F2: reserved reads executor.
        try:
            entries = await run_read(eng.list_sessions)
        except EngineBusyError:
            continue
        except Exception:  # noqa: BLE001 — dead child: keep the page alive
            continue
        for entry in entries:
            ds = entry.get("drafter_stats")
            if ds is not None:
                engine_stats[entry["session_id"]] = ds
    if engine_stats:
        for row in sessions:
            sid = row.get("id") or row.get("session_id")
            if sid in engine_stats:
                row["drafter_stats"] = engine_stats[sid]
    return sessions


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, req: CreateSessionRequest):
    await db.update_session(session_id, title=req.title, system_prompt=req.system_prompt)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    # U26 round 2 (codex F5a): the web delete used to remove ONLY the SQLite
    # rows — the engine kept the session's resident KV cache, its disk cache
    # file, AND its drafter-stats registry entry alive until process restart
    # (the registry is pruned only by engine delete_session/clear_caches).
    # Delete the engine-side session too, on EVERY engine (multi-model: the
    # session lives on whichever engine served it; same merge the
    # list_sessions enrichment uses). Best-effort and BEFORE the DB delete:
    # if the engine call fails (busy/dead child) the DB delete still
    # proceeds — the stale engine state then degrades to an honest MISS on
    # any future request, exactly like the mid-generation invalidation path.
    for eng in (_engines.values() if _engines else [engine] if engine else []):
        try:
            # U14/F2: synchronous mutating engine RPC — off the event loop,
            # long-ops lane (same as the OpenAI-compat delete route).
            await run_long(eng.delete_session, session_id)
        except Exception:  # noqa: BLE001 — the DB delete must still proceed
            logger.exception(
                f"[Session] engine-side delete failed for session="
                f"{session_id} (DB delete proceeds; stale engine state "
                f"degrades to an honest MISS)"
            )
    await db.delete_session(session_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, limit: int | None = None):
    total = await db.get_message_count(session_id)

    if limit:
        # Find messages after last compaction to ensure it's always included
        post_compact = await db.count_messages_after_last_compaction(session_id)
        effective_limit = max(limit, post_compact)
        messages = await db.get_messages(session_id, limit=effective_limit)
    else:
        messages = await db.get_messages(session_id)

    return {
        "messages": messages,
        "total": total,
        "returned": len(messages),
    }


# --- Chat endpoint (SSE streaming) ---

@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: SendMessageRequest, request: Request = None):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Resolve engine + availability preflight BEFORE any session mutation:
    # persisting the user message first and 503-ing afterwards would leave the
    # message in the DB, so the client's retry duplicates it. Raising HERE
    # (before StreamingResponse is returned) lets the server's
    # EngineRestartingError handler answer a real HTTP 503 instead of a 200
    # with a dead stream. In-process engines have no ensure_available.
    # NOTE: a mid-generation death window remains — if the child dies AFTER
    # this preflight but before/while generating, the user message is already
    # persisted and the stream ends with an in-band error frame; full
    # transactional rollback is out of scope.
    use_engine = _get_engine(req.model)
    ensure_available = getattr(use_engine, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    # Batch C: FIFO + bounded inference-queue admission. Acquire the single
    # generation slot BEFORE any DB mutation (user-message persist below), so
    # a saturated queue answers a clean 429 queue_full — and a shutting-down
    # server a 503 — with the DB untouched, exactly like the long-pool
    # reservation. Raised immediately (no await) by the gate and mapped by the
    # app-level EngineBusyError handler; FIFO waiting blocks here. The slot is
    # held for the whole turn: for streaming it is released by the response
    # (SlotStreamingResponse); for non-streaming it is released in this
    # function's finally after the generation completes.
    #
    # Finding 2: a real queued HTTP disconnect does NOT cancel this coroutine,
    # so acquire_or_disconnect RACES the acquire against a disconnect watcher on
    # the ASGI receive channel. Acquiring BEFORE the user-message persist means
    # a client that left while queued never lands a DB row nor enters the
    # engine — the gate ticket is dropped and we return a closed stream.
    gate = inference_queue.get_inference_gate()
    receive = request.receive if request is not None else None
    # Batch D metrics: queue_wait == the gate acquire duration (threaded down to
    # the generation body, which records the per-request metric at completion).
    _t_acquire = time.perf_counter()
    try:
        lease = await inference_queue.acquire_or_disconnect(gate, receive)
    except inference_queue.ClientDisconnected:
        logger.info(
            f"[Chat] session={session_id} | client disconnected while queued "
            f"on the inference gate — not generating, no user row persisted"
        )
        return closed_stream_response()
    queue_wait = time.perf_counter() - _t_acquire
    gate_handed_off = False
    # Finding 3: ARM the deferred metric at generation START (right after the
    # slot is acquired) so EVERY admitted request records exactly one metric even
    # when it dies before the generation's explicit terminal record — a streaming
    # disconnect (which returns before the terminal record) or a non-streaming
    # post-admission exception. The default terminal depends on the mode
    # (streaming most-commonly ends early via a client CANCEL; non-streaming via
    # an ERROR). The generation UPDATEs ttft/tokens/finish/cache/error as data
    # arrives. It is FIRED strictly AFTER the lease release — by
    # SlotStreamingResponse.on_release (streaming) or this endpoint's finally
    # (non-streaming) — so metrics recording never sits inside the lease window.
    metric_sink = metrics.DeferredRequestMetric()
    metric_sink.arm(
        model=getattr(use_engine, "model_id", None),
        queue_wait=queue_wait,
        finish_reason="cancel" if req.stream else "error",
        error_code="cancel" if req.stream else "server_error",
    )
    try:
        # Codex round 11, finding 1: the NON-STREAMING path submits the whole
        # generation through run_long AFTER the user message is persisted, and
        # ensure_available() does NOT reserve executor capacity — a saturated
        # long pool used to answer 503 with the message already in the DB (a
        # retry duplicates it). Reserve the admission slot BEFORE the mutation;
        # busy → clean 503 with the DB untouched. The streaming path bypasses
        # run_long entirely (the engine's own async plumbing), so no
        # reservation there.
        slot: LongReservation | None = None
        if not req.stream:
            slot = reserve_long_slot()
        try:
            response = await _chat_after_admission(
                session_id, req, session, use_engine, slot, gate, lease,
                queue_wait=queue_wait, metric_sink=metric_sink,
            )
        finally:
            # No-op once run_long consumed the reservation; releases the slot
            # when anything between the reserve and the submit threw.
            if slot is not None:
                slot.release()
        if req.stream:
            # SlotStreamingResponse now owns the gate lease and releases it
            # (after closing the inner stream's U13/C1 teardown) when the
            # stream ends — this function must not release it (and it fires the
            # deferred metric via on_release, after that release).
            gate_handed_off = True
        return response
    except EngineRestartingError:
        if not gate_handed_off:
            metric_sink.update(finish_reason="error", error_code="engine_not_ready")
        raise
    except EngineBusyError as exc:
        if not gate_handed_off:
            reason = getattr(exc, "reason", EngineBusyError.REASON_ENGINE_NOT_READY)
            metric_sink.update(
                finish_reason="error",
                error_code=(
                    "queue_full"
                    if reason == EngineBusyError.REASON_QUEUE_FULL
                    else "engine_not_ready"
                ),
            )
        raise
    except Exception:
        # Finding 3: any other post-admission exception on the NON-streaming path
        # (the streaming path already handed the lease off) still emits one metric
        # — error_code=server_error — fired after the lease release below.
        if not gate_handed_off:
            metric_sink.update(finish_reason="error", error_code="server_error")
        raise
    finally:
        if not gate_handed_off:
            gate.release(lease)
            # Finding 3(b): non-streaming path — fire the deferred metric AFTER
            # the lease is released (the arm-at-start baseline guarantees a record
            # even if the generation never reached its own terminal record).
            metric_sink.fire()


async def _chat_after_admission(
    session_id: str,
    req: SendMessageRequest,
    session: dict,
    use_engine: "MLXEngine",
    slot: "LongReservation | None",
    gate: "inference_queue.InferenceGate",
    lease: "inference_queue.Lease",
    queue_wait: float = 0.0,
    metric_sink=None,
):
    """Body of the chat endpoint AFTER the availability preflight, the FIFO
    inference-queue admission (Batch C) and the (non-streaming) long-pool
    reservation — split out so the reservation's try/finally in ``chat``
    covers every DB mutation below (codex round 11, finding 1).

    ``gate``/``lease`` are the already-ACQUIRED inference slot: the streaming
    branch hands the lease to ``SlotStreamingResponse``; the non-streaming
    branch leaves the release to ``chat``'s finally.

    Finding 5 (auto-compaction lease-reuse): the auto-compaction below runs
    INSIDE this held lease and reaches the engine DIRECTLY (never the gated
    standalone compaction endpoint), so it MUST NOT re-acquire the gate — doing
    so would self-deadlock at concurrency-1. Running under the existing lease is
    correct: it is still one admission, still one running slot, still counted in
    /ready. The standalone compaction endpoint (api/compaction.py) is the path
    that acquires its OWN lease."""
    # Add user message. Its row id is threaded through the request (codex
    # round 3, finding 2): the assistant row is persisted only AFTER
    # generation, so a delete-last landing in between removes this row —
    # the assistant insert is made CONDITIONAL on it still existing, or an
    # orphan assistant row (a turn with no originating user message) lands.
    user_row = await db.add_message(session_id, "user", content=req.content)
    user_message_id = user_row["id"]

    # Build messages from last compaction point (or all if no compaction)
    system_prompt = session.get("system_prompt", "")
    history = await db.get_messages(session_id)
    messages = build_post_compaction_messages(system_prompt, history)

    # Check if compaction is needed
    current_tokens = await db.get_session_total_tokens(session_id)
    window_limit = session.get("context_window_limit", 100000)
    utilization = (current_tokens / window_limit * 100) if window_limit > 0 else 0
    
    # Trigger compaction at 90% utilization
    #
    # TODO(pre-existing bug, out of Batch C scope): auto-compaction is
    # currently DEAD. ``CompactionEngine`` (engine/compaction.py) exposes
    # ``summarize`` / ``generate_summary_stream`` but NO ``compact()`` method,
    # so the ``compaction_engine.compact(...)`` call below raises AttributeError
    # every time — swallowed by the broad ``except Exception`` at the end of
    # this block ("[Compaction] Auto-compaction failed: ..."). The turn then
    # proceeds uncompacted. The Batch C invariant this block is SUPPOSED to
    # prove — auto-compaction runs under the chat's ALREADY-HELD gate lease and
    # never re-acquires the gate (which would self-deadlock at concurrency-1) —
    # is still correct BY CONSTRUCTION (no gate call here), but it cannot be
    # observed until compact() exists. See test_inference_queue.py
    # test_finding5_auto_compaction_reuses_lease_no_reacquire, which stubs
    # compact() so the lease-reuse is exercised honestly.
    if utilization >= 90:
        logger.info(f"[Compaction] Session {session_id} at {utilization:.1f}% - triggering auto-compaction")
        try:
            # Perform auto-compaction
            from mlx_soloheaven.engine.compaction import CompactionEngine, CompactionStrategy
            
            compaction_engine = CompactionEngine(use_engine)
            strategy_str = session.get("compaction_strategy", "summarize")
            strategy = CompactionStrategy(strategy_str)
            
            result = await compaction_engine.compact(
                messages=messages,
                strategy=strategy,
                target_tokens=window_limit // 2,  # Target 50% of limit
                keep_recent_turns=10,
            )
            
            # Record compaction
            await db.record_compaction(
                session_id=session_id,
                old_tokens=current_tokens,
                new_tokens=result["new_tokens"],
                reduction_percent=result["reduction_percent"],
                strategy=strategy.value,
                summary_content=result.get("summary"),
            )
            
            # Update session tokens
            await db.update_session_tokens(session_id, result["new_tokens"])
            
            # Rebuild messages with compacted state
            # For now, we'll use the original messages and let the engine handle it
            # In a full implementation, you'd rebuild the message history here
            
            logger.info(
                f"[Compaction] Auto-compaction complete: {current_tokens} → {result['new_tokens']} "
                f"({result['reduction_percent']:.1f}% reduction)"
            )
        except Exception as e:
            logger.error(f"[Compaction] Auto-compaction failed: {e}")
            # Continue with original messages even if compaction fails

    # Get generation parameters from session (engine resolved + availability
    # preflighted above, BEFORE the user message was persisted).
    # Batch-5 F7: the sampling columns exist in the sessions schema now, so
    # ``dict.get(key, default)`` no longer falls back for an uncustomized
    # session — the key is present with a NULL/None value. Coalesce None to
    # the engine config default explicitly (NULL = "not customized").
    def _session_value(key: str, default):
        v = session.get(key)
        return default if v is None else v

    temperature = _session_value("temperature", use_engine.cfg.default_temperature)
    top_p = _session_value("top_p", use_engine.cfg.default_top_p)
    min_p = _session_value("min_p", use_engine.cfg.default_min_p)
    top_k = _session_value("top_k", use_engine.cfg.default_top_k)
    repetition_penalty = _session_value(
        "repetition_penalty", use_engine.cfg.default_repetition_penalty
    )
    thinking_budget = _session_value("thinking_budget", use_engine.cfg.thinking_budget)
    max_tokens = _session_value("max_tokens", use_engine.cfg.default_max_tokens)

    if req.stream:
        # Batch C: the already-acquired gate lease is owned by
        # SlotStreamingResponse, which closes the inner generation stream (full
        # U13/C1 teardown) BEFORE releasing the slot on EVERY exit — normal
        # completion, error, a pre-body disconnect (finding 3), or a mid-send
        # disconnect (finding 4).
        return SlotStreamingResponse(
            gate,
            lease,
            _stream_chat(
                session_id, messages, use_engine,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                user_message_id=user_message_id,
                queue_wait=queue_wait,
                metric_sink=metric_sink,
            ),
            # Finding 3(b): fire the deferred metric AFTER the lease is released.
            on_release=metric_sink.fire if metric_sink is not None else None,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Batch-5 round 4 (codex finding 2): the non-streaming branch must
        # forward the SAME resolved session settings as the streaming branch
        # above — pre-fix it passed only session_id, so every customized
        # sampling setting fell back to the engine config default.
        return await _sync_chat(
            session_id, messages, use_engine, reservation=slot,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            user_message_id=user_message_id,
            queue_wait=queue_wait,
            metric_sink=metric_sink,
        )


async def _persist_assistant_turn(
    session_id: str,
    eng: "MLXEngine",
    *,
    user_message_id: str | None,
    content: str | None,
    thinking: str | None,
    token_count: int,
    stats: dict | None = None,
) -> bool:
    """Persist this turn's assistant row, conditional on its originating
    user row (codex round 3, finding 2).

    /chat inserts the user row BEFORE generation and the assistant row only
    AFTER, so a delete-last landing in between removes the user row; an
    unconditional insert then created an ORPHAN assistant row and the
    completing generation's engine-session install held a turn the DB no
    longer has (the U4/U25 class of DB↔engine divergence).

    Returns True when the row was inserted (normal completion — the caller
    proceeds with the engine commit and token bookkeeping). Returns False
    when the parent row is gone: the DB insert is skipped (WARNING only —
    the stream already delivered the content, so this is not a client
    error) and the ENGINE session is deleted so the next request rebuilds
    honestly from the DB instead of extending a state whose turn no longer
    exists. A failed engine delete is log-only: the stale engine messages
    can no longer prefix-match the shortened DB view, so the next request
    degrades to an honest MISS anyway.

    ``user_message_id=None`` (legacy/direct callers) keeps the historical
    unconditional insert."""
    if user_message_id is None:
        await db.add_message(
            session_id, "assistant",
            content=content, thinking=thinking,
            token_count=token_count, stats=stats,
        )
        return True
    inserted = await db.add_message_if_parent_exists(
        session_id, "assistant",
        parent_id=user_message_id,
        content=content, thinking=thinking,
        token_count=token_count, stats=stats,
    )
    if inserted is not None:
        return True
    logger.warning(
        f"[Chat] session={session_id} | turn deleted mid-generation "
        f"(user row {user_message_id} gone) — assistant row skipped; "
        f"invalidating the engine session so the next request rebuilds "
        f"from the DB"
    )
    try:
        await run_critical(eng.delete_session, session_id)
    except Exception:  # noqa: BLE001 — the terminal event must still go out
        logger.exception(
            f"[Chat] session={session_id} | engine session invalidation "
            f"after mid-generation delete failed (next request degrades to "
            f"an honest MISS via the message-prefix mismatch)"
        )
    return False


async def _stream_chat(
    session_id: str,
    messages: list[dict],
    eng: "MLXEngine | None" = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
    user_message_id: str | None = None,
    queue_wait: float = 0.0,
    metric_sink=None,
) -> AsyncGenerator[str, None]:
    """Stream chat response with real-time stats.

    Thin wrapper over ``_stream_chat_body``: once the SSE response started, a
    503 can no longer be sent — if the process-mode child worker dies
    mid-stream (EngineRestartingError), emit an in-band error event so the
    web client terminates with a clear message instead of a dead stream.

    Finding 1(a): the inner body generator is closed in a ``finally`` so an
    ``aclose()`` of this wrapper (client disconnect) CASCADES GeneratorExit into
    it — ``async for`` alone does NOT close a nested async generator, so without
    this the engine stream's C1 teardown would run only later (on GC).

    ``metric_sink`` (finding 3b): the deferred per-request metric the body arms
    at its terminal frame; SlotStreamingResponse fires it after lease release."""
    body = _stream_chat_body(
        session_id, messages, eng,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        user_message_id=user_message_id,
        gate_queue_wait=queue_wait,
        metric_sink=metric_sink,
    )
    try:
        async for chunk in body:
            yield chunk
    except EngineRestartingError as exc:
        logger.error(
            f"[Stream] session={session_id} | engine unavailable mid-stream: {exc}"
        )
        # Finding 3: the engine died mid-stream — the body's terminal record was
        # never reached, so UPDATE the early-armed (cancel/cancel) metric to an
        # error terminal (fired once via on_release after the lease release).
        # Mirrors the OpenAI streaming path (openai_compat._stream_completion).
        if metric_sink is not None:
            metric_sink.update(
                finish_reason="error", error_code="engine_not_ready",
            )
        event = json.dumps(
            {
                "type": "error",
                "error": "engine restarting, retry shortly",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )
        yield f"data: {event}\n\n"
    finally:
        # Cascade the close down to the engine stream so its C1 teardown runs.
        await body.aclose()


async def _stream_chat_body(
    session_id: str,
    messages: list[dict],
    eng: "MLXEngine | None" = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
    user_message_id: str | None = None,
    gate_queue_wait: float = 0.0,
    metric_sink=None,
) -> AsyncGenerator[str, None]:
    """Stream chat response with real-time stats.

    ``gate_queue_wait`` is the Batch-C inference-gate acquire duration (distinct
    from the engine-lock ``queue_wait`` shown in per-turn stats) — recorded as
    the Batch D queue_wait metric.

    ``user_message_id`` is the row id of the user message this turn
    originates from (codex round 3, finding 2): the assistant persist below
    is conditional on that row still existing, so a delete-last that removed
    the turn mid-generation never gains an orphan assistant row."""
    eng = eng or engine

    t_start = time.perf_counter()
    t_first_token = None
    # PERF: append-to-list + join at consumption points avoids the O(N^2)
    # cost of repeated ``str += text`` across the streaming loop.
    acc_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    gen_tps = 0.0
    prompt_tps = 0.0
    token_count = 0

    # Cache info for stats.
    # F5 (codex batch-3 review, round 2): the preflight is an ENGINE method
    # (session_cache_preflight) called on the reserved reads executor, and
    # it is METADATA-ONLY — a short bounded lock over in-memory dict
    # lookups, plus a lock-free plain-IO safetensors-header parse for the
    # disk fallback. It never loads KV caches (mx.load on a non-owning
    # thread violates the VLM thread-ownership rule) and never writes
    # _sessions; the authoritative load happens on the generation thread.
    # The old inline shape read/wrote eng._sessions and did disk loading
    # directly in this async stream: it blocked the event loop and raced an
    # in-process generation (bypassing U14/U15).
    # In process mode the engine is an EngineProcessProxy without the
    # method: the parent holds NO session/KV-cache state (the child owns
    # it) — skip the preflight (the child still reuses its own cache) and
    # report a neutral cache_info. A busy engine degrades like every other
    # busy path (the preflight is informational only — the generation
    # itself re-resolves the cache under its own lock).
    t_cache_check = time.perf_counter()
    cache_hit = False
    preflight = getattr(eng, "session_cache_preflight", None)
    if preflight is None:
        cache_info = {"type": "process", "detail": "Cache managed by generation process"}
    else:
        try:
            pf = await run_read(preflight, session_id, messages)
            cache_hit = bool(pf.get("cache_hit"))
            cache_info = pf.get("cache_info") or {"type": "none", "detail": "New session"}
        except EngineBusyError:
            cache_info = {
                "type": "busy",
                "detail": "Engine busy (generation in flight) — cache state unknown",
            }
        except Exception:  # noqa: BLE001 — preflight is informational only
            logger.exception(
                f"[Stream] session={session_id} | cache preflight failed (ignored)"
            )
            cache_info = {"type": "none", "detail": "Cache preflight unavailable"}
    t_cache_done = time.perf_counter()

    start_event = json.dumps(
        {"type": "start", "cache_hit": cache_hit, "cache_info": cache_info},
        ensure_ascii=False,
    )
    yield f"data: {start_event}\n\n"

    # In-process engine exposes a generation ._lock; the process-mode proxy
    # exposes .is_busy() instead (no parent-side lock). Use whichever exists.
    if hasattr(eng, "_lock"):
        is_queued = eng._lock.locked()
    elif hasattr(eng, "is_busy"):
        is_queued = eng.is_busy()
    else:
        is_queued = False
    if is_queued:
        queued_event = json.dumps(
            {"type": "queued", "message": "Another request is in progress. Please wait..."},
            ensure_ascii=False,
        )
        yield f"data: {queued_event}\n\n"

    t_gen_start = time.perf_counter()
    t_gen_actual = None
    queue_wait = 0.0
    client_disconnected = False
    engine_cache_info = None
    model_family = eng.model_family
    enable_thinking = eng.cfg.enable_thinking

    # COALESCING + reasoning routing: consume batches of GenerationResult.
    # Control results (status / finish_reason / empty-keepalive) keep their exact
    # per-result semantics. Normal content is routed through the shared
    # ThinkingRouter (same logic as the OpenAI-compat path): thinking-phase text
    # is emitted as "reasoning" SSE frames, the answer as "content" SSE frames.
    # The server now owns the thinking-vs-content split — the client no longer
    # parses raw <|channel>/<channel|>/<think> markers. The first content frame
    # after reasoning carries "thinking_done":True. The concatenation of all
    # emitted reasoning+content equals the routed split of the raw output.
    finished = False
    # U6/F1: engine terminal reason; "error" diverts to an error frame and
    # suppresses DB/session persistence of the truncated text.
    final_finish_reason = None
    # Router activation (shared policy with the OpenAI-compat path,
    # thinking_router_active): chatml/glm activate on the thinking flag
    # (pass-through otherwise, so non-thinking output is unchanged); codex
    # round 5, finding 3 — gemma4 is MARKER-ACTIVE regardless of the flag
    # (the model emits its own channel markers; a passthrough streamed
    # thought-span text as content, disagreeing with the batch parse and
    # the session store).
    # Codex round 7, finding 3: enable_thinking threads the effective
    # contract — the gemma4 router stays marker-active (full markers
    # authoritative), while the ambiguous bare ``thought\n`` opener heuristic
    # only fires when thinking is active.
    router = ThinkingRouter(
        active=thinking_router_active(model_family, enable_thinking),
        model_family=model_family,
        enable_thinking=enable_thinking,
    )
    reasoning_seen = False
    thinking_done_signaled = False

    def _content_frame(text: str, gen_tps: float, thinking_done: bool) -> str:
        event = json.dumps(
            {
                "type": "text",
                "content": text,
                "tps": round(gen_tps, 1) if gen_tps else 0,
                **({"thinking_done": True} if thinking_done else {}),
            },
            ensure_ascii=False,
        )
        return f"data: {event}\n\n"

    def _reasoning_frame(text: str, gen_tps: float) -> str:
        event = json.dumps(
            {
                "type": "text",
                "reasoning": text,
                "tps": round(gen_tps, 1) if gen_tps else 0,
            },
            ensure_ascii=False,
        )
        return f"data: {event}\n\n"

    # Finding 1(a): hold the engine stream so its aclose can be cascaded in the
    # finally below — closing THIS body generator (from _stream_chat's finally)
    # must drive the engine generator's GeneratorExit rescue (C1 teardown).
    engine_stream = eng.generate_stream_batches_async(
        messages,
        session_id=session_id,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
    )
    try:
        async for batch in engine_stream:
            # Accumulate this batch's text, route once at batch end into ordered
            # reasoning/content segments, and emit a frame per contiguous-channel
            # run (reasoning frames + content frames).
            batch_text: list[str] = []
            batch_tps = 0.0

            for result in batch:
                if result.status == "generating":
                    t_gen_actual = time.perf_counter()
                    queue_wait = t_gen_actual - t_gen_start
                    continue

                if result.finish_reason is not None:
                    # U6/F1: keep the terminal reason — an "error"
                    # (cache-corruption) terminal must not be persisted /
                    # reported as a normal completion below.
                    final_finish_reason = result.finish_reason
                    prompt_tokens = result.prompt_tokens
                    completion_tokens = result.completion_tokens
                    gen_tps = result.generation_tps
                    prompt_tps = result.prompt_tps
                    if result.cache_info:
                        engine_cache_info = result.cache_info
                    finished = True
                    break

                if not result.token_produced:
                    # Finding 4: a keepalive / no-token frame (empty heartbeat
                    # during long prefill) — forward as an SSE comment to keep
                    # the connection alive. The old ``not (text or token)`` check
                    # misread a REAL empty-detok token whose id is 0 as a
                    # keepalive (``token == 0`` is both the sentinel and a valid
                    # id); ``token_produced`` is the explicit discriminator.
                    yield ": keepalive\n\n"
                    continue

                token_count += 1
                # Finding 4/7: anchor TTFT on the first ACTUAL token frame (incl.
                # empty-detok), never on a keepalive — matches the OpenAI path.
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                acc_parts.append(result.text)
                batch_text.append(result.text)
                batch_tps = result.generation_tps

            # Route the batch's accumulated text into reasoning/content segments
            # and emit frames. Coalesce consecutive same-channel segments.
            segments = router.feed("".join(batch_text)) if batch_text else []
            run_channel: str | None = None
            run_parts: list[str] = []

            def _flush_run():
                nonlocal run_channel, run_parts, reasoning_seen
                nonlocal thinking_done_signaled
                if not run_parts:
                    return None
                text = "".join(run_parts)
                run_parts = []
                if run_channel == CHANNEL_REASONING:
                    reasoning_seen = True
                    return _reasoning_frame(text, batch_tps)
                # content channel: signal thinking_done on the FIRST content
                # frame that follows any reasoning.
                signal = reasoning_seen and not thinking_done_signaled
                if signal:
                    thinking_done_signaled = True
                return _content_frame(text, batch_tps, thinking_done=signal)

            for seg_channel, seg_text in segments:
                if not seg_text:
                    continue
                if run_channel is None:
                    run_channel = seg_channel
                elif seg_channel != run_channel:
                    frame = _flush_run()
                    if frame:
                        yield frame
                    run_channel = seg_channel
                run_parts.append(seg_text)
            frame = _flush_run()
            if frame:
                yield frame

            if finished:
                break
    except (asyncio.CancelledError, GeneratorExit) as exc:
        client_disconnected = True
        # Batch D log-leak scrub: do NOT log the generated-output tail (it is
        # conversation content, broadcast to the admin log SSE). Log only the
        # safe token count + generated char length.
        gen_len = len("".join(acc_parts))
        logger.info(
            f"[Stream] session={session_id} | client disconnected "
            f"({type(exc).__name__}) after {token_count} tokens | "
            f"generated_chars={gen_len} (content redacted)"
        )
    finally:
        # Finding 1(a): drive the engine generator's GeneratorExit rescue (C1
        # commit-or-invalidate) synchronously here. On normal completion the
        # engine stream is already exhausted (no-op); on a disconnect it is
        # still suspended, and this aclose does not return until C1 has run (in
        # BOTH engine modes — see MLXEngine._drain_worker_until_done /
        # EngineProcessProxy._await_child_cancel_ack). The web-chat path SWALLOWS
        # the disconnect above (persisting partial output), so this finally is
        # what actually winds the engine down before the gate lease is released.
        await engine_stream.aclose()

    # Flush the router's held tail (a partial marker that never completed is
    # real reasoning/content). Skip if the client already disconnected.
    if not client_disconnected:
        for seg_channel, seg_text in router.flush():
            if not seg_text:
                continue
            if seg_channel == CHANNEL_REASONING:
                reasoning_seen = True
                yield _reasoning_frame(seg_text, gen_tps)
            else:
                signal = reasoning_seen and not thinking_done_signaled
                if signal:
                    thinking_done_signaled = True
                yield _content_frame(seg_text, gen_tps, thinking_done=signal)

    # U6/F1: corruption-terminated stream — the partial text is unreliable
    # and must NOT be persisted to the DB / session (the engine has already
    # invalidated the session cache and skipped its own save). Tell the
    # client explicitly instead of sending a normal "done".
    if final_finish_reason == "error":
        logger.error(
            f"[Stream] session={session_id} | generation terminated by "
            f"cache corruption after {token_count} tokens — nothing persisted"
        )
        _record_chat_metrics(
            eng, finish_reason="error", error_code="server_error",
            cache_info=engine_cache_info, queue_wait=gate_queue_wait,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            sink=metric_sink,
        )
        if not client_disconnected:
            error_event = json.dumps(
                {
                    "type": "error",
                    "error": (
                        "generation terminated: session cache corruption "
                        "detected mid-stream; partial output was not saved — "
                        "please retry"
                    ),
                },
                ensure_ascii=False,
            )
            yield f"data: {error_event}\n\n"
        return

    t_end = time.perf_counter()
    engine_ttft = (t_first_token - (t_gen_actual or t_gen_start)) if t_first_token else 0
    # Finding 7: the RECORDED metric TTFT is anchored at BODY start (t_start) —
    # first token frame minus body start — measured the SAME way as the OpenAI
    # streaming path and matching the ttft histogram HELP. The per-turn
    # ``engine_ttft`` stat below stays engine-only for the UI (time from the
    # engine-lock grant to first token, excluding the queue/lock waits).
    metric_ttft = (t_first_token - t_start) if t_first_token else None
    total_time = t_end - t_start

    # PERF: single join at end of loop — replaces O(N^2) accumulation.
    accumulated_text = "".join(acc_parts)
    # Codex round 3, finding 4: the persisted thinking/content split must
    # match what the router above actually emitted on the wire. Thinking
    # DISABLED (chatml/glm) → the router was a pass-through, so the whole
    # text is content and a literal </think> in it is a quote, never a
    # boundary. Thinking ENABLED → the stream began inside the thought block
    # (started_in_thinking mirrors the active router, including the
    # degenerate no-</think> turn routing entirely to reasoning). gemma4
    # keeps its marker-driven split (the model emits its own channels).
    # harmony joins gemma4: its channel markers are model-emitted and
    # authoritative regardless of the thinking flag.
    if model_family not in ("gemma4", "harmony") and not enable_thinking:
        thinking, content = None, accumulated_text
    else:
        # Codex round 7, finding 3: thinking_active threads the contract into
        # the gemma4 split so its bare-opener recognition matches the router
        # that streamed the text.
        thinking, content = split_thinking_and_content(
            accumulated_text, model_family=eng.model_family,
            started_in_thinking=(
                enable_thinking
                and model_family not in ("gemma4", "harmony")
            ),
            thinking_active=enable_thinking,
        )

    # Include build_time from branch/regenerate BUILD if available
    build_time = 0.0
    if engine_cache_info and "build_time" in engine_cache_info:
        build_time = engine_cache_info["build_time"]

    # The engine's finish result carries the REAL cache decision. Prefer it for
    # the displayed stats — in process mode the parent preflight can't see the
    # child's cache, so cache_info/cache_hit above are a neutral stub.
    if engine_cache_info and "cache_mode" in engine_cache_info:
        cm = engine_cache_info["cache_mode"]
        cached = engine_cache_info.get("cached_tokens", 0)
        newt = engine_cache_info.get("new_tokens", 0)
        cache_hit = cm in ("hit", "base_hit")
        cache_info = {
            "type": "kv_cache_hit" if cache_hit else ("kv_cache_rebuild" if cm == "retry" else "kv_cache_miss"),
            "detail": (f"KV Cache reuse: {cached} tokens cached, {newt} new"
                       if cache_hit else f"Processing {newt} new tokens (cache {cm})"),
            "cached_tokens": cached,
        }

    stats = {
        "ttft": round(engine_ttft, 2),
        "queue_wait": round(queue_wait, 2),
        "total_time": round(total_time, 2),
        "build_time": build_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "gen_tps": round(gen_tps, 1),
        "prompt_tps": round(prompt_tps, 1),
        "cache_hit": cache_hit,
        "cache_info": cache_info,
    }

    if accumulated_text:
        # Codex round 3, finding 2: conditional persist — the insert commits
        # only if the originating user row still exists (one transaction).
        # A delete-last that removed the turn mid-generation gets NO orphan
        # assistant row, and the engine session (installed by the completing
        # generation with the deleted turn inside) is invalidated so the
        # next request rebuilds honestly from the DB.
        persisted = await _persist_assistant_turn(
            session_id, eng,
            user_message_id=user_message_id,
            content=content,
            thinking=thinking,
            token_count=completion_tokens,
            stats=stats,
        )

        if persisted:
            updated_messages = messages + [{"role": "assistant", "content": content}]
            # U14: synchronous engine RPC — keep it off the event loop.
            # Codex round 5, finding 1: this commit runs BETWEEN the last
            # streamed token and the terminal "done" event — a saturated long
            # pool used to reject it (EngineBusyError escaped mid-stream, no
            # done event, and pre-1a no dirty-mark for the fresh cache). It now
            # rides the guaranteed critical lane, and ANY failure is log-only:
            # the engine already marked the session dirty at install (finding
            # 1a), so the terminal event below must be unconditional.
            try:
                await run_critical(
                    eng.update_session_messages, session_id, updated_messages
                )
            except Exception:  # noqa: BLE001 — terminal event must still go out
                logger.exception(
                    f"[Stream] session={session_id} | post-stream session commit "
                    f"failed (persistence already guaranteed engine-side)"
                )

            # Update session total tokens
            new_total = await db.get_session_total_tokens(session_id)
            await db.update_session_tokens(session_id, new_total)

            if len(messages) <= 2:
                title = messages[-1].get("content", "")[:50]
                if title:
                    await db.update_session(session_id, title=title)

    if client_disconnected:
        # Finding 3: a disconnect returns BEFORE the terminal record below — so
        # enrich the early-armed (cancel) metric here with what streamed. It
        # stays finish_reason/error_code=cancel and fires exactly once via
        # on_release AFTER the lease release.
        if metric_sink is not None:
            metric_sink.update(
                finish_reason="cancel",
                error_code="cancel",
                queue_wait=gate_queue_wait,
                ttft=(t_first_token - t_start) if t_first_token else None,
                generation_time=time.perf_counter() - t_start,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return

    # Check if compaction is needed
    needs_compaction = False
    if accumulated_text:
        session_data = await db.get_session(session_id)
        if session_data:
            current_tokens = await db.get_session_total_tokens(session_id)
            window_limit = session_data.get("context_window_limit", 100000)
            if window_limit > 0 and current_tokens >= window_limit * 0.9:
                needs_compaction = True

    # Batch D metrics: record the completed streaming web-chat request. Finding
    # 7: TTFT = metric_ttft (first token frame - body start, same anchor as the
    # OpenAI path); generation_time = total_time; queue_wait = the inference-gate
    # acquire duration. Finding 3(b): armed here, fired after the lease release.
    _record_chat_metrics(
        eng, finish_reason=final_finish_reason or "stop", error_code="none",
        cache_info=engine_cache_info, queue_wait=gate_queue_wait,
        ttft=metric_ttft, generation_time=total_time,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        sink=metric_sink,
    )

    done_event = json.dumps(
        {
            "type": "done",
            "thinking": thinking,
            "content": content,
            "stats": stats,
            # U7: surface the engine's terminal reason ("length" when
            # max_tokens truncated the answer) instead of implying a clean
            # stop.
            "finish_reason": final_finish_reason or "stop",
            "needs_compaction": needs_compaction,
        },
        ensure_ascii=False,
    )
    yield f"data: {done_event}\n\n"


async def _sync_chat(
    session_id: str,
    messages: list[dict],
    eng: "MLXEngine | None" = None,
    reservation: "LongReservation | None" = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
    user_message_id: str | None = None,
    queue_wait: float = 0.0,
    metric_sink=None,
) -> dict:
    """Non-streaming chat response. ``reservation`` is the admission slot
    the chat endpoint acquired BEFORE persisting the user message (codex
    round 11, finding 1) — consumed by the run_long below.
    ``user_message_id`` gates the assistant persist on the originating user
    row (codex round 3, finding 2 — see ``_persist_assistant_turn``).

    Batch-5 round 4 (codex finding 2): the resolved session sampling
    settings (same names/shape as ``_stream_chat``) are threaded through to
    ``eng.complete`` — pre-fix the call passed only session_id, so a
    PATCHed session setting (e.g. top_k=1) reached the engine as None and
    silently resolved to the config default on the non-streaming path. In
    process mode ``complete`` is a generic RPC, so the kwargs pass through
    to the child identically."""
    eng = eng or engine
    # U14: complete() blocks for the WHOLE generation — off the event loop.
    # F2: non-streaming generation -> bounded long-ops executor.
    # ``reservation`` is consumed by run_long itself, never forwarded to
    # complete().
    _t_gen = time.perf_counter()
    result = await run_long(
        eng.complete, messages,
        session_id=session_id,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        reservation=reservation,
    )
    _gen_time = time.perf_counter() - _t_gen

    # U6/F1: corruption-terminated — return an error, persist nothing.
    if result.finish_reason == "error":
        _record_chat_metrics(
            eng, finish_reason="error", error_code="server_error",
            cache_info=getattr(result, "cache_info", None), queue_wait=queue_wait,
            generation_time=_gen_time,
            sink=metric_sink,
        )
        return {"error": (
            "generation terminated: session cache corruption detected; "
            "partial output was not saved — please retry"
        )}

    # Batch D metrics: record the completed non-streaming web-chat request.
    # Finding 3(b): armed here; fired by the chat endpoint's finally AFTER the
    # lease is released.
    _record_chat_metrics(
        eng, finish_reason=result.finish_reason, error_code="none",
        cache_info=getattr(result, "cache_info", None), queue_wait=queue_wait,
        generation_time=_gen_time,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        sink=metric_sink,
    )

    # Codex round 3, finding 2: conditional persist (see the streaming path
    # — same contract). Parent gone → no orphan row, engine session
    # invalidated; the completed response still goes out to the client.
    persisted = await _persist_assistant_turn(
        session_id, eng,
        user_message_id=user_message_id,
        content=result.content,
        thinking=result.thinking,
        token_count=result.completion_tokens,
    )

    if persisted:
        updated_messages = messages + [{"role": "assistant", "content": result.content}]
        # U14: synchronous engine RPC — keep it off the event loop.
        # Codex round 5, finding 1 (same shape as the streaming path): the
        # generation COMPLETED — failing the whole response over this
        # bookkeeping call would drop finished work, and the engine-side
        # dirty-mark (finding 1a) already guarantees persistence. Critical
        # lane + log-only failure.
        try:
            await run_critical(
                eng.update_session_messages, session_id, updated_messages
            )
        except Exception:  # noqa: BLE001 — the completed response must go out
            logger.exception(
                f"[Chat] session={session_id} | post-completion session commit "
                f"failed (persistence already guaranteed engine-side)"
            )

        # Update session total tokens
        new_total = await db.get_session_total_tokens(session_id)
        await db.update_session_tokens(session_id, new_total)

    return {
        "content": result.content,
        "thinking": result.thinking,
        # U7: surface the engine's terminal reason (e.g. "length").
        "finish_reason": result.finish_reason,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
    }


# --- Branch & Regenerate ---

@router.post("/sessions/{session_id}/branch")
async def branch_session(session_id: str, req: BranchRequest):
    """Branch a new session from a specific turn."""
    source = await db.get_session(session_id)
    if not source:
        raise HTTPException(404, "Session not found")

    # Availability preflight BEFORE any DB mutation: creating the branch
    # session + copying its messages, then failing the engine branch RPC with
    # a 503, would leave a half-built orphan session behind. (A death AFTER
    # this preflight can still orphan the copy — mid-flight death rollback is
    # out of scope; the preflight closes the common already-dead case.)
    eng = _get_engine(None)
    ensure_available = getattr(eng, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    source_messages = await db.get_messages(session_id)
    branch_messages = source_messages[:req.turn]

    # Codex round 11, finding 1: reserve the long-pool admission slot BEFORE
    # the durable session copy — ensure_available() does NOT reserve
    # executor capacity, so a saturated long pool used to reject run_long
    # only AFTER the branch session + messages were created (orphan
    # half-built session, zero engine calls). Busy now answers 503 with the
    # DB untouched; the context manager releases the slot if anything
    # before the submit throws.
    with reserve_long_slot() as slot:
        title = (source.get("title", "New Chat") + " (branch)")[:50]
        new_session = await db.create_session(
            title=title,
            system_prompt=source.get("system_prompt", ""),
            branched_from=session_id,
            branch_turn=req.turn,
        )
        new_id = new_session["id"]

        # Codex round 7, finding 1: remap the persisted turn-ownership links
        # onto the copied rows' NEW ids (a verbatim parent_id would dangle
        # into the source session; an unmapped parent degrades to NULL —
        # the walker's legacy positional behavior).
        branch_id_map: dict = {}
        for msg in branch_messages:
            new_row = await db.add_message(
                new_id, msg["role"],
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                thinking=msg.get("thinking"),
                token_count=msg.get("token_count", 0),
                stats=msg.get("stats"),
                parent_id=branch_id_map.get(msg.get("parent_id")),
            )
            branch_id_map[msg["id"]] = new_row["id"]

        # Build engine messages (with system prompt, same format as chat endpoint)
        engine_msgs = []
        system_prompt = source.get("system_prompt", "")
        if system_prompt:
            engine_msgs.append({"role": "system", "content": system_prompt})
        for msg in branch_messages:
            m = {"role": msg["role"]}
            if msg.get("content"):
                m["content"] = msg["content"]
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            engine_msgs.append(m)

        # Engine branch: checkpoint restore (fast) or build from scratch (slow)
        # (engine resolved + availability preflighted at the top, before the DB
        # mutations above)
        engine_turn = len(engine_msgs)
        # U14: heavy synchronous engine call — keep it off the event loop.
        # F2: mutating RPC -> long-ops executor (consumes the reservation).
        result = await run_long(
            eng.branch_from_turn, session_id, new_id, engine_turn,
            branch_messages=engine_msgs,
            reservation=slot,
        )

    return {
        "session_id": new_id,
        "title": title,
        "cached_tokens": result.get("cached_tokens", 0),
        "method": result.get("method", "none"),
        "messages": len(branch_messages),
    }


_COMPACTION_PREFIX = "The conversation history before this point was compacted"


def _is_compaction_message(msg: dict) -> bool:
    return ((msg.get("content") or "")).startswith(_COMPACTION_PREFIX)


def _delete_last_turn_count(messages: list[dict]) -> int:
    """U25: number of trailing DB messages that make up the last turn.

    Contract: the DB drives — the handler deletes exactly this suffix and
    the engine follows the resulting message list. Cases:
    - last message is a compaction summary → delete just it (1);
    - otherwise the turn spans from the LAST user message to the end
      (user+assistant pair = 2; tool chains like user / assistant(tool_calls)
      / tool / assistant delete as one unit), never crossing a compaction
      summary (it is not part of any turn);
    - no user message in the trailing region → 0 (caller answers 400).
    """
    if _is_compaction_message(messages[-1]):
        return 1
    for i in range(len(messages) - 1, -1, -1):
        if _is_compaction_message(messages[i]):
            return 0
        if messages[i].get("role") == "user":
            return len(messages) - i
    return 0


def _turn_delete_ids_from_anchor(
    messages: list[dict], anchor_id: str, *, compaction: bool,
) -> list[str]:
    """Codex round 5, finding 1 — pure rows→ids walker for the
    transactional delete-last (the turn-walk logic of
    ``_delete_last_turn_count``, re-run against the delete transaction's
    OWN row snapshot). ``anchor_id`` is the first row of the turn the
    handler identified on its pre-snapshot: the turn's user row, or the
    compaction summary row (``compaction=True``).

    - anchor gone (a concurrent delete already removed the turn) → ``[]``:
      nothing is deleted, mirroring the conditional insert's WHERE EXISTS;
    - ``compaction`` → exactly the anchor row (a compaction summary is not
      part of any turn — rows after it are other turns' territory);
    - otherwise the turn is the UNION of:
      (a) OWNERSHIP (codex round 7, finding 1): every row whose persisted
          ``parent_id`` equals the anchor, REGARDLESS of position. The
          positional walk alone cannot survive an INTERVENING user row —
          pre-snapshot [..., uA], a concurrent request appends uB, then
          the completing generation's conditional insert lands aA (uA
          still exists): the walk from uA stopped at the uB boundary,
          deleted ONLY uA, and aA survived as an orphan;
      (b) the POSITIONAL walk (covers legacy rows with NULL parent_id):
          from the anchor THROUGH every following non-user row up to the
          next user message / compaction summary — an assistant row a
          completing generation appended AFTER the handler's pre-snapshot
          belongs to THIS turn and is deleted with it (the round-4 shape
          deleted only the pre-snapshotted ids, leaving that row as an
          orphan); old sessions keep the round-6 delete behavior.
          Codex round 9, finding 1: the walk only CLAIMS rows with no
          persisted owner (parent_id IS NULL). A row carrying a DIFFERENT
          explicit owner belongs to its parent's turn and is skipped —
          after concurrency settles as [uA, uB, aA(parent=uA)], deleting
          the last turn (anchored on uB) must not take A's answer with it
          and leave uA incomplete. Rows owned by THIS anchor are covered
          by the ownership half regardless of position.
      A concurrently appended NEW user turn survives untouched in both
      (the U25 round-2 append-survival contract): it is never anchored
      here and its rows never carry the anchor's parent_id.
    """
    idx = next(
        (i for i, m in enumerate(messages) if m.get("id") == anchor_id),
        None,
    )
    if idx is None:
        return []
    if compaction:
        return [messages[idx]["id"]]
    end = idx + 1
    while end < len(messages):
        row = messages[end]
        if row.get("role") == "user" or _is_compaction_message(row):
            break
        end += 1
    # Positional fallback: the anchor itself plus trailing LEGACY rows only
    # (parent_id IS NULL). Explicitly-owned rows are the ownership union's
    # territory — a different owner means a different turn (round 9, f1).
    out = [messages[idx]["id"]]
    out.extend(
        m["id"] for m in messages[idx + 1 : end]
        if m.get("parent_id") is None
    )
    positional = set(out)
    # Ownership union: rows persisted for THIS turn that landed beyond the
    # positional boundary (snapshot order kept for determinism).
    out.extend(
        m["id"] for m in messages
        if m.get("parent_id") == anchor_id and m["id"] not in positional
    )
    return out


@router.post("/sessions/{session_id}/delete-last")
async def delete_last_turn(session_id: str):
    """Delete the last turn. Removes the whole trailing turn (user message
    through any assistant/tool messages), or a single compaction message.

    U25 contract: the DB drives and the engine follows. The handler deletes
    exactly the last-turn suffix and derives the engine's message view from
    the ACTUAL remaining rows; a compaction delete rebuilds via
    compact_session (the engine session holds the compacted view, which a
    count-slice cannot express).

    U25 round 2 (codex finding 5) — concurrency safety. The round-1 shape
    snapshotted messages once, then deleted 'whichever row is currently
    last' N times (N separate transactions) and derived the remaining view
    from the STALE snapshot: a chat append landing after the snapshot got
    DELETED while part of the old turn survived, and the engine was rebuilt
    from rows that no longer matched the DB.

    Codex round 5, finding 1 — the round-4 id-targeted delete was still a
    SEPARATE transaction from the snapshot, so the /chat conditional
    assistant insert could commit in between: its WHERE EXISTS saw the user
    row still present (insert lands), the delete then removed ONLY the
    snapshotted user row, and the assistant row survived as an ORPHAN the
    drift rebuild fed to the engine. Now snapshot + turn-walk + delete run
    as ONE write transaction (db.delete_last_turn_tx, BEGIN IMMEDIATE):
    - the write lock is held from the start, so the conditional insert
      lands entirely BEFORE the transaction (its row is in the tx snapshot
      and the anchored walk deletes the WHOLE turn including it) or
      entirely AFTER (WHERE EXISTS sees the user row gone → insert
      skipped) — no orphan in either ordering;
    - the walk is re-run INSIDE the transaction, anchored on the turn's
      first row from the handler's pre-snapshot: rows a concurrent writer
      appended to THIS turn are deleted with it, while a concurrently
      appended NEW user turn survives (U25 round-2 contract) and a turn a
      concurrent delete already removed walks to zero ids (no delete);
    - the engine view is built from the transaction's own remaining rows
      (exact under the held lock — no separate re-read can drift);
    - the truncate_session(count) fast path is used ONLY when the tx
      snapshot equals the handler's pre-snapshot (no concurrent write
      happened); any drift falls back to a full compact_session rebuild.
    There is no per-session asyncio lock in this API layer to serialize
    against a concurrent /chat append, and adding one cannot cover external
    writers anyway. The residual race (a write landing AFTER the delete
    transaction) is the pre-existing benign one — the writer's own request
    path hands the engine the full up-to-date view on its next generate
    call, and the conditional insert skips by construction."""
    messages = await db.get_messages(session_id)
    if not messages:
        raise HTTPException(400, "No messages to delete")

    # Availability preflight BEFORE deleting DB messages: mutating the DB and
    # then 503-ing on the engine truncate RPC would desync DB vs engine cache
    # (and a client retry would delete a SECOND turn). Mid-flight death after
    # this preflight remains possible; rollback is out of scope.
    eng = _get_engine(None)
    ensure_available = getattr(eng, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    is_compaction = _is_compaction_message(messages[-1])
    n_delete = _delete_last_turn_count(messages)
    if n_delete <= 0:
        raise HTTPException(400, "Not enough messages to delete")
    # The turn's ANCHOR row (its user message, or the compaction summary):
    # the transactional walk below re-identifies the turn from this row
    # against the transaction's own snapshot.
    anchor_id = messages[-n_delete]["id"]
    pre_snapshot_ids = [m["id"] for m in messages]

    tx_snapshot_ids: list[str] = []

    def _compute_delete_ids(rows: list[dict]) -> list[str]:
        """Pure walker handed to the delete transaction (runs under the
        write lock, against the tx's own row snapshot)."""
        tx_snapshot_ids[:] = [m["id"] for m in rows]
        return _turn_delete_ids_from_anchor(
            rows, anchor_id, compaction=is_compaction,
        )

    # Codex round 11, finding 1: reserve the long-pool admission slot BEFORE
    # deleting DB messages — a saturated pool used to reject run_long only
    # AFTER 1-2 messages were already removed (DB/engine desync; a client
    # retry deletes a SECOND turn). Busy now answers 503 with the DB
    # untouched.
    with reserve_long_slot() as slot:
        # ONE write transaction: snapshot + anchored turn-walk + delete
        # (codex round 5, finding 1 — see the docstring).
        deleted_ids, remaining_db = await db.delete_last_turn_tx(
            session_id, _compute_delete_ids,
        )
        if not deleted_ids:
            # The anchored turn vanished between the pre-check and the
            # transaction (a concurrent delete) — nothing was removed and
            # the DB is untouched.
            raise HTTPException(400, "Not enough messages to delete")

        # DB is the source of truth: the engine's message view is built from
        # the transaction's own remaining rows, through the same
        # compaction-aware assembly the chat endpoint uses.
        source = await db.get_session(session_id)
        system_prompt = source.get("system_prompt", "") if source else ""
        engine_msgs = build_post_compaction_messages(system_prompt, remaining_db)

        # The count-slice fast path is only valid when nothing else wrote
        # between the handler's pre-snapshot and the delete transaction
        # (the tx saw exactly the rows the handler saw). Any drift —
        # concurrent append, concurrent delete, a mid-generation assistant
        # row folded into the deleted turn — rebuilds from the tx view.
        concurrent_drift = tx_snapshot_ids != pre_snapshot_ids

        # Engine follows (engine resolved + preflighted at the top).
        # U14: heavy synchronous engine call — keep it off the event loop.
        # F2: mutating RPC -> long-ops executor (consumes the reservation).
        if is_compaction or concurrent_drift:
            # Compaction boundary changed, or the DB moved under us: the
            # engine session's view cannot be expressed as a count-slice —
            # rebuild from the fresh post-delete rows instead.
            result = await run_long(
                eng.compact_session, session_id, engine_msgs,
                reservation=slot,
            )
        else:
            result = await run_long(
                eng.truncate_session, session_id, len(engine_msgs),
                reservation=slot,
            )

    return {
        "status": "ok",
        "remaining_messages": len(remaining_db),
        **result,
    }


@router.post("/sessions/{session_id}/regenerate")
async def regenerate_session(session_id: str):
    """Remove last assistant+user pair and prepare for regeneration."""
    messages = await db.get_messages(session_id)
    if not messages or messages[-1]["role"] != "assistant":
        raise HTTPException(400, "Nothing to regenerate")

    # Availability preflight BEFORE deleting DB messages: deleting the
    # assistant message and then 503-ing on the engine RPC would strand the
    # session with a dangling user turn (and a retry deletes yet more).
    # Mid-flight death after this preflight remains possible; rollback is
    # out of scope.
    eng = _get_engine(None)
    ensure_available = getattr(eng, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    # Codex round 11, finding 1: reserve the long-pool admission slot BEFORE
    # deleting the assistant message — a saturated pool used to reject
    # run_long only AFTER the delete (codex reproduced: the session was
    # left with a dangling user turn and zero engine calls; a retry deletes
    # yet more). Busy now answers 503 with the DB untouched.
    with reserve_long_slot() as slot:
        # Delete assistant message
        await db.delete_last_message(session_id)

        # Restore engine cache
        # U14: heavy synchronous engine call — keep it off the event loop.
        # F2: mutating RPC -> long-ops executor (consumes the reservation).
        result = await run_long(
            eng.prepare_regenerate, session_id, reservation=slot
        )

        # Delete user message (frontend will re-send it)
        await db.delete_last_message(session_id)

    return {"status": "ok", "remaining_messages": len(messages) - 2, **result}


# --- Memory endpoints ---

@router.post("/memories")
async def add_memory(req: AddMemoryRequest):
    return await db.add_memory(
        content=req.content, category=req.category, importance=req.importance,
    )


@router.get("/memories")
async def get_memories(category: str | None = None):
    return await db.get_memories(category=category)


@router.get("/memories/search")
async def search_memories(q: str):
    return await db.search_memories(q)


# --- Cache stats ---

@router.get("/cache/stats")
async def cache_stats():
    # U14: engine reads off the event loop, bounded — a busy engine
    # (generation in flight) answers {"busy": true} instead of hanging
    # (process mode: cache_manager is the proxy shim whose stats() RPCs the
    # child's cache_overview; in-process: session_stats takes the bounded
    # read lock). F2: reserved reads executor.
    try:
        cm_stats = await run_read(engine.cache_manager.stats)
        sess_stats = await run_read(engine.session_stats)
    except EngineBusyError:
        return {"busy": True}
    return {**cm_stats, **sess_stats}
