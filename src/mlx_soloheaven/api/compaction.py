"""
Compaction API — summarize conversation history.

After compaction:
- Old messages are PRESERVED in DB (for research/history)
- A compaction summary message is INSERTED at the boundary
- Chat API sends only messages from the last compaction point onwards
- Engine cache is rebuilt with the compacted message set
"""

import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from mlx_soloheaven.storage import database as db
from mlx_soloheaven.engine.compaction import CompactionEngine, COMPACTION_SUMMARY_PREFIX, SUMMARIZATION_PROMPT
from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.types import EngineBusyError
from mlx_soloheaven.executors import reserve_long_slot, run_long
from mlx_soloheaven import inference_queue
from mlx_soloheaven.api.gate_stream import (
    SlotStreamingResponse,
    closed_stream_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

engine = None  # type: ignore


def set_engine(e):
    global engine
    engine = e


def set_engines(engines, default):
    global engine
    engine = default


class CompactionRequest(BaseModel):
    keep_recent_turns: Optional[int] = 3
    custom_prompt: Optional[str] = None


@router.post("/sessions/{session_id}/compact")
async def compact_session(
    session_id: str, req: CompactionRequest, request: Request = None
):
    """Compact conversation history via SSE streaming.

    Finding 5: the standalone compaction endpoint generates a summary via the
    engine, so it MUST go through the same FIFO inference gate as the completion
    handlers — otherwise a compaction runs CONCURRENTLY with a gated generation
    (the "FIFO fronting all generation" invariant breaks, /ready under-reports,
    and a standalone compaction can overtake queued requests). It acquires its
    OWN lease here (the engine lock still serializes the GPU underneath); the
    lease is owned by SlotStreamingResponse (findings 3+4). AUTO-compaction from
    within a gated chat request does NOT come here — it runs under the chat's
    existing lease and reaches the engine directly (see chat._chat_after_admission).
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if not engine:
        raise HTTPException(500, "Engine not initialized")

    # Fail FAST while the process-mode child worker is dead/respawning:
    # raising HERE (before StreamingResponse is returned) lets the server's
    # EngineRestartingError handler answer a real HTTP 503 instead of a 200
    # with a dead stream. In-process engines have no ensure_available.
    ensure_available = getattr(engine, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    # Batch C gate: acquire the single generation slot BEFORE returning the SSE
    # response (queue-full -> real 429, shutdown -> 503, both raised by the gate
    # and mapped by the app-level handler). Finding 2: race the acquire against
    # a queued-disconnect watcher.
    gate = inference_queue.get_inference_gate()
    receive = request.receive if request is not None else None
    try:
        lease = await inference_queue.acquire_or_disconnect(gate, receive)
    except inference_queue.ClientDisconnected:
        logger.info(
            f"[Compaction] session={session_id} | client disconnected while "
            f"queued on the inference gate — not compacting"
        )
        return closed_stream_response()

    return SlotStreamingResponse(
        gate,
        lease,
        _stream_compact(session_id, session, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _stream_compact(
    session_id: str, session: dict, req: CompactionRequest
) -> AsyncGenerator[str, None]:
    """SSE stream for compaction.

    Thin wrapper over ``_stream_compact_body``: once the SSE response
    started, a 503 can no longer be sent — if the process-mode child worker
    dies mid-stream (EngineRestartingError from the summary generation or
    the compact_session RPC), emit an in-band error event then terminate,
    mirroring the openai_compat/chat stream wrappers.

    Finding 1(a): the inner body generator is closed in a ``finally`` so an
    ``aclose()`` of this wrapper (client disconnect) CASCADES GeneratorExit into
    it — ``async for`` alone does NOT close a nested async generator, so without
    this the summary-generation stream's C1 teardown would run only later."""
    body = _stream_compact_body(session_id, session, req)
    try:
        async for chunk in body:
            yield chunk
    except EngineRestartingError as exc:
        logger.error(
            f"[Compaction] session={session_id} | engine unavailable "
            f"mid-stream: {exc}"
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
        # Cascade the close down to the summary-generation stream (C1 teardown).
        await body.aclose()


async def _stream_compact_body(
    session_id: str, session: dict, req: CompactionRequest
) -> AsyncGenerator[str, None]:
    """SSE stream for compaction: streams summary generation, then finalizes."""

    # Build full message list
    messages = []
    system_prompt = session.get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    db_messages = await db.get_messages(session_id)
    for msg in db_messages:
        messages.append(_to_engine_msg(msg))

    keep_recent = (req.keep_recent_turns or 3) * 2

    # Prepare summarization
    compaction_engine = CompactionEngine(engine)
    prep = await compaction_engine.summarize(
        messages=messages,
        keep_recent=keep_recent,
        custom_prompt=req.custom_prompt,
        session_id=session_id,
    )

    if "error" in prep:
        yield f"data: {json.dumps({'type': 'error', 'error': prep['error']})}\n\n"
        return

    summary_messages = prep["messages"]
    kept_from = prep["kept_from"]
    summarized_count = prep["summarized_count"]

    # Stream: start
    yield f"data: {json.dumps({'type': 'start', 'summarizing': summarized_count})}\n\n"

    # Stream: summary generation token-by-token
    summary = ""
    # Finding 1(a): hold the summary stream so its aclose is cascaded on a
    # client disconnect (closing this body from _stream_compact's finally),
    # driving the engine generator's GeneratorExit rescue (C1 teardown).
    summary_stream = compaction_engine.generate_summary_stream(
        summary_messages, session_id=session_id
    )
    try:
        async for event in summary_stream:
            if event["type"] == "text":
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            elif event["type"] == "result":
                summary = event["summary"]
    finally:
        # Exhausted on normal completion (no-op); on a disconnect this aclose
        # does not return until the engine C1 teardown has run.
        await summary_stream.aclose()

    # Finalize: insert compaction message + rebuild cache.
    # ORDERING NOTE: the summary DB insert intentionally precedes the
    # compact_session RPC — build_post_compaction_messages must read the
    # freshly inserted summary from the DB to assemble the message set the
    # engine rebuilds its cache from, so the two cannot be safely reordered.
    # If the engine dies between the insert and the RPC, the summary message
    # persists (the compacted history is already the DB truth and the next
    # chat turn rebuilds the KV cache from it lazily); the wrapper above
    # turns the failed RPC into an in-band error frame.
    #
    # Codex round 11, finding 1 (audit): admission saturation, unlike an
    # engine death, is detected BEFORE the durable insert — reserve the
    # long-pool slot first and degrade to an in-band error frame with the
    # DB untouched (pre-fix the EngineBusyError from run_long escaped the
    # generator mid-stream, after the summary was already persisted). The
    # streamed summary is discarded on a retry; acceptable — saturation is
    # exceptional and "retry shortly" is the uniform busy contract.
    try:
        slot = reserve_long_slot()
    except EngineBusyError as exc:
        logger.warning(
            f"[Compaction] session={session_id} | long pool saturated — "
            f"compaction not applied: {exc}"
        )
        event = json.dumps(
            {
                "type": "error",
                "error": "engine busy — compaction not applied, retry shortly",
                "detail": str(exc),
            },
            ensure_ascii=False,
        )
        yield f"data: {event}\n\n"
        return

    with slot:
        wrapped_summary = CompactionEngine.wrap_summary(summary, keep_recent=keep_recent)
        await db.add_message(session_id, "user", content=wrapped_summary, token_count=0)

        post_compact_msgs = build_post_compaction_messages(
            system_prompt, await db.get_messages(session_id)
        )
        # U14: the rebuild re-prefills the compacted prompt (seconds-scale) —
        # keep it off the event loop so other SSE streams stay live.
        # F2: mutating RPC -> long-ops executor (consumes the reservation).
        rebuild_result = await run_long(
            engine.compact_session, session_id, post_compact_msgs,
            reservation=slot,
        )

    old_tokens = session.get("total_prompt_tokens", 0)
    new_tokens = rebuild_result.get("cached_tokens", 0)
    reduction = ((old_tokens - new_tokens) / old_tokens * 100) if old_tokens > 0 else 0

    await db.record_compaction(
        session_id=session_id,
        old_tokens=old_tokens,
        new_tokens=new_tokens,
        reduction_percent=reduction,
        strategy="summarize",
        summary_content=summary,
    )
    await db.update_session_tokens(session_id, new_tokens)

    logger.info(
        f"[Compaction] session={session_id} | "
        f"summarized {summarized_count} msgs | "
        f"tokens: {old_tokens} -> {new_tokens}"
    )

    # Stream: done
    done_event = {
        "type": "done",
        "success": True,
        "summary": summary,
        "summarized_count": summarized_count,
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "reduction_percent": round(reduction, 1),
    }
    yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"


def build_post_compaction_messages(system_prompt: str, db_messages: list[dict]) -> list[dict]:
    """Build message list for the model, using compaction if available.

    Returns [system_prompt, compaction_summary, recent_messages, new_messages...].
    The compaction block is placed first (after system), followed by
    keep_recent messages from before it, then any messages after it.
    """
    import re

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Find last compaction message
    last_compact_idx = -1
    keep_recent = 0
    for i, msg in enumerate(db_messages):
        content = msg.get("content", "") or ""
        if content.startswith("The conversation history before this point was compacted"):
            last_compact_idx = i
            # Extract keep_recent from <!-- keep_recent:N -->
            m = re.search(r'<!-- keep_recent:(\d+) -->', content)
            if m:
                keep_recent = int(m.group(1))

    if last_compact_idx < 0:
        # No compaction — use all messages
        for msg in db_messages:
            messages.append(_to_engine_msg(msg))
        return messages

    # Compaction found: assemble [compaction] + [recent before it] + [after it]
    compact_msg = db_messages[last_compact_idx]

    # 1. Add compaction summary (strip keep_recent comment for model)
    compact_content = compact_msg.get("content", "")
    compact_content = re.sub(r'\n<!-- keep_recent:\d+ -->', '', compact_content)
    messages.append({"role": "user", "content": compact_content})

    # 2. Add keep_recent messages BEFORE compaction
    recent_start = max(0, last_compact_idx - keep_recent)
    for msg in db_messages[recent_start:last_compact_idx]:
        messages.append(_to_engine_msg(msg))

    # 3. Add messages AFTER compaction (new chats since compaction)
    for msg in db_messages[last_compact_idx + 1:]:
        messages.append(_to_engine_msg(msg))

    return messages


def _to_engine_msg(msg: dict) -> dict:
    """Convert a DB message to engine format."""
    m = {"role": msg["role"]}
    if msg.get("content"):
        m["content"] = msg["content"]
    if msg.get("tool_calls"):
        m["tool_calls"] = msg["tool_calls"]
    if msg.get("tool_call_id"):
        m["tool_call_id"] = msg["tool_call_id"]
    return m


@router.get("/sessions/{session_id}/compactions")
async def list_compactions(session_id: str, limit: int = Query(50, le=200)):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await db.get_compactions(session_id, limit=limit)


@router.get("/sessions/{session_id}/compaction-status")
async def get_compaction_status(session_id: str):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    current_tokens = await db.get_session_total_tokens(session_id)
    window_limit = session.get("context_window_limit", 100000)
    utilization = (current_tokens / window_limit * 100) if window_limit > 0 else 0

    return {
        "session_id": session_id,
        "current_tokens": current_tokens,
        "window_limit": window_limit,
        "remaining_tokens": max(0, window_limit - current_tokens),
        "utilization_percent": round(utilization, 1),
        "needs_compaction": utilization >= 90,
        "last_compacted_at": session.get("last_compacted_at"),
    }


@router.get("/sessions/{session_id}/compaction-prompt")
async def get_compaction_prompt():
    return {"prompt": SUMMARIZATION_PROMPT}
