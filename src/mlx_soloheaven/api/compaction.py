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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_soloheaven.storage import database as db
from mlx_soloheaven.engine.compaction import CompactionEngine, COMPACTION_SUMMARY_PREFIX, SUMMARIZATION_PROMPT

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
async def compact_session(session_id: str, req: CompactionRequest):
    """Compact conversation history via SSE streaming."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if not engine:
        raise HTTPException(500, "Engine not initialized")

    return StreamingResponse(
        _stream_compact(session_id, session, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _stream_compact(
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
    async for event in compaction_engine.generate_summary_stream(summary_messages, session_id=session_id):
        if event["type"] == "text":
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        elif event["type"] == "result":
            summary = event["summary"]

    # Finalize: insert compaction message + rebuild cache
    wrapped_summary = CompactionEngine.wrap_summary(summary, keep_recent=keep_recent)
    await db.add_message(session_id, "user", content=wrapped_summary, token_count=0)

    post_compact_msgs = build_post_compaction_messages(
        system_prompt, await db.get_messages(session_id)
    )
    rebuild_result = engine.compact_session(session_id, post_compact_msgs)

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
