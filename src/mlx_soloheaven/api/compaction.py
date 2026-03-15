"""
Compaction API — summarize conversation history.

After compaction:
- Old messages are PRESERVED in DB (for research/history)
- A compaction summary message is INSERTED at the boundary
- Chat API sends only messages from the last compaction point onwards
- Engine cache is rebuilt with the compacted message set
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
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
    """Compact conversation history into a structured summary.

    Old messages are preserved. A compaction summary is inserted at the boundary.
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if not engine:
        raise HTTPException(500, "Engine not initialized")

    # Build full message list for summarization
    messages = []
    system_prompt = session.get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    db_messages = await db.get_messages(session_id)
    for msg in db_messages:
        m = {"role": msg["role"]}
        if msg.get("content"):
            m["content"] = msg["content"]
        if msg.get("tool_calls"):
            m["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            m["tool_call_id"] = msg["tool_call_id"]
        messages.append(m)

    keep_recent = (req.keep_recent_turns or 3) * 2

    # Generate summary
    compaction_engine = CompactionEngine(engine)
    result = await compaction_engine.summarize(
        messages=messages,
        keep_recent=keep_recent,
        custom_prompt=req.custom_prompt,
    )

    if "error" in result:
        return {"success": False, "error": result["error"]}

    summary = result["summary"]
    kept_from = result["kept_from"]  # index in full messages array
    summarized_count = result["summarized_count"]

    # Insert compaction summary message at the boundary
    # Timestamp: just before the first kept message
    has_system = 1 if system_prompt else 0
    db_kept_from = kept_from - has_system
    if db_kept_from > 0 and db_kept_from < len(db_messages):
        boundary_time = db_messages[db_kept_from]["created_at"] - 0.001
    else:
        boundary_time = None  # use current time

    wrapped_summary = CompactionEngine.wrap_summary(summary)
    compaction_msg = await db.add_message(
        session_id, "user",
        content=wrapped_summary,
        token_count=0,
    )

    # Fix timestamp to be at the boundary
    if boundary_time is not None:
        async with db.get_db() as conn:
            await conn.execute(
                "UPDATE messages SET created_at = ? WHERE id = ?",
                (boundary_time, compaction_msg["id"]),
            )
            await conn.commit()

    # Rebuild engine cache with post-compaction messages
    post_compact_msgs = build_post_compaction_messages(
        system_prompt, await db.get_messages(session_id)
    )
    rebuild_result = engine.compact_session(session_id, post_compact_msgs)

    # Record compaction
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

    return {
        "success": True,
        "summary": summary,
        "summarized_count": summarized_count,
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "reduction_percent": round(reduction, 1),
    }


def build_post_compaction_messages(system_prompt: str, db_messages: list[dict]) -> list[dict]:
    """Build message list starting from the last compaction point.

    Scans for the last compaction summary message and returns
    [system_prompt, compaction_summary, messages_after...].
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Find last compaction message
    last_compact_idx = -1
    for i, msg in enumerate(db_messages):
        content = msg.get("content", "") or ""
        if content.startswith(COMPACTION_SUMMARY_PREFIX.split("\n")[0]):
            last_compact_idx = i

    # If compaction exists, start from it; otherwise use all messages
    start_idx = last_compact_idx if last_compact_idx >= 0 else 0

    for msg in db_messages[start_idx:]:
        m = {"role": msg["role"]}
        if msg.get("content"):
            m["content"] = msg["content"]
        if msg.get("tool_calls"):
            m["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            m["tool_call_id"] = msg["tool_call_id"]
        messages.append(m)

    return messages


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
