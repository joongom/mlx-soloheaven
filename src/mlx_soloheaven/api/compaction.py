"""
Compaction API

Endpoints for managing context window compaction.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from mlx_soloheaven.storage import database as db
from mlx_soloheaven.engine.compaction import CompactionEngine, CompactionStrategy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Engine registry (set by server.py)
engine = None  # type: ignore


def set_engine(e):
    global engine
    engine = e


def set_engines(engines, default):
    global engine
    engine = default


class CompactionRequest(BaseModel):
    """Request for manual compaction."""
    strategy: Optional[str] = "summarize"
    keep_recent_turns: Optional[int] = 10


class CompactionResult(BaseModel):
    """Result of compaction operation."""
    success: bool
    old_tokens: int
    new_tokens: int
    reduction_percent: float
    strategy: str
    summary_content: Optional[str] = None
    error: Optional[str] = None


@router.post("/sessions/{session_id}/compact")
async def manual_compact(session_id: str, req: CompactionRequest):
    """
    Manually trigger compaction on a session.
    
    Strategies:
    - summarize: Summarize entire conversation
    - summarize_recent: Summarize recent portion
    - memory_extract: Extract important facts
    - key_points: Retain only key points
    """
    # Check if session exists
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    
    try:
        # Get messages
        messages = await db.get_messages(session_id)
        
        # Add system prompt if present
        system_prompt = session.get("system_prompt", "")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Get current token count
        old_tokens = await db.get_session_total_tokens(session_id)
        
        # Resolve strategy
        strategy = CompactionStrategy(req.strategy or "summarize")
        
        # Create compaction engine
        compaction_engine = CompactionEngine(engine)
        
        # Perform compaction
        result = await compaction_engine.compact(
            messages=messages,
            strategy=strategy,
            target_tokens=session.get("context_window_limit", 100000) // 2,  # Target 50% of limit
            keep_recent_turns=req.keep_recent_turns or 10,
        )
        
        # Record compaction
        compaction_record = await db.record_compaction(
            session_id=session_id,
            old_tokens=old_tokens,
            new_tokens=result["new_tokens"],
            reduction_percent=result["reduction_percent"],
            strategy=strategy.value,
            summary_content=result.get("summary"),
        )
        
        # Update session tokens
        await db.update_session_tokens(session_id, result["new_tokens"])
        
        logger.info(
            f"[Compaction] Session {session_id}: {old_tokens} → {result['new_tokens']} "
            f"tokens ({result['reduction_percent']:.1f}% reduction)"
        )
        
        return {
            "success": True,
            "old_tokens": old_tokens,
            "new_tokens": result["new_tokens"],
            "reduction_percent": result["reduction_percent"],
            "strategy": strategy.value,
            "summary_content": result.get("summary"),
            "compaction_id": compaction_record["id"],
        }
        
    except Exception as e:
        logger.error(f"[Compaction] Failed for session {session_id}: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/sessions/{session_id}/compactions")
async def list_compactions(session_id: str, limit: int = Query(50, le=200)):
    """Get compaction history for a session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    compactions = await db.get_compactions(session_id, limit=limit)
    return compactions


@router.get("/sessions/{session_id}/compaction-status")
async def get_compaction_status(session_id: str):
    """
    Get current compaction status for a session.
    
    Returns:
    - current_tokens: Current token count
    - window_limit: Configured limit
    - remaining_tokens: Tokens remaining
    - needs_compaction: Whether compaction is needed
    - utilization_percent: Current utilization
    """
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    current_tokens = await db.get_session_total_tokens(session_id)
    window_limit = session.get("context_window_limit", 100000)
    remaining = max(0, window_limit - current_tokens)
    utilization = (current_tokens / window_limit * 100) if window_limit > 0 else 0
    
    # Trigger compaction at 90% utilization
    needs_compaction = utilization >= 90
    
    return {
        "session_id": session_id,
        "current_tokens": current_tokens,
        "window_limit": window_limit,
        "remaining_tokens": remaining,
        "utilization_percent": round(utilization, 1),
        "needs_compaction": needs_compaction,
        "last_compacted_at": session.get("last_compacted_at"),
    }
