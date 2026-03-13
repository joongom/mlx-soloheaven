
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from mlx_soloheaven.storage import database as db

router = APIRouter(prefix="/api")

class SessionSettings(BaseModel):
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    thinking_budget: Optional[int] = None
    max_tokens: Optional[int] = None
    context_window_limit: Optional[int] = None

@router.get("/sessions/{session_id}/settings")
async def get_settings(session_id: str):
    s = await db.get_session(session_id)
    if not s: raise HTTPException(404)
    return s

@router.patch("/sessions/{session_id}/settings")
async def update_settings(session_id: str, req: SessionSettings):
    await db.update_session(session_id, **req.model_dump(exclude_none=True))
    return {"ok": True}
