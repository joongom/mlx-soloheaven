
import math

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

def _validate_settings(req: SessionSettings) -> str | None:
    """U20: sampling values persisted here feed generation verbatim on the
    web-chat path (chat.py reads them off the DB session), so bad values
    (repetition_penalty=0 -> divide-by-zero logits, negative temperature)
    must be rejected at this boundary too. Returns an error message or None.
    Ranges mirror the OpenAI endpoint's _validate_sampling_params.

    Round 2 (codex F6): floats must be FINITE first — JSON NaN/Infinity
    literals reach pydantic floats and pass every range comparison below
    (``nan < 0`` is False), then persist into the DB and feed generation."""
    for name, val in (
        ("temperature", req.temperature),
        ("top_p", req.top_p),
        ("min_p", req.min_p),
        ("repetition_penalty", req.repetition_penalty),
    ):
        if val is not None and not math.isfinite(val):
            return f"'{name}': must be a finite number, got {val}"
    if req.temperature is not None and req.temperature < 0:
        return f"'temperature': must be >= 0, got {req.temperature}"
    if req.top_p is not None and not (0 < req.top_p <= 1):
        return f"'top_p': must be in (0, 1], got {req.top_p}"
    if req.min_p is not None and not (0 <= req.min_p <= 1):
        return f"'min_p': must be in [0, 1], got {req.min_p}"
    if req.top_k is not None and req.top_k < 0:
        return f"'top_k': must be >= 0, got {req.top_k}"
    if req.repetition_penalty is not None and req.repetition_penalty <= 0:
        return (
            f"'repetition_penalty': must be > 0 (1.0 disables, sane range "
            f"0.1-2.0), got {req.repetition_penalty}"
        )
    if req.thinking_budget is not None and req.thinking_budget < 0:
        return f"'thinking_budget': must be >= 0, got {req.thinking_budget}"
    if req.max_tokens is not None and req.max_tokens < 1:
        return f"'max_tokens': must be >= 1, got {req.max_tokens}"
    if req.context_window_limit is not None and req.context_window_limit < 1:
        return f"'context_window_limit': must be >= 1, got {req.context_window_limit}"
    return None


@router.patch("/sessions/{session_id}/settings")
async def update_settings(session_id: str, req: SessionSettings):
    error = _validate_settings(req)
    if error:
        raise HTTPException(400, error)
    await db.update_session(session_id, **req.model_dump(exclude_none=True))
    return {"ok": True}
