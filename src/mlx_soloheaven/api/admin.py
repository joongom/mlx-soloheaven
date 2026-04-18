"""
Admin API — real-time logs, cache/DB overview, and cache reset.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from mlx_soloheaven.storage import database as db

router = APIRouter(prefix="/api/admin")

# Engine registry — set by server.py
_engines: dict[str, "MLXEngine"] = {}
_default_engine = None


def set_engines(engines: dict, default):
    global _engines, _default_engine
    _engines = engines
    _default_engine = default


# --- Real-time log streaming via SSE ---

class LogBuffer(logging.Handler):
    """Captures log records and broadcasts to SSE subscribers."""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.buffer: deque[dict] = deque(maxlen=maxlen)
        self.subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def emit(self, record: logging.LogRecord):
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        }
        self.buffer.append(entry)
        for q in list(self.subscribers):
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(q.put_nowait, entry)
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self.subscribers:
            self.subscribers.remove(q)


# Global log buffer
log_buffer = LogBuffer()
log_buffer.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))


def install_log_handler():
    """Install the log buffer on root logger to capture everything."""
    root = logging.getLogger()
    log_buffer.setLevel(logging.DEBUG)
    root.addHandler(log_buffer)
    # Set event loop for thread-safe puts
    try:
        loop = asyncio.get_event_loop()
        log_buffer.set_loop(loop)
    except RuntimeError:
        pass


@router.get("/logs/stream")
async def stream_logs():
    """SSE endpoint for real-time log streaming."""
    # Ensure loop is set
    log_buffer.set_loop(asyncio.get_event_loop())

    async def _generate() -> AsyncGenerator[str, None]:
        q = log_buffer.subscribe()
        try:
            # Send recent history first
            for entry in list(log_buffer.buffer)[-100:]:
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            # Stream new logs
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            log_buffer.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/logs/recent")
async def recent_logs(limit: int = 200):
    """Get recent log entries."""
    entries = list(log_buffer.buffer)[-limit:]
    return entries


# --- Models overview ---

@router.get("/models")
async def models_overview():
    """List loaded models with their default parameters."""
    models = []
    for model_id, engine in _engines.items():
        cfg = engine.cfg
        models.append({
            "model_id": engine.model_id,
            "model_path": cfg.model_path,
            "defaults": {
                "temperature": cfg.default_temperature,
                "top_p": cfg.default_top_p,
                "min_p": cfg.default_min_p,
                "top_k": cfg.default_top_k,
                "repetition_penalty": cfg.default_repetition_penalty,
                "max_tokens": cfg.default_max_tokens,
            },
            "thinking": {
                "enabled": cfg.enable_thinking,
                "budget": cfg.thinking_budget,
                "think_end_token": cfg.think_end_token,
                "think_start_token": cfg.think_start_token,
            },
            "cache_budget": {
                "memory_gb": cfg.memory_budget_gb,
                "disk_gb": cfg.disk_budget_gb,
            },
            "sessions": len(engine._sessions),
        })
    return {"models": models}


# --- Cache overview ---

@router.get("/cache")
async def cache_overview():
    """Detailed cache overview across all engines."""
    result = {
        "engines": {},
        "disk_files": [],
        "total_memory_gb": 0.0,
        "total_disk_gb": 0.0,
    }

    for model_id, engine in _engines.items():
        sessions = []
        for sid, s in engine._sessions.items():
            cache = s.cache_state.cache if s.cache_state else None
            cache_size = engine.cache_manager._estimate_cache_size(cache) if cache else 0
            sessions.append({
                "session_id": sid,
                "messages": len(s.messages),
                "cache_tokens": s.total_cache_tokens,
                "cache_size_mb": round(cache_size / 1e6, 1),
                "last_used": s.last_used,
                "age_s": round(time.time() - s.last_used, 0),
            })
        sessions.sort(key=lambda x: x["last_used"], reverse=True)

        base_caches = []
        for h, bc in engine._base_caches.items():
            base_caches.append({
                "hash": h,
                "token_count": bc.token_count,
                "hit_count": bc.hit_count,
                "created": bc.created,
            })

        result["engines"][model_id] = {
            "model_id": engine.model_id,
            "enable_thinking": engine.cfg.enable_thinking,
            "sessions": sessions,
            "session_count": len(sessions),
            "base_caches": base_caches,
            "cache_manager": engine.cache_manager.stats(),
        }

    # Disk files
    for model_id, engine in _engines.items():
        cache_dir = engine.cfg.cache_dir
        if os.path.isdir(cache_dir):
            for fname in sorted(os.listdir(cache_dir)):
                if fname.endswith(".safetensors"):
                    fpath = os.path.join(cache_dir, fname)
                    fsize = os.path.getsize(fpath)
                    result["disk_files"].append({
                        "file": fname,
                        "size_mb": round(fsize / 1e6, 1),
                        "model": model_id,
                    })
                    result["total_disk_gb"] += fsize / 1e9

    # Total memory
    for model_id, engine in _engines.items():
        for sid, s in engine._sessions.items():
            cache = s.cache_state.cache if s.cache_state else None
            if cache:
                result["total_memory_gb"] += engine.cache_manager._estimate_cache_size(cache) / 1e9
    result["total_memory_gb"] = round(result["total_memory_gb"], 2)
    result["total_disk_gb"] = round(result["total_disk_gb"], 2)

    return result


# --- DB overview ---

@router.get("/db")
async def db_overview():
    """Database tables overview."""
    async with db.get_db() as conn:
        # Sessions
        sessions = await conn.execute_fetchall(
            "SELECT s.id, s.title, s.created_at, s.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) as msg_count "
            "FROM sessions s ORDER BY s.updated_at DESC"
        )
        session_list = [dict(r) for r in sessions]

        # Message stats
        msg_stats = await conn.execute_fetchall(
            "SELECT role, COUNT(*) as cnt FROM messages GROUP BY role"
        )
        msg_summary = {r["role"]: r["cnt"] for r in msg_stats}

        # Total counts
        total_sessions = len(session_list)
        total_messages = await conn.execute_fetchall("SELECT COUNT(*) as cnt FROM messages")
        total_memories = await conn.execute_fetchall("SELECT COUNT(*) as cnt FROM memories")

        # DB file size
        db_size = 0
        if db._db_path and os.path.exists(db._db_path):
            db_size = os.path.getsize(db._db_path)

    return {
        "db_path": db._db_path,
        "db_size_mb": round(db_size / 1e6, 2),
        "total_sessions": total_sessions,
        "total_messages": total_messages[0]["cnt"] if total_messages else 0,
        "total_memories": total_memories[0]["cnt"] if total_memories else 0,
        "message_by_role": msg_summary,
        "sessions": session_list,
    }


# --- Cache reset ---

@router.post("/cache/reset")
async def reset_cache():
    """Clear all KV caches (memory + disk) and DB cache references."""
    cleared = {"memory_sessions": 0, "disk_files": 0, "base_caches": 0}

    for model_id, engine in _engines.items():
        # Clear in-memory sessions
        cleared["memory_sessions"] += len(engine._sessions)
        engine._sessions.clear()

        # Clear base caches
        cleared["base_caches"] += len(engine._base_caches)
        engine._base_caches.clear()

        # Clear cache manager
        engine.cache_manager.memory_caches.clear()
        engine.cache_manager.disk_index.clear()

        # Delete disk cache files
        cache_dir = engine.cfg.cache_dir
        if os.path.isdir(cache_dir):
            for fname in os.listdir(cache_dir):
                if fname.endswith(".safetensors"):
                    try:
                        os.remove(os.path.join(cache_dir, fname))
                        cleared["disk_files"] += 1
                    except OSError:
                        pass

        # Clear disk index
        if hasattr(engine, "_disk_session_ids"):
            engine._disk_session_ids.clear()

    return {"status": "ok", "cleared": cleared}


# --- DB reset ---

@router.post("/db/reset")
async def reset_db():
    """Clear all data from DB tables (sessions, messages, memories)."""
    async with db.get_db() as conn:
        await conn.execute("DELETE FROM messages")
        await conn.execute("DELETE FROM sessions")
        await conn.execute("DELETE FROM memories")
        await conn.commit()
    return {"status": "ok"}


# --- Full reset (cache + DB) ---

@router.post("/reset-all")
async def reset_all():
    """Clear everything: KV caches + DB data."""
    cache_result = await reset_cache()
    db_result = await reset_db()
    return {
        "status": "ok",
        "cache": cache_result["cleared"],
        "db": "cleared",
    }
