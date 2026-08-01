"""
Admin API — real-time logs, cache/DB overview, and cache reset.
"""

import asyncio
import ipaddress
import json
import logging
import os
from collections import deque
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_soloheaven.api.errors import error_dict
from mlx_soloheaven.engine.types import EngineBusyError
from mlx_soloheaven.executors import run_long, run_read
from mlx_soloheaven.storage import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin")

# Hostnames (not IP literals) treated as loopback for the admin log gate.
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def _client_is_loopback(host: str | None) -> bool:
    """True iff ``host`` is a loopback address (127.0.0.0/8, ::1) or a known
    local hostname. Batch D log-leak control: the admin LOG endpoints are bound
    to loopback ONLY (there is no auth system — user decision), so tracebacks
    and operational logs are visible to a LOCAL operator alone, never to an
    unauthenticated remote viewer."""
    if not host:
        return False
    if host in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback(request: "Request | None") -> "JSONResponse | None":
    """Return a 403 envelope when the caller is NOT loopback, else None.

    ``request is None`` (direct unit-test invocation, no ASGI scope) is treated
    as local — the gate protects the network boundary, and a direct call has no
    remote peer."""
    if request is None:
        return None
    client = request.client
    host = client.host if client is not None else None
    if _client_is_loopback(host):
        return None
    logger.warning(
        f"[Admin] rejected non-loopback access to {request.url.path} "
        f"from {host!r} — admin log endpoints are localhost-only"
    )
    return JSONResponse(
        status_code=403,
        content=error_dict(
            "admin log endpoints are restricted to localhost",
            "forbidden",
            "forbidden",
        ),
    )

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
async def stream_logs(request: Request = None):
    """SSE endpoint for real-time log streaming.

    Batch D: localhost-gated — logs (incl. tracebacks) are visible only to a
    local operator, never an unauthenticated remote viewer."""
    denied = _require_loopback(request)
    if denied is not None:
        return denied
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
async def recent_logs(limit: int = 200, request: Request = None):
    """Get recent log entries.

    Batch D: localhost-gated (see ``stream_logs``)."""
    denied = _require_loopback(request)
    if denied is not None:
        return denied
    entries = list(log_buffer.buffer)[-limit:]
    return entries


# --- Models overview ---

@router.get("/models")
async def models_overview():
    """List loaded models with their default parameters.

    Includes the process-mode proxy's liveness snapshot (alive / respawning /
    respawn attempts) so a dead or restarting child worker is visible in the
    admin UI. session_stats is an RPC to the child in process mode — guard it
    so this overview still renders while the child is dead/respawning, and
    run it off the event loop (U14: a blocking RPC here would freeze every
    in-flight SSE stream); a busy engine (generation in flight) degrades to
    sessions=None like a dead one."""
    models = []
    for model_id, engine in _engines.items():
        cfg = engine.cfg
        liveness = None
        if hasattr(engine, "liveness"):
            try:
                liveness = engine.liveness()
            except Exception:  # noqa: BLE001
                liveness = None
        try:
            # F2: bounded read -> reserved reads executor.
            stats = await run_read(engine.session_stats)
            sessions = stats.get("active_sessions", 0)
        except Exception:  # noqa: BLE001 — dead/busy child: keep the page alive
            sessions = None
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
            "sessions": sessions,
            # Worker liveness (process mode): alive/respawning/attempts.
            # None for in-process engines (no child worker to die).
            "engine": liveness,
        })
    return {"models": models}


# --- Cache overview ---

@router.get("/cache")
async def cache_overview():
    """Detailed cache overview across all engines.

    Reads each engine's cache state via ``engine.cache_overview()`` so the
    same code path works for in-process engines AND process-mode proxies (the
    proxy RPCs this to the child, the authoritative cache owner).

    U14: the overview call runs off the event loop with a bounded wait — a
    generation in flight degrades that engine's entry to ``{"busy": true}``
    instead of hanging the whole admin page (and every SSE stream with it)."""
    result = {
        "engines": {},
        "disk_files": [],
        "total_memory_gb": 0.0,
        "total_disk_gb": 0.0,
    }

    for model_id, engine in _engines.items():
        try:
            # F2: bounded read -> reserved reads executor.
            ov = await run_read(engine.cache_overview)
        except EngineBusyError:
            result["engines"][model_id] = {"busy": True}
            continue
        result["engines"][model_id] = {
            "model_id": ov.get("model_id"),
            "enable_thinking": ov.get("enable_thinking"),
            "sessions": ov.get("sessions", []),
            "session_count": ov.get("session_count", 0),
            "base_caches": ov.get("base_caches", []),
            "cache_manager": ov.get("cache_manager", {}),
            # Resident KV / MLX process memory vs the configured budget, so the
            # OOM-causing active-session memory is visible (active-session LRU
            # eviction bounds it to memory_budget_gb).
            "memory": ov.get("memory", {}),
        }
        for df in ov.get("disk_files", []):
            result["disk_files"].append({
                "file": df.get("file"),
                "size_mb": df.get("size_mb"),
                "model": model_id,
            })
        result["total_disk_gb"] += ov.get("disk_bytes", 0) / 1e9
        result["total_memory_gb"] += ov.get("memory_bytes", 0) / 1e9

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
    """Clear all KV caches (memory + disk) and DB cache references.

    Delegates to ``engine.clear_caches()`` so the same path works for
    in-process engines AND process-mode proxies (the proxy RPCs the clear to
    the child, the authoritative cache owner)."""
    cleared = {"memory_sessions": 0, "disk_files": 0, "base_caches": 0}

    for model_id, engine in _engines.items():
        # U14: mutating admin op — full wait, but off the event loop.
        # F2: mutating RPC -> long-ops executor.
        # Codex round 11, finding 1 (audit): NO admission reservation here —
        # no DB mutation precedes this run_long, and a saturation rejection
        # mid-loop leaves at worst a PARTIAL multi-engine clear, which is
        # harmless: clear_caches is idempotent (converges to the empty
        # state), so the client's 503-driven retry completes the rest.
        c = await run_long(engine.clear_caches)
        cleared["memory_sessions"] += c.get("memory_sessions", 0)
        cleared["base_caches"] += c.get("base_caches", 0)
        cleared["disk_files"] += c.get("disk_files", 0)

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
