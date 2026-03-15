"""
Chat session API for the web frontend.
Manages sessions, messages, and provides SSE streaming with stats.
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.tool_parser import split_thinking_and_content
from mlx_soloheaven.storage import database as db
from mlx_soloheaven.api.compaction import build_post_compaction_messages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

engine: MLXEngine = None  # type: ignore
_engines: dict[str, MLXEngine] = {}


def set_engine(e: MLXEngine):
    global engine
    engine = e


def set_engines(engines: dict[str, MLXEngine], default: MLXEngine):
    global engine, _engines
    _engines = engines
    engine = default


def _get_engine(model: str | None) -> MLXEngine:
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
    return await db.list_sessions()


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
    await db.delete_session(session_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    return await db.get_messages(session_id)


# --- Chat endpoint (SSE streaming) ---

@router.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: SendMessageRequest):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Add user message
    await db.add_message(session_id, "user", content=req.content)

    # Build messages from last compaction point (or all if no compaction)
    system_prompt = session.get("system_prompt", "")
    history = await db.get_messages(session_id)
    messages = build_post_compaction_messages(system_prompt, history)

    # Check if compaction is needed
    current_tokens = await db.get_session_total_tokens(session_id)
    window_limit = session.get("context_window_limit", 100000)
    utilization = (current_tokens / window_limit * 100) if window_limit > 0 else 0
    
    # Trigger compaction at 90% utilization
    if utilization >= 90:
        logger.info(f"[Compaction] Session {session_id} at {utilization:.1f}% - triggering auto-compaction")
        try:
            # Perform auto-compaction
            from mlx_soloheaven.engine.compaction import CompactionEngine, CompactionStrategy
            
            compaction_engine = CompactionEngine(_get_engine(req.model))
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

    # Get generation parameters from session
    use_engine = _get_engine(req.model)
    temperature = session.get("temperature", use_engine.cfg.default_temperature)
    top_p = session.get("top_p", use_engine.cfg.default_top_p)
    min_p = session.get("min_p", use_engine.cfg.default_min_p)
    top_k = session.get("top_k", use_engine.cfg.default_top_k)
    repetition_penalty = session.get("repetition_penalty", use_engine.cfg.default_repetition_penalty)
    thinking_budget = session.get("thinking_budget", use_engine.cfg.thinking_budget)
    max_tokens = session.get("max_tokens", use_engine.cfg.default_max_tokens)

    if req.stream:
        return StreamingResponse(
            _stream_chat(
                session_id, messages, use_engine,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _sync_chat(session_id, messages, use_engine)


async def _stream_chat(
    session_id: str,
    messages: list[dict],
    eng: MLXEngine | None = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
) -> AsyncGenerator[str, None]:
    """Stream chat response with real-time stats."""
    eng = eng or engine

    t_start = time.perf_counter()
    t_first_token = None
    accumulated_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    gen_tps = 0.0
    prompt_tps = 0.0
    token_count = 0

    # Cache info for stats
    t_cache_check = time.perf_counter()
    session_state = eng._sessions.get(session_id)
    if not session_state and eng._has_disk_cache(session_id):
        session_state = eng._load_session_from_disk(session_id)
        if session_state:
            eng._sessions[session_id] = session_state
    cache_hit = False
    cache_info = {"type": "none", "detail": "New session"}
    if session_state:
        if eng._messages_match(session_state.messages, messages):
            cache_hit = session_state.cache is not None
            cached_tokens = session_state.total_cache_tokens
            new_msgs = messages[len(session_state.messages):]
            suffix_desc = f"{len(new_msgs)} new message(s)" if new_msgs else "retry"
            from_disk = (
                session_id not in eng._sessions
                or eng._sessions.get(session_id) is not session_state
            )
            source = "disk -> memory" if from_disk else "memory"
            cache_info = {
                "type": "kv_cache_hit" if cache_hit else "kv_cache_rebuild",
                "detail": (
                    f"KV Cache reuse ({source}): {cached_tokens} tokens cached, {suffix_desc}"
                    if cache_hit
                    else f"Rebuilding KV cache for {len(messages)} messages"
                ),
                "cached_tokens": cached_tokens,
                "stored_msgs": len(session_state.messages),
                "source": source,
            }
        else:
            cache_info = {
                "type": "kv_cache_miss",
                "detail": f"Conversation changed, reprocessing {len(messages)} messages",
                "stored_msgs": len(session_state.messages),
                "incoming_msgs": len(messages),
            }
    t_cache_done = time.perf_counter()

    start_event = json.dumps(
        {"type": "start", "cache_hit": cache_hit, "cache_info": cache_info},
        ensure_ascii=False,
    )
    yield f"data: {start_event}\n\n"

    is_queued = eng._lock.locked()
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

    try:
        async for result in eng.generate_stream_async(
            messages,
            session_id=session_id,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
        ):
            if result.status == "generating":
                t_gen_actual = time.perf_counter()
                queue_wait = t_gen_actual - t_gen_start
                continue

            if result.finish_reason is not None:
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                gen_tps = result.generation_tps
                prompt_tps = result.prompt_tps
                # Merge engine cache_info (includes build_time from branch/regenerate)
                if result.cache_info:
                    engine_cache_info = result.cache_info
                break

            if result.text:
                token_count += 1
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                accumulated_text += result.text

                event = json.dumps(
                    {
                        "type": "text",
                        "content": result.text,
                        "tps": round(result.generation_tps, 1) if result.generation_tps else 0,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {event}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        client_disconnected = True
        logger.info(f"[Stream] session={session_id} | client disconnected")

    t_end = time.perf_counter()
    engine_ttft = (t_first_token - (t_gen_actual or t_gen_start)) if t_first_token else 0
    total_time = t_end - t_start

    thinking, content = split_thinking_and_content(accumulated_text)

    # Include build_time from branch/regenerate BUILD if available
    build_time = 0.0
    if engine_cache_info and "build_time" in engine_cache_info:
        build_time = engine_cache_info["build_time"]

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
        await db.add_message(
            session_id,
            "assistant",
            content=content,
            thinking=thinking,
            token_count=completion_tokens,
            stats=stats,
        )

        updated_messages = messages + [{"role": "assistant", "content": content}]
        eng.update_session_messages(session_id, updated_messages)

        # Update session total tokens
        new_total = await db.get_session_total_tokens(session_id)
        await db.update_session_tokens(session_id, new_total)

        if len(messages) <= 2:
            title = messages[-1].get("content", "")[:50]
            if title:
                await db.update_session(session_id, title=title)

    if client_disconnected:
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

    done_event = json.dumps(
        {
            "type": "done",
            "thinking": thinking,
            "content": content,
            "stats": stats,
            "needs_compaction": needs_compaction,
        },
        ensure_ascii=False,
    )
    yield f"data: {done_event}\n\n"


async def _sync_chat(session_id: str, messages: list[dict], eng: MLXEngine | None = None) -> dict:
    """Non-streaming chat response."""
    eng = eng or engine
    result = eng.complete(messages, session_id=session_id)

    await db.add_message(
        session_id,
        "assistant",
        content=result.content,
        thinking=result.thinking,
        token_count=result.completion_tokens,
    )

    updated_messages = messages + [{"role": "assistant", "content": result.content}]
    eng.update_session_messages(session_id, updated_messages)

    # Update session total tokens
    new_total = await db.get_session_total_tokens(session_id)
    await db.update_session_tokens(session_id, new_total)

    return {
        "content": result.content,
        "thinking": result.thinking,
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

    source_messages = await db.get_messages(session_id)
    branch_messages = source_messages[:req.turn]

    title = (source.get("title", "New Chat") + " (branch)")[:50]
    new_session = await db.create_session(
        title=title,
        system_prompt=source.get("system_prompt", ""),
        branched_from=session_id,
        branch_turn=req.turn,
    )
    new_id = new_session["id"]

    for msg in branch_messages:
        await db.add_message(
            new_id, msg["role"],
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
            thinking=msg.get("thinking"),
            token_count=msg.get("token_count", 0),
            stats=msg.get("stats"),
        )

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
    eng = _get_engine(None)
    engine_turn = len(engine_msgs)
    result = eng.branch_from_turn(session_id, new_id, engine_turn, branch_messages=engine_msgs)

    return {
        "session_id": new_id,
        "title": title,
        "cached_tokens": result.get("cached_tokens", 0),
        "method": result.get("method", "none"),
        "messages": len(branch_messages),
    }


@router.post("/sessions/{session_id}/delete-last")
async def delete_last_turn(session_id: str):
    """Delete the last user+assistant pair and restore cache state."""
    messages = await db.get_messages(session_id)
    if len(messages) < 2:
        raise HTTPException(400, "Not enough messages to delete")

    # Delete assistant then user (last pair)
    await db.delete_last_message(session_id)
    await db.delete_last_message(session_id)

    # Build engine messages for remaining
    source = await db.get_session(session_id)
    remaining_db = messages[:-2]
    engine_msgs = []
    system_prompt = source.get("system_prompt", "") if source else ""
    if system_prompt:
        engine_msgs.append({"role": "system", "content": system_prompt})
    for msg in remaining_db:
        m = {"role": msg["role"]}
        if msg.get("content"):
            m["content"] = msg["content"]
        if msg.get("tool_calls"):
            m["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            m["tool_call_id"] = msg["tool_call_id"]
        engine_msgs.append(m)

    # Truncate engine session
    eng = _get_engine(None)
    result = eng.truncate_session(session_id, len(engine_msgs))

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

    # Delete assistant message
    await db.delete_last_message(session_id)

    # Restore engine cache
    eng = _get_engine(None)
    result = eng.prepare_regenerate(session_id)

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
    return {
        **engine.cache_manager.stats(),
        **engine.session_stats(),
    }
