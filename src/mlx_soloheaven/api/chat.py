"""
Chat session API for the web frontend.
Manages sessions, messages, and provides SSE streaming with stats.
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator, TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.tool_parser import (
    CHANNEL_REASONING,
    ThinkingRouter,
    split_thinking_and_content,
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
        for entry in eng.list_sessions():
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
async def chat(session_id: str, req: SendMessageRequest):
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
    eng: "MLXEngine | None" = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    thinking_budget: int = 8192,
    max_tokens: int = 32768,
) -> AsyncGenerator[str, None]:
    """Stream chat response with real-time stats.

    Thin wrapper over ``_stream_chat_body``: once the SSE response started, a
    503 can no longer be sent — if the process-mode child worker dies
    mid-stream (EngineRestartingError), emit an in-band error event so the
    web client terminates with a clear message instead of a dead stream."""
    try:
        async for chunk in _stream_chat_body(
            session_id, messages, eng,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
        ):
            yield chunk
    except EngineRestartingError as exc:
        logger.error(
            f"[Stream] session={session_id} | engine unavailable mid-stream: {exc}"
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
) -> AsyncGenerator[str, None]:
    """Stream chat response with real-time stats."""
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
    # In process mode the engine is an EngineProcessProxy: the parent holds NO
    # session/KV-cache state (the child owns it), so the in-memory/disk peek
    # below would raise AttributeError on ._sessions / ._has_disk_cache. Web
    # chat is not a Stage-1 process-mode target; skip the preflight (the child
    # still reuses its own cache) and report a neutral cache_info.
    t_cache_check = time.perf_counter()
    in_process_engine = hasattr(eng, "_sessions") and hasattr(eng, "_has_disk_cache")
    session_state = None
    if in_process_engine:
        session_state = eng._sessions.get(session_id)
        if not session_state and eng._has_disk_cache(session_id):
            session_state = eng._load_session_from_disk(session_id)
            if session_state:
                eng._sessions[session_id] = session_state
    cache_hit = False
    cache_info = (
        {"type": "none", "detail": "New session"}
        if in_process_engine
        else {"type": "process", "detail": "Cache managed by generation process"}
    )
    if session_state:
        if eng._messages_match(session_state.messages, messages):
            cache_hit = (
                session_state.cache_state is not None
                and session_state.cache_state.cache is not None
            )
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
    # Router active when thinking is enabled for this model. Pass-through
    # (all content) otherwise, so the non-thinking path is unchanged.
    router = ThinkingRouter(active=enable_thinking, model_family=model_family)
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

    try:
        async for batch in eng.generate_stream_batches_async(
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
                    prompt_tokens = result.prompt_tokens
                    completion_tokens = result.completion_tokens
                    gen_tps = result.generation_tps
                    prompt_tps = result.prompt_tps
                    if result.cache_info:
                        engine_cache_info = result.cache_info
                    finished = True
                    break

                if not (result.text or result.token):
                    # Empty heartbeat from engine during long prefill — forward
                    # as SSE comment to keep client connection alive
                    yield ": keepalive\n\n"
                    continue

                token_count += 1
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
        tail = ("".join(acc_parts))[-200:].replace('\n', '\\n')
        logger.info(
            f"[Stream] session={session_id} | client disconnected "
            f"({type(exc).__name__}) after {token_count} tokens | "
            f"tail={tail!r}"
        )

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

    t_end = time.perf_counter()
    engine_ttft = (t_first_token - (t_gen_actual or t_gen_start)) if t_first_token else 0
    total_time = t_end - t_start

    # PERF: single join at end of loop — replaces O(N^2) accumulation.
    accumulated_text = "".join(acc_parts)
    thinking, content = split_thinking_and_content(
        accumulated_text, model_family=eng.model_family
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


async def _sync_chat(session_id: str, messages: list[dict], eng: "MLXEngine | None" = None) -> dict:
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
    # (engine resolved + availability preflighted at the top, before the DB
    # mutations above)
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
    """Delete the last turn. Removes user+assistant pair, or single compaction message."""
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

    last_content = messages[-1].get("content", "") or ""
    is_compaction = last_content.startswith("The conversation history before this point was compacted")

    if is_compaction:
        # Compaction message: delete just this one
        await db.delete_last_message(session_id)
    else:
        # Normal turn: delete assistant + user pair
        if len(messages) < 2:
            raise HTTPException(400, "Not enough messages to delete")
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

    # Truncate engine session (engine resolved + preflighted at the top)
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

    # Availability preflight BEFORE deleting DB messages: deleting the
    # assistant message and then 503-ing on the engine RPC would strand the
    # session with a dangling user turn (and a retry deletes yet more).
    # Mid-flight death after this preflight remains possible; rollback is
    # out of scope.
    eng = _get_engine(None)
    ensure_available = getattr(eng, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

    # Delete assistant message
    await db.delete_last_message(session_id)

    # Restore engine cache
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
