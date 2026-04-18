"""
OpenAI-compatible API endpoints.
POST /v1/chat/completions — with streaming SSE and tool calling
GET  /v1/models
"""

import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from pydantic import BaseModel

from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChunkChoice,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    ResponseMessage,
    ToolCall,
    FunctionCall,
    UsageInfo,
)
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.tool_parser import (
    generate_call_id,
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
    strip_thinking_tags,
    try_extract_tool_name,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Engine registry set by server.py on startup
_engines: dict[str, MLXEngine] = {}
_default_engine: MLXEngine = None  # type: ignore


def set_engines(engines: dict[str, MLXEngine], default: MLXEngine):
    global _engines, _default_engine
    _engines = engines
    _default_engine = default


def _get_engine(model: str) -> MLXEngine:
    """Resolve model name to engine. Tries exact match, then substring match."""
    if model in _engines:
        return _engines[model]
    # Substring match: "qwen3.5-122b" matches "Qwen3.5-122B-A10B-8bit"
    model_lower = model.lower()
    for key, engine in _engines.items():
        if model_lower in key.lower() or model_lower in engine.model_id.lower():
            return engine
    return _default_engine


# --- POST /v1/chat/completions ---

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = _get_engine(request.model)

    # Validate response_format.json_schema early — return 400 on malformed
    # schemas (matches OpenAI's behavior; avoids silent fallback to
    # unconstrained generation).
    if request.response_format and request.response_format.type == "json_schema":
        js = request.response_format.json_schema
        if not js or not js.schema_:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "response_format.json_schema.schema is required when type=json_schema",
                    "type": "invalid_request_error",
                }},
            )
        try:
            from outlines_core.json_schema import build_regex_from_schema
            import json as _json
            build_regex_from_schema(_json.dumps(js.schema_))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": f"Invalid JSON schema in response_format: {e}",
                    "type": "invalid_request_error",
                }},
            )

    # Build message preview for logging
    msg_preview = []
    for m in request.messages[:3]:
        role = m.role
        raw = m.content
        if isinstance(raw, list):
            content = str(raw)[:80]
        else:
            content = (raw or "")[:80].replace('\n', '\\n')
        msg_preview.append(f"{role}:{content!r}")
    preview_str = " | ".join(msg_preview)
    if len(request.messages) > 3:
        preview_str += f" | ...+{len(request.messages)-3} more"
    logger.info(
        f"[Request] user={request.user!r}, model={request.model} -> {engine.model_id}, "
        f"stream={request.stream}, thinking={request.thinking}, "
        f"max_tokens={request.max_tokens or request.max_completion_tokens}, "
        f"messages={len(request.messages)} | {preview_str}"
    )
    if request.stream:
        return StreamingResponse(
            _stream_completion(request, engine),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return _sync_completion(request, engine)


def _sync_completion(request: ChatCompletionRequest, engine: MLXEngine) -> ChatCompletionResponse:
    """Non-streaming completion."""
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages]
    )
    tools = [t.model_dump() for t in request.tools] if request.tools else None

    enable_thinking = request.thinking if request.thinking is not None else engine.cfg.enable_thinking
    # Map OpenAI frequency/presence_penalty to repetition_penalty if not explicitly set
    rep_penalty = request.repetition_penalty
    if rep_penalty is None and (request.frequency_penalty or request.presence_penalty):
        # Approximate: OpenAI penalties are additive [-2,2], repetition_penalty is multiplicative [0.1, 2.0]
        fp = request.frequency_penalty or 0.0
        pp = request.presence_penalty or 0.0
        rep_penalty = 1.0 + (fp + pp) * 0.25  # rough mapping

    response_format = request.response_format
    if response_format and tools:
        logger.warning(
            f"[Structured] response_format={response_format.type} ignored: "
            f"tools are present (OpenAI behavior)."
        )
        response_format = None

    result = engine.complete(
        messages,
        max_tokens=request.max_tokens or request.max_completion_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        min_p=request.min_p,
        top_k=request.top_k,
        repetition_penalty=rep_penalty,
        tools=tools,
        session_id=request.user,
        thinking=enable_thinking,
        thinking_budget=request.thinking_budget,
        response_format=response_format,
    )

    msg = ResponseMessage(content=result.content)
    if result.tool_calls:
        msg.tool_calls = [
            ToolCall(
                id=tc["id"],
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in result.tool_calls
        ]
        msg.content = result.content

    if request.user:
        assistant_msg: dict = {"role": "assistant", "content": result.content or ""}
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in result.tool_calls
            ]
        engine.update_session_messages(request.user, messages + [assistant_msg])

    return ChatCompletionResponse(
        model=request.model,
        choices=[Choice(message=msg, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            cache_info=result.cache_info,
        ),
    )


async def _stream_completion(
    request: ChatCompletionRequest,
    engine: MLXEngine,
) -> AsyncGenerator[str, None]:
    """Streaming SSE completion with tool call detection."""
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages]
    )
    tools = [t.model_dump() for t in request.tools] if request.tools else None
    has_tools = bool(tools)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model

    # First chunk: role
    first_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Determine thinking mode
    enable_thinking = request.thinking if request.thinking is not None else engine.cfg.enable_thinking
    thinking_budget = request.thinking_budget

    # Map OpenAI frequency/presence_penalty to repetition_penalty if not explicitly set
    rep_penalty = request.repetition_penalty
    if rep_penalty is None and (request.frequency_penalty or request.presence_penalty):
        fp = request.frequency_penalty or 0.0
        pp = request.presence_penalty or 0.0
        rep_penalty = 1.0 + (fp + pp) * 0.25

    model_family = engine.model_family

    # Send opening <think> tag if thinking is enabled
    # Skip for Gemma 4 — it uses <|channel>thought...<channel|> natively
    if enable_thinking and model_family != "gemma4":
        think_chunk = _make_content_chunk(chunk_id, created, model, "<think>\n")
        yield f"data: {think_chunk}\n\n"

    # Generate and stream
    accumulated_text = ""
    final_prompt_tokens = 0
    final_completion_tokens = 0
    final_cache_info = None
    token_count = 0  # tracked for disconnect diagnostics
    TOOL_START, TOOL_END = get_tool_markers(model_family)
    holdback = ""

    # === Incremental tool_call emission state ===
    # When a <tool_call> block starts, we buffer per-block text (tc_block) and
    # try to emit the OpenAI first chunk (id + name) as soon as the function
    # name is determinable. The args chunk is emitted when the block closes.
    # Parallel calls are tracked by monotonically increasing tc_index.
    tc_active = False           # inside a tool_call block
    tc_block = ""               # buffered text after TOOL_START (excl. start tag itself)
    tc_name_sent = False        # whether first chunk (name) was emitted
    tc_id: Optional[str] = None
    tc_index = -1
    parsed_tool_calls: list[dict] = []   # completed calls (for session persistence)

    # Structured output (response_format): build constraint but skip if
    # tools are present (tools take priority per OpenAI semantics).
    response_format = request.response_format
    if response_format and has_tools:
        logger.warning(
            f"[Structured] response_format={response_format.type} ignored: "
            f"tools are present (OpenAI behavior)."
        )
        response_format = None

    try:
        async for result in engine.generate_stream_async(
            messages,
            max_tokens=request.max_tokens or request.max_completion_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
            repetition_penalty=rep_penalty,
            session_id=request.user,
            tools=tools,
            thinking=enable_thinking,
            thinking_budget=thinking_budget,
            response_format=response_format,
        ):
            if result.finish_reason is not None:
                final_prompt_tokens = result.prompt_tokens
                final_completion_tokens = result.completion_tokens
                final_cache_info = result.cache_info
                break

            text = result.text
            if not text:
                # Send SSE comment as keepalive during prompt processing
                yield ": keepalive\n\n"
                continue

            accumulated_text += text
            token_count += 1

            if tc_active:
                tc_block += text

                # Try to emit first chunk (id + name) as soon as name is known.
                if not tc_name_sent:
                    name = try_extract_tool_name(tc_block, model_family)
                    if name:
                        first = ChatCompletionChunk(
                            id=chunk_id, created=created, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                "index": tc_index,
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }]))],
                        )
                        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
                        tc_name_sent = True

                # Check for block close.
                if TOOL_END in tc_block:
                    end_idx = tc_block.index(TOOL_END)
                    block_text = TOOL_START + tc_block[:end_idx] + TOOL_END
                    _, calls = parse_tool_calls(block_text, model_family=model_family)
                    if calls:
                        tc = calls[0]
                        # If name chunk wasn't emitted yet (rare: whole block
                        # arrived in one token), emit it now.
                        if not tc_name_sent:
                            first = ChatCompletionChunk(
                                id=chunk_id, created=created, model=model,
                                choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                    "index": tc_index,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": "",
                                    },
                                }]))],
                            )
                            yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
                        args_chunk = ChatCompletionChunk(
                            id=chunk_id, created=created, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                "index": tc_index,
                                "function": {"arguments": tc["function"]["arguments"]},
                            }]))],
                        )
                        yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n"
                        # Use the id we generated at block-start so session
                        # + SSE agree.
                        tc["id"] = tc_id
                        parsed_tool_calls.append(tc)

                    # Reset for next block; any trailing text after TOOL_END
                    # goes back into the holdback/content path.
                    trailing = tc_block[end_idx + len(TOOL_END):]
                    tc_active = False
                    tc_block = ""
                    tc_name_sent = False
                    tc_id = None
                    if trailing:
                        holdback += trailing
                continue

            holdback += text

            if has_tools and TOOL_START.startswith(holdback.lstrip()):
                continue

            if has_tools and TOOL_START in holdback:
                idx = holdback.index(TOOL_START)
                before = holdback[:idx]
                if before:
                    chunk = _make_content_chunk(chunk_id, created, model, before)
                    yield f"data: {chunk}\n\n"
                tc_active = True
                tc_index += 1
                tc_id = generate_call_id()
                tc_name_sent = False
                tc_block = holdback[idx + len(TOOL_START):]
                holdback = ""
                # The just-received text may already contain the full block;
                # fall through by re-triggering on next iteration is fine,
                # but we can also short-circuit by running detection now.
                if tc_block:
                    name = try_extract_tool_name(tc_block, model_family)
                    if name and not tc_name_sent:
                        first = ChatCompletionChunk(
                            id=chunk_id, created=created, model=model,
                            choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                                "index": tc_index,
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }]))],
                        )
                        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
                        tc_name_sent = True
                continue

            if holdback:
                chunk = _make_content_chunk(chunk_id, created, model, holdback)
                yield f"data: {chunk}\n\n"
                holdback = ""
    except (asyncio.CancelledError, GeneratorExit) as exc:
        tail = accumulated_text[-200:].replace('\n', '\\n')
        logger.info(
            f"[Stream] user={request.user!r} | client disconnected "
            f"({type(exc).__name__}) after {token_count} tokens | "
            f"tail={tail!r}"
        )
        raise

    # Flush remaining holdback (only content path; tool_call active is handled below)
    if holdback and not tc_active:
        chunk = _make_content_chunk(chunk_id, created, model, holdback)
        yield f"data: {chunk}\n\n"

    # If generation ended mid-block (no TOOL_END seen), try best-effort parse
    # so we don't silently drop a tool_call the model truncated.
    if tc_active and tc_block:
        block_text = TOOL_START + tc_block
        _, calls = parse_tool_calls(block_text, model_family=model_family)
        if calls:
            tc = calls[0]
            if not tc_name_sent:
                first = ChatCompletionChunk(
                    id=chunk_id, created=created, model=model,
                    choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                        "index": tc_index,
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": "",
                        },
                    }]))],
                )
                yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
            args_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=model,
                choices=[ChunkChoice(delta=DeltaMessage(tool_calls=[{
                    "index": tc_index,
                    "function": {"arguments": tc["function"]["arguments"]},
                }]))],
            )
            yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n"
            tc["id"] = tc_id
            parsed_tool_calls.append(tc)

    finish_reason = "tool_calls" if parsed_tool_calls else "stop"

    # Update session — persist tool_calls in assistant message so next turn's
    # chat template can render {% if m.tool_calls %} block (required for
    # multi-turn tool use with stateful clients like OpenClaw).
    if request.user:
        thinking, content = split_thinking_and_content(
            accumulated_text, model_family=model_family
        )
        assistant_msg: dict = {"role": "assistant", "content": content or ""}
        if parsed_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in parsed_tool_calls
            ]
        engine.update_session_messages(request.user, messages + [assistant_msg])

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
            total_tokens=final_prompt_tokens + final_completion_tokens,
            cache_info=final_cache_info,
        ),
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def _make_content_chunk(chunk_id: str, created: int, model: str, text: str) -> str:
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(content=text))],
    )
    return chunk.model_dump_json(exclude_none=True)


# --- GET /v1/models ---

@router.get("/v1/models")
async def list_models():
    return ModelListResponse(
        data=[
            ModelInfo(
                id=engine.model_id,
                owned_by="mlx-soloheaven",
            )
            for engine in _engines.values()
        ]
    )


# --- Session management ---

class CompactRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/v1/sessions/{session_id}/compact")
async def compact_session(session_id: str, request: CompactRequest):
    """Rebuild KV cache for a session with new (compressed) messages."""
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    result = _default_engine.compact_session(session_id, messages)
    return result


@router.get("/v1/sessions")
async def list_sessions():
    """List all active sessions with cache stats."""
    return {
        "sessions": {
            model_id: engine.list_sessions()
            for model_id, engine in _engines.items()
        },
        "base_caches": {
            model_id: engine.base_cache_stats()
            for model_id, engine in _engines.items()
        },
    }


@router.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    info = _default_engine.get_session(session_id)
    if not info:
        return JSONResponse(status_code=404, content={"error": "session not found"})
    return info


@router.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its cache."""
    _default_engine.delete_session(session_id)
    return {"status": "ok", "session_id": session_id}
