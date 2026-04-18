"""
OpenAI-compatible API endpoints.
POST /v1/chat/completions — with streaming SSE and tool calling
GET  /v1/models
"""

import json
import time
import uuid
import logging
from typing import AsyncGenerator

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
    parse_tool_calls,
    split_thinking_and_content,
    strip_thinking_tags,
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
        assistant_msg = {"role": "assistant", "content": result.content or ""}
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
    tool_call_buffer = ""
    in_tool_call = False
    final_prompt_tokens = 0
    final_completion_tokens = 0
    final_cache_info = None
    TOOL_CALL_TAG = "<|tool_call>" if model_family == "gemma4" else "<tool_call>"
    holdback = ""

    # Structured output (response_format): build constraint but skip if
    # tools are present (tools take priority per OpenAI semantics).
    response_format = request.response_format
    if response_format and has_tools:
        logger.warning(
            f"[Structured] response_format={response_format.type} ignored: "
            f"tools are present (OpenAI behavior)."
        )
        response_format = None

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

        if in_tool_call:
            tool_call_buffer += text
            continue

        holdback += text

        if has_tools and TOOL_CALL_TAG.startswith(holdback.lstrip()):
            continue

        if has_tools and TOOL_CALL_TAG in holdback:
            idx = holdback.index(TOOL_CALL_TAG)
            before = holdback[:idx]
            if before:
                chunk = _make_content_chunk(chunk_id, created, model, before)
                yield f"data: {chunk}\n\n"
            in_tool_call = True
            tool_call_buffer = holdback[idx:]
            holdback = ""
            continue

        if holdback:
            chunk = _make_content_chunk(chunk_id, created, model, holdback)
            yield f"data: {chunk}\n\n"
            holdback = ""

    # Flush remaining holdback
    if holdback and not in_tool_call:
        chunk = _make_content_chunk(chunk_id, created, model, holdback)
        yield f"data: {chunk}\n\n"

    # Handle tool calls
    finish_reason = "stop"
    if tool_call_buffer:
        _, tool_calls = parse_tool_calls(
            tool_call_buffer,
            model_family=model_family,
        )
        if tool_calls:
            finish_reason = "tool_calls"
            for i, tc in enumerate(tool_calls):
                tc_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    choices=[
                        ChunkChoice(
                            delta=DeltaMessage(
                                tool_calls=[
                                    {
                                        "index": i,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": "",
                                        },
                                    }
                                ]
                            )
                        )
                    ],
                )
                yield f"data: {tc_chunk.model_dump_json(exclude_none=True)}\n\n"

                args_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    choices=[
                        ChunkChoice(
                            delta=DeltaMessage(
                                tool_calls=[
                                    {
                                        "index": i,
                                        "function": {
                                            "arguments": tc["function"]["arguments"],
                                        },
                                    }
                                ]
                            )
                        )
                    ],
                )
                yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Update session
    if request.user:
        thinking, content = split_thinking_and_content(
            accumulated_text, model_family=model_family
        )
        assistant_msg = {"role": "assistant", "content": content or ""}
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
