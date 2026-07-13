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
from typing import AsyncGenerator, Optional, TYPE_CHECKING

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
from mlx_soloheaven.engine.process_client import EngineRestartingError
from mlx_soloheaven.engine.tool_parser import (
    CHANNEL_REASONING,
    ThinkingRouter,
    _partial_marker_tail,
    generate_call_id,
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
    strip_thinking_tags,
    try_extract_tool_name,
)

if TYPE_CHECKING:
    # Type-only import: keeps mlx.core/mlx_vlm out of the FastAPI parent
    # process so `--engine-mode process` actually isolates MLX in the child.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Engine registry set by server.py on startup
_engines: dict[str, "MLXEngine"] = {}
_default_engine: "MLXEngine" = None  # type: ignore


def set_engines(engines: dict[str, "MLXEngine"], default: "MLXEngine"):
    global _engines, _default_engine
    _engines = engines
    _default_engine = default


def _get_engine(model: str) -> "MLXEngine":
    """Resolve model name to engine. Tries exact match, then substring match."""
    if model in _engines:
        return _engines[model]
    # Substring match: "qwen3.5-122b" matches "Qwen3.5-122B-A10B-8bit"
    model_lower = model.lower()
    for key, engine in _engines.items():
        if model_lower in key.lower() or model_lower in engine.model_id.lower():
            return engine
    return _default_engine


class _Gemma4ThinkingStripper:
    """Backward-compatible shim over the shared ``ThinkingRouter`` (FIX 3).

    The reasoning-channel feature replaced the old strip-and-drop stripper with
    ``tool_parser.ThinkingRouter``, which ROUTES the gemma4 thought channel to a
    reasoning stream instead of dropping it. This shim keeps the old
    ``feed(text) -> content_str`` / ``flush() -> content_str`` surface (returns
    ONLY the content-channel text, dropping reasoning) so prior callers/tests
    that just want the stripped answer still work. New code should use
    ``ThinkingRouter`` directly to obtain BOTH channels.
    """

    def __init__(self, active: bool):
        self.active = active
        self._router = ThinkingRouter(active=active, model_family="gemma4")

    def feed(self, text: str) -> str:
        if not self.active:
            return text
        return "".join(
            t for ch, t in self._router.feed(text) if ch != CHANNEL_REASONING
        )

    def flush(self) -> str:
        return "".join(
            t for ch, t in self._router.flush() if ch != CHANNEL_REASONING
        )


# --- POST /v1/chat/completions ---

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = _get_engine(request.model)

    # Fail FAST while the process-mode child worker is dead/respawning:
    # raising HERE (before StreamingResponse is returned) lets the server's
    # EngineRestartingError handler answer a real HTTP 503 instead of a 200
    # with a dead stream. In-process engines have no ensure_available.
    ensure_available = getattr(engine, "ensure_available", None)
    if ensure_available is not None:
        ensure_available()

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


def _sync_completion(request: ChatCompletionRequest, engine: "MLXEngine"):
    """Non-streaming completion.

    Returns a ChatCompletionResponse, or a 500 JSONResponse error object
    when the engine terminated the stream fail-closed (U6/F1: 'error' is an
    engine-internal reason, never a valid OpenAI finish_reason)."""
    # FIX 3: pass model_family so Gemma 4 <|channel>thought...<channel|> spans
    # in the INPUT history are actually stripped (the default "chatml" left
    # them — and degenerate trailing reasoning — to replay raw into the prompt).
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages],
        model_family=engine.model_family,
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

    # U6/F1: corruption-terminated generation — return an error response,
    # never a completion (and never tool_calls parsed from truncated text).
    # Nothing is persisted: the engine already skipped its session save and
    # invalidated the cache, so the retry takes an honest MISS.
    if result.finish_reason == "error":
        logger.error(
            f"[Request] user={request.user!r} | generation terminated by "
            f"cache corruption — returning 500 error object"
        )
        return JSONResponse(
            status_code=500,
            content={"error": {
                "message": (
                    "generation terminated: session cache corruption "
                    "detected; partial output is unreliable and was not "
                    "persisted — retry the request"
                ),
                "type": "server_error",
                "code": 500,
            }},
        )

    msg = ResponseMessage(content=result.content)
    # Expose the model's thinking as a SEPARATE reasoning channel (matches LM
    # Studio's reasoning_content). The engine already split it via
    # split_thinking_and_content; content stays the clean answer.
    if result.thinking:
        msg.reasoning_content = result.thinking
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
    engine: "MLXEngine",
) -> AsyncGenerator[str, None]:
    """Streaming SSE completion with tool call detection.

    Thin wrapper over ``_stream_completion_body``: once the SSE response has
    started, a 503 can no longer be sent — if the process-mode child worker
    dies mid-stream (EngineRestartingError), emit a proper in-band error
    frame and a terminating [DONE] so the client sees a clean, explained
    close instead of a silently-dead stream."""
    try:
        async for chunk in _stream_completion_body(request, engine):
            yield chunk
    except EngineRestartingError as exc:
        logger.error(
            f"[Stream] user={request.user!r} | engine unavailable mid-stream: {exc}"
        )
        err = {
            "error": {
                "message": "engine restarting, retry shortly",
                "type": "engine_restarting",
                "code": 503,
                "detail": str(exc),
            }
        }
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


async def _stream_completion_body(
    request: ChatCompletionRequest,
    engine: "MLXEngine",
) -> AsyncGenerator[str, None]:
    """Streaming SSE completion with tool call detection."""
    # FIX 3: pass model_family so Gemma 4 thinking channels (incl. degenerate
    # multi-cycle / trailing reasoning) are stripped from the INPUT history
    # rather than replayed raw into the prompt.
    messages = strip_thinking_tags(
        [m.model_dump(exclude_none=True) for m in request.messages],
        model_family=engine.model_family,
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

    # Reasoning-channel router (shared with the web chat path). Instead of
    # dropping the thought channel, route it: thinking-phase text becomes
    # ``delta.reasoning_content`` (LM-Studio shape), the post-thinking answer
    # becomes ``delta.content``. Active when thinking is enabled (gemma4 detects
    # <|channel>thought...<channel|>; chatml/glm stream begins inside <think>
    # and routes up to </think>). A pass-through when thinking is disabled, so
    # non-thinking output stays byte-identical.
    thinking_router = ThinkingRouter(
        active=enable_thinking, model_family=model_family
    )
    # NOTE: we no longer inject an opening "<think>\n" content chunk for
    # non-gemma4 — reasoning is now routed to reasoning_content instead of being
    # wrapped in <think> tags inside content.

    # Generate and stream
    # PERF: append-to-list + join at consumption points avoids the O(N^2)
    # cost of repeated ``str += text`` across the streaming loop.
    acc_parts: list[str] = []
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

    # COALESCING: consume batches of GenerationResult.
    #  - has_tools False: concatenate the batch's content and emit ONE content
    #    delta chunk per batch (reuse the existing chunk builder).
    #  - has_tools True: feed the batch's concatenated text through the existing
    #    tool-call parser, restructured as a loop so an arbitrary chunk that may
    #    contain TOOL_START...TOOL_END (or partial / multiple blocks / trailing
    #    content) is processed correctly. The emitted JSON for both content and
    #    tool_calls is byte-identical to the prior per-token behavior; only the
    #    BATCHING of the content deltas changes.
    finished = False
    # U6/F1: the engine's terminal reason. "error" (fail-closed cache
    # corruption) diverts to the in-band error envelope below instead of a
    # normal completion frame.
    final_finish_reason: Optional[str] = None
    try:
        async for batch in engine.generate_stream_batches_async(
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
            # Concatenate the batch's content text; capture finish separately.
            chunk_text_parts: list[str] = []
            for result in batch:
                if result.finish_reason is not None:
                    # U6/F1: keep the engine's terminal reason — an "error"
                    # (cache-corruption) terminal must not be dressed up as
                    # a normal stop/tool_calls completion below.
                    final_finish_reason = result.finish_reason
                    final_prompt_tokens = result.prompt_tokens
                    final_completion_tokens = result.completion_tokens
                    final_cache_info = result.cache_info
                    finished = True
                    break
                if result.status == "generating":
                    continue
                if not result.text:
                    continue
                chunk_text_parts.append(result.text)

            chunk_text = "".join(chunk_text_parts)
            if not chunk_text:
                if finished:
                    break
                # Empty batch (keepalive / status-only) during prompt processing
                yield ": keepalive\n\n"
                continue

            # Keep acc_parts RAW (drives session-persistence split + disconnect
            # tail). Route the OUTBOUND text through the shared ThinkingRouter so
            # thinking-phase text is emitted as reasoning_content deltas and the
            # post-thinking answer as content deltas. For non-thinking output the
            # router is a pass-through (all content). Segments are processed IN
            # ORDER so reasoning/content interleaving (multi-cycle) is preserved.
            acc_parts.append(chunk_text)
            token_count += 1
            segments = thinking_router.feed(chunk_text)
            if not segments:
                # Reasoning-only held partial marker / nothing emittable yet.
                if finished:
                    break
                yield ": keepalive\n\n"
                continue

            # Process each routed segment in order. Reasoning segments emit a
            # reasoning_content delta; content segments go through the existing
            # content / tool-call path (the tool state machine consumes content
            # text only — tool XML never appears on the reasoning channel).
            for seg_channel, seg_text in segments:
                if not seg_text:
                    continue
                if seg_channel == CHANNEL_REASONING:
                    yield f"data: {_make_reasoning_chunk(chunk_id, created, model, seg_text)}\n\n"
                    continue

                if not has_tools:
                    # Plain content path — one delta chunk per content segment.
                    chunk = _make_content_chunk(chunk_id, created, model, seg_text)
                    yield f"data: {chunk}\n\n"
                    continue

                # has_tools True — drive the tool-call state machine over this
                # content segment. ``chunk`` is the remaining unconsumed text;
                # loop until empty or no further progress (partial block / start).
                chunk = seg_text
                while chunk:
                    if tc_active:
                        tc_block += chunk
                        chunk = ""

                        # Emit first chunk (id + name) as soon as name is known.
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
                                # If name chunk wasn't emitted yet (whole block
                                # arrived at once), emit it now.
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
                            # is re-processed through the holdback/content path.
                            trailing = tc_block[end_idx + len(TOOL_END):]
                            tc_active = False
                            tc_block = ""
                            tc_name_sent = False
                            tc_id = None
                            if trailing:
                                chunk = trailing
                        # else: block still open, wait for more text (chunk empty).
                        continue

                    holdback += chunk
                    chunk = ""

                    if TOOL_START in holdback:
                        idx = holdback.index(TOOL_START)
                        before = holdback[:idx]
                        if before:
                            content_chunk = _make_content_chunk(
                                chunk_id, created, model, before
                            )
                            yield f"data: {content_chunk}\n\n"
                        tc_active = True
                        tc_index += 1
                        tc_id = generate_call_id()
                        tc_name_sent = False
                        # Re-feed everything after TOOL_START into the active-block
                        # branch on the next loop turn (handles full block in chunk).
                        chunk = holdback[idx + len(TOOL_START):]
                        holdback = ""
                        continue

                    # FIX 3: a partial start marker may trail VISIBLE content
                    # (e.g. "before <too"). Hold back only the LONGEST SUFFIX of
                    # the buffer that is a prefix of TOOL_START, and emit the
                    # preceding text now. The old check only held when the WHOLE
                    # (stripped) buffer was a prefix, so it leaked the partial
                    # marker as content. ``keep`` covers the all-prefix case too
                    # (keep == len(holdback) -> emit nothing, hold everything).
                    keep = _partial_marker_tail(holdback, (TOOL_START,))
                    emit = holdback[: len(holdback) - keep]
                    holdback = holdback[len(holdback) - keep:]
                    if emit:
                        content_chunk = _make_content_chunk(
                            chunk_id, created, model, emit
                        )
                        yield f"data: {content_chunk}\n\n"

            if finished:
                break
    except (asyncio.CancelledError, GeneratorExit) as exc:
        tail = ("".join(acc_parts))[-200:].replace('\n', '\\n')
        logger.info(
            f"[Stream] user={request.user!r} | client disconnected "
            f"({type(exc).__name__}) after {token_count} tokens | "
            f"tail={tail!r}"
        )
        raise

    # U6/F1: corruption-terminated stream. "error" is NOT a valid OpenAI
    # finish_reason, and the already-streamed text is truncated at an
    # arbitrary point — synthesizing stop/tool_calls here (or best-effort
    # emitting a truncated tool call below) would hand the client corrupt
    # partial output as a successful completion. Emit the same in-band error
    # envelope shape the liveness work uses (data:{"error":{...}} then
    # [DONE]) and SUPPRESS tool-call parsing and session persistence of the
    # truncated text. The engine has already invalidated the session cache;
    # the client's retry takes an honest MISS.
    if final_finish_reason == "error":
        logger.error(
            f"[Stream] user={request.user!r} | generation terminated by "
            f"cache corruption after {token_count} tokens — emitting error "
            f"envelope (no tool_calls, nothing persisted)"
        )
        err = {
            "error": {
                "message": (
                    "generation terminated: session cache corruption "
                    "detected mid-stream; partial output is unreliable and "
                    "was not persisted — retry the request"
                ),
                "type": "server_error",
                "code": 500,
            }
        }
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Flush the router's held tail. A partial-opener prefix held back as a
    # possible split marker is real content/reasoning at stream end and must be
    # emitted (a marker that never completed is not a marker). Reasoning that
    # never closed its channel (degenerate) is dropped by the router itself.
    for seg_channel, seg_text in thinking_router.flush():
        if not seg_text:
            continue
        if seg_channel == CHANNEL_REASONING:
            yield f"data: {_make_reasoning_chunk(chunk_id, created, model, seg_text)}\n\n"
        elif not has_tools:
            chunk = _make_content_chunk(chunk_id, created, model, seg_text)
            yield f"data: {chunk}\n\n"
        else:
            # Route content through the post-loop holdback flush below (the held
            # tail cannot be a tool-start fragment — those are held in
            # ``holdback``, not the router — so appending is safe).
            holdback += seg_text

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
        # FIX 1: the chatml/glm stream began inside <think> when thinking was
        # enabled (opener lives in the prompt suffix). Pass that so a degenerate
        # no-</think> turn is persisted as reasoning (content="") — matching what
        # the streaming router already emitted on the wire.
        thinking, content = split_thinking_and_content(
            "".join(acc_parts),
            model_family=model_family,
            started_in_thinking=enable_thinking and model_family != "gemma4",
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


def _make_reasoning_chunk(chunk_id: str, created: int, model: str, text: str) -> str:
    """Reasoning-channel delta (LM-Studio shape): delta.reasoning_content."""
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=DeltaMessage(reasoning_content=text))],
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
