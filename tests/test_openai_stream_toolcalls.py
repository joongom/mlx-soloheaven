"""Integration tests for the OpenAI-compat streaming tool_call emission.

Exercises the full _stream_completion path by feeding token-level chunks
through a stub engine and asserting the SSE stream matches OpenAI spec:

- First chunk with role=assistant
- First tool_call chunk with id + name + empty arguments (emitted as soon
  as name is determinable, NOT after the whole block closes)
- Second tool_call chunk with arguments JSON string (same index)
- Final chunk with finish_reason="tool_calls"

Runs across Qwen / GLM / Gemma 4 families with fixtures representative of
what each model template emits.
"""
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import pytest

from mlx_soloheaven.api.openai_compat import _stream_completion
from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    ToolDef,
)


@dataclass
class StubResult:
    """Matches the fields the stream reader touches on a GenerationResult."""
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None


class StubEngine:
    """Minimal engine stub that replays a canned token stream."""

    def __init__(self, model_family: str, token_stream: list[str]):
        self.model_family = model_family
        self._stream = token_stream

        class _Cfg:
            enable_thinking = False
        self.cfg = _Cfg()

    def _iter_results(self):
        for tok in self._stream:
            yield StubResult(text=tok)
        yield StubResult(
            text="",
            finish_reason="tool_calls",
            prompt_tokens=10,
            completion_tokens=len(self._stream),
            prompt_tps=100.0,
            generation_tps=20.0,
        )

    async def generate_stream_async(self, *args, **kwargs) -> AsyncGenerator:
        for result in self._iter_results():
            yield result

    async def generate_stream_batches_async(self, *args, **kwargs) -> AsyncGenerator:
        # Singleton batches preserve scalar ordering — exactly what the
        # tool-call byte-identity assertions in this module expect.
        for result in self._iter_results():
            yield [result]

    def update_session_messages(self, *args, **kwargs):
        pass


def _build_request(model_family: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=f"test-{model_family}",
        messages=[ChatMessage(role="user", content="search please")],
        tools=[ToolDef(function=FunctionDef(name="web_search", parameters={}))],
        stream=True,
        thinking=False,
    )


async def _collect(engine: StubEngine, request: ChatCompletionRequest) -> list[dict]:
    """Run the stream and return parsed SSE JSON events (excluding [DONE])."""
    events = []
    async for line in _stream_completion(request, engine):
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return events


def _extract_tool_call_chunks(events: list[dict]) -> list[dict]:
    out = []
    for ev in events:
        for choice in ev.get("choices", []):
            tcs = choice.get("delta", {}).get("tool_calls")
            if tcs:
                out.append(tcs[0])
    return out


def _final_finish_reason(events: list[dict]) -> Optional[str]:
    for ev in reversed(events):
        for choice in ev.get("choices", []):
            if choice.get("finish_reason"):
                return choice["finish_reason"]
    return None


# ---------- Qwen ----------

QWEN_TOKENS = [
    "Looking", " it ", "up.\n",
    "<tool_call>",
    "<function=web_search>",
    "<parameter=query>",
    "apple silicon",
    "</parameter>",
    "</function>",
    "</tool_call>",
]


@pytest.mark.asyncio
async def test_qwen_incremental_stream():
    engine = StubEngine("chatml", QWEN_TOKENS)
    req = _build_request("chatml")
    events = await _collect(engine, req)

    tc_chunks = _extract_tool_call_chunks(events)
    # Expect exactly 2 tool_call chunks: (id+name+empty args), (args delta)
    assert len(tc_chunks) == 2

    first, second = tc_chunks
    assert first["index"] == 0
    assert first["id"].startswith("call_")
    assert first["type"] == "function"
    assert first["function"]["name"] == "web_search"
    assert first["function"]["arguments"] == ""

    assert second["index"] == 0
    args = json.loads(second["function"]["arguments"])
    assert args == {"query": "apple silicon"}

    assert _final_finish_reason(events) == "tool_calls"


# ---------- GLM ----------

GLM_TOKENS = [
    "<tool_call>",
    "web_search",
    "<arg_key>",
    "query",
    "</arg_key>",
    "<arg_value>",
    '"hello"',
    "</arg_value>",
    "</tool_call>",
]


@pytest.mark.asyncio
async def test_glm_incremental_stream():
    engine = StubEngine("glm", GLM_TOKENS)
    req = _build_request("glm")
    events = await _collect(engine, req)

    tc_chunks = _extract_tool_call_chunks(events)
    assert len(tc_chunks) == 2
    first, second = tc_chunks

    assert first["id"].startswith("call_")
    assert first["function"]["name"] == "web_search"
    assert first["function"]["arguments"] == ""

    args = json.loads(second["function"]["arguments"])
    assert args == {"query": "hello"}

    assert _final_finish_reason(events) == "tool_calls"


# ---------- Gemma 4 ----------

GEMMA4_TOKENS = [
    "<|tool_call>",
    "call:get_weather",
    "{",
    "location:",
    '<|"|>',
    "San Francisco",
    '<|"|>',
    "}",
    "<tool_call|>",
]


@pytest.mark.asyncio
async def test_gemma4_incremental_stream():
    engine = StubEngine("gemma4", GEMMA4_TOKENS)
    req = _build_request("gemma4")
    events = await _collect(engine, req)

    tc_chunks = _extract_tool_call_chunks(events)
    assert len(tc_chunks) == 2
    first, second = tc_chunks

    assert first["function"]["name"] == "get_weather"
    assert first["function"]["arguments"] == ""

    args = json.loads(second["function"]["arguments"])
    assert args == {"location": "San Francisco"}

    assert _final_finish_reason(events) == "tool_calls"


# ---------- FIX 3: Gemma 4 thinking-channel stripped from streamed content ----------

GEMMA4_THINKING_TOKENS = [
    "<|channel>thought\n",
    "let me ", "reason about ", "this carefully",
    "<channel|>",
    "The ", "final ", "answer.",
]


class _ThinkingStubEngine(StubEngine):
    """StubEngine with thinking ENABLED and a content (no-tool) finish."""

    def __init__(self, model_family: str, token_stream: list[str]):
        super().__init__(model_family, token_stream)

        class _Cfg:
            enable_thinking = True
        self.cfg = _Cfg()

    def _iter_results(self):
        for tok in self._stream:
            yield StubResult(text=tok)
        yield StubResult(
            text="",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=len(self._stream),
            prompt_tps=100.0,
            generation_tps=20.0,
        )


@pytest.mark.asyncio
async def test_gemma4_streaming_strips_thinking_channel():
    """FIX 3: the raw <|channel>thought...<channel|> markers (and the thought
    text) must NOT appear in streamed content deltas; only the post-close
    answer is emitted."""
    engine = _ThinkingStubEngine("gemma4", GEMMA4_THINKING_TOKENS)
    req = ChatCompletionRequest(
        model="test-gemma4",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=True,
    )
    events = await _collect(engine, req)

    content = "".join(
        choice.get("delta", {}).get("content", "")
        for ev in events
        for choice in ev.get("choices", [])
        if choice.get("delta", {}).get("content")
    )
    assert "<|channel>" not in content
    assert "<channel|>" not in content
    assert "thought" not in content
    assert "reason about" not in content  # thought text suppressed
    assert content == "The final answer."
    assert _final_finish_reason(events) == "stop"


def _stream_content(events: list[dict]) -> str:
    return "".join(
        choice.get("delta", {}).get("content", "")
        for ev in events
        for choice in ev.get("choices", [])
        if choice.get("delta", {}).get("content")
    )


# CORRECTION 1: a degenerate MULTI-CYCLE stream (a 2nd <|channel>thought opener
# after visible content) must have ALL channels stripped end-to-end through
# _stream_completion — no later marker/thought may leak into content deltas.
GEMMA4_MULTICYCLE_TOKENS = [
    "<|channel>thought\n", "reason A", "<channel|>",
    "partial answer ",
    "<|channel>thought\n", "reason B", "<channel|>",
    "final answer.",
]


@pytest.mark.asyncio
async def test_gemma4_streaming_strips_multicycle_thinking():
    engine = _ThinkingStubEngine("gemma4", GEMMA4_MULTICYCLE_TOKENS)
    req = ChatCompletionRequest(
        model="test-gemma4",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=True,
    )
    events = await _collect(engine, req)
    content = _stream_content(events)
    assert "<|channel>" not in content
    assert "<channel|>" not in content
    assert "thought" not in content
    assert "reason A" not in content and "reason B" not in content
    assert content == "partial answer final answer."
    assert _final_finish_reason(events) == "stop"


# CORRECTION 2: gemma4 thinking ENABLED but the model emits a plain answer with
# NO channel markers → content must pass through unstripped (not be dropped).
GEMMA4_PLAIN_TOKENS = ["Hello", ", this is ", "a plain ", "answer."]


@pytest.mark.asyncio
async def test_gemma4_streaming_plain_answer_passes_through():
    engine = _ThinkingStubEngine("gemma4", GEMMA4_PLAIN_TOKENS)
    req = ChatCompletionRequest(
        model="test-gemma4",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=True,
    )
    events = await _collect(engine, req)
    content = _stream_content(events)
    assert content == "Hello, this is a plain answer."
    assert _final_finish_reason(events) == "stop"


# ---------- Name emitted BEFORE block closes ----------

@pytest.mark.asyncio
async def test_qwen_name_emitted_before_block_close():
    """The FIRST chunk (name) must arrive before the LAST token of the block.

    If emission is still end-of-block batched, the relative ordering below
    would fail: tool_call name chunk comes after </tool_call> token was fed.
    """
    events_seen_before_end = 0
    events_seen_after_end = 0
    engine = StubEngine("chatml", QWEN_TOKENS)
    req = _build_request("chatml")

    # Re-stream with fine-grained tracking: consume one SSE line at a time
    # and count when tool_call chunks appear relative to total stream length.
    lines = []
    async for line in _stream_completion(req, engine):
        lines.append(line)

    # Find index of first tool_call chunk and last content/block token event
    first_tc_idx = None
    for i, line in enumerate(lines):
        if '"tool_calls"' in line:
            first_tc_idx = i
            break
    assert first_tc_idx is not None
    # There must be further events after the first tool_call chunk (the args
    # chunk + final finish_reason) — this asserts incremental emission.
    assert first_tc_idx < len(lines) - 2


# ---------- Parallel tool_calls get distinct indices ----------

QWEN_PARALLEL_TOKENS = [
    "<tool_call>",
    "<function=read_file>",
    "<parameter=path>",
    "a.txt",
    "</parameter>",
    "</function>",
    "</tool_call>",
    "<tool_call>",
    "<function=read_file>",
    "<parameter=path>",
    "b.txt",
    "</parameter>",
    "</function>",
    "</tool_call>",
]


@pytest.mark.asyncio
async def test_qwen_parallel_tool_calls():
    engine = StubEngine("chatml", QWEN_PARALLEL_TOKENS)
    req = _build_request("chatml")
    events = await _collect(engine, req)

    tc_chunks = _extract_tool_call_chunks(events)
    # Expect 4 chunks: [name#0, args#0, name#1, args#1]
    assert len(tc_chunks) == 4
    indices = [c["index"] for c in tc_chunks]
    assert indices == [0, 0, 1, 1]
    # First call's args
    assert json.loads(tc_chunks[1]["function"]["arguments"]) == {"path": "a.txt"}
    # Second call's args
    assert json.loads(tc_chunks[3]["function"]["arguments"]) == {"path": "b.txt"}
    # Distinct ids on the name chunks
    assert tc_chunks[0]["id"] != tc_chunks[2]["id"]


# ---------- Content before tool_call is emitted as content delta ----------

@pytest.mark.asyncio
async def test_content_before_tool_call_emitted():
    engine = StubEngine("chatml", QWEN_TOKENS)
    req = _build_request("chatml")
    events = await _collect(engine, req)

    # Concatenate all content deltas
    content_parts = []
    for ev in events:
        for choice in ev.get("choices", []):
            c = choice.get("delta", {}).get("content")
            if c:
                content_parts.append(c)
    joined = "".join(content_parts)
    assert "Looking" in joined
    assert "up." in joined
    # Raw XML must not leak into content
    assert "<tool_call>" not in joined
    assert "<function=" not in joined


# ---------- FIX 3: partial tool-start marker AFTER visible text in one token ----------

# A single token carries visible content AND the start of the tool marker
# ("before <too"). The holdback must emit only the preceding visible text and
# RETAIN the partial-marker suffix — the old code leaked "before <too" because
# it only held when the WHOLE stripped buffer was a prefix of TOOL_START.
QWEN_SPLIT_START_TOKENS = [
    "Here is the result: ",
    "<too",                         # visible text already emitted; this is a pure prefix
    "l_call><function=web_search>",
    "<parameter=query>",
    "mlx",
    "</parameter>",
    "</function>",
    "</tool_call>",
]

# Even harder: visible text and the partial marker share ONE token.
QWEN_GLUED_START_TOKENS = [
    "before <too",                  # visible "before " + partial "<too" together
    "l_call><function=web_search>",
    "<parameter=query>",
    "mlx",
    "</parameter>",
    "</function>",
    "</tool_call>",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [QWEN_SPLIT_START_TOKENS, QWEN_GLUED_START_TOKENS])
async def test_partial_tool_start_after_visible_text_not_leaked(tokens):
    engine = StubEngine("chatml", tokens)
    req = _build_request("chatml")
    events = await _collect(engine, req)

    content = "".join(
        choice.get("delta", {}).get("content", "")
        for ev in events
        for choice in ev.get("choices", [])
        if choice.get("delta", {}).get("content")
    )
    # The partial start marker must NOT leak into content...
    assert "<too" not in content
    assert "<tool_call>" not in content
    assert "<function=" not in content
    # ...but the preceding visible text must still be emitted intact.
    if tokens is QWEN_SPLIT_START_TOKENS:
        assert content == "Here is the result: "
    else:
        assert content == "before "

    # And the tool call still parses correctly.
    tc_chunks = _extract_tool_call_chunks(events)
    assert tc_chunks[0]["function"]["name"] == "web_search"
    args_chunk = next(c for c in tc_chunks if c.get("function", {}).get("arguments"))
    assert json.loads(args_chunk["function"]["arguments"]) == {"query": "mlx"}
    assert _final_finish_reason(events) == "tool_calls"


@pytest.mark.asyncio
async def test_partial_tool_start_at_stream_end_flushed_as_content():
    """FIX 3 complement: a partial tool-start that NEVER completes (stream ends
    on 'answer text <too') is real content — the held suffix must be flushed,
    not dropped, and the visible prefix emitted before it."""
    engine = StubEngine("chatml", ["answer text <too"])
    # No real tool call -> finish must fall back to "stop"; reuse _build_request
    # (tools present) but the model never closes a block.
    req = _build_request("chatml")
    events = await _collect(engine, req)
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for ev in events
        for choice in ev.get("choices", [])
        if choice.get("delta", {}).get("content")
    )
    assert content == "answer text <too"  # nothing dropped


# ---------- U6/F1: corruption-terminated stream -> in-band error envelope ----------


class ErrorTerminalStubEngine(StubEngine):
    """Stub whose stream ends with the engine-internal 'error' terminal
    (fail-closed cache corruption) and which records persistence calls."""

    def __init__(self, model_family: str, token_stream: list[str]):
        super().__init__(model_family, token_stream)
        self.persist_calls: list = []

    def _iter_results(self):
        for tok in self._stream:
            yield StubResult(text=tok)
        yield StubResult(
            text="",
            finish_reason="error",
            prompt_tokens=10,
            completion_tokens=len(self._stream),
        )

    def update_session_messages(self, *args, **kwargs):
        self.persist_calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_error_terminal_emits_error_envelope_not_completion():
    """U6/F1: a corruption-terminated stream must NOT be dressed up as a
    normal completion. The adapter emits the in-band error envelope
    (data:{"error":{...}} then [DONE]) — 'error' is not a valid OpenAI
    finish_reason — and suppresses (a) the best-effort truncated tool-call
    emission and (b) session persistence of the truncated text."""
    # Stream dies inside a tool_call block, before the name is complete.
    tokens = ["Partial answer ", "<tool_call>", "<function=web_se"]
    engine = ErrorTerminalStubEngine("chatml", tokens)
    req = ChatCompletionRequest(
        model="test-chatml",
        messages=[ChatMessage(role="user", content="search please")],
        tools=[ToolDef(function=FunctionDef(name="web_search", parameters={}))],
        stream=True,
        thinking=False,
        user="sess-err",  # persistence WOULD run on the normal path
    )
    from mlx_soloheaven.api.openai_compat import _stream_completion as _sc
    lines = [line async for line in _sc(req, engine)]

    # Terminal shape: error envelope followed by [DONE], nothing after.
    assert lines[-1] == "data: [DONE]\n\n"
    err = json.loads(lines[-2][len("data: "):])
    assert "error" in err
    assert err["error"]["type"] == "server_error"

    # No tool_call chunk (not even the truncated best-effort one) and no
    # normal finish_reason frame anywhere in the stream.
    payloads = []
    for line in lines:
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            payloads.append(json.loads(line[len("data: "):]))
    for ev in payloads:
        for choice in ev.get("choices", []):
            assert not choice.get("delta", {}).get("tool_calls")
            assert choice.get("finish_reason") is None

    # Nothing persisted.
    assert engine.persist_calls == []


@pytest.mark.asyncio
async def test_error_terminal_never_synthesizes_toolcalls_finish():
    """Even when a COMPLETE tool_call block streamed before the corruption
    terminal, the stream must end with the error envelope — never a
    finish_reason='tool_calls' completion frame."""
    engine = ErrorTerminalStubEngine("chatml", QWEN_TOKENS)
    req = _build_request("chatml")
    events = await _collect(engine, req)
    assert _final_finish_reason(events) is None  # no completion frame at all
    assert any("error" in ev for ev in events)
    assert engine.persist_calls == []


def test_sync_completion_error_returns_500_not_toolcalls():
    """U6/F1 non-streaming: an 'error' CompletionResult becomes a 500 error
    object — never a completion (and never tool_calls), and nothing is
    persisted."""
    from types import SimpleNamespace

    from fastapi.responses import JSONResponse

    from mlx_soloheaven.api.openai_compat import _sync_completion
    from mlx_soloheaven.engine.types import CompletionResult

    class _Eng:
        model_family = "chatml"
        model_id = "test-model"
        cfg = SimpleNamespace(enable_thinking=False)

        def complete(self, *args, **kwargs):
            return CompletionResult(
                content="<tool_call>\n<function=hack>\n</function>\n</tool_call>",
                finish_reason="error",
            )

        def update_session_messages(self, *args, **kwargs):
            raise AssertionError("must not persist on the error path")

    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        thinking=False,
        user="sess-err",
    )
    resp = _sync_completion(req, _Eng())
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["type"] == "server_error"
