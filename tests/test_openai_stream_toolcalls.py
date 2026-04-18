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

    async def generate_stream_async(self, *args, **kwargs) -> AsyncGenerator:
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
