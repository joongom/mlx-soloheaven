"""Tests for the reasoning-channel feature (LM-Studio-style reasoning_content).

Covers the SHARED ``ThinkingRouter`` plus both API surfaces:

* Non-streaming OpenAI-compat ``_sync_completion`` sets ``reasoning_content``
  and a clean ``content`` for gemma4 AND chatml.
* Streaming OpenAI-compat ``_stream_completion`` emits ``reasoning_content``
  deltas during the thought phase, then ``content`` deltas for the answer —
  including multi-cycle gemma4 output and markers split across chunks — and the
  ``content`` deltas never contain ``<|channel>`` / ``<channel|>`` / ``<think>``
  / ``</think>`` markers.
* Tool calls still parse from the CONTENT portion (reasoning is never tool XML).
"""
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import pytest

from mlx_soloheaven.api.openai_compat import _stream_completion, _sync_completion
from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    ToolDef,
)
from mlx_soloheaven.engine.tool_parser import (
    CHANNEL_CONTENT,
    CHANNEL_REASONING,
    ThinkingRouter,
    parse_tool_calls,
    split_thinking_and_content,
)


# ---------------------------------------------------------------------------
# ThinkingRouter unit tests (the shared engine)
# ---------------------------------------------------------------------------


def _route_all(router: ThinkingRouter, chunks):
    segs = []
    for c in chunks:
        segs.extend(router.feed(c))
    segs.extend(router.flush())
    return segs


def _join(segs, channel):
    return "".join(t for ch, t in segs if ch == channel)


def test_router_gemma4_single_chunk_split():
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, ["<|channel>thought\nreasoning here<channel|>The answer"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning here"
    assert _join(segs, CHANNEL_CONTENT) == "The answer"


def test_router_gemma4_reasoning_then_content_across_chunks():
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(
        r, ["<|channel>thought\nthink", "ing more", "<channel|>ans", "wer"]
    )
    assert _join(segs, CHANNEL_REASONING) == "thinking more"
    assert _join(segs, CHANNEL_CONTENT) == "answer"


def test_router_gemma4_close_marker_split_across_chunks():
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, ["thought\nreason<chan", "nel|>visible"])
    assert _join(segs, CHANNEL_REASONING) == "reason"
    assert _join(segs, CHANNEL_CONTENT) == "visible"
    # No marker fragment in any segment.
    for _, t in segs:
        assert "channel" not in t


def test_router_gemma4_multicycle():
    r = ThinkingRouter(active=True, model_family="gemma4")
    raw = (
        "<|channel>thought\nA<channel|>partial"
        "<|channel>thought\nB<channel|>final"
    )
    segs = _route_all(r, [raw])
    assert _join(segs, CHANNEL_REASONING) == "AB"
    assert _join(segs, CHANNEL_CONTENT) == "partialfinal"


def test_router_gemma4_multicycle_char_by_char():
    raw = "<|channel>thought\nA<channel|>partial<|channel>thought\nB<channel|>final"
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, list(raw))
    assert _join(segs, CHANNEL_CONTENT) == "partialfinal"
    for marker in ("<|channel>", "<channel|>", "thought\n"):
        for _, t in segs:
            assert marker not in t


def test_router_gemma4_plain_answer_no_markers_is_content():
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, ["Hello, ", "plain answer."])
    assert _join(segs, CHANNEL_CONTENT) == "Hello, plain answer."
    assert _join(segs, CHANNEL_REASONING) == ""


def test_router_chatml_starts_in_reasoning():
    # chatml: opener <think> is in the prompt suffix; stream begins inside the
    # thought block and routes up to </think>.
    r = ThinkingRouter(active=True, model_family="chatml")
    segs = _route_all(r, ["reasoning text", "</think>", "the answer"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning text"
    assert _join(segs, CHANNEL_CONTENT) == "the answer"


def test_router_chatml_close_split_across_chunks():
    r = ThinkingRouter(active=True, model_family="chatml")
    segs = _route_all(r, ["reasoning</thi", "nk>answer"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning"
    assert _join(segs, CHANNEL_CONTENT) == "answer"
    for _, t in segs:
        assert "think" not in t


def test_router_inactive_passthrough_is_content():
    r = ThinkingRouter(active=False, model_family="chatml")
    segs = _route_all(r, ["anything </think> verbatim"])
    assert _join(segs, CHANNEL_CONTENT) == "anything </think> verbatim"
    assert _join(segs, CHANNEL_REASONING) == ""


# ---------------------------------------------------------------------------
# FIX 1: chatml/glm degenerate "no </think>" — stream AND non-stream both route
# the whole output to reasoning (it started inside <think> from the prompt, so
# with no close it is all reasoning, no final answer).
# ---------------------------------------------------------------------------


def test_router_chatml_no_close_routes_all_to_reasoning():
    r = ThinkingRouter(active=True, model_family="chatml")
    segs = _route_all(r, ["reasoning that ", "never closes the block"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning that never closes the block"
    assert _join(segs, CHANNEL_CONTENT) == ""


def test_split_chatml_no_close_is_reasoning_when_started_in_thinking():
    # started_in_thinking=True mirrors the streaming router (active=True).
    thinking, content = split_thinking_and_content(
        "reasoning that never closes the block",
        model_family="chatml",
        started_in_thinking=True,
    )
    assert thinking == "reasoning that never closes the block"
    assert content == ""


def test_split_chatml_no_close_stream_equals_non_stream():
    """FIX 1: the non-streaming split (started_in_thinking=True) must agree with
    the streaming router for a chatml stream that never emits </think>."""
    raw = "let me think step by step but I never finish"
    r = ThinkingRouter(active=True, model_family="chatml")
    segs = _route_all(r, [raw])
    stream_reasoning = _join(segs, CHANNEL_REASONING)
    stream_content = _join(segs, CHANNEL_CONTENT)
    split_reasoning, split_content = split_thinking_and_content(
        raw, model_family="chatml", started_in_thinking=True
    )
    assert stream_reasoning == split_reasoning == raw
    assert stream_content == split_content == ""


def test_split_chatml_no_close_default_is_legacy_content():
    """Without started_in_thinking (non-thinking output / older callers), a
    no-marker chatml answer must stay CONTENT — NOT be misrouted to reasoning
    (which would empty the answer)."""
    thinking, content = split_thinking_and_content(
        "just a plain answer", model_family="chatml"
    )
    assert thinking is None
    assert content == "just a plain answer"


def test_split_chatml_with_close_unaffected_by_flag():
    """The closed case is unchanged regardless of started_in_thinking."""
    for flag in (True, False):
        t, c = split_thinking_and_content(
            "secret reasoning</think>the answer",
            model_family="chatml",
            started_in_thinking=flag,
        )
        assert t == "secret reasoning"
        assert c == "the answer"


def test_split_gemma4_unaffected_by_started_in_thinking_flag():
    """gemma4 split must be unchanged by the chatml-only flag."""
    for flag in (True, False):
        t, c = split_thinking_and_content(
            "<|channel>thought\nreason<channel|>answer",
            model_family="gemma4",
            started_in_thinking=flag,
        )
        assert t == "reason"
        assert c == "answer"


# ---------------------------------------------------------------------------
# FIX 2: flush() must NOT leak a partial chatml marker fragment held in _pending
# while still in reasoning (a split </think> or <think> that never completed).
# ---------------------------------------------------------------------------


def test_flush_drops_partial_chatml_close_marker():
    r = ThinkingRouter(active=True, model_family="chatml")
    # Stream ends mid-</think>: "reasoning</thi" — the "</thi" tail is a partial
    # close marker and must be dropped, never emitted as reasoning text.
    segs = _route_all(r, ["reasoning</thi"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning"
    assert _join(segs, CHANNEL_CONTENT) == ""
    for _, t in segs:
        assert "</thi" not in t
        assert "think" not in t


def test_flush_drops_partial_chatml_open_marker():
    r = ThinkingRouter(active=True, model_family="chatml")
    # A stray opening <think> split at stream end: "reasoning <thi" — the
    # "<thi" tail is a partial open marker and must be dropped.
    segs = _route_all(r, ["reasoning <thi"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning "
    for _, t in segs:
        assert "<thi" not in t


def test_flush_emits_real_reasoning_remainder():
    """A held tail that is NOT a partial marker is real reasoning text and must
    still be emitted on flush (e.g. a trailing '<' that is not a marker start
    is held as a possible <think> prefix and emitted)."""
    r = ThinkingRouter(active=True, model_family="chatml")
    segs = _route_all(r, ["plain reasoning text"])
    assert _join(segs, CHANNEL_REASONING) == "plain reasoning text"


# ---------------------------------------------------------------------------
# FIX 4: gemma4 bare ``thought\n`` opener only at generation START. A literal
# ``thought\n`` line in content / tool args (after visible content, no
# <|channel>) must stay CONTENT — only the FULL <|channel>thought may re-open.
# ---------------------------------------------------------------------------


def test_gemma4_bare_thought_in_content_stays_content():
    r = ThinkingRouter(active=True, model_family="gemma4")
    # Visible content first, then a literal "thought\n" line (no <|channel>).
    segs = _route_all(
        r, ["Here is a code sample:\n", "thought\nis a variable name here"]
    )
    assert _join(segs, CHANNEL_REASONING) == ""
    assert (
        _join(segs, CHANNEL_CONTENT)
        == "Here is a code sample:\nthought\nis a variable name here"
    )


def test_gemma4_bare_thought_in_toolargs_stays_content():
    r = ThinkingRouter(active=True, model_family="gemma4")
    # A tool-arg value that literally contains a "thought\n" line after content.
    segs = _route_all(
        r, ['{"note": "first line\n', 'thought\nsecond line"}']
    )
    assert _join(segs, CHANNEL_REASONING) == ""
    assert "thought" in _join(segs, CHANNEL_CONTENT)
    assert _join(segs, CHANNEL_CONTENT) == '{"note": "first line\nthought\nsecond line"}'


def test_gemma4_bare_thought_at_generation_start_still_opens():
    """The sliding-window first-token bare ``thought\\n`` must STILL open
    reasoning (FIX 4 only constrains the MID-content re-opener)."""
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, ["thought\nreasoning here<channel|>the answer"])
    assert _join(segs, CHANNEL_REASONING) == "reasoning here"
    assert _join(segs, CHANNEL_CONTENT) == "the answer"


def test_gemma4_full_channel_reopener_after_content_still_works():
    """The FULL <|channel>thought opener must STILL re-open mid-content (real
    multi-cycle), even though the bare opener no longer does."""
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(
        r,
        ["answer one <|channel>thought\nmore reasoning<channel|>answer two"],
    )
    assert _join(segs, CHANNEL_REASONING) == "more reasoning"
    assert _join(segs, CHANNEL_CONTENT) == "answer one answer two"


def test_gemma4_bare_thought_after_content_split_across_chunks_stays_content():
    """Even split across chunks, a bare ``thought\\n`` after visible content is
    plain content (the partial-prefix holdback must not suppress it forever)."""
    r = ThinkingRouter(active=True, model_family="gemma4")
    segs = _route_all(r, ["visible ", "thou", "ght\nmore text"])
    assert _join(segs, CHANNEL_REASONING) == ""
    assert _join(segs, CHANNEL_CONTENT) == "visible thought\nmore text"


# ---------------------------------------------------------------------------
# Non-streaming _sync_completion: reasoning_content + clean content
# ---------------------------------------------------------------------------


@dataclass
class _SyncResult:
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[list] = None
    finish_reason: str = "stop"
    prompt_tokens: int = 5
    completion_tokens: int = 10
    cache_info: Optional[dict] = None


class _SyncEngine:
    """Stub engine whose ``complete`` mirrors the real engine: split raw output
    into thinking/content (and parse tool calls from content)."""

    def __init__(self, model_family: str, raw_output: str):
        self.model_family = model_family
        self.model_id = f"test-{model_family}"
        self._raw = raw_output

        class _Cfg:
            enable_thinking = True
        self.cfg = _Cfg()

    def complete(self, messages, *, tools=None, **kwargs) -> _SyncResult:
        thinking, content = split_thinking_and_content(
            self._raw, model_family=self.model_family
        )
        res = _SyncResult(thinking=thinking)
        if tools:
            text_part, calls = parse_tool_calls(content, model_family=self.model_family)
            if calls:
                res.tool_calls = calls
                res.content = text_part or None
                res.finish_reason = "tool_calls"
            else:
                res.content = content
        else:
            res.content = content
        return res

    def update_session_messages(self, *a, **k):
        pass


def test_sync_gemma4_sets_reasoning_content_and_clean_content():
    raw = "<|channel>thought\nthe reasoning<channel|>391"
    engine = _SyncEngine("gemma4", raw)
    req = ChatCompletionRequest(
        model="test-gemma4",
        messages=[ChatMessage(role="user", content="what is 17*23")],
        stream=False,
        thinking=True,
    )
    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.content == "391"
    assert msg.reasoning_content == "the reasoning"
    assert "<|channel>" not in (msg.content or "")
    assert "<channel|>" not in (msg.content or "")


def test_sync_chatml_sets_reasoning_content_and_clean_content():
    raw = "<think>let me think</think>Hello there"
    engine = _SyncEngine("chatml", raw)
    req = ChatCompletionRequest(
        model="test-chatml",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        thinking=True,
    )
    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.content == "Hello there"
    assert msg.reasoning_content == "let me think"
    assert "<think>" not in (msg.content or "")
    assert "</think>" not in (msg.content or "")


def test_sync_no_thinking_leaves_reasoning_content_none():
    raw = "just a plain answer"
    engine = _SyncEngine("chatml", raw)
    req = ChatCompletionRequest(
        model="test-chatml",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        thinking=False,
    )
    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.content == "just a plain answer"
    assert msg.reasoning_content is None


def test_sync_tool_call_parses_from_content_not_reasoning():
    # Reasoning channel, then a tool call in the answer portion.
    raw = (
        "<|channel>thought\nI should search<channel|>"
        "<|tool_call>call:web_search{query:<|\"|>mlx<|\"|>}<tool_call|>"
    )
    engine = _SyncEngine("gemma4", raw)
    req = ChatCompletionRequest(
        model="test-gemma4",
        messages=[ChatMessage(role="user", content="search mlx")],
        tools=[ToolDef(function=FunctionDef(name="web_search", parameters={}))],
        stream=False,
        thinking=True,
    )
    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.reasoning_content == "I should search"
    assert msg.tool_calls and len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "web_search"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"query": "mlx"}
    assert resp.choices[0].finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Streaming _stream_completion: reasoning_content deltas then content deltas
# ---------------------------------------------------------------------------


@dataclass
class _StubResult:
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None
    # Finding 4: a real GenerationResult carries a token_produced flag; these
    # stubs never emit keepalives, so any CONTENT frame is a real token.
    token_produced: bool = False

    def __post_init__(self):
        if (
            self.status is None
            and self.finish_reason is None
            and not self.token_produced
        ):
            self.token_produced = True


class _StreamEngine:
    def __init__(self, model_family: str, token_stream: list[str], finish="stop"):
        self.model_family = model_family
        self.model_id = f"test-{model_family}"
        self._stream = token_stream
        self._finish = finish

        class _Cfg:
            enable_thinking = True
        self.cfg = _Cfg()

    def _iter(self):
        for tok in self._stream:
            yield _StubResult(text=tok)
        yield _StubResult(
            text="",
            finish_reason=self._finish,
            prompt_tokens=10,
            completion_tokens=len(self._stream),
        )

    async def generate_stream_batches_async(self, *a, **k) -> AsyncGenerator:
        for r in self._iter():
            yield [r]

    def update_session_messages(self, *a, **k):
        pass


async def _collect(engine, request) -> list[dict]:
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


def _delta_field(events, field):
    return "".join(
        choice.get("delta", {}).get(field, "")
        for ev in events
        for choice in ev.get("choices", [])
        if choice.get("delta", {}).get(field)
    )


def _req(model_family, tools=None):
    return ChatCompletionRequest(
        model=f"test-{model_family}",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=True,
        tools=tools,
    )


GEMMA4_THINKING = [
    "<|channel>thought\n", "let me ", "reason ", "carefully",
    "<channel|>", "The ", "final ", "answer.",
]


@pytest.mark.asyncio
async def test_stream_gemma4_reasoning_then_content():
    engine = _StreamEngine("gemma4", GEMMA4_THINKING)
    events = await _collect(engine, _req("gemma4"))
    reasoning = _delta_field(events, "reasoning_content")
    content = _delta_field(events, "content")
    assert reasoning == "let me reason carefully"
    assert content == "The final answer."
    for marker in ("<|channel>", "<channel|>", "thought"):
        assert marker not in content
    # Reasoning deltas precede content deltas.
    first_content = next(
        i for i, ev in enumerate(events)
        for c in ev.get("choices", []) if c.get("delta", {}).get("content")
    )
    last_reasoning = max(
        i for i, ev in enumerate(events)
        for c in ev.get("choices", []) if c.get("delta", {}).get("reasoning_content")
    )
    assert last_reasoning < first_content


GEMMA4_MULTICYCLE = [
    "<|channel>thought\n", "reason A", "<channel|>", "partial answer ",
    "<|channel>thought\n", "reason B", "<channel|>", "final answer.",
]


@pytest.mark.asyncio
async def test_stream_gemma4_multicycle_routes_all_reasoning():
    engine = _StreamEngine("gemma4", GEMMA4_MULTICYCLE)
    events = await _collect(engine, _req("gemma4"))
    reasoning = _delta_field(events, "reasoning_content")
    content = _delta_field(events, "content")
    assert reasoning == "reason Areason B"
    assert content == "partial answer final answer."
    for marker in ("<|channel>", "<channel|>", "thought"):
        assert marker not in content


@pytest.mark.asyncio
async def test_stream_multicycle_preserves_interleave_order():
    """Multi-cycle: content_1 -> reasoning_2 -> content_2. The emitted delta
    ORDER must preserve the interleave (content_1 before reasoning_2 before
    content_2), even when several tokens land in one coalesced batch."""
    engine = _StreamEngine("gemma4", GEMMA4_MULTICYCLE)
    events = await _collect(engine, _req("gemma4"))
    # Build the ordered channel sequence of emitted deltas.
    order = []
    for ev in events:
        for c in ev.get("choices", []):
            d = c.get("delta", {})
            if d.get("reasoning_content"):
                order.append(("r", d["reasoning_content"]))
            if d.get("content"):
                order.append(("c", d["content"]))
    texts = [t for _, t in order]
    # "partial answer " (content) must appear before "reason B" (reasoning),
    # which must appear before "final answer." (content).
    assert texts.index("partial answer ") < texts.index("reason B")
    assert texts.index("reason B") < texts.index("final answer.")


# Close marker split across chunk boundaries.
GEMMA4_SPLIT = [
    "<|channel>thought\n", "secret rea", "soning<chan", "nel|>", "vis", "ible"
]


@pytest.mark.asyncio
async def test_stream_gemma4_close_marker_split_across_chunks():
    engine = _StreamEngine("gemma4", GEMMA4_SPLIT)
    events = await _collect(engine, _req("gemma4"))
    reasoning = _delta_field(events, "reasoning_content")
    content = _delta_field(events, "content")
    assert reasoning == "secret reasoning"
    assert content == "visible"
    assert "chan" not in content
    assert "channel" not in content


CHATML_THINKING = [
    "thinking ", "about it", "</think>", "Here is ", "the answer."
]


@pytest.mark.asyncio
async def test_stream_chatml_routes_think_to_reasoning():
    engine = _StreamEngine("chatml", CHATML_THINKING)
    events = await _collect(engine, _req("chatml"))
    reasoning = _delta_field(events, "reasoning_content")
    content = _delta_field(events, "content")
    assert reasoning == "thinking about it"
    assert content == "Here is the answer."
    for marker in ("<think>", "</think>"):
        assert marker not in content
        assert marker not in reasoning


@pytest.mark.asyncio
async def test_stream_content_deltas_never_contain_markers():
    for fam, toks in (
        ("gemma4", GEMMA4_THINKING),
        ("gemma4", GEMMA4_MULTICYCLE),
        ("chatml", CHATML_THINKING),
    ):
        engine = _StreamEngine(fam, toks)
        events = await _collect(engine, _req(fam))
        content = _delta_field(events, "content")
        for marker in ("<|channel>", "<channel|>", "<think>", "</think>", "thought\n"):
            assert marker not in content, (fam, marker, content)


# Tool call after reasoning: reasoning -> reasoning_content, the tool XML is
# parsed from content (never appears as a content delta, never as reasoning).
GEMMA4_TOOL_AFTER_THINK = [
    "<|channel>thought\n", "need to search", "<channel|>",
    "<|tool_call>", "call:web_search", "{", "query:", '<|"|>', "mlx", '<|"|>',
    "}", "<tool_call|>",
]


@pytest.mark.asyncio
async def test_stream_tool_call_parses_from_content_after_reasoning():
    engine = _StreamEngine("gemma4", GEMMA4_TOOL_AFTER_THINK, finish="tool_calls")
    tool = ToolDef(function=FunctionDef(name="web_search", parameters={}))
    events = await _collect(engine, _req("gemma4", tools=[tool]))

    reasoning = _delta_field(events, "reasoning_content")
    content = _delta_field(events, "content")
    assert reasoning == "need to search"
    # Tool XML must not leak into content deltas.
    assert "<|tool_call>" not in content
    assert "web_search" not in content

    tcs = [
        tc
        for ev in events
        for c in ev.get("choices", [])
        for tc in (c.get("delta", {}).get("tool_calls") or [])
    ]
    names = [tc.get("function", {}).get("name") for tc in tcs if tc.get("function", {}).get("name")]
    assert names == ["web_search"]
    finish = next(
        (c.get("finish_reason") for ev in events for c in ev.get("choices", [])
         if c.get("finish_reason")),
        None,
    )
    assert finish == "tool_calls"
