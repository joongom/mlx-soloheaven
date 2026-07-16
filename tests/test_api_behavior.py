"""API-behavior regression tests — review batch 4 (U7/U8/U9/U11/U12/U24/U25).

U7  — max_tokens exhaustion reports finish_reason="length" end-to-end: the
      engine detects it (runner signal or max_tokens/EOS heuristic) and the
      OpenAI-compat surfaces no longer overwrite it with "stop"; tool-call
      turns keep "tool_calls", corruption keeps the batch-2 "error" envelope.
U8  — _pld_response_adapter finalizes the detokenizer on max_tokens
      exhaustion and flushes the buffered tail as a TEXT-ONLY frame
      (token=None — ids were already recorded per-frame, so the cache/token
      accounting contract is untouched).
U9  — structured output (response_format json_schema/json_object) SUPPRESSES
      thinking for the request: the FSM constrains output to the JSON grammar
      from the first token, so pairing it with the <think> prompt opener
      routed the whole JSON answer to reasoning_content (content empty).
      Suppression is applied engine-side (prompt contract) and API-side
      (router / split), keeping stream == non-stream.
U11 — tool/function/parameter name parsing accepts OpenAI's legal charset
      ([A-Za-z0-9_.-], e.g. MCP-style "server.tool-name"); an unparseable
      block is passed through as raw content (fail-closed), never silently
      deleted — in both the batch parser and the streaming FSM. Round 2:
      preservation is PER BLOCK — a malformed block next to a parseable one
      (and all surrounding ordinary text) is retained in content in original
      order, converging batch content with what streaming emits.
U12 — tool-call XML inside the THINKING region is a rehearsal: never parsed
      into a real tool_calls entry (engine batch parse now runs on the
      content channel only, mirroring the streaming router) and never
      counted as a residual-marker/extractable-call conflict at match time.
      Round 2: the content channel is the UNION of ALL content segments —
      gemma4 multi-cycle output (thought → content with a call → thought →
      content) keeps the earlier cycle's call strippable/extractable while
      thinking segments stay excluded.
U24 — the ``stop`` request parameter is implemented engine-side: scan with a
      cross-token holdback buffer, stop BEFORE emitting the stop sequence,
      finish_reason="stop". Round 2 contract (commit-or-invalidate): the
      withheld stop text must NEVER survive in reusable KV state — when the
      visible text ends exactly on a token boundary AND every cache layer is
      trimmable, the recorded ids and the cache are trimmed back to the
      boundary (messages ↔ ids ↔ cache agree); otherwise (mid-token match,
      hybrid/untrimmable layers, MTP head layouts) the truncated text is
      committed and the session cache is INVALIDATED fail-closed (honest
      MISS next turn). Multi-stop scanning is deterministic under any chunk
      segmentation (earliest-START semantics with viable-partial holdback).
U25 — delete-last: the DB drives and the engine follows. The handler deletes
      exactly the last-turn suffix (compaction summary = 1; a turn spans from
      the last user message, covering tool chains) and derives the engine
      view from the ACTUAL remaining rows; a compaction delete rebuilds via
      compact_session (the engine session holds the compacted view, which a
      count-slice cannot express). Round 2 (concurrency): the turn's rows are
      deleted by their snapshot IDS in ONE transaction, the post-delete state
      is re-read fresh, and the truncate fast path is used only when the
      fresh rows equal snapshot-minus-deleted — any drift rebuilds from the
      fresh view (a concurrent append survives and is never mis-deleted).

Harness pieces are shared with tests/test_qwen_mtp.py (same-directory import
under pytest's prepend import mode).
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import AsyncGenerator, Optional

import pytest

from mlx_soloheaven.api.openai_compat import _stream_completion, _sync_completion
from mlx_soloheaven.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    ResponseFormat,
    ToolDef,
)
from mlx_soloheaven.engine.tool_parser import (
    parse_tool_calls,
    try_extract_tool_name,
)

from test_qwen_mtp import (
    MockKV,
    _generate_engine,
    _restamp_contract,
    _scripted_lm_stream,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class _StubResult:
    """Matches the fields the API stream readers touch on a GenerationResult."""
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None


class _StreamStubEngine:
    """Engine stub replaying a canned token stream with a chosen terminal
    finish_reason; records the kwargs of the generation call."""

    def __init__(
        self,
        token_stream: list[str],
        *,
        model_family: str = "chatml",
        finish_reason: str = "stop",
        enable_thinking: bool = False,
    ):
        self.model_family = model_family
        self.model_id = f"test-{model_family}"
        self._stream = token_stream
        self._finish = finish_reason
        self.captured_kwargs: dict = {}

        class _Cfg:
            pass
        _Cfg.enable_thinking = enable_thinking
        self.cfg = _Cfg()

    def _iter_results(self):
        for tok in self._stream:
            yield _StubResult(text=tok)
        yield _StubResult(
            text="", finish_reason=self._finish,
            prompt_tokens=10, completion_tokens=len(self._stream),
        )

    async def generate_stream_batches_async(self, *args, **kwargs) -> AsyncGenerator:
        self.captured_kwargs = kwargs
        for result in self._iter_results():
            yield [result]

    def update_session_messages(self, *args, **kwargs):
        pass


async def _collect_sse(engine, request) -> list[dict]:
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


def _final_finish_reason(events: list[dict]) -> Optional[str]:
    for ev in reversed(events):
        for choice in ev.get("choices", []):
            if choice.get("finish_reason"):
                return choice["finish_reason"]
    return None


def _joined_delta(events: list[dict], key: str) -> str:
    return "".join(
        c["delta"].get(key) or ""
        for ev in events
        for c in ev.get("choices", [])
    )


def _drive_locked(eng, messages, *, max_tokens=8, thinking=False, tools=None,
                  response_format=None, stop_sequences=None):
    """Drive _generate_locked to completion with batch-4 knobs exposed."""
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=tools,
            cancel_event=None,
            thinking=thinking,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=response_format,
            stop_sequences=stop_sequences,
        )
    )


def _plain_engine(script, monkeypatch, *, suffix=(52, 99)):
    """mlx-lm-path harness engine with a HIT session and a scripted stream."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=list(suffix))
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [script])
    return eng, cs, messages, stored


# ---------------------------------------------------------------------------
# U7 — finish_reason="length" survives end-to-end
# ---------------------------------------------------------------------------


def test_u7_engine_length_heuristic_scripted(monkeypatch):
    """ENGINE: a scripted stream that exhausts max_tokens without an EOS must
    terminate with finish_reason="length" (pre-fix: unconditional "stop")."""
    eng, cs, messages, stored = _plain_engine([(101, "a"), (102, "b")], monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=2)
    assert chunks[-1].finish_reason == "length"
    # Accounting untouched: both tokens recorded, cache consistent.
    assert cs.token_ids == stored + [52, 99] + [101, 102]


def test_u7_engine_short_stream_still_stop(monkeypatch):
    """ENGINE: a stream ending well under max_tokens stays "stop"."""
    eng, cs, messages, stored = _plain_engine([(101, "a")], monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=8)
    assert chunks[-1].finish_reason == "stop"


def test_u7_engine_explicit_runner_stop_wins_over_heuristic(monkeypatch):
    """ENGINE: an explicit runner finish_reason="stop" (mlx-lm's EOS terminal
    frame) is respected even when the token count reaches max_tokens."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            script = [(101, "a", None), (102, "", "stop")]
            for tid, text, fr in script:
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset += 1
                yield SimpleNamespace(
                    text=text, token=tid, prompt_tps=1.0, generation_tps=1.0,
                    finish_reason=fr,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)
    chunks = _drive_locked(eng, messages, max_tokens=2)
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_u7_stream_length_survives_to_final_chunk():
    """API STREAM: the engine's "length" terminal must reach the client's
    final chunk (pre-fix the API overwrote it with "stop")."""
    engine = _StreamStubEngine(["Hello", " wor"], finish_reason="length")
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _final_finish_reason(events) == "length"
    assert _joined_delta(events, "content") == "Hello wor"


@pytest.mark.asyncio
async def test_u7_stream_tool_calls_still_win():
    """API STREAM: a parsed tool call keeps finish_reason="tool_calls" even
    when the engine reported "length"."""
    engine = _StreamStubEngine(
        ["<tool_call><function=get_x>", "</function></tool_call>"],
        finish_reason="length",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _final_finish_reason(events) == "tool_calls"


def test_u7_sync_completion_passes_length_through():
    """API NON-STREAM: engine finish_reason="length" flows into the choice."""

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def complete(self, messages, **kwargs):
            return SimpleNamespace(
                content="truncated answer", thinking=None, tool_calls=None,
                finish_reason="length", prompt_tokens=5, completion_tokens=8,
                cache_info=None,
            )

        def update_session_messages(self, *a, **k):
            pass

    req = ChatCompletionRequest(
        model="m", stream=False, thinking=False,
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _sync_completion(req, _Eng())
    assert resp.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_u7_web_chat_done_event_carries_finish_reason():
    """Internal /chat SSE: the done event surfaces the engine terminal reason."""
    from mlx_soloheaven.api.chat import _stream_chat_body

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def is_busy(self):
            return False

        async def generate_stream_batches_async(self, *a, **k):
            yield [_StubResult(status="generating")]
            yield [_StubResult(finish_reason="length", completion_tokens=3)]

        def update_session_messages(self, *a, **k):
            pass

    events = []
    async for line in _stream_chat_body("s1", [{"role": "user", "content": "q"}], _Eng()):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# U8 — adapter finalizes the detokenizer on max_tokens exhaustion
# ---------------------------------------------------------------------------


class _HoldingDetok:
    """Detokenizer stub that buffers ALL text until finalize() (models the
    BPE-held-whitespace / partial-UTF-8 shape)."""

    def __init__(self, tail: str):
        self._tail = tail
        self._out = ""
        self.finalized = False

    def reset(self):
        pass

    def add_token(self, tok):
        self._out = ""

    def finalize(self):
        self.finalized = True
        self._out = self._tail

    @property
    def last_segment(self):
        return self._out


def test_u8_adapter_flushes_tail_on_max_tokens_exhaustion():
    """The inner runner exhausts WITHOUT an EOS (max_tokens): the adapter must
    finalize the detokenizer and flush the buffered tail as a TEXT-ONLY frame
    tagged finish_reason="length" (pre-fix the tail was silently lost)."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter

    detok = _HoldingDetok(" tail!")
    tok = SimpleNamespace(eos_token_ids=[9], decode=lambda ids: "x", detokenizer=detok)
    frames = list(
        _pld_response_adapter(
            iter([(5, None, False), (6, None, False)]), tok, label="T"
        )
    )
    assert detok.finalized
    # Two token frames + the text-only flush frame.
    assert [f.token for f in frames] == [5, 6, None]
    assert frames[-1].text == " tail!"
    assert frames[-1].finish_reason == "length"
    # Token ids were all delivered on their own frames (accounting contract).
    assert "".join(f.text for f in frames) == " tail!"


def test_u8_adapter_no_flush_frame_without_buffered_tail():
    """Bare exhaustion with nothing buffered emits NO extra frame — an early
    (corruption/cancel) termination must not be mislabeled "length" here; the
    engine's max_tokens heuristic owns that case."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter

    tok = SimpleNamespace(eos_token_ids=[9], decode=lambda ids: "x")
    frames = list(
        _pld_response_adapter(
            iter([(5, None, False), (6, None, False)]), tok, label="T"
        )
    )
    assert [f.token for f in frames] == [5, 6]


def test_u8_adapter_eos_frame_still_finalizes_and_reports_stop():
    """EOS path (already finalizing pre-fix) keeps the tail flush and now tags
    the terminal frame finish_reason="stop" (U7)."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter

    detok = _HoldingDetok("END")
    tok = SimpleNamespace(eos_token_ids=[9], decode=lambda ids: "x", detokenizer=detok)
    frames = list(
        _pld_response_adapter(
            iter([(5, None, False), (9, None, False)]), tok, label="T"
        )
    )
    assert [f.token for f in frames] == [5, 9]
    assert frames[-1].text == "END"
    assert frames[-1].finish_reason == "stop"


def test_u8_engine_delivers_adapter_tail_text(monkeypatch):
    """ENGINE-LEVEL: the adapter's flush frame reaches the stream (text) and
    is NOT recorded as a token (token_ids unchanged by the flush)."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step
    from test_qwen_mtp import MockArrays, MockOps

    next_map = lambda t: (t + 1) % 50
    stored = [1, 2, 3, 4, 90]
    suffix = [40, 20, 21]
    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    import mlx.core as mx
    detok = _HoldingDetok("~tail")
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x", eos_token_ids=[], detokenizer=detok,
    )
    cs.mtp_last_hidden = mx.array([[[90.0]]])
    cs.mtp_hidden_offset = len(stored)

    def fake_step(prompt, model, head, **kwargs):
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)
    chunks = _drive_locked(eng, messages, max_tokens=4)

    # The flushed tail text reaches the stream...
    assert "".join(c.text for c in chunks).endswith("~tail")
    # ...but is NOT a recorded token: exactly the 4 generated ids landed.
    assert cs.token_ids == stored + suffix + [22, 23, 24, 25]
    assert chunks[-1].finish_reason == "length"


# ---------------------------------------------------------------------------
# U9 — structured output suppresses thinking (stream == non-stream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u9_stream_structured_json_lands_in_content():
    """STREAM + thinking model + response_format: the JSON must arrive as
    content deltas (pre-fix the active router put ALL of it in
    reasoning_content), and the engine must be asked with thinking=False."""
    engine = _StreamStubEngine(
        ['{"answer"', ': 42}'], enable_thinking=True,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        response_format=ResponseFormat(type="json_object"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _joined_delta(events, "content") == '{"answer": 42}'
    assert _joined_delta(events, "reasoning_content") == ""
    assert engine.captured_kwargs.get("thinking") is False
    assert _final_finish_reason(events) == "stop"


@pytest.mark.asyncio
async def test_u9_stream_thinking_unchanged_without_response_format():
    """Guard: without response_format the thinking routing is untouched."""
    engine = _StreamStubEngine(
        ["reasoning...", "</think>", "answer"], enable_thinking=True,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _joined_delta(events, "reasoning_content") == "reasoning..."
    assert _joined_delta(events, "content") == "answer"
    assert engine.captured_kwargs.get("thinking") is True


def test_u9_sync_structured_suppresses_thinking_flag():
    """NON-STREAM: the engine is asked with thinking=False for structured
    requests (pre-fix it received thinking=True)."""
    captured = {}

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=True)

        def complete(self, messages, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content='{"a": 1}', thinking=None, tool_calls=None,
                finish_reason="stop", prompt_tokens=1, completion_tokens=1,
                cache_info=None,
            )

        def update_session_messages(self, *a, **k):
            pass

    req = ChatCompletionRequest(
        model="m", stream=False, thinking=True,
        response_format=ResponseFormat(type="json_object"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = _sync_completion(req, _Eng())
    assert captured.get("thinking") is False
    assert resp.choices[0].message.content == '{"a": 1}'


def test_u9_engine_structured_renders_thinking_off(monkeypatch):
    """ENGINE: a structured request on a thinking=True call renders the
    prompt WITHOUT the thinking opener (suffix built thinking=False) and
    stamps the session contract with thinking=False. Pre-fix the request
    kept thinking=True: the fingerprint mismatched the stored contract and
    the drive took a MISS into the (unstubbed) tokenizer."""
    eng, cs, messages, stored = _plain_engine([(101, '{"a": 1}')], monkeypatch)
    seen_thinking: list = []
    orig_suffix = eng._suffix_tokens
    eng._suffix_tokens = lambda msgs, thinking=True: (
        seen_thinking.append(thinking) or orig_suffix(msgs, thinking=thinking)
    )
    chunks = _drive_locked(
        eng, messages, max_tokens=8, thinking=True,
        response_format=SimpleNamespace(type="json_object", json_schema=None),
    )
    assert chunks[-1].finish_reason == "stop"
    assert seen_thinking == [False]
    sess = eng._sessions["s"]
    assert sess.thinking is False


# ---------------------------------------------------------------------------
# U11 — legal tool-name charset + fail-closed passthrough
# ---------------------------------------------------------------------------


def test_u11_chatml_dashed_dotted_names_parse():
    text = (
        "<tool_call><function=mcp.server-tool_name>"
        "<parameter=user-id>42</parameter>"
        "<parameter=path.name>\"a/b\"</parameter>"
        "</function></tool_call>"
    )
    content, calls = parse_tool_calls(text, model_family="chatml")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "mcp.server-tool_name"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"user-id": 42, "path.name": "a/b"}
    assert content == ""


def test_u11_gemma4_dashed_name_parses():
    text = '<|tool_call>call:web-search.run{query:<|"|>hi<|"|>}<tool_call|>'
    _, calls = parse_tool_calls(text, model_family="gemma4")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web-search.run"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "hi"}


@pytest.mark.parametrize("family,buf,expected", [
    ("chatml", "<function=mcp.server-tool_name><parameter=q>", "mcp.server-tool_name"),
    ("gemma4", "call:web-search.run{", "web-search.run"),
    ("glm", "<function=data.load-csv>", "data.load-csv"),
])
def test_u11_try_extract_widened_names(family, buf, expected):
    assert try_extract_tool_name(buf, family) == expected


def test_u11_unparseable_block_passes_through_raw():
    """A start marker whose block never parses must NOT vanish: the full raw
    text is returned as content (pre-fix everything from the marker on was
    silently deleted)."""
    text = "before <tool_call>garbage that is not a call</tool_call> after"
    content, calls = parse_tool_calls(text, model_family="chatml")
    assert calls == []
    assert content == text  # raw passthrough, nothing deleted


def test_u11_gemma4_unparseable_block_passes_through_raw():
    text = "pre <|tool_call>not-a-call<tool_call|> post"
    content, calls = parse_tool_calls(text, model_family="gemma4")
    assert calls == []
    assert content == text


def test_u11_r2_chatml_malformed_block_preserved_next_to_good():
    """codex finding 3 repro: with round 1's all-or-nothing passthrough, one
    parseable block made the content get cut at the FIRST marker — the
    malformed sibling block, ' mid ', and ' tail' VANISHED (while streaming
    passed them through). Per-block preservation keeps them, in order."""
    text = (
        "pre <tool_call><function=good><parameter=x>1</parameter>"
        "</function></tool_call> mid <tool_call>garbage</tool_call> tail"
    )
    content, calls = parse_tool_calls(text, model_family="chatml")
    assert [c["function"]["name"] for c in calls] == ["good"]
    assert content == "pre  mid <tool_call>garbage</tool_call> tail"


def test_u11_r2_glm_malformed_block_preserved_next_to_good():
    text = (
        "pre <tool_call>good<arg_key>x</arg_key><arg_value>1</arg_value>"
        "</tool_call> mid <tool_call></tool_call> tail"
    )
    content, calls = parse_tool_calls(text, model_family="glm")
    assert [c["function"]["name"] for c in calls] == ["good"]
    assert content == "pre  mid <tool_call></tool_call> tail"


def test_u11_r2_gemma4_malformed_block_preserved_next_to_good():
    text = (
        'pre <|tool_call>call:good{x:1}<tool_call|>'
        ' mid <|tool_call>garbage<tool_call|> tail'
    )
    content, calls = parse_tool_calls(text, model_family="gemma4")
    assert [c["function"]["name"] for c in calls] == ["good"]
    assert content == "pre  mid <|tool_call>garbage<tool_call|> tail"


def test_u11_r2_good_bad_good_ordering_chatml():
    """Ordinary text and the malformed middle block keep their original
    order; both good blocks parse."""
    good1 = "<tool_call><function=a></function></tool_call>"
    bad = "<tool_call>nope</tool_call>"
    good2 = "<tool_call><function=b></function></tool_call>"
    text = f"x {good1} y {bad} z {good2} w"
    content, calls = parse_tool_calls(text, model_family="chatml")
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert content == "x  y <tool_call>nope</tool_call> z  w"


@pytest.mark.parametrize("family,good,expected_names", [
    (
        "chatml",
        "<tool_call><function=a></function></tool_call>"
        "<tool_call><function=b></function></tool_call>",
        ["a", "b"],
    ),
    (
        "glm",
        "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value></tool_call>",
        ["a", "b"],
    ),
    (
        "gemma4",
        '<|tool_call>call:a{}<tool_call|><|tool_call>call:b{}<tool_call|>',
        ["a", "b"],
    ),
])
def test_u11_r2_all_good_blocks_consume_fully(family, good, expected_names):
    content, calls = parse_tool_calls("go " + good, model_family=family)
    assert [c["function"]["name"] for c in calls] == expected_names
    assert content == "go"


@pytest.mark.parametrize("family,text", [
    ("chatml", "a <tool_call>junk</tool_call> b <tool_call>junk2</tool_call> c"),
    ("glm", "a <tool_call></tool_call> b <tool_call>\n \n</tool_call> c"),
    ("gemma4", "a <|tool_call>junk<tool_call|> b <|tool_call>junk2<tool_call|> c"),
])
def test_u11_r2_all_bad_blocks_raw_passthrough(family, text):
    """No call parses → byte-identical raw passthrough (round-1 rule kept)."""
    content, calls = parse_tool_calls(text, model_family=family)
    assert calls == []
    assert content == text


def test_u11_r2_unclosed_trailing_block_retained_next_to_good_chatml():
    """A good block plus an UNCLOSED trailing block: round 1 deleted the
    trailing fragment (content cut at the first marker); now it is retained
    — matching the streaming FSM's end-of-stream raw flush."""
    text = (
        "go <tool_call><function=a></function></tool_call>"
        " then <tool_call><function=trunca"
    )
    content, calls = parse_tool_calls(text, model_family="chatml")
    assert [c["function"]["name"] for c in calls] == ["a"]
    assert content == "go  then <tool_call><function=trunca"


@pytest.mark.asyncio
async def test_u11_stream_dashed_name_emits_tool_call():
    """STREAM: a dashed/dotted tool name must produce the OpenAI chunk pair
    (pre-fix the name never matched and the whole block vanished)."""
    engine = _StreamStubEngine(
        ["<tool_call><function=mcp.server-tool_name>",
         "<parameter=q>hi</parameter>",
         "</function></tool_call>"],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="mcp.server-tool_name", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    tc_chunks = [
        tc
        for ev in events
        for c in ev.get("choices", [])
        for tc in (c["delta"].get("tool_calls") or [])
    ]
    assert tc_chunks and tc_chunks[0]["function"]["name"] == "mcp.server-tool_name"
    assert _final_finish_reason(events) == "tool_calls"
    # Nothing leaked into content.
    assert "<tool_call>" not in _joined_delta(events, "content")


@pytest.mark.asyncio
async def test_u11_stream_unparseable_block_passthrough():
    """STREAM: a closed block that fails to parse is passed through as raw
    content (pre-fix it silently vanished from the stream)."""
    engine = _StreamStubEngine(
        ["before ", "<tool_call>garbage", "</tool_call>", " after"],
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    content = _joined_delta(events, "content")
    assert "before " in content
    assert "<tool_call>garbage</tool_call>" in content
    assert " after" in content
    assert _final_finish_reason(events) == "stop"


@pytest.mark.asyncio
async def test_u11_stream_truncated_block_passthrough():
    """STREAM: generation ends mid-block and the partial block does not parse
    — flush the raw text instead of dropping it."""
    engine = _StreamStubEngine(
        ["answer: ", "<tool_call>trunca"],
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    content = _joined_delta(events, "content")
    assert "answer: " in content
    assert "<tool_call>trunca" in content


# ---------------------------------------------------------------------------
# U12 — tool calls inside <think> are rehearsals, not calls
# ---------------------------------------------------------------------------

_TOOLS = [{"type": "function", "function": {"name": "get_x", "parameters": {}}}]


def test_u12_engine_ignores_rehearsed_call_in_thinking(monkeypatch):
    """ENGINE: a complete tool-call block inside the thinking region must not
    be recorded/reported as a real call (pre-fix: finish_reason="tool_calls"
    and a persisted tool_calls entry)."""
    script = [
        (101, "I could run <tool_call><function=get_x></function></tool_call> here"),
        (102, "</think>plain answer"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    _restamp_contract(
        eng, thinking=True,
        tools=type(eng)._canonical_tools(_TOOLS),
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, thinking=True, tools=_TOOLS)
    assert chunks[-1].finish_reason == "stop"
    sess = eng._sessions["s"]
    assert "tool_calls" not in sess.messages[-1]
    # The rehearsal stays intact inside the stored thinking.
    assert "<tool_call>" in sess.messages[-1]["content"]


def test_u12_engine_parses_real_call_after_thinking(monkeypatch):
    """ENGINE: a real call in the CONTENT region still parses, and the stored
    content is stripped from the CONTENT-region block onward (the rehearsed
    marker inside thinking must not truncate the stored thinking)."""
    script = [
        (101, "rehearse <tool_call><function=other></function></tool_call> ok"),
        (102, "</think>go "),
        (103, "<tool_call><function=get_x><parameter=q>1</parameter></function></tool_call>"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    _restamp_contract(
        eng, thinking=True,
        tools=type(eng)._canonical_tools(_TOOLS),
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, thinking=True, tools=_TOOLS)
    assert chunks[-1].finish_reason == "tool_calls"
    sess = eng._sessions["s"]
    calls = sess.messages[-1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_x"]
    content = sess.messages[-1]["content"]
    # Thinking (with its rehearsal) preserved; content-region XML stripped.
    assert "rehearse <tool_call><function=other>" in content
    assert content.endswith("</think>go")


def test_u12_match_helpers_ignore_thinking_region_xml():
    """MATCH SIDE: XML inside the thinking region is neither an extractable
    call nor a residual marker (a rehearsal must not force every later turn
    into a one-sided/residual mismatch against the client's clean resend)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    rehearsal = (
        "<think>maybe <tool_call><function=get_x></function></tool_call> "
        "hmm</think>the actual answer"
    )
    assert MLXEngine._content_xml_calls_for_match(rehearsal, "chatml") == []
    assert MLXEngine._has_residual_tool_markers(rehearsal, "chatml") is False
    # A REAL call in the content region still extracts.
    real = (
        "<think>plan</think>"
        "<tool_call><function=get_x></function></tool_call>"
    )
    assert MLXEngine._content_xml_calls_for_match(real, "chatml") == [
        ("get_x", "{}"),
    ]
    # And a structured message whose content only rehearses in thinking is
    # NOT a self-conflict.
    msg = {
        "tool_calls": [
            {"function": {"name": "get_x", "arguments": "{}"}}
        ]
    }
    assert MLXEngine._structured_content_call_conflict(
        msg, rehearsal, "chatml"
    ) is False


# ---------------------------------------------------------------------------
# U12 round 2 — gemma4 multi-cycle: the content channel is the UNION of ALL
# content segments (thought → content → thought → content), not just the
# suffix after the last close marker.
# ---------------------------------------------------------------------------

_G4_CALL = '<|tool_call>call:get_x{q:<|"|>1<|"|>}<tool_call|>'
_G4_MULTI_CYCLE = (
    "<|channel>thought\nplan one<channel|>"
    "Let me check " + _G4_CALL +
    "<|channel>thought\nplan two<channel|>done"
)


def test_u12_r2_gemma4_multicycle_engine_extracts_and_strips(monkeypatch):
    """codex finding 4 repro: thought → content with a call → thought →
    content. The batch split finds the call, but round 1's last-close-marker
    reduction made the stored-content strip MISS it — the session stored
    BOTH the raw XML and the structured tool_calls (a rebuild double-
    renders). Now the call is extracted once, its XML stripped from the
    earlier content cycle, and both thinking cycles survive verbatim."""
    script = [
        (101, "<|channel>thought\nplan one<channel|>Let me check "),
        (102, _G4_CALL),
        (103, "<|channel>thought\nplan two<channel|>done"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    eng.model_family = "gemma4"
    _restamp_contract(
        eng, thinking=False, tools=type(eng)._canonical_tools(_TOOLS),
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, tools=_TOOLS)
    assert chunks[-1].finish_reason == "tool_calls"
    sess = eng._sessions["s"]
    calls = sess.messages[-1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_x"]
    content = sess.messages[-1]["content"]
    # Extracted exactly once, raw XML stripped (no double representation)...
    assert "<|tool_call>" not in content
    # ...while BOTH thinking cycles and the surrounding content survive.
    assert "thought\nplan one" in content
    assert "thought\nplan two" in content
    assert "Let me check" in content
    assert content.endswith("done")


def test_u12_r2_match_helpers_see_earlier_cycle_call():
    """MATCH SIDE (legacy stored shape — raw XML still in content): the call
    in the EARLIER content cycle must be extractable, not a residual, and
    must satisfy (not conflict with) a matching structured list. Round 1's
    rfind reduction hid it from all three helpers."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    raw = _G4_MULTI_CYCLE
    assert MLXEngine._content_xml_calls_for_match(raw, "gemma4") == [
        ("get_x", '{"q": "1"}'),
    ]
    assert MLXEngine._has_residual_tool_markers(raw, "gemma4") is False
    same = {
        "tool_calls": [
            {"function": {"name": "get_x", "arguments": '{"q": "1"}'}}
        ]
    }
    assert MLXEngine._structured_content_call_conflict(
        same, raw, "gemma4"
    ) is False
    other = {
        "tool_calls": [
            {"function": {"name": "other", "arguments": "{}"}}
        ]
    }
    assert MLXEngine._structured_content_call_conflict(
        other, raw, "gemma4"
    ) is True


def test_u12_r2_gemma4_rehearsal_inside_thought_still_ignored():
    """The U12 rehearsal rule must keep holding: a call rehearsed INSIDE a
    thought span is neither extractable nor a residual marker."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    raw = (
        "<|channel>thought\nmaybe " + _G4_CALL + " hmm<channel|>plain answer"
    )
    assert MLXEngine._content_xml_calls_for_match(raw, "gemma4") == []
    assert MLXEngine._has_residual_tool_markers(raw, "gemma4") is False


def test_u12_r2_content_segments_union_shapes():
    """content_segments mirrors split_thinking_and_content's channel
    semantics positionally: joined spans == the split's content channel."""
    from mlx_soloheaven.engine.tool_parser import (
        content_segments,
        split_thinking_and_content,
    )

    for family, text in [
        ("gemma4", _G4_MULTI_CYCLE),
        ("gemma4", "no markers at all"),
        ("gemma4", "leftover reasoning<channel|>real content"),
        ("chatml", "<think>plan</think>the answer"),
        ("chatml", "plain answer"),
    ]:
        union = "".join(
            text[s:e] for s, e in content_segments(text, family)
        )
        _, split_content = split_thinking_and_content(text, family)
        # The positional union preserves raw bytes (no marker scrubbing /
        # strip()), so compare modulo whitespace on the split side.
        assert union.strip() == split_content.strip() or (
            split_content.strip() in union
        )


# ---------------------------------------------------------------------------
# U24 — stop sequences (engine-side, holdback across token boundaries)
# ---------------------------------------------------------------------------


def test_u24_stop_sequence_basic(monkeypatch):
    """Stop text and everything after it is withheld; finish_reason="stop";
    generation terminates early (the runner is closed at the hit).

    Round 2 contract (finding 1, case a): the stop starts exactly at a token
    boundary and every layer is trimmable — the recorded ids AND the cache
    are trimmed back to the visible text, so messages ↔ ids ↔ cache agree
    (the round-1 contract kept the withheld " STOP" token in the reusable KV
    while the stored content said otherwise — a contract violation)."""
    script = [
        (101, "Hello "), (102, "world"), (103, " STOP"), (104, " never"),
        (105, " ever"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    emitted = "".join(c.text for c in chunks)
    assert emitted == "Hello world"
    # Early termination: frames 104/105 were never pulled.
    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [101, 102, 103]
    # Commit-or-invalidate: the withheld stop token (103) is trimmed from
    # the recorded ids and the cache — the reusable state holds EXACTLY the
    # persisted "Hello world".
    assert cs.token_ids == stored + [52, 99] + [101, 102]
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)
    # Persisted content excludes the stop text.
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


def test_u24_stop_sequence_split_across_tokens(monkeypatch):
    """A stop sequence split across token boundaries is still caught, and the
    partial prefix is never leaked to the stream."""
    script = [(101, "ab"), (102, "S"), (103, "TO"), (104, "P rest"), (105, "x")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=["STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "ab"
    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [101, 102, 103, 104]


def test_u24_stop_holdback_flushed_when_no_match(monkeypatch):
    """A partial-match tail that never completes is real output: it must be
    delivered (flush frame) and persisted; the terminal is not "stop"-by-
    sequence (here: length, the script exhausts max_tokens)."""
    script = [(101, "value: "), (102, "ST")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=2, stop_sequences=["STOP"])
    assert "".join(c.text for c in chunks) == "value: ST"
    assert chunks[-1].finish_reason == "length"
    assert eng._sessions["s"].messages[-1]["content"] == "value: ST"
    # The flush frame is text-only (its ids were recorded on their frames).
    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames if c.token] == [101, 102]
    assert cs.token_ids == stored + [52, 99] + [101, 102]


def test_u24_stop_on_mtp_path_invalidates_hybrid_cache(monkeypatch):
    """QwenMTP path (round 2, finding 1 case b via untrimmable layers): the
    stop break closes the runner (natural finalize — the settle contract),
    then the reconcile must trim the withheld stop tokens back out of the
    cache. The hybrid cache (ArraysCache layers, MTP head) cannot be trimmed
    verifiably, so the session cache is INVALIDATED fail-closed — including
    the on_finalize resume stash re-set during the close — and the next turn
    takes an honest MISS. (The round-1 contract kept the stop tokens live in
    the reusable KV while the persisted content excluded them.)"""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step
    from test_qwen_mtp import MockArrays, MockOps, _assert_no_desynced_cache

    import mlx.core as mx

    next_map = lambda t: (t + 1) % 50
    stored = [1, 2, 3, 4, 90]
    suffix = [40, 20, 21]
    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    eng.tokenizer = SimpleNamespace(decode=lambda ids: "x", eos_token_ids=[])
    cs.mtp_last_hidden = mx.array([[[90.0]]])
    cs.mtp_hidden_offset = len(stored)

    def fake_step(prompt, model, head, **kwargs):
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)
    # Each token decodes to "x": the stop completes on the 2nd token.
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=["xx"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == ""
    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [22, 23]
    # No stop token survives in reusable state: the hybrid cache cannot be
    # trimmed back to the (empty) visible text, so it was invalidated —
    # cache, token_ids AND the MTP finalize-hidden resume stash.
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)
    # The truncated (empty) turn was still persisted; the dead cache forces
    # an honest MISS on the next request (the HIT gate needs a live cache).
    assert eng._sessions["s"].messages[-1]["content"] == ""


def test_u24_multiple_stop_sequences_earliest_wins(monkeypatch):
    script = [(101, "abc DEF ghi"), (102, "never")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(
        eng, messages, max_tokens=8, stop_sequences=["ghi", "DEF"],
    )
    assert "".join(c.text for c in chunks) == "abc "
    assert chunks[-1].finish_reason == "stop"


def test_u24_stop_next_hit_reuses_trimmed_cache(monkeypatch):
    """Round 2, finding 1 case (a) end-to-end: after a boundary-aligned stop
    hit on a fully trimmable cache, the NEXT same-session turn takes a HIT
    whose model context extends the TRIMMED ids — the withheld stop token
    never reaches the model (round 1 silently kept ' STOP' in the context)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "Hello "), (102, "world"), (103, " STOP"), (104, " x")],
        [(111, "next")],
    ])

    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert cs.token_ids == stored + [52, 99] + [101, 102]

    # Turn 2: follow-up on the same session — HIT, suffix spliced onto the
    # trimmed ids; byte-consistent with the stored "Hello world".
    messages2 = list(eng._sessions["s"].messages) + [
        {"role": "user", "content": "u3"},
    ]
    chunks2 = _drive_locked(eng, messages2, max_tokens=8)
    assert chunks2[-1].finish_reason == "stop"
    # The engine fed ONLY the new suffix (prefix reuse on the trimmed ids).
    assert prompts_seen[1] == [52, 99]
    assert cs.token_ids == stored + [52, 99] + [101, 102] + [52, 99] + [111]
    assert 103 not in cs.token_ids  # the ' STOP' token id
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


def test_u24_stop_mid_token_invalidates_cache(monkeypatch):
    """Round 2, finding 1 case (b): the stop shares a token with visible
    text ("rld STOP" is ONE token) — no boundary trim can remove the stop
    text from the KV, so the session cache is invalidated fail-closed while
    the truncated text is still persisted; the next turn must MISS (the HIT
    gate requires a live cache), never resurrect ' STOP' into the context."""
    script = [(101, "Hello wo"), (102, "rld STOP"), (103, "never")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    # Case (b): everything reusable is gone...
    assert cs.cache is None
    assert cs.token_ids is None
    # ...but the truncated turn is committed (messages hold the visible text
    # only) and the dead cache forces the honest MISS next turn.
    sess = eng._sessions["s"]
    assert sess.messages[-1]["content"] == "Hello world"
    assert sess.cache_state.cache is None


def test_u24_stop_boundary_aligned_untrimmable_invalidates(monkeypatch):
    """Round 2, finding 1 case (b) via layer type: the match IS boundary-
    aligned but one cache layer is untrimmable (recurrent ArraysCache — the
    production Qwen hybrid shape) — the reconcile's trimmability check must
    invalidate rather than leave the stop tokens alive in the KV."""
    from test_qwen_mtp import MockArrays

    stored = [1, 2, 3, 4, 5]
    cache = [MockArrays(), MockKV(), MockKV(), MockKV(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "Hello "), (102, "world"), (103, " STOP")],
    ])
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    assert cs.cache is None
    assert cs.token_ids is None
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


@pytest.mark.parametrize("script", [
    [(101, "aba")],
    [(101, "ab"), (102, "a")],
    [(101, "a"), (102, "b"), (103, "a")],
])
def test_u24_multi_stop_chunk_permutations_deterministic(script, monkeypatch):
    """Round 2, finding 2 (codex repro): stops ["b","aba"] over the same
    decoded text "aba" must produce IDENTICAL output, finish_reason and
    final cache state for EVERY chunk segmentation. Round 1 accepted the
    completed "b" in frames ["ab","a"] (output "a") but "aba" in the single
    frame (output "") — earliest-START semantics with viable-partial
    holdback makes "aba" win everywhere."""
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(
        eng, messages, max_tokens=8, stop_sequences=["b", "aba"],
    )
    assert "".join(c.text for c in chunks) == ""
    assert chunks[-1].finish_reason == "stop"
    # Identical persisted state too: the whole output was stop territory
    # (match at position 0 → boundary-aligned trim back to the prompt).
    assert eng._sessions["s"].messages[-1]["content"] == ""
    assert cs.token_ids == stored + [52, 99]
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


def test_u24_multi_stop_earlier_partial_dies_completed_match_wins(monkeypatch):
    """Round 2, finding 2: the earlier viable partial ("ab" of "aba") DIES
    on the next frame ("x") — the held completed match ("b" at 1) then wins:
    output "a", finish "stop", for any segmentation."""
    script = [(101, "ab"), (102, "x")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(
        eng, messages, max_tokens=8, stop_sequences=["b", "aba"],
    )
    assert "".join(c.text for c in chunks) == "a"
    assert chunks[-1].finish_reason == "stop"
    # "b" starts mid-token (token 101 decoded "ab") → case (b) invalidation.
    assert cs.cache is None
    assert eng._sessions["s"].messages[-1]["content"] == "a"


def test_u24_stream_end_held_completed_match_accepted(monkeypatch):
    """Round 2, finding 2: the stream ends while a completed match is held
    back by a still-viable earlier partial ("ab" could become "aba"). At
    stream end partials can never complete — the completed "b" must be
    accepted (finish "stop", output "a"), identical to what any other
    segmentation of the same text would have produced."""
    script = [(101, "ab")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(
        eng, messages, max_tokens=1, stop_sequences=["b", "aba"],
    )
    assert "".join(c.text for c in chunks) == "a"
    assert chunks[-1].finish_reason == "stop"
    assert eng._sessions["s"].messages[-1]["content"] == "a"
    # Mid-token match → case (b) invalidation.
    assert cs.cache is None


@pytest.mark.asyncio
async def test_u24_stream_stop_param_threaded_to_engine():
    """API STREAM: request.stop reaches the engine call."""
    engine = _StreamStubEngine(["hi"])
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        stop=["\n\n", "END"],
        messages=[ChatMessage(role="user", content="hi")],
    )
    await _collect_sse(engine, req)
    assert engine.captured_kwargs.get("stop") == ["\n\n", "END"]


def test_u24_sync_stop_param_threaded_to_engine():
    captured = {}

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def complete(self, messages, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content="x", thinking=None, tool_calls=None,
                finish_reason="stop", prompt_tokens=1, completion_tokens=1,
                cache_info=None,
            )

        def update_session_messages(self, *a, **k):
            pass

    req = ChatCompletionRequest(
        model="m", stream=False, thinking=False, stop="END",
        messages=[ChatMessage(role="user", content="hi")],
    )
    _sync_completion(req, _Eng())
    assert captured.get("stop") == "END"


def test_u24_stop_validation_rejects_more_than_four():
    """OpenAI allows at most 4 stop sequences — a 5th is a 400."""
    from mlx_soloheaven.api import openai_compat

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

    old_engines = openai_compat._engines
    old_default = openai_compat._default_engine
    try:
        eng = _Eng()
        openai_compat.set_engines({"m": eng}, eng)
        req = ChatCompletionRequest(
            model="m", stream=False,
            stop=["a", "b", "c", "d", "e"],
            messages=[ChatMessage(role="user", content="hi")],
        )
        resp = asyncio.run(openai_compat.chat_completions(req))
    finally:
        openai_compat.set_engines(old_engines, old_default)
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"


def test_u24_stop_validation_rejects_empty_string():
    from mlx_soloheaven.api import openai_compat

    class _Eng:
        model_family = "chatml"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

    old_engines = openai_compat._engines
    old_default = openai_compat._default_engine
    try:
        eng = _Eng()
        openai_compat.set_engines({"m": eng}, eng)
        req = ChatCompletionRequest(
            model="m", stream=False, stop="",
            messages=[ChatMessage(role="user", content="hi")],
        )
        # "" is falsy — treated as absent by pydantic? No: it is a string.
        resp = asyncio.run(openai_compat.chat_completions(req))
    finally:
        openai_compat.set_engines(old_engines, old_default)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# U25 — delete-last: DB drives, engine follows (counts always agree)
# ---------------------------------------------------------------------------

_COMPACTION_TEXT = (
    "The conversation history before this point was compacted into the "
    "summary above.\n<!-- keep_recent:2 -->"
)


class _DeleteStubDB:
    """DB stub for the U25 round-2 contract: rows carry ids, deletion is
    id-based and atomic, and reads always reflect the CURRENT row set.
    Round 6 (codex round 5, finding 1): deletion is the ONE-transaction
    ``delete_last_turn_tx`` shape — snapshot + walker callback + delete
    under a single write lock. ``mutate_before_delete`` simulates a
    concurrent writer's transaction committing BEFORE the delete
    transaction acquires that lock (BEGIN IMMEDIATE serializes it there),
    so the walker sees the writer's rows in the tx's own snapshot."""

    def __init__(self, messages, system_prompt="sys"):
        self._messages = [
            {**m, "id": m.get("id", f"m{i}")} for i, m in enumerate(messages)
        ]
        self._system_prompt = system_prompt
        self.deleted_ids: list = []
        self.delete_calls = 0
        self.mutate_before_delete = None  # callable(stub) | None

    def ids(self):
        return [m["id"] for m in self._messages]

    async def get_messages(self, session_id, limit=None):
        return [dict(m) for m in self._messages]

    async def get_session(self, session_id):
        return {"id": session_id, "system_prompt": self._system_prompt}

    async def delete_last_turn_tx(self, session_id, compute_delete_ids):
        self.delete_calls += 1
        if self.mutate_before_delete is not None:
            # The concurrent writer's transaction committed BEFORE the
            # write lock — its rows are part of the tx snapshot below.
            mutate, self.mutate_before_delete = self.mutate_before_delete, None
            mutate(self)
        snapshot = [dict(m) for m in self._messages]
        ids = list(compute_delete_ids(snapshot) or [])
        id_set = set(ids)
        self._messages = [m for m in self._messages if m["id"] not in id_set]
        self.deleted_ids.extend(ids)
        return ids, [m for m in snapshot if m["id"] not in id_set]


class _DeleteStubEngine:
    model_id = "stub-model"
    model_family = "chatml"

    def __init__(self):
        self.truncate_calls: list = []
        self.compact_calls: list = []

    def truncate_session(self, session_id, n):
        self.truncate_calls.append(n)
        return {"truncated_to": n}

    def compact_session(self, session_id, messages):
        self.compact_calls.append(messages)
        return {"method": "build", "cached_tokens": 0, "messages": len(messages)}


def _delete_last_client(monkeypatch, stub_db, engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    monkeypatch.setattr(chat_mod, "db", stub_db)
    monkeypatch.setattr(chat_mod, "engine", engine)
    monkeypatch.setattr(chat_mod, "_engines", {})

    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(chat_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def test_u25_compaction_delete_removes_one_and_rebuilds(monkeypatch):
    """THE FINDING: deleting a trailing compaction summary removed 1 DB row
    but the engine view was built from messages[:-2] — trimming a message
    that was never deleted. Now: exactly 1 DB delete, and the engine is
    REBUILT from the actual remaining rows (compaction boundary changed, a
    count-slice of the engine's compacted view cannot express it)."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": _COMPACTION_TEXT},
    ]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    assert len(stub_db.deleted_ids) == 1  # round-1 shape deleted 1 but trimmed 2
    assert r.json()["remaining_messages"] == 2
    # Engine followed the DB: rebuilt from the ACTUAL remaining rows.
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert eng.compact_calls[0] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_u25_normal_pair_delete_counts_agree(monkeypatch):
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    assert stub_db.deleted_ids == ["m2", "m3"]
    assert stub_db.delete_calls == 1  # ONE atomic transaction, not N
    assert r.json()["remaining_messages"] == 2
    # system + 2 remaining rows.
    assert eng.truncate_calls == [3]
    assert eng.compact_calls == []


def test_u25_tool_chain_deletes_whole_turn(monkeypatch):
    """A tool-call turn spans user → assistant(tool_calls) → tool →
    assistant: deleting only the trailing 2 rows (old behavior) left a
    dangling tool_call. The whole turn is one delete unit."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
        {"role": "assistant", "content": "final"},
    ]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    assert stub_db.deleted_ids == ["m2", "m3", "m4", "m5"]
    assert stub_db.delete_calls == 1
    assert r.json()["remaining_messages"] == 2
    assert eng.truncate_calls == [3]  # system + q1 + a1


def test_u25_no_user_turn_400(monkeypatch):
    msgs = [{"role": "assistant", "content": "orphan"}]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 400
    assert stub_db.deleted_ids == []
    assert eng.truncate_calls == [] and eng.compact_calls == []


def test_u25_turn_never_crosses_compaction(monkeypatch):
    """A turn walk stops at a compaction summary: [compaction, assistant]
    has no deletable turn (the summary is not part of any turn)."""
    msgs = [
        {"role": "user", "content": _COMPACTION_TEXT},
        {"role": "assistant", "content": "post-compact note"},
    ]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 400
    assert stub_db.deleted_ids == []


def test_u25_r2_concurrent_append_survives_delete(monkeypatch):
    """codex finding 5 repro: a chat append lands between the handler's
    snapshot and the delete. Round 1 deleted 'whichever row is currently
    last' N times — the NEW user row got deleted and part of the old turn
    survived. Now: deletion targets the snapshot's row ids, the appended row
    survives, exactly the snapshot turn rows are gone, and the engine is
    REBUILT from the fresh post-delete rows (the count fast path is invalid
    once the DB moved)."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    stub_db = _DeleteStubDB(msgs)
    stub_db.mutate_before_delete = lambda db: db._messages.append(
        {"id": "new1", "role": "user", "content": "concurrent!"},
    )
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    # Exactly the snapshot turn rows are gone; the concurrent append lives.
    assert stub_db.deleted_ids == ["m2", "m3"]
    assert stub_db.ids() == ["m0", "m1", "new1"]
    assert r.json()["remaining_messages"] == 3
    # DB moved under us → no count-slice truncate; full rebuild from the
    # ACTUAL post-delete rows (including the concurrent append).
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert eng.compact_calls[0] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "concurrent!"},
    ]


def test_u25_r2_compaction_delete_with_concurrent_append(monkeypatch):
    """The worst case codex called out: deleting a trailing compaction
    summary while a concurrent user message lands. The summary row (and
    ONLY it) is deleted by id; the appended message survives and the engine
    rebuild sees it — never a rebuild missing rows that still exist in
    the DB."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": _COMPACTION_TEXT},
    ]
    stub_db = _DeleteStubDB(msgs)
    stub_db.mutate_before_delete = lambda db: db._messages.append(
        {"id": "new1", "role": "user", "content": "racing question"},
    )
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    assert stub_db.deleted_ids == ["m2"]  # the summary, by id
    assert stub_db.ids() == ["m0", "m1", "new1"]
    assert r.json()["remaining_messages"] == 3
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert eng.compact_calls[0] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "racing question"},
    ]


def test_u25_r2_concurrent_delete_falls_back_to_rebuild(monkeypatch):
    """Drift in the OTHER direction: a snapshot row disappeared between the
    snapshot and the delete (a concurrent regenerate/delete). Deletion by id
    is a no-op for missing rows; the fresh re-read drives a rebuild instead
    of a stale count-slice truncate."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    stub_db = _DeleteStubDB(msgs)
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    # Concurrent writer removes a1 between our snapshot and our delete.
    def _drop_a1(db):
        db._messages = [m for m in db._messages if m["id"] != "m1"]

    stub_db.mutate_before_delete = _drop_a1

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    assert stub_db.ids() == ["m0"]
    assert r.json()["remaining_messages"] == 1
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert eng.compact_calls[0] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
    ]


# ===========================================================================
# Round 4 (codex round 3 residuals) — R4F1..R4F5 + nits
# ===========================================================================
#
# R4F1 — stop-hit state leaked for OFFSET-OPAQUE caches (GLM CacheList):
#        offset discovery now recurses through CacheList-like containers,
#        and the stop-hit epilogue is a FAIL-CLOSED backstop — a stop hit
#        keeps the cache alive ONLY when its alignment is positively
#        verified (readable offset == len(trimmed ids)).
# R4F2 — delete-last racing an ACTIVE /chat generation: the assistant row
#        insert is conditional on the originating user row (one atomic
#        statement); parent gone → row skipped (WARNING), engine session
#        deleted (honest rebuild from DB next request).
# R4F3 — per-segment tool-XML stripping no longer right-strips INTERNAL
#        whitespace (segment-mode parse; one final top-level rstrip).
# R4F4 — content_segments()/the batch parse/the save strip follow the
#        streaming router (the authority): thinking inactive → whole text
#        is content (literal </think> is a quote); gemma4 orphan-close with
#        no prior thought-open is content.
# R4F5 — the streaming tool-call name delta is DEFERRED until the closed
#        block parses: a malformed block produces raw content only, never
#        a phantom tool_calls delta.


def _tool_call_deltas(events: list[dict]) -> list[dict]:
    return [
        tc
        for ev in events
        for c in ev.get("choices", [])
        for tc in (c["delta"].get("tool_calls") or [])
    ]


# ---------------------------------------------------------------------------
# R4F1 — offset-opaque caches: recursion + fail-closed stop backstop
# ---------------------------------------------------------------------------


class _FakeCacheList:
    """CacheList-like container (the GLM MoE/DSA per-layer layout): children
    live under ``.caches``, ``trim`` delegates, and the container itself has
    NO ``offset``."""

    def __init__(self, *children):
        self.caches = tuple(children)

    def trim(self, n):
        m = 0
        for c in self.caches:
            m = c.trim(n)
        return m


class _OpaqueLayer:
    """Fully offset-opaque cache layer: no offset, no trim, no children."""


def test_r4_f1_get_cache_offset_recurses_cachelist():
    """Offset discovery must see the KVCache offsets INSIDE a CacheList
    (pre-fix: top-level-only inspection read 0 for the whole cache)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    kv1, kv2 = MockKV(), MockKV()
    kv1.offset = 7
    kv2.offset = 7
    assert MLXEngine._get_cache_offset([_FakeCacheList(kv1, kv2)]) == 7
    # Leaf-only caches unchanged.
    plain = MockKV()
    plain.offset = 3
    assert MLXEngine._get_cache_offset([plain]) == 3
    assert MLXEngine._get_cache_offset([_OpaqueLayer()]) == 0


def test_r4_f1_stop_on_offset_opaque_cache_invalidates(monkeypatch):
    """codex repro: offset-opaque layout + boundary-aligned stop spanning
    >=2 tokens. The recorded ids are trimmed at the hit, but the cache
    offset reads 0, so the ahead/trim branch is skipped — pre-fix the cache
    stayed 'reusable' with the withheld stop tokens inside (keep_tokens was
    non-None, so the mid-token epilogue did not invalidate either). The
    fail-closed backstop must invalidate: alignment cannot be positively
    established."""
    stored = [1, 2, 3, 4, 5]
    cache = [_OpaqueLayer() for _ in range(4)]
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "Hello "), (102, "world"), (103, " ST"), (104, "OP"), (105, " x")],
    ])
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    # FAIL-CLOSED: unverifiable alignment → the cache must not survive.
    assert cs.cache is None
    assert cs.token_ids is None
    # The truncated turn is still committed; next turn takes an honest MISS.
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


def test_r4_f1_stop_on_cachelist_trims_and_survives(monkeypatch):
    """Recursion test: CacheList-wrapped KVCache offsets are now readable,
    so the NORMAL boundary-aligned trim path runs — the withheld stop token
    is trimmed out of every leaf and the cache survives, verified. Pre-fix
    the offset read 0: the trim was skipped and the leaves stayed AHEAD of
    the recorded ids (stop tokens alive in 'reusable' state)."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stored = [1, 2, 3, 4, 5]
    cache = [
        _FakeCacheList(MockKV(), MockKV()),
        _FakeCacheList(MockKV(), MockKV()),
    ]
    for layer in cache:
        for leaf in layer.caches:
            leaf.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])

    script = [(101, "Hello "), (102, "world"), (103, " STOP")]

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def _advance(n):
            for c in prompt_cache:
                for leaf in getattr(c, "caches", (c,)):
                    if hasattr(leaf, "offset"):
                        leaf.offset += n

        def gen():
            _advance(len(prompt))
            for tid, text in script:
                _advance(1)
                yield SimpleNamespace(
                    text=text, token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    # Genuinely aligned → the cache SURVIVES, trimmed to the visible text.
    assert cs.cache is not None
    assert cs.token_ids == stored + [52, 99] + [101, 102]
    for leaf in MLXEngine._flatten_cache_layers(cs.cache):
        assert leaf.offset == len(cs.token_ids)
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


def test_r4_nit_stop_bounds_pruned_late_hit_exact(monkeypatch):
    """NIT guard: _stop_tok_bounds is pruned past 4096 entries; a
    boundary-aligned stop hit LATE in a long stream must still resolve the
    exact absolute token boundary (the dropped-entries base offset) and
    trim cache/ids precisely."""
    n_body = 4600
    script = [(1000 + i, "a") for i in range(n_body)]
    script += [(9000, " ST"), (9001, "OP")]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    chunks = _drive_locked(
        eng, messages, max_tokens=5000, stop_sequences=[" STOP"],
    )
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "a" * n_body
    expected_ids = stored + [52, 99] + [1000 + i for i in range(n_body)]
    assert cs.token_ids == expected_ids
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


@pytest.mark.asyncio
async def test_r4_nit_delete_messages_by_ids_chunks_in_clause(tmp_path):
    """NIT: the IN clause is chunked (500/batch, ONE transaction) so
    low-SQLITE_MAX_VARIABLE_NUMBER builds survive large turn deletes."""
    from mlx_soloheaven.storage import database as real_db

    old_path = real_db._db_path
    try:
        real_db.set_db_path(str(tmp_path / "t.db"))
        await real_db.init_db()
        async with real_db.get_db() as conn:
            await conn.executemany(
                "INSERT INTO messages (id, session_id, role, content, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                [(f"m{i}", "s1", "user", "x", float(i)) for i in range(1200)],
            )
            await conn.commit()
        deleted = await real_db.delete_messages_by_ids(
            "s1", [f"m{i}" for i in range(1100)],
        )
        assert deleted == 1100
        remaining = await real_db.get_messages("s1")
        assert len(remaining) == 100
        assert remaining[0]["id"] == "m1100"
    finally:
        real_db.set_db_path(old_path)


# ---------------------------------------------------------------------------
# R4F2 — delete-last vs ACTIVE generation: no orphan assistant row
# ---------------------------------------------------------------------------


class _RaceStubDB:
    """DB stub with real row semantics for the finding-2 race: rows carry
    ids, the conditional insert honors parent existence, id-deletes remove
    rows."""

    def __init__(self, messages):
        self._n = 0
        self._messages = []
        for m in messages:
            self._append(m)

    def _append(self, m):
        row = {**m, "id": m.get("id", f"m{self._n}")}
        self._n += 1
        self._messages.append(row)
        return row

    def roles(self):
        return [m["role"] for m in self._messages]

    async def get_session(self, session_id):
        return {"id": session_id, "system_prompt": "sys",
                "context_window_limit": 100000}

    async def get_messages(self, session_id, limit=None):
        return [dict(m) for m in self._messages]

    async def add_message(self, session_id, role, **kw):
        return dict(self._append({"role": role, **kw}))

    async def add_message_if_parent_exists(self, session_id, role, *,
                                           parent_id, **kw):
        if not any(m["id"] == parent_id for m in self._messages):
            return None
        return dict(self._append({"role": role, **kw}))

    async def delete_messages_by_ids(self, session_id, ids):
        id_set = set(ids)
        before = len(self._messages)
        self._messages = [m for m in self._messages if m["id"] not in id_set]
        return before - len(self._messages)

    async def delete_last_turn_tx(self, session_id, compute_delete_ids):
        # Round 6 (codex round 5, finding 1): one-transaction delete shape.
        snapshot = [dict(m) for m in self._messages]
        ids = list(compute_delete_ids(snapshot) or [])
        id_set = set(ids)
        self._messages = [m for m in self._messages if m["id"] not in id_set]
        return ids, [m for m in snapshot if m["id"] not in id_set]

    async def get_session_total_tokens(self, session_id):
        return 0

    async def update_session_tokens(self, session_id, total):
        pass

    async def update_session(self, session_id, **kw):
        pass


class _RaceStubEngine:
    """Engine stub whose generation BLOCKS mid-stream on an asyncio event so
    the test can land a delete-last deterministically between the streamed
    content and the assistant persist."""

    model_id = "race-model"
    model_family = "chatml"

    def __init__(self, *, block=True):
        self.cfg = SimpleNamespace(
            enable_thinking=False, default_temperature=0.0, default_top_p=1.0,
            default_min_p=0.0, default_top_k=0,
            default_repetition_penalty=1.0, thinking_budget=0,
            default_max_tokens=64,
        )
        self._block = block
        self.started = asyncio.Event()
        self.resume = asyncio.Event()
        self.update_calls: list = []
        self.delete_calls: list = []
        self.truncate_calls: list = []
        self.compact_calls: list = []

    def is_busy(self):
        return False

    async def generate_stream_batches_async(self, *a, **k):
        yield [_StubResult(text="Hello", token=1)]
        self.started.set()
        if self._block:
            await self.resume.wait()
        yield [_StubResult(finish_reason="stop", completion_tokens=1)]

    def update_session_messages(self, session_id, messages):
        self.update_calls.append((session_id, len(messages)))

    def delete_session(self, session_id):
        self.delete_calls.append(session_id)
        return True

    def truncate_session(self, session_id, n):
        self.truncate_calls.append(n)
        return {"truncated_to": n}

    def compact_session(self, session_id, messages):
        self.compact_calls.append(messages)
        return {"method": "build"}


def _patch_chat_module(monkeypatch, stub_db, eng):
    from mlx_soloheaven.api import chat as chat_mod

    monkeypatch.setattr(chat_mod, "db", stub_db)
    monkeypatch.setattr(chat_mod, "engine", eng)
    monkeypatch.setattr(chat_mod, "_engines", {})
    return chat_mod


async def _drive_chat_stream(chat_mod, session_id, content):
    from mlx_soloheaven.api.chat import SendMessageRequest

    resp = await chat_mod.chat(session_id, SendMessageRequest(content=content))
    events = []
    async for line in resp.body_iterator:
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


@pytest.mark.asyncio
async def test_r4_f2_delete_last_during_generation_no_orphan(
    monkeypatch, caplog,
):
    """codex race, deterministic: /chat inserts u3 and blocks in generation;
    delete-last removes u3 (its turn) and rebuilds the engine; generation
    then completes. Pre-fix: the unconditional insert created an ORPHAN
    assistant row whose user turn no longer exists, and the completing
    generation's engine commit reinstalled the deleted turn. Post-fix: the
    conditional insert skips (WARNING), the engine session is deleted
    (honest rebuild from DB next request), and the client still gets its
    normal done event (the content was already delivered)."""
    import logging as _logging

    stub_db = _RaceStubDB([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ])
    eng = _RaceStubEngine(block=True)
    chat_mod = _patch_chat_module(monkeypatch, stub_db, eng)
    caplog.set_level(_logging.WARNING, logger="mlx_soloheaven.api.chat")

    chat_task = asyncio.create_task(
        _drive_chat_stream(chat_mod, "s1", "q3"),
    )
    await asyncio.wait_for(eng.started.wait(), timeout=5)
    # u3 is in the DB and generation is mid-flight — delete the last turn.
    assert stub_db.roles() == ["user", "assistant", "user", "assistant", "user"]
    delete_resp = await chat_mod.delete_last_turn("s1")
    assert delete_resp["remaining_messages"] == 4  # fresh post-delete reread
    # Let the generation complete and the persist epilogue run.
    eng.resume.set()
    events = await asyncio.wait_for(chat_task, timeout=10)

    # NO orphan assistant row: the DB holds exactly the post-delete rows.
    assert stub_db.roles() == ["user", "assistant", "user", "assistant"]
    # WARNING logged; not an error for the client — done event went out.
    assert any(
        "turn deleted mid-generation" in r.message for r in caplog.records
    )
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["content"] == "Hello"
    # Engine session honestly invalidated; the stale install never committed.
    assert eng.delete_calls == ["s1"]
    assert eng.update_calls == []


@pytest.mark.asyncio
async def test_r4_f2_normal_completion_still_persists(monkeypatch):
    """Guard: without a concurrent delete, the conditional insert lands the
    assistant row exactly as before and the engine commit runs."""
    stub_db = _RaceStubDB([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ])
    eng = _RaceStubEngine(block=False)
    eng.resume.set()
    chat_mod = _patch_chat_module(monkeypatch, stub_db, eng)

    events = await asyncio.wait_for(
        _drive_chat_stream(chat_mod, "s1", "q2"), timeout=10,
    )
    assert stub_db.roles() == ["user", "assistant", "user", "assistant"]
    assert stub_db._messages[-1]["content"] == "Hello"
    assert eng.update_calls and eng.delete_calls == []
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["content"] == "Hello"


# ---------------------------------------------------------------------------
# R4F3 — per-segment stripping keeps INTERNAL whitespace
# ---------------------------------------------------------------------------


def test_r4_f3_strip_preserves_internal_whitespace_gemma4():
    """codex repro: thought-A + 'before ' + CALL + '   ' + thought-B +
    'done'. Pre-fix the per-segment parse right-stripped its result, gluing
    'before' straight onto the next thinking marker; the whitespace is
    INTERNAL to the reassembled string and must survive. Only the final
    string is right-stripped."""
    from mlx_soloheaven.engine.mlx_engine import _strip_content_channel_tool_xml

    text = (
        "<|channel>thought\nA<channel|>before " + _G4_CALL + "   "
        "<|channel>thought\nB<channel|>done"
    )
    out = _strip_content_channel_tool_xml(text, "gemma4")
    assert "<|tool_call>" not in out
    # 'before ' + the 3 spaces that followed the call — preserved, so the
    # content does NOT fuse with the thought-B opener.
    assert "before    <|channel>thought\nB" in out
    assert out.endswith("done")


def test_r4_f3_final_result_still_right_stripped():
    """The historical top-level right-strip is intact: trailing whitespace
    of the WHOLE result (after the last call) is still removed."""
    from mlx_soloheaven.engine.mlx_engine import _strip_content_channel_tool_xml

    text = "<think>plan</think>go " + (
        "<tool_call><function=get_x></function></tool_call>"
    ) + "   \n"
    out = _strip_content_channel_tool_xml(text, "chatml")
    assert out == "<think>plan</think>go"


def test_r4_f3_parse_tool_calls_default_batch_rstrip_unchanged():
    """Guard: the normal (whole-output) parse path keeps its historical
    right-strip; segment mode is opt-in."""
    call = "<tool_call><function=get_x></function></tool_call>"
    content, calls = parse_tool_calls("go " + call + "  ", model_family="chatml")
    assert [c["function"]["name"] for c in calls] == ["get_x"]
    assert content == "go"
    content_seg, _ = parse_tool_calls(
        "go " + call + "  ", model_family="chatml", rstrip_content=False,
    )
    assert content_seg == "go   "


# ---------------------------------------------------------------------------
# R4F4 — content channel follows the streaming router (the authority)
# ---------------------------------------------------------------------------


def test_r4_f4_content_segments_router_policy():
    from mlx_soloheaven.engine.tool_parser import content_segments

    # (a) gemma4 orphan close with NO prior thought-open: the active router
    # starts in content mode and scans only for thought openers — everything
    # is content (pre-fix: everything before the orphan close was dropped).
    text_a = "prefix " + _G4_CALL + "<channel|>tail"
    assert content_segments(text_a, "gemma4") == [(0, len(text_a))]
    # (b) chatml with thinking DISABLED: the inactive router passes all text
    # through — the literal </think> is content, not a boundary.
    text_b = "prefix CALL quote: </think> tail"
    assert content_segments(text_b, "chatml", thinking_active=False) == [
        (0, len(text_b)),
    ]
    # Thinking ACTIVE chatml unchanged: content is after the first close.
    text_c = "reasoning</think>answer"
    assert content_segments(text_c, "chatml") == [
        (text_c.index("answer"), len(text_c)),
    ]
    # gemma4 well-formed multi-cycle unchanged: both content cycles present,
    # thought spans excluded (rehearsal protection intact).
    spans = content_segments(_G4_MULTI_CYCLE, "gemma4")
    union = "".join(_G4_MULTI_CYCLE[s:e] for s, e in spans)
    assert union == "Let me check " + _G4_CALL + "done"
    # A call inside a thought span stays excluded.
    rehearsal = "<|channel>thought\nmaybe " + _G4_CALL + " hmm<channel|>ok"
    spans = content_segments(rehearsal, "gemma4")
    assert "".join(rehearsal[s:e] for s, e in spans) == "ok"


def test_r4_f4_gemma4_orphan_close_engine_parse_and_strip(monkeypatch):
    """codex repro (a) end-to-end: 'prefix CALL<channel|>tail'. The
    streaming FSM parses CALL (router: all content); pre-fix the batch
    parse/save strip used the extract-based split, which treated everything
    before the orphan close as reasoning — the call vanished from the batch
    parse (finish 'stop') and the raw XML stayed in the stored content."""
    script = [
        (101, "prefix "),
        (102, _G4_CALL),
        (103, "<channel|>tail"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    eng.model_family = "gemma4"
    _restamp_contract(
        eng, thinking=False, tools=type(eng)._canonical_tools(_TOOLS),
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, tools=_TOOLS)
    assert chunks[-1].finish_reason == "tool_calls"
    sess = eng._sessions["s"]
    calls = sess.messages[-1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_x"]
    content = sess.messages[-1]["content"]
    # Stored content: XML stripped once, surrounding content preserved.
    assert "<|tool_call>" not in content
    assert "prefix" in content
    assert content.endswith("tail")


def test_r4_f4_chatml_thinking_disabled_literal_close_engine(monkeypatch):
    """codex repro (b) end-to-end: thinking DISABLED, output
    'prefix CALL quote: </think> tail'. Streaming (inactive router) parses
    CALL; pre-fix the batch parse split at the literal </think> and saw
    only ' tail' — no call (finish 'stop'), raw XML left in the stored
    content while streaming had already emitted a tool call."""
    call_xml = (
        "<tool_call><function=get_x><parameter=q>1</parameter>"
        "</function></tool_call>"
    )
    script = [
        (101, "prefix "),
        (102, call_xml),
        (103, " quote: </think> tail"),
    ]
    eng, cs, messages, stored = _plain_engine(script, monkeypatch)
    _restamp_contract(
        eng, thinking=False, tools=type(eng)._canonical_tools(_TOOLS),
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, tools=_TOOLS)
    assert chunks[-1].finish_reason == "tool_calls"
    sess = eng._sessions["s"]
    calls = sess.messages[-1]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_x"]
    content = sess.messages[-1]["content"]
    assert "<tool_call>" not in content  # stripped, no double representation
    assert content.startswith("prefix")
    assert "</think>" in content  # the quote is content, preserved
    assert content.endswith("tail")


def test_r4_f4_match_helpers_thread_thinking_state():
    """MATCH SIDE: with the stored turn's thinking DISABLED, a call before a
    literal </think> quote is extractable and not a residual; the gemma4
    orphan-close shape likewise (pre-fix both were hidden by the boundary
    reduction, forcing one-sided mismatches / missed conflicts)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    chatml_raw = (
        "prefix <tool_call><function=get_x></function></tool_call>"
        " quote: </think> tail"
    )
    assert MLXEngine._content_xml_calls_for_match(
        chatml_raw, "chatml", thinking_active=False,
    ) == [("get_x", "{}")]
    assert MLXEngine._has_residual_tool_markers(
        chatml_raw, "chatml", thinking_active=False,
    ) is False
    same = {"tool_calls": [{"function": {"name": "get_x", "arguments": "{}"}}]}
    assert MLXEngine._structured_content_call_conflict(
        same, chatml_raw, "chatml", thinking_active=False,
    ) is False

    gemma_raw = "prefix " + _G4_CALL + "<channel|>tail"
    assert MLXEngine._content_xml_calls_for_match(gemma_raw, "gemma4") == [
        ("get_x", '{"q": "1"}'),
    ]
    assert MLXEngine._has_residual_tool_markers(gemma_raw, "gemma4") is False


# ---------------------------------------------------------------------------
# R4F5 — no phantom tool-call name delta before raw passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r4_f5_malformed_block_no_phantom_delta():
    """codex repro: '<tool_call><function=get_x>broken</tool_call>' — the
    name is extractable early but the closed block fails to parse. Pre-fix
    the client saw a tool_calls delta (id+name) for a call that never
    materialized, then the raw block as content, then finish 'stop'. Now:
    raw content passthrough ONLY, no tool_calls delta at all."""
    engine = _StreamStubEngine(
        ["<tool_call><function=get_x>", "broken", "</tool_call>"],
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _tool_call_deltas(events) == []
    content = _joined_delta(events, "content")
    assert "<tool_call><function=get_x>broken</tool_call>" in content
    assert _final_finish_reason(events) == "stop"


@pytest.mark.asyncio
async def test_r4_f5_valid_block_emits_name_then_args():
    """Guard: a valid block still emits the OpenAI chunk pair in order —
    id+name first (deferred to block close), then arguments."""
    engine = _StreamStubEngine(
        ["<tool_call><function=get_x>",
         "<parameter=q>1</parameter>",
         "</function></tool_call>"],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    deltas = _tool_call_deltas(events)
    assert len(deltas) == 2
    assert deltas[0]["function"]["name"] == "get_x"
    assert deltas[0]["id"]
    assert json.loads(deltas[1]["function"]["arguments"]) == {"q": 1}
    assert _final_finish_reason(events) == "tool_calls"
    assert "<tool_call>" not in _joined_delta(events, "content")


@pytest.mark.asyncio
async def test_r4_f5_mixed_valid_invalid_blocks_stream():
    """Mixed stream: valid block -> deltas; malformed block -> raw content,
    no delta; ordering and finish_reason correct."""
    engine = _StreamStubEngine(
        ["go <tool_call><function=a></function></tool_call>",
         " mid <tool_call>garbage</tool_call> tail"],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="a", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    deltas = _tool_call_deltas(events)
    # Exactly one call (name+args chunks), for the VALID block.
    named = [d for d in deltas if d.get("function", {}).get("name")]
    assert [d["function"]["name"] for d in named] == ["a"]
    content = _joined_delta(events, "content")
    assert "go " in content
    assert "<tool_call>garbage</tool_call>" in content
    assert " tail" in content
    assert _final_finish_reason(events) == "tool_calls"


@pytest.mark.asyncio
async def test_r4_f5_truncated_block_stream_end_no_dangling_delta():
    """Round-1 note updated: the stream ends mid-block and the partial
    block does not parse — raw passthrough with NO tool_calls delta at all
    (pre-fix a name delta could already have been emitted mid-block,
    leaving a dangling name chunk before the raw flush)."""
    engine = _StreamStubEngine(
        ["<tool_call><function=get_x>", "<parameter=q>1</par"],
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _tool_call_deltas(events) == []
    content = _joined_delta(events, "content")
    assert "<tool_call><function=get_x><parameter=q>1</par" in content


# ===========================================================================
# Round 6 (codex round 5 residuals) — R6F1..R6F4
# ===========================================================================
#
# R6F1 — delete-last vs the conditional assistant insert, REVERSE ordering:
#        snapshot + turn-walk + delete are now ONE write transaction
#        (db.delete_last_turn_tx, BEGIN IMMEDIATE). The insert lands
#        entirely BEFORE the transaction (its row is in the tx snapshot →
#        the anchored walk deletes the whole turn, u3+a3) or entirely
#        AFTER (WHERE EXISTS sees u3 gone → insert skipped) — no orphan in
#        either interleaving. A concurrent NEW user turn still survives
#        (U25 round-2 contract: the walk is anchored on the identified
#        turn's first row, not re-run blindly from the tail).
# R6F2 — stop-hit survival is verified PER FLATTENED LEAF (every offset
#        readable AND == len(trimmed ids)); the ahead/trim branch honors
#        is_trimmable() (method presence alone insufficient — a wrapped
#        RotatingKVCache trims 'successfully' while the evicted ring
#        entries are unrecoverable).
# R6F3 — gemma4 markers are ALWAYS authoritative: both streaming APIs keep
#        the router MARKER-ACTIVE regardless of enable_thinking (a thought-
#        span tool rehearsal never reaches the tool FSM), and the gemma4
#        split follows the positional router (bare opener after content =
#        content; orphan close = content) so stream == batch == store.
# R6F4 — a stream ending EXACTLY at the tool start marker emits the bare
#        marker as content (the old EOF guard required a non-empty block
#        body and silently dropped it; batch preserves it byte-for-byte).


# ---------------------------------------------------------------------------
# R6F1 — one-transaction delete-last: no orphan in either interleaving
# ---------------------------------------------------------------------------


async def _r6_f1_setup(tmp_path, monkeypatch):
    """Real database module on a temp file + real delete-last handler with a
    stub engine. Returns (chat_mod, real_db, eng, sid, u3_row)."""
    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.storage import database as real_db

    monkeypatch.setattr(real_db, "_db_path", str(tmp_path / "t.db"))
    await real_db.init_db()
    sess = await real_db.create_session(title="t")
    sid = sess["id"]
    await real_db.add_message(sid, "user", content="q1")
    await real_db.add_message(sid, "assistant", content="a1")
    u3 = await real_db.add_message(sid, "user", content="q3")

    eng = _DeleteStubEngine()
    monkeypatch.setattr(chat_mod, "engine", eng)
    monkeypatch.setattr(chat_mod, "_engines", {})
    assert chat_mod.db is real_db  # the handler drives the REAL module
    return chat_mod, real_db, eng, sid, u3


@pytest.mark.asyncio
async def test_r6_f1_insert_lands_before_tx_whole_turn_deleted(
    monkeypatch, tmp_path,
):
    """codex round 5, finding 1 — interleaving A (the orphan repro): the
    generation's conditional assistant insert commits BETWEEN the handler's
    pre-snapshot and the delete transaction. With BEGIN IMMEDIATE holding
    the write lock from the start, the insert serializes entirely BEFORE
    the transaction — its row is part of the tx's own snapshot and the
    anchored turn-walk deletes the WHOLE turn (u3 + a3). Pre-fix the
    handler deleted only the pre-snapshotted ids ([u3]): the insert's
    WHERE EXISTS had seen u3 still present, so a3 landed, survived the
    delete as an ORPHAN, and the drift rebuild handed it to the engine."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)

    real_tx = real_db.delete_last_turn_tx
    inserted: list = []

    async def racing_tx(session_id, compute_delete_ids):
        # The generation's conditional insert commits AFTER the handler's
        # pre-snapshot but BEFORE the delete transaction's write lock.
        row = await real_db.add_message_if_parent_exists(
            session_id, "assistant", parent_id=u3["id"], content="a3",
        )
        inserted.append(row)
        return await real_tx(session_id, compute_delete_ids)

    monkeypatch.setattr(real_db, "delete_last_turn_tx", racing_tx)

    resp = await chat_mod.delete_last_turn(sid)

    # The insert DID land (its parent existed at insert time)...
    assert inserted and inserted[0] is not None
    # ...and the whole turn is gone — NO orphan assistant row survives.
    remaining = await real_db.get_messages(sid)
    assert [(m["role"], m["content"]) for m in remaining] == [
        ("user", "q1"), ("assistant", "a1"),
    ]
    assert resp["remaining_messages"] == 2
    # The DB moved between pre-snapshot and tx → drift → the engine is
    # rebuilt from the transaction's own remaining rows (no orphan in it).
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert [m["content"] for m in eng.compact_calls[0]] == ["q1", "a1"]


@pytest.mark.asyncio
async def test_r6_f1_insert_after_tx_is_skipped(monkeypatch, tmp_path):
    """Interleaving B: the delete transaction commits FIRST — the
    generation's conditional insert then serializes AFTER it, its WHERE
    EXISTS sees the user row gone, and the insert is SKIPPED. No orphan in
    this ordering either."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)

    resp = await chat_mod.delete_last_turn(sid)
    assert resp["remaining_messages"] == 2

    row = await real_db.add_message_if_parent_exists(
        sid, "assistant", parent_id=u3["id"], content="a3",
    )
    assert row is None  # parent gone → insert skipped

    remaining = await real_db.get_messages(sid)
    assert [(m["role"], m["content"]) for m in remaining] == [
        ("user", "q1"), ("assistant", "a1"),
    ]
    # No concurrent write before the tx → the count-slice fast path ran.
    assert eng.compact_calls == []
    assert eng.truncate_calls == [2]  # q1 + a1 (empty system prompt)


@pytest.mark.asyncio
async def test_r6_f1_anchored_walk_spares_concurrent_new_user_turn(
    monkeypatch,
):
    """The anchored walk deletes the IDENTIFIED turn plus rows appended TO
    IT (the completing generation's a3) but never a concurrently appended
    NEW user turn — re-running the tail-walk blindly against the tx
    snapshot would have deleted the new user row instead (the exact U25
    round-2 bug shape, resurrected)."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q3"},
    ]
    stub_db = _DeleteStubDB(msgs)

    def _land_a3_and_new_turn(db):
        db._messages.append(
            {"id": "a3", "role": "assistant", "content": "late answer"},
        )
        db._messages.append(
            {"id": "new1", "role": "user", "content": "racing question"},
        )

    stub_db.mutate_before_delete = _land_a3_and_new_turn
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 200
    # The identified turn (u3) AND the assistant row that joined it are
    # deleted together; the new user turn survives.
    assert stub_db.deleted_ids == ["m2", "a3"]
    assert stub_db.ids() == ["m0", "m1", "new1"]
    assert r.json()["remaining_messages"] == 3
    # Drift → rebuild from the tx's own remaining rows.
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert eng.compact_calls[0] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "racing question"},
    ]


@pytest.mark.asyncio
async def test_r6_f1_turn_vanished_inside_tx_is_400_and_no_delete(
    monkeypatch,
):
    """A concurrent delete removed the anchored turn before the tx: the
    walker returns no ids, NOTHING is deleted (DB untouched) and the
    handler answers 400 without any engine mutation."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    stub_db = _DeleteStubDB(msgs)

    def _drop_turn(db):
        db._messages = [m for m in db._messages if m["id"] not in ("m2", "m3")]

    stub_db.mutate_before_delete = _drop_turn
    eng = _DeleteStubEngine()
    client = _delete_last_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/delete-last")
    assert r.status_code == 400
    assert stub_db.deleted_ids == []
    assert stub_db.ids() == ["m0", "m1"]
    assert eng.truncate_calls == [] and eng.compact_calls == []


# ---------------------------------------------------------------------------
# R6F2 — stop-hit survival: EVERY flattened leaf verified; is_trimmable
# ---------------------------------------------------------------------------


def _stop_lagging_leaf_stream(monkeypatch, script, *, lag_leaf=0):
    """lm_stream stub: every offset-bearing layer advances per token EXCEPT
    ``lag_leaf``, which skips the LAST scripted token — after the stop trim
    it lands exactly on the recorded ids while the others stay one ahead."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            for i, (tid, text) in enumerate(script):
                last = i == len(script) - 1
                for j, c in enumerate(prompt_cache):
                    if hasattr(c, "offset") and not (last and j == lag_leaf):
                        c.offset += 1
                yield SimpleNamespace(
                    text=text, token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)


def test_r6_f2_stop_second_leaf_ahead_invalidates(monkeypatch):
    """codex round 5, finding 2 repro: leaves land on [N, N+1] with
    N == len(trimmed ids). The scalar offset read (FIRST leaf) reported N,
    so the ahead/trim branch was skipped AND the round-4 backstop accepted
    the cache — the SECOND leaf stayed ahead with the withheld stop token
    inside 'reusable' state. Survival now requires EVERY flattened leaf
    aligned → invalidate (fails pre-fix)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV(), MockKV()]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    _stop_lagging_leaf_stream(
        monkeypatch,
        [(101, "Hello "), (102, "world"), (103, " STOP")],
        lag_leaf=0,
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    # FAIL-CLOSED: one leaf ahead → the whole cache must not survive.
    assert cs.cache is None
    assert cs.token_ids is None
    # The truncated turn is still committed; next turn takes an honest MISS.
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


def test_r6_f2_stop_opaque_leaf_next_to_aligned_invalidates(monkeypatch):
    """Same hole, opaque flavor: ONE aligned visible leaf next to an
    offset-opaque leaf. The scalar read saw only the aligned leaf and
    accepted; the opaque leaf's state (holding the withheld stop tokens) was
    unverifiable. ANY opaque leaf now invalidates (fails pre-fix)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV(), _OpaqueLayer()]
    cache[0].offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    _stop_lagging_leaf_stream(
        monkeypatch,
        [(101, "Hello "), (102, "world"), (103, " STOP")],
        lag_leaf=0,
    )
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    assert cs.cache is None
    assert cs.token_ids is None
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


class _WrappedRotatingKV(MockKV):
    """RotatingKVCache-like leaf whose ring has WRAPPED: trim() exists and
    decrements offset (the post-trim offset verification would pass) but
    is_trimmable() is False — the entries the trimmed appends evicted are
    unrecoverable, so a trim-back is semantically invalid."""

    def is_trimmable(self):
        return False


class _TrimmableRotatingKV(MockKV):
    """Rotating-like leaf that has NOT wrapped: is_trimmable() is True and
    trim is honest — the semantic gate must let it through."""

    def is_trimmable(self):
        return True


def test_r6_f2_wrapped_rotating_with_trim_method_invalidates(monkeypatch):
    """codex round 5, finding 2 (trimmability): the ahead/trim branch used
    METHOD PRESENCE only — a wrapped RotatingKVCache passed, its trim
    decremented offsets (verification green), and the semantically corrupt
    rewound ring survived as 'reusable'. is_trimmable() is now honored →
    invalidate (fails pre-fix)."""
    stored = [1, 2, 3, 4, 5]
    cache = [_WrappedRotatingKV() for _ in range(3)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "Hello "), (102, "world"), (103, " STOP")],
    ])
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "Hello world"
    assert cs.cache is None
    assert cs.token_ids is None
    assert eng._sessions["s"].messages[-1]["content"] == "Hello world"


def test_r6_f2_trimmable_rotating_still_trims_and_survives(monkeypatch):
    """Guard for the semantic gate's positive path: a rotating-like leaf
    that has NOT wrapped (is_trimmable() True) keeps the round-2 behavior —
    trimmed back to the boundary, verified per leaf, cache survives."""
    stored = [1, 2, 3, 4, 5]
    cache = [_TrimmableRotatingKV() for _ in range(3)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "Hello "), (102, "world"), (103, " STOP")],
    ])
    chunks = _drive_locked(eng, messages, max_tokens=8, stop_sequences=[" STOP"])
    assert chunks[-1].finish_reason == "stop"
    assert cs.cache is not None
    assert cs.token_ids == stored + [52, 99] + [101, 102]
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)
    # (The all-aligned CacheList survival guard is
    # test_r4_f1_stop_on_cachelist_trims_and_survives — still green under
    # the per-leaf epilogue.)


# ---------------------------------------------------------------------------
# R6F3 — gemma4 markers always authoritative (stream == batch == store)
# ---------------------------------------------------------------------------

_G4_REHEARSAL_STREAM = [
    "<|channel>thought\n",
    "<|tool_call>call:danger{}",
    "<tool_call|>",
    "<channel|>answer",
]


@pytest.mark.asyncio
async def test_r6_f3_thinking_false_rehearsal_never_becomes_a_call():
    """codex round 5, finding 3 repro (fails pre-fix): thinking=False made
    the gemma4 router a passthrough, so the thought-span rehearsal reached
    the tool FSM and streamed as a REAL call (finish_reason=tool_calls) —
    while the batch parse and the session store excluded it. The router is
    now marker-active regardless of the flag: the rehearsal streams as
    reasoning_content, NO tool_calls delta is emitted, and the finish
    reason stays 'stop'."""
    engine = _StreamStubEngine(
        _G4_REHEARSAL_STREAM, model_family="gemma4", enable_thinking=False,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="danger", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _tool_call_deltas(events) == []
    assert _final_finish_reason(events) == "stop"
    assert _joined_delta(events, "content") == "answer"
    # The rehearsal is on the reasoning channel, never content.
    reasoning = _joined_delta(events, "reasoning_content")
    assert "call:danger" in reasoning
    assert "danger" not in _joined_delta(events, "content")


@pytest.mark.asyncio
async def test_r6_f3_stream_persistence_matches_wire_thinking_false():
    """The session store of the thinking=False stream matches the wire:
    content 'answer' only, no tool_calls entry (pre-fix the passthrough
    persisted a call the batch parse would have rejected)."""

    class _CapturingEngine(_StreamStubEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.saved: list = []

        def update_session_messages(self, sid, messages):
            self.saved.append((sid, messages))

    engine = _CapturingEngine(
        _G4_REHEARSAL_STREAM, model_family="gemma4", enable_thinking=False,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False, user="u1",
        tools=[ToolDef(function=FunctionDef(name="danger", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    await _collect_sse(engine, req)
    assert engine.saved
    assistant = engine.saved[-1][1][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "answer"
    assert "tool_calls" not in assistant


@pytest.mark.asyncio
async def test_r6_f3_web_chat_gemma4_thinking_false_routes_reasoning(
    monkeypatch,
):
    """Web /chat SSE path, same policy: gemma4 with thinking disabled still
    routes thought spans to reasoning frames; the done event's content is
    the router-parity split (fails pre-fix: everything streamed as content
    frames)."""
    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.api.chat import _stream_chat_body

    # The text-bearing stream reaches the persistence epilogue — stub the
    # storage module (the stream is driven directly, engine passed in).
    monkeypatch.setattr(
        chat_mod, "db", _RaceStubDB([{"role": "user", "content": "q"}]),
    )

    class _Eng:
        model_family = "gemma4"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def is_busy(self):
            return False

        async def generate_stream_batches_async(self, *a, **k):
            for tok in _G4_REHEARSAL_STREAM:
                yield [_StubResult(text=tok, token=1)]
            yield [_StubResult(finish_reason="stop", completion_tokens=4)]

        def update_session_messages(self, *a, **k):
            pass

    events = []
    async for line in _stream_chat_body(
        "s1", [{"role": "user", "content": "q"}], _Eng(),
    ):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    content = "".join(
        e.get("content") or "" for e in events if e.get("type") == "text"
    )
    reasoning = "".join(
        e.get("reasoning") or "" for e in events if e.get("type") == "text"
    )
    assert content == "answer"
    assert "call:danger" in reasoning
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["content"] == "answer"
    assert "call:danger" in (done[-1]["thinking"] or "")


def test_r6_f3_split_bare_opener_after_content_is_content():
    """codex repro (fails pre-fix): 'prefix thought\\nsecret<channel|>tail'.
    The router emits ALL of it as content (bare openers only count at
    absolute generation start — FIX 4); the legacy regex extraction split
    it into thinking 'secret' + content 'prefix tail', so persistence
    diverged from the wire. The split now follows the router."""
    from mlx_soloheaven.engine.tool_parser import split_thinking_and_content

    raw = "prefix thought\nsecret<channel|>tail"
    thinking, content = split_thinking_and_content(raw, "gemma4")
    assert thinking is None
    assert content == raw


@pytest.mark.parametrize("raw", [
    "<|channel>thought\nplan<channel|>answer",
    _G4_MULTI_CYCLE,
    "prefix thought\nsecret<channel|>tail",
    "prefix " + _G4_CALL + "<channel|>tail",
    "thought\nsliding reasoning<channel|>the answer",
    "leftover reasoning<channel|>real content",
    "answer text <|channel> dangling open",
    "no markers at all",
    "<|channel>thought\nnever closes",
    "lead <|channel>thought\nA<channel|>mid<|channel>thought\nB<channel|>fin",
])
def test_r6_f3_router_split_union_property(raw):
    """PROPERTY (finding 3 verification): for tricky gemma outputs, the
    router-streamed accumulation == the split output == the
    content_segments union — under multiple chunkings of the same text.
    One marker-authoritative policy, three implementations, zero drift."""
    from mlx_soloheaven.engine.tool_parser import (
        CHANNEL_CONTENT,
        CHANNEL_REASONING,
        ThinkingRouter,
        content_segments,
        split_thinking_and_content,
    )

    thinking, content = split_thinking_and_content(raw, "gemma4")
    union = "".join(raw[s:e] for s, e in content_segments(raw, "gemma4"))
    assert content == union

    for size in (len(raw) or 1, 1, 3, 7):
        chunks = [raw[i:i + size] for i in range(0, len(raw), size)] or [""]
        router = ThinkingRouter(active=True, model_family="gemma4")
        segs = []
        for c in chunks:
            segs.extend(router.feed(c))
        segs.extend(router.flush())
        stream_content = "".join(t for ch, t in segs if ch == CHANNEL_CONTENT)
        stream_reasoning = "".join(
            t for ch, t in segs if ch == CHANNEL_REASONING
        )
        assert stream_content == content
        assert stream_reasoning.strip() == (thinking or "")


@pytest.mark.asyncio
async def test_r6_f3_thinking_true_behavior_unchanged():
    """Guard: with thinking=True the gemma4 stream routes exactly as
    before (reasoning first, then content; a real content-region call still
    parses via the FSM)."""
    engine = _StreamStubEngine(
        [
            "<|channel>thought\nplan<channel|>",
            'go <|tool_call>call:get_x{q:<|"|>1<|"|>}',
            "<tool_call|>",
        ],
        model_family="gemma4", enable_thinking=True,
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _joined_delta(events, "reasoning_content") == "plan"
    deltas = _tool_call_deltas(events)
    named = [d for d in deltas if d.get("function", {}).get("name")]
    assert [d["function"]["name"] for d in named] == ["get_x"]
    assert _final_finish_reason(events) == "tool_calls"


# ---------------------------------------------------------------------------
# R6F4 — exact trailing tool start marker survives stream end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,marker", [
    ("chatml", "<tool_call>"),
    ("glm", "<tool_call>"),
    ("gemma4", "<|tool_call>"),
])
@pytest.mark.asyncio
async def test_r6_f4_stream_ending_exactly_at_start_marker(family, marker):
    """codex round 5, finding 4 (fails pre-fix for every family sharing the
    FSM): the final chunk ends EXACTLY at the start marker — tc_active was
    set with an empty block body, and the old ``tc_active and tc_block``
    EOF guard silently dropped the marker (batch preserves it byte-for-
    byte). The bare marker must be emitted as raw content, no tool_calls
    delta, finish 'stop'."""
    engine = _StreamStubEngine(
        ["answer: ", marker], model_family=family,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _tool_call_deltas(events) == []
    assert _joined_delta(events, "content") == "answer: " + marker
    assert _final_finish_reason(events) == "stop"


@pytest.mark.asyncio
async def test_r6_f4_marker_only_stream_preserved():
    """Degenerate: the WHOLE stream is exactly the start marker."""
    engine = _StreamStubEngine(["<tool_call>"])
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        tools=[ToolDef(function=FunctionDef(name="get_x", parameters={}))],
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _tool_call_deltas(events) == []
    assert _joined_delta(events, "content") == "<tool_call>"
    assert _final_finish_reason(events) == "stop"


# ---------------------------------------------------------------------------
# R8F1 — delete-last turn OWNERSHIP (parent_id), not adjacency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r8_f1_intervening_user_row_no_orphan(monkeypatch, tmp_path):
    """codex round 7, finding 1 repro (fails pre-fix): the concurrent
    request's NEW user row uB lands first, THEN request A's conditional
    assistant insert aA passes (its parent uA still exists). The delete
    transaction's positional walk from uA stops at the uB boundary — pre-fix
    it deleted ONLY uA and aA survived as an orphan. The persisted
    parent_id link now deletes aA WITH its turn regardless of position,
    while uB (a different turn) survives untouched."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)

    real_tx = real_db.delete_last_turn_tx
    inserted: list = []

    async def racing_tx(session_id, compute_delete_ids):
        # Both writes commit AFTER the handler's pre-snapshot and BEFORE the
        # delete transaction's write lock: the intervening user row first...
        await real_db.add_message(session_id, "user", content="racing question")
        # ...then the completing generation's conditional insert (uA exists).
        row = await real_db.add_message_if_parent_exists(
            session_id, "assistant", parent_id=u3["id"], content="late answer",
        )
        inserted.append(row)
        return await real_tx(session_id, compute_delete_ids)

    monkeypatch.setattr(real_db, "delete_last_turn_tx", racing_tx)

    resp = await chat_mod.delete_last_turn(sid)

    # The insert DID land (its parent existed at insert time)...
    assert inserted and inserted[0] is not None
    # ...and it carries the persisted ownership link.
    assert inserted[0]["parent_id"] == u3["id"]
    # The WHOLE turn (u3 + its late assistant row) is gone; the concurrent
    # NEW user turn survives — no orphan, no mis-deleted neighbor.
    remaining = await real_db.get_messages(sid)
    assert [(m["role"], m["content"]) for m in remaining] == [
        ("user", "q1"), ("assistant", "a1"), ("user", "racing question"),
    ]
    assert resp["remaining_messages"] == 3
    # Drift between pre-snapshot and tx → engine rebuilt from the tx's own
    # remaining rows (the surviving new user turn included, no orphan).
    assert eng.truncate_calls == []
    assert len(eng.compact_calls) == 1
    assert [m["content"] for m in eng.compact_calls[0]] == [
        "q1", "a1", "racing question",
    ]


@pytest.mark.asyncio
async def test_r8_f1_persisted_parent_id_lands_on_assistant_rows(
    monkeypatch, tmp_path,
):
    """The write paths persist the ownership link: assistant rows store the
    originating user row's id; user rows keep NULL."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)
    row = await real_db.add_message_if_parent_exists(
        sid, "assistant", parent_id=u3["id"], content="a3",
    )
    assert row is not None
    stored = await real_db.get_messages(sid)
    by_content = {m["content"]: m for m in stored}
    assert by_content["a3"]["parent_id"] == u3["id"]
    assert by_content["q3"].get("parent_id") is None
    assert by_content["q1"].get("parent_id") is None


@pytest.mark.asyncio
async def test_r8_f1_legacy_null_parent_rows_keep_positional_delete(
    monkeypatch, tmp_path,
):
    """Legacy rows (no parent_id — pre-migration sessions) keep the round-6
    behavior: the positional walk still deletes the whole trailing turn."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)
    # Legacy shape: the assistant row was persisted with NO ownership link.
    await real_db.add_message(sid, "assistant", content="a3-legacy")

    resp = await chat_mod.delete_last_turn(sid)
    assert resp["remaining_messages"] == 2
    remaining = await real_db.get_messages(sid)
    assert [(m["role"], m["content"]) for m in remaining] == [
        ("user", "q1"), ("assistant", "a1"),
    ]


@pytest.mark.asyncio
async def test_r8_f1_migration_adds_parent_id_to_existing_db(
    monkeypatch, tmp_path,
):
    """Startup migration: a pre-round-8 database (messages table without
    parent_id) gains the nullable column on init_db, legacy rows stay NULL,
    and the conditional insert persists the link on the migrated schema."""
    import aiosqlite

    from mlx_soloheaven.storage import database as real_db

    db_path = tmp_path / "legacy.db"
    async with aiosqlite.connect(db_path) as db:
        # The round-6 messages schema — no parent_id column.
        await db.executescript("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                thinking TEXT,
                token_count INTEGER DEFAULT 0,
                stats TEXT,
                created_at REAL
            );
        """)
        await db.commit()

    monkeypatch.setattr(real_db, "_db_path", str(db_path))
    await real_db.init_db()

    async with aiosqlite.connect(db_path) as db:
        cols = [
            r[1] for r in await db.execute_fetchall("PRAGMA table_info(messages)")
        ]
    assert "parent_id" in cols

    sess = await real_db.create_session(title="t")
    u = await real_db.add_message(sess["id"], "user", content="q")
    a = await real_db.add_message_if_parent_exists(
        sess["id"], "assistant", parent_id=u["id"], content="ans",
    )
    assert a is not None
    rows = await real_db.get_messages(sess["id"])
    assert rows[0].get("parent_id") is None
    assert rows[1]["parent_id"] == u["id"]


# ---------------------------------------------------------------------------
# R8F2 — cancellation must not commit stop-prefix bytes hidden in the KV
# ---------------------------------------------------------------------------

_R8_EOT = 77


def _r8_close_capable_engine(monkeypatch, script):
    """HIT harness whose interrupted-turn commit CAN template-close (callable
    target + <|im_end|> in vocab). Pre-fix, a cancelled/torn-down stream with
    a withheld stop-prefix tail committed 'answer ' to messages while the
    recorded ids/KV kept the hidden 'ST' token ALIVE in reusable state —
    these tests need an engine whose cache would otherwise survive the
    cancel commit, so the hidden-bytes hole is observable."""
    from test_qwen_mtp import _CallableTargetModel

    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": _R8_EOT},
    )
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [script])
    return eng, cs, messages, stored


def _r8_stop_gen(eng, messages, *, cancel_event=None, max_tokens=8):
    return eng._generate_locked(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        session_id="s",
        tools=None,
        cancel_event=cancel_event,
        thinking=False,
        thinking_budget=0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=1.0,
        response_format=None,
        stop_sequences=["STOP"],
    )


def test_r8_f2_generator_exit_pending_boundary_trims(monkeypatch):
    """codex round 7, finding 2 repro (fails pre-fix): stop='STOP', the
    stream emitted 'answer ' and holds the partial prefix 'ST' when the
    client disconnects (GeneratorExit at the streaming yield). The committed
    message is 'answer ' (byte-what-the-client-received), and the held
    token's bytes must NOT survive in reusable state: the pending text
    starts exactly on a token boundary here, so the ids AND the cache are
    trimmed back to the boundary, verified per leaf, and the cache SURVIVES
    holding exactly the committed text (+ the template close). Pre-fix the
    hidden 'ST' token stayed in ids/KV while messages said 'answer '."""
    script = [(101, "answer"), (102, " "), (103, "ST"), (104, "never")]
    eng, cs, messages, stored = _r8_close_capable_engine(monkeypatch, script)

    gen = _r8_stop_gen(eng, messages)
    received = []
    for ch in gen:
        received.append(ch.text)
        if len(received) == 3:
            break
    gen.close()  # GeneratorExit at the yield — the client disconnected

    # The wire only ever saw 'answer ' (the 'ST' frame was withheld).
    assert "".join(received) == "answer "
    sess = eng._sessions["s"]
    assert sess.messages[-1]["content"] == "answer "
    # The hidden token is OUT of the recorded ids; boundary trim + template
    # close leave ids exactly matching the committed text.
    assert cs.token_ids == stored + [52, 99] + [101, 102, _R8_EOT]
    assert 103 not in cs.token_ids
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)
    assert sess.total_cache_tokens == len(cs.token_ids)


def test_r8_f2_cancel_flag_pending_boundary_trims(monkeypatch):
    """Same contract via the cancelled-flag path (cancel_event observed in
    the stream loop) — the normal post-loop flush is guarded by ``not
    cancelled``, so pre-fix the pending 'ST' was silently committed into
    ids/KV while messages stored 'answer ' (fails pre-fix)."""
    script = [(101, "answer"), (102, " "), (103, "ST"), (104, "never")]
    eng, cs, messages, stored = _r8_close_capable_engine(monkeypatch, script)

    cancel = threading.Event()
    received = []
    for ch in _r8_stop_gen(eng, messages, cancel_event=cancel):
        received.append(ch.text)
        if len(received) == 3:
            cancel.set()

    assert "".join(received) == "answer "
    sess = eng._sessions["s"]
    assert sess.messages[-1]["content"] == "answer "
    assert cs.token_ids == stored + [52, 99] + [101, 102, _R8_EOT]
    assert 103 not in cs.token_ids
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


def test_r8_f2_cancel_pending_mid_token_invalidates(monkeypatch):
    """Fail-closed leg (fails pre-fix): the withheld tail starts MID-TOKEN
    ('r ST' is one token — 'r ' was emitted, 'ST' withheld), so no boundary
    trim can remove the hidden bytes from the KV. The cancel reconcile must
    INVALIDATE the session cache — pre-fix it survived with the hidden 'ST'
    inside and the next HIT resumed from it."""
    script = [(101, "answe"), (102, "r ST"), (103, "x")]
    eng, cs, messages, stored = _r8_close_capable_engine(monkeypatch, script)

    cancel = threading.Event()
    received = []
    for ch in _r8_stop_gen(eng, messages, cancel_event=cancel):
        received.append(ch.text)
        if len(received) == 2:
            cancel.set()

    from test_qwen_mtp import _assert_no_desynced_cache

    assert "".join(received) == "answer "
    # NEVER a live cache containing the hidden 'ST': mid-token → invalidate.
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)


def test_r8_f2_cancel_pending_untrimmable_layers_invalidate(monkeypatch):
    """Fail-closed leg via layer type (machinery guard): the pending tail IS
    boundary-aligned but a recurrent (untrimmable) layer — the production
    hybrid shape — cannot be verifiably rewound, so the pending-trim path
    invalidates instead of leaving the hidden bytes in a surviving cache."""
    from test_qwen_mtp import MockArrays, _assert_no_desynced_cache

    stored = [1, 2, 3, 4, 5]
    cache = [MockArrays(), MockKV(), MockKV(), MockKV(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[52, 99])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [
        [(101, "answer"), (102, " "), (103, "ST"), (104, "never")],
    ])

    cancel = threading.Event()
    received = []
    for ch in _r8_stop_gen(eng, messages, cancel_event=cancel):
        received.append(ch.text)
        if len(received) == 3:
            cancel.set()

    assert "".join(received) == "answer "
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)


def test_r8_f2_normal_cancel_without_pending_unchanged(monkeypatch):
    """Guard: a cancel with an EMPTY pending buffer (all emitted text was
    released — nothing withheld) keeps the existing C1 cancel-commit shape
    untouched: ids/KV hold exactly the emitted text + close."""
    script = [(101, "plain"), (102, " text"), (103, " more")]
    eng, cs, messages, stored = _r8_close_capable_engine(monkeypatch, script)

    cancel = threading.Event()
    received = []
    for ch in _r8_stop_gen(eng, messages, cancel_event=cancel):
        received.append(ch.text)
        if len(received) == 2:
            cancel.set()

    assert "".join(received) == "plain text"
    sess = eng._sessions["s"]
    assert sess.messages[-1]["content"] == "plain text"
    assert cs.token_ids == stored + [52, 99] + [101, 102, _R8_EOT]
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


# ---------------------------------------------------------------------------
# R8F3 — gemma4 bare 'thought\n' opener obeys the thinking contract
# ---------------------------------------------------------------------------

_R8_BARE_TEXT = "thought\nThis is the final answer."


@pytest.mark.asyncio
async def test_r8_f3_stream_thinking_false_bare_opener_stays_content():
    """codex round 7, finding 3 repro (fails pre-fix): thinking=False, the
    model's output genuinely starts with the words 'thought\\n...'. Round 6
    made the gemma4 router always marker-active, so the AMBIGUOUS bare
    opener fired too and the whole answer streamed as reasoning_content
    (content empty). The bare-opener heuristic is now gated on the thinking
    contract: the text stays CONTENT."""
    engine = _StreamStubEngine(
        ["thought\n", "This is the final answer."],
        model_family="gemma4", enable_thinking=False,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    assert _joined_delta(events, "content") == _R8_BARE_TEXT
    assert _joined_delta(events, "reasoning_content") == ""
    assert _final_finish_reason(events) == "stop"


@pytest.mark.asyncio
async def test_r8_f3_web_chat_thinking_false_bare_opener_stays_content(
    monkeypatch,
):
    """Web /chat SSE path, same policy (fails pre-fix): the router AND the
    done event's split both keep the bare-opener text on the content
    channel when thinking is disabled."""
    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.api.chat import _stream_chat_body

    monkeypatch.setattr(
        chat_mod, "db", _RaceStubDB([{"role": "user", "content": "q"}]),
    )

    class _Eng:
        model_family = "gemma4"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=False)

        def is_busy(self):
            return False

        async def generate_stream_batches_async(self, *a, **k):
            for tok in ("thought\n", "This is the final answer."):
                yield [_StubResult(text=tok, token=1)]
            yield [_StubResult(finish_reason="stop", completion_tokens=2)]

        def update_session_messages(self, *a, **k):
            pass

    events = []
    async for line in _stream_chat_body(
        "s1", [{"role": "user", "content": "q"}], _Eng(),
    ):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    content = "".join(
        e.get("content") or "" for e in events if e.get("type") == "text"
    )
    reasoning = "".join(
        e.get("reasoning") or "" for e in events if e.get("type") == "text"
    )
    assert content == _R8_BARE_TEXT
    assert reasoning == ""
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["content"] == _R8_BARE_TEXT
    assert not (done[-1]["thinking"] or "")


def test_r8_f3_split_segments_strip_thinking_false_bare_opener():
    """Unit sweep (fails pre-fix): with the thinking contract INACTIVE the
    bare opener is content across the batch split, the positional
    segmentation, and the input-history strip."""
    from mlx_soloheaven.engine.tool_parser import (
        content_segments,
        split_thinking_and_content,
        strip_thinking_tags,
    )

    thinking, content = split_thinking_and_content(
        _R8_BARE_TEXT, "gemma4", thinking_active=False,
    )
    assert thinking is None
    assert content == _R8_BARE_TEXT

    assert content_segments(_R8_BARE_TEXT, "gemma4", thinking_active=False) == [
        (0, len(_R8_BARE_TEXT)),
    ]

    msgs = strip_thinking_tags(
        [{"role": "assistant", "content": _R8_BARE_TEXT}],
        model_family="gemma4", thinking_active=False,
    )
    assert msgs[0]["content"] == _R8_BARE_TEXT


def test_r8_f3_strip_thinking_false_closed_bare_span_keeps_content():
    """Input-history hygiene, the divergent shape (fails pre-fix): a bare
    opener followed by an orphan close. Under an INACTIVE contract nothing
    before the orphan close is reasoning — only the stray marker itself is
    dropped from the replayed prompt. Under an ACTIVE contract the round-6
    hygiene is unchanged (the closed bare span strips as thinking)."""
    from mlx_soloheaven.engine.tool_parser import strip_thinking_tags

    raw = "thought\nThe answer<channel|> is 42."
    inactive = strip_thinking_tags(
        [{"role": "assistant", "content": raw}],
        model_family="gemma4", thinking_active=False,
    )
    assert inactive[0]["content"] == "thought\nThe answer is 42."

    active = strip_thinking_tags(
        [{"role": "assistant", "content": raw}],
        model_family="gemma4", thinking_active=True,
    )
    assert active[0]["content"] == "is 42."


def test_r8_f3_thinking_true_bare_opener_still_reasoning():
    """Guard: with the thinking contract ACTIVE the bare opener keeps its
    round-6 sliding-window semantics — recognized at generation start and
    routed to reasoning, in the router and the split alike."""
    from mlx_soloheaven.engine.tool_parser import (
        CHANNEL_CONTENT,
        CHANNEL_REASONING,
        ThinkingRouter,
        split_thinking_and_content,
    )

    raw = "thought\nplan<channel|>answer"
    router = ThinkingRouter(
        active=True, model_family="gemma4", enable_thinking=True,
    )
    segs = router.feed(raw) + router.flush()
    assert "".join(t for ch, t in segs if ch == CHANNEL_REASONING) == "plan"
    assert "".join(t for ch, t in segs if ch == CHANNEL_CONTENT) == "answer"

    thinking, content = split_thinking_and_content(
        raw, "gemma4", thinking_active=True,
    )
    assert thinking == "plan"
    assert content == "answer"


def test_r8_f3_full_markers_still_authoritative_thinking_false():
    """Guard (round-6 / U12 protection preserved): FULL channel markers are
    model-emitted and stay authoritative under thinking=False — the thought
    span routes to reasoning / strips everywhere, exactly as in round 6."""
    from mlx_soloheaven.engine.tool_parser import (
        CHANNEL_CONTENT,
        CHANNEL_REASONING,
        ThinkingRouter,
        content_segments,
        split_thinking_and_content,
        strip_thinking_tags,
    )

    raw = "<|channel>thought\nsecret<channel|>answer"
    router = ThinkingRouter(
        active=True, model_family="gemma4", enable_thinking=False,
    )
    segs = router.feed(raw) + router.flush()
    assert "".join(t for ch, t in segs if ch == CHANNEL_REASONING) == "secret"
    assert "".join(t for ch, t in segs if ch == CHANNEL_CONTENT) == "answer"

    thinking, content = split_thinking_and_content(
        raw, "gemma4", thinking_active=False,
    )
    assert thinking == "secret"
    assert content == "answer"

    assert [
        raw[s:e]
        for s, e in content_segments(raw, "gemma4", thinking_active=False)
    ] == ["answer"]

    stripped = strip_thinking_tags(
        [{"role": "assistant", "content": raw}],
        model_family="gemma4", thinking_active=False,
    )
    assert stripped[0]["content"] == "answer"


@pytest.mark.parametrize("raw", [
    _R8_BARE_TEXT,
    "<|channel>thought\nplan<channel|>answer",
    "prefix thought\nsecret<channel|>tail",
    "leftover reasoning<channel|>real content",
    "thought\nis fine <|channel>thought\nX<channel|>done",
    "no markers at all",
])
def test_r8_f3_router_split_parity_thinking_false(raw):
    """PROPERTY: under an INACTIVE thinking contract the router-streamed
    accumulation == the split output == the content_segments union — for
    every chunking. One policy, three implementations, zero drift (the
    round-6 parity property, re-proven for the new flag value)."""
    from mlx_soloheaven.engine.tool_parser import (
        CHANNEL_CONTENT,
        CHANNEL_REASONING,
        ThinkingRouter,
        content_segments,
        split_thinking_and_content,
    )

    thinking, content = split_thinking_and_content(
        raw, "gemma4", thinking_active=False,
    )
    union = "".join(
        raw[s:e] for s, e in content_segments(raw, "gemma4", thinking_active=False)
    )
    assert content == union

    for size in (len(raw) or 1, 1, 3, 7):
        chunks = [raw[i:i + size] for i in range(0, len(raw), size)] or [""]
        router = ThinkingRouter(
            active=True, model_family="gemma4", enable_thinking=False,
        )
        segs = []
        for c in chunks:
            segs.extend(router.feed(c))
        segs.extend(router.flush())
        stream_content = "".join(t for ch, t in segs if ch == CHANNEL_CONTENT)
        stream_reasoning = "".join(
            t for ch, t in segs if ch == CHANNEL_REASONING
        )
        assert stream_content == content
        assert stream_reasoning.strip() == (thinking or "")


# ===========================================================================
# Round 10 (codex round 9 residuals) — R10F1..R10F3
# ===========================================================================
#
# R10F1 — the delete-last POSITIONAL walk is the legacy fallback: it may
#         only claim rows with no persisted owner (parent_id IS NULL). A
#         trailing row explicitly owned by ANOTHER turn belongs to its
#         parent ([uA, uB, aA(parent=uA)] anchored on uB deletes uB only);
#         the anchor's own rows keep flowing in via the ownership union.
# R10F2 — cache matching runs the router-authoritative channel reduction
#         under the stored thinking contract: _normalize_for_match uses the
#         multi-cycle content union (gemma4) with the bare opener gated on
#         the flag, and _interrupted_resend_equiv threads the flag into its
#         split — no wrong-HIT from flag-blind suffix-only reduction.
# R10F3 — OpenAI structured requests strip input history under the
#         EFFECTIVE thinking contract (request flag AND NOT structured
#         suppression), sync and streaming; engine-side forcing unchanged.


# ---------------------------------------------------------------------------
# R10F1 — positional walk must not claim rows owned by another turn
# ---------------------------------------------------------------------------


def test_r10_f1_walker_skips_assistant_owned_by_other_turn():
    """codex round 9, finding 1 repro (fails pre-fix returning
    ['uB', 'aA']): after concurrency settles as [uA, uB, aA(parent=uA)],
    delete-last anchored on uB deletes uB ONLY — aA is A's answer; taking
    it with uB's turn leaves uA permanently incomplete."""
    from mlx_soloheaven.api.chat import _turn_delete_ids_from_anchor

    rows = [
        {"id": "uA", "role": "user", "content": "qA", "parent_id": None},
        {"id": "uB", "role": "user", "content": "qB", "parent_id": None},
        {"id": "aA", "role": "assistant", "content": "answer A",
         "parent_id": "uA"},
    ]
    assert _turn_delete_ids_from_anchor(rows, "uB", compaction=False) == ["uB"]


def test_r10_f1_walker_legacy_and_owned_rows_unchanged():
    """Guards: legacy NULL-parent trailing rows keep the positional delete
    (round-6 behavior); the ownership union still folds the anchor's own
    rows in regardless of position (round-8 contract)."""
    from mlx_soloheaven.api.chat import _turn_delete_ids_from_anchor

    legacy = [
        {"id": "u1", "role": "user", "content": "q"},
        {"id": "a1", "role": "assistant", "content": "a"},
        {"id": "t1", "role": "tool", "content": "r"},
    ]
    assert _turn_delete_ids_from_anchor(legacy, "u1", compaction=False) == [
        "u1", "a1", "t1",
    ]

    mixed = [
        {"id": "uA", "role": "user", "content": "qA"},
        {"id": "uB", "role": "user", "content": "qB"},
        {"id": "aA", "role": "assistant", "content": "A", "parent_id": "uA"},
        {"id": "aB", "role": "assistant", "content": "B", "parent_id": "uB"},
        {"id": "aX", "role": "assistant", "content": "X"},  # legacy NULL
    ]
    # Anchored on uB: aA (another turn's) is skipped by the walk; aB joins
    # via the ownership union; aX (legacy) stays positional.
    assert _turn_delete_ids_from_anchor(mixed, "uB", compaction=False) == [
        "uB", "aX", "aB",
    ]
    # Anchored on uA: the union reaches aA beyond the positional boundary
    # (the uB row), exactly as in round 8.
    assert _turn_delete_ids_from_anchor(mixed, "uA", compaction=False) == [
        "uA", "aA",
    ]


@pytest.mark.asyncio
async def test_r10_f1_delete_last_spares_other_turns_late_answer(
    monkeypatch, tmp_path,
):
    """End-to-end repro on the real DB/handler (fails pre-fix): a new user
    turn uB lands, then A's completing generation commits its answer for u3
    (parent=u3). delete-last anchors on uB and must remove ONLY uB — u3's
    turn keeps its answer."""
    chat_mod, real_db, eng, sid, u3 = await _r6_f1_setup(tmp_path, monkeypatch)
    await real_db.add_message(sid, "user", content="qB")
    row = await real_db.add_message_if_parent_exists(
        sid, "assistant", parent_id=u3["id"], content="answer A",
    )
    assert row is not None and row["parent_id"] == u3["id"]

    resp = await chat_mod.delete_last_turn(sid)

    remaining = await real_db.get_messages(sid)
    assert [(m["role"], m["content"]) for m in remaining] == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q3"), ("assistant", "answer A"),
    ]
    assert resp["remaining_messages"] == 4


# ---------------------------------------------------------------------------
# R10F2 — flag-aware, multi-cycle-aware cache-match normalization
# ---------------------------------------------------------------------------


def _r10_match_stub(family="gemma4"):
    """Lightweight engine shell binding the real match helpers (the
    test_cache_match_* pattern) with a model family."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stub = SimpleNamespace()
    stub.model_family = family
    stub._flatten_multipart = MLXEngine._flatten_multipart
    stub._normalize_for_match = MLXEngine._normalize_for_match
    stub._is_compacted_tool = MLXEngine._is_compacted_tool
    stub._interrupted_resend_equiv = (
        MLXEngine._interrupted_resend_equiv.__get__(stub)
    )
    stub._match = MLXEngine._messages_match.__get__(stub)
    return stub


def test_r10_f2_thinking_disabled_bare_opener_no_wrong_hit():
    """codex round 9, finding 2 repro (a) — STRICT match (fails pre-fix by
    matching): under thinking=False a stored gemma4 turn genuinely starting
    'thought\\n...' is ALL content (orphan close preserved by the router);
    the flag-blind normalize reduced it to the post-<channel|> suffix, so a
    bare incoming 'answer' selected the wrong anon session."""
    stub = _r10_match_stub()
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thought\nSECRET<channel|>answer"},
    ]
    incoming = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored, incoming,
        last_assistant_wildcard=False, thinking_active=False,
    ) is False
    # Under an ACTIVE contract the same shapes ARE equivalent (the bare
    # opener is reasoning; 'answer' is what streamed on the wire) — HIT.
    assert stub._match(
        stored, incoming,
        last_assistant_wildcard=False, thinking_active=True,
    ) is True


def test_r10_f2_multicycle_union_no_suffix_wrong_hit():
    """codex round 9, finding 2 repro (b) — STRICT match (fails pre-fix by
    matching): the stored multi-cycle turn's content channel is the UNION
    of all content segments; the last-<channel|> slice compared suffix-only
    and accepted an incoming that lost the earlier cycle's content."""
    stub = _r10_match_stub()
    stored_raw = (
        "<|channel>thought\nplan<channel|>IMPORTANT"
        "<|channel>thought\nmore<channel|>done"
    )
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": stored_raw},
    ]

    def _incoming(content):
        return [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "next"},
        ]

    assert stub._match(
        stored, _incoming("done"),
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # Legit resends still match: the wire content (multi-cycle union) and
    # the byte-identical raw replay.
    assert stub._match(
        stored, _incoming("IMPORTANTdone"),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True
    assert stub._match(
        stored, _incoming(stored_raw),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True


def test_r10_f2_interrupted_resend_equiv_threads_flag():
    """codex round 9, finding 2 — _interrupted_resend_equiv called the split
    with the default ACTIVE policy: a thinking-DISABLED stored gemma4 turn
    'thought\\nThe answer<channel|> is 42.' shrank to 'is 42.' and a forged
    bare resend passed even the EXACT (anon/strict) rule (fails pre-fix)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stub = _r10_match_stub()
    stored_raw = "thought\nThe answer<channel|> is 42."
    # thinking=False: the content channel is the whole text — the forged
    # suffix must NOT be equivalent...
    assert stub._interrupted_resend_equiv(
        stored_raw, "is 42.", exact=True, thinking_active=False,
    ) is False
    # ...while the honest full resend (normalized like _messages_match
    # normalizes the incoming side) is.
    full_norm = MLXEngine._normalize_for_match(
        stored_raw, "assistant", "gemma4", False,
    )
    assert stub._interrupted_resend_equiv(
        stored_raw, full_norm, exact=True, thinking_active=False,
    ) is True
    # Active contract keeps the round-7 shape: bare opener = reasoning, the
    # wire content channel was 'is 42.'.
    assert stub._interrupted_resend_equiv(
        stored_raw, "is 42.", exact=True, thinking_active=True,
    ) is True


# ---------------------------------------------------------------------------
# R10F3 — structured requests strip history under the EFFECTIVE contract
# ---------------------------------------------------------------------------


class _CaptureMsgsEngine(_StreamStubEngine):
    """_StreamStubEngine that also records the positional messages arg."""

    async def generate_stream_batches_async(self, *args, **kwargs):
        self.captured_messages = args[0] if args else kwargs.get("messages")
        async for batch in super().generate_stream_batches_async(
            *args, **kwargs,
        ):
            yield batch


_R10_HISTORY_RAW = "thought\nThe answer<channel|> is 42."
# strip_thinking_tags under thinking=False: the bare opener is content, only
# the orphan full marker is dropped (the r8_f3 inactive contract).
_R10_HISTORY_INACTIVE = "thought\nThe answer is 42."


def test_r10_f3_sync_structured_history_strip_effective_flag():
    """codex round 9, finding 3 repro (fails pre-fix): gemma4, thinking=True,
    response_format=json_object. The engine is correctly called with
    thinking=False (U9), but the history was stripped under the PRE-
    suppression thinking=True, reducing the bare-opener turn to 'is 42.'.
    The strip must preserve what the thinking=False contract preserves."""
    captured = {}

    class _Eng:
        model_family = "gemma4"
        model_id = "m"
        cfg = SimpleNamespace(enable_thinking=True)

        def complete(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return SimpleNamespace(
                content='{"a": 1}', thinking=None, tool_calls=None,
                finish_reason="stop", prompt_tokens=1, completion_tokens=1,
                cache_info=None,
            )

        def update_session_messages(self, *a, **k):
            pass

    req = ChatCompletionRequest(
        model="m", stream=False, thinking=True,
        response_format=ResponseFormat(type="json_object"),
        messages=[
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content=_R10_HISTORY_RAW),
            ChatMessage(role="user", content="again as json"),
        ],
    )
    _sync_completion(req, _Eng())
    # Engine-side forcing unchanged (defense in depth)...
    assert captured.get("thinking") is False
    # ...and the history strip ran under the SAME effective contract.
    assert captured["messages"][1]["content"] == _R10_HISTORY_INACTIVE


@pytest.mark.asyncio
async def test_r10_f3_stream_structured_history_strip_effective_flag():
    """STREAMING path of the same repro (fails pre-fix): ordering moved the
    suppression above the strip, so the effective thinking=False contract
    drives the history preprocessing."""
    engine = _CaptureMsgsEngine(
        ['{"a"', ': 1}'], model_family="gemma4", enable_thinking=True,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        response_format=ResponseFormat(type="json_object"),
        messages=[
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content=_R10_HISTORY_RAW),
            ChatMessage(role="user", content="again as json"),
        ],
    )
    events = await _collect_sse(engine, req)
    assert engine.captured_kwargs.get("thinking") is False
    assert engine.captured_messages[1]["content"] == _R10_HISTORY_INACTIVE
    # The JSON still lands in content (U9 router contract unchanged).
    assert _joined_delta(events, "content") == '{"a": 1}'


@pytest.mark.asyncio
async def test_r10_f3_non_structured_strip_unchanged():
    """Guard: without response_format the strip keeps the request contract —
    thinking=True still strips the bare-opener reasoning span (round 7/8
    behavior untouched)."""
    engine = _CaptureMsgsEngine(
        ["answer"], model_family="gemma4", enable_thinking=True,
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=True,
        messages=[
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content=_R10_HISTORY_RAW),
            ChatMessage(role="user", content="next"),
        ],
    )
    await _collect_sse(engine, req)
    assert engine.captured_kwargs.get("thinking") is True
    assert engine.captured_messages[1]["content"] == "is 42."


# ===========================================================================
# Round 12 (codex round 11 residuals) — R12F1/R12F2
# ===========================================================================
#
# R12F1 — chatml/glm active-contract matching uses FIRST-close semantics:
#         the router never re-enters thinking (tool_parser.content_segments),
#         so the first '</think>' is authoritative and any later one is a
#         literal quote INSIDE the content channel. The old rindex
#         (LAST-close) reduction collapsed distinct content channels onto
#         their final suffix, so a forged plain resend of that suffix
#         wrong-HIT even on the strict anon path — in _normalize_for_match
#         AND in _interrupted_resend_equiv's stored-side split. A side
#         without the leading opener (the engine raw always carries it —
#         _make_full_assistant_content) is a content channel and is kept
#         whole; ambiguous raw shapes degrade to an honest MISS.
# R12F2 — streamed tool_calls indices are assigned POST-validation
#         (len(parsed_tool_calls) at emission), so a malformed block that
#         passes through as raw content no longer consumes an index — the
#         emitted sequence is contiguous from 0 (clients reconstructing
#         calls by index no longer see holes).


# ---------------------------------------------------------------------------
# R12F1 — first-close semantics: quoted '</think>' is content, not a boundary
# ---------------------------------------------------------------------------

_R12_QUOTED_RAW = "<think>private</think>PUBLIC PREFIX </think>PUBLIC SUFFIX"
_R12_QUOTED_CONTENT = "PUBLIC PREFIX </think>PUBLIC SUFFIX"


def _r12_incoming(content):
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "next"},
    ]


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r12_f1_quoted_close_no_suffix_wrong_hit(family):
    """codex round 11, finding 1 repro — STRICT match (fails pre-fix by
    matching): stored raw quotes a second '</think>' inside the content
    channel; the rindex reduction cut both sides down to 'PUBLIC SUFFIX',
    so a forged plain resend of the suffix was accepted although the real
    content channel is 'PUBLIC PREFIX </think>PUBLIC SUFFIX'."""
    stub = _r10_match_stub(family)
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _R12_QUOTED_RAW},
    ]
    assert stub._match(
        stored, _r12_incoming("PUBLIC SUFFIX"),
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # The true content channel (byte-identical quote included) still HITs...
    assert stub._match(
        stored, _r12_incoming(_R12_QUOTED_CONTENT),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True
    # ...and so does the byte-identical raw replay (first-close reduces the
    # incoming raw to the same content channel).
    assert stub._match(
        stored, _r12_incoming(_R12_QUOTED_RAW),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r12_f1_stored_raw_vs_split_still_match(family):
    """Guard — the pair the rindex form was protecting: stored engine raw
    ('<think>t</think>answer') vs the client's already-split resend
    ('answer') must still MATCH under first-close semantics."""
    stub = _r10_match_stub(family)
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>t</think>answer"},
    ]
    assert stub._match(
        stored, _r12_incoming("answer"),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True
    # An unrelated resend still misses.
    assert stub._match(
        stored, _r12_incoming("different"),
        last_assistant_wildcard=False, thinking_active=True,
    ) is False


def test_r12_f1_normalize_first_close_shapes():
    """Shape table for the chatml/glm active-contract normalization."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    def norm(text, family="chatml", active=True):
        return MLXEngine._normalize_for_match(
            text, "assistant", family, active,
        )

    # Raw with leading opener: content after the FIRST close (quote kept).
    assert norm("<think>t</think>answer") == "answer"
    assert norm(_R12_QUOTED_RAW) == _R12_QUOTED_CONTENT
    # Content quoting the marker (no leading opener): kept whole — it can
    # only match its byte-identical self.
    assert norm(_R12_QUOTED_CONTENT) == _R12_QUOTED_CONTENT
    # Budget-forced raw (empty thinking, opener stored by
    # _make_full_assistant_content): router-consistent — content after the
    # first close, exactly what split_thinking_and_content yields.
    assert norm("<think>\n</think>The answer is 42.") == "The answer is 42."
    # Bare leading close with NO open: the router cannot align it
    # one-to-one against a stored raw (it could equally be content quoting
    # the marker at position 0) — kept raw, so it can never collide with a
    # plain resend of its suffix (honest MISS, fails pre-fix by reducing).
    assert norm("</think>The answer is 42.") == "</think>The answer is 42."
    # Degenerate unclosed raw is kept RAW (codex round 13, finding 1): the
    # router routes an unclosed stream entirely to reasoning with EMPTY
    # content, so the legacy bare-opener strip made it a forgeable suffix
    # ('<think>SECRET' == plain 'SECRET'), while reducing to '' would
    # collide with genuinely-empty content — it equals only itself.
    assert norm("<think>unclosed reasoning") == "<think>unclosed reasoning"
    # Thinking DISABLED stays a pure pass-through (round 3, finding 4).
    assert norm(_R12_QUOTED_CONTENT, active=False) == _R12_QUOTED_CONTENT
    # GLM shares the chatml think markers ('<think>' prefix, '</think>'
    # close) — identical reduction.
    assert norm(_R12_QUOTED_RAW, family="glm") == _R12_QUOTED_CONTENT


def test_r12_f1_budget_forced_leading_close_router_consistent():
    """The budget-forced shapes end-to-end on the STRICT path: the engine
    raw carries the opener and reduces exactly as the router splits it; a
    bare no-opener close (a shape this engine never stores) is kept whole
    and cannot be selected by a plain resend (fails pre-fix by matching)."""
    stub = _r10_match_stub("chatml")
    stored_engine = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>\n</think>The answer is 42."},
    ]
    assert stub._match(
        stored_engine, _r12_incoming("The answer is 42."),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True

    stored_bare = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "</think>The answer is 42."},
    ]
    assert stub._match(
        stored_bare, _r12_incoming("The answer is 42."),
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # Its byte-identical self still matches.
    assert stub._match(
        stored_bare, _r12_incoming("</think>The answer is 42."),
        last_assistant_wildcard=False, thinking_active=True,
    ) is True


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r12_f1_interrupted_exact_first_close(family):
    """codex round 11, finding 1 — the same lossy reduction lived in
    _interrupted_resend_equiv (stored-side split at the last close via
    _normalize_for_match): the forged suffix passed even the EXACT
    (anon/strict) rule (fails pre-fix)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stub = _r10_match_stub(family)

    def i_norm(content):
        # Normalize the incoming side exactly as _messages_match does.
        return MLXEngine._normalize_for_match(
            content, "assistant", family, True,
        )

    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_RAW, i_norm("PUBLIC SUFFIX"),
        exact=True, thinking_active=True,
    ) is False
    # The true content channel is exact-equal...
    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_RAW, i_norm(_R12_QUOTED_CONTENT),
        exact=True, thinking_active=True,
    ) is True
    # ...and the legacy protected pair keeps matching.
    assert stub._interrupted_resend_equiv(
        "<think>t</think>answer", i_norm("answer"),
        exact=True, thinking_active=True,
    ) is True


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r12_f1_interrupted_narrow_prefix_first_close(family):
    """NARROW (explicit-session) prefix mode under first-close semantics: a
    wire-truncated resend of the QUOTED content channel is a prefix of the
    stored channel (fails pre-fix — the rindex channel was 'PUBLIC SUFFIX'
    and the truncation is no prefix of it), while the forged suffix is
    not a prefix of anything that streamed."""
    stub = _r10_match_stub(family)
    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_RAW, "PUBLIC PREFIX </th",
        exact=False, thinking_active=True,
    ) is True
    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_RAW, "PUBLIC SUFFIX",
        exact=False, thinking_active=True,
    ) is False


def test_r12_f1_interrupted_disabled_and_ambiguous_kept_whole():
    """Stored sides without a raw shape are never split: with thinking
    DISABLED a literal '</think>' is a quote (fails pre-fix — the
    unconditional split cut at it), and under an ACTIVE contract a raw
    WITHOUT the leading opener (this engine always stores it) is ambiguous
    — kept whole, honest MISS instead of a forgeable suffix HIT."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stub = _r10_match_stub("chatml")
    # thinking disabled: quote, not boundary.
    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_CONTENT,
        MLXEngine._normalize_for_match(
            "PUBLIC SUFFIX", "assistant", "chatml", False,
        ),
        exact=True, thinking_active=False,
    ) is False
    assert stub._interrupted_resend_equiv(
        _R12_QUOTED_CONTENT,
        MLXEngine._normalize_for_match(
            _R12_QUOTED_CONTENT, "assistant", "chatml", False,
        ),
        exact=True, thinking_active=False,
    ) is True
    # active contract, no leading opener: ambiguous raw — honest MISS.
    assert stub._interrupted_resend_equiv(
        "t</think>C", "C", exact=True, thinking_active=True,
    ) is False


# ---------------------------------------------------------------------------
# R12F2 — streamed tool_calls indices stay contiguous past malformed blocks
# ---------------------------------------------------------------------------


def _r12_tc_deltas(events):
    """All streamed tool_calls delta entries, in emission order."""
    return [
        tc
        for ev in events
        for c in ev.get("choices", [])
        for tc in (c["delta"].get("tool_calls") or [])
    ]


_R12_TOOLS = [
    ToolDef(function=FunctionDef(name=n, parameters={}))
    for n in ("a", "b", "c", "good", "get_x")
]


@pytest.mark.asyncio
async def test_r12_f2_malformed_block_no_index_hole():
    """codex round 11, finding 2 repro (fails pre-fix with index 1): the
    malformed first block passes through as content, and the sole VALID
    call must stream with index 0 — no hole for index-reconstructing
    clients."""
    engine = _StreamStubEngine(
        ["<tool_call>bad</tool_call>"
         "<tool_call><function=good></function></tool_call>"],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False, tools=_R12_TOOLS,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    deltas = _r12_tc_deltas(events)
    # id+name chunk then arguments chunk, both index 0.
    assert [d["index"] for d in deltas] == [0, 0]
    assert deltas[0]["function"]["name"] == "good"
    assert deltas[0]["id"]
    # The malformed block is still preserved as raw content (U11).
    assert "<tool_call>bad</tool_call>" in _joined_delta(events, "content")
    assert _final_finish_reason(events) == "tool_calls"


@pytest.mark.asyncio
async def test_r12_f2_interleaved_invalid_valid_contiguous():
    """Valid calls interleaved with malformed blocks stream 0,1,2 (pre-fix
    0,2,4 — every start marker charged an index)."""
    engine = _StreamStubEngine(
        [
            "<tool_call><function=a></function></tool_call>",
            "<tool_call>junk</tool_call>",
            "<tool_call><function=b></function></tool_call>",
            "<tool_call>junk2</tool_call>",
            "<tool_call><function=c></function></tool_call>",
        ],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False, tools=_R12_TOOLS,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    name_deltas = [d for d in _r12_tc_deltas(events) if "id" in d]
    assert [d["index"] for d in name_deltas] == [0, 1, 2]
    assert [d["function"]["name"] for d in name_deltas] == ["a", "b", "c"]
    # Every arguments chunk shares its call's index — the full emitted
    # index sequence is pairwise (0,0,1,1,2,2).
    assert [d["index"] for d in _r12_tc_deltas(events)] == [0, 0, 1, 1, 2, 2]


@pytest.mark.asyncio
async def test_r12_f2_all_valid_unchanged():
    """Guard: an all-valid parallel-call stream keeps 0,1."""
    engine = _StreamStubEngine(
        [
            "<tool_call><function=a></function></tool_call>",
            "<tool_call><function=b></function></tool_call>",
        ],
        finish_reason="tool_calls",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False, tools=_R12_TOOLS,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    name_deltas = [d for d in _r12_tc_deltas(events) if "id" in d]
    assert [d["index"] for d in name_deltas] == [0, 1]
    assert [d["function"]["name"] for d in name_deltas] == ["a", "b"]


@pytest.mark.asyncio
async def test_r12_f2_end_of_stream_flush_index_contiguous():
    """The end-of-stream truncated-block flush (GLM's lenient no-closer
    parse) uses the post-validation index too: an earlier malformed closed
    block must not shift the flushed call to index 1 (fails pre-fix)."""
    engine = _StreamStubEngine(
        [
            "<tool_call></tool_call>",  # closed, empty name — unparseable
            "<tool_call>get_x\n<arg_key>q</arg_key><arg_value>\"hi\"</arg_value>",
        ],
        model_family="glm",
    )
    req = ChatCompletionRequest(
        model="m", stream=True, thinking=False, tools=_R12_TOOLS,
        messages=[ChatMessage(role="user", content="hi")],
    )
    events = await _collect_sse(engine, req)
    deltas = _r12_tc_deltas(events)
    assert [d["index"] for d in deltas] == [0, 0]
    assert deltas[0]["function"]["name"] == "get_x"
    assert _final_finish_reason(events) == "tool_calls"


# ---------------------------------------------------------------------------
# R14F1 — unclosed '<think>' raw is never a forgeable suffix (codex round 13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r14_f1_unclosed_thinking_forged_plain_resend_strict_miss(family):
    """codex round 13, finding 1 repro (fails pre-fix by matching): the
    no-close branch stripped the bare opener, so a stored INTERRUPTED
    '<think>SECRET' turn normalized equal to a plain incoming 'SECRET' —
    plain equality accepted BEFORE the interrupted-exact guard, selecting
    the anon session despite last_assistant_wildcard=False. Under the
    router contract an unclosed stream is ENTIRELY reasoning with EMPTY
    content: 'SECRET' never streamed, so this is an honest MISS."""
    stub = _r10_match_stub(family)
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET", "interrupted": True},
    ]
    incoming = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "SECRET"},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored, incoming,
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # The uninterrupted stored shape (budget cut / crash) misses too.
    stored_plain = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET"},
    ]
    assert stub._match(
        stored_plain, incoming,
        last_assistant_wildcard=False, thinking_active=True,
    ) is False


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r14_f1_unclosed_byte_identical_replay_still_matches(family):
    """Guard: the unclosed raw equals its byte-identical self — a verbatim
    replay keeps matching on the strict path (interrupted or not)."""
    stub = _r10_match_stub(family)
    incoming = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET"},
        {"role": "user", "content": "next"},
    ]
    for interrupted in (True, False):
        stored = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>SECRET",
             **({"interrupted": True} if interrupted else {})},
        ]
        assert stub._match(
            stored, incoming,
            last_assistant_wildcard=False, thinking_active=True,
        ) is True


def test_r14_f1_unclosed_never_equals_genuinely_empty():
    """Normalizing the unclosed raw to '' (the router-faithful content
    channel) would be another many-to-one — a genuinely-empty assistant
    turn must not equal '<think>SECRET' via PLAIN normalization, in either
    direction. (The interrupted-EXACT empty-resend equivalence is a
    separate, deliberate rule — see the wire-faithful guard below.)"""
    stub = _r10_match_stub("chatml")
    stored_unclosed = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET"},
    ]
    incoming_empty = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored_unclosed, incoming_empty,
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # Mirror direction: genuinely-empty stored vs incoming unclosed raw.
    stored_empty = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},
    ]
    incoming_unclosed = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET"},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored_empty, incoming_unclosed,
        last_assistant_wildcard=False, thinking_active=True,
    ) is False


def test_r14_f1_interrupted_wire_faithful_resends_still_match():
    """Guard: the LEGITIMATE resend shapes ride _interrupted_resend_equiv's
    own split, not plain equality. The wire showed NOTHING (unclosed
    thinking is entirely reasoning), so the client's faithful EMPTY resend
    matches the interrupted turn even on the strict path; the
    explicit-session narrow rule still accepts the thinking replayed as
    plain content."""
    stub = _r10_match_stub("chatml")
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>SECRET", "interrupted": True},
    ]
    empty_resend = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored, empty_resend,
        last_assistant_wildcard=False, thinking_active=True,
    ) is True
    # Explicit session: thinking replayed as plain content (narrow prefix).
    thinking_replay = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "SECRET"},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored, thinking_replay,
        last_assistant_wildcard=True, thinking_active=True,
    ) is True


# ---------------------------------------------------------------------------
# R14F2 — first-<tool_call> prefix shortcut removed (codex round 13)
# ---------------------------------------------------------------------------


def _r14_rehearsal(content_suffix):
    """Raw turn whose ONLY '<tool_call>' block is a rehearsal INSIDE the
    thinking segment (U12: not an extractable call), followed by real
    content-channel text."""
    return (
        "<think>plan <tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "seoul\n</parameter>\n</function>\n</tool_call></think>"
        + content_suffix
    )


_R14_CALL_XML = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nseoul\n"
    "</parameter>\n</function>\n</tool_call>"
)


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r14_f2_rehearsal_prefix_equal_divergent_content_strict_miss(family):
    """codex round 13, finding 2 repro (fails pre-fix by matching): both
    sides are byte-equal up to the FIRST '<tool_call>' — a rehearsal
    inside the thinking segment, so the structural call comparison sees
    None on both sides — and the removed prefix shortcut then accepted the
    message without ever comparing the content channels 'ALPHA' vs 'BETA',
    even with last_assistant_wildcard=False (wrong anon session
    selection)."""
    stub = _r10_match_stub(family)
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _r14_rehearsal("ALPHA")},
    ]
    incoming = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _r14_rehearsal("BETA")},
        {"role": "user", "content": "next"},
    ]
    assert stub._match(
        stored, incoming,
        last_assistant_wildcard=False, thinking_active=True,
    ) is False
    # Wildcard (explicit-session) path: with a LATER stored turn pinning
    # the position — the last-stored-assistant wildcard is a separate,
    # deliberate leniency — the divergence must fail there too (pre-fix
    # the shortcut accepted it on both paths).
    stored_mid = stored + [{"role": "user", "content": "next"}]
    assert stub._match(
        stored_mid, incoming,
        last_assistant_wildcard=True, thinking_active=True,
    ) is False


@pytest.mark.parametrize("family", ["chatml", "glm"])
def test_r14_f2_identical_rehearsal_messages_still_match(family):
    """Guard: identical full messages with a rehearsal call keep matching
    on BOTH paths (the reduction sees equal content channels)."""
    stub = _r10_match_stub(family)
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _r14_rehearsal("ALPHA")},
    ]
    incoming = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _r14_rehearsal("ALPHA")},
        {"role": "user", "content": "next"},
    ]
    for wildcard in (False, True):
        assert stub._match(
            stored, incoming,
            last_assistant_wildcard=wildcard, thinking_active=True,
        ) is True


def test_r14_f2_divergence_after_validated_call_block_strict_miss():
    """The shortcut also hid divergence AFTER a legitimate CLOSED call
    block in the content channel: equal 'text' prefixes and equal
    canonical calls, but trailing 'ALPHA' vs 'BETA' (fails pre-fix by
    matching on both paths). The normalized comparison — validated blocks
    stripped — now sees the divergent surrounding content."""
    stub = _r10_match_stub("chatml")
    tool = {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    stored = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": f"text\n\n{_R14_CALL_XML}\nALPHA"},
        tool,
    ]
    incoming = [
        stored[0],
        {"role": "assistant", "content": f"text\n\n{_R14_CALL_XML}\nBETA"},
        tool,
        {"role": "user", "content": "next"},
    ]
    for wildcard in (False, True):
        assert stub._match(
            stored, incoming,
            last_assistant_wildcard=wildcard, thinking_active=True,
        ) is False


def test_r14_f2_opencode_reconstruction_shapes_still_match():
    """Guard — the two shapes the shortcut was born for (initial commit:
    OpenCode moves tool-call XML from content into the structured
    tool_calls field) keep matching on BOTH paths via the modern
    machinery: (a) structural call comparison, (b) content compared with
    validated blocks stripped."""
    stub = _r10_match_stub("chatml")
    tc = {
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather",
                     "arguments": '{"city": "seoul"}'},
    }
    tool = {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    # 1. Pure tool call: stored raw XML vs incoming empty content + field.
    stored1 = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _R14_CALL_XML},
        tool,
    ]
    incoming1 = [
        stored1[0],
        {"role": "assistant", "content": "", "tool_calls": [tc]},
        tool,
        {"role": "user", "content": "next"},
    ]
    # 2. Text + tool call: stored 'text\n\n<tool_call>…' vs incoming 'text'.
    stored2 = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": f"text\n\n{_R14_CALL_XML}"},
        tool,
    ]
    incoming2 = [
        stored2[0],
        {"role": "assistant", "content": "text", "tool_calls": [tc]},
        tool,
        {"role": "user", "content": "next"},
    ]
    for stored, incoming in ((stored1, incoming1), (stored2, incoming2)):
        for wildcard in (False, True):
            assert stub._match(
                stored, incoming,
                last_assistant_wildcard=wildcard, thinking_active=True,
            ) is True


def test_r14_f2_channel_strip_exact_gemma4_markers_only():
    """Companion guard for the regex tightening: the tool-call CHANNEL
    strip matches ONLY the genuine piped gemma4 markers. The old
    optional-pipe pattern consumed 'block1 ... <tool_call>' across two
    chatml blocks (first opener through the SECOND opener) and left the
    second block's body behind as phantom content — masked pre-fix by the
    removed prefix shortcut, and breaking the strict-path equality for
    the OpenAI two-call reconstruction once the shortcut was gone."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    two_blocks = (
        "<tool_call>\n<function=a>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n</function>\n</tool_call>"
    )
    assert MLXEngine._normalize_for_match(
        two_blocks, "assistant", "chatml", True,
    ) == ""
    # The genuine gemma4 channel form still strips.
    assert MLXEngine._normalize_for_match(
        "before <|tool_call>call:a{}<tool_call|> after",
        "assistant", "gemma4", True,
    ) == "before  after"
