"""Harmony (OpenAI gpt-oss, model_type "gpt_oss") family support.

THE BUG (user report): gpt-oss fell through to the chatml fallback, so raw
Harmony channel markers leaked verbatim into ``content`` — an OpenCode user
literally saw::

    <|channel|>analysis<|message|>We need to respond...<|end|><|start|>
    assistant<|channel|>final<|message|>안녕하세요! ...

THE FIX (family "harmony", threaded through the same seams as gemma4/glm):

* detection: model_type containing "gpt_oss"/"gpt-oss" -> "harmony";
* streaming: a harmony ThinkingRouter branch — an incremental channel FSM
  (analysis -> reasoning_content, final/preamble -> content, functions-
  commentary -> a canonical tool block for the streaming tool FSM), with
  partial-marker holdback across chunk boundaries;
* batch: split_thinking_and_content drives the SAME router (batch == stream
  by construction); content_segments walks the positional twin (U12: a
  rehearsal inside analysis is never extractable);
* tool parse: _TOOL_MARKERS["harmony"] + _parse_harmony_tool_calls with U11
  per-block preservation (unparseable block -> raw passthrough);
* prompt build: strip_thinking_tags reduces marker-bearing assistant history
  to the content channel (the template RAISES on <|channel|> in content, and
  its own history semantics DROP analysis);
* cache HIT: _suffix_tokens_harmony renders the template-exact suffix
  ``<|end|><|start|>user<|message|>{q}<|end|><|start|>assistant`` (leading
  <|end|> = the C1 interrupted-turn closer), and
  _reconcile_harmony_recorded_eos reconciles the mlx-lm recorded terminal
  (U18 lesson: the fired chat-EOS is recorded into token_ids):
    - <|end|> or <|call|> tail  -> drop the duplicate leading closer,
    - <|return|>/<|endoftext|>  -> REPLACE (verified 1-token cache trim; the
      template renders history finals with <|end|>, and <|return|> must
      never be model input),
    - untrimmable -> honest MISS; trim failure -> fail-closed invalidation;
* budget: ThinkingBudgetProcessor is DISABLED for harmony (think_end_token
  -1): the analysis->final transition is a multi-token header sequence and a
  single forced token would corrupt output (see _detect_special_tokens).

TOKEN-EXACT GROUND TRUTH: the real gpt-oss chat_template.jinja + o200k
tokenizer (skip-if-missing local model dir, tokenizer/template only — no
weights). Measured facts the reconcile design follows:
  - history finals render ``...{content}<|end|>``, a natural stop records
    ``<|return|>`` (200002) — trim+replace makes the 2-turn/3-turn splice
    EQUAL full retokenization for analysis-free turns;
  - analysis is dropped from history by the template, but the KV physically
    contains what the model generated — the RETAINED-REASONING deviation
    (chatml campaign precedent): splice == full retokenization PLUS the
    recorded analysis block, byte-exact at every boundary (proven below);
  - a tool-call stop records ``<|call|>`` (200012), which history rendering
    keeps with the tool result ADJACENT (no <|end|>) — the reconcile drops
    the suffix's leading closer there.
"""

from __future__ import annotations

import json
import random
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_qwen_mtp import (  # noqa: E402
    LyingKV,
    MockKV,
    _drive_full,
    _generate_engine,
    _scripted_lm_stream,
)

from mlx_soloheaven.engine.tool_parser import (  # noqa: E402
    CHANNEL_CONTENT,
    CHANNEL_REASONING,
    ThinkingRouter,
    _harmony_walk,
    content_segments,
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
    strip_thinking_tags,
    thinking_router_active,
    try_extract_tool_name,
)

# Real-model tokenizer/template (tokenizer ONLY — weights are never loaded).
MODEL_DIR = Path(
    "~/.lmstudio/models/cloudyu/gpt-oss-120b-Fable-5-Distilled"
).expanduser()

# Harmony special-token ids (o200k_harmony; verified against the real
# tokenizer in the token-exact section below).
RETURN_ID = 200002
CHANNEL_ID = 200005
START_ID = 200006
END_ID = 200007
MESSAGE_ID = 200008
CALL_ID = 200012
ENDOFTEXT_ID = 199999

# The EXACT leaked sequence from the bug report.
LEAKED = (
    "<|channel|>analysis<|message|>We need to respond to the greeting."
    "<|end|><|start|>assistant<|channel|>final<|message|>"
    "안녕하세요! 무엇을 도와드릴까요?"
)
LEAKED_REASONING = "We need to respond to the greeting."
LEAKED_CONTENT = "안녕하세요! 무엇을 도와드릴까요?"

TOOL_BLOCK = (
    '<|channel|>commentary to=functions.get_weather <|constrain|>json'
    '<|message|>{"location": "Tokyo"}<|call|>'
)
TOOL_STREAM = (
    "<|channel|>analysis<|message|>Need the weather tool.<|end|>"
    "<|start|>assistant" + TOOL_BLOCK
)

ALL_MARKERS = (
    "<|start|>", "<|channel|>", "<|message|>",
    "<|end|>", "<|return|>", "<|call|>",
)


def _route(chunks, *, enable_thinking=None):
    r = ThinkingRouter(
        active=True, model_family="harmony", enable_thinking=enable_thinking,
    )
    segs = []
    for c in chunks:
        segs.extend(r.feed(c))
    segs.extend(r.flush())
    return segs


def _join(segs, channel):
    return "".join(t for ch, t in segs if ch == channel)


def _chunkings(text):
    """Deterministic set of segmentations for permutation tests."""
    rng = random.Random(42)
    outs = [[text], list(text)]
    for size in (2, 3, 5, 7, 11):
        outs.append([text[i:i + size] for i in range(0, len(text), size)])
    for _ in range(5):
        cuts = sorted(rng.sample(range(1, len(text)), min(9, len(text) - 1)))
        outs.append([
            text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])
        ])
    return outs


# ---------------------------------------------------------------------------
# Router: the leaked sequence routes cleanly, deterministically
# ---------------------------------------------------------------------------


def test_router_leaked_sequence_routes_reasoning_and_content():
    """The exact bug-report stream: analysis -> reasoning_content, final ->
    content, ZERO markers leaked on either channel. PRE-FIX: the chatml
    fallback passed the whole thing through as content."""
    segs = _route([LEAKED])
    assert _join(segs, CHANNEL_REASONING) == LEAKED_REASONING
    assert _join(segs, CHANNEL_CONTENT) == LEAKED_CONTENT
    for _, t in segs:
        for m in ALL_MARKERS:
            assert m not in t


def test_router_chunk_permutation_determinism():
    """Every segmentation of the same text yields identical channel output
    (markers split across chunk boundaries are held, never mis-routed)."""
    streams = [
        LEAKED,
        LEAKED + "<|return|>",
        TOOL_STREAM,
        "<|channel|>analysis<|message|>only thinking, never closed",
        "plain text with no markers at all",
    ]
    for text in streams:
        base = _route([text])
        for chunks in _chunkings(text):
            segs = _route(chunks)
            assert _join(segs, CHANNEL_REASONING) == _join(base, CHANNEL_REASONING)
            assert _join(segs, CHANNEL_CONTENT) == _join(base, CHANNEL_CONTENT)


def test_router_return_terminated_final():
    segs = _route([LEAKED + "<|return|>"])
    assert _join(segs, CHANNEL_CONTENT) == LEAKED_CONTENT
    assert _join(segs, CHANNEL_REASONING) == LEAKED_REASONING


def test_router_tool_block_reemitted_canonically_on_content():
    """A functions-commentary block is buffered and re-emitted as ONE
    canonical content segment (the streaming tool FSM's input)."""
    segs = _route([TOOL_STREAM])
    assert _join(segs, CHANNEL_REASONING) == "Need the weather tool."
    content_segs = [t for ch, t in segs if ch == CHANNEL_CONTENT]
    assert content_segs == [TOOL_BLOCK]


def test_router_end_terminated_tool_block_is_raw_content_not_a_call():
    """FINDING A (codex batch-6): a functions-commentary block closed by
    <|end|> (NOT the <|call|> tool-call eos) is NOT a tool call — it is emitted
    RAW as content (no fabricated <|call|>), never promoted, so it never
    executes and the positional twin agrees. A genuine tool call is ALWAYS
    <|call|>-terminated (P1-1 re-attaches <|call|> on a natural stop).

    SUPERSEDES the old <|end|>->  <|call|> canonicalization: that promoted a
    non-<|call|>-terminated block to an executable call, diverging batch from
    stream (batch saw no <|call|> terminator and parsed 0 calls)."""
    raw = TOOL_STREAM[: -len("<|call|>")] + "<|end|>"
    segs = _route([raw])
    # Analysis still routes to reasoning; the commentary block is RAW content
    # with its <|end|> terminator consumed (a message terminator).
    assert _join(segs, CHANNEL_REASONING) == "Need the weather tool."
    content = _join(segs, CHANNEL_CONTENT)
    assert content == TOOL_BLOCK[: -len("<|call|>")]
    _, calls = parse_tool_calls(content, model_family="harmony")
    assert calls == []


def test_router_preamble_commentary_without_recipient_is_content():
    text = (
        "<|channel|>commentary<|message|>I'll fetch that for you.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    )
    segs = _route([text])
    assert _join(segs, CHANNEL_CONTENT) == "I'll fetch that for you.Done."
    assert _join(segs, CHANNEL_REASONING) == ""


def test_router_analysis_always_reasoning_even_thinking_off():
    """U9-style policy, mirroring gemma4 thinking=False: the router stays
    marker-active (thinking_router_active) and analysis routes to
    reasoning_content — NEVER content."""
    assert thinking_router_active("harmony", False) is True
    segs = _route([LEAKED], enable_thinking=False)
    assert _join(segs, CHANNEL_REASONING) == LEAKED_REASONING
    assert _join(segs, CHANNEL_CONTENT) == LEAKED_CONTENT


def test_router_plain_text_without_markers_is_content():
    segs = _route(["Just a plain answer."])
    assert _join(segs, CHANNEL_CONTENT) == "Just a plain answer."
    assert _join(segs, CHANNEL_REASONING) == ""


def test_router_flush_drops_incomplete_header_keeps_partial_marker_text():
    # Incomplete header at stream end: structural — dropped, never leaked.
    segs = _route(["<|channel|>fin"])
    assert segs == []
    # A partial marker in CONTENT that never completes is real text.
    segs2 = _route([
        "<|channel|>final<|message|>tail ends with <|ret"
    ])
    assert _join(segs2, CHANNEL_CONTENT) == "tail ends with <|ret"


def test_router_truncated_tool_block_flushes_raw():
    """Stream ends inside a tool block: flushed RAW (no fabricated
    terminator) — downstream it stays an unparseable raw passthrough
    (chatml truncated-block parity, U11)."""
    raw = '<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"a": 1'
    segs = _route([raw])
    assert segs == [(CHANNEL_CONTENT, raw)]
    _, calls = parse_tool_calls(segs[0][1], model_family="harmony")
    assert calls == []


def test_router_missing_end_before_next_header_still_routes():
    """Degenerate: analysis runs straight into the next header without
    <|end|> — the header opener terminates the body (harmony messages are
    header-delimited)."""
    text = (
        "<|channel|>analysis<|message|>thinking"
        "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
    )
    segs = _route([text])
    assert _join(segs, CHANNEL_REASONING) == "thinking"
    assert _join(segs, CHANNEL_CONTENT) == "answer"


# ---------------------------------------------------------------------------
# Batch split == router (single-authority property, gemma4 round-6 pattern)
# ---------------------------------------------------------------------------


BATCH_STREAMS = [
    LEAKED,
    LEAKED + "<|return|>",
    TOOL_STREAM,
    TOOL_STREAM + "<|start|>assistant<|channel|>final<|message|>Sunny!<|return|>",
    "<|channel|>commentary<|message|>preamble<|end|>"
    "<|start|>assistant<|channel|>final<|message|>body<|return|>",
    "<|channel|>analysis<|message|>unclosed thinking only",
    "plain, marker-free",
    "",
    "<|channel|>final<|message|>direct answer, no analysis<|return|>",
]


@pytest.mark.parametrize("text", BATCH_STREAMS)
def test_batch_split_equals_router(text):
    thinking, content = split_thinking_and_content(text, model_family="harmony")
    for chunks in _chunkings(text) if text else [[""]]:
        segs = _route(chunks)
        assert (_join(segs, CHANNEL_REASONING).strip() or None) == thinking
        assert _join(segs, CHANNEL_CONTENT) == content


@pytest.mark.parametrize("text", BATCH_STREAMS)
def test_content_segments_union_matches_router_content(text):
    """The positional twin's union equals the router's content channel for
    canonical-terminator streams (the <|call|>-canonicalization divergence
    only exists for <|end|>-terminated tool blocks, tested separately)."""
    union = "".join(text[s:e] for s, e in content_segments(text, "harmony"))
    segs = _route([text] if text else [""])
    assert union == _join(segs, CHANNEL_CONTENT)


def test_content_segments_excludes_analysis_rehearsal_u12():
    """A tool-call-looking rehearsal INSIDE analysis is never part of the
    content channel — not extractable, not a residual marker (U12)."""
    text = (
        '<|channel|>analysis<|message|>maybe {"location": "T"} via '
        'functions.get_weather... or <tool_call><function=x></function>'
        '</tool_call><|end|><|start|>assistant<|channel|>final<|message|>'
        'no call made<|return|>'
    )
    union = "".join(text[s:e] for s, e in content_segments(text, "harmony"))
    assert union == "no call made"
    _, calls = parse_tool_calls(union, model_family="harmony")
    assert calls == []


# ---------------------------------------------------------------------------
# parse_tool_calls harmony branch (U11 per-block rules)
# ---------------------------------------------------------------------------


def test_parse_harmony_tool_call_block():
    text, calls = parse_tool_calls(TOOL_BLOCK, model_family="harmony")
    assert text == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_parse_harmony_end_terminator_not_a_call_finding_a():
    """FINDING A: ONLY a <|call|>-terminated commentary block is an executable
    tool call. An <|end|>-terminated block is NOT parsed as a call — it is
    retained raw (U11 passthrough). A real generated tool call ALWAYS ends
    with the <|call|> tool-call eos, so this is fail-closed and safe.

    SUPERSEDES test_parse_harmony_accepts_end_terminator (which codified the
    now-removed <|end|> degenerate-terminator acceptance)."""
    raw = TOOL_BLOCK[: -len("<|call|>")] + "<|end|>"
    text, calls = parse_tool_calls(raw, model_family="harmony")
    assert calls == []
    assert text == raw


def test_parse_harmony_unparseable_block_raw_passthrough():
    """U11: garbage body / missing recipient / non-object args -> the block
    is retained byte-identical, never silently deleted, no phantom call."""
    for raw in (
        '<|channel|>commentary to=functions.f<|message|>not json<|call|>',
        '<|channel|>commentary<|message|>{"a": 1}<|call|>',  # no recipient
        '<|channel|>commentary to=functions.f<|message|>[1, 2]<|call|>',  # non-object
        '<|channel|>commentary to=functions.f<|message|>{"a": 1',  # unterminated
    ):
        text, calls = parse_tool_calls(raw, model_family="harmony")
        assert calls == []
        assert text == raw


def test_parse_harmony_mixed_blocks_preserve_order():
    good = '<|channel|>commentary to=functions.ok<|message|>{"x": 1}<|call|>'
    bad = '<|channel|>commentary to=functions.bad<|message|>oops<|call|>'
    text, calls = parse_tool_calls(
        "before " + good + " mid " + bad + " after", model_family="harmony",
    )
    assert len(calls) == 1 and calls[0]["function"]["name"] == "ok"
    assert text == "before  mid " + bad + " after"


def test_harmony_tool_markers_and_name_extraction():
    start, end = get_tool_markers("harmony")
    assert start == "<|channel|>commentary"
    assert end == "<|call|>"
    assert try_extract_tool_name(
        " to=functions.web.search-v2 <|constrain|>json<|message|>", "harmony",
    ) == "web.search-v2"
    assert try_extract_tool_name(" nonsense", "harmony") is None


# ---------------------------------------------------------------------------
# Streaming OpenAI-compat wiring: reasoning deltas + tool_calls deltas
# ---------------------------------------------------------------------------


from test_reasoning_channel import (  # noqa: E402
    _StreamEngine,
    _collect,
    _delta_field,
    _req,
)
from mlx_soloheaven.api.schemas import (  # noqa: E402
    ChatMessage,
    ChatCompletionRequest,
    FunctionDef,
    ToolDef,
)


@pytest.mark.asyncio
async def test_stream_harmony_leaked_sequence_clean_channels():
    """The bug-report stream over the SSE surface: reasoning_content gets the
    analysis, content gets the final text, no marker ever leaks."""
    chunks = [
        "<|channel|>", "analysis", "<|mess", "age|>We need to respond",
        " to the greeting.", "<|end|><|start|>assist", "ant<|channel|>final",
        "<|message|>안녕하세요! ", "무엇을 도와드릴까요?",
    ]
    engine = _StreamEngine("harmony", chunks)
    events = await _collect(engine, _req("harmony"))
    assert _delta_field(events, "reasoning_content") == LEAKED_REASONING
    assert _delta_field(events, "content") == LEAKED_CONTENT
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {})
            for field in ("content", "reasoning_content"):
                for m in ALL_MARKERS:
                    assert m not in (delta.get(field) or "")


@pytest.mark.asyncio
async def test_stream_harmony_tool_call_emits_tool_calls_delta():
    """A commentary tool block becomes proper tool_calls deltas (id+name
    first, then arguments) with finish_reason=tool_calls — no phantom."""
    chunks = [
        "<|channel|>analysis<|message|>Need weather.<|end|>",
        "<|start|>assistant<|channel|>commentary to=functions.get_",
        "weather <|constrain|>json<|message|>{\"location\": ",
        "\"Tokyo\"}<|call|>",
    ]
    engine = _StreamEngine("harmony", chunks)
    tools = [ToolDef(function=FunctionDef(name="get_weather", parameters={}))]
    events = await _collect(engine, _req("harmony", tools=tools))
    tc_deltas = [
        tc
        for ev in events
        for choice in ev.get("choices", [])
        for tc in (choice.get("delta", {}).get("tool_calls") or [])
    ]
    assert tc_deltas, "no tool_calls delta emitted"
    assert tc_deltas[0]["index"] == 0
    assert tc_deltas[0]["function"]["name"] == "get_weather"
    args = "".join(
        tc["function"].get("arguments", "") for tc in tc_deltas
    )
    assert json.loads(args) == {"location": "Tokyo"}
    finishes = [
        choice.get("finish_reason")
        for ev in events for choice in ev.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finishes == ["tool_calls"]
    assert _delta_field(events, "reasoning_content") == "Need weather."


@pytest.mark.asyncio
async def test_stream_harmony_malformed_tool_block_no_phantom():
    """U11/round-4: a commentary block whose body fails to parse produces NO
    tool_calls delta — the raw block passes through as content."""
    bad = '<|channel|>commentary to=functions.f<|message|>not json<|call|>'
    engine = _StreamEngine("harmony", [bad])
    tools = [ToolDef(function=FunctionDef(name="f", parameters={}))]
    events = await _collect(engine, _req("harmony", tools=tools))
    tc_deltas = [
        tc
        for ev in events
        for choice in ev.get("choices", [])
        for tc in (choice.get("delta", {}).get("tool_calls") or [])
    ]
    assert tc_deltas == []
    assert _delta_field(events, "content") == bad
    finishes = [
        choice.get("finish_reason")
        for ev in events for choice in ev.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finishes == ["stop"]


@pytest.mark.asyncio
async def test_stream_harmony_rehearsal_in_analysis_not_a_call():
    """U12 over the wire: a rehearsed call inside analysis reaches neither
    the tool FSM nor content."""
    chunks = [
        '<|channel|>analysis<|message|>I could run functions.get_weather '
        'with {"location": "Tokyo"}...<|end|>',
        '<|start|>assistant<|channel|>final<|message|>No tool needed.',
    ]
    engine = _StreamEngine("harmony", chunks)
    tools = [ToolDef(function=FunctionDef(name="get_weather", parameters={}))]
    events = await _collect(engine, _req("harmony", tools=tools))
    tc_deltas = [
        tc
        for ev in events
        for choice in ev.get("choices", [])
        for tc in (choice.get("delta", {}).get("tool_calls") or [])
    ]
    assert tc_deltas == []
    assert _delta_field(events, "content") == "No tool needed."


@pytest.mark.asyncio
async def test_stream_harmony_thinking_false_analysis_stays_reasoning():
    """thinking=False: analysis still streams as reasoning_content (never
    content) — the gemma4 marker-active policy mirrored."""
    engine = _StreamEngine("harmony", [LEAKED + "<|return|>"])
    req = ChatCompletionRequest(
        model="test-harmony",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        thinking=False,
    )
    events = await _collect(engine, req)
    assert _delta_field(events, "reasoning_content") == LEAKED_REASONING
    assert _delta_field(events, "content") == LEAKED_CONTENT


# ---------------------------------------------------------------------------
# Non-streaming _sync_completion wiring (batch parse mirrors the stream)
# ---------------------------------------------------------------------------


from test_reasoning_channel import _SyncEngine  # noqa: E402


def test_sync_harmony_sets_reasoning_content_and_clean_content():
    engine = _SyncEngine("harmony", LEAKED + "<|return|>")
    req = ChatCompletionRequest(
        model="test-harmony",
        messages=[ChatMessage(role="user", content="안녕")],
        stream=False,
        thinking=True,
    )
    from mlx_soloheaven.api.openai_compat import _sync_completion

    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.content == LEAKED_CONTENT
    assert msg.reasoning_content == LEAKED_REASONING
    for m in ALL_MARKERS:
        assert m not in (msg.content or "")


def test_sync_harmony_tool_call_parses_from_content_not_reasoning():
    engine = _SyncEngine("harmony", TOOL_STREAM)
    req = ChatCompletionRequest(
        model="test-harmony",
        messages=[ChatMessage(role="user", content="weather?")],
        tools=[ToolDef(function=FunctionDef(name="get_weather", parameters={}))],
        stream=False,
        thinking=True,
    )
    from mlx_soloheaven.api.openai_compat import _sync_completion

    resp = _sync_completion(req, engine)
    msg = resp.choices[0].message
    assert msg.reasoning_content == "Need the weather tool."
    assert msg.tool_calls and len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "get_weather"
    assert json.loads(msg.tool_calls[0].function.arguments) == {
        "location": "Tokyo",
    }
    assert resp.choices[0].finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# strip_thinking_tags: input-history hygiene (the template raises on markers)
# ---------------------------------------------------------------------------


def test_strip_thinking_tags_harmony_reduces_to_content_channel():
    msgs = strip_thinking_tags(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": LEAKED + "<|return|>"},
        ],
        model_family="harmony",
    )
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == LEAKED_CONTENT
    for m in ALL_MARKERS:
        assert m not in msgs[1]["content"]


def test_strip_thinking_tags_harmony_drops_tool_xml_from_content():
    raw = TOOL_STREAM + "<|start|>assistant<|channel|>final<|message|>Sunny!<|return|>"
    msgs = strip_thinking_tags(
        [{"role": "assistant", "content": raw}], model_family="harmony",
    )
    assert msgs[0]["content"] == "Sunny!"


def test_strip_thinking_tags_harmony_plain_content_untouched():
    msgs = strip_thinking_tags(
        [{"role": "assistant", "content": "plain reply"}],
        model_family="harmony",
    )
    assert msgs[0]["content"] == "plain reply"


def test_strip_thinking_tags_other_families_unaffected():
    """Regression guard: the harmony branch never fires for other families."""
    msgs = strip_thinking_tags(
        [{"role": "assistant", "content": "<think>t</think>answer"}],
        model_family="chatml",
    )
    assert msgs[0]["content"] == "answer"


# ---------------------------------------------------------------------------
# Family detection / special tokens / budget-processor decision
# ---------------------------------------------------------------------------


def _family_of(model_type: str) -> str:
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng._model_type = model_type
    return eng._detect_model_family()


def test_family_detection():
    assert _family_of("gpt_oss") == "harmony"
    assert _family_of("GPT-OSS") == "harmony"
    assert _family_of("gpt_oss_moe") == "harmony"
    # Existing families untouched.
    assert _family_of("gemma4") == "gemma4"
    assert _family_of("glm4_moe") == "glm"
    assert _family_of("qwen3_5_moe") == "chatml"


def test_detect_special_tokens_harmony_disables_budget_token():
    """harmony: think_end_token is forced to -1 (no single-token analysis
    close exists), which disables the ThinkingBudgetProcessor gate
    (use_thinking and budget>0 and think_end_token>=0) — the documented
    safe fallback instead of forcing a token that would corrupt a
    direct-to-final answer."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    eng.model_id = "test-harmony"
    eng.cfg = SimpleNamespace(think_end_token=999)  # even a user-set id
    eng.tokenizer = SimpleNamespace(get_vocab=lambda: {"<|end|>": END_ID})
    eng._detect_special_tokens()
    assert eng.cfg.think_end_token == -1


def test_suffix_template_rev_harmony_stamped_others_unchanged():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    # P2-5: the harmony revision carries the base revision PLUS the current
    # template date (the system prefix embeds ``Current date: {today}``).
    rev = eng._suffix_template_rev()
    assert rev.startswith(MLXEngine._HARMONY_SUFFIX_REV + "|date=")
    assert rev.endswith(eng._harmony_template_date())
    base = MLXEngine._prompt_fingerprint(None, True)
    assert MLXEngine._prompt_fingerprint(None, True, template_rev=rev) != base
    # chatml/gemma4 fingerprints stay byte-identical (no spurious MISS).
    eng.model_family = "chatml"
    assert eng._suffix_template_rev() == ""
    eng.model_family = "gemma4"
    assert eng._suffix_template_rev() == ""
    eng.model_family = "glm"
    assert eng._suffix_template_rev() == MLXEngine._GLM_SUFFIX_REV


def test_prompt_fingerprint_harmony_date_component_p2_5(monkeypatch):
    """P2-5: the harmony fingerprint carries the template date, so a session
    crossing midnight takes an honest MISS (its stored ids embed yesterday's
    ``Current date:`` line while a fresh retokenization renders today's).
    Same day -> same fingerprint (HIT); a different day -> different
    fingerprint (MISS/cold rebuild)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"

    def _fp():
        return MLXEngine._prompt_fingerprint(
            None, True, template_rev=eng._suffix_template_rev(),
        )

    # Same day: identical fingerprint (HIT).
    monkeypatch.setattr(
        MLXEngine, "_harmony_template_date", staticmethod(lambda: "2026-07-19"),
    )
    same_day_a = _fp()
    same_day_b = _fp()
    assert same_day_a == same_day_b

    # Next day: the date rolls, the fingerprint MUST differ (honest MISS).
    monkeypatch.setattr(
        MLXEngine, "_harmony_template_date", staticmethod(lambda: "2026-07-20"),
    )
    next_day = _fp()
    assert next_day != same_day_a

    # Other families never embed a date — their revision is date-free.
    eng.model_family = "chatml"
    assert eng._suffix_template_rev() == ""


# ---------------------------------------------------------------------------
# Token-exact ground truth (real gpt-oss tokenizer + template; skip-if-missing)
# ---------------------------------------------------------------------------


requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists()
    or not (MODEL_DIR / "chat_template.jinja").exists(),
    reason=f"local gpt-oss model dir not found: {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def hf_tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))


def _tmpl(tok, messages, tools=None):
    r = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=True,
    )
    if hasattr(r, "input_ids"):
        r = r.input_ids
    return list(r)


def _enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def _harmony_shell(tok):
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    eng.tokenizer = tok
    return eng


def _cs(token_ids, cache):
    return SimpleNamespace(
        token_ids=list(token_ids), cache=cache,
        mtp_last_hidden=None, mtp_hidden_offset=None,
    )


SYS_U1 = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "hi"},
]
Q2 = "and again?"
Q3 = "third question 세 번째"
A1 = "hello there"
A2 = "second answer"


@requires_model
def test_marker_ids_match_real_vocab(hf_tok):
    vocab = hf_tok.get_vocab()
    assert vocab["<|return|>"] == RETURN_ID
    assert vocab["<|end|>"] == END_ID
    assert vocab["<|call|>"] == CALL_ID
    assert vocab["<|start|>"] == START_ID
    assert vocab["<|channel|>"] == CHANNEL_ID
    assert vocab["<|message|>"] == MESSAGE_ID
    # The generation config's chat-EOS set the recording contract fires on.
    gc = json.loads((MODEL_DIR / "generation_config.json").read_text())
    assert set(gc["eos_token_id"]) == {RETURN_ID, ENDOFTEXT_ID, CALL_ID}


@requires_model
def test_generation_prompt_ends_with_start_assistant(hf_tok):
    p = _tmpl(hf_tok, SYS_U1)
    assert p[-1] == _enc(hf_tok, "assistant")[0]
    assert p[-2] == START_ID
    tail = hf_tok.decode(p[-6:])
    assert tail.endswith("<|message|>hi<|end|><|start|>assistant")


@requires_model
def test_suffix_builder_user_turn_shape(hf_tok):
    eng = _harmony_shell(hf_tok)
    suffix = eng._suffix_tokens_harmony(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    assert suffix == _enc(
        hf_tok, f"<|end|><|start|>user<|message|>{Q2}<|end|><|start|>assistant",
    )
    assert suffix[0] == END_ID
    # thinking flag is a no-op for harmony (template has no rendering for it).
    assert suffix == eng._suffix_tokens_harmony(
        [{"role": "user", "content": Q2}], thinking=False,
    )


@requires_model
def test_two_turn_splice_token_exact_u18(hf_tok):
    """THE U18-class check: stored ids = turn-1 prompt + generated final +
    the RECORDED <|return|> terminal. The reconcile trims the foreign
    <|return|> (history renders <|end|>) and keeps the suffix's leading
    <|end|> — the splice equals apply_chat_template over the full 2-turn
    conversation, byte-exact. Without the reconcile it provably diverges."""
    eng = _harmony_shell(hf_tok)
    p1 = _tmpl(hf_tok, SYS_U1)
    stored = p1 + _enc(hf_tok, f"<|channel|>final<|message|>{A1}") + [RETURN_ID]
    cache = [MockKV() for _ in range(3)]
    for c in cache:
        c.offset = len(stored)
    cs = _cs(stored, cache)

    suffix = eng._suffix_tokens_harmony(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    out = eng._reconcile_harmony_recorded_eos(suffix, cs, "s")
    assert out is not None and out[0] == END_ID  # replace keeps the closer
    assert cs.token_ids == stored[:-1]  # <|return|> trimmed
    assert all(c.offset == len(stored) - 1 for c in cache)

    full = _tmpl(hf_tok, SYS_U1 + [
        {"role": "assistant", "content": A1},
        {"role": "user", "content": Q2},
    ])
    assert cs.token_ids + out == full

    # Regression documentation: the unreconciled splice diverges (recorded
    # <|return|> where the template renders <|end|>).
    assert stored + suffix != full


@requires_model
def test_three_turn_no_drift(hf_tok):
    """Two natural <|return|> stops in a row: reconciled splices stay equal
    to full retokenization — no per-turn drift."""
    eng = _harmony_shell(hf_tok)
    p1 = _tmpl(hf_tok, SYS_U1)
    stored = p1 + _enc(hf_tok, f"<|channel|>final<|message|>{A1}") + [RETURN_ID]
    cache = [MockKV()]
    cache[0].offset = len(stored)
    cs = _cs(stored, cache)

    s2 = eng._reconcile_harmony_recorded_eos(
        eng._suffix_tokens_harmony([{"role": "user", "content": Q2}], True),
        cs, "s",
    )
    cs.token_ids = cs.token_ids + s2 + _enc(
        hf_tok, f"<|channel|>final<|message|>{A2}",
    ) + [RETURN_ID]
    cache[0].offset = len(cs.token_ids)

    s3 = eng._reconcile_harmony_recorded_eos(
        eng._suffix_tokens_harmony([{"role": "user", "content": Q3}], True),
        cs, "s",
    )
    spliced = cs.token_ids + s3
    full = _tmpl(hf_tok, SYS_U1 + [
        {"role": "assistant", "content": A1},
        {"role": "user", "content": Q2},
        {"role": "assistant", "content": A2},
        {"role": "user", "content": Q3},
    ])
    assert spliced == full


@requires_model
def test_analysis_retained_splice_boundary_exact(hf_tok):
    """RETAINED-REASONING invariant (the chatml-campaign deviation applied
    to harmony): with a generated analysis block the cached stream keeps it
    (the KV physically processed it), so full-sequence equality cannot hold
    — but removing exactly the recorded analysis block from the splice
    yields the full retokenization byte-for-byte, proving every boundary
    (prompt end, analysis block, final body, next-turn suffix) token-exact."""
    eng = _harmony_shell(hf_tok)
    p1 = _tmpl(hf_tok, SYS_U1)
    ana = _enc(
        hf_tok,
        "<|channel|>analysis<|message|>Think hard.<|end|><|start|>assistant",
    )
    stored = p1 + ana + _enc(
        hf_tok, f"<|channel|>final<|message|>{A1}",
    ) + [RETURN_ID]
    cs = _cs(stored, [MockKV()])
    cs.cache[0].offset = len(stored)

    out = eng._reconcile_harmony_recorded_eos(
        eng._suffix_tokens_harmony([{"role": "user", "content": Q2}], True),
        cs, "s",
    )
    spliced = cs.token_ids + out
    full = _tmpl(hf_tok, SYS_U1 + [
        {"role": "assistant", "content": A1},
        {"role": "user", "content": Q2},
    ])
    n0 = len(p1)
    assert spliced[n0:n0 + len(ana)] == ana
    assert spliced[:n0] + spliced[n0 + len(ana):] == full


@requires_model
def test_tool_result_suffix_after_call_stop_token_exact(hf_tok):
    """Natural tool-call stop: the recorded <|call|> is template-native
    (history keeps the tool result ADJACENT — no <|end|>), so the reconcile
    drops the suffix's leading closer and the tool-result rendering matches
    the template byte-for-byte (incl. tojson content encoding — quotes,
    newline, Korean). The tool name resolves via prior_messages
    (last_tool_call.name semantics)."""
    eng = _harmony_shell(hf_tok)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]
    p1 = _tmpl(hf_tok, SYS_U1, tools=tools)
    gen = _enc(
        hf_tok,
        '<|channel|>analysis<|message|>Need tool.<|end|><|start|>assistant'
        '<|channel|>commentary to=functions.get_weather <|constrain|>json'
        '<|message|>{"location": "Tokyo"}',
    ) + [CALL_ID]
    stored = p1 + gen
    cs = _cs(stored, [MockKV()])
    cs.cache[0].offset = len(stored)

    result_content = 'sunny "quoted"\n한국어 25C'
    prior = SYS_U1 + [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "get_weather",
                         "arguments": '{"location": "Tokyo"}'},
        }],
    }]
    suffix = eng._suffix_tokens_harmony(
        [{"role": "tool", "tool_call_id": "c1", "content": result_content}],
        thinking=True, prior_messages=prior,
    )
    out = eng._reconcile_harmony_recorded_eos(suffix, cs, "s")
    assert out == suffix[1:]  # leading <|end|> dropped after <|call|>
    assert cs.token_ids == stored  # nothing trimmed

    spliced = cs.token_ids + out
    full = _tmpl(hf_tok, SYS_U1 + [
        {
            "role": "assistant",
            "thinking": "Need tool.",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": {"location": "Tokyo"}},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "content": result_content},
    ], tools=tools)
    # The generated tool-call HEADER form differs from the template's
    # history form (to= position / <|constrain|>) — a retained-wire
    # deviation like the analysis block — but the SUFFIX REGION (tool
    # result + generation prompt) must equal the template's rendering.
    assert spliced[-len(out):] == full[-len(out):]
    # And the tool-result text itself round-trips the tojson encoding.
    assert hf_tok.decode(out).startswith(
        '<|start|>functions.get_weather to=assistant<|channel|>commentary'
        '<|message|>"sunny \\"quoted\\"\\n한국어 25C"<|end|>'
    )


@requires_model
def test_reconcile_edge_cases(hf_tok):
    eng = _harmony_shell(hf_tok)
    suffix = [END_ID, START_ID, 1428, MESSAGE_ID, 3686]

    # <|end|> tail (stop-sequence close): dedupe, cache untouched.
    cs = _cs([1, 2, END_ID], [MockKV()])
    assert eng._reconcile_harmony_recorded_eos(suffix, cs, "s") == suffix[1:]
    assert cs.token_ids == [1, 2, END_ID]

    # Interrupted content tail: splice as built (the closer closes it).
    cs2 = _cs([1, 2, 3], [MockKV()])
    assert eng._reconcile_harmony_recorded_eos(suffix, cs2, "s") == suffix
    assert cs2.token_ids == [1, 2, 3]

    # <|endoftext|> tail: foreign — trimmed like <|return|>.
    cache3 = [MockKV()]
    cache3[0].offset = 3
    cs3 = _cs([1, 2, ENDOFTEXT_ID], cache3)
    assert eng._reconcile_harmony_recorded_eos(suffix, cs3, "s") == suffix
    assert cs3.token_ids == [1, 2]

    # Empty suffix / empty stored: untouched.
    assert eng._reconcile_harmony_recorded_eos([], cs2, "s") == []
    cs_empty = _cs([], [MockKV()])
    assert eng._reconcile_harmony_recorded_eos(suffix, cs_empty, "s") == suffix

    # Non-harmony family: passthrough.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    other = MLXEngine.__new__(MLXEngine)
    other.model_family = "chatml"
    other.tokenizer = hf_tok
    cs4 = _cs([1, RETURN_ID], [MockKV()])
    assert other._reconcile_harmony_recorded_eos(suffix, cs4, "s") == suffix


@requires_model
def test_reconcile_untrimmable_demotes_to_miss(hf_tok):
    eng = _harmony_shell(hf_tok)
    stored = [1, 2, RETURN_ID]
    untrimmable = SimpleNamespace(offset=len(stored))  # no trim()
    cs = _cs(stored, [untrimmable])
    suffix = [END_ID, 5]
    assert eng._reconcile_harmony_recorded_eos(suffix, cs, "s") is None
    assert cs.token_ids == stored
    assert cs.cache is not None
    assert untrimmable.offset == len(stored)


@requires_model
def test_reconcile_trim_failure_invalidates_fail_closed(hf_tok):
    eng = _harmony_shell(hf_tok)
    stored = [1, 2, RETURN_ID]
    good, liar = MockKV(), LyingKV()
    good.offset = liar.offset = len(stored)
    cs = _cs(stored, [good, liar])
    assert eng._reconcile_harmony_recorded_eos([END_ID, 5], cs, "s") is None
    assert cs.cache is None
    assert cs.token_ids is None


def test_reconcile_degraded_vocab_foreign_tail_honest_miss():
    """P2-4 CORRECTED POLARITY: marker detection failed AND no reliable eos
    set — the recorded tail could be a FOREIGN eos (<|return|>) we cannot
    verify, so returning the suffix as-is would leave <|return|> as model
    input. Honest MISS (None), NOT a corrupt splice (the pre-fix test
    codified the splice)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    eng.tokenizer = SimpleNamespace(
        get_vocab=lambda: {},
        encode=lambda t, add_special_tokens=False: [],
        eos_token_ids=[],
    )
    cs = _cs([1, RETURN_ID], [MockKV()])
    assert eng._reconcile_harmony_recorded_eos([END_ID, 5], cs, "s") is None
    # Cache untouched (the honest MISS rebuild replaces it, not this path).
    assert cs.token_ids == [1, RETURN_ID]


def test_reconcile_degraded_vocab_content_tail_splices_when_provable():
    """P2-4: marker detection failed BUT a reliable eos set is available and
    the recorded tail is provably NOT a foreign eos — a normal content token
    (interrupted turn). The leading closer legitimately closes it: splice as
    built (the genuinely-safe unknown-tail case)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    # get_vocab empty -> the <|end|> MARKER probe fails, but eos_token_ids
    # still surfaces the chat-EOS set, so a content tail is provably safe.
    eng.tokenizer = SimpleNamespace(
        get_vocab=lambda: {},
        encode=lambda t, add_special_tokens=False: [],
        eos_token_ids=[RETURN_ID, ENDOFTEXT_ID, CALL_ID],
    )
    cs = _cs([1, 2, 3], [MockKV()])  # tail 3 is a content token
    assert eng._reconcile_harmony_recorded_eos([END_ID, 5], cs, "s") == [END_ID, 5]
    assert cs.token_ids == [1, 2, 3]

    # A foreign-eos tail under the SAME reliable set -> still honest MISS.
    cs2 = _cs([1, 2, RETURN_ID], [MockKV()])
    assert eng._reconcile_harmony_recorded_eos([END_ID, 5], cs2, "s") is None
    assert cs2.token_ids == [1, 2, RETURN_ID]


# ---------------------------------------------------------------------------
# Engine wiring — the reconcile runs on the real HIT path (mlx-lm harness)
# ---------------------------------------------------------------------------


def _harmony_hit_engine(stored, suffix, *, cache=None):
    """HIT-capable mlx-lm-path harness re-labeled harmony (conventions from
    test_glm_hit_suffix_eos._glm_hit_engine): vocab resolves the harmony
    markers, and the session contract is restamped with the harmony
    suffix-builder revision."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    if cache is None:
        cache = [MockKV() for _ in range(5)]
        for c in cache:
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)
    eng.model_family = "harmony"
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {
            "<|end|>": END_ID,
            "<|call|>": CALL_ID,
            "<|return|>": RETURN_ID,
        },
        encode=lambda text, add_special_tokens=False: [],
    )
    # The engine passes prior_messages for harmony — widen the stub.
    eng._suffix_tokens = (
        lambda msgs, thinking=True, prior_messages=None: list(suffix)
    )
    sess = eng._sessions["s"]
    # P2-5: stamp with the SAME (date-inclusive) revision the engine computes
    # on the HIT so same-day fingerprints match (the date component is folded
    # into _suffix_template_rev for harmony).
    sess.prompt_fingerprint = MLXEngine._prompt_fingerprint(
        None, False, template_rev=eng._suffix_template_rev(),
    )
    return eng, cs, messages


def test_hit_replaces_recorded_return_with_end(monkeypatch):
    """Stored tail is <|return|> (natural stop): the HIT trims it from
    cache + token_ids and splices the suffix WITH its leading <|end|>."""
    stored = [1, 2, 3, 4, RETURN_ID]
    eng, cs, messages = _harmony_hit_engine(
        stored, suffix=[END_ID, 55, 66],
    )
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [END_ID, 55, 66]
    assert cs.token_ids == stored[:-1] + [END_ID, 55, 66] + [101, 102]
    assert all(c.offset == len(cs.token_ids) for c in cs.cache)


def test_hit_dedupes_leading_end_after_recorded_end(monkeypatch):
    stored = [1, 2, 3, 4, END_ID]
    eng, cs, messages = _harmony_hit_engine(stored, suffix=[END_ID, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [55, 66]
    assert cs.token_ids == stored + [55, 66] + [101, 102]


def test_hit_call_tail_keeps_adjacency(monkeypatch):
    """Stored tail is <|call|> (natural tool-call stop): the suffix's
    leading closer is dropped — history keeps the next message adjacent."""
    stored = [1, 2, 3, 4, CALL_ID]
    eng, cs, messages = _harmony_hit_engine(stored, suffix=[END_ID, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [55, 66]
    assert cs.token_ids == stored + [55, 66] + [101, 102]


def test_hit_interrupted_tail_splices_with_closer(monkeypatch):
    """Interrupted commit (content tail): the leading <|end|> closes the
    unterminated turn (C1 NOT_REQUIRED contract)."""
    stored = [1, 2, 3, 4, 5]
    eng, cs, messages = _harmony_hit_engine(stored, suffix=[END_ID, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [END_ID, 55, 66]
    assert cs.token_ids == stored + [END_ID, 55, 66] + [101, 102]


def test_hit_foreign_eos_untrimmable_honest_miss(monkeypatch):
    stored = [1, 2, 3, 4, RETURN_ID]
    untrimmable = [SimpleNamespace(offset=len(stored)) for _ in range(3)]
    eng, cs, messages = _harmony_hit_engine(
        stored, suffix=[END_ID, 55, 66], cache=untrimmable,
    )
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [11, 12, 13]  # honest MISS re-prefill
    new_cs = eng._sessions["s"].cache_state
    assert new_cs is not cs
    assert new_cs.token_ids == [11, 12, 13, 101, 102]


def test_try_close_interrupted_turn_not_required_for_harmony():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine
    from mlx_soloheaven.engine.session_cache_types import TurnCloseResult

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    assert eng._try_close_interrupted_turn(
        "s", _cs([1, 2], [MockKV()]),
    ) is TurnCloseResult.NOT_REQUIRED


# ---------------------------------------------------------------------------
# Match normalization: stored raw wire text vs the client's clean resend
# ---------------------------------------------------------------------------


def test_normalize_for_match_harmony_reduces_stored_raw():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stored_raw = LEAKED + "<|return|>"
    n_stored = MLXEngine._normalize_for_match(
        stored_raw, "assistant", "harmony", True,
    )
    n_incoming = MLXEngine._normalize_for_match(
        LEAKED_CONTENT, "assistant", "harmony", True,
    )
    assert n_stored == n_incoming == LEAKED_CONTENT


def test_normalize_for_match_harmony_strips_tool_blocks():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    raw = (
        TOOL_STREAM
        + "<|start|>assistant<|channel|>final<|message|>Sunny!<|return|>"
    )
    assert MLXEngine._normalize_for_match(
        raw, "assistant", "harmony", True,
    ) == "Sunny!"


def test_normalize_for_match_other_families_unchanged():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    assert MLXEngine._normalize_for_match(
        "<think>t</think>answer", "assistant", "chatml", True,
    ) == "answer"
    # A literal harmony-like marker in chatml content is NOT stripped by the
    # harmony regex... it IS a commentary block shape; guard the common case:
    assert MLXEngine._normalize_for_match(
        "plain text", "assistant", "chatml", True,
    ) == "plain text"


# ===========================================================================
# CODEX BATCH-5 FINDINGS
# ===========================================================================


# ---------------------------------------------------------------------------
# P1-1: a natural CALL stop reaches the tool FSM / batch parse as text
# ---------------------------------------------------------------------------


def _drive_locked(eng, messages, *, tools, thinking=False, max_tokens=8):
    return list(
        eng._generate_locked(
            messages, max_tokens=max_tokens, temperature=0.0, session_id="s",
            tools=tools, cancel_event=None, thinking=thinking, thinking_budget=0,
            top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=1.0,
            response_format=None,
        )
    )


def test_engine_natural_call_stop_synthesizes_terminator_p1_1(monkeypatch):
    """P1-1 REAL BACKEND CONTRACT: the mlx-lm stream records the terminal CALL
    eos TOKEN but never detokenizes its ``<|call|>`` marker into the text (the
    frame carries token=CALL_ID with EMPTY text — this test injects NO literal
    marker). The engine must re-attach the terminator so the tool block parses
    on the router AND the batch parse -> finish_reason=tool_calls and the
    session stores the structured call.

    PRE-FIX: accumulated_text lacked <|call|>, _parse_harmony_tool_calls found
    no terminated block, parsed_tool_calls was empty -> finish_reason=stop and
    the raw marker block leaked as content. So the finish_reason and joined-
    text assertions below both FAIL pre-fix."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stored = [1, 2, 3, 4, 5]  # interrupted content tail (splice with closer)
    eng, cs, messages = _harmony_hit_engine(stored, suffix=[END_ID, 55, 66])
    tools = [{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}},
    }]
    # Re-stamp the session contract to include the tools so the HIT gate
    # matches (a tools mismatch would demote to MISS, changing the path).
    tc = eng._canonical_tools(tools)
    eng._sessions["s"].tools = tc
    eng._sessions["s"].prompt_fingerprint = MLXEngine._prompt_fingerprint(
        tc, False, template_rev=eng._suffix_template_rev(),
    )

    # The scripted mlx-lm stream: the tool block as TEXT, then the CALL eos
    # frame with EMPTY text — the marker is NEVER in resp.text.
    block = (
        "<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"location": "Tokyo"}'
    )
    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen, [[(101, block), (CALL_ID, "")]],
    )

    chunks = _drive_locked(eng, messages, tools=tools)

    # The synthesized terminator reached the outgoing text stream even though
    # NO frame carried it (the real-backend contract).
    joined = "".join(c.text for c in chunks)
    assert "<|call|>" in joined
    # ... completing the block into a real tool call.
    assert chunks[-1].finish_reason == "tool_calls"
    stored_msg = eng._sessions["s"].messages[-1]
    assert stored_msg["role"] == "assistant"
    assert stored_msg.get("tool_calls")
    assert stored_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    # The stored content channel has the tool XML stripped (no double render).
    for m in ALL_MARKERS:
        assert m not in stored_msg.get("content", "")


def test_engine_return_stop_mid_final_clean_content_finish_stop_p1_1(monkeypatch):
    """P1-1 companion: a RETURN stop mid-final is NOT a tool call — no
    terminator is synthesized, finish_reason=stop, and the content channel is
    clean (the router split of the engine's raw output yields the answer)."""
    stored = [1, 2, 3, 4, 5]
    eng, cs, messages = _harmony_hit_engine(stored, suffix=[END_ID, 55, 66])

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "<|channel|>final<|message|>Hello there"), (RETURN_ID, "")]],
    )

    chunks = _drive_locked(eng, messages, tools=None)

    joined = "".join(c.text for c in chunks)
    assert "<|call|>" not in joined  # nothing synthesized for a RETURN stop
    assert chunks[-1].finish_reason == "stop"
    thinking, content = split_thinking_and_content(joined, model_family="harmony")
    assert content == "Hello there"
    _, calls = parse_tool_calls(content, model_family="harmony")
    assert calls == []


# ---------------------------------------------------------------------------
# P1-2: unclosed analysis is kept RAW (no wrong-HIT vs a plain resend)
# ---------------------------------------------------------------------------


def test_normalize_for_match_harmony_unclosed_analysis_kept_raw_p1_2():
    """P1-2 (batch-4 R13/14 twin): a stored raw turn with an EMPTY visible
    content channel (unclosed/closed analysis, no final) must NOT reduce to ""
    and plain-equality collide with a DIFFERENT plain resend. Kept RAW ->
    honest MISS vs empty/text; a clean closed final still reduces + matches.

    PRE-FIX: norm(unclosed) == "" == norm("") == norm("secret plan"), a
    wrong-HIT — so the inequality assertions below FAIL pre-fix."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    def norm(c):
        return MLXEngine._normalize_for_match(c, "assistant", "harmony", True)

    unclosed = "<|channel|>analysis<|message|>secret internal plan"
    assert norm(unclosed) != norm("")                       # vs empty resend
    assert norm(unclosed) != norm("secret internal plan")   # vs text resend
    assert norm(unclosed) == norm(unclosed)                 # deterministic replay

    closed_analysis = "<|channel|>analysis<|message|>done thinking<|end|>"
    assert norm(closed_analysis) != norm("")                # empty content -> raw

    # A CLEAN closed final still reduces + matches its clean resend.
    final = "<|channel|>final<|message|>the answer<|return|>"
    assert norm(final) == norm("the answer") == "the answer"
    # A marker-less plain resend reduces to itself (not treated as ambiguous).
    assert norm("just a plain answer") == "just a plain answer"


def test_messages_match_harmony_unclosed_analysis_strict_anon_p1_2():
    """Integration: on the STRICT anon path a stored unclosed-analysis turn
    does NOT match an empty (or differing) plain resend, but a byte-identical
    resend DOES."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "harmony"
    unclosed = "<|channel|>analysis<|message|>secret internal plan"
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": unclosed},
    ]
    empty_resend = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ]
    same_resend = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": unclosed},
    ]
    assert eng._messages_match(
        stored, empty_resend, last_assistant_wildcard=False,
    ) is False
    assert eng._messages_match(
        stored, same_resend, last_assistant_wildcard=False,
    ) is True


# ---------------------------------------------------------------------------
# P1-3: a quoted commentary block inside a final body is NOT a tool call
# ---------------------------------------------------------------------------


QUOTED_IN_FINAL = (
    "<|channel|>final<|message|>here is an example: "
    "<|channel|>commentary to=functions.x <|constrain|>json<|message|>{}"
    "<|call|> end"
)


def test_router_quoted_commentary_in_final_not_a_tool_call_p1_3():
    """P1-3 STRUCTURAL PRECEDENCE: a functions-commentary block QUOTED inside
    a final CONTENT body (no preceding <|start|>) is quoted content, never a
    real channel switch — so it is NOT promoted to an executable tool call.
    ZERO calls on the streaming router, the batch union, AND the router split.

    PRE-FIX: the mid-body <|channel|> terminated the final body, was
    reprocessed structurally, and the quoted commentary block parsed to ONE
    tool call from visible content — so the call-count assertions FAIL."""
    # Streaming router content -> zero calls.
    segs = _route([QUOTED_IN_FINAL])
    assert _join(segs, CHANNEL_REASONING) == ""
    content = _join(segs, CHANNEL_CONTENT)
    _, calls = parse_tool_calls(content, model_family="harmony")
    assert calls == []
    assert "example" in content  # the quoted example survived as content

    # Batch positional union -> zero calls (keep batch == stream).
    union = "".join(
        QUOTED_IN_FINAL[s:e]
        for s, e in content_segments(QUOTED_IN_FINAL, "harmony")
    )
    _, batch_calls = parse_tool_calls(union, model_family="harmony")
    assert batch_calls == []

    # Router-authoritative whole-text split -> zero calls.
    _, split_content = split_thinking_and_content(
        QUOTED_IN_FINAL, model_family="harmony",
    )
    _, split_calls = parse_tool_calls(split_content, model_family="harmony")
    assert split_calls == []


def test_router_genuine_tool_call_still_parses_p1_3():
    """The structural-precedence fix must NOT break a REAL tool call — one
    reached AFTER a <|start|> boundary, or the bare gen-start opener."""
    # After <|start|> (analysis -> commentary tool):
    segs = _route([TOOL_STREAM])
    _, calls = parse_tool_calls(
        _join(segs, CHANNEL_CONTENT), model_family="harmony",
    )
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    # At gen start (bare <|channel|>commentary opener):
    segs2 = _route([TOOL_BLOCK])
    _, calls2 = parse_tool_calls(
        _join(segs2, CHANNEL_CONTENT), model_family="harmony",
    )
    assert len(calls2) == 1 and calls2[0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize("chunks", _chunkings(QUOTED_IN_FINAL))
def test_router_quoted_commentary_chunk_permutation_p1_3(chunks):
    """Any segmentation of the quoted-commentary stream still yields zero
    tool calls (markers split across chunks are held, never mis-promoted)."""
    segs = _route(chunks)
    _, calls = parse_tool_calls(
        _join(segs, CHANNEL_CONTENT), model_family="harmony",
    )
    assert calls == []


# ---------------------------------------------------------------------------
# P2-6: an unknown / misspelled channel fails SAFE to reasoning
# ---------------------------------------------------------------------------


def test_router_unknown_channel_routes_to_reasoning_p2_6():
    """P2-6: an UNKNOWN / misspelled channel fails SAFE toward HIDING —
    reasoning_content, NEVER content (the original confidentiality bug was the
    fail-OPEN-to-content default).

    PRE-FIX: an unknown channel body defaulted to CONTENT, so the reasoning
    text leaked into the visible answer — the content assertion FAILS."""
    text = "<|channel|>analysiss<|message|>secret internal reasoning<|end|>"
    segs = _route([text])
    assert _join(segs, CHANNEL_CONTENT) == ""
    assert _join(segs, CHANNEL_REASONING) == "secret internal reasoning"

    # Batch twin agrees.
    thinking, content = split_thinking_and_content(text, model_family="harmony")
    assert content == ""
    assert thinking == "secret internal reasoning"

    # final + preamble-commentary still stay CONTENT (unchanged).
    ok = (
        "<|channel|>final<|message|>visible<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>preamble too<|return|>"
    )
    seg2 = _route([ok])
    assert _join(seg2, CHANNEL_CONTENT) == "visiblepreamble too"
    assert _join(seg2, CHANNEL_REASONING) == ""


# ---------------------------------------------------------------------------
# Ground-truth correction: strip only on the template's real raise sequences
# ---------------------------------------------------------------------------


def test_strip_thinking_tags_harmony_quoted_marker_passes_through():
    """GROUND-TRUTH CORRECTION: content that merely QUOTES or partially types
    a marker — but is NOT one of the template's raise-triggering headers
    (``<|channel|>analysis<|message|>`` / ``<|channel|>final<|message|>``) —
    passes through UNMODIFIED (the template would not raise on it).

    PRE-FIX: the broad 'any marker present' gate reduced these strings,
    destructively rewriting legitimate content — the equality assertions
    FAIL."""
    for content in (
        "Use the <|channel|> token to switch channels.",
        "An incomplete <|call| fragment appears in this sentence.",
        "This mentions <|message|> and <|end|> but forms no real header.",
        "<|channel|>analysis<|message| missing the closing angle bracket",
    ):
        msgs = strip_thinking_tags(
            [{"role": "assistant", "content": content}], model_family="harmony",
        )
        assert msgs[0]["content"] == content, content


def test_strip_thinking_tags_harmony_real_header_still_reduced():
    """A REAL analysis/final header (the template WOULD raise on it) is still
    reduced to the content channel."""
    msgs = strip_thinking_tags(
        [{"role": "assistant", "content": LEAKED + "<|return|>"}],
        model_family="harmony",
    )
    assert msgs[0]["content"] == LEAKED_CONTENT
    msgs2 = strip_thinking_tags(
        [{"role": "assistant",
          "content": "<|channel|>final<|message|>hi there<|return|>"}],
        model_family="harmony",
    )
    assert msgs2[0]["content"] == "hi there"


# ===========================================================================
# CODEX BATCH-6 FINDINGS A & B — the <|call|>-ONLY-terminator rule
#
# THE RULE (enforced in BOTH the streaming FSM ``_feed_harmony`` and the
# positional twin ``_harmony_walk``): a functions-commentary block is an
# EXECUTABLE tool call ONLY when terminated by ``<|call|>`` (the tool-call
# eos; P1-1 re-attaches it on a natural stop). EVERY other termination —
# ``<|start|>``/``<|channel|>`` header opener, ``<|end|>``/``<|return|>``, or
# EOF/flush — makes the block raw CONTENT, never promoted.
#
# FINDING A: a commentary-functions block NOT terminated by ``<|call|>`` was
#   canonicalized to a ``<|call|>`` block by the STREAMING router (1 call)
#   while the BATCH positional twin left it raw (0 calls) — batch != stream.
# FINDING B: a full ``<|start|>...<|channel|>commentary...<|call|>`` frame
#   QUOTED inside a final CONTENT body executed as a PHANTOM tool call, because
#   the mid-body ``<|start|>`` sent the quoted frame back to structural
#   processing. Now a ``<|start|>`` inside an OPEN final/content body is quoted
#   content (a final is terminal — the model does not open a new message after
#   it in one response), so the frame stays content and never executes.
#
# These property tests are the SAFETY NET (the external reviewer could not
# finish). All randomness is seeded by INDEX, never an RNG, so the fuzz is
# deterministic across repeated runs.
# ===========================================================================


def _stream_channels(chunks):
    """(reasoning, content) the streaming router emits for a chunking."""
    segs = _route(chunks)
    return _join(segs, CHANNEL_REASONING), _join(segs, CHANNEL_CONTENT)


def _batch_content_union(text):
    """The positional twin's CONTENT channel (content_segments -> _harmony_walk
    non-reasoning spans), reassembled — computed INDEPENDENTLY of the streaming
    feed (this is the 'batch' side of the batch == stream invariant)."""
    return "".join(text[s:e] for s, e in content_segments(text, "harmony"))


def _batch_reasoning(text):
    """The positional twin's REASONING channel (_harmony_walk analysis spans)."""
    return "".join(
        text[s:e] for kind, s, e in _harmony_walk(text) if kind == "reasoning"
    )


def _calls_sig(content):
    """(name, arguments) signatures of the tool calls parsed from content."""
    _, calls = parse_tool_calls(content, model_family="harmony")
    return [
        (c["function"]["name"], c["function"]["arguments"]) for c in calls
    ]


def _assert_batch_equals_stream(text, chunks):
    """The core invariant: the streamed channels (over ANY chunking + flush)
    are BYTE-EQUAL to the positional twin's channels, so the wire's parsed
    tool calls == the batch union's parsed tool calls == the same set."""
    r_stream, c_stream = _stream_channels(chunks)
    c_batch = _batch_content_union(text)
    r_batch = _batch_reasoning(text)
    assert c_stream == c_batch, (text, chunks, c_stream, c_batch)
    assert r_stream == r_batch, (text, chunks, r_stream, r_batch)
    # Call sets agree (implied by content equality, asserted for clarity).
    assert _calls_sig(c_stream) == _calls_sig(c_batch), (text, chunks)


# --- Deterministic adversarial corpus generator -----------------------------

_CORP_TERMS = ("<|call|>", "<|end|>", "<|return|>", "")  # "" = missing/EOF
_CORP_KINDS = ("A", "F", "P", "C")  # analysis / final / preamble / functions


def _corp_block(kind, i, term):
    """One harmony block. Bodies NEVER end mid-marker, so a string that ends in
    an open reasoning body has no partial-marker tail to disagree on (the FSM
    flush drops a reasoning partial tail; the walk keeps it — an orthogonal,
    pre-existing quirk the corpus deliberately avoids)."""
    if kind == "A":
        return f"<|channel|>analysis<|message|>think {i} deeply now" + term
    if kind == "F":
        return f"<|channel|>final<|message|>answer {i} to the user" + term
    if kind == "P":
        return f"<|channel|>commentary<|message|>preamble note {i}" + term
    # C — functions-commentary (GENERATED form: recipient in the header).
    name = ("get_weather", "web.search", "do-thing")[i % 3]
    body = ("{}", '{"q": %d}' % i, '{"a": "x", "b": %d}' % i)[i % 3]
    return (
        f"<|channel|>commentary to=functions.{name} <|constrain|>json"
        f"<|message|>{body}" + term
    )


def _build_batch_eq_corpus():
    """~650 harmony wire strings mixing analysis / final / preamble / functions
    blocks with EVERY terminator variant (<|call|>, <|end|>, <|return|>,
    missing/EOF), with and without a <|start|>assistant boundary between
    blocks (adjacent, degenerate-missing-terminator, nested). All choices are
    a deterministic function of the loop indices."""
    seqs: list[str] = []
    for k in _CORP_KINDS:  # length-1
        for t in range(len(_CORP_TERMS)):
            seqs.append(_corp_block(k, 0, _CORP_TERMS[t]))
    for length in (2, 3):  # length-2 and length-3 sequences
        for combo in product(_CORP_KINDS, repeat=length):
            for ti in range(len(_CORP_TERMS)):
                for bi in range(2):
                    parts: list[str] = []
                    for j, k in enumerate(combo):
                        if j > 0:
                            # Boundary present iff (bi + j) is even — a fixed
                            # per-position rule, so both the "<|start|> present"
                            # and "missing boundary" degenerate shapes appear.
                            present = (bi + j) % 2 == 0
                            parts.append("<|start|>assistant" if present else "")
                        term = _CORP_TERMS[(ti + j) % len(_CORP_TERMS)]
                        parts.append(_corp_block(k, j, term))
                    seqs.append("".join(parts))
    return seqs


_BATCH_EQ_CORPUS = _build_batch_eq_corpus()


def _lite_chunkings(text):
    """A deterministic, index-seeded set of segmentations (no RNG state)."""
    outs = [[text], list(text)]
    for size in (3, 8):
        outs.append([text[i:i + size] for i in range(0, len(text), size)])
    if len(text) > 6:
        cuts = sorted({(j * 7 + 3) % len(text) for j in range(1, 6)} - {0})
        pts = sorted(set([0] + cuts + [len(text)]))
        outs.append([text[a:b] for a, b in zip(pts, pts[1:])])
    return outs


def test_batch_equals_stream_large_adversarial_corpus():
    """PROPERTY (the safety net): for EVERY corpus string and EVERY chunking,
    the streamed channels equal the positional twin's channels — so the wire's
    parsed tool calls == the batch union's parsed calls, byte-for-byte.

    This FAILS pre-fix on FINDING A: a non-<|call|>-terminated functions block
    (e.g. C(<|end|>), C(<|return|>), or a C block run into the next <|start|>)
    was canonicalized to a <|call|> block on the STREAM (1 call) but left raw
    in the WALK (0 calls) — content bytes and call sets diverged."""
    assert len(_BATCH_EQ_CORPUS) > 600  # a genuinely large corpus
    for text in _BATCH_EQ_CORPUS:
        for chunks in _lite_chunkings(text):
            _assert_batch_equals_stream(text, chunks)


# --- Explicit finding repros (exact probe strings) --------------------------

# FINDING A probe (⟪⟫ = the special-token delimiters): a commentary-functions
# block terminated by <|start|> (NOT <|call|>), then a real final turn.
FINDING_A = (
    "<|channel|>commentary to=functions.x <|message|>{}"
    "<|start|>assistant<|channel|>final<|message|>done<|return|>"
)
# FINDING B probes: a full quoted frame INSIDE a final CONTENT body.
FINDING_B_EOF = (
    "<|channel|>final<|message|>Here is the tool-call format: "
    "<|start|>assistant<|channel|>commentary to=functions.x<|message|>{}<|call|>"
)
FINDING_B_MID = (
    "<|channel|>final<|message|>Example: "
    "<|start|>assistant<|channel|>commentary to=functions.x<|message|>{}"
    "<|call|> — do NOT actually run that.<|return|>"
)


def test_finding_a_zero_calls_batch_equals_stream():
    """FINDING A probe: 0 executable tool calls on BOTH the stream and the
    batch union, all chunkings; the whole block is CONTENT (never promoted)."""
    for chunks in _chunkings(FINDING_A):
        r_stream, c_stream = _stream_channels(chunks)
        assert _calls_sig(c_stream) == []
        _assert_batch_equals_stream(FINDING_A, chunks)
    # The real final turn survives as visible content; the commentary markers
    # pass through raw (not executed).
    _, c = _stream_channels([FINDING_A])
    assert "done" in c
    assert _calls_sig(_batch_content_union(FINDING_A)) == []


def test_finding_b_quoted_frame_in_final_zero_calls():
    """FINDING B probes: a quoted <|start|>...<|channel|>commentary...<|call|>
    frame inside a final body is CONTENT (the model showing an example), NEVER
    an executable call — 0 calls on stream AND batch, all chunkings."""
    for probe in (FINDING_B_EOF, FINDING_B_MID):
        for chunks in _chunkings(probe):
            r_stream, c_stream = _stream_channels(chunks)
            assert _calls_sig(c_stream) == [], (probe, chunks)
            _assert_batch_equals_stream(probe, chunks)
        # The quoted frame text survives as visible content.
        _, c = _stream_channels([probe])
        assert "functions.x" in c
        assert _calls_sig(_batch_content_union(probe)) == []


# --- Genuine tool calls still parse (both stream and batch) ------------------

def test_genuine_call_terminated_by_call_parses_both_sides():
    """A GENUINE <|call|>-terminated tool call parses to exactly ONE call on
    BOTH the stream and the batch union — at gen-start (bare opener) and after
    a real <|start|> boundary (the P1-1-synthesized-terminator shape)."""
    for text, name in ((TOOL_BLOCK, "get_weather"), (TOOL_STREAM, "get_weather")):
        for chunks in _chunkings(text):
            _, c_stream = _stream_channels(chunks)
            assert [n for n, _ in _calls_sig(c_stream)] == [name], (text, chunks)
            _assert_batch_equals_stream(text, chunks)
        assert [n for n, _ in _calls_sig(_batch_content_union(text))] == [name]


def test_multi_tool_two_genuine_call_blocks_two_calls_both_sides():
    """Two genuine <|call|>-terminated commentary blocks -> exactly 2 calls, in
    order, on BOTH the stream and the batch union (all chunkings)."""
    text = (
        "<|channel|>analysis<|message|>need two tools<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.alpha "
        '<|constrain|>json<|message|>{"x": 1}<|call|>'
        "<|start|>assistant<|channel|>commentary to=functions.beta "
        '<|constrain|>json<|message|>{"y": 2}<|call|>'
    )
    for chunks in _chunkings(text):
        _, c_stream = _stream_channels(chunks)
        assert [n for n, _ in _calls_sig(c_stream)] == ["alpha", "beta"], chunks
        _assert_batch_equals_stream(text, chunks)
    sig = _calls_sig(_batch_content_union(text))
    assert [n for n, _ in sig] == ["alpha", "beta"]
    assert json.loads(sig[0][1]) == {"x": 1}
    assert json.loads(sig[1][1]) == {"y": 2}


def test_non_call_terminated_functions_blocks_never_execute_all_terms():
    """Every non-<|call|> termination of a functions-commentary block yields
    ZERO calls on BOTH sides (finding A generalized): <|end|>, <|return|>,
    EOF (missing terminator), and a truncated mid-JSON body."""
    zero_call = [
        # <|end|> / <|return|> / missing-terminator terminations.
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|end|>",
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|return|>",
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}",
        # Truncated mid-JSON (stream cut before <|call|>).
        '<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"a": 1',
        # Analysis, then a tool block closed by <|end|> (not <|call|>).
        "<|channel|>analysis<|message|>maybe<|end|><|start|>assistant"
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|end|>",
        # A functions block run straight into the next final turn (no <|call|>).
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}"
        "<|start|>assistant<|channel|>final<|message|>hi<|return|>",
    ]
    for text in zero_call:
        for chunks in _chunkings(text):
            _, c_stream = _stream_channels(chunks)
            assert _calls_sig(c_stream) == [], (text, chunks)
            _assert_batch_equals_stream(text, chunks)


def test_genuine_call_after_analysis_missing_end_still_executes():
    """The reasoning-body <|start|> recovery is PRESERVED: a degenerate
    analysis missing its <|end|> that runs straight into a real
    <|start|>assistant commentary tool call still yields the call (the
    <|start|> is a genuine boundary after analysis, unlike inside a final)."""
    text = (
        "<|channel|>analysis<|message|>thinking without a close"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>'
    )
    for chunks in _chunkings(text):
        r_stream, c_stream = _stream_channels(chunks)
        assert [n for n, _ in _calls_sig(c_stream)] == ["get_weather"], chunks
        assert r_stream == "thinking without a close"
        _assert_batch_equals_stream(text, chunks)
