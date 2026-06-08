"""Tests for the streaming-safe Gemma 4 thinking-channel stripper (FIX 3).

The OpenAI-compatible streaming path used to yield the raw Gemma 4
``<|channel>thought...<channel|>`` markers to the client (OpenCode), which
then replayed into conversation history. ``_Gemma4ThinkingStripper`` buffers
output until the ``<channel|>`` close is seen and emits only the post-close
answer, holding back a partial close marker split across chunks.
"""

from __future__ import annotations

from mlx_soloheaven.api.openai_compat import _Gemma4ThinkingStripper


def _feed_all(stripper, chunks):
    return "".join(stripper.feed(c) for c in chunks) + stripper.flush()


def test_strip_single_chunk():
    s = _Gemma4ThinkingStripper(active=True)
    assert s.feed("<|channel>thought\nreasoning<channel|>Hello") == "Hello"
    assert s.feed(" world") == " world"
    assert s.flush() == ""


def test_strip_thought_across_chunks():
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s,
        ["<|channel>thought\nthink", "ing more", "<channel|>ans", "wer"],
    )
    assert out == "answer"


def test_close_marker_split_across_chunk_boundary():
    """The <channel|> close marker may straddle two streamed chunks; the
    partial-marker tail must be held back, never emitted as content."""
    s = _Gemma4ThinkingStripper(active=True)
    assert s.feed("thought\nreason<chan") == ""  # partial marker buffered
    assert s.feed("nel|>visible") == "visible"


def test_close_marker_char_by_char():
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s, ["thought\nx", "<", "channel", "|", ">", "Z", "Z"]
    )
    assert out == "ZZ"


def test_no_thought_text_leaks_before_close():
    """Everything before <channel|> is thought and must be suppressed."""
    s = _Gemma4ThinkingStripper(active=True)
    pre = s.feed("<|channel>thought\nsecret reasoning that should not leak")
    assert pre == ""
    assert "secret" not in pre


def test_degenerate_never_closes_emits_nothing():
    """Output that never closes the channel yields no content (the buffered
    text is all thought and is dropped on flush)."""
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(s, ["thought\nendless reasoning ", "with no close marker"])
    assert out == ""


def test_post_close_passthrough_in_later_chunks():
    s = _Gemma4ThinkingStripper(active=True)
    assert s.feed("<|channel>thought\nr<channel|>a") == "a"
    # Once past the close, later chunks pass straight through.
    assert s.feed("bc") == "bc"
    assert s.feed("def") == "def"


def test_inactive_is_passthrough():
    """Non-gemma4 / thinking-disabled callers must be byte-identical."""
    s = _Gemma4ThinkingStripper(active=False)
    assert s.feed("anything <channel|> stays verbatim") == (
        "anything <channel|> stays verbatim"
    )
    assert s.flush() == ""


# ---------------------------------------------------------------------------
# CORRECTION 1: degenerate MULTI-CYCLE output. A new <|channel>thought opener
# can appear AFTER visible content; the stripper must re-enter thought mode and
# emit ONLY the post-<channel|> content of EACH cycle (mirrors the batch
# parser's "remove ALL spans" behavior, incrementally).
# ---------------------------------------------------------------------------


def test_multi_cycle_two_channels_single_chunk():
    """Two thought cycles in one chunk: only the post-close content of each
    cycle survives; the 2nd channel marker + thought text must NOT leak."""
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s,
        [
            "<|channel>thought\nA<channel|>partial"
            "<|channel>thought\nB<channel|>final"
        ],
    )
    assert out == "partialfinal"
    assert "thought" not in out
    assert "channel" not in out


def test_multi_cycle_three_channels():
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s,
        [
            "<|channel>thought\nr1<channel|>one",
            "<|channel>thought\nr2<channel|>two",
            "<|channel>thought\nr3<channel|>three",
        ],
    )
    assert out == "onetwothree"


def test_multi_cycle_bare_reopener_after_content():
    """FIX 4 (policy change): a BARE ``thought\\n`` opener is recognized ONLY at
    the very START of generation (sliding-window first token). AFTER visible
    content it is NOT a re-opener — a literal ``thought\\n`` line in content /
    tool args must stay CONTENT, not be mis-routed to reasoning. So here only
    the FIRST (full-marker) cycle is stripped; the bare ``thought\\n`` after
    ``visible `` is content. (A real multi-cycle re-opener uses the FULL
    ``<|channel>thought`` marker — see test_multi_cycle_two_channels_single_chunk.)
    The orphan ``<channel|>`` left in content is a degenerate-input artifact, not
    reasoning leakage."""
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s,
        ["<|channel>thought\nA<channel|>visible thought\nB<channel|>tail"],
    )
    # OLD policy (pre-FIX-4) treated the bare reopener as reasoning -> "visible tail".
    assert out == "visible thought\nB<channel|>tail"
    assert out.startswith("visible ")  # cycle-1 reasoning "A" stripped, content kept


def test_multi_cycle_markers_split_char_by_char():
    """Feed a 2-cycle degenerate stream one character at a time: no marker
    fragment (open or close) may ever leak into the emitted content."""
    raw = "<|channel>thought\nA<channel|>partial<|channel>thought\nB<channel|>final"
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(s, list(raw))
    assert out == "partialfinal"
    for marker in ("<|channel>", "<channel|>", "thought\n"):
        assert marker not in out


def test_second_channel_open_split_across_chunks():
    """The SECOND cycle's <|channel> open marker straddles a chunk boundary —
    the partial open tail must be held, not emitted as content."""
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s,
        ["<|channel>thought\nA<channel|>vis<|chan", "nel>thought\nB<channel|>end"],
    )
    assert out == "visend"
    assert "chan" not in out


# ---------------------------------------------------------------------------
# CORRECTION 2: bounded guard. The stream does NOT begin assumed-inside-thought.
# A gemma4 thinking-enabled request whose model emits a plain answer with NO
# channel markers must pass through UNSTRIPPED (not be dropped).
# ---------------------------------------------------------------------------


def test_plain_answer_no_channel_markers_passes_through():
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(s, ["Hello, ", "this is a ", "plain answer."])
    assert out == "Hello, this is a plain answer."


def test_plain_answer_single_chunk_passes_through():
    # ``content`` ends in "t" — a 1-char prefix of the bare "thought\n" opener,
    # so it is correctly held back as a possible split-marker tail and emitted
    # on flush (a marker that never completed is real content).
    s = _Gemma4ThinkingStripper(active=True)
    out = s.feed("just content") + s.flush()
    assert out == "just content"


def test_content_then_thought_then_content():
    """Content before the first opener is preserved; thought is stripped;
    post-close content resumes."""
    s = _Gemma4ThinkingStripper(active=True)
    out = _feed_all(
        s, ["lead-in <|channel>thought\nsecret<channel|>after"]
    )
    assert out == "lead-in after"
    assert "secret" not in out
