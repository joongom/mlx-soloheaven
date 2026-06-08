"""Unit tests for generate_stream_batches_async coalescing.

Exercises MLXEngine.generate_stream_batches_async with a *fake* engine whose
``generate_stream`` replays a canned GenerationResult sequence — no model load.
Asserts the batching flush rules:

  (a) order is preserved across all batches,
  (b) the first content token flushes immediately (preserve TTFT),
  (c) status="generating" and finish_reason results flush as their own batch,
  (d) N <= 1 yields singleton batches (coalescing disabled),
  (e) N = 4 groups consecutive normal content tokens,

and that the concatenation of all content across batches equals the input
token texts (content byte-identity — the CRITICAL correctness invariant).
"""
import json
from dataclasses import dataclass

import pytest

from mlx_soloheaven.engine.mlx_engine import MLXEngine, GenerationResult


@dataclass
class _Cfg:
    stream_coalesce_n: int = 4
    stream_coalesce_ms: int = 30


class FakeEngine:
    """Minimal engine exposing the surface generate_stream_batches_async uses.

    We bind the real (unbound) MLXEngine.generate_stream_batches_async to this
    instance so the production code path is exercised verbatim.
    """

    def __init__(self, results: list[GenerationResult], coalesce_n=4, coalesce_ms=30):
        self.cfg = _Cfg(stream_coalesce_n=coalesce_n, stream_coalesce_ms=coalesce_ms)
        self._results = results
        self._use_vlm = False  # use the daemon-thread submit path

    def generate_stream(self, *args, **kwargs):
        for r in self._results:
            yield r

    # Bind the production batched generator onto the fake.
    generate_stream_batches_async = MLXEngine.generate_stream_batches_async


# A representative sequence:
#   status="generating" → first content token → think_end content token →
#   more content tokens → finish_reason terminal.
THINK_END_TOKEN = 999


def _make_results():
    return [
        GenerationResult(status="generating"),
        GenerationResult(text="Hello", token=1),
        GenerationResult(text=" world", token=2),
        GenerationResult(text="</think>", token=THINK_END_TOKEN),
        GenerationResult(text="A", token=3),
        GenerationResult(text="B", token=4),
        GenerationResult(text="C", token=5),
        GenerationResult(text="D", token=6),
        GenerationResult(text="E", token=7),
        GenerationResult(
            text="", finish_reason="stop", prompt_tokens=10, completion_tokens=8
        ),
    ]


async def _collect(engine: FakeEngine) -> list[list[GenerationResult]]:
    batches = []
    async for batch in engine.generate_stream_batches_async([]):
        # Skip empty keepalive batches (single empty result) — they only appear
        # on the 1s queue timeout, which our fast fake stream never triggers.
        batches.append(batch)
    return batches


def _concat_content(batches: list[list[GenerationResult]]) -> str:
    return "".join(
        r.text
        for batch in batches
        for r in batch
        if r.status is None and r.finish_reason is None
    )


@pytest.mark.asyncio
async def test_order_and_content_byte_identity():
    results = _make_results()
    engine = FakeEngine(results, coalesce_n=4)
    batches = await _collect(engine)

    # (a) order preserved: flatten and compare token order to input order.
    flat = [r for batch in batches for r in batch]
    assert [r.token for r in flat] == [r.token for r in results]

    # content byte-identity vs the raw token texts
    expected = "".join(
        r.text for r in results if r.status is None and r.finish_reason is None
    )
    assert _concat_content(batches) == expected


@pytest.mark.asyncio
async def test_first_content_and_control_flush_immediately():
    results = _make_results()
    engine = FakeEngine(results, coalesce_n=4)
    batches = await _collect(engine)

    # (c) status="generating" is its own singleton batch (first batch).
    assert len(batches[0]) == 1
    assert batches[0][0].status == "generating"

    # (b) first content token ("Hello") flushes immediately as its own batch.
    assert len(batches[1]) == 1
    assert batches[1][0].text == "Hello"

    # (c) finish_reason result is its own singleton batch (last batch).
    assert len(batches[-1]) == 1
    assert batches[-1][0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_n4_groups_normal_tokens():
    results = _make_results()
    engine = FakeEngine(results, coalesce_n=4, coalesce_ms=100_000)
    batches = await _collect(engine)

    # Content batches after the first content token: " world","</think>" then
    # the run "A","B","C","D","E". With N=4 (and a huge ms so time never forces
    # a flush) at least one batch must contain >1 normal content result.
    content_batches = [
        b
        for b in batches
        if all(r.status is None and r.finish_reason is None for r in b)
        and b[0].text  # exclude the lone first-content singleton already checked
    ]
    multi = [b for b in content_batches if len(b) > 1]
    assert multi, f"expected at least one coalesced batch, got {[len(b) for b in batches]}"
    # No coalesced batch exceeds N.
    assert all(len(b) <= 4 for b in batches)


@pytest.mark.asyncio
async def test_n1_disables_coalescing():
    results = _make_results()
    engine = FakeEngine(results, coalesce_n=1)
    batches = await _collect(engine)

    # (d) every batch is a singleton; count == number of source results.
    assert all(len(b) == 1 for b in batches)
    assert len(batches) == len(results)

    # content byte-identity still holds with coalescing disabled.
    expected = "".join(
        r.text for r in results if r.status is None and r.finish_reason is None
    )
    assert _concat_content(batches) == expected


@pytest.mark.asyncio
async def test_think_end_token_preserved_in_stream():
    """think_end token rides through unchanged so chat.py can split on it."""
    results = _make_results()
    engine = FakeEngine(results, coalesce_n=4)
    batches = await _collect(engine)

    flat = [r for batch in batches for r in batch]
    think_ends = [r for r in flat if r.token == THINK_END_TOKEN]
    assert len(think_ends) == 1
    assert think_ends[0].text == "</think>"


# --- chat.py layer: content byte-identity + thinking_done split ---


class _ChatFakeEngine(FakeEngine):
    """FakeEngine extended with the minimal surface _stream_chat touches."""

    def __init__(self, results, coalesce_n=4):
        super().__init__(results, coalesce_n=coalesce_n)
        import threading

        class _ChatCfg(_Cfg):
            think_end_token = THINK_END_TOKEN
            enable_thinking = True
            default_max_tokens = 32768

        self.cfg = _ChatCfg(stream_coalesce_n=coalesce_n)
        self.model_family = "chatml"
        self._sessions = {}
        self._lock = threading.Lock()
        self.updated = None

    def _has_disk_cache(self, sid):
        return False

    def update_session_messages(self, sid, msgs):
        self.updated = msgs


@pytest.mark.asyncio
async def test_chat_layer_reasoning_content_split_and_thinking_done(monkeypatch):
    """Reasoning-channel protocol: the chat SSE stream now routes thinking into
    "reasoning" frames and the answer into "content" frames (no raw <think>/
    </think>/<channel|> markers anywhere). thinking_done rides the FIRST content
    frame that follows reasoning."""
    from mlx_soloheaven.api import chat as chat_mod

    # Stub the async db surface _stream_chat calls after the stream loop.
    class _StubDB:
        async def add_message(self, *a, **k):
            pass

        async def get_session_total_tokens(self, *a, **k):
            return 0

        async def update_session_tokens(self, *a, **k):
            pass

        async def update_session(self, *a, **k):
            pass

        async def get_session(self, *a, **k):
            return None

    monkeypatch.setattr(chat_mod, "db", _StubDB())

    results = _make_results()
    eng = _ChatFakeEngine(results, coalesce_n=4)

    frames = []
    async for line in chat_mod._stream_chat("s1", [{"role": "user", "content": "hi"}], eng):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        frames.append(json.loads(line[len("data: "):]))

    text_frames = [f for f in frames if f.get("type") == "text"]
    reasoning = "".join(f.get("reasoning", "") for f in text_frames)
    content = "".join(f.get("content", "") for f in text_frames)

    # chatml stream begins inside the <think> block (opener is in the prompt
    # suffix). Reasoning is everything before </think>; content is the rest.
    # No markers leak into either channel.
    assert reasoning == "Hello world"
    assert content == "ABCDE"
    for marker in ("<think>", "</think>", "<channel|>", "<|channel>"):
        assert marker not in reasoning
        assert marker not in content

    # Exactly one content frame carries thinking_done=True (the first content
    # frame after reasoning); no reasoning frame carries it.
    done_frames = [f for f in text_frames if f.get("thinking_done")]
    assert len(done_frames) == 1
    assert done_frames[0].get("content")
    assert not done_frames[0].get("reasoning")
