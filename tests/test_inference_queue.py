"""Batch C — the parent-side FIFO inference-queue admission gate.

The gate (``mlx_soloheaven.inference_queue.InferenceGate``) fronts ALL GPU
generation entry points with a bounded, FIFO-ordered, concurrency-1 admission
layer. These tests prove every invariant the batch must preserve, HERMETICALLY
(pure asyncio, no MLX, no pipes) for the gate logic, and via FastAPI stubs for
the endpoint wiring:

- FIFO order: N waiters acquire the single slot in submission order.
- Concurrency 1: an overlap detector never sees two generations at once; the
  engine lock stays the underlying serializer, the gate the FIFO+bound on top.
- Max length: fill to max -> the next acquire is 429 queue_full IMMEDIATELY
  (no long wait); after capacity frees, a new request is admitted.
- Queued disconnect (U13): a waiter cancelled while queued is removed, the
  counter decremented, and it NEVER generates; the queue continues for others.
- Shutdown while queued: shutdown fails every queued waiter with
  engine_not_ready (503) — no hang — rejects new acquires, and lets the
  running one finish.
- Counter accounting: after a mix of complete / error / cancel, running +
  waiting returns to 0 (no leak).
- /ready reports the REAL gate depth and stays 200 when saturated-but-ready.
- Endpoint wiring: streaming completion queue-full -> 429 + Retry-After;
  shutdown -> 503; two concurrent streaming requests never overlap in the
  engine and run FIFO; a queued streaming request whose task is cancelled
  never reaches the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from mlx_soloheaven import inference_queue
from mlx_soloheaven.inference_queue import (
    ClientDisconnected,
    InferenceGate,
    Lease,
    acquire_or_disconnect,
    inference_slot,
)
from mlx_soloheaven.api.gate_stream import SlotStreamingResponse
from mlx_soloheaven.engine.types import EngineBusyError, GenerationResult


# ---------------------------------------------------------------------------
# Helpers for the codex concurrency-bug regressions.
# ---------------------------------------------------------------------------


class _FakeAsgiRequest:
    """Minimal stand-in for a Starlette Request exposing the ASGI ``receive``
    channel the handlers watch for ``http.disconnect`` (finding 2). Its
    ``receive`` blocks until ``disconnect()`` is called, then reports the
    disconnect — modelling a client that drops WHILE QUEUED on the gate."""

    def __init__(self):
        self._disc = asyncio.Event()

        async def _receive():
            await self._disc.wait()
            return {"type": "http.disconnect"}

        self.receive = _receive

    def disconnect(self):
        self._disc.set()


# ---------------------------------------------------------------------------
# Hermetic gate logic — pure asyncio, no MLX.
# ---------------------------------------------------------------------------


def test_fifo_order():
    """N requests acquire the single slot strictly in submission order."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        order: list[int] = []

        async def worker(i: int):
            async with inference_slot(gate):
                order.append(i)
                await asyncio.sleep(0.005)  # hold the slot briefly

        # Creation order == scheduling order == enqueue order (call_soon FIFO):
        # the first task takes the slot, the rest queue in order.
        tasks = [asyncio.ensure_future(worker(i)) for i in range(6)]
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(drive()) == list(range(6))


def test_concurrency_exactly_one_no_overlap():
    """An overlap detector proves at most ONE 'generation' holds the slot at a
    time — the second request waits for the first (FIFO)."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        active = 0
        max_active = 0
        order: list[int] = []

        async def gen(i: int):
            nonlocal active, max_active
            async with inference_slot(gate):
                order.append(i)
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)  # window in which an overlap would show
                active -= 1

        tasks = [asyncio.ensure_future(gen(i)) for i in range(5)]
        await asyncio.gather(*tasks)
        return max_active, order

    max_active, order = asyncio.run(drive())
    assert max_active == 1  # never two at once
    assert order == list(range(5))  # and FIFO


def test_max_length_rejects_immediately_then_admits_after_free():
    """Fill to max (1 running + max-1 queued): the next acquire fails FAST with
    429 queue_full (no long wait). After the running one frees, a new acquire
    is admitted (queues, does NOT reject)."""

    async def drive():
        gate = InferenceGate(max_queue_length=3)
        holds = [asyncio.Event() for _ in range(3)]
        started: list[int] = []

        async def holder(i: int):
            async with inference_slot(gate):
                started.append(i)
                await holds[i].wait()

        tasks = [asyncio.ensure_future(holder(i)) for i in range(3)]
        await asyncio.sleep(0.02)  # 1 running + 2 queued
        assert gate.queue_length() == 3

        # Over the bound -> immediate 429 (no await; well under any timeout).
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError) as ei:
            await gate.acquire()
        dt = time.monotonic() - t0
        assert ei.value.reason == EngineBusyError.REASON_QUEUE_FULL
        assert dt < 0.5  # fail-FAST, not a queue wait

        # Free the running one -> a waiter is promoted, depth drops to 2, so a
        # NEW acquire is now ADMITTED (enqueues as the 3rd; no 429).
        holds[0].set()
        await asyncio.sleep(0.02)
        assert gate.queue_length() == 2

        admitted = asyncio.ensure_future(_acquire_and_release(gate))
        await asyncio.sleep(0.02)
        assert not admitted.done()  # queued, waiting — NOT rejected

        for ev in holds:
            ev.set()
        await asyncio.gather(*tasks, admitted)
        assert gate.queue_length() == 0

    asyncio.run(drive())


async def _acquire_and_release(gate: InferenceGate):
    async with inference_slot(gate):
        await asyncio.sleep(0)


def test_queued_disconnect_dropped_never_generates():
    """U13 (queued): a waiter cancelled while queued is removed from the FIFO,
    the counter decremented, and it NEVER runs — the queue continues correctly
    for the remaining waiters."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        ran: list[int] = []
        hold = asyncio.Event()

        async def holder():
            async with inference_slot(gate):
                await hold.wait()

        async def queued(i: int):
            async with inference_slot(gate):
                ran.append(i)

        h = asyncio.ensure_future(holder())
        await asyncio.sleep(0.01)  # holder running
        q1 = asyncio.ensure_future(queued(1))
        q2 = asyncio.ensure_future(queued(2))
        await asyncio.sleep(0.01)  # both queued
        assert gate.queue_length() == 3

        q1.cancel()  # disconnect while queued
        with pytest.raises(asyncio.CancelledError):
            await q1
        assert gate.queue_length() == 2  # counter decremented, q1 gone

        hold.set()  # release the running one -> q2 (only remaining waiter) runs
        await asyncio.gather(h, q2)
        return ran, gate.queue_length()

    ran, depth = asyncio.run(drive())
    assert ran == [2]  # q1 never generated; q2 proceeded
    assert depth == 0  # no leak


def test_shutdown_while_queued_503_no_hang():
    """Shutdown fails every QUEUED waiter with engine_not_ready (503) — no
    hang — and rejects new acquires; the RUNNING one still finishes."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        hold = asyncio.Event()
        results: dict[int, str] = {}

        async def holder():
            async with inference_slot(gate):
                await hold.wait()

        async def queued(i: int):
            try:
                async with inference_slot(gate):
                    results[i] = "ran"
            except EngineBusyError as e:
                results[i] = e.reason

        h = asyncio.ensure_future(holder())
        await asyncio.sleep(0.01)
        q1 = asyncio.ensure_future(queued(1))
        q2 = asyncio.ensure_future(queued(2))
        await asyncio.sleep(0.01)
        assert gate.queue_length() == 3

        gate.shutdown()
        # No hang — the queued waiters resolve promptly with 503.
        await asyncio.wait_for(asyncio.gather(q1, q2), timeout=2.0)
        assert results[1] == EngineBusyError.REASON_ENGINE_NOT_READY
        assert results[2] == EngineBusyError.REASON_ENGINE_NOT_READY

        # A NEW acquire after shutdown is rejected immediately (503).
        with pytest.raises(EngineBusyError) as ei:
            await gate.acquire()
        assert ei.value.reason == EngineBusyError.REASON_ENGINE_NOT_READY

        # The running one finishes cleanly and the slot frees.
        hold.set()
        await asyncio.wait_for(h, timeout=2.0)
        return gate.queue_length()

    assert asyncio.run(drive()) == 0


def test_counter_accounting_no_leak_across_mixed_terminations():
    """After a mix of complete / error / cancel-while-running /
    cancel-while-queued, running + waiting returns to 0."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)

        async def ok():
            async with inference_slot(gate):
                await asyncio.sleep(0.005)

        async def boom():
            async with inference_slot(gate):
                raise ValueError("boom")

        await ok()  # completes cleanly
        with pytest.raises(ValueError):
            await boom()  # errors inside the slot

        # Cancel while RUNNING.
        hold = asyncio.Event()

        async def running_cancel():
            async with inference_slot(gate):
                await hold.wait()

        t = asyncio.ensure_future(running_cancel())
        await asyncio.sleep(0.01)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

        # Cancel while QUEUED (behind a running holder).
        hold2 = asyncio.Event()

        async def holder():
            async with inference_slot(gate):
                await hold2.wait()

        async def queued():
            async with inference_slot(gate):
                pass

        h = asyncio.ensure_future(holder())
        await asyncio.sleep(0.01)
        q = asyncio.ensure_future(queued())
        await asyncio.sleep(0.01)
        q.cancel()
        with pytest.raises(asyncio.CancelledError):
            await q
        hold2.set()
        await h
        return gate.running(), gate.queue_length()

    running, depth = asyncio.run(drive())
    assert running == 0
    assert depth == 0


def test_current_queue_length_and_configure_module_api():
    """The module-level singleton accessors: configure() sets the bound,
    current_queue_length() reports running + waiting live."""
    inference_queue._reset_for_tests()
    inference_queue.configure(7)
    gate = inference_queue.get_inference_gate()
    assert gate.max_queue_length == 7
    assert inference_queue.current_queue_length() == 0
    gate._running = 1
    gate._waiting = 3
    assert inference_queue.current_queue_length() == 4


def test_release_is_safe_when_slot_free():
    """release() on an already-free slot only nudges the queue — never goes
    negative, never raises."""
    gate = InferenceGate(max_queue_length=4)
    gate.release()  # nothing held
    assert gate.running() == 0
    assert gate.queue_length() == 0


# ---------------------------------------------------------------------------
# Endpoint wiring — the gate fronts the OpenAI-compat generation entry points.
# ---------------------------------------------------------------------------


class _StubApiEngine:
    """Minimal engine for driving openai_compat.chat_completions past its
    pre-gate validation (mirrors the batch-B / batch-5 stub shape)."""

    model_family = "chatml"
    model_id = "m"
    cfg = SimpleNamespace(enable_thinking=False)


def _openai_client(engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.server import register_engine_error_handlers

    openai_compat.set_engines({"m": engine}, engine)
    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(openai_compat.router)
    return TestClient(app, raise_server_exceptions=False)


def _stream_req_body():
    return {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_streaming_completion_queue_full_returns_429(monkeypatch):
    """A saturated inference queue answers a REAL 429 queue_full + Retry-After
    BEFORE the SSE response starts (the gate rejects in the handler)."""
    from mlx_soloheaven.api import openai_compat

    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    gate = inference_queue.get_inference_gate()
    gate.set_max_queue_length(2)
    gate._running = 1
    gate._waiting = 1  # depth 2 == max
    try:
        client = _openai_client(_StubApiEngine())
        r = client.post("/v1/chat/completions", json=_stream_req_body())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "queue_full"
    assert body["error"]["code"] == "queue_full"
    assert r.headers.get("Retry-After") is not None


def test_streaming_completion_shutdown_returns_503():
    """A shutting-down server answers 503 engine_not_ready at the gate."""
    from mlx_soloheaven.api import openai_compat

    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    inference_queue.get_inference_gate().shutdown()
    try:
        client = _openai_client(_StubApiEngine())
        r = client.post("/v1/chat/completions", json=_stream_req_body())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_not_ready"
    assert r.headers.get("Retry-After") is not None


class _OverlapEngine:
    """Engine stub whose streaming generation records concurrent entries — an
    overlap (>1 at once) would mean the gate failed to serialize."""

    model_family = "chatml"
    model_id = "m"

    def __init__(self):
        self.cfg = SimpleNamespace(enable_thinking=False)
        self.active = 0
        self.max_active = 0
        self.enter_order: list[str] = []

    async def generate_stream_batches_async(self, messages, **kwargs):
        # Identify the caller by its first user message content.
        tag = messages[-1].get("content") if messages else "?"
        self.enter_order.append(tag)
        # The GPU-busy region is the sleep window: enter, hold, then leave
        # BEFORE yielding. (Decrementing in a finally would be deferred to the
        # async-generator's GC/aclose — long after _stream_completion_body
        # breaks out of the loop — and would falsely read as an overlap even
        # though the gate serialized correctly.)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)  # window in which a real overlap would show
        self.active -= 1
        yield [GenerationResult(text="ok", status=None, finish_reason=None)]
        yield [GenerationResult(
            finish_reason="stop", prompt_tokens=1, completion_tokens=1
        )]


def test_two_concurrent_streams_never_overlap_and_run_fifo():
    """END-TO-END concurrency-1 proof: two concurrent streaming completions
    driven through the real handler + gate never run the engine simultaneously,
    and the second runs only after the first finished (FIFO)."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    eng = _OverlapEngine()
    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    openai_compat.set_engines({"m": eng}, eng)

    async def run_one(text: str):
        req = ChatCompletionRequest(
            model="m", stream=True, user=None,
            messages=[ChatMessage(role="user", content=text)],
        )
        resp = await openai_compat.chat_completions(req)
        # Consume the SSE stream fully -> runs the generation, then the gate
        # slot is released by stream_with_slot's finally.
        async for _chunk in resp.body_iterator:
            pass

    async def drive():
        t1 = asyncio.ensure_future(run_one("first"))
        t2 = asyncio.ensure_future(run_one("second"))
        await asyncio.gather(t1, t2)

    try:
        asyncio.run(drive())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert eng.max_active == 1  # engine never ran two generations at once
    assert eng.enter_order == ["first", "second"]  # FIFO
    assert inference_queue.get_inference_gate().queue_length() == 0  # no leak


def test_queued_streaming_request_cancelled_never_reaches_engine():
    """A streaming request queued behind a running one, whose client
    disconnects (task cancelled) while still waiting on the gate, never enters
    the engine — and the running one completes normally."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    class _BlockingEngine:
        model_family = "chatml"
        model_id = "m"

        def __init__(self):
            self.cfg = SimpleNamespace(enable_thinking=False)
            self.seen: list[str] = []
            self.release = None  # set inside the loop

        async def generate_stream_batches_async(self, messages, **kwargs):
            tag = messages[-1].get("content") if messages else "?"
            self.seen.append(tag)
            if tag == "first":
                await self.release.wait()  # hold the slot open
            yield [GenerationResult(
                finish_reason="stop", prompt_tokens=1, completion_tokens=1
            )]

    eng = _BlockingEngine()
    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    openai_compat.set_engines({"m": eng}, eng)

    async def run_one(text: str):
        req = ChatCompletionRequest(
            model="m", stream=True, user=None,
            messages=[ChatMessage(role="user", content=text)],
        )
        resp = await openai_compat.chat_completions(req)
        async for _chunk in resp.body_iterator:
            pass

    async def drive():
        eng.release = asyncio.Event()
        t1 = asyncio.ensure_future(run_one("first"))
        await asyncio.sleep(0.03)  # first is running (holding the slot open)
        t2 = asyncio.ensure_future(run_one("second"))
        await asyncio.sleep(0.03)  # second is QUEUED on the gate (handler acquire)
        gate = inference_queue.get_inference_gate()
        assert gate.queue_length() == 2

        t2.cancel()  # client disconnects while queued
        with pytest.raises(asyncio.CancelledError):
            await t2
        assert gate.queue_length() == 1  # only the running one remains

        eng.release.set()  # let the first finish
        await asyncio.wait_for(t1, timeout=2.0)
        return gate.queue_length()

    try:
        depth = asyncio.run(drive())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert eng.seen == ["first"]  # the cancelled queued request NEVER generated
    assert depth == 0  # no leak


# ===========================================================================
# Codex concurrency-bug regressions (findings 1-6). Each reproduces the EXACT
# interleaving codex gave; each FAILS against the pre-fix gate.
# ===========================================================================


# --- [1] head-only wake: cancelling the FIFO head must not double-admit ------


def test_finding1_cancel_head_then_release_same_turn_no_double_admission():
    """CODEX FINDING 1 — the exact interleaving: A holds the slot; queue
    [B,C,D]; B is cancelled (its future is done but B has NOT resumed to
    unlink); A releases in the SAME loop turn. The old ``_wake_up_first``
    scanned PAST the done head B and woke C; then B resumed, saw the slot free
    and woke D — handing the ONE slot to BOTH C and D (reproduced max_active==2).

    Head-only wake (mirrors CPython asyncio.Lock): release sees the head B is
    done and does nothing; B, on resume, unlinks itself and wakes the NEW head
    C; D stays queued. Exactly ONE of C/D runs, FIFO (C before D), and the
    counters return to 0."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        active = 0
        max_active = 0
        ran: list[str] = []

        lease_a = await gate.acquire()  # A holds the slot (fast path)

        async def worker(tag: str):
            nonlocal active, max_active
            lease = await gate.acquire()
            ran.append(tag)
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            gate.release(lease)

        b = asyncio.ensure_future(worker("B"))
        c = asyncio.ensure_future(worker("C"))
        d = asyncio.ensure_future(worker("D"))
        for _ in range(5):
            await asyncio.sleep(0)  # B, C, D all enqueue behind A
        assert gate.queue_length() == 4

        # SAME loop turn: cancel the head B, then release A — NO await between,
        # so B's future is done (cancelled) but B has not yet resumed to unlink.
        b.cancel()
        gate.release(lease_a)

        await asyncio.gather(c, d, return_exceptions=True)
        with pytest.raises(asyncio.CancelledError):
            await b
        return max_active, ran, gate.stats()

    max_active, ran, stats = asyncio.run(drive())
    assert max_active == 1  # NEVER two holders at once (pre-fix: 2)
    assert ran == ["C", "D"]  # FIFO — C promoted first, D after C finished
    assert stats["running"] == 0 and stats["waiting"] == 0  # no leak


# --- [2] a queued ASGI disconnect unlinks the ticket -------------------------


def test_finding2_queued_asgi_disconnect_unlinks_ticket_gate_level():
    """CODEX FINDING 2 (gate level): a request QUEUED on the gate whose ASGI
    connection reports http.disconnect while it waits is unlinked from the FIFO
    (never promoted), and the queue continues for the others. A real queued
    disconnect does NOT cancel the handler task — acquire_or_disconnect races
    the acquire against a receive-channel watcher."""

    async def drive():
        gate = InferenceGate(max_queue_length=10)
        lease_a = await gate.acquire()  # A holds the slot

        fake = _FakeAsgiRequest()
        # A second caller QUEUES behind A, watching its (fake) receive channel.
        queued = asyncio.ensure_future(
            acquire_or_disconnect(gate, fake.receive)
        )
        # A third, live caller queues behind the second.
        live = asyncio.ensure_future(gate.acquire())
        for _ in range(5):
            await asyncio.sleep(0)
        assert gate.queue_length() == 3  # A running + 2 queued

        # The queued client disconnects WHILE WAITING.
        fake.disconnect()
        with pytest.raises(ClientDisconnected):
            await queued
        # Its ticket was unlinked; the live waiter is still queued.
        assert gate.queue_length() == 2

        # A releases: the slot goes to the LIVE waiter (not the disconnected
        # one, which is gone), FIFO.
        gate.release(lease_a)
        lease_live = await asyncio.wait_for(live, timeout=2.0)
        assert gate.running() == 1
        gate.release(lease_live)
        return gate.stats()

    stats = asyncio.run(drive())
    assert stats["running"] == 0 and stats["waiting"] == 0


def test_finding2_queued_disconnect_openai_handler_never_generates():
    """CODEX FINDING 2 (handler level): a streaming completion QUEUED behind a
    running one, whose ASGI connection reports http.disconnect while waiting,
    returns a closed stream WITHOUT entering the engine — and the queue
    continues correctly for the running one. Pre-fix the handler awaited the
    bare gate and a real queued disconnect (which does NOT cancel the task) let
    the dead client be promoted and generate."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    class _BlockingEngine:
        model_family = "chatml"
        model_id = "m"

        def __init__(self):
            self.cfg = SimpleNamespace(enable_thinking=False)
            self.seen: list[str] = []
            self.release = None

        async def generate_stream_batches_async(self, messages, **kwargs):
            tag = messages[-1].get("content") if messages else "?"
            self.seen.append(tag)
            if tag == "first":
                await self.release.wait()
            yield [GenerationResult(
                finish_reason="stop", prompt_tokens=1, completion_tokens=1
            )]

    eng = _BlockingEngine()
    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    openai_compat.set_engines({"m": eng}, eng)

    async def run_one(text: str, raw_request=None):
        req = ChatCompletionRequest(
            model="m", stream=True, user=None,
            messages=[ChatMessage(role="user", content=text)],
        )
        resp = await openai_compat.chat_completions(req, raw_request=raw_request)
        async for _chunk in resp.body_iterator:
            pass

    async def drive():
        eng.release = asyncio.Event()
        gate = inference_queue.get_inference_gate()
        t1 = asyncio.ensure_future(run_one("first"))
        await asyncio.sleep(0.03)  # first running (holds the slot open)

        fake = _FakeAsgiRequest()
        t2 = asyncio.ensure_future(run_one("second", raw_request=fake))
        await asyncio.sleep(0.03)  # second QUEUED on the gate (handler acquire)
        assert gate.queue_length() == 2

        fake.disconnect()  # the QUEUED client's ASGI connection drops
        # The handler returns cleanly (closed stream) — no exception, no engine.
        await asyncio.wait_for(t2, timeout=2.0)
        assert gate.queue_length() == 1  # ticket dropped, running one remains

        eng.release.set()
        await asyncio.wait_for(t1, timeout=2.0)
        return gate.queue_length()

    try:
        depth = asyncio.run(drive())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert eng.seen == ["first"]  # the disconnected queued request NEVER ran
    assert depth == 0


def test_finding3_nonstreaming_openai_queued_disconnect_never_generates(monkeypatch):
    """CODEX ROUND 3, FINDING 3 — the NON-STREAMING OpenAI branch must be
    disconnect-aware too. Pre-fix it used a bare ``inference_slot()`` with no
    disconnect watcher; uvicorn does NOT cancel the app task on socket loss, so a
    dead queued non-streaming request was PROMOTED and ran the full
    ``_sync_completion`` / engine generation. Post-fix it races the acquire
    against the ASGI disconnect watcher (like the streaming path): a client that
    left while queued is dropped BEFORE ``_sync_completion``, returns a clean
    closed (204) response, releases the lease, and the queue continues."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    called = {"sync": 0}

    def _fake_sync(request, engine):
        # If a queued-then-disconnected request ever reaches here, generation
        # ran — the exact bug. Post-fix this is never called for that request.
        called["sync"] += 1
        return {"ok": True}

    monkeypatch.setattr(openai_compat, "_sync_completion", _fake_sync)

    eng = _StubApiEngine()
    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    openai_compat.set_engines({"m": eng}, eng)

    async def drive():
        gate = inference_queue.get_inference_gate()
        holder_lease = await gate.acquire()  # occupy the single slot
        assert gate.queue_length() == 1

        fake = _FakeAsgiRequest()
        req = ChatCompletionRequest(
            model="m", stream=False, user=None,
            messages=[ChatMessage(role="user", content="hi")],
        )
        t = asyncio.ensure_future(
            openai_compat.chat_completions(req, raw_request=fake)
        )
        await asyncio.sleep(0.03)  # the non-streaming request is QUEUED on the gate
        assert gate.queue_length() == 2

        fake.disconnect()  # the QUEUED client's ASGI connection drops
        resp = await asyncio.wait_for(t, timeout=2.0)
        # A clean closed 204 (client is gone) and generation NEVER entered.
        assert resp.status_code == 204
        assert called["sync"] == 0
        assert gate.queue_length() == 1  # ticket dropped, only the holder remains

        gate.release(holder_lease)
        return gate.queue_length()

    try:
        depth = asyncio.run(drive())
    finally:
        openai_compat.set_engines(old_e, old_d)

    assert depth == 0  # lease/counter clean, no leak


# --- [3] pre-body disconnect still releases the slot (unmasked) --------------


@pytest.mark.no_gate_reset
def test_finding3_pre_body_disconnect_releases_slot_without_reset_mask():
    """CODEX FINDING 3 — pre-body disconnect. Under uvicorn's ASGI
    spec_version 2.3, Starlette can cancel body streaming BEFORE the body
    generator's first __anext__ (the connection is already gone when the
    handler returns). An unstarted async generator never runs its try/finally,
    so a release living ONLY in the body generator's finally leaks _running=1
    forever (server wedged at concurrency-1).

    SlotStreamingResponse.__call__ releases via the RESPONSE lifecycle, so the
    never-started case still releases. This test runs WITHOUT the autouse gate
    reset (``no_gate_reset``) so the leak is not masked — the assertion itself
    is the detector (pre-fix: running stays 1)."""
    inference_queue._reset_for_tests()  # start clean (our own hygiene)
    try:
        gate = inference_queue.get_inference_gate()

        async def drive():
            started = {"body": False}

            async def inner():
                started["body"] = True
                try:
                    yield "data: x\n\n"
                finally:
                    started["inner_finally"] = True

            lease = await gate.acquire()
            assert gate.running() == 1
            resp = SlotStreamingResponse(
                gate, lease, inner(), media_type="text/event-stream"
            )

            async def receive():
                # Already disconnected when the response runs.
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    # Yield so the disconnect listener cancels body streaming
                    # BEFORE the body generator's first __anext__.
                    await asyncio.sleep(0.02)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "method": "POST",
                "headers": [],
            }
            await resp(scope, receive, send)
            return started

        started = asyncio.run(drive())
        # The body generator NEVER started (pre-body cancel) ...
        assert started["body"] is False
        assert "inner_finally" not in started
        # ... yet the slot was released by the response lifecycle (no leak).
        assert gate.running() == 0
        assert gate.queue_length() == 0
    finally:
        inference_queue._reset_for_tests()  # hygiene for the next test


# --- [4] disconnect during send: inner teardown BEFORE slot release ----------


def test_finding4_inner_teardown_completes_before_slot_release():
    """CODEX FINDING 4 — disconnect while a chunk is being sent. The body
    generator is closed at its yield; the slot must be released ONLY AFTER the
    inner generation stream's teardown (its cancel_event set / C1
    commit-or-invalidate) has fully run — otherwise the next request is admitted
    before the prior turn wound down. This asserts the STRICT ordering: inner
    teardown, THEN release."""
    events: list[str] = []

    async def inner():
        try:
            yield "a"
            yield "b"
        finally:
            # Stand-in for the engine's U13/C1 teardown (runs on aclose).
            events.append("inner_teardown")

    class _RecordingGate:
        def release(self, lease):
            events.append("released")

    async def drive():
        resp = SlotStreamingResponse(
            _RecordingGate(), Lease(), inner(),
            media_type="text/event-stream",
        )
        it = resp.body_iterator
        first = await it.__anext__()  # "a" — now suspended AT the next yield
        assert first == "a"
        # Disconnect during send == the outer stream closes the body iterator
        # while it is suspended at a yield.
        await it.aclose()

    asyncio.run(drive())
    # Inner teardown ran to completion BEFORE the slot was released (so the next
    # request cannot be admitted until this turn is wound down).
    assert events == ["inner_teardown", "released"]


# --- round 3 [1] release strictly AFTER the NESTED engine teardown ------------


def test_finding1_release_strictly_after_nested_engine_teardown():
    """CODEX ROUND 3, FINDING 1 — the lease must release ONLY AFTER the engine
    C1 teardown, ACROSS THE NESTED WRAPPER CHAIN. The round-1 finding-4 test
    above used a single FLAT generator whose whole teardown lives inside its own
    aclose — it never proved the CASCADE. Here the REAL openai wrapper chain is
    driven end-to-end:

        SlotStreamingResponse._inner
          -> openai_compat._stream_completion          (top wrapper)
            -> _stream_completion_body                 (nested wrapper)
              -> engine.generate_stream_batches_async  (the engine generator)

    with a fake engine whose teardown AWAITS a done-ack (an async step — the
    "engine-completion-ack" that in production is the in-proc worker's C1 done
    sentinel / the process child's done frame).

    Pre-fix, closing the TOP wrapper did NOT cascade — ``async for`` does not
    close a nested async generator — so the engine generator's GeneratorExit
    rescue (its C1 commit-or-invalidate) ran only LATER (on GC), i.e. the lease
    was released BEFORE the engine teardown (events == ['released']). Post-fix
    each wrapper closes its inner in a finally, so the close cascades all the way
    to the engine generator and its teardown is awaited BEFORE the release."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    events: list[str] = []

    class _NestedEngine:
        model_family = "chatml"
        model_id = "m"

        def __init__(self):
            self.cfg = SimpleNamespace(enable_thinking=False)
            self.hold = asyncio.Event()  # never set -> generation suspends

        async def generate_stream_batches_async(self, messages, **kwargs):
            try:
                yield [GenerationResult(text="hi", status=None, finish_reason=None)]
                # Suspend mid-generation so the disconnect closes us WHILE the
                # engine loop is in flight (a real interrupted turn).
                await self.hold.wait()
                yield [GenerationResult(
                    finish_reason="stop", prompt_tokens=1, completion_tokens=1
                )]
            finally:
                # The engine C1 teardown, which AWAITS a worker/child done-ack.
                # The two markers straddle the await so a release that slipped in
                # mid-teardown (pre-fix) would be observable between them.
                events.append("engine_teardown_start")
                await asyncio.sleep(0)
                events.append("engine_teardown_done")

    class _RecordingGate:
        def release(self, lease):
            events.append("released")

    eng = _NestedEngine()
    old_e, old_d = openai_compat._engines, openai_compat._default_engine
    openai_compat.set_engines({"m": eng}, eng)

    async def drive():
        req = ChatCompletionRequest(
            model="m", stream=True, user=None,
            messages=[ChatMessage(role="user", content="hi")],
        )
        resp = SlotStreamingResponse(
            _RecordingGate(), Lease(),
            openai_compat._stream_completion(req, eng),
            media_type="text/event-stream",
        )
        it = resp.body_iterator
        # Pull the role chunk + the first content chunk ("hi") so the engine loop
        # is genuinely in flight, then disconnect during send.
        await it.__anext__()  # role delta
        await it.__anext__()  # content "hi"
        await it.aclose()     # client disconnect mid-stream

    try:
        asyncio.run(drive())
    finally:
        openai_compat.set_engines(old_e, old_d)

    # The engine teardown ran to COMPLETION (both markers, across the nested
    # chain) STRICTLY before the lease was released.
    assert events == [
        "engine_teardown_start",
        "engine_teardown_done",
        "released",
    ]


# --- round 3 [2] teardown is cancellation-safe / shielded ---------------------


def test_finding2_cancel_during_c1_completes_shielded_then_releases():
    """CODEX ROUND 3, FINDING 2 — teardown must be cancellation-safe. Pre-fix the
    close-and-release used ``suppress(Exception)`` (CancelledError is a
    BaseException, so it was NOT suppressed) with no shield: a cancellation of
    the app task while ``_inner.aclose()`` was mid-C1 aborted the C1 teardown AND
    skipped the release (leaked slot) OR left it for a later retry that could
    release with C1 incomplete.

    Post-fix the inner close runs as a SHIELDED task: cancelling the outer task
    keeps C1 alive to its done marker, the release runs EXACTLY once in an
    outermost finally (order: c1-done THEN released), and the CancelledError is
    re-raised afterwards (not swallowed)."""
    events: list[str] = []
    released = {"count": 0}

    async def inner():
        try:
            yield "a"
        finally:
            events.append("c1-start")
            await asyncio.sleep(0.05)  # C1 in progress (the mid-commit window)
            events.append("c1-done")

    class _CountingGate:
        def release(self, lease):
            released["count"] += 1
            events.append("released")

    async def drive():
        resp = SlotStreamingResponse(
            _CountingGate(), Lease(), inner(),
            media_type="text/event-stream",
        )
        it = resp.body_iterator
        await it.__anext__()  # "a" — inner suspended at its yield
        # Run the close-and-release as its own task, let it enter the C1 await,
        # then cancel it mid-C1.
        t = asyncio.ensure_future(resp._teardown())
        await asyncio.sleep(0.01)  # teardown running; C1 mid-flight
        assert events == ["c1-start"]  # C1 started, not done, nothing released
        t.cancel()  # app-task cancellation lands DURING the C1 teardown
        with pytest.raises(asyncio.CancelledError):
            await t  # the cancellation is re-raised (not swallowed)

    asyncio.run(drive())
    # C1 reached its done marker (shielded) AND the lease was released exactly
    # once, strictly AFTER C1 — no leaked/held slot, no double release.
    assert events == ["c1-start", "c1-done", "released"]
    assert released["count"] == 1


# --- [5] compaction gated + auto-compaction lease-reuse ----------------------


def test_finding5_standalone_compaction_is_gated_429_when_full(monkeypatch):
    """CODEX FINDING 5 — the standalone compaction endpoint bypassed the gate,
    so a compaction could run CONCURRENTLY with a gated generation and overtake
    queued requests. It now acquires the SAME FIFO gate: with the gate at its
    bound, the compaction endpoint answers a real 429 queue_full (pre-fix it
    proceeded — 200 — overtaking every queued completion)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import compaction as compaction_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    class _DB:
        async def get_session(self, session_id):
            return {"id": session_id, "system_prompt": ""}

    monkeypatch.setattr(compaction_mod, "db", _DB())
    compaction_mod.set_engine(SimpleNamespace(model_family="chatml", model_id="m"))

    gate = inference_queue.get_inference_gate()
    gate.set_max_queue_length(2)
    gate._running = 1
    gate._waiting = 1  # depth 2 == max -> the gate is FULL

    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(compaction_mod.router)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/api/sessions/s1/compact", json={})
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "queue_full"


def test_finding5_auto_compaction_reuses_lease_no_reacquire(monkeypatch):
    """CODEX FINDING 5 — auto-compaction lease-reuse (round 3 observation 4: made
    HONEST). Auto-compaction triggered from WITHIN a gated chat request must run
    under the chat's EXISTING lease, never re-acquire the gate (that
    self-deadlocks at concurrency-1) and never double-count the bound.

    Round-3 note: the production ``CompactionEngine`` exposes NO ``compact()``
    method, so the auto-compaction call in chat.py raises AttributeError that is
    SWALLOWED — auto-compaction is currently DEAD (a separate pre-existing bug,
    see the TODO at chat.py's utilization>=90 block). The prior version of THIS
    test was therefore a FALSE POSITIVE: it proved 'one acquire + a swallowed
    failure', not real lease reuse. Here we STUB ``compact()`` so it actually
    GENERATES (reaches the engine) under the chat's held lease, and assert it
    (a) truly RAN (not swallowed — the compaction DB records land), (b) did NOT
    re-acquire the gate (acquire_count stays 1, even mid-compact), and (c) did
    not deadlock (a response is returned)."""
    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.engine import compaction as compaction_engine_mod

    class _CountingGate(InferenceGate):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.acquire_count = 0

        async def acquire(self):
            self.acquire_count += 1
            return await super().acquire()

    counting = _CountingGate(max_queue_length=1)  # bound 1: a reacquire => 429
    monkeypatch.setattr(inference_queue, "_GATE", counting)

    ran = {"compacted": False, "acquire_count_during_compact": None}

    class _StubCompactionEngine:
        """Honest stub: ``compact()`` EXISTS and actually runs (reaches the
        engine) under the chat's ALREADY-held lease, WITHOUT touching the gate."""

        def __init__(self, engine):
            self.engine = engine

        async def compact(self, *, messages, strategy, target_tokens,
                          keep_recent_turns):
            # Prove we run under the EXISTING lease: no gate re-acquire here.
            ran["acquire_count_during_compact"] = counting.acquire_count
            # Model reaching the engine (a real generation would happen here).
            _ = self.engine.model_family
            ran["compacted"] = True
            return {"new_tokens": 50, "reduction_percent": 47.4, "summary": "S"}

    # The chat handler does ``from ...engine.compaction import CompactionEngine``
    # INSIDE the utilization block, so patching the module attribute is picked up.
    monkeypatch.setattr(
        compaction_engine_mod, "CompactionEngine", _StubCompactionEngine
    )

    db_calls = {"record_compaction": 0, "update_session_tokens": 0}

    class _DB:
        async def get_session(self, session_id):
            return {
                "id": session_id,
                "system_prompt": "",
                "context_window_limit": 100,
            }

        async def add_message(self, *a, **k):
            return {"id": "u1"}

        async def get_messages(self, *a, **k):
            return [{"role": "user", "content": "hi", "id": "u1"}]

        async def get_session_total_tokens(self, *a, **k):
            return 95  # 95% of the 100-token window -> auto-compaction fires

        async def record_compaction(self, **k):
            db_calls["record_compaction"] += 1

        async def update_session_tokens(self, *a, **k):
            db_calls["update_session_tokens"] += 1

    monkeypatch.setattr(chat_mod, "db", _DB())

    cfg = SimpleNamespace(
        default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
        default_top_k=0, default_repetition_penalty=1.0, thinking_budget=2048,
        default_max_tokens=256, enable_thinking=False,
    )
    eng = SimpleNamespace(model_family="chatml", model_id="m", cfg=cfg)
    chat_mod.set_engines({"m": eng}, eng)

    from mlx_soloheaven.api.chat import SendMessageRequest

    async def drive():
        req = SendMessageRequest(content="hi", stream=True, model="m")
        resp = await chat_mod.chat("s1", req, request=None)
        # No deadlock: the gated chat + its auto-compaction returned a response.
        assert isinstance(resp, SlotStreamingResponse)
        # Auto-compaction TRULY RAN (reached the engine, not swallowed) — the
        # post-compaction DB records only land on a successful compact().
        assert ran["compacted"] is True
        assert db_calls["record_compaction"] == 1
        assert db_calls["update_session_tokens"] == 1
        # ... and it ran under the EXISTING lease: exactly ONE acquire for the
        # whole turn, no re-acquire mid-compact, and the bound was not
        # double-counted (one running slot, no queued waiter).
        assert ran["acquire_count_during_compact"] == 1
        assert counting.acquire_count == 1
        assert counting.stats()["running"] == 1
        assert counting.stats()["waiting"] == 0
        # Release the streaming lease (the response owns it) for hygiene.
        with contextlib.suppress(Exception):
            await resp.body_iterator.aclose()
        await resp._teardown()

    asyncio.run(drive())
    chat_mod.set_engines({}, None)


# --- [6] release is lease-owned: idempotent + ownership-safe -----------------


def test_finding6_release_is_idempotent_and_ownership_safe():
    """CODEX FINDING 6 — release() ownership safety via a lease token. A double
    release (after successor B has taken the slot), a stale release, and a
    foreign/no-lease release are all NO-OPS — none clears the live owner's slot
    nor wakes a waiter behind its back. Pre-fix release() was lease-less and a
    second call cleared B's ownership and woke C."""

    async def drive():
        gate = InferenceGate(max_queue_length=4)
        lease_a = await gate.acquire()  # A owns
        b_task = asyncio.ensure_future(gate.acquire())  # B queues
        for _ in range(3):
            await asyncio.sleep(0)
        assert gate.queue_length() == 2

        gate.release(lease_a)          # hand the slot to B
        lease_b = await b_task          # B resumes, now OWNS (running=1)
        assert gate.running() == 1 and gate._owner is lease_b

        # DOUBLE / STALE release of A's lease: A is no longer the owner -> no-op.
        gate.release(lease_a)
        assert gate.running() == 1 and gate._owner is lease_b

        # FOREIGN release (a lease that never held the slot) -> no-op.
        gate.release(Lease())
        assert gate.running() == 1 and gate._owner is lease_b

        # NO-LEASE release -> no-op (never disturbs the live owner).
        gate.release(None)
        assert gate.running() == 1 and gate._owner is lease_b

        # The paired normal path still works: the true owner releases the slot.
        gate.release(lease_b)
        assert gate.running() == 0 and gate._owner is None
        return gate.queue_length()

    assert asyncio.run(drive()) == 0


# ---------------------------------------------------------------------------
# Round 4 — REAL-chain cascade-close coverage (codex: the round-3 tests were
# false-green because they used fakes that BYPASS the real streaming wrappers).
#
# These drive the ACTUAL wrappers end-to-end so a missed hoist / unclosed inner
# generator fails the assertion:
#   [1] EngineProcessProxy.generate_stream_batches_async / generate_stream_async
#       -> the REAL private _stream cancel-ack path (child cancel + done ack).
#   [2] the REAL api/compaction._stream_compact chain -> the REAL
#       CompactionEngine.generate_summary_stream -> engine.generate_stream_async.
#   [3] the REAL MLXEngine.generate_stream_batches_async sentinel/drain logic,
#       bound to a lightweight fake engine — NOT a flat fake generator.
# ---------------------------------------------------------------------------

from mlx_soloheaven.engine.mlx_engine import MLXEngine  # noqa: E402
from mlx_soloheaven.engine.process_client import EngineProcessProxy  # noqa: E402


class _RecordingConn:
    """Pipe-connection stand-in: records frames sent parent -> child."""

    def __init__(self):
        self.sent: list = []

    def send(self, obj):
        self.sent.append(obj)


def _make_proxy_shell():
    """An EngineProcessProxy with exactly the surface _stream +
    _await_child_cancel_ack touch — no real child process/pipes."""
    proxy = EngineProcessProxy.__new__(EngineProcessProxy)
    proxy.ensure_available = lambda: None  # instance shadow of the method
    proxy._queues = {}
    proxy._loops = {}
    proxy._routing_lock = threading.Lock()
    proxy._inflight = 0
    proxy._inflight_lock = threading.Lock()
    proxy._cmd_parent = _RecordingConn()
    proxy._ctrl_parent = _RecordingConn()
    proxy._cmd_send_lock = threading.Lock()
    proxy._ctrl_send_lock = threading.Lock()
    proxy._dead = False
    proxy._permanently_dead = False
    proxy._closed = False
    proxy._dead_reason = ""
    return proxy


async def _await_stream_queue(proxy):
    """Spin until _stream registers its per-request queue; return (rid, q)."""
    for _ in range(500):
        await asyncio.sleep(0)
        with proxy._routing_lock:
            if proxy._queues:
                rid = next(iter(proxy._queues))
                return rid, proxy._queues[rid]
    raise AssertionError("process _stream never registered its queue")


def test_process_proxy_batches_close_cascades_cancel_ack_before_release():
    """CODEX ROUND 4, FINDING 1 (batched) — closing the PUBLIC process-proxy
    generator must CASCADE into the private _stream: send the child cancel and
    AWAIT the child's terminal done-ack (the child's C1 teardown) BEFORE the
    aclose returns (== before the gate lease is released).

    Pre-fix ``generate_stream_batches_async`` was a plain
    ``async for batch in self._stream(...): yield`` that never closed _stream —
    closing the public generator sent NO cancel and the aclose returned
    immediately (no done-ack awaited)."""
    proxy = _make_proxy_shell()

    async def drive():
        gen = proxy.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}]
        )
        first = asyncio.ensure_future(gen.__anext__())
        rid, q = await _await_stream_queue(proxy)
        # A generate command reached the child.
        assert any(f.get("op") == "generate" for f in proxy._cmd_parent.sent)
        # The child emits a content batch; the consumer observes it.
        q.put_nowait({"id": rid, "type": "batch", "items": [{"text": "hi"}]})
        batch = await asyncio.wait_for(first, timeout=5.0)
        assert [r.text for r in batch] == ["hi"]

        # Close the PUBLIC proxy generator (client disconnect / the response
        # wrapper closing it). This must reach _stream's cancel handler.
        close_task = asyncio.ensure_future(gen.aclose())
        for _ in range(500):
            await asyncio.sleep(0)
            if proxy._ctrl_parent.sent:
                break
        # (1) the cascade reached _stream — a cancel was sent to the child ...
        assert any(
            f.get("op") == "cancel" and f.get("id") == rid
            for f in proxy._ctrl_parent.sent
        ), "closing the public proxy generator did not cascade a cancel into _stream"
        # (2) ... and the aclose is BLOCKED awaiting the child's terminal done
        # ack (C1) — it must NOT return before the child winds the turn down.
        assert not close_task.done(), "aclose returned before the child done-ack"

        # The child acks: closes generate_stream (C1) then emits done.
        q.put_nowait({"id": rid, "type": "done"})
        await asyncio.wait_for(close_task, timeout=5.0)
        assert proxy._inflight == 0  # routing bookkeeping unwound

    asyncio.run(drive())


def test_process_proxy_scalar_close_cascades_cancel_ack_before_release():
    """CODEX ROUND 4, FINDING 1 (scalar) — same cascade for
    ``generate_stream_async`` (used by the compaction summarizer)."""
    proxy = _make_proxy_shell()

    async def drive():
        gen = proxy.generate_stream_async([{"role": "user", "content": "hi"}])
        first = asyncio.ensure_future(gen.__anext__())
        rid, q = await _await_stream_queue(proxy)
        assert any(
            f.get("op") == "generate_scalar" for f in proxy._cmd_parent.sent
        )
        q.put_nowait({"id": rid, "type": "batch", "items": [{"text": "hi"}]})
        item = await asyncio.wait_for(first, timeout=5.0)
        assert item.text == "hi"

        close_task = asyncio.ensure_future(gen.aclose())
        for _ in range(500):
            await asyncio.sleep(0)
            if proxy._ctrl_parent.sent:
                break
        assert any(
            f.get("op") == "cancel" and f.get("id") == rid
            for f in proxy._ctrl_parent.sent
        ), "closing the public scalar proxy generator did not cascade a cancel"
        assert not close_task.done(), "aclose returned before the child done-ack"

        q.put_nowait({"id": rid, "type": "done"})
        await asyncio.wait_for(close_task, timeout=5.0)
        assert proxy._inflight == 0

    asyncio.run(drive())


def test_finding2_compaction_chain_release_after_engine_teardown(monkeypatch):
    """CODEX ROUND 4, FINDING 2 — the REAL compaction chain must cascade close
    to the engine generator's teardown BEFORE the lease releases:

        SlotStreamingResponse._inner
          -> api/compaction._stream_compact          (top wrapper)
            -> _stream_compact_body                  (nested wrapper)
              -> CompactionEngine.generate_summary_stream   (the fixed layer)
                -> engine.generate_stream_async      (the engine generator)

    Pre-fix ``generate_summary_stream`` was a plain
    ``async for chunk in self.engine.generate_stream_async(...): yield`` that
    never closed the engine generator — a compaction client disconnect released
    the lease WITHOUT the engine C1 teardown (events == ['released'])."""
    from mlx_soloheaven.api import compaction as compaction_mod

    events: list[str] = []

    class _CompactionLeafEngine:
        model_family = "chatml"
        model_id = "m"

        def __init__(self):
            self.hold = asyncio.Event()  # never set -> mid-generation suspend

        async def generate_stream_async(self, messages, **kwargs):
            try:
                yield GenerationResult(text="draft ")
                await self.hold.wait()
                yield GenerationResult(text="more", finish_reason="stop")
            finally:
                # The engine C1 teardown (straddles an await so a release that
                # slipped in mid-teardown would be observable between them).
                events.append("engine_teardown_start")
                await asyncio.sleep(0)
                events.append("engine_teardown_done")

    class _DB:
        async def get_messages(self, session_id):
            # > keep_recent(6) + 2 so summarize() actually proceeds.
            return [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"m{i}",
                }
                for i in range(12)
            ]

    class _RecordingGate:
        def release(self, lease):
            events.append("released")

    eng = _CompactionLeafEngine()
    monkeypatch.setattr(compaction_mod, "db", _DB())
    old_engine = compaction_mod.engine
    compaction_mod.set_engine(eng)

    req = compaction_mod.CompactionRequest()
    session = {"id": "s1", "system_prompt": "sys"}

    async def drive():
        resp = SlotStreamingResponse(
            _RecordingGate(), Lease(),
            compaction_mod._stream_compact("s1", session, req),
            media_type="text/event-stream",
        )
        it = resp.body_iterator
        # Pull the 'start' event + the first summary text chunk so the engine
        # leaf is genuinely mid-generation (suspended at its first yield).
        await it.__anext__()  # {"type": "start", ...}
        await it.__anext__()  # {"type": "text", "content": "draft "}
        await it.aclose()     # client disconnect mid-stream

    try:
        asyncio.run(drive())
    finally:
        compaction_mod.set_engine(old_engine)

    # The engine C1 teardown ran to completion (across the nested compaction
    # chain) STRICTLY before the lease was released.
    assert events == [
        "engine_teardown_start",
        "engine_teardown_done",
        "released",
    ]


# ---------------------------------------------------------------------------
# Round 5 — WORKER-side C1-teardown ordering (APPROVE-WITH-NITS, NIT 2).
#
# The round-4 cascade tests above prove the PROXY side (closing the public
# generator cancels the child and awaits its terminal ack before releasing the
# lease) but INJECT the ``done`` frame by hand, so they never exercise the
# CHILD actually running generate_stream's C1 teardown (its GeneratorExit
# rescue) BEFORE emitting a terminal frame. These drive the REAL worker
# generation path (process_worker._run_generate) with a recording fake engine
# whose generator records when its C1 rescue / idempotent teardown runs, and
# assert close-BEFORE-frame ordering on BOTH:
#   (a) NORMAL completion  -> teardown precedes the ``done`` frame;
#   (b) a DRIVER-SIDE fault -> teardown precedes the ``error`` frame.
#
# The proxy frees the lease on EITHER a ``done`` OR an ``error`` ack
# (process_client._await_child_cancel_ack treats both as C1-complete), so C1
# must precede BOTH. Pre-NIT-1 gen.close() lived only on the straight-line
# success path below the loop, so a driver-side exception emitted the ``error``
# frame with the generator still suspended (C1 unrun) — case (b) FAILS pre-fix
# (error frame before teardown) and PASSES post-fix (finally runs gen.close()).
# ---------------------------------------------------------------------------

from mlx_soloheaven.engine import process_protocol as _proto  # noqa: E402
from mlx_soloheaven.engine.process_worker import _run_generate  # noqa: E402


class _OrderingConn:
    """resp-pipe stand-in recording each frame's type into a SHARED timeline
    so a generator teardown recorded into the same list can be strictly
    ordered against the emitted frames."""

    def __init__(self, events):
        self._events = events
        self.frames: list = []

    def send(self, frame):
        self.frames.append(frame)
        self._events.append(("frame", frame["type"]))


class _BadYieldedResult:
    """A yielded object that fails the driver's ``result.status`` read
    (process_worker ~775) — models generate_stream handing back an invalid
    object mid-stream: a DRIVER-SIDE fault (not an engine cancel), leaving the
    generator suspended so only gen.close() can tear it down."""

    @property
    def status(self):
        raise ValueError("invalid yielded object")


def _recording_c1_engine(events, *, fail_mid_stream):
    """Engine stand-in whose generate_stream returns a generator that records
    its C1 GeneratorExit rescue + idempotent teardown into ``events`` —
    mirroring MLXEngine.generate_stream's commit-or-invalidate teardown, which
    runs exactly once whether reached via the normal post-loop path or the
    GeneratorExit rescue (guarded there by _turn_committed / _stream_reconciled;
    modelled here by the ``torn_down`` guard so a redundant close is a no-op)."""

    class _Engine:
        def generate_stream(self, messages, **kwargs):
            torn_down = False

            def _gen():
                nonlocal torn_down
                try:
                    yield GenerationResult(text="hello")
                    if fail_mid_stream:
                        # Suspend at a yield of an INVALID object: the driver
                        # raises when it reads result.status, leaving this
                        # generator suspended (only gen.close() closes it).
                        yield _BadYieldedResult()
                    else:
                        yield GenerationResult(text="", finish_reason="stop")
                except GeneratorExit:
                    # The C1 GeneratorExit rescue (cancel / driver-error close).
                    events.append("c1_rescue")
                    raise
                finally:
                    # Idempotent teardown marker — the single convergence point
                    # for the normal post-loop path AND the rescue.
                    if not torn_down:
                        torn_down = True
                        events.append("teardown")

            return _gen()

    return _Engine()


def _drive_worker_generate(engine, events):
    """Run the REAL worker generation path exactly as worker_main does:
    ``_run_generate`` inside the outer try/except that turns a driver-side
    exception into an ``error`` frame on the SAME resp conn (worker_main ~640).
    Returns the recording conn (its ``.frames`` + the shared ``events``)."""
    import traceback

    conn = _OrderingConn(events)
    rid = "r-worker"
    cancel_event = threading.Event()
    try:
        _run_generate(
            engine, rid, [{"role": "user", "content": "hi"}], {},
            cancel_event, conn,
            coalesce_n=8, coalesce_ms=1000, coalescing=True,
        )
    except Exception as e:  # noqa: BLE001 — mirror worker_main's outer sender
        conn.send(_proto.make_error(rid, str(e), traceback.format_exc()))
    return conn


def test_worker_run_generate_normal_completion_c1_before_done():
    """NIT 2 case (a): on NORMAL completion the worker runs the engine
    generator's C1 teardown BEFORE pushing the ``done`` frame, and the
    redundant gen.close() below the loop is a no-op (teardown recorded once)."""
    events: list = []
    conn = _drive_worker_generate(
        _recording_c1_engine(events, fail_mid_stream=False), events
    )

    # A terminal ``done`` frame was emitted (normal completion) — no error.
    types = [f["type"] for f in conn.frames]
    assert types[-1] == "done"
    assert "error" not in types

    # Teardown ran exactly once, STRICTLY before the done frame.
    assert events.count("teardown") == 1
    assert events.index("teardown") < events.index(("frame", "done"))
    # Natural exhaustion never throws GeneratorExit -> no rescue path taken.
    assert "c1_rescue" not in events


def test_worker_run_generate_driver_error_c1_before_error_frame():
    """NIT 2 case (b): a DRIVER-SIDE fault (invalid yielded object) must STILL
    run the engine generator's C1 teardown (via the new finally) BEFORE the
    worker's outer ``error`` frame. FAILS pre-fix (error frame emitted with the
    generator still suspended, C1 unrun) and PASSES post-fix."""
    events: list = []
    conn = _drive_worker_generate(
        _recording_c1_engine(events, fail_mid_stream=True), events
    )

    # The driver fault surfaced as an ``error`` frame (worker_main's sender).
    types = [f["type"] for f in conn.frames]
    assert types[-1] == "error"
    assert "done" not in types
    # The original driver error is carried on the frame (not swallowed).
    assert "invalid yielded object" in conn.frames[-1]["error"]

    # The C1 rescue + idempotent teardown ran, STRICTLY before the error frame.
    assert "c1_rescue" in events
    assert events.count("teardown") == 1
    assert events.index("c1_rescue") < events.index(("frame", "error"))
    assert events.index("teardown") < events.index(("frame", "error"))


class _FakeBatchEngine:
    """Lightweight engine exposing exactly what the REAL
    ``generate_stream_batches_async`` touches, so the test drives the REAL
    sentinel/drain logic (NOT a flat fake generator that bypasses it)."""

    # The REAL method, bound onto this fake engine as `self`.
    generate_stream_batches_async = MLXEngine.generate_stream_batches_async

    def __init__(self):
        # coalesce_n <= 1 -> every result is its own 1-item batch.
        self.cfg = SimpleNamespace(stream_coalesce_n=1, stream_coalesce_ms=0)
        self._use_vlm = False  # -> one-shot daemon worker thread
        self.gen_running = threading.Event()
        self.proceed = threading.Event()
        self.c1_done = threading.Event()
        self.c1_count = 0

    def generate_stream(self, messages, **kwargs):
        try:
            self.gen_running.set()
            assert self.proceed.wait(timeout=5.0)
            # A single TERMINAL result: the worker flushes it as its own batch,
            # then (loop exhausted) closes this generator (C1) and posts None.
            yield GenerationResult(
                finish_reason="stop", prompt_tokens=1, completion_tokens=1
            )
        finally:
            self.c1_count += 1  # C1 commit-or-invalidate (runs on gen.close())
            self.c1_done.set()


def test_finding3_inproc_normal_completion_aclose_immediate_no_30s_stall():
    """CODEX ROUND 4, FINDING 3 — in-proc NORMAL completion must not stall 30s.

    The terminal batch AND the None sentinel are both queued BEFORE the consumer
    handles the terminal batch; the consumer observes the terminal batch and
    BREAKS, suspending the engine generator at the backlog-drain yield with the
    ONLY None already consumed. Pre-fix the GeneratorExit handler called
    ``_drain_worker_until_done`` for a None that never arrives -> a full
    30s (_CANCEL_C1_DRAIN_TIMEOUT_S) stall holding the FIFO lease. Post-fix the
    sentinel-aware drain skips it (C1 already ran) -> aclose is IMMEDIATE."""
    engine = _FakeBatchEngine()

    async def drive():
        gen = engine.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}], session_id="s"
        )
        first = asyncio.ensure_future(gen.__anext__())
        # Let __anext__ spawn the worker + reach `await q.get()`; the worker
        # blocks in generate_stream on `proceed` (the queue is still empty).
        for _ in range(500):
            await asyncio.sleep(0)
            if engine.gen_running.is_set():
                break
        assert engine.gen_running.is_set()

        # Release the worker and BLOCK the loop thread until it has posted the
        # terminal batch, run C1 (gen.close), and scheduled the None sentinel —
        # so BOTH are queued before the consumer's drain runs (the exact
        # interleaving codex reproduced). The loop is frozen during this wait, so
        # the two call_soon_threadsafe puts land in _ready before the getter
        # resumes and the drain then consumes the None.
        engine.proceed.set()
        assert engine.c1_done.wait(timeout=5.0)  # gen.close() (C1) completed
        time.sleep(0.05)                          # let the None post land

        # The consumer drains: terminal batch yielded, None consumed; the
        # generator is now suspended at that yield with the sentinel gone.
        batch = await asyncio.wait_for(first, timeout=2.0)
        assert batch[0].finish_reason == "stop"
        assert engine.c1_count == 1

        # The consumer observed the terminal batch and BREAKS -> aclose. This
        # must be IMMEDIATE (no 30s drain of an already-consumed sentinel).
        t0 = time.monotonic()
        await asyncio.wait_for(gen.aclose(), timeout=10.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"aclose stalled {elapsed:.2f}s — the sentinel-aware drain regressed "
            f"(30s normal-completion stall is back)"
        )
        assert engine.c1_count == 1  # C1 ran exactly once

    asyncio.run(drive())
