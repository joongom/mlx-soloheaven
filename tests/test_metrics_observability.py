"""Batch D — observability: /metrics (Prometheus), request-ID, per-request
metrics.

Covers:
  * the hand-rolled Prometheus exporter (text format, metric names, counters +
    histograms);
  * observe_request increments requests_total / tokens_total and observes the
    latency histograms;
  * GET /metrics returns valid Prometheus text and is NOT gated by the inference
    queue (answers 200 even when the gate is shut);
  * label cardinality is bounded — no request_id / session_id labels ever;
  * the X-Request-ID middleware echoes a safe incoming id, generates one when
    absent, and rejects an unsafe (injection) id.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_soloheaven.api import metrics
from mlx_soloheaven.api.request_id import RequestIDMiddleware


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


# --------------------------------------------------------------------------
# Exporter / render.
# --------------------------------------------------------------------------

def test_render_lists_all_metric_names():
    text = metrics.render()
    for name in (
        "soloheaven_requests_total",
        "soloheaven_tokens_total",
        "soloheaven_cache_reused_tokens_total",
        "soloheaven_queue_wait_seconds",
        "soloheaven_ttft_seconds",
        "soloheaven_generation_seconds",
    ):
        assert f"# TYPE {name} " in text, f"{name} missing from exposition"


def test_render_types_are_counter_and_histogram():
    text = metrics.render()
    assert "# TYPE soloheaven_requests_total counter" in text
    assert "# TYPE soloheaven_queue_wait_seconds histogram" in text


def test_observe_request_increments_counter_and_histograms():
    metrics.observe_request(
        model="modelX",
        finish_reason="stop",
        error_code="none",
        cache_result="hit",
        reused_tokens=42,
        queue_wait=0.01,
        ttft=0.2,
        generation_time=1.5,
        prompt_tokens=100,
        completion_tokens=20,
    )
    text = metrics.render()
    # Counter with the exact bounded labels.
    assert (
        'soloheaven_requests_total{model="modelX",finish_reason="stop",'
        'error_code="none",cache_result="hit"} 1'
    ) in text
    # Token counters.
    assert 'soloheaven_tokens_total{model="modelX",kind="prompt"} 100' in text
    assert 'soloheaven_tokens_total{model="modelX",kind="completion"} 20' in text
    assert 'soloheaven_cache_reused_tokens_total{model="modelX"} 42' in text
    # Histograms produced bucket / sum / count series for the model.
    assert 'soloheaven_queue_wait_seconds_count{model="modelX"} 1' in text
    assert 'soloheaven_ttft_seconds_count{model="modelX"} 1' in text
    assert 'soloheaven_generation_seconds_count{model="modelX"} 1' in text
    assert 'soloheaven_generation_seconds_bucket{model="modelX",le="+Inf"} 1' in text


def test_histogram_buckets_are_cumulative_and_monotonic():
    # Two observations at known latencies; le-buckets must be cumulative,
    # monotonically non-decreasing, and bounded by _count.
    metrics.observe_request(model="m", finish_reason="stop", generation_time=0.06)
    metrics.observe_request(model="m", finish_reason="stop", generation_time=3.0)
    text = metrics.render()
    lines = [
        ln for ln in text.splitlines()
        if ln.startswith("soloheaven_generation_seconds_bucket")
    ]
    counts = [int(ln.rsplit(" ", 1)[1]) for ln in lines]
    # Non-decreasing across buckets.
    assert counts == sorted(counts), counts
    # le=+Inf equals the total count (2).
    assert 'soloheaven_generation_seconds_bucket{model="m",le="+Inf"} 2' in text
    assert 'soloheaven_generation_seconds_count{model="m"} 2' in text
    # le="0.1" catches only the 0.06 observation (1), le="5" catches both (2).
    assert 'soloheaven_generation_seconds_bucket{model="m",le="0.1"} 1' in text
    assert 'soloheaven_generation_seconds_bucket{model="m",le="5"} 2' in text


def test_observe_request_accumulates():
    for _ in range(3):
        metrics.observe_request(
            model="m", finish_reason="stop", cache_result="miss",
            queue_wait=0.0, generation_time=0.5,
            prompt_tokens=10, completion_tokens=5,
        )
    text = metrics.render()
    assert (
        'soloheaven_requests_total{model="m",finish_reason="stop",'
        'error_code="none",cache_result="miss"} 3'
    ) in text
    assert 'soloheaven_tokens_total{model="m",kind="prompt"} 30' in text


def test_observe_request_is_defensive():
    # Weird / missing values must never raise.
    metrics.observe_request(model=None)
    metrics.observe_request(
        model="m", finish_reason=None, queue_wait=float("nan"),
        ttft=float("inf"), prompt_tokens=0,
    )
    # A NaN/Inf observation is dropped (no series created for it), but the
    # request counter still increments with defaulted labels.
    text = metrics.render()
    assert "soloheaven_requests_total" in text


def test_cache_result_from_info_shapes():
    assert metrics.cache_result_from_info({"cache_mode": "hit", "cached_tokens": 7}) == ("hit", 7)
    assert metrics.cache_result_from_info({"cache_mode": "base_hit", "cached_tokens": 3}) == ("hit", 3)
    assert metrics.cache_result_from_info({"cache_mode": "miss"})[0] == "miss"
    assert metrics.cache_result_from_info({"type": "kv_cache_hit", "cached_tokens": 9}) == ("hit", 9)
    assert metrics.cache_result_from_info({"type": "kv_cache_miss"})[0] == "miss"
    assert metrics.cache_result_from_info(None) == ("unknown", 0)


# --------------------------------------------------------------------------
# Label cardinality is bounded — NEVER request_id / session_id.
# --------------------------------------------------------------------------

def test_labels_never_include_request_or_session_id():
    metrics.observe_request(
        model="m", finish_reason="stop", cache_result="hit",
        queue_wait=0.1, prompt_tokens=1, completion_tokens=1,
    )
    text = metrics.render()
    assert "request_id" not in text
    assert "session_id" not in text


# --------------------------------------------------------------------------
# GET /metrics endpoint.
# --------------------------------------------------------------------------

def _metrics_client() -> TestClient:
    app = FastAPI()
    app.include_router(metrics.router)
    return TestClient(app, raise_server_exceptions=False)


def test_metrics_endpoint_returns_prometheus_text():
    metrics.observe_request(
        model="m", finish_reason="stop", cache_result="hit",
        queue_wait=0.01, generation_time=0.2, prompt_tokens=5, completion_tokens=2,
    )
    r = _metrics_client().get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in r.headers["content-type"]
    assert "soloheaven_requests_total" in r.text
    assert "soloheaven_generation_seconds_bucket" in r.text


def test_metrics_endpoint_not_gated_by_inference_queue():
    """A read endpoint: it must answer even when the inference gate is CLOSED
    (a gated generation endpoint would 503 in this state)."""
    from mlx_soloheaven import inference_queue

    gate = inference_queue.get_inference_gate()
    # Occupy the single running slot AND shut the gate: any generation acquire
    # would now be rejected. /metrics must still answer 200.
    gate._running = 1
    inference_queue.shutdown()
    try:
        r = _metrics_client().get("/metrics")
        assert r.status_code == 200
        assert "soloheaven_requests_total" in r.text
    finally:
        inference_queue._reset_for_tests()


# --------------------------------------------------------------------------
# X-Request-ID middleware.
# --------------------------------------------------------------------------

def _request_id_client() -> TestClient:
    from mlx_soloheaven.api.request_id import get_request_id

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/echo")
    async def echo():
        return {"request_id": get_request_id()}

    return TestClient(app, raise_server_exceptions=False)


def test_request_id_generated_when_absent():
    r = _request_id_client().get("/echo")
    assert r.status_code == 200
    rid = r.headers.get("x-request-id")
    assert rid and len(rid) == 32  # uuid4 hex
    # The handler saw the same id via the ContextVar.
    assert r.json()["request_id"] == rid


def test_request_id_echoes_safe_incoming():
    incoming = "abc-123_DEF.456"
    r = _request_id_client().get("/echo", headers={"X-Request-ID": incoming})
    assert r.headers.get("x-request-id") == incoming
    assert r.json()["request_id"] == incoming


def test_request_id_rejects_unsafe_incoming():
    # A header-injection / oversized payload is NOT echoed — a fresh id is used.
    unsafe = "bad id with spaces and \r\n injection"
    r = _request_id_client().get("/echo", headers={"X-Request-ID": unsafe})
    rid = r.headers.get("x-request-id")
    assert rid != unsafe
    assert rid and len(rid) == 32
    assert " " not in rid and "\n" not in rid


def test_request_id_rejects_overlong_incoming():
    r = _request_id_client().get("/echo", headers={"X-Request-ID": "a" * 500})
    rid = r.headers.get("x-request-id")
    assert rid and len(rid) == 32  # replaced by a generated id


# --------------------------------------------------------------------------
# Finding 3(a): /metrics render is LOCK-FREE while formatting — a stalled
# scrape must not block a concurrent recorder (so metrics can never sit in /
# extend the batch-C inference lease).
# --------------------------------------------------------------------------

def test_metrics_render_does_not_hold_lock_during_format(monkeypatch):
    metrics.observe_request(model="m", finish_reason="stop", generation_time=0.1)

    started = threading.Event()
    release = threading.Event()
    orig_render = metrics._REQUESTS.render

    def blocking_render(snap):
        # This is the FORMAT phase — it runs OUTSIDE the module lock (render
        # only holds the lock to snapshot), so stalling here must not block a
        # concurrent recorder.
        started.set()
        assert release.wait(2.0)
        return orig_render(snap)

    monkeypatch.setattr(metrics._REQUESTS, "render", blocking_render)

    render_done = threading.Event()

    def do_render():
        metrics.render()
        render_done.set()

    threading.Thread(target=do_render, daemon=True).start()
    assert started.wait(1.0), "render never reached the format phase"

    recorded = threading.Event()

    def do_record():
        metrics.observe_request(model="m2", finish_reason="stop")
        recorded.set()

    threading.Thread(target=do_record, daemon=True).start()
    # The recorder MUST proceed even though a render is stalled mid-format: the
    # lock is free (only briefly held for the snapshot).
    assert recorded.wait(1.0), "a stalled /metrics render blocked a recorder"

    release.set()
    assert render_done.wait(2.0)


# --------------------------------------------------------------------------
# Finding 3(b): the per-request metric is recorded strictly AFTER the
# inference-gate lease is released (never inside the lease window).
# --------------------------------------------------------------------------

def test_deferred_metric_idempotent_and_noop_when_unarmed(monkeypatch):
    calls = []
    monkeypatch.setattr(metrics, "observe_request", lambda **k: calls.append(k))
    d = metrics.DeferredRequestMetric()
    assert not d.armed
    d.fire()  # never armed -> no record (e.g. a disconnect before terminal)
    assert calls == []
    d.arm(model="x", finish_reason="stop")
    assert d.armed
    d.fire()
    d.fire()  # exactly once
    assert len(calls) == 1 and calls[0]["model"] == "x"


def test_slot_streaming_records_metric_after_lease_release(monkeypatch):
    """The streaming body ARMS the metric while holding the lease; the record
    fires only AFTER SlotStreamingResponse releases the slot (batch-C
    invariant: metrics recording is never inside the lease window)."""
    from mlx_soloheaven import inference_queue
    from mlx_soloheaven.api.gate_stream import SlotStreamingResponse

    gate = inference_queue.InferenceGate()
    observed = []

    def spy(**kwargs):
        # At record time the slot MUST already be free (lease released).
        observed.append(("record", gate.running(), kwargs.get("model")))

    monkeypatch.setattr(metrics, "observe_request", spy)

    sink = metrics.DeferredRequestMetric()

    async def inner():
        yield "data: chunk\n\n"
        # Terminal frame: arm while STILL holding the lease.
        sink.arm(model="slotmodel", finish_reason="stop")

    async def run():
        lease = await gate.acquire()
        assert gate.running() == 1
        resp = SlotStreamingResponse(gate, lease, inner(), on_release=sink.fire)
        async for _chunk in resp.body_iterator:
            pass
        return resp

    asyncio.run(run())
    assert gate.running() == 0  # lease released
    # Recorded exactly once, and the slot was already free at record time.
    assert observed == [("record", 0, "slotmodel")], observed


# --------------------------------------------------------------------------
# Finding 3 (round 2): EVERY admitted request emits exactly ONE metric —
# armed-at-start, fired-at-teardown — on every terminal (normal / disconnect /
# error), still strictly AFTER the lease release. Disconnect => one CANCEL
# record (the old contract "disconnect => no record" is gone).
# --------------------------------------------------------------------------

def test_arm_early_fire_records_baseline_without_update(monkeypatch):
    """A request that dies before any explicit terminal record still fires ONE
    metric from the arm-at-start baseline (finding 3)."""
    calls = []
    monkeypatch.setattr(metrics, "observe_request", lambda **k: calls.append(k))
    d = metrics.DeferredRequestMetric()
    d.arm(model="m", queue_wait=0.0, finish_reason="cancel", error_code="cancel")
    d.fire()
    d.fire()  # idempotent
    assert len(calls) == 1
    assert calls[0]["finish_reason"] == "cancel"
    assert calls[0]["error_code"] == "cancel"


def test_update_merges_onto_baseline(monkeypatch):
    calls = []
    monkeypatch.setattr(metrics, "observe_request", lambda **k: calls.append(k))
    d = metrics.DeferredRequestMetric()
    d.arm(model="m", queue_wait=0.3, finish_reason="cancel", error_code="cancel")
    d.update(finish_reason="stop", error_code="none", ttft=0.5)
    d.fire()
    assert len(calls) == 1
    # Baseline model/queue_wait survive; terminal fields override.
    assert calls[0]["model"] == "m" and calls[0]["queue_wait"] == 0.3
    assert calls[0]["finish_reason"] == "stop" and calls[0]["error_code"] == "none"
    assert calls[0]["ttft"] == 0.5


class _HangingEngine:
    """One real token, then the stream hangs — a client that abandons mid-stream
    exercises the disconnect (GeneratorExit) terminal."""

    model_family = "chatml"
    model_id = "mD"

    class _Cfg:
        enable_thinking = False

    cfg = _Cfg()

    def __init__(self, first_frame):
        self._first = first_frame

    async def generate_stream_batches_async(self, *a, **k):
        yield [self._first]
        await asyncio.Event().wait()  # hang; the client disconnects
        yield  # pragma: no cover — unreachable

    def update_session_messages(self, *a, **k):
        pass


def test_openai_disconnect_records_one_cancel_metric(monkeypatch):
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    recorded = []
    monkeypatch.setattr(metrics, "observe_request", lambda **k: recorded.append(k))

    sink = metrics.DeferredRequestMetric()
    # arm-at-start baseline exactly as chat_completions does for a stream.
    sink.arm(model="mD", queue_wait=0.0, finish_reason="cancel", error_code="cancel")
    req = ChatCompletionRequest(
        model="mD", messages=[ChatMessage(role="user", content="hi")],
        stream=True, thinking=False,
    )
    engine = _HangingEngine(_TFrame(text="partial", token=8, token_produced=True))

    async def run():
        body = openai_compat._stream_completion_body(req, engine, metric_sink=sink)
        await body.__anext__()  # role chunk
        await body.__anext__()  # first content chunk
        await body.aclose()     # client disconnect -> GeneratorExit -> cancel
        sink.fire()             # SlotStreamingResponse fires this AFTER release

    asyncio.run(run())
    assert len(recorded) == 1
    assert recorded[0]["finish_reason"] == "cancel"
    assert recorded[0]["error_code"] == "cancel"


def test_chat_disconnect_records_one_cancel_metric(monkeypatch):
    from mlx_soloheaven.api import chat as chat_mod

    class _StubDB:
        async def get_session_total_tokens(self, *a, **k):
            return 0

        async def update_session_tokens(self, *a, **k):
            pass

    monkeypatch.setattr(chat_mod, "db", _StubDB())

    recorded = []
    monkeypatch.setattr(metrics, "observe_request", lambda **k: recorded.append(k))

    sink = metrics.DeferredRequestMetric()
    sink.arm(model="mD", queue_wait=0.0, finish_reason="cancel", error_code="cancel")
    # A keepalive-first stream that then hangs: disconnect BEFORE any real token
    # (accumulated_text stays empty -> the persist block is skipped).
    engine = _HangingEngine(_TFrame(text="", token=0))  # keepalive frame

    async def run():
        body = chat_mod._stream_chat_body(
            "s", [{"role": "user", "content": "hi"}], engine, metric_sink=sink
        )
        await body.__anext__()  # start event
        await body.__anext__()  # keepalive comment
        await body.aclose()     # disconnect -> swallowed -> cancel update
        sink.fire()

    asyncio.run(run())
    assert len(recorded) == 1
    assert recorded[0]["finish_reason"] == "cancel"
    assert recorded[0]["error_code"] == "cancel"


class _RestartingEngine:
    """Yields one real token frame, then the child worker dies mid-stream
    (EngineRestartingError) — exercises the streaming ERROR terminal."""

    model_family = "chatml"
    model_id = "mR"

    class _Cfg:
        enable_thinking = False

    cfg = _Cfg()

    async def generate_stream_batches_async(self, *a, **k):
        from mlx_soloheaven.engine.process_client import EngineRestartingError

        yield [_TFrame(text="partial", token=8, token_produced=True)]
        raise EngineRestartingError("child worker restarting")

    def update_session_messages(self, *a, **k):
        pass


def test_chat_stream_engine_restart_records_one_error_metric(monkeypatch):
    """P2: a mid-stream engine restart on the WEB-CHAT streaming path records
    EXACTLY ONE metric with finish_reason=error / error_code=engine_not_ready
    (NOT the early-armed cancel/cancel baseline), fired strictly AFTER the
    inference-gate lease is released (batch-C invariant). Mirrors the OpenAI
    streaming path. Fails pre-fix (records cancel/cancel)."""
    from mlx_soloheaven import inference_queue
    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.api.gate_stream import SlotStreamingResponse

    gate = inference_queue.InferenceGate()
    observed = []

    def spy(**kwargs):
        # At record time the slot MUST already be free (lease released).
        observed.append((gate.running(), kwargs))

    monkeypatch.setattr(metrics, "observe_request", spy)

    sink = metrics.DeferredRequestMetric()
    # arm-at-start baseline exactly as the chat endpoint does for a stream.
    sink.arm(model="mR", queue_wait=0.0, finish_reason="cancel", error_code="cancel")
    engine = _RestartingEngine()

    async def run():
        lease = await gate.acquire()
        assert gate.running() == 1
        inner = chat_mod._stream_chat(
            "s", [{"role": "user", "content": "hi"}], engine, metric_sink=sink,
        )
        # SlotStreamingResponse owns the lease and fires the metric on release.
        resp = SlotStreamingResponse(gate, lease, inner, on_release=sink.fire)
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    assert gate.running() == 0  # lease released
    # The wrapper emitted an in-band error event (mirrors the OpenAI path).
    assert any('"type": "error"' in c for c in chunks), chunks
    # Recorded EXACTLY once, after the lease release (slot free), as an ERROR.
    assert len(observed) == 1, observed
    running_at_record, kwargs = observed[0]
    assert running_at_record == 0  # fired strictly after gate.release
    assert kwargs["finish_reason"] == "error"
    assert kwargs["error_code"] == "engine_not_ready"


# --------------------------------------------------------------------------
# Finding 6: an UNCAUGHT 500 still carries X-Request-ID (canonical envelope,
# no exception-text leak).
# --------------------------------------------------------------------------

def _uncaught_500_app() -> FastAPI:
    from mlx_soloheaven.server import register_uncaught_error_handler

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    register_uncaught_error_handler(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom-INTERNAL-DETAIL")

    return app


def test_uncaught_500_carries_incoming_request_id():
    client = TestClient(_uncaught_500_app(), raise_server_exceptions=False)
    rid = "known-rid-123"
    r = client.get("/boom", headers={"X-Request-ID": rid})
    assert r.status_code == 500
    assert r.headers.get("x-request-id") == rid  # matches the request's id
    body = r.json()
    assert body["error"]["code"] == "server_error"  # canonical batch-B envelope
    assert body["error"]["type"] == "server_error"
    assert "kaboom" not in r.text  # exception text never leaks into the body


def test_uncaught_500_carries_generated_request_id():
    client = TestClient(_uncaught_500_app(), raise_server_exceptions=False)
    r = client.get("/boom")
    assert r.status_code == 500
    rid = r.headers.get("x-request-id")
    assert rid and len(rid) == 32  # a fresh uuid4 hex was generated + echoed


# --------------------------------------------------------------------------
# Finding 7: TTFT is anchored at generation/body start and stops at the FIRST
# token frame (INCLUDING an empty-detok frame) — measured the SAME way in the
# OpenAI streaming and web-chat paths — and the histogram HELP text matches.
# --------------------------------------------------------------------------

@dataclass
class _TFrame:
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None
    # Finding 4: explicit "a real token was produced this frame" signal — True on
    # genuine token frames (incl. empty-detok), False on keepalive/status/finish.
    token_produced: bool = False


def test_openai_ttft_anchored_on_first_token_frame_including_empty(monkeypatch):
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    clock = {"t": 0.0}
    monkeypatch.setattr(openai_compat.time, "perf_counter", lambda: clock["t"])

    class _Engine:
        model_family = "chatml"
        model_id = "m7"

        class _Cfg:
            enable_thinking = False

        cfg = _Cfg()

        async def generate_stream_batches_async(self, *a, **k):
            clock["t"] = 5.0
            # EMPTY-detok FIRST token — token_produced=True anchors TTFT here.
            yield [_TFrame(text="", token=7, token_produced=True)]
            clock["t"] = 9.0
            yield [_TFrame(text="hello", token=8, token_produced=True)]
            clock["t"] = 12.0
            yield [_TFrame(finish_reason="stop", prompt_tokens=3, completion_tokens=2)]

        def update_session_messages(self, *a, **k):
            pass

    sink = metrics.DeferredRequestMetric()
    req = ChatCompletionRequest(
        model="m7", messages=[ChatMessage(role="user", content="hi")],
        stream=True, thinking=False,
    )

    async def run():
        async for _ in openai_compat._stream_completion_body(
            req, _Engine(), metric_sink=sink
        ):
            pass

    asyncio.run(run())
    assert sink.armed
    # 5.0 == first TOKEN frame (empty text) minus body start (0.0) — NOT 9.0
    # (the first visible-text batch), which is what the pre-fix anchor measured.
    assert sink._kwargs["ttft"] == 5.0


def test_openai_ttft_ignores_keepalive_before_first_token(monkeypatch):
    """Finding 4: a KEEPALIVE frame (token_produced=False, empty text emitted
    during long prefill) must NOT anchor TTFT — only the first REAL token frame
    does, even when its detok text is empty."""
    from mlx_soloheaven.api import openai_compat
    from mlx_soloheaven.api.schemas import ChatCompletionRequest, ChatMessage

    clock = {"t": 0.0}
    monkeypatch.setattr(openai_compat.time, "perf_counter", lambda: clock["t"])

    class _Engine:
        model_family = "chatml"
        model_id = "m7"

        class _Cfg:
            enable_thinking = False

        cfg = _Cfg()

        async def generate_stream_batches_async(self, *a, **k):
            clock["t"] = 1.0
            yield [_TFrame(text="", token=0)]        # KEEPALIVE (no token)
            clock["t"] = 5.0
            # First REAL token — empty detok but token_produced=True.
            yield [_TFrame(text="", token=0, token_produced=True)]
            clock["t"] = 9.0
            yield [_TFrame(text="hi", token=8, token_produced=True)]
            clock["t"] = 12.0
            yield [_TFrame(finish_reason="stop", prompt_tokens=3, completion_tokens=2)]

        def update_session_messages(self, *a, **k):
            pass

    sink = metrics.DeferredRequestMetric()
    req = ChatCompletionRequest(
        model="m7", messages=[ChatMessage(role="user", content="hi")],
        stream=True, thinking=False,
    )

    async def run():
        async for _ in openai_compat._stream_completion_body(
            req, _Engine(), metric_sink=sink
        ):
            pass

    asyncio.run(run())
    assert sink.armed
    # TTFT == 5.0 (first real token), NOT 1.0 (the keepalive).
    assert sink._kwargs["ttft"] == 5.0


def test_chat_ttft_uses_same_body_start_anchor(monkeypatch):
    from mlx_soloheaven.api import chat as chat_mod

    class _StubDB:
        async def add_message(self, *a, **k):
            return {"id": "x"}

        async def get_session_total_tokens(self, *a, **k):
            return 0

        async def update_session_tokens(self, *a, **k):
            pass

        async def get_session(self, *a, **k):
            return None

        async def update_session(self, *a, **k):
            pass

    monkeypatch.setattr(chat_mod, "db", _StubDB())
    clock = {"t": 0.0}
    monkeypatch.setattr(chat_mod.time, "perf_counter", lambda: clock["t"])

    class _Engine:
        model_family = "chatml"
        model_id = "m7"

        class _Cfg:
            enable_thinking = False

        cfg = _Cfg()

        async def generate_stream_batches_async(self, *a, **k):
            clock["t"] = 3.0
            yield [_TFrame(status="generating")]     # engine-lock grant marker
            clock["t"] = 4.0
            yield [_TFrame(text="", token=0)]        # KEEPALIVE (finding 4)
            clock["t"] = 5.0
            # EMPTY-detok FIRST token — token_produced=True anchors here.
            yield [_TFrame(text="", token=0, token_produced=True)]
            clock["t"] = 9.0
            yield [_TFrame(text="hello", token=8, token_produced=True)]
            clock["t"] = 12.0
            yield [_TFrame(finish_reason="stop", prompt_tokens=3, completion_tokens=2)]

        def update_session_messages(self, *a, **k):
            pass

    sink = metrics.DeferredRequestMetric()

    async def run():
        async for _ in chat_mod._stream_chat_body(
            "s", [{"role": "user", "content": "hi"}], _Engine(), metric_sink=sink
        ):
            pass

    asyncio.run(run())
    assert sink.armed
    # BODY-start anchor (5.0 - 0.0), SAME as the OpenAI path — NOT the engine-
    # lock-grant anchor the pre-fix code used (5.0 - 3.0 = 2.0), and NOT the
    # keepalive at t=4.0 (finding 4).
    assert sink._kwargs["ttft"] == 5.0


def test_generation_and_ttft_help_text_matches_implementation():
    text = metrics.render()
    # generation_seconds measures body/generation start -> completion, NOT
    # first-token -> completion (the old, wrong HELP wording).
    assert (
        "# HELP soloheaven_generation_seconds Whole-generation wall time "
        "(generation/body start through completion)."
    ) in text
    assert "first token through completion" not in text  # old wording gone
    # ttft_seconds is anchored at body start and stops at the first token frame
    # (including empty-detok).
    ttft_help = next(
        ln for ln in text.splitlines()
        if ln.startswith("# HELP soloheaven_ttft_seconds")
    )
    assert "generation/body start" in ttft_help
    assert "first token frame" in ttft_help
