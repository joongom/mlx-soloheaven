"""U14 — synchronous proxy RPCs must not freeze the event loop / SSE.

The parent-side proxy RPC blocks its calling thread and the child services
RPCs only BETWEEN generations — so one admin/metadata call during a long
generation used to block the FastAPI event loop AND every in-flight SSE
stream. Two halves are covered here:

PARENT (mandatory): every async endpoint that calls a synchronous engine
method now runs it through ``asyncio.to_thread`` — a fake engine that BLOCKS
inside the call must not stop a concurrently-scheduled asyncio task from
ticking.

BOUNDED READS: read-only RPCs (list/stats/overview/get) get a bounded wait —
``EngineBusyError`` on expiry instead of hanging for the generation's full
duration — and every caller DEGRADES: admin cache overview and the web cache
stats answer a ``busy`` marker, /v1/sessions marks the busy model, session
GET answers 503-shaped busy, the admin models overview keeps rendering with
sessions=None.

HERMETIC — no real engine/child/pipes (conventions from
tests/test_worker_liveness.py: asyncio.run drivers + SimpleNamespace stubs).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine import process_protocol as proto
from mlx_soloheaven.engine.process_client import (
    EngineProcessProxy,
    _RpcError,
)
from mlx_soloheaven.engine.types import EngineBusyError


# --- helpers ------------------------------------------------------------------


def _proxy_shell() -> EngineProcessProxy:
    """Real __init__ (cheap), once-ready, with a cmd pipe that swallows sends
    and a child that never answers — the RPC wait must expire on its own."""
    proxy = EngineProcessProxy(SimpleNamespace())
    proxy._ever_ready = True
    proxy._cmd_parent = SimpleNamespace(send=lambda obj: None)
    return proxy


def _stub_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model_path="/tmp/x",
        default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
        default_top_k=0, default_repetition_penalty=1.0,
        default_max_tokens=100,
        enable_thinking=False, thinking_budget=0,
        think_end_token=-1, think_start_token=-1,
        memory_budget_gb=1, disk_budget_gb=1,
    )


async def _run_with_ticker(coro):
    """Run ``coro`` while a 5ms ticker task increments a counter — returns
    (result, ticks). A blocked event loop yields ~0 ticks."""
    ticks = {"n": 0}
    stop = asyncio.Event()

    async def _ticker():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    t = asyncio.ensure_future(_ticker())
    try:
        result = await coro
    finally:
        stop.set()
        await t
    return result, ticks["n"]


# --- proxy: bounded read RPCs raise EngineBusyError -----------------------------


def test_rpc_read_timeout_raises_engine_busy(monkeypatch):
    """A read-only RPC that the (busy) child never answers expires within
    its bounded wait and raises EngineBusyError — not the generic 600s
    _RpcError path."""
    proxy = _proxy_shell()
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.15)

    t0 = time.monotonic()
    with pytest.raises(EngineBusyError):
        proxy.list_sessions()
    assert time.monotonic() - t0 < 2.0  # bounded, sliced wait honors sub-1s

    # RPC slot cleaned up (no leak).
    assert proxy._rpc_results == {}


@pytest.mark.parametrize("method", [
    "session_stats", "base_cache_stats", "cache_overview",
])
def test_read_methods_use_bounded_busy_wait(monkeypatch, method):
    proxy = _proxy_shell()
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)
    with pytest.raises(EngineBusyError):
        getattr(proxy, method)()


def test_get_session_read_is_bounded(monkeypatch):
    proxy = _proxy_shell()
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)
    with pytest.raises(EngineBusyError):
        proxy.get_session("s1")


def test_default_rpc_timeout_still_rpc_error():
    """Non-read RPCs keep the generic timeout error (a busy child is not the
    right diagnosis for e.g. a wedged delete)."""
    proxy = _proxy_shell()
    with pytest.raises(_RpcError):
        proxy._rpc("delete_session", "s1", timeout=0.15)


def test_cache_manager_shim_propagates_busy(monkeypatch):
    """chat.py's /api/cache/stats reads engine.cache_manager.stats() — the
    proxy shim forwards to the bounded cache_overview read, so the busy
    signal reaches that caller too."""
    proxy = _proxy_shell()
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)
    with pytest.raises(EngineBusyError):
        proxy.cache_manager.stats()


# --- API endpoints: blocking engine call does NOT block the event loop ----------


def test_admin_models_overview_nonblocking_event_loop():
    from mlx_soloheaven.api import admin

    class BlockingEngine:
        model_id = "m"
        cfg = _stub_cfg()

        def session_stats(self):
            time.sleep(0.3)  # blocking engine RPC (generation in flight)
            return {"active_sessions": 3}

    eng = BlockingEngine()
    old_engines, old_default = admin._engines, admin._default_engine
    try:
        admin.set_engines({"m": eng}, eng)
        out, ticks = asyncio.run(_run_with_ticker(admin.models_overview()))
    finally:
        admin.set_engines(old_engines, old_default)

    assert out["models"][0]["sessions"] == 3
    # The event loop kept scheduling during the 0.3s engine block.
    assert ticks >= 10


def test_openai_sessions_list_nonblocking_event_loop():
    from mlx_soloheaven.api import openai_compat

    class BlockingEngine:
        def list_sessions(self):
            time.sleep(0.3)
            return [{"session_id": "s1"}]

        def base_cache_stats(self):
            return []

    eng = BlockingEngine()
    old_engines, old_default = openai_compat._engines, openai_compat._default_engine
    try:
        openai_compat.set_engines({"m": eng}, eng)
        out, ticks = asyncio.run(_run_with_ticker(openai_compat.list_sessions()))
    finally:
        openai_compat.set_engines(old_engines, old_default)

    assert out["sessions"]["m"] == [{"session_id": "s1"}]
    assert ticks >= 10


def test_openai_sync_completion_nonblocking_event_loop(monkeypatch):
    """The non-streaming /v1/chat/completions path blocks for the WHOLE
    generation — it must run off the event loop."""
    from mlx_soloheaven.api import openai_compat

    sentinel = object()

    def blocking_sync_completion(request, engine):
        time.sleep(0.3)
        return sentinel

    monkeypatch.setattr(
        openai_compat, "_sync_completion", blocking_sync_completion
    )

    class StubEngine:
        model_id = "m"
        model_family = "chatml"
        cfg = SimpleNamespace(enable_thinking=False)

        def ensure_available(self):
            pass

    eng = StubEngine()
    old_engines, old_default = openai_compat._engines, openai_compat._default_engine
    try:
        openai_compat.set_engines({"m": eng}, eng)
        request = SimpleNamespace(
            model="m", stream=False, user=None, thinking=None,
            max_tokens=8, max_completion_tokens=None,
            response_format=None, stop=None,
            # U20: the endpoint now validates sampling params up front — the
            # stub must carry them (all unset) like a real request would.
            temperature=None, top_p=None, min_p=None, top_k=None,
            repetition_penalty=None, frequency_penalty=None,
            presence_penalty=None,
            messages=[SimpleNamespace(role="user", content="hi")],
        )
        out, ticks = asyncio.run(
            _run_with_ticker(openai_compat.chat_completions(request))
        )
    finally:
        openai_compat.set_engines(old_engines, old_default)

    assert out is sentinel
    assert ticks >= 10


# --- API endpoints: EngineBusyError degrades instead of failing -----------------


def test_admin_cache_overview_busy_degrades():
    from mlx_soloheaven.api import admin

    class BusyEngine:
        def cache_overview(self):
            raise EngineBusyError("generation in flight")

    class OkEngine:
        def cache_overview(self):
            return {
                "model_id": "ok", "enable_thinking": False, "sessions": [],
                "session_count": 0, "base_caches": [], "cache_manager": {},
                "memory": {}, "disk_files": [], "disk_bytes": 0,
                "memory_bytes": 0,
            }

    old_engines, old_default = admin._engines, admin._default_engine
    try:
        admin.set_engines({"busy": BusyEngine(), "ok": OkEngine()}, OkEngine())
        out = asyncio.run(admin.cache_overview())
    finally:
        admin.set_engines(old_engines, old_default)

    # Busy engine degrades to a marker; the healthy one still renders fully.
    assert out["engines"]["busy"] == {"busy": True}
    assert out["engines"]["ok"]["model_id"] == "ok"


def test_admin_models_overview_busy_degrades_to_none():
    from mlx_soloheaven.api import admin

    class BusyEngine:
        model_id = "m"
        cfg = _stub_cfg()

        def session_stats(self):
            raise EngineBusyError("generation in flight")

    eng = BusyEngine()
    old_engines, old_default = admin._engines, admin._default_engine
    try:
        admin.set_engines({"m": eng}, eng)
        out = asyncio.run(admin.models_overview())
    finally:
        admin.set_engines(old_engines, old_default)

    assert out["models"][0]["sessions"] is None  # page renders regardless


def test_openai_sessions_list_busy_marks_model():
    from mlx_soloheaven.api import openai_compat

    class BusyEngine:
        def list_sessions(self):
            raise EngineBusyError("generation in flight")

        def base_cache_stats(self):  # pragma: no cover — list raises first
            return []

    eng = BusyEngine()
    old_engines, old_default = openai_compat._engines, openai_compat._default_engine
    try:
        openai_compat.set_engines({"m": eng}, eng)
        out = asyncio.run(openai_compat.list_sessions())
    finally:
        openai_compat.set_engines(old_engines, old_default)

    assert out["sessions"]["m"] == {"busy": True}
    assert out["base_caches"]["m"] == {"busy": True}


def test_openai_get_session_busy_503():
    from mlx_soloheaven.api import openai_compat

    class BusyEngine:
        def get_session(self, sid):
            raise EngineBusyError("generation in flight")

    eng = BusyEngine()
    old_engines, old_default = openai_compat._engines, openai_compat._default_engine
    try:
        openai_compat.set_engines({"m": eng}, eng)
        resp = asyncio.run(openai_compat.get_session("s1"))
    finally:
        openai_compat.set_engines(old_engines, old_default)

    assert resp.status_code == 503


def test_chat_cache_stats_busy_degrades():
    from mlx_soloheaven.api import chat

    class BusyEngine:
        cache_manager = SimpleNamespace(
            stats=lambda: (_ for _ in ()).throw(
                EngineBusyError("generation in flight")
            )
        )

        def session_stats(self):  # pragma: no cover — stats raises first
            return {}

    old_engine = chat.engine
    try:
        chat.set_engine(BusyEngine())
        out = asyncio.run(chat.cache_stats())
    finally:
        chat.set_engine(old_engine)

    assert out == {"busy": True}


# --- F3: parent-side RPC timeout sends an rpc_cancel tombstone -------------------


def test_rpc_read_timeout_sends_rpc_cancel(monkeypatch):
    """A timed-out READ leaves its command queued child-side; the parent must
    tombstone it over the ctrl pipe so the worker skips execution (late
    replies already drop at the reader)."""
    proxy = _proxy_shell()
    ctrl_sent: list = []
    proxy._ctrl_parent = SimpleNamespace(send=lambda obj: ctrl_sent.append(obj))
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)

    with pytest.raises(EngineBusyError):
        proxy.session_stats()

    assert len(ctrl_sent) == 1
    assert ctrl_sent[0]["op"] == "rpc_cancel"
    assert isinstance(ctrl_sent[0]["id"], str) and ctrl_sent[0]["id"]


def test_generic_rpc_timeout_sends_rpc_cancel():
    """The generic (non-read) timeout path tombstones too — a wedged mutating
    RPC's queued command must not execute after its caller gave up."""
    proxy = _proxy_shell()
    ctrl_sent: list = []
    proxy._ctrl_parent = SimpleNamespace(send=lambda obj: ctrl_sent.append(obj))

    with pytest.raises(_RpcError):
        proxy._rpc("delete_session", "s1", timeout=0.1)

    assert [f["op"] for f in ctrl_sent] == ["rpc_cancel"]


def test_rpc_timeout_tombstone_send_failure_still_raises(monkeypatch):
    """A dead ctrl pipe must not mask the timeout error (best-effort send)."""
    proxy = _proxy_shell()

    def _boom(obj):
        raise OSError("ctrl pipe closed")

    proxy._ctrl_parent = SimpleNamespace(send=_boom)
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)
    with pytest.raises(EngineBusyError):
        proxy.list_sessions()


# --- F5 (round 2): mutating-RPC timeout is OUTCOME-UNKNOWN ------------------------


def test_mutating_rpc_timeout_warns_outcome_unknown(caplog):
    """A timed-out MUTATING RPC may still have executed child-side (the
    tombstone only stops a still-queued command) — the parent must log an
    explicit OUTCOME-UNKNOWN warning naming the method instead of implying
    the operation didn't run."""
    proxy = _proxy_shell()
    proxy._ctrl_parent = SimpleNamespace(send=lambda obj: None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(_RpcError) as exc_info:
            proxy._rpc("delete_session", "s1", timeout=0.1)

    assert "outcome unknown" in str(exc_info.value)
    warned = [r for r in caplog.records if "OUTCOME UNKNOWN" in r.getMessage()]
    assert len(warned) == 1
    assert "delete_session" in warned[0].getMessage()


def test_read_rpc_timeout_keeps_busy_semantics_no_unknown_warning(
    monkeypatch, caplog
):
    """Read-only RPCs keep the existing behavior: EngineBusyError, no
    outcome-unknown warning (a stale read is harmless — the tombstone
    guarantees it never runs)."""
    proxy = _proxy_shell()
    proxy._ctrl_parent = SimpleNamespace(send=lambda obj: None)
    monkeypatch.setattr(EngineProcessProxy, "RPC_READ_TIMEOUT_S", 0.1)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(EngineBusyError):
            proxy.list_sessions()

    assert not [
        r for r in caplog.records if "OUTCOME UNKNOWN" in r.getMessage()
    ]


def test_reader_logs_rpc_cancel_ack_outcomes(caplog):
    """rpc_cancel_ack frames from the worker route to the parent log — the
    definitive outcome for an RPC the parent already timed out on:
    "already_started"/"unknown" warn (it ran), "cancelled" is info."""

    class ScriptedResp:
        def __init__(self, frames):
            self.frames = list(frames)

        def recv(self):
            if not self.frames:
                raise EOFError
            return self.frames.pop(0)

    proxy = EngineProcessProxy(SimpleNamespace())
    conn = ScriptedResp([
        proto.make_rpc_cancel_ack("r1", "already_started"),
        proto.make_rpc_cancel_ack("r2", "cancelled"),
    ])
    with caplog.at_level(logging.INFO):
        # EOF before first ready -> the reader exits via the startup path
        # (no death handling) — safe to drive inline.
        proxy._reader_loop(resp_conn=conn)

    acks = [r for r in caplog.records if "rpc_cancel ack" in r.getMessage()]
    assert len(acks) == 2
    started = next(r for r in acks if "already_started" in r.getMessage())
    assert started.levelno == logging.WARNING
    cancelled = next(r for r in acks if "cancelled" in r.getMessage())
    assert cancelled.levelno == logging.INFO


# --- F2: dedicated executors — reads never starve behind long operations ---------


def test_run_read_gets_worker_while_long_pool_saturated():
    """Saturate the LONG pool (more blocking calls than workers); a bounded
    read must still START immediately on the reserved read pool — pre-fix
    (shared to_thread default executor) it queued behind the generations and
    its 10s bound never even began."""
    from mlx_soloheaven import executors

    release = threading.Event()

    def _long_op():
        release.wait(timeout=10.0)
        return "long-done"

    async def _drive():
        longs = [
            asyncio.ensure_future(executors.run_long(_long_op))
            for _ in range(executors.LONG_EXECUTOR_MAX_WORKERS + 2)
        ]
        await asyncio.sleep(0.05)  # let them occupy every long worker
        t0 = time.monotonic()
        out = await executors.run_read(lambda: "read-done")
        dt = time.monotonic() - t0
        release.set()
        long_results = await asyncio.gather(*longs)
        return out, dt, long_results

    from mlx_soloheaven import executors as _ex

    out, dt, long_results = asyncio.run(_drive())
    assert out == "read-done"
    assert dt < 2.0  # never waited for the long ops to finish
    assert long_results == ["long-done"] * (_ex.LONG_EXECUTOR_MAX_WORKERS + 2)


def test_run_long_queues_excess_instead_of_failing():
    """More long ops than workers: the extras QUEUE and all complete."""
    from mlx_soloheaven import executors

    n = executors.LONG_EXECUTOR_MAX_WORKERS * 2

    async def _drive():
        return await asyncio.gather(
            *[executors.run_long(lambda i=i: i) for i in range(n)]
        )

    assert sorted(asyncio.run(_drive())) == list(range(n))


def test_run_helpers_forward_args_and_kwargs():
    from mlx_soloheaven.executors import run_long, run_read

    def _fn(a, b, *, c=0):
        return (a, b, c)

    async def _drive():
        return (
            await run_long(_fn, 1, 2, c=3),
            await run_read(_fn, 4, 5, c=6),
        )

    assert asyncio.run(_drive()) == ((1, 2, 3), (4, 5, 6))


def test_health_wired_to_reserved_read_executor():
    """server.create_app's /health must run session_stats via run_read (the
    reserved pool), never asyncio.to_thread (the shared default pool a burst
    of long ops can exhaust). Source-level wiring check — same pattern as
    test_drafter_on_worker_load's create_app propagation assertion."""
    import inspect
    from mlx_soloheaven import server

    src = inspect.getsource(server.create_app)
    assert "async def health" in src
    health_src = src.split("async def health", 1)[1].split(
        "app.include_router", 1
    )[0]
    assert "run_read(engine.session_stats)" in health_src
    assert "asyncio.to_thread" not in health_src


@pytest.mark.parametrize("module_name", [
    "mlx_soloheaven.api.openai_compat",
    "mlx_soloheaven.api.chat",
    "mlx_soloheaven.api.admin",
    "mlx_soloheaven.api.compaction",
])
def test_api_modules_use_dedicated_executors(module_name):
    """No engine call site may fall back to the shared default executor —
    every off-loop engine call goes through run_long / run_read (F2)."""
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    assert "asyncio.to_thread(" not in inspect.getsource(mod)
