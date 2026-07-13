"""GPU-keepalive coverage for PROCESS mode (worker poll-timeout branch).

MEASURED GAP being fixed: with --gpu-keepalive, thread mode runs a background
keepalive thread, but main_thread/process mode skipped it entirely ("no
background MLX access permitted") — so in process mode the Metal GPU
downclocked after ~40s idle and the next request's TTFT roughly doubled
(2k-token prompt: 727ms warm -> 1,422ms idle). The child's main loop idles in
``cmd_conn.poll(...)`` between requests — that IS the main thread — so the
poll-timeout branch (where the idle flush already runs) now invokes the same
core GPU op as the thread-mode keepalive loop (MLXEngine._gpu_keepalive_ping),
rate-limited to the thread-mode interval (MLXEngine.GPU_KEEPALIVE_INTERVAL).

These tests are HERMETIC (no model, no real pipes, no real signals — same
conventions as tests/test_shutdown_flush.py):
  - the keepalive touch runs from the poll-timeout branch when enabled + the
    lock is free + the interval elapsed,
  - it is NOT invoked when --gpu-keepalive is off, when the engine lock is
    held (non-blocking acquire — NEVER waits behind a generation), when the
    interval has not elapsed, or when the shutdown flag is set (sentinel
    unwinding),
  - a raising ping is fail-closed: logged ONCE (later failures suppressed),
    the lock is released, and the worker loop keeps running to a clean
    shutdown flush,
  - the periodic idle flush still runs alongside the keepalive (flush first —
    the touch must never starve or delay it),
  - thread mode's contract holds: MLXEngine exposes GPU_KEEPALIVE_INTERVAL
    and _gpu_keepalive_ping (the shared core op the worker calls).
"""

from __future__ import annotations

import logging
import signal
import threading
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine import process_worker
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.process_worker import (
    _gpu_keepalive_touch,
    worker_main,
)


# --- helpers ----------------------------------------------------------------


class FakeConn:
    """Minimal Pipe-connection stand-in: scripted recv frames + send log."""

    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.sent = []
        self.poll_timeouts: list = []

    def poll(self, timeout=None):
        self.poll_timeouts.append(timeout)
        return True

    def recv(self):
        if not self.frames:
            raise EOFError
        return self.frames.pop(0)

    def send(self, obj):
        self.sent.append(obj)


class TimeoutThenShutdownConn(FakeConn):
    """cmd pipe stand-in: poll() times out ``n_timeouts`` times (driving the
    worker's poll-timeout branch), then reports readable and recv() serves
    the scripted frames (default: a graceful 'shutdown' op)."""

    def __init__(self, n_timeouts, frames=None):
        super().__init__(frames if frames is not None else [{"op": "shutdown"}])
        self.n_timeouts = n_timeouts

    def poll(self, timeout=None):
        self.poll_timeouts.append(timeout)
        if self.n_timeouts > 0:
            self.n_timeouts -= 1
            return False
        return True


class EOFConn(FakeConn):
    """ctrl pipe stand-in — immediate EOF so the ctrl thread exits."""

    def recv(self):
        raise EOFError


class KeepaliveFakeEngine:
    """Engine stand-in — records keepalive pings, idle flushes, shutdown
    flushes. ``GPU_KEEPALIVE_INTERVAL`` mirrors the MLXEngine class constant
    the worker reads for its rate limit."""

    GPU_KEEPALIVE_INTERVAL = 0.0  # overridden per test via the factory

    def __init__(self, cfg, execution_mode="worker"):
        self.cfg = cfg
        self.execution_mode = execution_mode
        self.model_id = "fake-model"
        self.model_family = "chatml"
        self.pings = []
        self.idle_flushes = []
        self.flush_calls = []
        self._lock = threading.Lock()

    def load_model(self):
        pass

    def _gpu_keepalive_ping(self):
        self.pings.append(1)

    def _flush_dirty_sessions(self):
        self.idle_flushes.append(1)

    def _flush_all_on_shutdown(self):
        self.flush_calls.append("flush")


@pytest.fixture
def restore_signals():
    """worker_main installs real SIGTERM/SIGINT handlers on the pytest main
    thread — restore the originals so the test process keeps its behavior."""
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)


@pytest.fixture(autouse=True)
def reset_keepalive_error_flag(monkeypatch):
    """The 'log once' latch is module-global — reset per test for order
    independence."""
    monkeypatch.setattr(process_worker, "_keepalive_error_logged", False)


def _run_worker(cmd_conn, monkeypatch, *, gpu_keepalive, interval=0.0,
                engine_cls=KeepaliveFakeEngine):
    """Drive worker_main with a KeepaliveFakeEngine and fake pipes; returns
    the engine. Synchronous — worker_main returns when the loop exits."""
    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = engine_cls(cfg, execution_mode=execution_mode)
        eng.GPU_KEEPALIVE_INTERVAL = interval
        created.append(eng)
        return eng

    # worker_main does `from ...mlx_engine import MLXEngine` at call time, so
    # patching the symbol on the mlx_engine module intercepts construction.
    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    worker_main(
        {"model_path": "/tmp/x", "verbose": False, "gpu_keepalive": gpu_keepalive},
        cmd_conn,
        FakeConn(),
        EOFConn(),
    )
    assert len(created) == 1
    return created[0]


# --- loop-level: poll-timeout branch ------------------------------------------


def test_keepalive_invoked_on_poll_timeout_when_enabled(
    monkeypatch, restore_signals
):
    """Enabled + lock free + interval elapsed: every timeout poll pings."""
    conn = TimeoutThenShutdownConn(n_timeouts=3)
    engine = _run_worker(conn, monkeypatch, gpu_keepalive=True, interval=0.0)

    assert engine.pings == [1, 1, 1]  # one touch per timed-out poll
    # Poll granularity matched the keepalive interval (min with the flush
    # poll), so the touch cadence equals thread mode's.
    assert all(t == min(process_worker.IDLE_FLUSH_POLL_S, 0.0)
               for t in conn.poll_timeouts)
    # Shutdown path unaffected — the final flush still ran exactly once.
    assert engine.flush_calls == ["flush"]


def test_keepalive_not_invoked_when_disabled(monkeypatch, restore_signals):
    conn = TimeoutThenShutdownConn(n_timeouts=3)
    engine = _run_worker(conn, monkeypatch, gpu_keepalive=False, interval=0.0)

    assert engine.pings == []
    # Disabled: the loop keeps the plain idle-flush poll granularity.
    assert all(t == process_worker.IDLE_FLUSH_POLL_S for t in conn.poll_timeouts)
    assert engine.flush_calls == ["flush"]


def test_keepalive_not_invoked_when_interval_not_elapsed(
    monkeypatch, restore_signals
):
    """last_keepalive starts at loop entry; a huge interval means the rate
    limit never elapses across the timeout polls — no ping."""
    conn = TimeoutThenShutdownConn(n_timeouts=3)
    engine = _run_worker(conn, monkeypatch, gpu_keepalive=True, interval=3600.0)

    assert engine.pings == []
    # Poll stays bounded by the flush poll even with a huge interval.
    assert all(t == process_worker.IDLE_FLUSH_POLL_S for t in conn.poll_timeouts)
    assert engine.flush_calls == ["flush"]


def test_keepalive_exception_does_not_kill_loop_and_logs_once(
    monkeypatch, restore_signals, caplog
):
    """Fail-closed: a ping that raises on EVERY touch is logged exactly once,
    the loop keeps polling, and the graceful shutdown flush still runs."""

    class ExplodingPingEngine(KeepaliveFakeEngine):
        def _gpu_keepalive_ping(self):
            self.pings.append(1)
            raise RuntimeError("metal on fire")

    with caplog.at_level(
        logging.ERROR, logger="mlx_soloheaven.engine.process_worker"
    ):
        engine = _run_worker(
            TimeoutThenShutdownConn(n_timeouts=3),
            monkeypatch,
            gpu_keepalive=True,
            interval=0.0,
            engine_cls=ExplodingPingEngine,
        )

    assert engine.pings == [1, 1, 1]  # kept retrying at the normal cadence
    assert engine.flush_calls == ["flush"]  # loop survived to the flush
    assert not engine._lock.locked()  # lock released on the error path
    failures = [r for r in caplog.records
                if "GPU keepalive touch failed" in r.getMessage()]
    assert len(failures) == 1  # logged once, later failures suppressed


def test_idle_flush_still_runs_alongside_keepalive(monkeypatch, restore_signals):
    """The keepalive must not starve or delay the periodic idle flush: with
    the flush threshold at 0 the SAME timeout poll runs both (flush first)."""
    monkeypatch.setattr(process_worker, "IDLE_FLUSH_AFTER_S", 0.0)
    engine = _run_worker(
        TimeoutThenShutdownConn(n_timeouts=2),
        monkeypatch,
        gpu_keepalive=True,
        interval=0.0,
    )

    assert engine.idle_flushes == [1, 1]  # one per timed-out poll
    assert engine.pings == [1, 1]
    assert engine.flush_calls == ["flush"]


# --- helper-level: _gpu_keepalive_touch ----------------------------------------


def _touch_engine(pings: list):
    return SimpleNamespace(
        _lock=threading.Lock(),
        _gpu_keepalive_ping=lambda: pings.append(1),
    )


def test_touch_pings_under_free_lock():
    pings: list = []
    eng = _touch_engine(pings)
    assert _gpu_keepalive_touch(eng) is True
    assert pings == [1]
    assert not eng._lock.locked()  # released afterwards


def test_touch_skips_when_lock_held():
    """Non-blocking acquire: a held lock (in-flight generation) means skip
    immediately — the touch must NEVER wait behind a generation."""
    pings: list = []
    eng = _touch_engine(pings)
    assert eng._lock.acquire(blocking=False)
    try:
        assert _gpu_keepalive_touch(eng) is False  # returns immediately
    finally:
        eng._lock.release()
    assert pings == []


def test_touch_skips_when_shutdown_flag_set():
    """Shutdown unwinding (sentinel flag set): no touch, lock untouched."""
    pings: list = []
    eng = _touch_engine(pings)
    flag = threading.Event()
    flag.set()
    assert _gpu_keepalive_touch(eng, flag) is False
    assert pings == []
    assert not eng._lock.locked()


def test_touch_runs_when_shutdown_flag_clear():
    pings: list = []
    eng = _touch_engine(pings)
    assert _gpu_keepalive_touch(eng, threading.Event()) is True
    assert pings == [1]


def test_touch_failclosed_releases_lock_and_reports_attempted():
    """A raising ping never propagates, releases the lock, and still counts
    as ATTEMPTED (True) so the caller's interval timer resets — a broken
    ping retries at the normal cadence, not in a tight loop."""
    eng = SimpleNamespace(
        _lock=threading.Lock(),
        _gpu_keepalive_ping=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _gpu_keepalive_touch(eng) is True  # must not raise
    assert not eng._lock.locked()
    assert process_worker._keepalive_error_logged is True

    # Second failure: still True, still silent (once-latch already set).
    assert _gpu_keepalive_touch(eng) is True


# --- engine-side contract the worker relies on ---------------------------------


def test_engine_exposes_keepalive_interval_and_ping():
    """The worker reads MLXEngine.GPU_KEEPALIVE_INTERVAL (thread mode's
    constant) and calls _gpu_keepalive_ping — the SAME core op the
    thread-mode keepalive loop runs. Guard against attribute drift, and run
    the real op once (a 32x32 matmul — trivially cheap, no model)."""
    assert isinstance(MLXEngine.GPU_KEEPALIVE_INTERVAL, float)
    assert MLXEngine.GPU_KEEPALIVE_INTERVAL > 0
    eng = MLXEngine.__new__(MLXEngine)  # no __init__ — ping uses no state
    eng._gpu_keepalive_ping()  # must not raise
