"""Shutdown-flush coverage for process mode (and main_thread mode).

CONFIRMED GAP being fixed: in process mode (the production default) sessions
were NEVER saved to disk on a normal server stop — the shutdown flush was
registered only inside ``_start_gpu_keepalive``, which is skipped in
main_thread/process mode, and the worker's 'shutdown' op just ``break``-ed.

These tests are HERMETIC (no model, no MLX work, no real signals delivered to
the pytest process):
  - worker_main flushes dirty sessions on the 'shutdown' op BEFORE exiting,
  - worker_main's SIGTERM/SIGINT handler (invoked directly, never via a real
    signal) raises the graceful sentinel, the loop catches it, and the flush
    still runs; a second handler invocation never re-raises (it must not
    abort an in-progress flush),
  - the final flush sets the shutdown flag FIRST, so a SIGTERM landing
    mid-flush on a NON-signal exit path ('shutdown' op / EOF / None) takes
    the 'already unwinding' path instead of raising the sentinel and
    aborting the flush after the dirty set was drained,
  - MLXEngine._flush_all_on_shutdown is idempotent and lock-guarded (bounded
    acquire — skips + re-marks dirty instead of deadlocking),
  - MLXEngine._register_shutdown_flush registers atexit + SIGINT/SIGTERM
    WITHOUT starting the keepalive thread (main_thread-mode contract), is
    idempotent at the registration level, CHAINS a previously-installed
    callable handler (Uvicorn owns process lifetime in server mode — our
    handler only prepends a flush), preserves the SIG_DFL re-delivery dance,
    respects SIG_IGN, and its atexit backstop RETRIES (re-reads the live
    dirty set) anything a timed-out lock acquire re-marked dirty,
  - the worker's periodic _idle_flush flushes under a free lock and bails
    (never blocks) when the lock is held,
  - EngineProcessProxy.close() sends the 'shutdown' op, waits bounded, and is
    idempotent.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine import process_worker
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.process_client import EngineProcessProxy
from mlx_soloheaven.engine.process_worker import (
    _GracefulShutdown,
    _idle_flush,
    worker_main,
)


# --- helpers ----------------------------------------------------------------


class FakeConn:
    """Minimal Pipe-connection stand-in: scripted recv frames + send log.

    poll() always reports readable — REAL pipe semantics: a pending frame OR
    a pending EOF both make poll() return True (recv then yields the frame or
    raises EOFError). Returning False on empty would busy-spin the worker
    loop forever in the EOF tests."""

    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.sent = []

    def poll(self, timeout=None):
        return True

    def recv(self):
        if not self.frames:
            raise EOFError
        return self.frames.pop(0)

    def send(self, obj):
        self.sent.append(obj)


class EOFConn(FakeConn):
    """ctrl pipe stand-in — immediate EOF so the ctrl thread exits."""

    def recv(self):
        raise EOFError


class FakeEngine:
    """Engine stand-in for worker_main — records shutdown-flush calls."""

    def __init__(self, cfg, execution_mode="worker"):
        self.cfg = cfg
        self.execution_mode = execution_mode
        self.model_id = "fake-model"
        self.model_family = "chatml"
        self.flush_calls = []
        self._lock = threading.Lock()

    def load_model(self):
        pass

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


def _run_worker(cmd_conn, monkeypatch):
    """Drive worker_main with a FakeEngine and fake pipes; returns
    (engine, resp_conn). Synchronous — worker_main returns when the loop
    exits."""
    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = FakeEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    # worker_main does `from ...mlx_engine import MLXEngine` at call time, so
    # patching the symbol on the mlx_engine module intercepts construction.
    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    resp_conn = FakeConn()
    worker_main(
        {"model_path": "/tmp/x", "verbose": False},
        cmd_conn,
        resp_conn,
        EOFConn(),
    )
    assert len(created) == 1
    return created[0], resp_conn


# --- worker_main: 'shutdown' op flushes before exit ---------------------------


def test_worker_shutdown_op_flushes_before_exit(monkeypatch, restore_signals):
    engine, resp_conn = _run_worker(FakeConn([{"op": "shutdown"}]), monkeypatch)
    # Flush ran exactly once, and worker_main returned (loop exited).
    assert engine.flush_calls == ["flush"]
    # Ready frame was still sent before the loop (sanity).
    assert resp_conn.sent and resp_conn.sent[0].get("type") == "ready"


def test_worker_pipe_eof_flushes_before_exit(monkeypatch, restore_signals):
    """Parent death (recv EOF) also triggers the shutdown flush."""
    engine, _ = _run_worker(FakeConn([]), monkeypatch)  # first recv -> EOFError
    assert engine.flush_calls == ["flush"]


def test_worker_none_cmd_flushes_before_exit(monkeypatch, restore_signals):
    engine, _ = _run_worker(FakeConn([None]), monkeypatch)
    assert engine.flush_calls == ["flush"]


# --- worker_main: SIGTERM sentinel path ---------------------------------------


class SigtermOnRecvConn(FakeConn):
    """recv() invokes the CURRENTLY-REGISTERED SIGTERM handler directly —
    covers the signal path without delivering a real signal to pytest."""

    def poll(self, timeout=None):
        return True

    def recv(self):
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)  # must raise _GracefulShutdown
        raise AssertionError("graceful handler should have raised the sentinel")


def test_worker_sigterm_sentinel_flushes_and_exits(monkeypatch, restore_signals):
    engine, _ = _run_worker(SigtermOnRecvConn(), monkeypatch)
    # The sentinel unwound the loop; the finally still flushed exactly once.
    assert engine.flush_calls == ["flush"]

    # Second signal while/after shutdown: handler must NOT raise again (it
    # would abort an in-progress flush). The worker's handler is still
    # installed here (restored by the fixture afterwards).
    handler = signal.getsignal(signal.SIGTERM)
    handler(signal.SIGTERM, None)  # no exception


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param([{"op": "shutdown"}], id="shutdown-op"),
        pytest.param([None], id="none-cmd"),
        pytest.param([], id="pipe-eof"),
    ],
)
def test_worker_signal_during_final_flush_completes(
    frames, monkeypatch, restore_signals
):
    """FINDING-1 regression: on NON-signal exit paths the shutdown flag was
    never set when the loop reached the final flush. A SIGTERM landing while
    that flush was running (the parent's close() join timeout firing
    terminate()) looked like a FIRST signal → raised _GracefulShutdown →
    aborted _flush_all_on_shutdown AFTER it drained the dirty set → the
    atexit backstop no-op'd → sessions lost.

    The flag is now set unconditionally as the first statement of the final
    flush, so the handler must take the 'already unwinding → return
    silently' path: the flush COMPLETES and no sentinel propagates out of
    worker_main."""
    import mlx_soloheaven.engine.mlx_engine as engine_module

    created = []

    class SignalDuringFlushEngine(FakeEngine):
        def _flush_all_on_shutdown(self):
            self.flush_calls.append("flush-start")
            # Parent terminate() lands mid-flush: invoke the CURRENTLY
            # registered SIGTERM handler directly (no real signal). It must
            # return silently — raising the sentinel here would abort the
            # flush and propagate out of worker_main.
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
            self.flush_calls.append("flush-end")

    def _factory(cfg, execution_mode="worker"):
        eng = SignalDuringFlushEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    # Must return normally — no _GracefulShutdown out of the finally.
    worker_main(
        {"model_path": "/tmp/x", "verbose": False},
        FakeConn(frames),
        FakeConn(),
        EOFConn(),
    )

    assert created[0].flush_calls == ["flush-start", "flush-end"]


def test_worker_flush_failure_never_propagates(monkeypatch, restore_signals):
    """Fail-closed: a flush that raises must not crash worker_main's exit."""
    import mlx_soloheaven.engine.mlx_engine as engine_module

    class ExplodingEngine(FakeEngine):
        def _flush_all_on_shutdown(self):
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(
        engine_module, "MLXEngine",
        lambda cfg, execution_mode="worker": ExplodingEngine(
            cfg, execution_mode=execution_mode
        ),
    )
    # Must return normally despite the exploding flush.
    worker_main(
        {"model_path": "/tmp/x", "verbose": False},
        FakeConn([{"op": "shutdown"}]),
        FakeConn(),
        EOFConn(),
    )


def test_graceful_shutdown_is_baseexception():
    """The sentinel must NOT be an Exception subclass, or the per-request
    `except Exception` blocks would swallow it and the loop would never
    unwind to the flush."""
    assert issubclass(_GracefulShutdown, BaseException)
    assert not issubclass(_GracefulShutdown, Exception)


# --- MLXEngine._flush_all_on_shutdown: idempotent + lock-guarded -------------


def _shell_engine(monkeypatch, saves: list):
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "shell"
    eng._lock = threading.Lock()
    eng._dirty_lock = threading.Lock()
    eng._dirty_sessions = {"s1"}
    eng._sessions = {"s1": SimpleNamespace()}
    monkeypatch.setattr(
        eng, "_save_session_to_disk", lambda sid, s: (saves.append(sid) or True)
    )
    monkeypatch.setattr(MLXEngine, "_all_engines", [eng])
    return eng


def test_flush_all_on_shutdown_saves_and_is_idempotent(monkeypatch):
    saves: list[str] = []
    eng = _shell_engine(monkeypatch, saves)

    MLXEngine._flush_all_on_shutdown()
    assert saves == ["s1"]
    assert eng._dirty_sessions == set()

    # Second call (atexit backstop after worker flush): no-op.
    MLXEngine._flush_all_on_shutdown()
    assert saves == ["s1"]


def test_flush_all_on_shutdown_bounded_lock_no_deadlock(monkeypatch):
    """Lock held (e.g. handler interrupted an in-flight generation) — the
    flush must SKIP within the bound, re-mark the ids dirty, and not save."""
    saves: list[str] = []
    eng = _shell_engine(monkeypatch, saves)

    assert eng._lock.acquire(blocking=False)
    try:
        MLXEngine._flush_all_on_shutdown(lock_timeout=0.05)
    finally:
        eng._lock.release()
    assert saves == []
    assert eng._dirty_sessions == {"s1"}  # re-marked for a later retry

    # Lock free again — the retry saves.
    MLXEngine._flush_all_on_shutdown(lock_timeout=0.05)
    assert saves == ["s1"]
    assert eng._dirty_sessions == set()


def test_flush_all_on_shutdown_save_error_continues(monkeypatch):
    """A failing per-session save is logged and never propagates."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "shell"
    eng._lock = threading.Lock()
    eng._dirty_lock = threading.Lock()
    eng._dirty_sessions = {"bad"}
    eng._sessions = {"bad": SimpleNamespace()}
    monkeypatch.setattr(
        eng, "_save_session_to_disk",
        lambda sid, s: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(MLXEngine, "_all_engines", [eng])
    MLXEngine._flush_all_on_shutdown()  # must not raise
    assert not eng._lock.locked()  # lock released on the error path


# --- MLXEngine._register_shutdown_flush: mode-agnostic registration ----------


def _fresh_registration(monkeypatch):
    """Reset the class-level once-guards and capture atexit/signal hooks.

    ``signal.getsignal`` is patched to read from the same ``sig_handlers``
    dict that the patched ``signal.signal`` writes — registration captures
    its 'previous handler' from there, so tests can pre-seed a fake previous
    handler (Uvicorn stand-in) BEFORE calling _register_shutdown_flush. An
    empty dict reads as SIG_DFL (bare-script default disposition)."""
    monkeypatch.setattr(MLXEngine, "_shutdown_flush_registered", False)
    monkeypatch.setattr(MLXEngine, "_shutdown_flush_fn", None)
    monkeypatch.setattr(MLXEngine, "_global_keepalive_started", False)
    monkeypatch.setattr(MLXEngine, "_global_keepalive_stop", threading.Event())

    atexit_fns: list = []
    sig_handlers: dict = {}
    monkeypatch.setattr(atexit, "register", lambda fn: atexit_fns.append(fn))
    monkeypatch.setattr(
        signal, "signal", lambda signum, fn: sig_handlers.__setitem__(signum, fn)
    )
    monkeypatch.setattr(
        signal, "getsignal",
        lambda signum: sig_handlers.get(signum, signal.SIG_DFL),
    )

    eng = MLXEngine.__new__(MLXEngine)
    eng.execution_mode = "main_thread"
    return eng, atexit_fns, sig_handlers


def test_register_shutdown_flush_registers_without_keepalive(monkeypatch):
    eng, atexit_fns, sig_handlers = _fresh_registration(monkeypatch)

    eng._register_shutdown_flush()

    assert len(atexit_fns) == 1
    assert signal.SIGINT in sig_handlers and signal.SIGTERM in sig_handlers
    # main_thread-mode contract: registration must NOT start the keepalive
    # background thread.
    assert MLXEngine._global_keepalive_started is False
    assert getattr(eng, "_keepalive_thread", None) is None

    # Idempotent: second registration adds nothing.
    eng._register_shutdown_flush()
    assert len(atexit_fns) == 1


def test_registered_atexit_flush_retries_not_once_guarded(monkeypatch):
    """The atexit backstop must NOT be once-guarded: every run re-invokes
    _flush_all_on_shutdown, which re-reads the LIVE dirty set — so a repeat
    is a cheap no-op when the first flush drained it, and a REAL retry when
    a timed-out lock acquire re-marked ids dirty (FINDING-2: the old
    _shutdown_done guard meant 'no retry ever')."""
    eng, atexit_fns, _ = _fresh_registration(monkeypatch)
    eng._register_shutdown_flush()

    calls: list = []
    monkeypatch.setattr(
        MLXEngine, "_flush_all_on_shutdown", classmethod(lambda cls: calls.append(1))
    )

    atexit_fns[0]()
    assert calls == [1]
    assert MLXEngine._global_keepalive_stop.is_set()
    # Second run (signal handler + atexit overlap) re-invokes the flush —
    # idempotency lives in _flush_all_on_shutdown's drained dirty set, not
    # in a once-guard, so re-marked-dirty sessions get retried.
    atexit_fns[0]()
    assert calls == [1, 1]


def test_registered_sigterm_handler_flushes_and_redelivers_sig_dfl(monkeypatch):
    """SIG_DFL previous disposition (bare script / tests — nothing installed
    before us): invoke the registered SIGTERM handler DIRECTLY (no real
    signal). It must flush, then restore SIG_DFL and re-deliver via os.kill
    so the process still dies of the signal."""
    eng, _, sig_handlers = _fresh_registration(monkeypatch)
    eng._register_shutdown_flush()

    calls: list = []
    monkeypatch.setattr(
        MLXEngine, "_flush_all_on_shutdown", classmethod(lambda cls: calls.append(1))
    )
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: kills.append(signum))

    sig_handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert calls == [1]
    assert kills == [signal.SIGTERM]
    # The handler restored the default disposition before re-raising.
    assert sig_handlers[signal.SIGTERM] == signal.SIG_DFL


def test_registered_signal_handler_chains_previous_handler(monkeypatch):
    """FINDING-2 regression (ownership contract): a callable handler
    installed BEFORE us (Uvicorn's — load_model runs during server startup)
    must be CHAINED after the flush, with the same (signum, frame) args, and
    the process must NOT be re-killed — Uvicorn owns process lifetime and
    drives its own graceful shutdown (lifespan shutdown → server.py hook)."""
    eng, _, sig_handlers = _fresh_registration(monkeypatch)

    prev_calls: list = []
    sig_handlers[signal.SIGTERM] = (
        lambda signum, frame: prev_calls.append((signum, frame))
    )

    eng._register_shutdown_flush()

    calls: list = []
    monkeypatch.setattr(
        MLXEngine, "_flush_all_on_shutdown", classmethod(lambda cls: calls.append(1))
    )
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: kills.append(signum))

    our_handler = sig_handlers[signal.SIGTERM]
    fake_frame = object()
    our_handler(signal.SIGTERM, fake_frame)

    assert calls == [1]  # flush ran first
    assert prev_calls == [(signal.SIGTERM, fake_frame)]  # chained, same args
    assert kills == []  # process NOT re-killed — previous handler owns it
    # Disposition untouched (no SIG_DFL reset on the chaining path).
    assert sig_handlers[signal.SIGTERM] is our_handler


def test_registered_signal_handler_respects_sig_ign(monkeypatch):
    """SIG_IGN previous disposition: flush, then just return — no chain
    call, no re-delivery."""
    eng, _, sig_handlers = _fresh_registration(monkeypatch)
    sig_handlers[signal.SIGTERM] = signal.SIG_IGN

    eng._register_shutdown_flush()

    calls: list = []
    monkeypatch.setattr(
        MLXEngine, "_flush_all_on_shutdown", classmethod(lambda cls: calls.append(1))
    )
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, signum: kills.append(signum))

    sig_handlers[signal.SIGTERM](signal.SIGTERM, None)  # must not raise

    assert calls == [1]
    assert kills == []


def test_timed_out_flush_remarks_dirty_then_atexit_backstop_saves(monkeypatch):
    """FINDING-2 regression (end-to-end retry): a signal interrupts a
    generation that holds the engine lock → the handler's flush times out
    (bounded acquire), re-marks the ids dirty, and the handler returns
    without terminating anything. Later the atexit backstop re-reads the
    LIVE dirty set and actually saves."""
    eng_reg, atexit_fns, sig_handlers = _fresh_registration(monkeypatch)
    eng_reg._register_shutdown_flush()

    saves: list[str] = []
    eng = _shell_engine(monkeypatch, saves)

    # The registered hooks call _flush_all_on_shutdown() with the default
    # 10s lock timeout — wrap the REAL flush with a fast bound for the test.
    orig_flush = MLXEngine._flush_all_on_shutdown.__func__
    monkeypatch.setattr(
        MLXEngine, "_flush_all_on_shutdown",
        classmethod(lambda cls: orig_flush(cls, lock_timeout=0.05)),
    )
    monkeypatch.setattr(os, "kill", lambda pid, signum: None)

    # In-flight generation holds the non-reentrant engine lock when SIGTERM
    # lands (main_thread mode: handler runs on the same thread).
    assert eng._lock.acquire(blocking=False)
    try:
        sig_handlers[signal.SIGTERM](signal.SIGTERM, None)
    finally:
        eng._lock.release()
    assert saves == []  # bounded acquire timed out — nothing saved yet
    assert eng._dirty_sessions == {"s1"}  # re-marked for the retry

    # atexit backstop: must re-read the dirty set (not a snapshot) and save.
    atexit_fns[0]()
    assert saves == ["s1"]
    assert eng._dirty_sessions == set()


def test_registered_flush_is_failclosed(monkeypatch):
    """A raising flush must not propagate out of the shutdown handler."""
    eng, atexit_fns, _ = _fresh_registration(monkeypatch)
    eng._register_shutdown_flush()
    monkeypatch.setattr(
        MLXEngine,
        "_flush_all_on_shutdown",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    atexit_fns[0]()  # must not raise


# --- worker _idle_flush -------------------------------------------------------


def test_idle_flush_flushes_under_free_lock():
    flushed: list = []
    eng = SimpleNamespace(
        _lock=threading.Lock(),
        _flush_dirty_sessions=lambda: flushed.append(1),
    )
    _idle_flush(eng)
    assert flushed == [1]
    assert not eng._lock.locked()  # released afterwards


def test_idle_flush_bails_when_lock_held():
    flushed: list = []
    eng = SimpleNamespace(
        _lock=threading.Lock(),
        _flush_dirty_sessions=lambda: flushed.append(1),
    )
    assert eng._lock.acquire(blocking=False)
    try:
        _idle_flush(eng)  # must return immediately, never block
    finally:
        eng._lock.release()
    assert flushed == []


def test_idle_flush_failclosed():
    eng = SimpleNamespace(
        _lock=threading.Lock(),
        _flush_dirty_sessions=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _idle_flush(eng)  # must not raise
    assert not eng._lock.locked()


# --- EngineProcessProxy.close(): graceful, bounded, idempotent ----------------


def test_proxy_close_sends_shutdown_waits_and_is_idempotent():
    proxy = EngineProcessProxy.__new__(EngineProcessProxy)
    proxy._closed = False
    proxy._cmd_send_lock = threading.Lock()
    sent: list = []
    proxy._cmd_parent = SimpleNamespace(send=lambda obj: sent.append(obj))
    joins: list = []
    proxy._proc = SimpleNamespace(
        join=lambda timeout=None: joins.append(timeout),
        is_alive=lambda: False,
        terminate=lambda: pytest.fail("graceful child must not be terminated"),
    )

    proxy.close()
    assert sent == [{"op": "shutdown"}]
    assert joins == [10.0]  # bounded wait for the child's flush

    proxy.close()  # idempotent — no second shutdown op / join
    assert sent == [{"op": "shutdown"}]
    assert joins == [10.0]


def test_proxy_close_terminates_wedged_child_then_reaps():
    proxy = EngineProcessProxy.__new__(EngineProcessProxy)
    proxy._closed = False
    proxy._cmd_send_lock = threading.Lock()
    proxy._cmd_parent = SimpleNamespace(send=lambda obj: None)
    events: list = []
    alive = {"v": True}

    def _join(timeout=None):
        events.append(("join", timeout))

    def _terminate():
        events.append(("terminate",))
        alive["v"] = False  # SIGTERM -> child's graceful handler exits it

    proxy._proc = SimpleNamespace(
        join=_join, is_alive=lambda: alive["v"], terminate=_terminate
    )

    proxy.close(join_timeout=0.01)
    # join -> still alive -> terminate -> join again (let the child's SIGTERM
    # flush finish).
    assert events == [("join", 0.01), ("terminate",), ("join", 0.01)]
