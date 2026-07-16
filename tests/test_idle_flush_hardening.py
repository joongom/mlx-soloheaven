"""U16 + U17 — idle-flush hardening (batch 3).

U16 — SIGTERM during the idle flush must not lose dirty sessions: the old
``_flush_dirty_sessions`` drained the WHOLE dirty set up front and only
``except Exception`` re-marked failures — the worker's SIGTERM sentinel
(``_GracefulShutdown``, a BaseException) unwinding mid-loop lost every
drained-but-unsaved id, so the shutdown flush saw an EMPTY set. CHOSEN FIX:
pop-one-save-one (each id leaves the dirty set only right before its own
save) + an ``except BaseException`` re-mark of the single in-flight id
before re-raising — an interrupt loses NOTHING and the shutdown flush's own
pass (which re-reads the live dirty set) saves the remainder. The
shutdown-flush flag interplay is untouched: ``_flush_engine_for_shutdown``
still sets the flag FIRST (covered by tests/test_shutdown_flush.py).

U17 — periodic admin/health RPC traffic must not starve the 60s idle flush:
every recv used to reset ``last_activity`` while the flush waits for 60s of
quiet, so a dashboard polling stats every 30s kept dirty sessions unflushed
forever. Now only an explicit ALLOWLIST of generation-shaped ops resets the
timer; read-only RPCs (stats/list/overview/health) and unknown future ops
default to non-resetting.

HERMETIC — same conventions as tests/test_shutdown_flush.py (shell engines,
fake pipes, no real signals; U17's loop test drives worker_main under a fake
clock).
"""

from __future__ import annotations

import signal
import threading
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine import process_worker
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.process_worker import (
    ACTIVITY_RPC_METHODS,
    GENERATION_OPS,
    _GracefulShutdown,
    _resets_activity,
    worker_main,
)
from mlx_soloheaven.engine.types import GenerationResult


# --- U16: BaseException during the idle flush -----------------------------------


def _flush_shell(save_fn) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "shell"
    eng._lock = threading.Lock()
    eng._dirty_lock = threading.Lock()
    eng._dirty_sessions = {"s1", "s2"}
    eng._sessions = {"s1": SimpleNamespace(), "s2": SimpleNamespace()}
    eng._save_session_to_disk = save_fn
    return eng


def test_flush_baseexception_loses_nothing(monkeypatch):
    """The SIGTERM sentinel lands during the FIRST save: the in-flight id is
    re-marked and the not-yet-popped id was never drained — BOTH survive in
    the dirty set for the shutdown flush, which then saves them."""
    calls: list[str] = []

    def _save(sid, session):
        calls.append(sid)
        if len(calls) == 1:
            raise _GracefulShutdown(signal.SIGTERM)
        return True

    eng = _flush_shell(_save)

    with pytest.raises(_GracefulShutdown):
        eng._flush_dirty_sessions()

    # NOTHING lost: the interrupted save's id was re-marked, the other never
    # left the set.
    assert eng._dirty_sessions == {"s1", "s2"}
    assert len(calls) == 1

    # The shutdown flush (re-reads the LIVE dirty set) now saves both.
    monkeypatch.setattr(MLXEngine, "_all_engines", [eng])
    MLXEngine._flush_all_on_shutdown()
    assert eng._dirty_sessions == set()
    assert set(calls[1:]) == {"s1", "s2"}


class _DiscardRaisingSet(set):
    """Dirty-set stand-in whose FIRST discard performs the removal and THEN
    raises the SIGTERM sentinel — the exact residual F1 window: the id has
    already left the dirty set but the save has not run yet."""

    def __init__(self, *args):
        super().__init__(*args)
        self.raised = False

    def discard(self, sid):
        super().discard(sid)
        if not self.raised:
            self.raised = True
            raise _GracefulShutdown(signal.SIGTERM)


def test_flush_baseexception_in_pop_save_gap_remarks_id(monkeypatch):
    """F1 (codex batch-3): a signal landing IMMEDIATELY AFTER the pop —
    before the save even starts — must re-mark the popped id. Pre-fix, the
    pop happened outside the BaseException guard, so an interrupt in the
    pop→save gap lost exactly that id (the shutdown flush saw it missing
    from the set and never saved it)."""
    calls: list[str] = []

    def _save(sid, session):
        calls.append(sid)
        return True

    eng = _flush_shell(_save)
    eng._dirty_sessions = _DiscardRaisingSet({"s1", "s2"})

    with pytest.raises(_GracefulShutdown):
        eng._flush_dirty_sessions()

    # NO save ran (the interrupt landed right after the pop), and NOTHING
    # was lost: the popped id was re-marked, the other never left the set.
    assert calls == []
    assert set(eng._dirty_sessions) == {"s1", "s2"}

    # The shutdown flush (re-reads the LIVE dirty set) saves both.
    monkeypatch.setattr(MLXEngine, "_all_engines", [eng])
    MLXEngine._flush_all_on_shutdown()
    assert set(eng._dirty_sessions) == set()
    assert sorted(calls) == ["s1", "s2"]


def test_flush_baseexception_inside_failure_handler_remarks_id(monkeypatch):
    """F1 residual (codex batch-3, round 2): the sentinel lands INSIDE the
    exception-handling path itself — the save fails with an ordinary
    Exception and the handler's LOGGING raises the SIGTERM sentinel before
    the handler's re-mark ran. A sibling ``except BaseException`` cannot
    catch an exception raised inside another except block, so pre-fix
    exactly that popped id was lost; the try/finally shape re-marks it
    regardless of where the unwind started."""
    import mlx_soloheaven.engine.mlx_engine as engine_module

    calls: list[str] = []

    def _save(sid, session):
        calls.append(sid)
        raise RuntimeError("disk on fire")

    def _raising_error(*args, **kwargs):
        raise _GracefulShutdown(signal.SIGTERM)

    eng = _flush_shell(_save)
    monkeypatch.setattr(engine_module.logger, "error", _raising_error)

    with pytest.raises(_GracefulShutdown):
        eng._flush_dirty_sessions()

    # ONE save was attempted (and failed); the sentinel then fired inside
    # the handler — the in-flight id was still re-marked and the other
    # never left the set. NOTHING lost for the shutdown flush.
    assert len(calls) == 1
    assert eng._dirty_sessions == {"s1", "s2"}


def test_flush_sentinel_propagates_not_swallowed():
    """The sentinel must RE-RAISE (BaseException) — swallowing it would keep
    the worker loop alive after SIGTERM."""
    def _save(sid, session):
        raise _GracefulShutdown(signal.SIGTERM)

    eng = _flush_shell(_save)
    with pytest.raises(_GracefulShutdown):
        eng._flush_dirty_sessions()


def test_flush_ordinary_exception_remarks_without_infinite_loop():
    """Pop-one-save-one regression guard: a failing save re-marks its id for
    the NEXT flush cycle — this call must attempt each id EXACTLY once and
    terminate (no re-pop of a just-re-marked id)."""
    calls: list[str] = []

    def _save(sid, session):
        calls.append(sid)
        raise RuntimeError("disk on fire")

    eng = _flush_shell(_save)
    eng._flush_dirty_sessions()  # must return (and not raise)
    assert sorted(calls) == ["s1", "s2"]  # each attempted exactly once
    assert eng._dirty_sessions == {"s1", "s2"}  # re-marked for the next cycle


def test_flush_normal_path_saves_and_drains():
    calls: list[str] = []

    def _save(sid, session):
        calls.append(sid)
        return True

    eng = _flush_shell(_save)
    eng._flush_dirty_sessions()
    assert sorted(calls) == ["s1", "s2"]
    assert eng._dirty_sessions == set()


def test_flush_skips_evicted_sessions():
    """A dirty id whose session left memory is dropped, not retried."""
    calls: list[str] = []
    eng = _flush_shell(lambda sid, s: calls.append(sid) or True)
    eng._sessions.pop("s2")
    eng._flush_dirty_sessions()
    assert calls == ["s1"]
    assert eng._dirty_sessions == set()


# --- U17: op classification -------------------------------------------------------


def test_generation_ops_reset_activity():
    for op in GENERATION_OPS:
        assert _resets_activity({"op": op, "id": "r1"})


def test_mutating_rpcs_reset_activity():
    for method in ACTIVITY_RPC_METHODS:
        assert _resets_activity({"op": "rpc", "id": "r1", "method": method})


@pytest.mark.parametrize("method", [
    "list_sessions", "session_stats", "get_session", "base_cache_stats",
    "cache_overview", "liveness", "some_future_read",
])
def test_readonly_rpcs_do_not_reset_activity(method):
    assert not _resets_activity({"op": "rpc", "id": "r1", "method": method})


def test_non_dict_and_control_ops_do_not_reset_activity():
    assert not _resets_activity(None)
    assert not _resets_activity("junk")
    assert not _resets_activity({"op": "shutdown"})
    assert not _resets_activity({"op": "totally_new_op"})


# --- U17: worker-loop integration under a fake clock ------------------------------


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def perf_counter(self):
        return self.t


class ProgrammedConn:
    """cmd pipe stand-in driven by a step program:
    ("timeout", advance_s) -> poll() advances the fake clock and reports no
    data; ("frame", obj) -> poll() True, recv() returns obj. Exhausted
    program -> recv EOF (ends the loop)."""

    def __init__(self, program, clock):
        self.program = list(program)
        self.clock = clock
        self.sent = []

    def poll(self, timeout=None):
        if not self.program:
            return True  # recv -> EOFError
        kind, val = self.program[0]
        if kind == "timeout":
            self.program.pop(0)
            self.clock.t += val
            return False
        return True

    def recv(self):
        if not self.program:
            raise EOFError
        kind, val = self.program.pop(0)
        assert kind == "frame"
        return val

    def send(self, obj):
        self.sent.append(obj)


class EOFConn:
    def recv(self):
        raise EOFError


class FlushRecordingEngine:
    def __init__(self, cfg, execution_mode="worker"):
        self.cfg = cfg
        self.execution_mode = execution_mode
        self.model_id = "fake-model"
        self.model_family = "chatml"
        self.idle_flushes: list = []
        self.flush_calls: list = []
        self._lock = threading.Lock()

    def load_model(self):
        pass

    def _flush_dirty_sessions(self):
        self.idle_flushes.append(1)

    def _flush_all_on_shutdown(self):
        self.flush_calls.append("flush")

    def list_sessions(self):
        return []

    def generate_stream(self, messages, **kwargs):
        yield GenerationResult(finish_reason="stop")


@pytest.fixture
def restore_signals():
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)


def _run_programmed_worker(program, monkeypatch, engine_factory=None, clock=None):
    clock = clock if clock is not None else FakeClock()
    monkeypatch.setattr(
        process_worker, "time",
        SimpleNamespace(monotonic=clock.monotonic, perf_counter=clock.perf_counter),
    )

    created = []

    def _factory(cfg, execution_mode="worker"):
        cls = engine_factory or FlushRecordingEngine
        eng = cls(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    cmd = ProgrammedConn(program, clock)
    resp = ProgrammedConn([], clock)
    worker_main({"model_path": "/tmp/x", "verbose": False}, cmd, resp, EOFConn())
    return created[0]


def test_readonly_rpc_does_not_starve_idle_flush(monkeypatch, restore_signals):
    """A read-only RPC at t=30 does NOT reset the flush timer: at t=61 the
    60s quiet window (measured from t=0) has elapsed and the flush runs —
    the pre-fix behavior (reset at every recv) would have deferred it."""
    engine = _run_programmed_worker([
        ("timeout", 30.0),  # t=30: quiet, window not yet elapsed
        ("frame", {"op": "rpc", "id": "r1", "method": "list_sessions",
                   "args": [], "kwargs": {}}),
        ("timeout", 31.0),  # t=61: 61s since the LAST activity (t=0)
        ("frame", {"op": "shutdown"}),
    ], monkeypatch)
    assert engine.idle_flushes == [1]


def test_generation_op_resets_idle_flush_timer(monkeypatch, restore_signals):
    """A generation at t=30 DOES reset the timer: at t=61 only 31s of quiet
    have passed — no flush yet."""
    engine = _run_programmed_worker([
        ("timeout", 30.0),
        ("frame", {"op": "generate", "id": "g1",
                   "payload": {"messages": [], "params": {}}}),
        ("timeout", 31.0),  # t=61: 31s since the generation at t=30
        ("frame", {"op": "shutdown"}),
    ], monkeypatch)
    assert engine.idle_flushes == []


def test_generation_then_quiet_window_flushes(monkeypatch, restore_signals):
    """Sanity: after a generation, a FULL quiet window still flushes."""
    engine = _run_programmed_worker([
        ("frame", {"op": "generate", "id": "g1",
                   "payload": {"messages": [], "params": {}}}),
        ("timeout", 61.0),  # 61s of quiet after the generation
        ("frame", {"op": "shutdown"}),
    ], monkeypatch)
    assert engine.idle_flushes == [1]


# --- codex round 11, finding 3: sustained read traffic must not starve the flush ---


def _read_frame(i):
    return ("frame", {"op": "rpc", "id": f"r{i}", "method": "list_sessions",
                      "args": [], "kwargs": {}})


def test_sustained_read_rpcs_do_not_starve_idle_flush(
    monkeypatch, restore_signals
):
    """CODEX ROUND 11 MED repro: reads arriving more often than the 5s poll
    timeout keep poll() returning data, so the poll-timeout branch —
    pre-fix the ONLY place the flush deadline was evaluated — never runs:
    61 one-second read RPCs = 61s elapsed, ZERO idle flushes while dirty
    sessions age unbounded (reads correctly do NOT reset last_activity, but
    they also never let the deadline be CHECKED). The command path now
    evaluates the same deadline after handling each non-activity command.
    Flush timestamps are recorded so the assertion pins WHERE the flush
    fired: exactly one, at t=60 IN THE COMMAND PATH (the window elapses
    after the 60th read and resets), and neither the final read (t=61) nor
    the trailing 5s poll timeout (t=66, which pre-fix produced the only —
    and far too late — flush) re-fires."""
    clock = FakeClock()

    class ReadTickEngine(FlushRecordingEngine):
        def list_sessions(self):
            clock.t += 1.0  # each read RPC costs 1s of wall clock
            return []

        def _flush_dirty_sessions(self):
            self.idle_flushes.append(clock.t)  # record WHEN it flushed

    program = (
        [_read_frame(i) for i in range(61)]
        + [("timeout", 5.0)]  # quiet poll AFTER the flush: no double-flush
        + [("frame", {"op": "shutdown"})]
    )
    engine = _run_programmed_worker(
        program, monkeypatch, engine_factory=ReadTickEngine, clock=clock
    )
    assert engine.idle_flushes == [60.0]


def test_activity_rpc_still_resets_deadline_under_read_traffic(
    monkeypatch, restore_signals
):
    """A mutating RPC in the middle of the read stream RESETS the deadline:
    the new command-path check must measure from the LAST activity, never
    from a stale window. 30 reads (t=30), an activity RPC (reset at t=30),
    then 40 more reads ending at t=71 — 71s total elapsed (the pre-reset
    window would have fired) but only 41s since the activity: NO flush."""
    clock = FakeClock()

    class TickEngine(FlushRecordingEngine):
        def list_sessions(self):
            clock.t += 1.0
            return []

        def update_session_messages(self, *a, **k):
            clock.t += 1.0
            return True

    program = (
        [_read_frame(i) for i in range(30)]  # t: 0 -> 30
        + [("frame", {"op": "rpc", "id": "m1",
                      "method": "update_session_messages",
                      "args": [], "kwargs": {}})]  # activity: reset at t=30
        + [_read_frame(100 + i) for i in range(40)]  # t: 31 -> 71
        + [("frame", {"op": "shutdown"})]
    )
    engine = _run_programmed_worker(
        program, monkeypatch, engine_factory=TickEngine, clock=clock
    )
    assert engine.idle_flushes == []
