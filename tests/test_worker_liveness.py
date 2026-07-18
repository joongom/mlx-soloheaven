"""Worker liveness + auto-respawn coverage for the process-mode proxy.

PRODUCTION INCIDENT being fixed: the child worker died mid-prefill (Metal GPU
Timeout, libc++abi abort — uncatchable in-process). The parent FastAPI kept
serving for 1.5 days: ``_reader_loop`` just ``break``-ed on EOFError/OSError
— nothing marked the proxy dead, failed pending requests, respawned the
child, or rejected new requests. Clients got HTTP 200 + dead streams.

These tests are HERMETIC (no real model, no real child process; follows
tests/test_shutdown_flush.py conventions for proxy tests):
  - reader EOF -> proxy marked dead, ALL pending RPC slots failed with an
    ``engine_dead`` error frame, ALL live streaming queues get the same frame
    (so waiting generators terminate with a clear message), respawn kicked,
  - dead proxy generate/RPC -> fail FAST with EngineRestartingError (and
    lazily kick the respawn),
  - dead proxy -> respawn (mocked _spawn) -> a subsequent generate AND RPC
    are served successfully through the fresh child conn,
  - respawn fails N times -> proxy permanently dead, ensure_available keeps
    failing fast, no further respawn is ever kicked,
  - close() prevents respawn (closed proxy never respawns), including the
    close()-races-respawn path (fresh child is shut down again),
  - a reader EOF BEFORE the first ready is a startup failure (start() raises)
    — no death handling / background respawn,
  - API layer: EngineRestartingError -> HTTP 503 JSON body (hermetic FastAPI
    app via TestClient); already-started SSE streams (OpenAI-compat + web
    chat + compaction) emit an in-band error frame then close; the admin
    models overview exposes the liveness snapshot even while the child is
    dead,
  - per-spawn GENERATION token: a stale reader's delayed EOF (or late ready
    frame) from an obsolete spawn never kills / confuses a healthy
    replacement,
  - replacement dies AFTER `ready` but BEFORE the respawn loop clears
    _dead/_respawning: the death is recorded (not swallowed by the
    already-dead early return), the loop counts the attempt as FAILED
    instead of declaring a dead child alive, and a later successful spawn
    (or attempt exhaustion -> permanently dead) proceeds normally,
  - reap escalation: terminate -> kill -> refuse-to-respawn when the old
    worker won't die (no double GPU residency),
  - _shutdown_spawned_child mirrors close()'s second bounded join after
    SIGTERM (the fresh child's shutdown flush is not cut off),
  - availability preflight BEFORE DB mutation: chat / branch / delete-last /
    regenerate / compact answer 503 with ZERO DB writes while the engine is
    dead (no duplicated user message on client retry).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest

from mlx_soloheaven.engine import process_protocol as proto
from mlx_soloheaven.engine.process_client import (
    EngineProcessProxy,
    EngineRestartingError,
)
from mlx_soloheaven.engine.types import GenerationResult


# --- helpers ----------------------------------------------------------------


class FakeConn:
    """Minimal Pipe-connection stand-in: scripted recv frames + send log.
    Empty frame list -> recv raises EOFError (child died / pipe closed)."""

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


class FakeChildConn:
    """cmd-pipe stand-in acting like an instantly-responding child worker.

    RPC frames are answered by filling the proxy's RPC slot ("pong");
    generate frames are answered by routing batch/final/done frames into the
    request's streaming queue — enough to prove a respawned proxy actually
    SERVES requests again."""

    def __init__(self, proxy):
        self.proxy = proxy
        self.sent = []

    def send(self, frame):
        self.sent.append(frame)
        rid = frame.get("id")
        op = frame.get("op")
        if op == "rpc":
            with self.proxy._rpc_lock:
                slot = self.proxy._rpc_results.get(rid)
            if slot is not None:
                slot["frame"] = proto.make_rpc_result(rid, "pong")
                slot["event"].set()
        elif op in ("generate", "generate_scalar"):
            with self.proxy._routing_lock:
                q = self.proxy._queues.get(rid)
                loop = self.proxy._loops.get(rid)
            if q is None or loop is None:
                return
            frames = [
                proto.make_batch(rid, [GenerationResult(text="ok").to_dict()]),
                proto.make_final(
                    rid, GenerationResult(finish_reason="stop").to_dict()
                ),
                proto.make_done(rid),
            ]
            for f in frames:
                loop.call_soon_threadsafe(q.put_nowait, f)


def _make_proxy() -> EngineProcessProxy:
    """Real __init__ (cheap since pipes/process moved into _spawn), with a
    once-ready history so death handling is armed, and zero backoff."""
    proxy = EngineProcessProxy(SimpleNamespace())
    proxy._ever_ready = True
    proxy._respawn_backoff_s = 0.0
    return proxy


def _wait_until(cond, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _collect(agen) -> list:
    return [item async for item in agen]


# --- death detection: reader EOF fails pending RPCs + streams -----------------


def test_reader_eof_marks_dead_and_fails_pending_rpc():
    proxy = _make_proxy()
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)
    proxy._proc = SimpleNamespace(exitcode=-6, is_alive=lambda: False)

    ev = threading.Event()
    proxy._rpc_results["r1"] = {"event": ev, "frame": None}

    proxy._reader_loop(FakeConn([]))  # first recv -> EOFError -> death

    assert proxy._dead is True
    assert "EOF" in proxy._dead_reason
    assert ev.is_set()
    frame = proxy._rpc_results["r1"]["frame"]
    assert frame["type"] == "error"
    assert frame["engine_dead"] is True
    assert "died" in frame["error"]
    assert kicks == [1]  # auto-respawn kicked exactly once


def test_reader_oserror_marks_dead():
    class OSErrorConn(FakeConn):
        def recv(self):
            raise OSError("handle is closed")

    proxy = _make_proxy()
    proxy._kick_respawn = lambda: None
    proxy._reader_loop(OSErrorConn())
    assert proxy._dead is True
    assert "OSError" in proxy._dead_reason


def test_reader_eof_fails_live_stream_queues():
    proxy = _make_proxy()
    proxy._kick_respawn = lambda: None

    loop = asyncio.new_event_loop()
    try:
        q: asyncio.Queue = asyncio.Queue()
        proxy._queues["s1"] = q
        proxy._loops["s1"] = loop

        proxy._reader_loop(FakeConn([]))

        # The death sweep scheduled the put via call_soon_threadsafe; run the
        # loop so it is delivered to the waiting generator's queue.
        frame = loop.run_until_complete(asyncio.wait_for(q.get(), timeout=2.0))
    finally:
        loop.close()

    assert frame["type"] == "error"
    assert frame["engine_dead"] is True
    assert "died" in frame["error"]


def test_stream_death_error_frame_raises_restarting():
    """Child dies MID-generation: the injected death frame terminates the
    waiting generator with EngineRestartingError (SSE paths turn this into an
    in-band error event)."""
    proxy = _make_proxy()
    proxy._kick_respawn = lambda: None
    proxy._proc = SimpleNamespace(exitcode=-6, is_alive=lambda: False)
    sent = []
    proxy._cmd_parent = SimpleNamespace(send=lambda obj: sent.append(obj))

    async def _drive():
        agen = proxy.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}]
        )
        it = agen.__aiter__()
        task = asyncio.ensure_future(it.__anext__())
        while not sent:  # wait for the generate cmd to be registered + sent
            await asyncio.sleep(0)
        proxy._handle_child_death("resp pipe EOF (child worker exited)")
        return await task

    with pytest.raises(EngineRestartingError, match="died mid-generation"):
        asyncio.run(_drive())
    assert proxy._dead is True


def test_rpc_death_mid_wait_raises_restarting():
    """Child dies while an RPC waits: the failed slot unblocks the waiter
    immediately with EngineRestartingError (not a 600s timeout)."""
    proxy = _make_proxy()
    proxy._kick_respawn = lambda: None
    proxy._proc = SimpleNamespace(exitcode=1, is_alive=lambda: False)

    def _send_then_die(frame):
        threading.Thread(
            target=proxy._handle_child_death,
            args=("resp pipe EOF (child worker exited)",),
            daemon=True,
        ).start()

    proxy._cmd_parent = SimpleNamespace(send=_send_then_die)

    t0 = time.monotonic()
    with pytest.raises(EngineRestartingError):
        proxy._rpc("session_stats", timeout=30.0)
    assert time.monotonic() - t0 < 10.0  # unblocked early, not at timeout


# --- fail fast while dead / respawning ----------------------------------------


def test_dead_proxy_generate_fails_fast_and_kicks_respawn():
    proxy = _make_proxy()
    proxy._dead = True
    proxy._dead_reason = "resp pipe EOF (child worker exited)"
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)

    async def _drive():
        async for _ in proxy.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}]
        ):
            pass

    with pytest.raises(EngineRestartingError, match="respawn"):
        asyncio.run(_drive())
    assert kicks == [1]


def test_dead_proxy_rpc_fails_fast():
    proxy = _make_proxy()
    proxy._dead = True
    proxy._kick_respawn = lambda: None
    with pytest.raises(EngineRestartingError):
        proxy._rpc("list_sessions")


def test_respawning_proxy_fails_fast_without_new_kick():
    proxy = _make_proxy()
    proxy._dead = True
    proxy._respawning = True
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)
    with pytest.raises(EngineRestartingError, match="restarting"):
        proxy.ensure_available()
    assert kicks == []  # already respawning — no duplicate kick


# --- auto-respawn: success path -----------------------------------------------


def test_dead_proxy_respawn_then_serves_generate_and_rpc():
    """Dead proxy -> generate fails fast + triggers respawn (mocked spawn) ->
    once the respawn completes, generate AND RPC are served again."""
    proxy = _make_proxy()
    proxy._dead = True
    proxy._dead_reason = "resp pipe EOF (child worker exited)"
    proxy._reap_dead_child = lambda: True

    def _fake_spawn(ready_timeout):
        proxy._cmd_parent = FakeChildConn(proxy)

    proxy._spawn = _fake_spawn

    async def _drive_fail():
        async for _ in proxy.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}]
        ):
            pass

    with pytest.raises(EngineRestartingError):
        asyncio.run(_drive_fail())  # fails fast, lazily kicks the respawn

    _wait_until(lambda: not proxy._dead and not proxy._respawning)
    assert proxy._respawn_attempts == 1
    assert proxy.liveness()["alive"] is True
    proxy.ensure_available()  # no raise anymore

    async def _drive_ok():
        out = []
        async for batch in proxy.generate_stream_batches_async(
            [{"role": "user", "content": "hi"}]
        ):
            out.extend(batch)
        return out

    results = asyncio.run(_drive_ok())
    assert any(r.text == "ok" for r in results)
    assert results[-1].finish_reason == "stop"
    assert proxy._rpc("list_sessions") == "pong"


def test_respawn_serialized_single_flight():
    """_kick_respawn's False->True _respawning transition is the gate: N
    concurrent kicks start exactly ONE respawn loop."""
    proxy = _make_proxy()
    proxy._dead = True
    proxy._reap_dead_child = lambda: True

    spawn_calls = []
    release = threading.Event()

    def _slow_spawn(ready_timeout):
        spawn_calls.append(1)
        release.wait(timeout=5.0)
        proxy._cmd_parent = FakeChildConn(proxy)

    proxy._spawn = _slow_spawn

    for _ in range(5):
        proxy._kick_respawn()
    time.sleep(0.1)  # let any (wrongly) extra threads reach _spawn
    release.set()
    _wait_until(lambda: not proxy._respawning)
    assert spawn_calls == [1]


# --- auto-respawn: bounded retries -> permanently dead --------------------------


def test_respawn_exhaustion_goes_permanently_dead():
    proxy = _make_proxy()
    proxy._dead = True
    proxy._reap_dead_child = lambda: True

    attempts = []

    def _fail_spawn(ready_timeout):
        attempts.append(1)
        raise RuntimeError("model load failed")

    proxy._spawn = _fail_spawn

    proxy._kick_respawn()
    _wait_until(lambda: not proxy._respawning)

    assert len(attempts) == 3  # bounded retries
    assert proxy._permanently_dead is True
    assert proxy.liveness()["alive"] is False
    assert proxy.liveness()["permanently_dead"] is True

    with pytest.raises(EngineRestartingError, match="permanently"):
        proxy.ensure_available()

    # No further respawn is EVER kicked once permanently dead.
    proxy._kick_respawn()
    time.sleep(0.05)
    assert proxy._respawning is False
    assert len(attempts) == 3


# --- close() interplay: closed proxy never respawns ----------------------------


def test_close_prevents_death_handling_and_respawn():
    proxy = _make_proxy()
    proxy._proc = SimpleNamespace(
        join=lambda timeout=None: None,
        is_alive=lambda: False,
        terminate=lambda: pytest.fail("graceful child must not be terminated"),
    )
    proxy._cmd_parent = FakeConn()
    proxy.close()
    assert proxy._cmd_parent.sent == [{"op": "shutdown"}]

    spawned = []
    proxy._spawn = lambda ready_timeout: spawned.append(1)

    # The reader EOF caused by close()'s own shutdown is the EXPECTED end of
    # life — no death marking, no respawn.
    proxy._reader_loop(FakeConn([]))
    assert proxy._dead is False

    proxy._kick_respawn()
    time.sleep(0.05)
    assert spawned == []
    assert proxy._respawning is False

    with pytest.raises(EngineRestartingError, match="shut down"):
        proxy.ensure_available()


def test_close_racing_respawn_stops_fresh_child():
    """close() lands while a respawn attempt is loading the model: the loop
    must notice and gracefully stop the FRESH child instead of resurrecting a
    closed proxy."""
    proxy = _make_proxy()
    proxy._dead = True
    proxy._reap_dead_child = lambda: True

    fresh_cmd = FakeConn()
    fresh_proc = SimpleNamespace(
        join=lambda timeout=None: None, is_alive=lambda: False,
        terminate=lambda: None,
    )

    def _spawn_then_closed(ready_timeout):
        proxy._cmd_parent = fresh_cmd
        proxy._proc = fresh_proc
        proxy._closed = True  # close() arrived mid-spawn

    proxy._spawn = _spawn_then_closed

    proxy._kick_respawn()
    _wait_until(lambda: not proxy._respawning)

    assert {"op": "shutdown"} in fresh_cmd.sent  # fresh child stopped again
    assert proxy._dead is True  # never resurrected


def test_reader_eof_before_first_ready_is_startup_failure_no_respawn():
    """A child that never reached `ready` is a STARTUP failure: start()
    raises to the caller — the reader EOF must not mark dead / kick respawn."""
    proxy = EngineProcessProxy(SimpleNamespace())  # _ever_ready is False
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)
    proxy._reader_loop(FakeConn([]))
    assert proxy._dead is False
    assert kicks == []


# --- per-spawn generation token: stale reader callbacks are ignored -------------


def test_stale_generation_eof_ignored_keeps_current_child_alive():
    """Late EOF from an OBSOLETE spawn's reader (e.g. a replacement that
    timed out on ready and was terminated) must NOT mark the CURRENT healthy
    child dead, fail its RPCs/streams, or kick a respawn."""
    proxy = _make_proxy()
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)

    # Gen 2 is the CURRENT healthy child; gen-1's reader reports late.
    proxy._spawn_gen = 2
    stale_proc = SimpleNamespace(exitcode=-15, is_alive=lambda: False)

    proxy._reader_loop(FakeConn([]), gen=1, proc=stale_proc)

    assert proxy._dead is False          # current child stays alive
    assert proxy._dead_reason == ""
    assert kicks == []                   # no respawn kicked
    proxy.ensure_available()             # no raise

    # Control: a CURRENT-gen EOF still runs the full death handling.
    proxy._reader_loop(FakeConn([]), gen=2, proc=stale_proc)
    assert proxy._dead is True
    assert kicks == [1]


def test_stale_generation_eof_after_respawn_keeps_fresh_child_alive():
    """END-TO-END shape of the incident: gen-A child dies -> respawn brings
    up gen B (ready, _dead/_respawning cleared) -> A's reader delivers its
    DELAYED EOF. B must stay alive and keep serving; no extra respawn."""
    proxy = _make_proxy()
    proxy._spawn_gen = 1  # gen A is the current child
    proxy._reap_dead_child = lambda: True

    def _fake_spawn(ready_timeout):
        # Mirror the real _spawn's generation bump for the fresh child.
        with proxy._state_lock:
            proxy._spawn_gen += 1
        proxy._cmd_parent = FakeChildConn(proxy)

    proxy._spawn = _fake_spawn
    stale_proc = SimpleNamespace(exitcode=-15, is_alive=lambda: False)

    # Gen A dies (current gen -> full death handling + auto-respawn).
    proxy._handle_child_death(
        "resp pipe EOF (child worker exited)", gen=1, proc=stale_proc
    )
    _wait_until(lambda: not proxy._dead and not proxy._respawning)
    assert proxy._spawn_gen == 2  # gen B is current now
    assert proxy._respawn_attempts == 1

    # Gen A's reader delivers a SECOND, delayed EOF (post-termination).
    proxy._handle_child_death(
        "resp pipe EOF (child worker exited)", gen=1, proc=stale_proc
    )

    assert proxy._dead is False          # B stays alive
    assert proxy._respawning is False    # no new respawn kicked
    assert proxy._respawn_attempts == 1
    proxy.ensure_available()             # still serving
    assert proxy._rpc("list_sessions") == "pong"  # B actually serves


def test_stale_generation_ready_frame_ignored():
    """A late `ready` from an obsolete spawn must not overwrite the current
    child's metadata or spuriously satisfy the CURRENT spawn's ready wait."""
    proxy = _make_proxy()
    proxy._spawn_gen = 2
    proxy.model_id = "current-model"
    proxy._ready_event = threading.Event()

    proxy._apply_ready({"model_id": "stale-model"}, gen=1)

    assert proxy.model_id == "current-model"
    assert not proxy._ready_event.is_set()

    # Control: a current-gen ready still applies.
    proxy._apply_ready({"model_id": "fresh-model"}, gen=2)
    assert proxy.model_id == "fresh-model"
    assert proxy._ready_event.is_set()


def test_stale_ready_never_releases_newer_spawn_wait():
    """codex round-3 race: a stale `ready` (gen N-1) delivered while gen N
    is spawning/waiting must NOT release gen N's ready wait. The ready event
    is generation-OWNED: _spawn creates it fresh and assigns it together
    with the gen bump in ONE _state_lock block, each reader sets its own
    CAPTURED event, and _apply_ready validates + sets in one critical
    section (no validate-then-release-then-set gap through the mutable
    self._ready_event alias)."""
    proxy = _make_proxy()
    # Gen 1 is current; its reader captured gen-1's OWNED event at spawn.
    proxy._spawn_gen = 1
    ev1 = threading.Event()
    proxy._ready_event = ev1
    proxy.model_id = "current-model"

    # Gen 2 spawns: fresh OWNED event + gen bump, atomically (mirrors
    # _spawn's single _state_lock block). Gen 2's start()/_respawn_loop
    # wait on ev2 — the LOCAL event — not the mutable alias.
    ev2 = threading.Event()
    with proxy._state_lock:
        proxy._spawn_gen += 1
        proxy._ready_event = ev2

    # The gen-1 reader delivers its late ready frame with ITS OWN captured
    # event (the post-fix reader contract) while gen 2 is still waiting.
    proxy._apply_ready({"model_id": "stale-model"}, gen=1, ready_event=ev1)

    assert not ev2.is_set()                  # gen 2's wait does NOT release
    assert not ev1.is_set()                  # stale gen rejected outright
    assert proxy.model_id == "current-model"  # no metadata overwrite

    # Gen 2's OWN ready still releases its wait and applies metadata.
    proxy._apply_ready({"model_id": "fresh-model"}, gen=2, ready_event=ev2)
    assert ev2.is_set()
    assert proxy.model_id == "fresh-model"


def test_reader_loop_stale_ready_sets_neither_event():
    """Same race driven through the reader loop itself: an obsolete spawn's
    reader (gen 1, its own captured event) delivers a late ready frame and
    then EOFs while gen 2 is current. Neither gen 2's event (would falsely
    release gen 2's wait) nor the stale reader's own event is set, metadata
    is untouched, and the stale EOF stays ignored."""
    proxy = _make_proxy()
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)
    proxy._spawn_gen = 2
    ev_current = threading.Event()
    proxy._ready_event = ev_current
    proxy.model_id = "current-model"

    ev_stale = threading.Event()
    stale_proc = SimpleNamespace(exitcode=-15, is_alive=lambda: False)
    proxy._reader_loop(
        FakeConn([{"type": "ready", "model_id": "stale-model"}]),
        gen=1, proc=stale_proc, ready_event=ev_stale,
    )

    assert not ev_current.is_set()           # gen 2's wait untouched
    assert not ev_stale.is_set()
    assert proxy.model_id == "current-model"
    assert proxy._dead is False              # stale EOF ignored too
    assert kicks == []

    # Control: the CURRENT gen's reader delivering ready sets ITS captured
    # event (which _spawn also waits on locally).
    proxy._kick_respawn = lambda: None
    proxy._reader_loop(
        FakeConn([{"type": "ready", "model_id": "fresh-model"}]),
        gen=2, proc=stale_proc, ready_event=ev_current,
    )
    assert ev_current.is_set()
    assert proxy.model_id == "fresh-model"


# --- replacement dies after ready, before the loop clears state ------------------


def test_replacement_dies_after_ready_before_state_clear_counts_failed():
    """codex round-2 race: the respawned child sends `ready`, then dies
    BEFORE the respawn loop clears _dead/_respawning. Its reader's death
    callback lands while _dead is still True — the old early-return swallowed
    it and the loop then cleared _dead, reporting a DEAD child alive (with no
    reader left to ever notice). Now: the death is RECORDED, the loop counts
    that attempt as FAILED (proxy never reports alive in between), and a
    subsequent successful spawn recovers."""
    proxy = _make_proxy()
    proxy._dead = True
    proxy._dead_reason = "resp pipe EOF (child worker exited)"
    proxy._reap_dead_child = lambda: True

    spawn_calls = []
    second_attempt_started = threading.Event()
    release_second_attempt = threading.Event()
    dead_proc = SimpleNamespace(exitcode=-6, is_alive=lambda: False)

    def _fake_spawn(ready_timeout):
        # Mirror the real _spawn's generation bump for the fresh child.
        with proxy._state_lock:
            proxy._spawn_gen += 1
            gen = proxy._spawn_gen
        spawn_calls.append(gen)
        if len(spawn_calls) == 1:
            # The child reached `ready` (so _spawn returns success), but its
            # reader's EOF fires HERE — i.e. before _spawn returns to the
            # respawn loop, guaranteeing the death callback runs between
            # ready-set and the loop's state-clear while _dead is still True
            # and the gen is CURRENT.
            proxy._handle_child_death(
                "resp pipe EOF (child worker exited)", gen=gen, proc=dead_proc
            )
        else:
            second_attempt_started.set()
            release_second_attempt.wait(timeout=5.0)
            proxy._cmd_parent = FakeChildConn(proxy)

    proxy._spawn = _fake_spawn

    proxy._kick_respawn()

    # Attempt 1 was counted FAILED: the loop moved on to attempt 2 and the
    # proxy NEVER reported alive in between (still dead + respawning).
    assert second_attempt_started.wait(timeout=5.0)
    assert proxy._dead is True
    assert proxy.liveness()["alive"] is False
    assert proxy._respawning is True
    with pytest.raises(EngineRestartingError):
        proxy.ensure_available()

    # Attempt 2 succeeds -> the proxy recovers and actually serves.
    release_second_attempt.set()
    _wait_until(lambda: not proxy._dead and not proxy._respawning)

    assert spawn_calls == [1, 2]
    assert proxy._respawn_attempts == 2
    assert proxy._gen_death_recorded is None  # record consumed, none pending
    assert proxy.liveness()["alive"] is True
    proxy.ensure_available()  # no raise
    assert proxy._rpc("list_sessions") == "pong"  # fresh child serves

    # Control for the OTHER side of the window: a current-gen death arriving
    # AFTER the loop cleared _dead takes the NORMAL death path (marks dead +
    # kicks a fresh respawn) — not the record.
    kicks = []
    proxy._kick_respawn = lambda: kicks.append(1)
    proxy._handle_child_death(
        "resp pipe EOF (child worker exited)", gen=proxy._spawn_gen,
        proc=dead_proc,
    )
    assert proxy._dead is True
    assert kicks == [1]
    assert proxy._gen_death_recorded is None


def test_replacement_dies_after_ready_every_attempt_goes_permanently_dead():
    """Every respawned child sends `ready` then immediately dies before the
    loop's state-clear: each attempt is counted FAILED via the death record,
    and the existing attempt counter drives the proxy permanently dead."""
    proxy = _make_proxy()
    proxy._dead = True
    proxy._reap_dead_child = lambda: True
    dead_proc = SimpleNamespace(exitcode=-6, is_alive=lambda: False)
    spawn_calls = []

    def _dying_spawn(ready_timeout):
        with proxy._state_lock:
            proxy._spawn_gen += 1
            gen = proxy._spawn_gen
        spawn_calls.append(gen)
        proxy._handle_child_death(
            "resp pipe EOF (child worker exited)", gen=gen, proc=dead_proc
        )

    proxy._spawn = _dying_spawn

    proxy._kick_respawn()
    _wait_until(lambda: not proxy._respawning)

    assert spawn_calls == [1, 2, 3]  # bounded retries, all counted failed
    assert proxy._permanently_dead is True
    assert proxy._dead is True  # never declared alive with a dead child
    assert proxy.liveness()["alive"] is False
    with pytest.raises(EngineRestartingError, match="permanently"):
        proxy.ensure_available()


# --- reap: a stuck old worker must never coexist with a fresh one ---------------


def test_reap_stuck_child_escalates_to_kill_and_returns_false():
    """terminate()+join leaves the old child alive -> kill()+second join; if
    it STILL lives, _reap_dead_child returns False and keeps the Process
    object (the caller must not spawn a replacement)."""
    proxy = _make_proxy()
    calls = []
    stuck = SimpleNamespace(
        is_alive=lambda: True,
        terminate=lambda: calls.append("terminate"),
        kill=lambda: calls.append("kill"),
        join=lambda timeout=None: calls.append(("join", timeout)),
        exitcode=None,
    )
    proxy._proc = stuck

    assert proxy._reap_dead_child() is False
    assert calls == [
        "terminate", ("join", 5.0), "kill", ("join", 5.0),
    ]
    assert proxy._proc is stuck  # NOT discarded while still alive


def test_reap_kill_fallback_succeeds():
    """SIGTERM ignored but SIGKILL lands: the reap succeeds and clears
    _proc so the respawn can proceed."""
    proxy = _make_proxy()
    state = {"alive": True}
    proc = SimpleNamespace(
        is_alive=lambda: state["alive"],
        terminate=lambda: None,
        kill=lambda: state.update(alive=False),
        join=lambda timeout=None: None,
        exitcode=None,
    )
    proxy._proc = proc

    assert proxy._reap_dead_child() is True
    assert proxy._proc is None


def test_respawn_refused_while_old_child_unkillable():
    """A stuck old worker (survives terminate+join+kill+join) must REFUSE
    the respawn attempt — never two live workers on the GPU. With every
    attempt refused, the proxy goes permanently dead instead of spawning."""
    proxy = _make_proxy()
    proxy._dead = True

    stuck = SimpleNamespace(
        is_alive=lambda: True,
        terminate=lambda: None,
        kill=lambda: None,
        join=lambda timeout=None: None,
        exitcode=None,
    )
    proxy._proc = stuck

    spawned = []
    proxy._spawn = lambda ready_timeout: spawned.append(1)

    proxy._kick_respawn()
    _wait_until(lambda: not proxy._respawning)

    assert spawned == []                     # NEVER spawned a second worker
    assert proxy._proc is stuck              # stuck proc not discarded
    assert proxy._permanently_dead is True   # attempts exhausted, stays dead
    with pytest.raises(EngineRestartingError, match="permanently"):
        proxy.ensure_available()


# --- shutdown of a freshly-respawned child: second bounded join ------------------


def test_shutdown_spawned_child_second_join_after_terminate():
    """_shutdown_spawned_child must mirror close(): after terminate() it
    joins AGAIN (bounded) so the fresh child's SIGTERM flush isn't cut off
    by the parent exiting first."""
    proxy = _make_proxy()
    proxy._cmd_parent = FakeConn()
    calls = []
    proc = SimpleNamespace(
        join=lambda timeout=None: calls.append(("join", timeout)),
        is_alive=lambda: True,
        terminate=lambda: calls.append("terminate"),
    )
    proxy._proc = proc

    proxy._shutdown_spawned_child()

    assert {"op": "shutdown"} in proxy._cmd_parent.sent
    assert calls == [("join", 10.0), "terminate", ("join", 10.0)]


def test_shutdown_spawned_child_graceful_exit_no_terminate():
    """A fresh child that exits on the shutdown op is never terminated."""
    proxy = _make_proxy()
    proxy._cmd_parent = FakeConn()
    calls = []
    proc = SimpleNamespace(
        join=lambda timeout=None: calls.append(("join", timeout)),
        is_alive=lambda: False,
        terminate=lambda: pytest.fail("graceful child must not be terminated"),
    )
    proxy._proc = proc

    proxy._shutdown_spawned_child()
    assert calls == [("join", 10.0)]


# --- API layer: 503 mapping (hermetic FastAPI app) ------------------------------


def test_engine_restarting_maps_to_503_json():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.server import register_engine_error_handlers

    app = FastAPI()
    register_engine_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise EngineRestartingError("child worker died (resp pipe EOF)")

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/boom")
    assert r.status_code == 503
    body = r.json()
    # BATCH B CONTRACT CHANGE: EngineRestartingError now maps to the canonical
    # {message,type,code} envelope with type "engine_not_ready" (renamed from
    # "engine_restarting"), and the body NO LONGER echoes str(exc) — the
    # "resp pipe EOF" internal text must NOT leak into the response.
    assert body["error"]["message"] == "engine not ready, retry shortly"
    assert body["error"]["type"] == "engine_not_ready"
    assert body["error"]["code"] == "engine_not_ready"
    assert "detail" not in body["error"]
    assert "resp pipe EOF" not in json.dumps(body)
    assert r.headers.get("retry-after") == "30"


def test_engine_unavailable_response_shape():
    from mlx_soloheaven.server import engine_unavailable_response

    resp = engine_unavailable_response(EngineRestartingError("x"))
    assert resp.status_code == 503
    body = json.loads(resp.body)
    # BATCH B CONTRACT CHANGE: engine_not_ready envelope (see above).
    assert body["error"]["message"] == "engine not ready, retry shortly"
    assert body["error"]["type"] == "engine_not_ready"
    assert body["error"]["code"] == "engine_not_ready"


# --- API layer: started SSE streams emit an in-band error frame -----------------


@dataclass
class StubResult:
    """Matches the fields the stream readers touch on a GenerationResult."""
    text: str = ""
    token: int = 0
    status: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None


class DyingStubEngine:
    """Yields one content batch, then dies (EngineRestartingError) — like a
    child worker aborting mid-generation."""

    model_id = "stub"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(enable_thinking=False)

    async def generate_stream_batches_async(self, *args, **kwargs):
        yield [StubResult(text="partial ")]
        raise EngineRestartingError(
            "engine child worker died mid-generation: resp pipe EOF"
        )

    def update_session_messages(self, *args, **kwargs):  # pragma: no cover
        pass


def test_openai_stream_emits_error_frame_then_done_on_death():
    from mlx_soloheaven.api.openai_compat import _stream_completion
    from mlx_soloheaven.api.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="stub",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = asyncio.run(_collect(_stream_completion(request, DyingStubEngine())))

    # The partial content that made it out is still there...
    assert any('"partial ' in c for c in chunks)
    # ...then a proper in-band error frame, then a terminating [DONE].
    # BATCH B CONTRACT CHANGE: the in-band frame now uses type/code
    # "engine_not_ready" (renamed from "engine_restarting") and code is a
    # string (was the int 503). The frame still fires under the SAME
    # mid-stream death condition.
    error_chunks = [c for c in chunks if '"engine_not_ready"' in c]
    assert len(error_chunks) == 1
    err = json.loads(error_chunks[0].removeprefix("data: "))
    assert err["error"]["message"] == "engine not ready, retry shortly"
    assert err["error"]["type"] == "engine_not_ready"
    assert err["error"]["code"] == "engine_not_ready"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_web_chat_stream_emits_error_event_on_death():
    from mlx_soloheaven.api.chat import _stream_chat

    frames = asyncio.run(
        _collect(
            _stream_chat("s1", [{"role": "user", "content": "hi"}],
                         DyingStubEngine())
        )
    )
    payloads = [
        json.loads(f.removeprefix("data: "))
        for f in frames
        if f.startswith("data: ")
    ]
    errors = [p for p in payloads if p.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error"] == "engine restarting, retry shortly"
    # The error event is the LAST frame — the stream closes cleanly after it.
    assert payloads[-1]["type"] == "error"


# --- admin: liveness exposure ----------------------------------------------------


def test_admin_models_overview_exposes_liveness_while_dead():
    from mlx_soloheaven.api import admin

    proxy = _make_proxy()
    proxy.model_id = "stub-model"
    # ensure_available raises on the respawning check (before the dead check)
    # so no real respawn is ever kicked from session_stats here.
    proxy._dead = True
    proxy._respawning = True
    proxy._respawn_attempts = 2
    proxy.cfg = SimpleNamespace(
        model_path="/tmp/x",
        default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
        default_top_k=0, default_repetition_penalty=1.0,
        default_max_tokens=100,
        enable_thinking=False, thinking_budget=0,
        think_end_token=-1, think_start_token=-1,
        memory_budget_gb=1, disk_budget_gb=1,
    )

    old_engines, old_default = admin._engines, admin._default_engine
    try:
        admin.set_engines({"stub-model": proxy}, proxy)
        out = asyncio.run(admin.models_overview())
    finally:
        admin.set_engines(old_engines, old_default)

    entry = out["models"][0]
    assert entry["engine"]["alive"] is False
    assert entry["engine"]["respawning"] is True
    assert entry["engine"]["respawn_attempts"] == 2
    # session_stats RPC failed fast (dead child) — page still renders.
    assert entry["sessions"] is None


# --- API layer: availability preflight BEFORE any DB mutation --------------------


class _RecordingStubDB:
    """Async stand-in for storage.database that RECORDS every mutation.

    Reads return canned data rich enough to get each endpoint past its
    guards and to its (now-preflighted) engine touchpoint; every mutating
    call is appended to ``self.writes`` so tests can assert a dead engine
    caused ZERO DB writes."""

    def __init__(self, messages=None):
        self.writes: list[tuple] = []
        self._messages = messages or []

    # reads
    async def get_session(self, session_id):
        return {
            "id": session_id,
            "system_prompt": "",
            "title": "T",
            "context_window_limit": 100000,
        }

    async def get_messages(self, session_id, limit=None):
        return list(self._messages)

    async def get_session_total_tokens(self, session_id):
        return 0

    async def get_message_count(self, session_id):
        return len(self._messages)

    async def count_messages_after_last_compaction(self, session_id):
        return 0

    # writes (must NOT happen when the engine is dead)
    async def add_message(self, *a, **k):
        self.writes.append(("add_message", a))

    async def create_session(self, *a, **k):
        self.writes.append(("create_session", a))
        return {"id": "branch-id"}

    async def delete_last_message(self, *a, **k):
        self.writes.append(("delete_last_message", a))

    async def update_session(self, *a, **k):
        self.writes.append(("update_session", a))

    async def update_session_tokens(self, *a, **k):
        self.writes.append(("update_session_tokens", a))

    async def record_compaction(self, *a, **k):
        self.writes.append(("record_compaction", a))


class _DeadPreflightEngine:
    """Engine whose availability preflight fails (child dead/respawning).
    Any OTHER engine touch means a preflight was missing — fail loudly."""

    model_id = "stub"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(
            enable_thinking=False,
            default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
            default_top_k=0, default_repetition_penalty=1.0,
            thinking_budget=0, default_max_tokens=100,
        )

    def ensure_available(self):
        raise EngineRestartingError(
            "engine child worker died (resp pipe EOF) — respawn in progress"
        )

    def __getattr__(self, name):  # any un-preflighted engine call
        pytest.fail(f"engine.{name} reached without availability preflight")


def _chat_test_client(monkeypatch, stub_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    monkeypatch.setattr(chat_mod, "db", stub_db)
    monkeypatch.setattr(chat_mod, "engine", _DeadPreflightEngine())
    monkeypatch.setattr(chat_mod, "_engines", {})

    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(chat_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def test_chat_dead_engine_503_before_user_message_persisted(monkeypatch):
    """Dead engine -> POST chat answers 503 AND the user message is NOT
    written to the DB (a retry after the 503 must not duplicate it)."""
    stub_db = _RecordingStubDB()
    client = _chat_test_client(monkeypatch, stub_db)

    r = client.post("/api/sessions/s1/chat", json={"content": "hi"})

    assert r.status_code == 503
    # BATCH B CONTRACT CHANGE: EngineRestartingError -> type "engine_not_ready".
    assert r.json()["error"]["type"] == "engine_not_ready"
    assert stub_db.writes == []  # NO user message persisted


def test_branch_dead_engine_503_no_db_mutation(monkeypatch):
    stub_db = _RecordingStubDB(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    client = _chat_test_client(monkeypatch, stub_db)

    r = client.post("/api/sessions/s1/branch", json={"turn": 2})

    assert r.status_code == 503
    assert stub_db.writes == []  # no orphan branch session / copied messages


def test_delete_last_dead_engine_503_no_db_mutation(monkeypatch):
    stub_db = _RecordingStubDB(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    client = _chat_test_client(monkeypatch, stub_db)

    r = client.post("/api/sessions/s1/delete-last")

    assert r.status_code == 503
    assert stub_db.writes == []  # no messages deleted


def test_regenerate_dead_engine_503_no_db_mutation(monkeypatch):
    stub_db = _RecordingStubDB(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    client = _chat_test_client(monkeypatch, stub_db)

    r = client.post("/api/sessions/s1/regenerate")

    assert r.status_code == 503
    assert stub_db.writes == []  # assistant/user messages NOT deleted


# --- compaction endpoint: preflight 503 + mid-stream error frame -----------------


def test_compaction_dead_engine_503_preflight(monkeypatch):
    """The original 200+dead-stream bug in compact_session: a dead engine
    must now answer a real HTTP 503 BEFORE the StreamingResponse starts."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import compaction as compaction_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    stub_db = _RecordingStubDB()
    monkeypatch.setattr(compaction_mod, "db", stub_db)
    monkeypatch.setattr(compaction_mod, "engine", _DeadPreflightEngine())

    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(compaction_mod.router)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/api/sessions/s1/compact", json={})

    assert r.status_code == 503
    # BATCH B CONTRACT CHANGE: EngineRestartingError -> type "engine_not_ready".
    assert r.json()["error"]["type"] == "engine_not_ready"
    assert stub_db.writes == []


class _DyingCompactionEngine:
    """Streams part of the summary, then dies (EngineRestartingError) — like
    the child worker aborting mid-summary."""

    model_id = "stub"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(enable_thinking=False)

    async def generate_stream_async(self, *args, **kwargs):
        yield SimpleNamespace(text="partial summary ")
        raise EngineRestartingError(
            "engine child worker died mid-generation: resp pipe EOF"
        )

    def compact_session(self, *a, **k):  # pragma: no cover
        pytest.fail("finalization must not run after a mid-stream death")


def test_compaction_stream_emits_error_event_on_mid_stream_death(monkeypatch):
    """Child dies while the summary streams: the SSE stream must terminate
    with an in-band error frame (mirroring the chat/openai wrappers) and the
    finalization (summary insert / compact RPC / record) must NOT run."""
    from mlx_soloheaven.api import compaction as compaction_mod

    # Enough history to pass summarize()'s minimum-message guards.
    stub_db = _RecordingStubDB(
        messages=[
            {"role": "user", "content": f"m{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"m{i}"}
            for i in range(8)
        ]
    )
    monkeypatch.setattr(compaction_mod, "db", stub_db)
    monkeypatch.setattr(compaction_mod, "engine", _DyingCompactionEngine())

    req = compaction_mod.CompactionRequest(keep_recent_turns=1)
    frames = asyncio.run(
        _collect(
            compaction_mod._stream_compact(
                "s1", {"system_prompt": "", "total_prompt_tokens": 100}, req
            )
        )
    )
    payloads = [
        json.loads(f.removeprefix("data: "))
        for f in frames
        if f.startswith("data: ")
    ]

    # The partial summary text that made it out is still streamed...
    assert any(p.get("type") == "text" for p in payloads)
    # ...then exactly one in-band error frame terminates the stream.
    errors = [p for p in payloads if p.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error"] == "engine restarting, retry shortly"
    assert payloads[-1]["type"] == "error"
    # Finalization never ran: no summary insert / token update / record.
    assert stub_db.writes == []
