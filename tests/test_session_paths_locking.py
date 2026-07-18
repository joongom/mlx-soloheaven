"""U15 — in-process session list/delete/clear/stats paths take the engine lock.

These paths used to read/mutate ``_sessions`` / ``_base_caches`` with NO
lock while a concurrent generation mutates the same structures in place
(rebuild paths already locked). Now:

  - READ paths (list_sessions / get_session / session_stats /
    base_cache_stats / cache_overview) take the lock with a BOUNDED acquire
    and raise ``EngineBusyError`` when a generation holds it — callers
    degrade instead of hanging for minutes,
  - MUTATING paths (delete_session / clear_caches) take the lock UNBOUNDED
    (correctness over latency; the U14 to_thread wrappers keep the event
    loop responsive while they wait).

RE-ENTRANCY: the engine lock is a plain (non-reentrant) threading.Lock —
verified here by exercising ``reset()`` (the one internal caller chain:
reset -> clear_caches) under the recording lock: exactly one acquire.

HERMETIC — shell engines via ``MLXEngine.__new__`` (conventions from
tests/test_shutdown_flush.py / test_cache_contract.py).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState
from mlx_soloheaven.engine.types import EngineBusyError


class RecordingLock:
    """Wraps a real threading.Lock and records acquire/release pairs."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquires = 0
        self.releases = 0

    def acquire(self, blocking=True, timeout=-1):
        if timeout is None:
            timeout = -1
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            self.acquires += 1
        return ok

    def release(self):
        self.releases += 1
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _session(sid_tokens=3) -> SessionState:
    return SessionState(
        cache_state=SimpleNamespace(cache=None, token_ids=None),
        messages=[{"role": "user", "content": "u1"}],
        total_cache_tokens=sid_tokens,
        last_used=time.time(),
    )


def _shell_engine(tmp_path) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "shell"
    eng._lock = RecordingLock()
    eng._dirty_lock = threading.Lock()
    eng._dirty_sessions = set()
    eng._sessions = {"s1": _session()}
    eng._anon_minted_ids = set()
    eng._base_caches = {}
    eng.cfg = SimpleNamespace(
        cache_dir=str(tmp_path), memory_budget_gb=0, mlx_cache_limit_gb=0,
        enable_thinking=False,
    )
    eng.cache_manager = SimpleNamespace(
        _estimate_cache_size=lambda cache: 0,
        _memory_usage_gb=lambda: 0.0,
        stats=lambda: {},
        memory_caches={},
        disk_index={},
    )
    return eng


# --- reads acquire + release the lock -------------------------------------------


@pytest.mark.parametrize("call", [
    lambda e: e.list_sessions(),
    lambda e: e.get_session("s1"),
    lambda e: e.session_stats(),
    lambda e: e.base_cache_stats(),
    lambda e: e.cache_overview(),
])
def test_read_paths_take_and_release_lock(tmp_path, call):
    eng = _shell_engine(tmp_path)
    call(eng)
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1
    assert not eng._lock.locked()


def test_read_results_unchanged_shape(tmp_path):
    eng = _shell_engine(tmp_path)
    assert eng.list_sessions()[0]["session_id"] == "s1"
    assert eng.get_session("s1")["cache_tokens"] == 3
    assert eng.get_session("nope") is None
    assert eng.session_stats()["active_sessions"] == 1
    assert eng.base_cache_stats() == []
    ov = eng.cache_overview()
    assert ov["session_count"] == 1
    assert ov["model_id"] == "shell"


# --- reads are BOUNDED: busy engine raises EngineBusyError -----------------------


@pytest.mark.parametrize("call", [
    lambda e: e.list_sessions(),
    lambda e: e.get_session("s1"),
    lambda e: e.session_stats(),
    lambda e: e.base_cache_stats(),
    lambda e: e.cache_overview(),
])
def test_read_paths_raise_busy_while_lock_held(tmp_path, call, monkeypatch):
    monkeypatch.setattr(MLXEngine, "_READ_LOCK_TIMEOUT_S", 0.05)
    eng = _shell_engine(tmp_path)
    assert eng._lock.acquire(blocking=False)  # a generation holds the lock
    try:
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError):
            call(eng)
        assert time.monotonic() - t0 < 2.0  # bounded, not the generation's length
    finally:
        eng._lock.release()
    # And once the lock frees, the same read succeeds.
    call(eng)
    assert not eng._lock.locked()


# --- mutating paths lock (unbounded) ---------------------------------------------


def test_delete_session_takes_and_releases_lock(tmp_path):
    eng = _shell_engine(tmp_path)
    eng._dirty_sessions.add("s1")
    assert eng.delete_session("s1") is True
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1
    assert "s1" not in eng._sessions
    assert "s1" not in eng._dirty_sessions
    assert not eng._lock.locked()


def test_clear_caches_takes_and_releases_lock(tmp_path):
    eng = _shell_engine(tmp_path)
    cleared = eng.clear_caches()
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1
    assert cleared["memory_sessions"] == 1
    assert eng._sessions == {}
    assert not eng._lock.locked()


def test_reset_delegates_without_double_lock(tmp_path):
    """reset() -> clear_caches() is the only internal caller chain; on the
    NON-reentrant engine lock a double acquire would deadlock — exactly one
    acquire must happen."""
    eng = _shell_engine(tmp_path)
    eng.reset()
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1


def test_mutating_paths_wait_for_generation(tmp_path):
    """delete_session BLOCKS (unbounded) behind a held lock and completes
    once the generation releases — never a busy error, never a skipped
    mutation."""
    eng = _shell_engine(tmp_path)
    assert eng._lock.acquire(blocking=False)
    done = threading.Event()
    result = {}

    def _delete():
        result["ok"] = eng.delete_session("s1")
        done.set()

    t = threading.Thread(target=_delete, daemon=True)
    t.start()
    time.sleep(0.05)
    assert not done.is_set()  # still waiting behind the generation
    eng._lock.release()
    assert done.wait(timeout=5.0)
    assert result["ok"] is True
    assert "s1" not in eng._sessions


# --- F5: web-chat cache preflight is a bounded, locked engine read ----------------
# (used to live inline in chat.py's async stream: it read/wrote _sessions and
# did disk IO on the event loop with NO lock, racing an in-process generation.
# Round 2: the preflight is METADATA-ONLY — the bounded lock covers only
# in-memory dict lookups, the disk fallback parses the safetensors header with
# plain file IO OUTSIDE the lock, and it never runs mx.load / never writes
# _sessions — the authoritative load stays on the generation thread.)


def _preflight_fp(eng) -> str:
    """The prompt-contract fingerprint session_cache_preflight computes for a
    web-chat request on this engine (tools=None, thinking=cfg.enable_thinking,
    the family suffix revision). A session/disk file must carry it to be
    reported as a HIT under the U18-round-3 fingerprint gate."""
    return eng._prompt_fingerprint(
        eng._canonical_tools(None),
        bool(eng.cfg.enable_thinking),
        template_rev=eng._suffix_template_rev(),
    )


def _preflight_engine(tmp_path, *, match=True, with_cache=True):
    eng = _shell_engine(tmp_path)
    eng._has_disk_cache = lambda sid: False
    eng._load_session_from_disk = lambda sid: None
    eng._messages_match = lambda stored, incoming, **kw: match
    if with_cache:
        eng._sessions["s1"].cache_state = SimpleNamespace(
            cache=object(), token_ids=[1, 2, 3],
        )
        # U18 round 3: a genuinely reusable session carries the matching
        # prompt-contract fingerprint (a live session built by generation
        # always stamps one) so the new fingerprint gate reports a HIT.
        eng._sessions["s1"].prompt_fingerprint = _preflight_fp(eng)
    return eng


def _write_session_cache_file(tmp_path, session_id: str, metadata: dict):
    """Hand-craft a minimal safetensors file with plain IO — the 8-byte LE
    header length + JSON header carrying ``__metadata__``, exactly the
    layout mx.save_safetensors writes. No MLX involved."""
    header = json.dumps({"__metadata__": metadata}).encode()
    path = tmp_path / f"session_{session_id}.safetensors"
    with open(path, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
    return path


def test_preflight_takes_and_releases_lock(tmp_path):
    eng = _preflight_engine(tmp_path)
    eng.session_cache_preflight("s1", list(eng._sessions["s1"].messages))
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1
    assert not eng._lock.locked()


def test_preflight_busy_raises_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(MLXEngine, "_READ_LOCK_TIMEOUT_S", 0.05)
    eng = _preflight_engine(tmp_path)
    assert eng._lock.acquire(blocking=False)  # a generation holds the lock
    try:
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError):
            eng.session_cache_preflight("s1", [])
        assert time.monotonic() - t0 < 2.0
    finally:
        eng._lock.release()
    # Once the lock frees, the same preflight succeeds.
    eng.session_cache_preflight("s1", list(eng._sessions["s1"].messages))
    assert not eng._lock.locked()


def test_preflight_hit_shape(tmp_path):
    eng = _preflight_engine(tmp_path)
    msgs = list(eng._sessions["s1"].messages) + [{"role": "user", "content": "u2"}]
    out = eng.session_cache_preflight("s1", msgs)
    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"
    assert out["cache_info"]["source"] == "memory"
    assert out["cache_info"]["cached_tokens"] == 3


def test_preflight_rebuild_when_cache_dead(tmp_path):
    eng = _preflight_engine(tmp_path, with_cache=False)
    out = eng.session_cache_preflight("s1", list(eng._sessions["s1"].messages))
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"


def test_preflight_miss_when_history_diverged(tmp_path):
    eng = _preflight_engine(tmp_path, match=False)
    out = eng.session_cache_preflight("s1", [{"role": "user", "content": "other"}])
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_miss"


def test_preflight_new_session(tmp_path):
    eng = _preflight_engine(tmp_path)
    out = eng.session_cache_preflight("nope", [{"role": "user", "content": "x"}])
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "none"


def test_preflight_disk_is_metadata_only_never_loads_cache(tmp_path, monkeypatch):
    """F5 round 2 (VLM thread ownership): the disk fallback parses ONLY the
    safetensors header — it must NOT run the MLX disk load (monkeypatched
    to explode), must NOT construct KV caches, and must NOT publish into
    ``_sessions``. Pre-fix it ran _load_session_from_disk (mx.load) on the
    engine-read executor thread and wrote _sessions — generation then
    consumed those arrays on the engine's owning thread."""
    import mlx_soloheaven.engine.session_cache_mixin as cache_module

    eng = _preflight_engine(tmp_path)
    stored = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    _write_session_cache_file(tmp_path, "s2", {
        "messages": json.dumps(stored),
        "total_cache_tokens": "42",
        "prompt_fingerprint": _preflight_fp(eng),
    })
    eng._has_disk_cache = lambda sid: sid == "s2"
    eng._load_session_from_disk = lambda sid: pytest.fail(
        "metadata-only preflight must never run the MLX disk load"
    )
    monkeypatch.setattr(
        cache_module, "load_prompt_cache",
        lambda *a, **k: pytest.fail("preflight must never touch mx.load"),
    )

    out = eng.session_cache_preflight(
        "s2", stored + [{"role": "user", "content": "u2"}]
    )
    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"
    assert out["cache_info"]["source"] == "disk"
    assert out["cache_info"]["cached_tokens"] == 42
    assert "s2" not in eng._sessions  # advisory only: never publishes state


def test_preflight_disk_io_runs_outside_engine_lock(tmp_path):
    """The header read must NOT hold the engine/GPU lock — the 10s read
    bound used to cover only the acquire while the critical section did
    arbitrarily large disk IO. The lock is still taken (bounded, once) for
    the in-memory phase."""
    eng = _preflight_engine(tmp_path)
    eng._has_disk_cache = lambda sid: sid == "s2"
    observed = {}

    def _meta(path):
        observed["locked_during_disk_io"] = eng._lock.locked()
        return {
            "messages": json.dumps([{"role": "user", "content": "u1"}]),
            "total_cache_tokens": "3",
            "prompt_fingerprint": _preflight_fp(eng),
        }

    eng._read_safetensors_metadata = _meta
    out = eng.session_cache_preflight("s2", [{"role": "user", "content": "u1"}])
    assert observed["locked_during_disk_io"] is False
    assert out["cache_hit"] is True
    assert eng._lock.acquires == 1
    assert eng._lock.releases == 1
    assert not eng._lock.locked()


def test_preflight_disk_metadata_mismatch_reports_miss(tmp_path):
    eng = _preflight_engine(tmp_path, match=False)
    _write_session_cache_file(tmp_path, "s2", {
        "messages": json.dumps([{"role": "user", "content": "other"}]),
        "total_cache_tokens": "3",
    })
    eng._has_disk_cache = lambda sid: sid == "s2"
    out = eng.session_cache_preflight("s2", [{"role": "user", "content": "x"}])
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_miss"
    assert "s2" not in eng._sessions


def test_preflight_disk_metadata_unreadable_degrades(tmp_path):
    """A corrupt header is an advisory 'unknown' — never an exception and
    never an MLX load attempt (the length prefix here claims a bogus giant
    header)."""
    eng = _preflight_engine(tmp_path)
    path = tmp_path / "session_s2.safetensors"
    path.write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xffgarbage")
    eng._has_disk_cache = lambda sid: sid == "s2"
    eng._load_session_from_disk = lambda sid: pytest.fail("must not MLX-load")
    out = eng.session_cache_preflight("s2", [{"role": "user", "content": "x"}])
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "none"
    assert "unreadable" in out["cache_info"]["detail"]


def test_read_safetensors_metadata_parses_and_fails_closed(tmp_path):
    """The plain-IO header parser: happy path + every malformed shape
    returns None (missing file, truncated prefix, non-dict header, no
    __metadata__ block)."""
    p = _write_session_cache_file(tmp_path, "meta", {"k": "v"})
    assert MLXEngine._read_safetensors_metadata(str(p)) == {"k": "v"}

    assert MLXEngine._read_safetensors_metadata(
        str(tmp_path / "nope.safetensors")
    ) is None

    trunc = tmp_path / "trunc.safetensors"
    trunc.write_bytes(b"\x04\x00")
    assert MLXEngine._read_safetensors_metadata(str(trunc)) is None

    nodict = tmp_path / "nodict.safetensors"
    body = json.dumps(["not", "a", "dict"]).encode()
    nodict.write_bytes(len(body).to_bytes(8, "little") + body)
    assert MLXEngine._read_safetensors_metadata(str(nodict)) is None

    nometa = tmp_path / "nometa.safetensors"
    body = json.dumps({"tensor": {"dtype": "F32"}}).encode()
    nometa.write_bytes(len(body).to_bytes(8, "little") + body)
    assert MLXEngine._read_safetensors_metadata(str(nometa)) is None

    # Round 3 finding 6: a TRUNCATED file whose available bytes still parse
    # as JSON (the length prefix promises more than the file holds) must
    # fail closed — pre-fix the short read was accepted and the preflight
    # reported a disk hit from a header it only partially saw.
    short = tmp_path / "short.safetensors"
    body = json.dumps({"__metadata__": {"k": "v"}}).encode()
    short.write_bytes((len(body) + 10).to_bytes(8, "little") + body)
    assert MLXEngine._read_safetensors_metadata(str(short)) is None


# --- round 3 finding 2: engine-side shutdown gate ---------------------------------
# (Future.cancel() cannot stop an executor worker that STARTED but is
# BLOCKED on the engine lock behind a generation — the executors' bounded
# quiesce wait returns with it "running", the server flushes, and the
# straggler then acquires the lock, mutates state and marks it dirty with
# nothing left to flush. begin_shutdown() closes an engine-side gate that
# every mutating lock acquisition checks IMMEDIATELY after acquiring.)


MUTATING_CALLS = [
    ("delete_session", lambda e: e.delete_session("s1")),
    ("clear_caches", lambda e: e.clear_caches()),
    ("reset", lambda e: e.reset()),
    ("compact_session",
     lambda e: e.compact_session("s1", [{"role": "user", "content": "x"}])),
    ("_rebuild_session",
     lambda e: e._rebuild_session("s1", [{"role": "user", "content": "x"}])),
]


@pytest.mark.parametrize("name,call", MUTATING_CALLS)
def test_shutdown_gate_rejects_mutations_without_mutating(tmp_path, name, call):
    """After begin_shutdown(), every mutating path aborts with
    EngineBusyError IMMEDIATELY after acquiring the lock — no session /
    dirty-set / base-cache state is touched, and the lock is released."""
    eng = _shell_engine(tmp_path)
    eng._dirty_sessions.add("s1")
    eng.begin_shutdown()

    with pytest.raises(EngineBusyError, match="shutting down"):
        call(eng)

    assert not eng._lock.locked()
    assert eng._lock.acquires == eng._lock.releases  # acquired, then freed
    assert "s1" in eng._sessions  # nothing deleted/cleared/rebuilt
    assert eng._dirty_sessions == {"s1"}  # nothing re-marked or drained


def test_shutdown_gate_straggler_blocked_on_lock_becomes_noop(tmp_path):
    """THE round-3 repro shape: a mutation is BLOCKED on the engine lock
    behind a generation when shutdown begins. The gate is set and the
    'flush' happens while it waits; when the generation finally releases
    the lock (post-flush), the straggler must abort WITHOUT mutating —
    pre-fix it deleted the session and could re-dirty state after the last
    flush."""
    eng = _shell_engine(tmp_path)
    assert eng._lock.acquire(blocking=False)  # a generation holds the lock

    started = threading.Event()
    outcome: dict = {}

    def _straggler():
        started.set()
        try:
            outcome["result"] = eng.delete_session("s1")
        except EngineBusyError as e:
            outcome["rejected"] = e

    t = threading.Thread(target=_straggler, daemon=True)
    t.start()
    assert started.wait(timeout=5.0)
    time.sleep(0.05)  # let it block on the engine lock
    assert not outcome  # still waiting behind the generation

    eng.begin_shutdown()  # shutdown hook: gate BEFORE the flush
    # <-- the shutdown flush runs here, while the straggler still waits -->
    eng._lock.release()  # the generation ends AFTER the flush

    t.join(timeout=5.0)
    assert not t.is_alive()
    assert "rejected" in outcome  # aborted, never mutated
    assert "s1" in eng._sessions  # the post-flush mutation never happened
    assert not eng._lock.locked()


def test_shutdown_gate_leaves_flush_exempt(tmp_path):
    """The flush itself MUST still run after the gate closes (it is the
    whole point of the ordering): _flush_dirty_sessions drains the dirty
    set normally with the gate set."""
    eng = _shell_engine(tmp_path)
    saved: list = []
    eng._save_session_to_disk = lambda sid, session: saved.append(sid) or True
    eng._dirty_sessions.add("s1")

    eng.begin_shutdown()
    eng.begin_shutdown()  # idempotent
    eng._flush_dirty_sessions()

    assert saved == ["s1"]
    assert eng._dirty_sessions == set()


def test_shutdown_gate_reads_stay_live(tmp_path):
    """Reads are deliberately NOT gated — /health keeps answering through a
    graceful shutdown (reads mutate nothing)."""
    eng = _shell_engine(tmp_path)
    eng.begin_shutdown()
    assert eng.list_sessions()[0]["session_id"] == "s1"
    assert eng.session_stats()["active_sessions"] == 1


def test_shutdown_gate_blocks_new_generation_lock_acquire(tmp_path):
    """A generation QUEUED behind the engine lock when shutdown begins must
    not start post-flush either (it would advance session state + mark it
    dirty): the gate fires immediately after the acquire, the lock is
    released, and EngineBusyError surfaces at first iteration."""
    eng = _shell_engine(tmp_path)
    eng.begin_shutdown()
    with pytest.raises(EngineBusyError, match="shutting down"):
        list(eng.generate_stream([{"role": "user", "content": "x"}]))
    assert not eng._lock.locked()


def test_shutdown_gate_spares_generation_already_holding_lock(tmp_path):
    """The gate stops NEW acquisitions only: begin_shutdown() while the
    lock is HELD (an in-flight generation) neither raises nor steals the
    lock — Uvicorn's graceful shutdown owns live connections."""
    eng = _shell_engine(tmp_path)
    assert eng._lock.acquire(blocking=False)
    try:
        eng.begin_shutdown()  # must not raise or touch the held lock
        assert eng._lock.locked()
    finally:
        eng._lock.release()


# --- round 5 finding 3b: the gate covers the WHOLE wrapper op (preludes too) ------
# branch_from_turn / prepare_regenerate / truncate_session used to inspect,
# disk-load and PUBLISH _sessions in their prelude BEFORE reaching the gated
# _rebuild_session — post-shutdown calls mutated _sessions outside the gate.


def _preluded_shell(tmp_path):
    """Shell whose disk-reload prelude is OBSERVABLE: the target session is
    NOT resident but has a disk copy, so a pre-fix prelude would publish the
    loaded session into _sessions before hitting the inner gate."""
    eng = _shell_engine(tmp_path)
    loads: list = []
    eng._has_disk_cache = lambda sid: True

    def _load(sid):
        loads.append(sid)
        s = _session()
        s.messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        return s

    eng._load_session_from_disk = _load
    return eng, loads


WRAPPER_CALLS = [
    ("branch_from_turn",
     lambda e: e.branch_from_turn("evicted", "new-id", 1)),
    ("prepare_regenerate", lambda e: e.prepare_regenerate("evicted")),
    ("truncate_session", lambda e: e.truncate_session("evicted", 1)),
    ("compact_session",
     lambda e: e.compact_session("evicted", [{"role": "user", "content": "x"}])),
]


@pytest.mark.parametrize("name,call", WRAPPER_CALLS)
def test_shutdown_gate_covers_wrapper_preludes(tmp_path, name, call):
    """codex round-5 3b repro: each wrapper called AFTER begin_shutdown is
    rejected WITHOUT any prelude side effect — no disk load, no _sessions
    publication, no dirty-mark. Pre-fix the prelude ran ungated and
    published the disk-loaded session before the inner gate fired."""
    eng, loads = _preluded_shell(tmp_path)
    eng.begin_shutdown()

    with pytest.raises(EngineBusyError, match="shutting down"):
        call(eng)

    assert loads == []  # the prelude never ran
    assert set(eng._sessions) == {"s1"}  # nothing published
    assert eng._dirty_sessions == set()  # nothing marked
    assert not eng._lock.locked()
    assert eng._lock.acquires == eng._lock.releases


@pytest.mark.parametrize("name,call", WRAPPER_CALLS[:3])
def test_wrapper_disk_resume_still_works_inside_gate(tmp_path, name, call):
    """Sanity for the 3b restructure: with the server NOT shutting down,
    the wrappers still disk-resume the evicted session (the prelude runs —
    now inside the gate) and reach the rebuild body."""
    eng, loads = _preluded_shell(tmp_path)
    rebuilt: list = []
    eng._rebuild_session_locked = (
        lambda sid, msgs, tools=None, thinking=True:
            rebuilt.append(sid) or {"status": "ok", "cached_tokens": 1}
    )

    result = call(eng)

    assert loads == ["evicted"]  # prelude ran (disk resume)
    assert "evicted" in eng._sessions  # published under the gate
    assert result.get("status") == "ok"
    assert rebuilt  # reached the rebuild body
    assert not eng._lock.locked()


# --- round 5 finding 3a: in-flight mutation tracking + self-flush-on-exit ---------


def _enter_mutation(eng, what="test-op"):
    """Run a controllable mutation body inside _mutate_locked on a thread.
    Returns (thread, entered_event, release_event, outcome_dict)."""
    entered = threading.Event()
    release = threading.Event()
    outcome: dict = {}

    def _body():
        try:
            with eng._mutate_locked(what):
                entered.set()
                assert release.wait(timeout=10.0)
                # The mutation publishes + marks dirty INSIDE its critical
                # section (the compaction shape codex described).
                eng._sessions["s-new"] = _session()
                eng._mark_dirty("s-new")
            outcome["ok"] = True
        except EngineBusyError as e:
            outcome["rejected"] = e

    t = threading.Thread(target=_body, daemon=True)
    t.start()
    return t, entered, release, outcome


def test_mutations_in_flight_counter_and_bounded_wait(tmp_path):
    """wait_mutations_settled reports an entered mutation within its bound
    and settles to 0 once it exits."""
    eng = _shell_engine(tmp_path)
    assert eng.wait_mutations_settled(0.05) == 0  # nothing in flight

    t, entered, release, outcome = _enter_mutation(eng)
    assert entered.wait(timeout=5.0)
    assert eng.wait_mutations_settled(0.15) == 1  # bounded, reports truthfully

    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert outcome.get("ok") is True
    assert eng.wait_mutations_settled(1.0) == 0


def test_straggler_mutation_self_flushes_on_exit_during_shutdown(tmp_path):
    """codex round-5 3a repro: a mutation is INSIDE its critical section
    when the gate closes; the shutdown flush's bounded lock acquire times
    out (round 7, 2b: it now drains NOTHING — pre-2b it drained a snapshot
    and re-marked it); the straggler then publishes + marks dirty AFTER
    that final flush. Its _mutate_locked exit path must self-flush —
    saving BOTH the still-marked ids and its own — so nothing dirty
    survives un-persisted."""
    eng = _shell_engine(tmp_path)
    saved: list = []
    eng._save_session_to_disk = lambda sid, session: saved.append(sid) or True
    # A session already dirty BEFORE shutdown (the flush's bounded lock
    # acquire will time out against the straggler and leave it marked).
    eng._dirty_sessions.add("s1")

    t, entered, release, outcome = _enter_mutation(eng, "compaction-ish")
    assert entered.wait(timeout=5.0)

    # Server shutdown while the mutation holds the engine lock:
    eng.begin_shutdown()
    assert eng.wait_mutations_settled(0.1) == 1  # straggler reported

    import mlx_soloheaven.engine.mlx_engine as mlx_engine_module
    orig_engines = mlx_engine_module.MLXEngine._all_engines
    mlx_engine_module.MLXEngine._all_engines = [eng]
    try:
        # The final flush cannot get the lock: it skips, draining nothing.
        MLXEngine._flush_all_on_shutdown(lock_timeout=0.1)
    finally:
        mlx_engine_module.MLXEngine._all_engines = orig_engines
    assert saved == []  # nothing was flushable yet
    assert eng._dirty_sessions == {"s1"}  # never drained (2b: lock first)

    # The straggler finishes AFTER the final flush: publish + mark dirty,
    # then exit — the exit path self-flushes everything.
    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert outcome.get("ok") is True

    assert set(saved) == {"s1", "s-new"}  # re-marked AND fresh both saved
    assert eng._dirty_sessions == set()  # nothing dirty survives
    assert not eng._lock.locked()


# --- codex round 7 finding 2: self-flush vs final-flush overlap -------------------
#
# Three layers, individually testable:
#   (a) _mutations_in_flight decrements only AFTER the self-flush completes,
#       so wait_mutations_settled can never report zero while a straggler is
#       still mid-save under the engine lock;
#   (b) the final flush acquires the engine lock BEFORE draining the dirty
#       set — a lock-acquire timeout re-marks NOTHING (it never drained),
#       killing the stale re-mark window;
#   (c) the self-flush RESCANS (bounded) until the dirty set is empty at
#       exit, so ids marked while it was mid-save are still saved.


def test_wait_mutations_settled_covers_self_flush_in_progress(tmp_path):
    """Layer (a): while the straggler's self-flush is MID-SAVE (still
    holding the engine lock, doing disk IO), wait_mutations_settled must
    still report it in flight. Pre-fix the counter was decremented BEFORE
    _self_flush_on_shutdown_exit ran, so this returned 0 and the server
    proceeded to the final flush against a mid-save engine."""
    eng = _shell_engine(tmp_path)
    in_save = threading.Event()
    finish_save = threading.Event()
    saved: list = []

    def _slow_save(sid, session):
        in_save.set()
        assert finish_save.wait(timeout=10.0)
        saved.append(sid)
        return True

    eng._save_session_to_disk = _slow_save

    t, entered, release, outcome = _enter_mutation(eng)
    assert entered.wait(timeout=5.0)
    eng.begin_shutdown()
    release.set()
    assert in_save.wait(timeout=5.0)  # straggler is mid-self-flush save

    # The mutation is still in flight — its self-flush has not completed.
    assert eng.wait_mutations_settled(0.1) == 1

    finish_save.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert outcome.get("ok") is True
    assert eng.wait_mutations_settled(2.0) == 0  # settles once save is done
    assert "s-new" in saved
    assert eng._dirty_sessions == set()


def test_final_flush_timeout_during_self_flush_strands_nothing(tmp_path):
    """CODEX ROUND 7, FINDING 2 — the exact overlap: an id is dirty BEFORE
    shutdown; the final flush runs while a straggler holds the engine lock;
    the straggler publishes a NEW dirty session and enters its self-flush,
    whose (slow) save is still in flight when the final flush's bounded
    lock acquire times out. Pre-fix the final flush had ALREADY drained the
    old id and now RE-MARKED it — after the self-flush's single drain — so
    the re-marked id stayed dirty forever (no later flush ever runs) and
    was never persisted. Post-fix (b) the timed-out flush drained nothing
    and (c) the self-flush's rescan owns everything: nothing remains dirty
    and every session is persisted."""
    eng = _shell_engine(tmp_path)
    flush_done = threading.Event()
    saved: list = []

    def _save(sid, session):
        # The FIRST save (straggler's self-flush, engine lock held) stalls
        # until the final flush has timed out — the codex window.
        if not saved:
            assert flush_done.wait(timeout=10.0)
        saved.append(sid)
        return True

    eng._save_session_to_disk = _save
    eng._dirty_sessions.add("s1")  # dirty BEFORE shutdown

    t, entered, release, outcome = _enter_mutation(eng, "compaction-ish")
    assert entered.wait(timeout=5.0)
    eng.begin_shutdown()

    import mlx_soloheaven.engine.mlx_engine as mlx_engine_module
    orig_engines = mlx_engine_module.MLXEngine._all_engines
    mlx_engine_module.MLXEngine._all_engines = [eng]
    flush_thread = threading.Thread(
        target=lambda: MLXEngine._flush_all_on_shutdown(lock_timeout=1.0),
        daemon=True,
    )
    try:
        flush_thread.start()
        # Let the final flush reach its bounded lock acquire, then release
        # the straggler: it publishes s-new, marks it dirty, and its
        # self-flush starts the STALLED save while the final flush is still
        # waiting on the lock. The lock stays held until after the flush
        # thread finished (the save waits on flush_done), so the timeout is
        # deterministic.
        time.sleep(0.2)
        release.set()
        flush_thread.join(timeout=10.0)
        assert not flush_thread.is_alive()  # timed out + skipped
        flush_done.set()
        t.join(timeout=10.0)
        assert not t.is_alive()
    finally:
        mlx_engine_module.MLXEngine._all_engines = orig_engines

    assert outcome.get("ok") is True
    assert eng._dirty_sessions == set()   # NOTHING stranded dirty
    assert set(saved) == {"s1", "s-new"}  # everything persisted


def test_self_flush_rescans_concurrent_remark(tmp_path):
    """Layer (c) in isolation: an id marked dirty by a lock-free marker
    (update_session_messages' touch shape) WHILE the self-flush is mid-save
    must be picked up by a rescan pass before the straggler releases the
    engine lock — pre-fix the single up-front drain missed it and nothing
    later could ever flush it."""
    eng = _shell_engine(tmp_path)
    eng._sessions["s2"] = _session()
    in_save = threading.Event()
    resume = threading.Event()
    saved: list = []

    def _save(sid, session):
        saved.append(sid)
        if len(saved) == 1:
            in_save.set()
            assert resume.wait(timeout=10.0)
        return True

    eng._save_session_to_disk = _save

    t, entered, release, outcome = _enter_mutation(eng)
    assert entered.wait(timeout=5.0)
    eng.begin_shutdown()
    release.set()
    assert in_save.wait(timeout=5.0)  # self-flush mid-save (pass 1)

    eng._mark_dirty("s2")  # lock-free re-mark lands during the save
    resume.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert outcome.get("ok") is True
    assert eng._dirty_sessions == set()   # the rescan drained it
    assert set(saved) == {"s-new", "s2"}  # both persisted


def test_self_flush_rescan_bounded_on_persistent_failure(tmp_path):
    """The rescan bound's anti-livelock contract: a permanently failing
    save is re-marked ONCE and excluded from later passes — the exit path
    terminates (each failing id attempted exactly once) and leaves the id
    dirty rather than spinning."""
    eng = _shell_engine(tmp_path)
    attempts: list = []

    def _always_fail(sid, session):
        attempts.append(sid)
        raise RuntimeError("disk full")

    eng._save_session_to_disk = _always_fail

    t, entered, release, outcome = _enter_mutation(eng)
    assert entered.wait(timeout=5.0)
    eng.begin_shutdown()
    release.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert outcome.get("ok") is True    # self-flush never masks the outcome
    assert attempts == ["s-new"]        # exactly one attempt, no livelock
    assert eng._dirty_sessions == {"s-new"}  # honestly left marked
    assert not eng._lock.locked()


def test_begin_shutdown_fires_cooperative_prefill_cancel(tmp_path):
    """begin_shutdown doubles as a cancellation source for minutes-long
    compaction/rebuild prefills: it sets _shutdown_cancel_event, and the
    rebuild body threads exactly that event into _prefill_cache."""
    import inspect as _inspect

    eng = _shell_engine(tmp_path)
    eng._shutdown_cancel_event = threading.Event()

    # (a) the gate fires the event.
    eng.begin_shutdown()
    assert eng._shutdown_cancel_event.is_set()

    # (b) the rebuild body wires THE SAME event into its prefill.
    eng2 = _shell_engine(tmp_path)
    eng2._shutdown_cancel_event = threading.Event()
    seen: dict = {}
    eng2._touch_gpu = lambda: None
    eng2._tokenize_prompt = lambda msgs, thinking=True, tools=None: [1, 2, 3]
    eng2._find_base_cache = lambda msgs, tools=None: None
    eng2._prefill_cache = (
        lambda cache, tokens, cancel_event=None:
            seen.__setitem__("cancel_event", cancel_event)
    )
    eng2._evict_active_sessions_if_needed = (
        lambda protect_session_id=None: None
    )
    import mlx_soloheaven.engine.mlx_engine as mlx_engine_module
    orig_mpc = mlx_engine_module.make_prompt_cache
    mlx_engine_module.make_prompt_cache = lambda lm: [
        SimpleNamespace(keys=None, values=None, offset=0)
    ]
    try:
        eng2._language_model = SimpleNamespace()
        result = eng2.truncate_session("s1", 0)
    finally:
        mlx_engine_module.make_prompt_cache = orig_mpc
    assert result.get("status") == "ok"
    assert seen["cancel_event"] is eng2._shutdown_cancel_event

    # (c) compact_session wires the same event (source-level, same pattern
    # as the server shutdown ordering test).
    src = _inspect.getsource(MLXEngine.compact_session)
    assert src.count('getattr(self, "_shutdown_cancel_event", None)') >= 2


# --- F5: chat.py wiring — preflight via the reads executor, busy degrades ---------


def _first_stream_frame(eng):
    """Drive chat's SSE stream to its FIRST frame (the start event, which
    carries the preflight result), then close the stream."""
    import asyncio
    import json
    from mlx_soloheaven.api import chat as chat_mod

    async def _drive():
        agen = chat_mod._stream_chat_body(
            "s1", [{"role": "user", "content": "hi"}], eng
        )
        frame = await agen.__anext__()
        await agen.aclose()
        return frame

    line = asyncio.run(_drive())
    assert line.startswith("data: ")
    return json.loads(line[len("data: "):])


def test_chat_stream_preflight_busy_degrades():
    """A busy engine (generation holds the lock) degrades the start event to
    a 'busy' cache_info instead of hanging the stream or racing _sessions."""
    class BusyPreflightEngine:
        def session_cache_preflight(self, sid, messages):
            raise EngineBusyError("generation in flight")

    frame = _first_stream_frame(BusyPreflightEngine())
    assert frame["type"] == "start"
    assert frame["cache_hit"] is False
    assert frame["cache_info"]["type"] == "busy"


def test_chat_stream_preflight_proxy_reports_neutral():
    """Process-mode proxy (no preflight method — the child owns the cache):
    neutral marker, no attribute pokes."""
    class ProxyShaped:
        pass

    frame = _first_stream_frame(ProxyShaped())
    assert frame["cache_info"]["type"] == "process"


def test_chat_stream_preflight_runs_on_reads_executor():
    """The preflight result flows into the start event AND the call ran on
    the RESERVED reads pool (never the event loop / shared default pool)."""
    observed = {}

    class RecordingEngine:
        def session_cache_preflight(self, sid, messages):
            observed["thread"] = threading.current_thread().name
            return {
                "cache_hit": True,
                "cache_info": {"type": "kv_cache_hit", "detail": "d"},
            }

    frame = _first_stream_frame(RecordingEngine())
    assert frame["cache_hit"] is True
    assert frame["cache_info"]["type"] == "kv_cache_hit"
    assert observed["thread"].startswith("engine-read")


def test_chat_stream_preflight_error_is_nonfatal():
    """An unexpected preflight failure must never kill the stream — the
    preflight is informational only."""
    class ExplodingEngine:
        def session_cache_preflight(self, sid, messages):
            raise RuntimeError("boom")

    frame = _first_stream_frame(ExplodingEngine())
    assert frame["type"] == "start"
    assert frame["cache_info"]["type"] == "none"
