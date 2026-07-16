"""U13 — cancellation gaps (batch 3).

(a) QUEUED-REQUEST CANCEL: the worker used to consult its cancel registry
only for ACTIVE requests — a cancel op arriving BEFORE the request activated
(client disconnected while its generate command sat queued behind a long
generation) was silently dropped and the child ran the FULL generation for a
dead client. The worker now remembers pre-activation cancels in a bounded,
insertion-ordered pending set consulted (and consumed) at activation.

(b) CANCEL-AWARE LOCK WAIT + PREFILL: in-process, the engine lock acquire
and the chunked prefill never looked at cancel_event — a disconnect during a
90K-token prefill burned minutes of GPU. Now:
  - generate_stream polls the lock with bounded acquires and gives up when
    the event is set (no generation started),
  - the engine's own chunk loop (``_prefill_cache``), mlx-lm's prefill (via
    ``prompt_progress_callback``), the qwen_mtp prefill loops
    (``mtp_prefill_base`` + ``qwen_mtp_generate_step``) and PLD's
    ``_prefill`` all abort between chunks with ``GenerationCancelled``,
  - the engine stream loop converts that into the NORMAL cancel path, so a
    HIT session gets the C1 fail-closed reconcile (verified suffix rollback
    on trimmable caches, invalidation on hybrid/MTP caches) and a MISS's
    partially-filled fresh cache is simply discarded.

HERMETIC — no real model, pipes, or signals (conventions from
tests/test_shutdown_flush.py; the _generate_locked harness is shared with
tests/test_qwen_mtp.py via same-directory import).
"""

from __future__ import annotations

import signal
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_soloheaven.engine import process_protocol as proto
from mlx_soloheaven.engine import process_worker
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.process_worker import PENDING_CANCEL_MAX, worker_main
from mlx_soloheaven.engine.types import GenerationCancelled, GenerationResult

from test_qwen_mtp import (
    MockArrays,
    MockKV,
    MockOps,
    _content_token_suffix,
    _generate_engine,
    _mk_caches,
    _use_real_messages_match,
)


# --- helpers ------------------------------------------------------------------


class FakeConn:
    """Minimal Pipe-connection stand-in: scripted recv frames + send log."""

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


class CtrlScriptConn(FakeConn):
    """ctrl pipe stand-in: delivers scripted cancel frames, then sets
    ``drained`` and EOFs. The ctrl thread calls recv() again only AFTER it
    fully processed the previous frame, so ``drained`` set == every scripted
    cancel has been recorded."""

    def __init__(self, frames, drained: threading.Event):
        super().__init__(frames)
        self.drained = drained

    def recv(self):
        if not self.frames:
            self.drained.set()
            raise EOFError
        return self.frames.pop(0)


class GatedCmdConn(FakeConn):
    """cmd pipe stand-in that blocks poll() until ``gate`` is set — used to
    guarantee the ctrl thread recorded its cancels BEFORE the main loop
    activates the request (deterministic ordering, no sleeps)."""

    def __init__(self, gate: threading.Event, frames):
        super().__init__(frames)
        self.gate = gate

    def poll(self, timeout=None):
        return self.gate.wait(timeout=10.0)


class GenRecordingEngine:
    """Engine stand-in for worker_main — records generate_stream calls and
    generic RPC dispatches (``probe``)."""

    def __init__(self, cfg, execution_mode="worker"):
        self.cfg = cfg
        self.execution_mode = execution_mode
        self.model_id = "fake-model"
        self.model_family = "chatml"
        self.generate_calls: list = []
        self.flush_calls: list = []
        self.rpc_calls: list = []
        self._lock = threading.Lock()

    def load_model(self):
        pass

    def generate_stream(self, messages, **kwargs):
        self.generate_calls.append((messages, kwargs.get("cancel_event")))
        yield GenerationResult(finish_reason="stop")

    def probe(self, tag=None):
        # Generic RPC target for the F3 tombstone tests.
        self.rpc_calls.append(tag)
        return {"ok": True, "tag": tag}

    def _flush_all_on_shutdown(self):
        self.flush_calls.append("flush")


@pytest.fixture
def restore_signals():
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)


def _run_worker(cmd_conn, ctrl_conn, monkeypatch):
    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = GenRecordingEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    resp_conn = FakeConn()
    worker_main(
        {"model_path": "/tmp/x", "verbose": False},
        cmd_conn,
        resp_conn,
        ctrl_conn,
    )
    assert len(created) == 1
    return created[0], resp_conn


def _frames_for(resp_conn, rid):
    return [f for f in resp_conn.sent if isinstance(f, dict) and f.get("id") == rid]


# --- U13(a): pre-activation cancel consumed at activation ----------------------


def test_pre_activation_cancel_skips_generation(monkeypatch, restore_signals):
    """A cancel that arrives BEFORE its request activates is remembered and
    CONSUMED at activation: the generation never runs; the request still
    terminates with final(None)+done. A later, un-cancelled request runs
    normally (the pending entry was consumed, not sticky)."""
    drained = threading.Event()
    ctrl = CtrlScriptConn([proto.make_cancel("r-cancelled")], drained)
    cmd = GatedCmdConn(drained, [
        proto.make_generate("r-cancelled", [{"role": "user", "content": "hi"}], {}),
        proto.make_generate("r-normal", [{"role": "user", "content": "yo"}], {}),
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)

    # NO generation ran for the cancelled request; the normal one ran.
    assert len(engine.generate_calls) == 1
    assert engine.generate_calls[0][0] == [{"role": "user", "content": "yo"}]

    # The cancelled request still closed its frame contract (final + done).
    cancelled_frames = _frames_for(resp, "r-cancelled")
    assert [f["type"] for f in cancelled_frames] == ["final", "done"]
    assert cancelled_frames[0]["result"] is None

    normal_frames = _frames_for(resp, "r-normal")
    assert [f["type"] for f in normal_frames] == ["final", "done"]
    assert normal_frames[0]["result"] is not None  # the stop terminal


def test_pending_cancel_set_is_bounded(monkeypatch, restore_signals):
    """The pending-cancel set is bounded: past PENDING_CANCEL_MAX entries the
    OLDEST is evicted (its later activation generates normally), while the
    newest entries are retained (their activations are skipped)."""
    n = PENDING_CANCEL_MAX + 1
    drained = threading.Event()
    ctrl = CtrlScriptConn(
        [proto.make_cancel(f"c{i}") for i in range(n)], drained
    )
    cmd = GatedCmdConn(drained, [
        # c0 was evicted (oldest) -> generation RUNS.
        proto.make_generate("c0", [{"role": "user", "content": "evicted"}], {}),
        # The newest is retained -> generation SKIPPED.
        proto.make_generate(
            f"c{n - 1}", [{"role": "user", "content": "retained"}], {}
        ),
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)

    assert len(engine.generate_calls) == 1
    assert engine.generate_calls[0][0] == [{"role": "user", "content": "evicted"}]
    retained_frames = _frames_for(resp, f"c{n - 1}")
    assert [f["type"] for f in retained_frames] == ["final", "done"]
    assert retained_frames[0]["result"] is None


def test_pending_cancel_dropped_after_request_completes(monkeypatch, restore_signals):
    """A cancel arriving mid-generation goes to the ACTIVE event (existing
    path) — nothing lingers in the pending set to poison an unrelated later
    request that happens to reuse the id (worker cleanup drops both)."""
    # Ordering is made deterministic with two events: the ctrl thread
    # releases its cancel frame only AFTER r1's generation started (so the
    # cancel targets the ACTIVE registry, never the pending set), and the
    # first generation finishes only AFTER the ctrl thread processed it.
    first_gen_started = threading.Event()
    cancel_seen = threading.Event()

    class MidGenEngine(GenRecordingEngine):
        def generate_stream(self, messages, **kwargs):
            ev = kwargs.get("cancel_event")
            self.generate_calls.append((messages, ev))
            if len(self.generate_calls) == 1:
                first_gen_started.set()
                # Block until the ctrl thread processed the cancel — it must
                # land on THIS active event, not pending.
                assert cancel_seen.wait(timeout=10.0)
            yield GenerationResult(finish_reason="stop")

    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = MidGenEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    class MidGenCtrl(FakeConn):
        def recv(self):
            if self.frames:
                # Hold the cancel until r1 is ACTIVE (mid-generation).
                assert first_gen_started.wait(timeout=10.0)
                return self.frames.pop(0)
            cancel_seen.set()
            raise EOFError

    ctrl = MidGenCtrl([proto.make_cancel("r1")])
    cmd = FakeConn([
        proto.make_generate("r1", [{"role": "user", "content": "hi"}], {}),
        # Same id re-used later: must NOT be pre-cancelled by leftovers.
        proto.make_generate("r1", [{"role": "user", "content": "again"}], {}),
        {"op": "shutdown"},
    ])

    resp = FakeConn()
    worker_main({"model_path": "/tmp/x", "verbose": False}, cmd, resp, ctrl)
    engine = created[0]
    assert len(engine.generate_calls) == 2  # both ran (no sticky cancel)
    # First request's ACTIVE event was set by the mid-generation cancel;
    # the second (same id, later) got a fresh, un-set event — the cancel
    # never leaked into the pending set.
    assert engine.generate_calls[0][1].is_set()
    assert not engine.generate_calls[1][1].is_set()


# --- U13(b): cancel-aware engine lock wait --------------------------------------


def _lock_shell():
    eng = MLXEngine.__new__(MLXEngine)
    eng._lock = threading.Lock()
    return eng


def test_generate_stream_lock_wait_cancelled_no_generation():
    """Cancel while queued behind the engine lock: generate_stream yields
    NOTHING, never acquires the lock, and never reaches session resolution
    (no generation side effects)."""
    eng = _lock_shell()
    eng._resolve_anon_session_id_locked = lambda *a, **k: pytest.fail(
        "cancelled queue wait must never resolve a session"
    )
    cancel = threading.Event()
    cancel.set()

    assert eng._lock.acquire(blocking=False)  # another request holds the lock
    try:
        out = list(eng.generate_stream([{"role": "user", "content": "x"}],
                                       cancel_event=cancel))
    finally:
        eng._lock.release()
    assert out == []
    assert not eng._lock.locked()  # never stolen / left held


def test_generate_stream_lock_wait_cancel_mid_wait():
    """The cancel can land WHILE polling (not only up front) — the poll loop
    observes it within one bounded acquire."""
    eng = _lock_shell()
    cancel = threading.Event()
    assert eng._lock.acquire(blocking=False)
    try:
        result = {}

        def _consume():
            result["out"] = list(
                eng.generate_stream([{"role": "user", "content": "x"}],
                                    cancel_event=cancel)
            )

        t = threading.Thread(target=_consume, daemon=True)
        t.start()
        cancel.set()
        t.join(timeout=5.0)
        assert not t.is_alive()
    finally:
        eng._lock.release()
    assert result["out"] == []


def test_acquire_lock_cancellable_no_event_blocks_normally():
    eng = _lock_shell()
    assert eng._acquire_lock_cancellable(None) is True
    assert eng._lock.locked()
    eng._lock.release()


# --- U13(b): engine chunked prefill (_prefill_cache) -----------------------------


def test_prefill_cache_cancels_between_chunks():
    """_prefill_cache checks the event at each chunk boundary: with the event
    set by the FIRST model call, exactly one chunk runs before the abort."""
    eng = MLXEngine.__new__(MLXEngine)
    cancel = threading.Event()
    calls: list = []

    def fake_model(chunk, cache=None):
        calls.append(int(chunk.shape[1]))
        cancel.set()  # disconnect lands during the first chunk

    eng._language_model = fake_model
    tokens = list(range(MLXEngine._PREFILL_STEP * 3))
    with pytest.raises(GenerationCancelled):
        eng._prefill_cache([], tokens, cancel_event=cancel)
    assert calls == [MLXEngine._PREFILL_STEP]  # aborted BETWEEN chunks


def test_prefill_cache_pre_set_cancel_runs_nothing():
    eng = MLXEngine.__new__(MLXEngine)
    eng._language_model = lambda *a, **k: pytest.fail("must not forward")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        eng._prefill_cache([], [1, 2, 3], cancel_event=cancel)


def test_prefill_cache_without_event_unchanged():
    eng = MLXEngine.__new__(MLXEngine)
    calls: list = []
    eng._language_model = lambda chunk, cache=None: calls.append(1)
    eng._prefill_cache([], [1, 2, 3])  # no event -> full prefill
    assert calls == [1]


# --- U13(b): mlx-lm path prefill cancel -> C1 fail-closed reconcile ---------------


def _prefill_cancel_lm_stream(monkeypatch, prompts_seen, cancel, *, chunk=1):
    """lm_stream_generate stand-in that simulates mlx-lm's chunked prefill:
    it invokes ``prompt_progress_callback`` between chunks (advancing cache
    offsets per chunk) and sets ``cancel`` after the first chunk — the
    callback must then raise GenerationCancelled (never reaching a token)."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompts_seen.append(list(prompt))
        cb = kwargs.get("prompt_progress_callback")
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            # (a real generator, like mlx-lm's stream_generate: the prefill —
            # and therefore the cancel raise — runs at first next(), inside
            # the engine's stream loop)
            assert cb is not None, "engine must wire the prefill cancel hook"
            total = len(prompt)
            done = 0
            cb(0, total)
            while total - done > 1:
                n = min(chunk, total - done - 1)
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset += n
                done += n
                cancel.set()  # client disconnects mid-prefill
                cb(done, total)  # must raise GenerationCancelled
            pytest.fail("prefill should have aborted between chunks")
            yield  # pragma: no cover — makes this function a generator

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)


def _drive_with_event(eng, messages, cancel, *, max_tokens=8):
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=cancel,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )


def test_prefill_cancel_hit_session_rolls_suffix_back(monkeypatch):
    """HIT session, trimmable cache, cancel between prefill chunks: the C1
    rules apply — the partially prefilled suffix is ROLLED BACK (verified
    per-layer) to the pre-turn history and session.messages (unchanged)
    exactly describes the result. Nothing reached the client."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)

    cancel = threading.Event()
    prompts_seen: list = []
    _prefill_cancel_lm_stream(monkeypatch, prompts_seen, cancel)

    orig_sess = eng._sessions["s"]
    chunks = _drive_with_event(eng, messages, cancel)
    assert chunks == []  # nothing reached the client

    # The suffix chunk that DID prefill was rolled back fail-closed.
    assert prompts_seen[0] == [52, 99]  # HIT: only the suffix was fed
    assert cs.token_ids == stored
    for c in cache:
        assert c.offset == len(stored)
    assert eng._sessions["s"] is orig_sess
    assert orig_sess.total_cache_tokens == len(stored)
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]


def test_prefill_cancel_hit_hybrid_cache_invalidates(monkeypatch):
    """Same shape on a hybrid cache (untrimmable ArraysCache layers): the
    partial suffix cannot be rolled back -> INVALIDATE fail-closed (next
    turn takes an honest MISS cold-fill)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)

    cancel = threading.Event()
    prompts_seen: list = []
    _prefill_cancel_lm_stream(monkeypatch, prompts_seen, cancel)

    orig_sess = eng._sessions["s"]
    chunks = _drive_with_event(eng, messages, cancel)
    assert chunks == []
    assert cs.cache is None
    assert cs.token_ids is None
    assert eng._sessions["s"] is orig_sess
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]


def test_prefill_cancel_miss_discards_fresh_cache(monkeypatch):
    """MISS path: the partially-filled FRESH cache is simply discarded — no
    session is persisted, nothing to roll back."""
    eng, cs, messages = _generate_engine(
        [MockKV() for _ in range(5)], [1, 2, 3], mtp=False, suffix=[],
    )
    # Force a MISS: no stored session for this id.
    eng._sessions.pop("s")
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: list(range(10))
    eng._language_model.make_cache = lambda: [MockKV() for _ in range(5)]

    cancel = threading.Event()
    prompts_seen: list = []
    _prefill_cancel_lm_stream(monkeypatch, prompts_seen, cancel, chunk=4)

    chunks = _drive_with_event(eng, messages, cancel)
    assert chunks == []
    assert prompts_seen[0] == list(range(10))  # full cold-fill prompt
    assert "s" not in eng._sessions  # partial fresh cache discarded


# --- U13(b): qwen_mtp prefill loops ----------------------------------------------


def test_mtp_prefill_base_cancel_pre_set_runs_nothing():
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        qwen_mtp_mod.mtp_prefill_base(
            [1, 2, 3, 4],
            model=None,
            head=None,
            prompt_cache=cache,
            n_target_layers=5,
            prefill_step_size=2,
            ops=ops,
            cancel_event=cancel,
        )
    assert ops.head_calls == []  # nothing forwarded


def test_mtp_prefill_base_cancel_between_chunks():
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)

    class AfterFirstChunk:
        def is_set(self):
            return len(ops.head_calls) >= 1

    with pytest.raises(GenerationCancelled):
        qwen_mtp_mod.mtp_prefill_base(
            list(range(1, 8)),
            model=None,
            head=None,
            prompt_cache=cache,
            n_target_layers=5,
            prefill_step_size=2,
            ops=ops,
            cancel_event=AfterFirstChunk(),
        )
    assert len(ops.head_calls) == 1  # exactly one chunk before the abort


def test_qwen_mtp_generate_step_prefill_cancel_invalidates():
    """Cancel during the runner's own prefill loop: the stream terminates
    with GenerationCancelled and the corruption callback fired exactly once
    (fail-closed — a mid-prefill target+head cache cannot be verifiably
    rewound, so the engine's cancel commit invalidates)."""
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    fired = {"corruption": 0}

    class AfterFirstChunk:
        def is_set(self):
            return len(ops.head_calls) >= 1

    gen = qwen_mtp_generate_step(
        mx.array(list(range(1, 8)), mx.uint32),
        model=None,
        head=None,
        block_size=3,
        max_tokens=8,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        prefill_step_size=2,
        on_cache_corruption=lambda: fired.__setitem__(
            "corruption", fired["corruption"] + 1
        ),
        ops=ops,
        cancel_event=AfterFirstChunk(),
    )
    with pytest.raises(GenerationCancelled):
        list(gen)
    assert fired["corruption"] == 1


# --- F3: timed-out RPC tombstones — the worker skips stale queued RPCs ------------


def test_stale_rpc_skipped_by_worker(monkeypatch, restore_signals):
    """An ``rpc_cancel`` arriving BEFORE the queued RPC executes (its parent
    wait timed out) tombstones the rid: the worker never executes the stale
    request and sends it NO frames; a different, live RPC runs normally."""
    drained = threading.Event()
    ctrl = CtrlScriptConn([proto.make_rpc_cancel("rpc-stale")], drained)
    cmd = GatedCmdConn(drained, [
        proto.make_rpc("rpc-stale", "probe", ["stale"], {}),
        proto.make_rpc("rpc-live", "probe", ["live"], {}),
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)

    # The stale RPC never executed; the live one did.
    assert engine.rpc_calls == ["live"]
    # The stale rid got NO result/error frames (its parent slot is gone; a
    # reply would be dropped anyway) — only the F5 cancel-ack stating the
    # DEFINITIVE outcome: tombstoned before dispatch == "cancelled".
    stale_frames = _frames_for(resp, "rpc-stale")
    assert [f["type"] for f in stale_frames] == ["rpc_cancel_ack"]
    assert stale_frames[0]["outcome"] == "cancelled"
    live_frames = _frames_for(resp, "rpc-live")
    assert [f["type"] for f in live_frames] == ["rpc_result"]
    assert live_frames[0]["result"] == {"ok": True, "tag": "live"}


def test_rpc_cancel_tombstone_consumed_not_sticky(monkeypatch, restore_signals):
    """A consumed tombstone must not poison a LATER rpc that reuses the id
    (uuid reuse can't happen in production; this pins the consume-on-skip
    semantics)."""
    drained = threading.Event()
    ctrl = CtrlScriptConn([proto.make_rpc_cancel("r1")], drained)
    cmd = GatedCmdConn(drained, [
        proto.make_rpc("r1", "probe", ["first"], {}),   # skipped (tombstone)
        proto.make_rpc("r1", "probe", ["second"], {}),  # runs (consumed)
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)
    assert engine.rpc_calls == ["second"]


def test_rpc_cancel_tombstone_set_is_bounded(monkeypatch, restore_signals):
    """Past CANCELLED_RPC_MAX tombstones the OLDEST is evicted (its RPC runs)
    while the newest is retained (its RPC is skipped) — mirrors the U13
    pending-cancel bound.

    Round-3 finding 3 contract: the eviction REVOKES the skip promise, so
    the evicted rid is acked "unknown" (definitive honesty — its RPC then
    executes, consistent with "unknown") and NEVER "cancelled"; the
    retained rid is acked "cancelled" only when the main loop actually
    consumes its tombstone and skips dispatch."""
    from mlx_soloheaven.engine.process_worker import CANCELLED_RPC_MAX

    n = CANCELLED_RPC_MAX + 1
    drained = threading.Event()
    ctrl = CtrlScriptConn(
        [proto.make_rpc_cancel(f"c{i}") for i in range(n)], drained
    )
    cmd = GatedCmdConn(drained, [
        # c0 was evicted (oldest) -> executes.
        proto.make_rpc("c0", "probe", ["evicted"], {}),
        # The newest is retained -> skipped.
        proto.make_rpc(f"c{n - 1}", "probe", ["retained"], {}),
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)
    assert engine.rpc_calls == ["evicted"]
    # The EVICTED rid: the evictor acked "unknown" (we no longer know), and
    # its RPC executed — an "unknown"-then-executed sequence is consistent;
    # a "cancelled"-then-executed one (the pre-fix contradiction) is not.
    evicted_frames = _frames_for(resp, "c0")
    assert [f["type"] for f in evicted_frames] == [
        "rpc_cancel_ack", "rpc_result",
    ]
    assert evicted_frames[0]["outcome"] == "unknown"
    # The retained rid was skipped: its only frame is the cancel-ack, sent
    # by the MAIN loop at dispatch-skip time with the (now true) outcome.
    retained_frames = _frames_for(resp, f"c{n - 1}")
    assert [f["type"] for f in retained_frames] == ["rpc_cancel_ack"]
    assert retained_frames[0]["outcome"] == "cancelled"


def test_rpc_cancel_pending_tombstone_gets_no_premature_ack(
    monkeypatch, restore_signals,
):
    """Round-3 finding 3: tombstone registration is PENDING-ACK. A tombstone
    that is never consumed (its command never arrives) and never evicted
    gets NO ack at all — pre-fix the ctrl thread acked "cancelled"
    immediately, a promise the bounded set could not keep."""
    drained = threading.Event()
    ctrl = CtrlScriptConn([proto.make_rpc_cancel("ghost")], drained)
    cmd = GatedCmdConn(drained, [
        # An unrelated live RPC proves the loop ran normally.
        proto.make_rpc("rpc-live", "probe", ["live"], {}),
        {"op": "shutdown"},
    ])

    engine, resp = _run_worker(cmd, ctrl, monkeypatch)
    assert engine.rpc_calls == ["live"]
    assert _frames_for(resp, "ghost") == []  # no premature "cancelled"


# --- F5 (round 2): rpc_cancel outcome acks -----------------------------------------
# (a cancel cannot stop an RPC that already started or finished — the worker
# must ack the DEFINITIVE outcome so the parent never believes a mutation
# didn't run; "cancelled" for a pre-dispatch tombstone is asserted in
# test_stale_rpc_skipped_by_worker above)


def test_rpc_cancel_during_execution_acks_already_started(monkeypatch, restore_signals):
    """A cancel arriving while the RPC is EXECUTING cannot stop it: the ctrl
    thread acks {outcome: "already_started"} mid-execution (send-locked resp
    pipe) and the RPC still runs to completion with its rpc_result —
    pre-fix the cancel was silently tombstoned while the mutation ran."""
    rpc_started = threading.Event()
    ctrl_drained = threading.Event()

    class BlockingRpcEngine(GenRecordingEngine):
        def probe(self, tag=None):
            rpc_started.set()
            # Resume only after the ctrl thread processed the cancel — it
            # must observe THIS rpc as started, not queued.
            assert ctrl_drained.wait(timeout=10.0)
            return super().probe(tag)

    class WaitingCtrl(FakeConn):
        def recv(self):
            if self.frames:
                assert rpc_started.wait(timeout=10.0)
                return self.frames.pop(0)
            ctrl_drained.set()
            raise EOFError

    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = BlockingRpcEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    ctrl = WaitingCtrl([proto.make_rpc_cancel("r1")])
    cmd = FakeConn([
        proto.make_rpc("r1", "probe", ["running"], {}),
        {"op": "shutdown"},
    ])
    resp = FakeConn()
    worker_main({"model_path": "/tmp/x", "verbose": False}, cmd, resp, ctrl)

    engine = created[0]
    assert engine.rpc_calls == ["running"]  # the cancel could NOT stop it
    frames = _frames_for(resp, "r1")
    assert [f["type"] for f in frames] == ["rpc_cancel_ack", "rpc_result"]
    assert frames[0]["outcome"] == "already_started"


def test_rpc_cancel_after_completion_acks_unknown(monkeypatch, restore_signals):
    """A cancel that arrives AFTER the RPC completed (its reply crossed the
    cancel on the wire) acks {outcome: "unknown"} and records NO tombstone —
    the parent's log learns the operation actually ran."""
    rpc_finished = threading.Event()
    ctrl_drained = threading.Event()

    class GatedCmd(FakeConn):
        """poll() for the SECOND command blocks until the ctrl thread
        processed its cancel — the main loop only re-polls after the first
        RPC's dispatch (including its completion bookkeeping) finished."""

        def __init__(self, frames):
            super().__init__(frames)
            self.recvs = 0

        def poll(self, timeout=None):
            if self.recvs >= 1:
                rpc_finished.set()
                assert ctrl_drained.wait(timeout=10.0)
            return True

        def recv(self):
            self.recvs += 1
            return super().recv()

    class WaitingCtrl(FakeConn):
        def recv(self):
            if self.frames:
                assert rpc_finished.wait(timeout=10.0)
                return self.frames.pop(0)
            ctrl_drained.set()
            raise EOFError

    created = []

    def _factory(cfg, execution_mode="worker"):
        eng = GenRecordingEngine(cfg, execution_mode=execution_mode)
        created.append(eng)
        return eng

    import mlx_soloheaven.engine.mlx_engine as engine_module
    monkeypatch.setattr(engine_module, "MLXEngine", _factory)

    ctrl = WaitingCtrl([proto.make_rpc_cancel("r1")])
    cmd = GatedCmd([
        proto.make_rpc("r1", "probe", ["done"], {}),
        {"op": "shutdown"},
    ])
    resp = FakeConn()
    worker_main({"model_path": "/tmp/x", "verbose": False}, cmd, resp, ctrl)

    engine = created[0]
    assert engine.rpc_calls == ["done"]  # it RAN, before the cancel arrived
    frames = _frames_for(resp, "r1")
    assert [f["type"] for f in frames] == ["rpc_result", "rpc_cancel_ack"]
    assert frames[1]["outcome"] == "unknown"


def test_make_rpc_cancel_ack_shape():
    frame = proto.make_rpc_cancel_ack("abc", "cancelled")
    assert frame == {"id": "abc", "type": "rpc_cancel_ack", "outcome": "cancelled"}


# --- F4: base-cache registration honors cancel_event ------------------------------


def _base_cache_shell():
    """Engine shell exposing exactly the surface _maybe_register_base_cache
    touches, with the REAL _prefill_cache / _maybe_register_base_cache."""
    eng = MLXEngine.__new__(MLXEngine)
    eng._has_rotating_cache = False
    eng._base_caches = {}
    eng._system_hash = lambda msgs, tools=None: "sys-h"
    eng._extract_system_tokens = (
        lambda msgs, tokens, tools=None, thinking=True: tokens[:2]
    )
    eng._mtp_base_caches_active = lambda: False
    eng._language_model = SimpleNamespace(
        make_cache=lambda: [MockKV() for _ in range(3)],
        layers=[],
    )
    eng.cfg = SimpleNamespace(prefill_step_size=512)
    eng._register_base_cache = lambda *a, **k: pytest.fail(
        "cancelled base prefill must not register"
    )
    return eng


def test_base_cache_registration_cancelled_skips_silently():
    """Cancel during the secondary system-prompt prefill: registration is
    skipped, NO exception escapes, and the base-cache pool is unchanged."""
    eng = _base_cache_shell()
    cancel = threading.Event()
    cancel.set()  # disconnect landed before/at the base prefill

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    # Must not raise (GenerationCancelled is swallowed as a debug skip).
    eng._maybe_register_base_cache(
        messages, [1, 2, 3, 4], cancel_event=cancel,
    )
    assert eng._base_caches == {}


def test_base_cache_registration_without_cancel_still_registers():
    """No cancel event: the same shell registers normally (the F4 plumbing
    must not break the plain path)."""
    eng = _base_cache_shell()
    registered: list = []
    eng._register_base_cache = (
        lambda *a, **k: registered.append((a, k))
    )

    class _Model:
        layers: list = []

        def make_cache(self):
            return [MockKV() for _ in range(3)]

        def __call__(self, chunk, cache=None):
            for c in cache or []:
                c.offset += int(chunk.shape[1])

    eng._language_model = _Model()
    eng._eval_cache = lambda cache: None

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    eng._maybe_register_base_cache(messages, [1, 2, 3, 4], cancel_event=None)
    assert len(registered) == 1


def test_base_cache_mtp_prefill_receives_cancel_event(monkeypatch):
    """MTP branch wiring: mtp_prefill_base gets the SAME cancel_event, and a
    GenerationCancelled from it is swallowed as a skip (pool unchanged)."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    eng = _base_cache_shell()
    eng._mtp_base_caches_active = lambda: True
    eng._drafter = SimpleNamespace(layers=[object()])

    recorded = {}

    def _fake_prefill(tokens, **kwargs):
        recorded["cancel_event"] = kwargs.get("cancel_event")
        raise GenerationCancelled("base prefill cancelled")

    monkeypatch.setattr(qwen_mtp_mod, "mtp_prefill_base", _fake_prefill)
    monkeypatch.setattr(qwen_mtp_mod, "make_head_cache", lambda n: [])

    cancel = threading.Event()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    eng._maybe_register_base_cache(
        messages, [1, 2, 3, 4], cancel_event=cancel,
    )
    assert recorded["cancel_event"] is cancel
    assert eng._base_caches == {}


def test_generate_locked_wires_cancel_event_into_base_registration(monkeypatch):
    """INTEGRATION (the codex F4 gap): _generate_locked's MISS path must pass
    the request's cancel_event into _maybe_register_base_cache — pre-fix it
    passed nothing and a disconnect during the secondary system-prompt
    prefill ran to completion."""
    from test_qwen_mtp import _scripted_lm_stream

    eng, cs, messages = _generate_engine(
        [MockKV() for _ in range(5)], [1, 2, 3], mtp=False, suffix=[],
    )
    # Force a MISS: no stored session for this id.
    eng._sessions.pop("s")
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: list(range(10))
    eng._language_model.make_cache = lambda: [MockKV() for _ in range(5)]

    recorded = {}

    def _record_register(msgs, prompt_tokens, tools=None, thinking=True,
                         cancel_event=None):
        recorded["cancel_event"] = cancel_event

    eng._maybe_register_base_cache = _record_register

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "x"), (102, "x")]],
        advance="post",
    )

    cancel = threading.Event()  # never set — the generation completes
    chunks = _drive_with_event(eng, messages, cancel)
    assert chunks, "generation must complete and yield frames"
    assert chunks[-1].finish_reason == "stop"
    assert recorded.get("cancel_event") is cancel


# --- F4 (round 2): cancel DURING the final prefill chunk ---------------------------
# (the between-chunk checks miss a disconnect that lands while the LAST chunk
# is running — execution used to slip through to base-cache registration and
# follow-up model work)


def test_prefill_cache_cancel_during_final_chunk_raises():
    """Disconnect DURING the last chunk: the post-loop check must raise
    before any further model work (cache eval here, registration upstream).
    Pre-fix this prefill returned normally."""
    eng = MLXEngine.__new__(MLXEngine)
    cancel = threading.Event()
    calls: list = []

    def fake_model(chunk, cache=None):
        calls.append(int(chunk.shape[1]))
        if len(calls) == 2:  # disconnect lands during the FINAL chunk
            cancel.set()

    eng._language_model = fake_model
    eng._eval_cache = lambda cache: pytest.fail(
        "no model work may follow a cancelled final chunk"
    )
    tokens = list(range(MLXEngine._PREFILL_STEP * 2))  # exactly two chunks
    with pytest.raises(GenerationCancelled):
        eng._prefill_cache([], tokens, cancel_event=cancel)
    assert len(calls) == 2  # both chunks ran; the abort came AFTER the last


def test_mtp_prefill_base_cancel_during_final_chunk_raises():
    """Same gap in the MTP base prefill: cancel set DURING the last chunk →
    the post-loop check aborts BEFORE the held-out-token forward (target
    offsets stay at the chunked count; nothing reaches registration)."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)

    class DuringLastChunk:
        def is_set(self):
            return len(ops.head_calls) >= 3  # true only once chunk 3 ran

    with pytest.raises(GenerationCancelled):
        qwen_mtp_mod.mtp_prefill_base(
            list(range(1, 8)),  # 7 tokens: 3 chunks of 2 + held-out last
            model=None,
            head=None,
            prompt_cache=cache,
            n_target_layers=5,
            prefill_step_size=2,
            ops=ops,
            cancel_event=DuringLastChunk(),
        )
    assert len(ops.head_calls) == 3  # every chunk ran; abort came after
    # The held-out final forward never ran: target KV offsets stopped at 6.
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 6


def test_base_cache_registration_cancel_during_final_chunk_not_registered():
    """THE round-2 F4 gap: the prefill COMPLETES (the disconnect landed
    during its final chunk, so every between-chunk check passed) —
    registration must still be skipped by the pre-registration check.
    Pre-fix a base cache was registered for a dead request."""
    eng = _base_cache_shell()
    cancel = threading.Event()

    def _prefill_sets_cancel_at_end(cache, tokens, cancel_event=None):
        # Simulates a disconnect DURING the last chunk of a prefill whose
        # own final check is absent: it returns normally, event set.
        cancel.set()

    eng._prefill_cache = _prefill_sets_cancel_at_end
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    # _register_base_cache is armed to pytest.fail in the shell — the
    # GenerationCancelled raised pre-registration is swallowed as a skip.
    eng._maybe_register_base_cache(messages, [1, 2, 3, 4], cancel_event=cancel)
    assert eng._base_caches == {}


def test_base_cache_mtp_cancel_during_final_chunk_not_registered(monkeypatch):
    """MTP branch of the same gap: mtp_prefill_base returns normally with
    the event set — the pre-registration check still skips registration."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    eng = _base_cache_shell()
    eng._mtp_base_caches_active = lambda: True
    eng._drafter = SimpleNamespace(layers=[object()])
    cancel = threading.Event()

    def _fake_prefill(tokens, **kwargs):
        cancel.set()  # disconnect during the final chunk; prefill completes
        return object()

    monkeypatch.setattr(qwen_mtp_mod, "mtp_prefill_base", _fake_prefill)
    monkeypatch.setattr(qwen_mtp_mod, "make_head_cache", lambda n: [])

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    eng._maybe_register_base_cache(messages, [1, 2, 3, 4], cancel_event=cancel)
    assert eng._base_caches == {}


# --- F4 (round 3): final-chunk cancellation gaps in the LIVE runners ---------------
# (round 2 fixed the base-cache HELPERS; the live generation runners kept the
# same gap: PLD proceeded from its prefill loop straight into the remaining/
# bootstrap forward, the qwen MTP runner into its held-out-token bootstrap,
# and a cancellation-driven generator close still ran the MTP finalize
# forwards)


def test_pld_prefill_cancel_during_final_chunk_stops_before_bootstrap():
    """PLD live runner: cancel lands DURING the last prefill chunk — the
    post-loop check must raise BEFORE the remaining/bootstrap forward.
    Pre-fix the runner went on to forward the remaining tokens and yield."""
    import mlx.nn as nn

    from mlx_soloheaven.engine.pld import pld_generate_step

    cancel = threading.Event()
    calls: list = []

    class _Cache:
        def __init__(self):
            self.offset = 0
            self.state = mx.array([0])

        def is_trimmable(self):
            return True

        def trim(self, n):
            n = min(self.offset, n)
            self.offset -= n
            return n

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [object()]

        def __call__(self, x, cache=None):
            T = int(x.shape[1])
            calls.append(T)
            if len(calls) == 2:  # disconnect DURING the final prefill chunk
                cancel.set()
            if cache is not None:
                cache[0].offset += T
            return mx.zeros((1, T, 8))

    gen = pld_generate_step(
        mx.array([1, 2, 3, 4, 5], mx.uint32),  # 2 chunks of 2 + 1 remaining
        _Model(),
        num_draft_tokens=2,
        max_tokens=4,
        prompt_cache=[_Cache()],
        prefill_step_size=2,
        cancel_event=cancel,
    )
    with pytest.raises(GenerationCancelled):
        next(gen)
    # Both prefill chunks ran; the remaining/bootstrap forward did NOT.
    assert calls == [2, 2]


def test_qwen_mtp_prefill_cancel_during_final_chunk_stops_before_bootstrap():
    """Qwen MTP live runner: cancel lands DURING the last prefill chunk —
    the post-loop check aborts BEFORE the held-out bootstrap forward, with
    the same fail-closed invalidate as the in-loop check (corruption fired
    exactly once). Pre-fix the runner forwarded the held-out token and
    yielded a bootstrap token for a dead request."""
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    fired = {"corruption": 0}

    class DuringLastChunk:
        def is_set(self):
            # True only once every prefill chunk (3 x 2 tokens) has run.
            return len(ops.head_calls) >= 3

    gen = qwen_mtp_generate_step(
        mx.array(list(range(1, 8)), mx.uint32),  # 7 tokens: 3 chunks + 1
        model=None,
        head=None,
        block_size=3,
        max_tokens=8,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        prefill_step_size=2,
        on_cache_corruption=lambda: fired.__setitem__(
            "corruption", fired["corruption"] + 1
        ),
        ops=ops,
        cancel_event=DuringLastChunk(),
    )
    with pytest.raises(GenerationCancelled):
        list(gen)
    assert fired["corruption"] == 1
    assert len(ops.head_calls) == 3  # every chunk ran; the abort came after
    # The held-out bootstrap forward never ran: target KV offsets stayed at
    # the chunked count (6 of 7 tokens).
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 6


def test_qwen_mtp_cancel_driven_close_skips_finalize_forwards():
    """Cancellation-driven generator close: the finally still SETTLES the
    round (cache-state consistency — C1 depends on settled offsets) but
    SKIPS the finalize forwards (pending-pair head commit + pending-token
    target forward) and reports on_finalize(None, None). Pre-fix both
    finalize forwards executed for a request that was already dead, and the
    finalized cache sat one token ahead of the recorded ids."""
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    cache = _mk_caches()
    # Model chain t -> t+1 and head chain t -> t+1: drafts fully accepted.
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    cancel = threading.Event()
    finalized: dict = {}

    gen = qwen_mtp_generate_step(
        mx.array([1, 2, 3], mx.uint32),
        model=None,
        head=None,
        block_size=2,
        max_tokens=16,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: finalized.__setitem__("corrupt", True),
        on_finalize=lambda tok, hid: finalized.update(tok=tok, hid=hid),
        ops=ops,
        cancel_event=cancel,
    )
    tok0, _lp0, _fd0 = next(gen)  # bootstrap token (4)
    tok1, _lp1, _fd1 = next(gen)  # first token of the speculative round (5)
    assert (tok0, tok1) == (4, 5)

    cancel.set()  # the engine observes cancellation ...
    head_calls_before = len(ops.head_calls)
    gen.close()  # ... and closes the generator

    # on_finalize reported honestly: NO resume hidden survives a cancel.
    assert finalized == {"tok": None, "hid": None}
    # The finalize forwards were SKIPPED: no extra head commit ran (settle
    # with consumed=1 commits nothing), and the target holds EXACTLY
    # prompt + yielded - 1 tokens (3 + 2 - 1 = 4) — the finalize's
    # pending-token forward (which would land it at 5) never ran.
    assert len(ops.head_calls) == head_calls_before
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 4
    # The settle DID run: the round invariant head == target - 1 holds.
    assert cache[5].offset == 3


def test_qwen_mtp_natural_exhaustion_with_late_cancel_still_finalizes():
    """Codex round 5, finding 2 — the exact interleaving: the engine
    records + delivers the LAST (max-token) frame, the client disconnects
    (cancel_event set), the engine requests the next frame and the runner
    exhausts NATURALLY into its finally. No further frame reaches the
    engine, so its cancel check never fires and it takes the NORMAL
    persistence path with the FULL recorded history — the finalize forwards
    MUST therefore run (cache == recorded ids, on_finalize reports the
    resume state). Pre-fix the finally read the live event and skipped
    finalization: the cache landed one token short of the recorded ids,
    the reconcile truncated token_ids while the complete assistant text was
    saved, and the next HIT's suffix splice silently dropped the final
    token."""
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    cache = _mk_caches()
    # Target chain t -> t+1, head identical: drafts fully accepted.
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    cancel = threading.Event()
    finalized: dict = {}
    suffix = [1, 2, 3]
    max_tokens = 4

    gen = qwen_mtp_generate_step(
        mx.array(suffix, mx.uint32),
        model=None,
        head=None,
        block_size=2,
        max_tokens=max_tokens,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: finalized.__setitem__("corrupt", True),
        on_finalize=lambda tok, hid: finalized.update(tok=tok, hid=hid),
        ops=ops,
        cancel_event=cancel,
    )
    # Engine loop: pull EVERY frame up to max_tokens (all delivered +
    # recorded engine-side).
    out = [next(gen)[0] for _ in range(max_tokens)]
    assert out == [4, 5, 6, 7]

    # ... the client disconnects BETWEEN the last frame and the runner's
    # natural exhaustion.
    cancel.set()
    with pytest.raises(StopIteration):
        next(gen)  # natural exhaustion — enters the finally with event SET

    # Finalization RAN: on_finalize reported the pending token + a real
    # resume hidden (pre-fix: (None, None)).
    assert "corrupt" not in finalized
    assert finalized["tok"] == out[-1]
    assert finalized["hid"] is not None
    # The pending-token target forward landed: target == suffix + ALL
    # emitted tokens (matches the engine's recorded ids exactly), head one
    # behind (the lazy last slot).
    full = len(suffix) + max_tokens
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == full
    assert cache[5].offset == full - 1


def test_qwen_mtp_cancel_close_after_late_event_still_skips():
    """Companion guard for finding 2: a REAL cancellation-driven close
    (GeneratorExit while the event is set) must keep the skip path — the
    fix must only re-route natural exhaustion, not weaken F4."""
    from mlx_soloheaven.engine.qwen_mtp import qwen_mtp_generate_step

    cache = _mk_caches()
    ops = MockOps(lambda t: t + 1, lambda t: t + 1)
    cancel = threading.Event()
    finalized: dict = {}

    gen = qwen_mtp_generate_step(
        mx.array([1, 2, 3], mx.uint32),
        model=None,
        head=None,
        block_size=2,
        max_tokens=16,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: finalized.__setitem__("corrupt", True),
        on_finalize=lambda tok, hid: finalized.update(tok=tok, hid=hid),
        ops=ops,
        cancel_event=cancel,
    )
    next(gen)
    next(gen)
    cancel.set()
    gen.close()  # engine cancel-break: close WITH the event set
    assert finalized == {"tok": None, "hid": None}  # skip path intact


# --- codex round 7 finding 1: EOS survives the ADAPTER teardown ------------------
#
# The production path wraps the QwenMTP runner in _pld_response_adapter. The
# round-6 closed_by_cancel inference fixed DIRECT natural exhaustion, but the
# adapter re-opened the hole: after the adapter yielded the turn's EOS frame
# (engine recorded + delivered it), a client disconnect set cancel_event and
# the engine closed the ADAPTER — the adapter teardown then closed the INNER
# runner, which saw GeneratorExit + event set, classified the close as a
# cancel and skipped finalization. The engine's normal persistence then
# truncated recorded ids to the one-short cache offset while saving the
# COMPLETE assistant message: silent next-HIT suffix corruption. The fix is
# the explicit close-reason channel (GeneratorCloseReason): once the EOS was
# produced the turn is COMPLETE and finalization runs regardless of the
# event.


def _adapter_wrapped_runner(*, eos_id, cancel, finalized, cache=None, ops=None):
    """The production wiring shape: runner + adapter sharing one
    close-reason cell (mirrors the engine's QwenMTP dispatch)."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter
    from mlx_soloheaven.engine.qwen_mtp import (
        GeneratorCloseReason,
        qwen_mtp_generate_step,
    )

    cache = cache if cache is not None else _mk_caches()
    ops = ops if ops is not None else MockOps(lambda t: t + 1, lambda t: t + 1)
    cell = GeneratorCloseReason()
    runner = qwen_mtp_generate_step(
        mx.array([1, 2, 3], mx.uint32),
        model=None,
        head=None,
        block_size=2,
        max_tokens=16,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: finalized.__setitem__("corrupt", True),
        on_finalize=lambda tok, hid: finalized.update(tok=tok, hid=hid),
        ops=ops,
        cancel_event=cancel,
        close_reason=cell,
    )
    tok = SimpleNamespace(eos_token_ids=[eos_id], decode=lambda ids: "x")
    adapter = _pld_response_adapter(
        runner, tok, label="QwenMTP", close_reason=cell,
    )
    return adapter, cache


def _assert_finalized_to_recorded(finalized, cache, frames, *, eos_id):
    """The post-fix contract: finalize RAN — on_finalize reports the resume
    state and the target cache offset equals the engine's recorded ids
    (suffix + every yielded frame token, EOS included); head one behind."""
    recorded = [1, 2, 3] + [f.token for f in frames]
    assert "corrupt" not in finalized
    assert finalized["tok"] == eos_id
    assert finalized["hid"] is not None
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == len(recorded)
    assert cache[5].offset == len(recorded) - 1


def test_adapter_eos_yield_close_with_cancel_still_finalizes():
    """CODEX ROUND 7, FINDING 1 — the exact interleaving: the adapter
    yields the EOS frame and is SUSPENDED AT THAT YIELD; the client
    disconnects (cancel_event set); the engine closes the adapter
    (disconnect-rescue / cancel-break teardown). The inner runner MUST
    finalize: the turn's EOS was already produced and recorded, so the
    close tears down a COMPLETE turn. Pre-fix the adapter teardown closed
    the inner runner without a reason, the runner inferred cancel from the
    set event, skipped the finalize forwards, and the cache landed one
    token short of the recorded ids while the complete assistant message
    was persisted."""
    EOS = 6
    cancel = threading.Event()
    finalized: dict = {}
    adapter, cache = _adapter_wrapped_runner(
        eos_id=EOS, cancel=cancel, finalized=finalized,
    )

    frames = []
    while True:
        f = next(adapter)  # engine pulls + records EVERY frame
        frames.append(f)
        if f.token == EOS:
            break  # adapter now suspended AT its EOS yield
    assert [f.token for f in frames] == [4, 5, EOS]

    cancel.set()      # client disconnect after the EOS frame went out
    adapter.close()   # engine closes the ADAPTER (GeneratorExit at EOS yield)

    _assert_finalized_to_recorded(finalized, cache, frames, eos_id=EOS)


def test_adapter_eos_break_resumption_with_cancel_still_finalizes():
    """Companion interleaving: the engine pulls the frame AFTER the EOS
    frame — the adapter resumes at its EOS yield and takes its natural
    break-on-resumption path (StopIteration) with the event already set.
    The adapter's teardown must mark the close NATURAL and close the inner
    runner explicitly (pre-fix the runner was only GC-closed from the dying
    adapter frame and inferred cancel from the set event)."""
    EOS = 6
    cancel = threading.Event()
    finalized: dict = {}
    adapter, cache = _adapter_wrapped_runner(
        eos_id=EOS, cancel=cancel, finalized=finalized,
    )

    frames = []
    while True:
        f = next(adapter)
        frames.append(f)
        if f.token == EOS:
            break
    cancel.set()  # disconnect between the EOS frame and the next pull
    with pytest.raises(StopIteration):
        next(adapter)  # break-on-resumption: adapter exhausts naturally

    _assert_finalized_to_recorded(finalized, cache, frames, eos_id=EOS)


def test_adapter_midstream_close_with_cancel_still_skips_finalize():
    """Regression guard: a MID-STREAM disconnect (adapter closed BEFORE any
    EOS frame, event set) must keep the cancel semantics — the cell stays
    unmarked, the runner infers cancel, the finalize forwards are skipped
    and on_finalize reports (None, None); the settle still reconciles the
    cache to exactly suffix + yielded - 1."""
    cancel = threading.Event()
    finalized: dict = {}
    adapter, cache = _adapter_wrapped_runner(
        eos_id=99, cancel=cancel, finalized=finalized,  # EOS never reached
    )

    assert next(adapter).token == 4
    assert next(adapter).token == 5
    cancel.set()      # engine observes the cancel ...
    adapter.close()   # ... and closes the adapter mid-stream

    assert finalized == {"tok": None, "hid": None}
    # Settled, not finalized: target holds suffix(3) + yielded(2) - 1 = 4.
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 4
    assert cache[5].offset == 3


def test_qwen_mtp_close_reason_cell_overrides_event_inference():
    """The channel itself, driven directly on the runner: an explicit mark
    OVERRIDES the cancel_event inference in BOTH directions — NATURAL with
    the event set finalizes; CANCEL with the event clear skips."""
    from mlx_soloheaven.engine.qwen_mtp import (
        GeneratorCloseReason,
        qwen_mtp_generate_step,
    )

    def _run_two_tokens(cell, cancel):
        cache = _mk_caches()
        ops = MockOps(lambda t: t + 1, lambda t: t + 1)
        finalized: dict = {}
        gen = qwen_mtp_generate_step(
            mx.array([1, 2, 3], mx.uint32),
            model=None,
            head=None,
            block_size=2,
            max_tokens=16,
            sampler=None,
            logits_processors=None,
            prompt_cache=cache,
            n_target_layers=5,
            on_cache_corruption=lambda: finalized.__setitem__("corrupt", True),
            on_finalize=lambda tok, hid: finalized.update(tok=tok, hid=hid),
            ops=ops,
            cancel_event=cancel,
            close_reason=cell,
        )
        toks = [next(gen)[0], next(gen)[0]]
        return gen, cache, finalized, toks

    # NATURAL beats a set event: finalize runs.
    cell = GeneratorCloseReason()
    cancel = threading.Event()
    gen, cache, finalized, toks = _run_two_tokens(cell, cancel)
    cancel.set()
    cell.mark_natural()
    gen.close()
    assert finalized["tok"] == toks[-1]
    assert finalized["hid"] is not None
    for c in cache[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 5  # suffix(3) + yielded(2): finalized

    # CANCEL beats a clear event: finalize forwards skipped.
    cell2 = GeneratorCloseReason()
    gen2, cache2, finalized2, _ = _run_two_tokens(cell2, threading.Event())
    cell2.mark_cancel()
    gen2.close()
    assert finalized2 == {"tok": None, "hid": None}
    for c in cache2[:5]:
        if hasattr(c, "offset"):
            assert c.offset == 4  # settled only: suffix(3) + yielded(2) - 1
    assert "corrupt" not in finalized
