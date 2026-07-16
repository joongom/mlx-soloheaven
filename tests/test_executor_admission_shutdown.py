"""F2 residual + executor shutdown lifecycle (batch 3, codex rounds 2+3).

READ ADMISSION (#4): the reserved read pool alone was not enough — two
10s-busy reads pin both workers and further reads queued UNBOUNDED, their
own timeout starting only once a worker freed up ("reads start promptly"
was false under concurrent load). run_read now has an admission bound
(fail-fast EngineBusyError past workers + a small backlog) and an
END-TO-END deadline measured from SUBMISSION (queue wait + execution); a
read that expires while still queued is cancelled and never runs late.

LONG ADMISSION (round 3, finding 5): run_long is bounded too — streaming
generation does NOT use run_long, so nothing legitimate needs a deep
backlog; past LONG_ADMISSION_LIMIT the queued entries would execute
arbitrarily late (long ops are minutes-scale) and fail fast instead. NO
end-to-end deadline: a started long op legitimately runs for minutes.

SHUTDOWN LIFECYCLE (#6): the pools used to live forever — queued mutations
could execute AFTER the server-shutdown engine flush, and interpreter exit
joined the worker threads. The server hook now runs quiesce() (stop
admission + cancel queued futures + bounded wait for running calls) BEFORE
the engine flush and shutdown_pools() AFTER it, all idempotent.

ATOMIC ADMISSION (round 3, finding 1): the admission check, pool submit and
pending-registry insert used to be three separate steps — a request could
pass the check, quiesce could then set the flag and snapshot an EMPTY
pending set, and the request submitted + tracked afterwards: the mutation
ran AFTER the shutdown flush while quiesce reported still_running=0. Now
all three run under ONE module lock and quiesce takes flag + snapshot under
the SAME lock, making the interleaving impossible.

ADMISSION RESERVATION (codex round 11, finding 1 — HIGH): run_long's
fail-fast admission runs AFTER several endpoints already mutated the DB
(regenerate deletes the assistant turn, delete-last removes the pair,
branch creates+populates a durable session, non-streaming chat persists
the user message, compaction inserts the summary) — saturation then left
the DB mutated with ZERO engine calls. ensure_available() (the batch-1
preflight) does NOT reserve executor capacity. reserve_long_slot() now
performs the same admission check+increment atomically BEFORE the DB step;
run_long(..., reservation=...) consumes the slot instead of re-admitting.

POISON-INSTEAD-OF-CANCEL (codex round 11, finding 2 — MED): the round-10
ownership check (task.cancelling() > 0) is blind to Task.cancel() +
Task.uncancel(), which drops cancelling() to 0 while the already-scheduled
CancelledError still delivers — a task-originated cancellation again
converted to the quiesce-labelled EngineBusyError. Shutdown now NEVER
communicates critical-lane rejection through Future.cancel(): quiesce /
shutdown_pools poison the commit's cell, the wrapper raises the internal
_ShutdownRejected INSIDE the pool, and the awaiter converts that ordinary
future exception to EngineBusyError. A CancelledError in run_critical is
therefore ALWAYS awaiter/task-originated and always propagates.

HERMETIC — pure executor/asyncio tests, no engine, no pipes (conventions
from tests/test_rpc_nonblocking.py's F2 section); the endpoint-level
reservation tests reuse test_worker_liveness.py's stub-DB/TestClient shape.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mlx_soloheaven import executors
from mlx_soloheaven.engine.types import EngineBusyError


@pytest.fixture(autouse=True)
def fresh_executors():
    """Every test starts (and leaves) pristine module state — the shutdown
    tests flip module-global flags that must never leak into the suite."""
    executors._reset_for_tests()
    yield
    executors._reset_for_tests()


# --- #4: read admission bound ------------------------------------------------------


def test_run_read_over_admission_limit_fails_fast():
    """Saturate the read pool AND its backlog: the next read fails FAST with
    EngineBusyError instead of queueing unboundedly (pre-fix its own
    timeout only started once a worker freed up)."""
    release = threading.Event()

    def _busy():
        release.wait(timeout=10.0)
        return "done"

    async def _drive():
        admitted = [
            asyncio.ensure_future(executors.run_read(_busy))
            for _ in range(executors.READ_ADMISSION_LIMIT)
        ]
        await asyncio.sleep(0.1)  # workers occupied + backlog queued
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError, match="saturated"):
            await executors.run_read(lambda: "late")
        dt = time.monotonic() - t0
        release.set()
        results = await asyncio.gather(*admitted)
        return dt, results

    dt, results = asyncio.run(_drive())
    assert dt < 1.0  # fail-FAST, not after any deadline
    assert results == ["done"] * executors.READ_ADMISSION_LIMIT


def test_run_read_admission_slots_recycle():
    """Completed reads free their admission slots — sequential traffic far
    beyond the limit keeps flowing."""

    async def _drive():
        out = []
        for i in range(executors.READ_ADMISSION_LIMIT * 2):
            out.append(await executors.run_read(lambda i=i: i))
        return out

    n = executors.READ_ADMISSION_LIMIT * 2
    assert asyncio.run(_drive()) == list(range(n))
    assert executors._read_admitted == 0  # every slot returned


def test_run_read_deadline_is_end_to_end_and_queued_task_never_runs(monkeypatch):
    """The deadline is measured from SUBMISSION (queue wait counts): a read
    stuck behind two busy workers expires on time with EngineBusyError AND
    is cancelled — it must never execute once the workers free up."""
    monkeypatch.setattr(executors, "READ_DEADLINE_S", 0.3)
    release = threading.Event()
    ran: list = []

    def _busy():
        release.wait(timeout=10.0)
        return "busy-done"

    async def _drive():
        blockers = [
            asyncio.ensure_future(executors.run_read(_busy))
            for _ in range(executors.READ_EXECUTOR_MAX_WORKERS)
        ]
        await asyncio.sleep(0.05)  # both workers occupied
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError, match="not served within"):
            await executors.run_read(lambda: ran.append(1))
        dt = time.monotonic() - t0

        release.set()
        # The blockers exceeded the shrunken deadline too — drain them.
        await asyncio.gather(*blockers, return_exceptions=True)
        # Prove the expired-in-queue read never runs: a fresh read completes
        # (pool FIFO would have executed the stale one first) while `ran`
        # stays empty.
        monkeypatch.setattr(executors, "READ_DEADLINE_S", 5.0)
        after = await executors.run_read(lambda: "after")
        return dt, after

    dt, after = asyncio.run(_drive())
    assert 0.2 < dt < 2.0  # expired from submission, not from start-of-run
    assert after == "after"
    assert ran == []  # the cancelled queued read NEVER executed


def test_run_long_burst_within_bound_all_completes():
    """A legitimate burst (several concurrent non-streaming completions +
    admin mutations, well under LONG_ADMISSION_LIMIT) queues and ALL of it
    completes — the round-3 bound must not break normal flows."""
    n = executors.LONG_EXECUTOR_MAX_WORKERS * 3
    assert n < executors.LONG_ADMISSION_LIMIT

    async def _drive():
        return await asyncio.gather(
            *[executors.run_long(lambda i=i: i) for i in range(n)]
        )

    assert sorted(asyncio.run(_drive())) == list(range(n))


# --- round 3 finding 5: run_long bounded backlog -----------------------------------


def test_run_long_over_admission_limit_fails_fast():
    """Saturate the long pool AND its backlog: the next long call fails FAST
    with EngineBusyError (server 503 backstop) instead of queueing
    unboundedly — pre-fix a burst of non-streaming completions / mutating
    admin calls queued futures that executed arbitrarily late."""
    release = threading.Event()

    def _busy():
        release.wait(timeout=30.0)
        return "done"

    async def _drive():
        admitted = [
            asyncio.ensure_future(executors.run_long(_busy))
            for _ in range(executors.LONG_ADMISSION_LIMIT)
        ]
        await asyncio.sleep(0.1)  # workers occupied + backlog queued
        t0 = time.monotonic()
        with pytest.raises(EngineBusyError, match="saturated"):
            await executors.run_long(lambda: "late")
        dt = time.monotonic() - t0
        release.set()
        results = await asyncio.gather(*admitted)
        return dt, results

    dt, results = asyncio.run(_drive())
    assert dt < 1.0  # fail-FAST — run_long has no deadline, only admission
    assert results == ["done"] * executors.LONG_ADMISSION_LIMIT


def test_run_long_admission_slots_recycle():
    """Completed long calls free their admission slots — sequential traffic
    far beyond the limit keeps flowing."""

    async def _drive():
        out = []
        for i in range(executors.LONG_ADMISSION_LIMIT + 4):
            out.append(await executors.run_long(lambda i=i: i))
        return out

    n = executors.LONG_ADMISSION_LIMIT + 4
    assert asyncio.run(_drive()) == list(range(n))
    assert executors._long_admitted == 0  # every slot returned


# --- round 5 finding 1: the CRITICAL post-stream commit lane ------------------------


def test_run_critical_bypasses_saturated_long_pool():
    """codex round-5 repro: 20 long calls saturate run_long's admission
    (streaming holds the engine lock, so they block/queue for minutes). The
    MANDATORY post-stream commit used to be rejected there — mid-stream, no
    503 possible. run_critical must complete promptly on its reserved
    worker while the long pool is fully saturated."""
    release = threading.Event()
    committed: list = []

    def _busy():
        release.wait(timeout=30.0)
        return "done"

    async def _drive():
        admitted = [
            asyncio.ensure_future(executors.run_long(_busy))
            for _ in range(executors.LONG_ADMISSION_LIMIT)
        ]
        await asyncio.sleep(0.1)  # workers occupied + backlog full
        # The next long call still fails fast (saturation is real) ...
        with pytest.raises(EngineBusyError, match="saturated"):
            await executors.run_long(lambda: "late")
        # ... but the critical commit is NOT droppable and NOT queued
        # behind the blocked long workers.
        t0 = time.monotonic()
        out = await executors.run_critical(lambda: committed.append(1) or "ok")
        dt = time.monotonic() - t0
        release.set()
        await asyncio.gather(*admitted)
        return out, dt

    out, dt = asyncio.run(_drive())
    assert out == "ok"
    assert committed == [1]
    assert dt < 1.0  # ran immediately — never behind the saturated pool


def test_run_critical_allowed_through_stop_admission_until_pools_down():
    """Shutdown contract (round 5, finding 1): a stream that is just
    finishing during a graceful shutdown still needs its commit —
    run_critical passes stop_admission (the final flush runs AFTER quiesce,
    so the commit's dirty-mark still lands in time) but is refused once
    shutdown_pools() tore the pools down (state already flushed; nothing
    may resurrect a pool)."""
    executors.stop_admission()

    async def _drive():
        # Normal lanes reject ...
        with pytest.raises(EngineBusyError, match="shutting down"):
            await executors.run_long(lambda: 1)
        # ... the critical commit still runs.
        return await executors.run_critical(lambda: "committed")

    assert asyncio.run(_drive()) == "committed"

    executors.shutdown_pools()

    async def _after_teardown():
        with pytest.raises(EngineBusyError, match="shut down"):
            await executors.run_critical(lambda: 1)

    asyncio.run(_after_teardown())


def test_run_critical_visible_to_quiesce():
    """Critical futures are TRACKED in the pending registry: quiesce's
    bounded wait sees a running commit (still_running) instead of ignoring
    it — the commit is never invisible to shutdown accounting."""
    release = threading.Event()
    entered = threading.Event()

    async def _drive():
        fut = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        stats = await asyncio.to_thread(executors.quiesce, 0.2)
        release.set()
        out = await fut
        return stats, out

    stats, out = asyncio.run(_drive())
    assert out == "ok"
    assert stats["still_running"] == 1


# --- codex round 7 finding 3: quiesce marker vs awaiter-task cancellation ----------


def test_run_critical_task_cancellation_propagates_not_busy():
    """CODEX ROUND 7, FINDING 3 repro: one BLOCKED critical op + a second
    QUEUED one; cancelling the second AWAITING TASK must propagate
    CancelledError. Pre-fix the awaiter read fut.cancelled() as
    quiesce-did-it — but a task cancel also reaches the queued pool future
    through asyncio.wrap_future — and converted the real cancellation into
    the quiesce-labelled EngineBusyError, which chat.py/openai_compat.py's
    broad post-stream handlers then swallowed."""
    release = threading.Event()
    entered = threading.Event()
    ran: list = []

    async def _drive():
        blocker = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        queued = asyncio.ensure_future(
            executors.run_critical(lambda: ran.append(1) or "queued")
        )
        await asyncio.sleep(0.05)  # queued behind the single critical worker

        queued.cancel()  # the AWAITING TASK is cancelled — NOT quiesce
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        release.set()
        out = await blocker
        return queued, queued_result, out

    queued_task, queued_result, out = asyncio.run(_drive())
    assert out == "ok"
    assert ran == []  # the queued op never executed
    # CancelledError PROPAGATED: the task ended cancelled — pre-fix it ended
    # with the quiesce-labelled EngineBusyError instead.
    assert queued_task.cancelled()
    assert isinstance(queued_result, asyncio.CancelledError)
    assert not isinstance(queued_result, EngineBusyError)


def test_run_critical_task_cancel_plus_uncancel_never_busy():
    """CODEX ROUND 11 MED repro: Task.cancel(); Task.uncancel() drops
    task.cancelling() to 0 while the already-scheduled CancelledError is
    still delivered — the round-10 ownership check cannot see this
    cancellation, and a concurrent quiesce that tagged + cancelled the
    still-queued pool future made the awaiter convert the GENUINE
    task-originated CancelledError into the quiesce-labelled
    EngineBusyError. Deterministic interleaving WITHOUT monkeypatched
    pauses: cancel + uncancel the awaiting task, then run quiesce
    SYNCHRONOUSLY before yielding to the event loop — the wrap_future
    cancel-chain needs a loop tick, so pre-fix quiesce still saw the pool
    future pending and tagged + cancelled it, producing exactly the
    ambiguous tagged+cancelled+cancelling()==0 state. Post-fix shutdown
    never cancels critical-lane futures (poison instead), so the
    CancelledError is unambiguous and propagates."""
    release = threading.Event()
    entered = threading.Event()
    ran: list = []

    async def _drive():
        blocker = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        queued = asyncio.ensure_future(
            executors.run_critical(lambda: ran.append(1) or "queued")
        )
        await asyncio.sleep(0.05)  # queued behind the single critical worker

        # SYNCHRONOUS window — no await between these three steps: the task
        # cancellation is recorded (delivery happens on the next tick),
        # uncancel() zeroes task.cancelling(), and quiesce runs while the
        # pool future is still pending.
        queued.cancel()
        queued.uncancel()
        stats = executors.quiesce(timeout_s=0.05)

        release.set()
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        out = await blocker
        return queued, queued_result, out, stats

    queued_task, queued_result, out, stats = asyncio.run(_drive())
    assert out == "ok"
    assert ran == []  # the queued op never executed
    # The task-originated cancellation PROPAGATED — pre-fix it surfaced as
    # the quiesce-labelled EngineBusyError (misclassification).
    assert not isinstance(queued_result, EngineBusyError)
    assert isinstance(queued_result, asyncio.CancelledError)
    assert queued_task.cancelled()
    # Shutdown accounting: the critical lane is POISONED, never cancelled.
    assert stats["cancelled"] == 0
    assert stats["poisoned_critical"] == 1


def test_run_critical_task_cancel_racing_quiesce_propagates():
    """Ordinary (no-uncancel) task cancellation racing quiesce: the same
    synchronous interleaving as the uncancel repro, but with
    task.cancelling() still > 0 — CancelledError must propagate under the
    poison design exactly as it did under the round-10 ownership check."""
    release = threading.Event()
    entered = threading.Event()
    ran: list = []

    async def _drive():
        blocker = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        queued = asyncio.ensure_future(
            executors.run_critical(lambda: ran.append(1) or "queued")
        )
        await asyncio.sleep(0.05)  # queued behind the single critical worker

        queued.cancel()  # the AWAITING TASK is cancelled, racing quiesce
        stats = executors.quiesce(timeout_s=0.05)

        release.set()
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        out = await blocker
        return queued, queued_result, out, stats

    queued_task, queued_result, out, stats = asyncio.run(_drive())
    assert out == "ok"
    assert ran == []  # the queued op never executed
    assert queued_task.cancelled()
    assert isinstance(queued_result, asyncio.CancelledError)
    assert not isinstance(queued_result, EngineBusyError)
    assert stats["cancelled"] == 0  # quiesce never cancels the critical lane


def test_run_critical_quiesce_rejection_still_degrades_to_busy():
    """The other half of the contract (guards the round-5 behavior): a
    QUEUED critical commit rejected by quiesce still degrades to the
    EngineBusyError shape callers log-and-continue on, so the stream's
    terminal event goes out. Round 11 (finding 2): the rejection is now a
    POISON — quiesce never cancels the pool future; the wrapper raises the
    internal sentinel INSIDE the pool the moment the worker reaches it
    (immediately after the blocker frees the 1-worker lane — the poisoned
    op 'runs' only to reject) and the awaiter converts that ordinary
    future exception to EngineBusyError."""
    release = threading.Event()
    entered = threading.Event()
    ran: list = []

    async def _drive():
        blocker = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        queued = asyncio.ensure_future(
            executors.run_critical(lambda: ran.append(1) or "queued")
        )
        await asyncio.sleep(0.05)  # queued behind the single critical worker

        stats = await asyncio.to_thread(executors.quiesce, 0.2)
        # Unblock the lane so the poisoned wrapper reaches the worker and
        # raises; only then can its awaiter observe the degraded shape.
        release.set()
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        out = await blocker
        return stats, queued_result, out

    stats, queued_result, out = asyncio.run(_drive())
    assert out == "ok"  # a STARTED commit is never poisoned
    assert ran == []  # the queued op's real payload never executed
    assert stats["cancelled"] == 0
    assert stats["poisoned_critical"] == 1
    assert isinstance(queued_result, EngineBusyError)
    assert "quiesce" in str(queued_result)


def test_shutdown_pools_rejects_queued_critical_as_busy():
    """Teardown backstop (round 11, finding 2): shutdown_pools also POISONS
    queued critical commits instead of cancelling them — the critical pool
    is torn down WITHOUT cancel_futures, so the poisoned wrapper still runs
    on the drained queue, raises inside the pool, and the awaiter degrades
    to EngineBusyError (never an ambiguous CancelledError)."""
    release = threading.Event()
    entered = threading.Event()
    ran: list = []

    async def _drive():
        blocker = asyncio.ensure_future(
            executors.run_critical(
                lambda: (entered.set(), release.wait(timeout=10.0), "ok")[-1]
            )
        )
        await asyncio.to_thread(entered.wait, 5.0)
        queued = asyncio.ensure_future(
            executors.run_critical(lambda: ran.append(1) or "queued")
        )
        await asyncio.sleep(0.05)  # queued behind the single critical worker

        await asyncio.to_thread(executors.shutdown_pools)
        release.set()
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        out = await blocker
        return queued_result, out

    queued_result, out = asyncio.run(_drive())
    assert out == "ok"  # already-running work is never poisoned
    assert ran == []  # the queued op's real payload never executed
    assert isinstance(queued_result, EngineBusyError)
    assert not isinstance(queued_result, asyncio.CancelledError)


def test_run_long_and_read_task_cancellation_propagates():
    """Audit pin (finding 3): run_long / run_read never had the
    cancelled()-means-quiesce conversion — an awaiting-task cancellation
    must keep propagating as CancelledError there too (the marker fix must
    not leak a busy-conversion into these lanes)."""
    release = threading.Event()
    ran: list = []

    async def _drive(run_fn, workers):
        blockers = [
            asyncio.ensure_future(run_fn(lambda: release.wait(timeout=10.0)))
            for _ in range(workers)
        ]
        await asyncio.sleep(0.05)  # every worker occupied
        queued = asyncio.ensure_future(run_fn(lambda: ran.append(1)))
        await asyncio.sleep(0.05)  # queued
        queued.cancel()
        result = (await asyncio.gather(queued, return_exceptions=True))[0]
        release.set()
        await asyncio.gather(*blockers, return_exceptions=True)
        return queued, result

    task, result = asyncio.run(
        _drive(executors.run_long, executors.LONG_EXECUTOR_MAX_WORKERS)
    )
    assert task.cancelled()
    assert isinstance(result, asyncio.CancelledError)

    release.clear()
    task, result = asyncio.run(
        _drive(executors.run_read, executors.READ_EXECUTOR_MAX_WORKERS)
    )
    assert task.cancelled()
    assert isinstance(result, asyncio.CancelledError)
    assert ran == []  # neither queued op ever executed


# --- round 3 finding 1: admission is ATOMIC against quiesce ------------------------


def test_shutdown_admission_toctou_closed(monkeypatch):
    """codex round-3 repro, made deterministic: a submission passes the
    admission check, then STALLS before submitting while quiesce() runs.
    Pre-fix quiesce set the flag + snapshotted an EMPTY pending set and the
    stalled request submitted + tracked afterwards — the work executed
    after the shutdown flush with quiesce reporting still_running=0. With
    the fix the check+submit+track section holds the module lock quiesce
    needs, so quiesce BLOCKS until the admission settles and its snapshot
    always contains the admitted future (cancelled or bounded-waited)."""
    in_admission = threading.Event()
    release_admission = threading.Event()
    real_check = executors._check_admission

    def stalling_check(kind):
        real_check(kind)
        in_admission.set()
        # Hold the admission section open (the fix runs this under the
        # module lock; pre-fix nothing stopped quiesce from interleaving).
        assert release_admission.wait(timeout=10.0)

    monkeypatch.setattr(executors, "_check_admission", stalling_check)

    work_release = threading.Event()
    ran: list = []

    def _work():
        work_release.wait(timeout=10.0)
        ran.append(1)

    submit_outcome: dict = {}

    def _submitter():
        try:
            submit_outcome["ok"] = asyncio.run(executors.run_long(_work))
        except (EngineBusyError, asyncio.CancelledError) as e:
            submit_outcome["rejected"] = e

    q_stats: dict = {}

    def _quiescer():
        q_stats.update(executors.quiesce(timeout_s=0.5))

    t_sub = threading.Thread(target=_submitter, daemon=True)
    t_sub.start()
    assert in_admission.wait(timeout=10.0)

    t_q = threading.Thread(target=_quiescer, daemon=True)
    t_q.start()
    time.sleep(0.2)
    # Quiesce must NOT have completed while the admission section is open —
    # it is blocked on the same lock (the atomicity under test).
    assert not q_stats, "quiesce interleaved with an in-flight admission"

    release_admission.set()
    t_q.join(timeout=10.0)
    assert not t_q.is_alive()

    # The admitted future was VISIBLE to quiesce: cancelled while queued or
    # counted still running — NEVER the pre-fix empty snapshot.
    assert q_stats["cancelled"] + q_stats["still_running"] == 1

    work_release.set()
    t_sub.join(timeout=10.0)
    assert not t_sub.is_alive()
    if q_stats["cancelled"]:
        # Cancelled while queued: it must never execute.
        assert ran == []
        assert "rejected" in submit_outcome
    else:
        # It had already started: quiesce reported it truthfully as still
        # running (the engine-side gate — finding 2 — neutralizes it).
        assert q_stats["still_running"] == 1


def test_submission_after_quiesce_always_rejected():
    """The other half of the atomicity: once quiesce holds/held the lock,
    ANY later submission fails the (now in-lock) admission check — no
    untracked future can slip in behind the flush."""
    executors.quiesce(timeout_s=0.1)

    async def _drive():
        with pytest.raises(EngineBusyError, match="shutting down"):
            await executors.run_long(lambda: 1)
        with pytest.raises(EngineBusyError, match="shutting down"):
            await executors.run_read(lambda: 1)

    asyncio.run(_drive())


# --- #6: shutdown lifecycle ---------------------------------------------------------


def test_stop_admission_rejects_new_submissions():
    executors.stop_admission()

    async def _drive():
        with pytest.raises(EngineBusyError, match="shutting down"):
            await executors.run_read(lambda: 1)
        with pytest.raises(EngineBusyError, match="shutting down"):
            await executors.run_long(lambda: 1)

    asyncio.run(_drive())


def test_quiesce_cancels_queued_work_before_flush():
    """The core #6 guarantee: a QUEUED (not-yet-started) mutation is
    cancelled by quiesce and NEVER executes afterwards — pre-fix it would
    have run after the shutdown flush persisted engine state. Running calls
    are bounded-waited and reported."""
    release = threading.Event()
    queued_ran: list = []

    async def _drive():
        blockers = [
            asyncio.ensure_future(
                executors.run_long(lambda: release.wait(timeout=10.0))
            )
            for _ in range(executors.LONG_EXECUTOR_MAX_WORKERS)
        ]
        await asyncio.sleep(0.05)  # every long worker occupied
        queued = asyncio.ensure_future(
            executors.run_long(lambda: queued_ran.append(1))
        )
        await asyncio.sleep(0.05)  # queued behind the blockers

        stats = await asyncio.to_thread(executors.quiesce, 0.2)

        release.set()
        await asyncio.gather(*blockers)
        queued_result = (
            await asyncio.gather(queued, return_exceptions=True)
        )[0]
        return stats, queued_result

    stats, queued_result = asyncio.run(_drive())
    assert stats["cancelled"] == 1  # exactly the queued mutation
    assert stats["still_running"] == executors.LONG_EXECUTOR_MAX_WORKERS
    assert queued_ran == []  # it NEVER executed
    assert isinstance(queued_result, asyncio.CancelledError)


def test_quiesce_and_shutdown_pools_idempotent():
    executors.quiesce()
    executors.quiesce()  # second run is a no-op
    executors.shutdown_pools()
    executors.shutdown_pools()  # second teardown is a no-op

    # After a reset, the pools work again (test-isolation contract).
    executors._reset_for_tests()

    async def _drive():
        return await executors.run_read(lambda: "ok")

    assert asyncio.run(_drive()) == "ok"


def test_callers_degrade_busy_when_admission_stopped():
    """Callers already handle the admission-stopped failure: the web
    cache-stats endpoint answers its 'busy' placeholder instead of a 500."""
    from mlx_soloheaven.api import chat

    class Engine:
        cache_manager = SimpleNamespace(stats=lambda: {})

        def session_stats(self):
            return {}

    executors.stop_admission()
    old_engine = chat.engine
    try:
        chat.set_engine(Engine())
        out = asyncio.run(chat.cache_stats())
    finally:
        chat.set_engine(old_engine)
    assert out == {"busy": True}


# --- codex round 11, finding 1: LONG-pool admission reservation ---------------------


def test_reserve_long_slot_counts_and_releases():
    """Reservations share the SAME admission counter as run_long: holding
    every slot saturates both APIs; releasing restores both. release() is
    idempotent (never double-decrements)."""
    slots = [
        executors.reserve_long_slot()
        for _ in range(executors.LONG_ADMISSION_LIMIT)
    ]
    assert executors._long_admitted == executors.LONG_ADMISSION_LIMIT
    with pytest.raises(EngineBusyError, match="saturated"):
        executors.reserve_long_slot()
    with pytest.raises(EngineBusyError, match="saturated"):
        asyncio.run(executors.run_long(lambda: 1))

    for s in slots:
        s.release()
        s.release()  # idempotent
    assert executors._long_admitted == 0
    assert asyncio.run(executors.run_long(lambda: "ok")) == "ok"


def test_reservation_context_manager_releases_on_error():
    """The endpoint shape: anything thrown between the reserve and the
    submit (a failing DB step) releases the slot via the context manager."""
    with pytest.raises(RuntimeError, match="db step failed"):
        with executors.reserve_long_slot():
            assert executors._long_admitted == 1
            raise RuntimeError("db step failed")
    assert executors._long_admitted == 0


def test_run_long_consumes_reservation_no_double_count():
    """run_long(..., reservation=...) consumes the pre-acquired slot instead
    of re-admitting; afterwards the future's done callback releases it and
    the context-manager exit is a no-op — accounting lands exactly at 0."""

    async def _drive():
        with executors.reserve_long_slot() as slot:
            assert executors._long_admitted == 1
            return await executors.run_long(lambda: "ran", reservation=slot)

    assert asyncio.run(_drive()) == "ran"
    assert executors._long_admitted == 0


def test_run_long_with_reservation_bypasses_saturation_recheck():
    """A reservation admitted BEFORE saturation must still submit while the
    pool is saturated (its slot is already counted — re-admitting would
    reject the endpoint against its own reservation)."""

    async def _drive():
        with executors.reserve_long_slot() as slot:
            others = [
                executors.reserve_long_slot()
                for _ in range(executors.LONG_ADMISSION_LIMIT - 1)
            ]
            try:
                with pytest.raises(EngineBusyError, match="saturated"):
                    await executors.run_long(lambda: "plain")
                return await executors.run_long(
                    lambda: "reserved", reservation=slot
                )
            finally:
                for o in others:
                    o.release()

    assert asyncio.run(_drive()) == "reserved"
    assert executors._long_admitted == 0


def test_run_long_reservation_shutdown_gate_leaves_reservation_held():
    """stop_admission between the reserve and the submit: run_long must
    still reject (a mutation may never land behind the shutdown flush) and
    leave the reservation HELD so the caller's context manager releases it."""

    async def _drive():
        with executors.reserve_long_slot() as slot:
            executors.stop_admission()
            with pytest.raises(EngineBusyError, match="shutting down"):
                await executors.run_long(lambda: 1, reservation=slot)

    asyncio.run(_drive())
    assert executors._long_admitted == 0


def test_reserved_run_long_releases_slot_on_error_and_cancel():
    """The consumed slot is released by the done callback on EVERY terminal
    path — an op that raises, and a queued future cancelled by its awaiting
    task — never leaking admission capacity."""

    def _boom():
        raise ValueError("boom")

    async def _err():
        with executors.reserve_long_slot() as slot:
            with pytest.raises(ValueError, match="boom"):
                await executors.run_long(_boom, reservation=slot)

    asyncio.run(_err())
    assert executors._long_admitted == 0

    release = threading.Event()

    async def _cancel():
        blockers = [
            asyncio.ensure_future(
                executors.run_long(lambda: release.wait(timeout=10.0))
            )
            for _ in range(executors.LONG_EXECUTOR_MAX_WORKERS)
        ]
        await asyncio.sleep(0.05)  # every worker occupied
        with executors.reserve_long_slot() as slot:
            queued = asyncio.ensure_future(
                executors.run_long(lambda: "q", reservation=slot)
            )
            await asyncio.sleep(0.05)  # queued behind the blockers
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
        release.set()
        await asyncio.gather(*blockers)

    asyncio.run(_cancel())
    assert executors._long_admitted == 0


def test_reservation_cannot_be_consumed_twice():
    """Reservation misuse (a second run_long against a consumed slot) fails
    loudly BEFORE submitting anything."""

    async def _drive():
        with executors.reserve_long_slot() as slot:
            await executors.run_long(lambda: 1, reservation=slot)
            with pytest.raises(RuntimeError, match="consumed"):
                await executors.run_long(lambda: 2, reservation=slot)

    asyncio.run(_drive())
    assert executors._long_admitted == 0


# --- codex round 11, finding 1: endpoints reserve BEFORE their DB mutations ---------


@contextmanager
def _saturated_long_pool():
    """Simulate a fully saturated LONG pool by occupying every admission
    slot — the counter is the single admission source of truth for both
    run_long and reserve_long_slot, so this is exactly the state a burst of
    in-flight long calls produces, without threads."""
    with executors._executor_lock:
        executors._long_admitted += executors.LONG_ADMISSION_LIMIT
    try:
        yield
    finally:
        with executors._executor_lock:
            executors._long_admitted -= executors.LONG_ADMISSION_LIMIT


class _EndpointStubDB:
    """Async storage.database stand-in recording every mutation (shape from
    test_worker_liveness.py's _RecordingStubDB)."""

    def __init__(self, messages=None, raise_on_delete=False):
        self.writes: list[tuple] = []
        self._messages = messages if messages is not None else [
            {"role": "user", "content": "q", "token_count": 1},
            {"role": "assistant", "content": "a", "token_count": 1},
        ]
        # U25 round 2: rows carry ids (delete-last deletes by id).
        self._messages = [
            {**m, "id": m.get("id", f"m{i}")}
            for i, m in enumerate(self._messages)
        ]
        self._raise_on_delete = raise_on_delete

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

    # writes (must NOT happen when admission is saturated)
    async def add_message(self, *a, **k):
        self.writes.append(("add_message", a))
        # Round 4 (batch 4, finding 2): the chat endpoint threads the user
        # row's id through the request — return the real function's row
        # shape.
        return {"id": f"w{len(self.writes)}", "role": a[1] if len(a) > 1 else None}

    async def add_message_if_parent_exists(self, *a, **k):
        # Round 4 (finding 2): conditional assistant insert. This stub does
        # not delete the parent mid-flow, so it always reports inserted.
        self.writes.append(("add_message_if_parent_exists", a))
        return {"id": f"w{len(self.writes)}", "parent_id": k.get("parent_id")}

    async def create_session(self, *a, **k):
        self.writes.append(("create_session", a))
        return {"id": "branch-id"}

    async def delete_last_message(self, *a, **k):
        if self._raise_on_delete:
            raise RuntimeError("db exploded")
        self.writes.append(("delete_last_message", a))

    async def delete_messages_by_ids(self, session_id, ids):
        # U25 round 2: id-based atomic turn delete. Must NOT run while the
        # pool is saturated (writes recorded like every other mutation);
        # actually removes the rows so the handler's fresh re-read sees the
        # post-delete state (no-drift fast path).
        if self._raise_on_delete:
            raise RuntimeError("db exploded")
        self.writes.append(("delete_messages_by_ids", (session_id, list(ids))))
        id_set = set(ids)
        self._messages = [m for m in self._messages if m["id"] not in id_set]
        return len(id_set)

    async def delete_last_turn_tx(self, session_id, compute_delete_ids):
        # Round 6 (codex round 5, finding 1): delete-last is now the
        # one-transaction snapshot + walk + delete. Must NOT run while the
        # pool is saturated (recorded like every other mutation).
        if self._raise_on_delete:
            raise RuntimeError("db exploded")
        snapshot = [dict(m) for m in self._messages]
        ids = list(compute_delete_ids(snapshot) or [])
        self.writes.append(("delete_last_turn_tx", (session_id, ids)))
        id_set = set(ids)
        self._messages = [m for m in self._messages if m["id"] not in id_set]
        return ids, [m for m in snapshot if m["id"] not in id_set]

    async def update_session_tokens(self, *a, **k):
        self.writes.append(("update_session_tokens", a))

    async def record_compaction(self, *a, **k):
        self.writes.append(("record_compaction", a))


class _LiveEngineStub:
    """Engine stub whose availability preflight passes (no ensure_available
    attribute) and whose mutating RPCs record calls — reaching one while the
    pool is saturated means the reservation was missing."""

    model_id = "stub-model"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(
            enable_thinking=False,
            default_temperature=0.6, default_top_p=1.0, default_min_p=0.0,
            default_top_k=0, default_repetition_penalty=1.0,
            thinking_budget=0, default_max_tokens=64,
        )
        self.calls: list[str] = []

    def prepare_regenerate(self, session_id):
        self.calls.append("prepare_regenerate")
        return {"restored": True}

    def truncate_session(self, session_id, n):
        self.calls.append("truncate_session")
        return {"truncated_to": n}

    def branch_from_turn(self, src, dst, turn, branch_messages=None):
        self.calls.append("branch_from_turn")
        return {"cached_tokens": 0, "method": "none"}

    def complete(self, messages, session_id=None):
        self.calls.append("complete")
        return SimpleNamespace(
            finish_reason="stop", content="hi", thinking=None,
            prompt_tokens=1, completion_tokens=1,
        )

    def update_session_messages(self, *a, **k):
        self.calls.append("update_session_messages")
        return True


def _reservation_test_client(monkeypatch, stub_db, engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mlx_soloheaven.api import chat as chat_mod
    from mlx_soloheaven.server import register_engine_error_handlers

    monkeypatch.setattr(chat_mod, "db", stub_db)
    monkeypatch.setattr(chat_mod, "engine", engine)
    monkeypatch.setattr(chat_mod, "_engines", {})

    app = FastAPI()
    register_engine_error_handlers(app)
    app.include_router(chat_mod.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("endpoint,payload", [
    ("/api/sessions/s1/regenerate", None),
    ("/api/sessions/s1/delete-last", None),
    ("/api/sessions/s1/branch", {"turn": 2}),
])
def test_endpoint_saturated_503_db_untouched(monkeypatch, endpoint, payload):
    """CODEX ROUND 11 HIGH repro: with the long pool saturated, the three
    mutate-then-submit endpoints must answer a clean 503 BEFORE touching
    the DB. Pre-fix, admission failed only at the run_long AFTER the
    mutations: regenerate stranded a dangling user turn, delete-last
    removed a pair a retry would double-delete, branch orphaned a durable
    half-built session — all with ZERO engine calls."""
    stub_db = _EndpointStubDB()
    eng = _LiveEngineStub()
    client = _reservation_test_client(monkeypatch, stub_db, eng)

    with _saturated_long_pool():
        if payload is None:
            r = client.post(endpoint)
        else:
            r = client.post(endpoint, json=payload)

    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_busy"
    assert stub_db.writes == []  # DB untouched (pre-fix: mutations landed)
    assert eng.calls == []  # zero engine calls
    assert executors._long_admitted == 0  # rejected reserve counted nothing


def test_chat_nonstream_saturated_503_user_message_not_persisted(monkeypatch):
    """Round 11 finding 1 audit: the NON-STREAMING chat path persists the
    user message and only then submits the whole generation through
    run_long — saturation used to answer 503 with the message already in
    the DB (a retry duplicates it). The reservation now runs first."""
    stub_db = _EndpointStubDB()
    eng = _LiveEngineStub()
    client = _reservation_test_client(monkeypatch, stub_db, eng)

    with _saturated_long_pool():
        r = client.post(
            "/api/sessions/s1/chat", json={"content": "hi", "stream": False}
        )

    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_busy"
    assert stub_db.writes == []  # user message NOT persisted
    assert eng.calls == []
    assert executors._long_admitted == 0


def test_endpoints_consume_reservation_and_return_slots(monkeypatch):
    """Success path: every reserving endpoint completes and the admission
    accounting returns to baseline — the reservation was consumed exactly
    once (context-manager exit must not double-release)."""
    stub_db = _EndpointStubDB()
    eng = _LiveEngineStub()
    client = _reservation_test_client(monkeypatch, stub_db, eng)

    assert client.post("/api/sessions/s1/regenerate").status_code == 200
    assert client.post("/api/sessions/s1/delete-last").status_code == 200
    assert client.post(
        "/api/sessions/s1/branch", json={"turn": 2}
    ).status_code == 200
    assert client.post(
        "/api/sessions/s1/chat", json={"content": "hi", "stream": False}
    ).status_code == 200

    assert executors._long_admitted == 0
    for call in (
        "prepare_regenerate", "truncate_session", "branch_from_turn",
        "complete",
    ):
        assert call in eng.calls


def test_reservation_released_when_db_step_fails(monkeypatch):
    """Failure path between the reserve and the submit: the DB mutation
    throws, the endpoint 500s, and the un-consumed reservation is released
    — slot accounting returns to baseline, engine untouched."""
    stub_db = _EndpointStubDB(raise_on_delete=True)
    eng = _LiveEngineStub()
    client = _reservation_test_client(monkeypatch, stub_db, eng)

    r = client.post("/api/sessions/s1/regenerate")

    assert r.status_code == 500
    assert eng.calls == []
    assert executors._long_admitted == 0


class _SummaryEngineStub:
    """Compaction engine stub: streams a summary successfully; reaching
    compact_session while the pool is saturated means the reservation was
    missing."""

    model_id = "stub-model"
    model_family = "chatml"

    def __init__(self):
        self.cfg = SimpleNamespace(enable_thinking=False)
        self.compact_calls: list = []

    async def generate_stream_async(self, *args, **kwargs):
        yield SimpleNamespace(text="a fine summary")

    def compact_session(self, *a, **k):
        self.compact_calls.append(a)
        return {"cached_tokens": 5}


def test_compaction_saturated_error_frame_no_summary_insert(monkeypatch):
    """Round 11 finding 1 audit: the compaction finalizer inserts the
    durable summary message BEFORE its compact_session run_long. Saturation
    is now detected by the reservation BEFORE the insert and surfaces as an
    in-band SSE error frame — DB untouched, no engine rebuild (pre-fix the
    EngineBusyError escaped the generator mid-stream with the summary
    already persisted)."""
    from mlx_soloheaven.api import compaction as compaction_mod

    stub_db = _EndpointStubDB(
        messages=[
            {"role": "user", "content": f"m{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"m{i}"}
            for i in range(8)
        ]
    )
    eng = _SummaryEngineStub()
    monkeypatch.setattr(compaction_mod, "db", stub_db)
    monkeypatch.setattr(compaction_mod, "engine", eng)

    req = compaction_mod.CompactionRequest(keep_recent_turns=1)

    async def _collect():
        frames = []
        async for f in compaction_mod._stream_compact(
            "s1", {"system_prompt": "", "total_prompt_tokens": 100}, req
        ):
            frames.append(f)
        return frames

    with _saturated_long_pool():
        frames = asyncio.run(_collect())

    payloads = [
        json.loads(f.removeprefix("data: "))
        for f in frames
        if f.startswith("data: ")
    ]
    assert payloads[-1]["type"] == "error"
    assert "busy" in payloads[-1]["error"]
    assert stub_db.writes == []  # summary NOT inserted
    assert eng.compact_calls == []  # no engine rebuild
    assert executors._long_admitted == 0


def test_server_shutdown_hook_orders_quiesce_gate_flush_teardown():
    """Source-level wiring (same pattern as
    test_health_wired_to_reserved_read_executor): quiesce runs BEFORE the
    engine flush/close so a queued mutation can't land after it; the
    engine-side shutdown gate (begin_shutdown — round 3, finding 2) closes
    BETWEEN quiesce and the flush so a straggler blocked on the engine lock
    can never mutate post-flush; the bounded in-flight-mutation drain
    (wait_mutations_settled — round 5, finding 3a) runs BETWEEN the gate
    and the flush; the pool teardown backstop runs AFTER."""
    from mlx_soloheaven import server

    src = inspect.getsource(server.create_app)
    shutdown_src = src.split("async def shutdown", 1)[1].split(
        '@app.get("/health")', 1
    )[0]
    i_quiesce = shutdown_src.index("executors.quiesce")
    i_gate = shutdown_src.index("begin_shutdown()")
    i_drain = shutdown_src.index("engine.wait_mutations_settled")
    i_flush = shutdown_src.index("_flush_all_on_shutdown")
    i_close = shutdown_src.index("engine.close()")
    i_pools = shutdown_src.index("executors.shutdown_pools")
    assert i_quiesce < i_gate
    assert i_gate < i_drain
    assert i_drain < i_flush
    assert i_drain < i_close
    assert i_gate < i_flush
    assert i_gate < i_close
    assert i_flush < i_pools
    assert i_close < i_pools
