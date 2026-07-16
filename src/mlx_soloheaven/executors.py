"""Dedicated thread pools for engine-facing endpoint work (batch-3 F2).

``asyncio.to_thread`` runs everything on the event loop's DEFAULT executor —
one shared ``ThreadPoolExecutor`` for every ``to_thread`` user. Non-streaming
generations and mutating engine RPCs each occupy a worker for the FULL call
(a whole generation, or the process-mode RPC's 600s bounded wait). Enough
concurrent long calls exhaust the shared pool, and then BOUNDED reads
(/health, admin overviews, session lists) cannot even START: their 10s
engine-side bound never begins ticking and the busy degradation
(``EngineBusyError`` -> "busy" placeholder / 503) never runs — exactly the
endpoints meant to stay live under load hang in the executor queue.

Two dedicated pools fix the priority inversion:

- LONG pool (small, bounded): non-streaming completions + mutating engine
  RPCs (complete / update_session_messages / compact / branch / truncate /
  regenerate / delete / clear). The engine serializes them anyway (single
  engine lock / single child worker), so a shallow queue is the intended
  admission control — but the queue is BOUNDED (codex round 3, finding 5):
  streaming generation does NOT go through run_long (it streams via the
  engine's own async plumbing), so nothing legitimate ever needs a deep
  run_long backlog. A burst past ``LONG_ADMISSION_LIMIT`` is stale work by
  construction (long ops are minutes-scale; entry N of a deep queue would
  execute arbitrarily late, long after its client gave up) and fails fast
  with ``EngineBusyError`` (the app-level 503 backstop maps it). There is
  deliberately NO end-to-end deadline for run_long — long ops legitimately
  take minutes; only ADMISSION is bounded.
- READ pool (tiny, reserved): bounded reads only (session_stats /
  list_sessions / get_session / base_cache_stats / cache_overview / cache
  preflight). A read always gets a worker promptly, so its own bound + busy
  degradation actually run. Reads keep their existing 10s engine-side bound
  (``_read_locked`` in-process, ``RPC_READ_TIMEOUT_S`` in process mode), so
  two workers turn over quickly.
- CRITICAL lane (codex round 5, finding 1): the post-stream session commit
  (``update_session_messages``) is MANDATORY bookkeeping submitted BETWEEN
  the last streamed token and the terminal done/[DONE] event — after tokens
  went out no 503 is possible, so it must never be droppable by admission
  control nor queue behind minutes-scale long ops. ``run_critical`` runs it
  on its own reserved single worker with NO admission limit and NO
  stop-admission rejection (see its docstring for the shutdown contract).
  Shutdown rejects a still-queued commit by POISONING it, never by
  cancelling its future (codex round 11, finding 2 — see the
  poison-instead-of-cancel note at ``_CriticalPoisonCell``), so a
  CancelledError in run_critical is always task-originated and propagates.

ADMISSION RESERVATION (codex round 11, finding 1): endpoints whose
irreversible DB mutation precedes their run_long submission acquire the
admission slot FIRST via ``reserve_long_slot()`` and pass it to
``run_long(..., reservation=...)`` — saturation then answers 503 with the
DB untouched instead of stranding a half-mutated session (see the note at
``LongReservation``).

F2 residual (codex batch-3 review, round 2): a tiny reserved pool is not
enough on its own — two 10s-busy reads pin both workers, and further reads
used to queue UNBOUNDED with their own timeout starting only once a worker
freed up ("reads start promptly" was false under concurrent load). run_read
therefore adds:

- an ADMISSION bound (``READ_ADMISSION_LIMIT``): over-limit submissions fail
  fast with ``EngineBusyError`` (every caller already degrades — busy
  placeholders / 503);
- an END-TO-END deadline from SUBMISSION time (``READ_DEADLINE_S``): the
  awaiter's ``asyncio.wait_for`` covers queue wait + execution, and a read
  that expires while still QUEUED is cancelled so it never runs late.

ATOMIC ADMISSION (codex round 3, finding 1): the admission check, the pool
submission, and the pending-registry insertion are ONE critical section
under ``_executor_lock``, and ``quiesce()`` flips ``_admission_stopped`` and
snapshots the pending set under the SAME lock. The round-2 shape ran those
as three separate steps, so a request could pass the admission check, lose
the CPU, let quiesce set the flag and snapshot an EMPTY pending set, and
only then submit + track — the mutation executed AFTER the shutdown engine
flush while quiesce had reported still_running=0 (codex reproduced this
deterministically). With the shared lock, once quiesce holds it no new
future can be admitted and no admitted future can be untracked.

SHUTDOWN LIFECYCLE (codex batch-3 review, round 2): the pools used to live
forever — queued mutations could execute AFTER the server-shutdown engine
flush persisted state, and interpreter exit joined the worker threads. The
server shutdown hook now runs, in order: ``quiesce()`` (stop admission +
cancel queued futures + bounded wait for running calls) BEFORE the engine
flush, then ``shutdown_pools()`` as the backstop AFTER it. Both are
idempotent (the FastAPI hook and the atexit backstop may each run them);
``_reset_for_tests`` restores pristine module state for the test suite.
A straggler that survives the bounded wait while BLOCKED on the engine
lock is handled engine-side: the engine's shutdown gate
(``MLXEngine.begin_shutdown``) makes any post-flush mutating lock
acquisition abort without mutating (codex round 3, finding 2).

Executors are created lazily (no threads until first use) and shared
process-wide. This module imports NO mlx (``engine.types`` is explicitly
mlx-free), so the process-mode parent stays mlx-free.
"""

import asyncio
import atexit
import functools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

from mlx_soloheaven.engine.types import EngineBusyError

# Long operations: non-streaming generations + mutating engine RPCs. Small on
# purpose — the engine serializes generations anyway; extra submissions queue.
LONG_EXECUTOR_MAX_WORKERS = 4

# Bounded reads: /health + admin/stats/list/preflight. Reads are 10s-bounded
# engine-side, so a tiny pool turns over quickly — but it is RESERVED, so a
# read can never sit behind a generation for a worker.
READ_EXECUTOR_MAX_WORKERS = 2

# Critical post-stream commits (codex round 5, finding 1): engine-side the
# op is lock-free and microsecond-scale (touch + mark-dirty); in process
# mode it is one bounded RPC served by the child's serial loop right after
# the stream it follows. One reserved worker keeps it off both other pools
# so it can never queue behind a minutes-scale long op.
CRITICAL_EXECUTOR_MAX_WORKERS = 1

# Finding 5 (codex round 3): admission bound for the LONG pool — workers + a
# generous backlog. Larger than the read backlog on purpose: long calls are
# heterogeneous (a 2s non-streaming completion queues behind a 3-minute
# compaction), so a legitimate burst of a handful of concurrent completions
# plus a few admin mutations must still fit. Past this, queued entries are
# stale by construction — long ops are minutes-scale, so entry N of a deeper
# queue would execute tens of minutes late, against a client that timed out
# long before — and fail fast with EngineBusyError instead of queueing
# unboundedly. NO end-to-end deadline (unlike run_read): a long op that has
# STARTED may legitimately run for minutes; only admission is bounded.
LONG_ADMISSION_LIMIT = LONG_EXECUTOR_MAX_WORKERS + 16

# F2 residual: admission bound for the READ pool — workers + a small backlog.
# The backlog (4) absorbs a normal burst (health poll + a couple of admin
# overviews landing together) while anything past it fails fast with
# EngineBusyError instead of queueing unboundedly behind two 10s-busy reads.
READ_ADMISSION_LIMIT = READ_EXECUTOR_MAX_WORKERS + 4

# F2 residual: END-TO-END read deadline measured from SUBMISSION (queue wait
# + execution). Slightly above the 10s engine-side bound (_read_locked /
# RPC_READ_TIMEOUT_S) so an immediately-started read is still bounded by the
# engine, while a queued one answers "busy" within this window instead of
# "10s starting whenever a worker frees up".
READ_DEADLINE_S = 12.0

# Finding 1: REENTRANT module lock. It guards pool creation, the admission
# counters, the shutdown flag AND the pending-futures registry — and the
# atomic admission section calls the pool getters while holding it, so it
# must be reentrant. Done callbacks (slot release / untrack) take it from
# worker threads; every critical section is tiny (no blocking work inside).
_executor_lock = threading.RLock()
_long_executor: ThreadPoolExecutor | None = None
_read_executor: ThreadPoolExecutor | None = None
_critical_executor: ThreadPoolExecutor | None = None

# Shutdown gate: once stop_admission() flips this, every NEW run_long /
# run_read fails fast — a queued mutation must never land after the
# server-shutdown engine flush. Written AND read under _executor_lock in the
# admission/quiesce paths (finding 1: the flag flip must be atomic with the
# pending-set snapshot).
_admission_stopped = False

# Terminal gate for the CRITICAL lane (codex round 5, finding 1): critical
# submissions deliberately pass ``_admission_stopped`` (a stream that is
# just finishing during shutdown still needs its commit) but must NOT
# resurrect a pool after ``shutdown_pools()`` tore everything down — the
# final engine flush has already run by then, so a late commit's dirty-mark
# could never be persisted anyway. Written AND read under _executor_lock.
_pools_shut_down = False

# In-flight admissions per pool (submitted, not yet done). Guarded by
# _executor_lock; released by the futures' done callbacks (which fire for
# completed, failed AND cancelled futures alike).
_long_admitted = 0
_read_admitted = 0

# Every not-yet-done future from BOTH pools, so quiesce() can cancel queued
# work and bounded-wait the running remainder. Guarded by _executor_lock.
# Futures are inserted in the SAME critical section that admits them
# (finding 1), so quiesce's snapshot can never miss an admitted future.
_pending_futures: set[Future] = set()

# POISON-INSTEAD-OF-CANCEL for the CRITICAL lane (codex round 11, finding 2).
#
# History: round 7 tagged shutdown-cancelled futures so run_critical could
# tell a quiesce-cancel (degrade to EngineBusyError) from an awaiting-task
# cancellation (must propagate); round 10 added an ownership check
# (task.cancelling() > 0 wins over tags) because an awaiting-task cancel
# can race the tag+cancel gap. Round 11 (codex, reproduced): the ownership
# check is still blind to Task.cancel() followed by Task.uncancel() —
# cancelling() drops to 0 while the already-scheduled CancelledError is
# still delivered, so a task-originated cancellation once again converted
# to EngineBusyError. Cancellation provenance simply cannot be recovered
# from a cancelled Future plus task state.
#
# The fix removes the ambiguity AT THE SOURCE: shutdown paths never call
# Future.cancel() on a critical-lane future. Instead each critical
# submission carries a poison cell shared with the shutdown paths; quiesce
# and shutdown_pools POISON the cell (under _executor_lock) and the wrapper
# callable checks it FIRST thing when the pool worker reaches it — poisoned
# means it raises the internal ``_ShutdownRejected`` INSIDE the pool, which
# the awaiter receives as an ordinary future exception and converts to the
# EngineBusyError shape its callers already handle. A CancelledError in
# run_critical is therefore ALWAYS awaiter/task-originated and ALWAYS
# re-raised — no tags, no ownership heuristics. Cost: a poisoned queued
# commit "runs" only to raise immediately on the 1-worker pool (trivial).
_CRITICAL_CELL_ATTR = "_soloheaven_critical_poison_cell"


class _ShutdownRejected(Exception):
    """INTERNAL sentinel raised INSIDE the critical pool by a poisoned
    wrapper (see the poison-instead-of-cancel note above). Never crosses
    the API boundary: run_critical converts it to EngineBusyError."""


class _CriticalPoisonCell:
    """State shared between ONE critical wrapper and the shutdown paths.

    Transitions happen under ``_executor_lock``: the wrapper atomically
    checks ``poisoned`` and sets ``started`` when it begins executing;
    quiesce/shutdown_pools poison only cells that have NOT started (an
    already-running commit is left to the bounded wait, exactly like a
    running future whose cancel() used to fail). The lock makes the
    outcome deterministic: a poisoned cell always rejects, an unpoisoned
    started cell always runs."""

    __slots__ = ("poisoned", "started")

    def __init__(self):
        self.poisoned = False
        self.started = False


def get_long_executor() -> ThreadPoolExecutor:
    """The bounded pool for LONG blocking engine calls (lazily created)."""
    global _long_executor
    with _executor_lock:
        if _long_executor is None:
            _long_executor = ThreadPoolExecutor(
                max_workers=LONG_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="engine-long",
            )
        return _long_executor


def get_read_executor() -> ThreadPoolExecutor:
    """The reserved pool for BOUNDED engine reads (lazily created)."""
    global _read_executor
    with _executor_lock:
        if _read_executor is None:
            _read_executor = ThreadPoolExecutor(
                max_workers=READ_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="engine-read",
            )
        return _read_executor


def get_critical_executor() -> ThreadPoolExecutor:
    """The reserved worker for CRITICAL post-stream commits (lazily
    created). Kept separate from BOTH other pools so a mandatory commit can
    never queue behind a saturated long-ops backlog or a busy read."""
    global _critical_executor
    with _executor_lock:
        if _critical_executor is None:
            _critical_executor = ThreadPoolExecutor(
                max_workers=CRITICAL_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="engine-critical",
            )
        return _critical_executor


def _check_admission(kind: str):
    """Fail fast once shutdown stopped admission (callers degrade 503-style,
    exactly like every other EngineBusyError path). Called INSIDE the atomic
    admission section (finding 1), so it can never race quiesce's flag+
    snapshot."""
    if _admission_stopped:
        raise EngineBusyError(
            f"server shutting down — {kind} engine call rejected"
        )


def _untrack(fut: Future):
    with _executor_lock:
        _pending_futures.discard(fut)


def _release_long_slot(_fut: Future):
    global _long_admitted
    with _executor_lock:
        _long_admitted -= 1


def _release_read_slot(_fut: Future):
    global _read_admitted
    with _executor_lock:
        _read_admitted -= 1


# --- LONG-pool admission reservation (codex round 11, finding 1) -----------
#
# Several endpoints perform IRREVERSIBLE DB mutations BEFORE their run_long
# submission (regenerate deletes the assistant turn, delete-last removes the
# last pair, branch creates+populates a durable session, non-streaming chat
# persists the user message, compaction inserts the durable summary).
# run_long's fail-fast admission then rejected the op AFTER the mutation:
# a 503 with the DB already changed and ZERO engine calls (codex reproduced:
# regenerate left only the user message behind). ensure_available() (the
# batch-1 preflight) checks child liveness but does NOT reserve executor
# capacity, so it cannot close this. reserve_long_slot() performs the SAME
# admission check + counter increment run_long does, atomically under
# _executor_lock, BEFORE the endpoint touches the DB; run_long(...,
# reservation=...) then CONSUMES the reservation instead of re-admitting.


class LongReservation:
    """A pre-acquired LONG-pool admission slot (see ``reserve_long_slot``).

    Lifecycle — the slot is returned EXACTLY once, through one of:

    - CONSUMED: passed to ``run_long(..., reservation=...)`` and submitted.
      From then on the submitted future's done callback releases the slot
      (completion, error and cancellation alike — the same
      ``_release_long_slot`` path unreserved admissions use) and
      ``release()`` becomes a no-op.
    - RELEASED: ``release()`` (or context-manager exit) before consumption
      — the endpoint's DB step threw, the shutdown gate rejected the
      submit, or the op was abandoned. Idempotent: double release never
      double-decrements.
    """

    __slots__ = ("_state",)

    _HELD = "held"
    _CONSUMED = "consumed"
    _RELEASED = "released"

    def __init__(self):
        self._state = LongReservation._HELD

    def __enter__(self) -> "LongReservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self):
        """Return the admission slot unless run_long already consumed it
        (then the future's done callback owns the release). Idempotent."""
        global _long_admitted
        with _executor_lock:
            if self._state == LongReservation._HELD:
                self._state = LongReservation._RELEASED
                _long_admitted -= 1

    def _require_held(self):
        """Misuse guard, called under _executor_lock BEFORE the submit —
        a reservation can be consumed exactly once and never after its
        release."""
        if self._state != LongReservation._HELD:
            raise RuntimeError(
                f"long-pool reservation is already {self._state}; it can "
                f"be consumed exactly once, before any release"
            )

    def _consume(self):
        """Transfer slot ownership to a submitted future (called by
        run_long under _executor_lock, AFTER a successful pool submit —
        cannot fail after _require_held because every state transition
        holds the same lock)."""
        self._require_held()
        self._state = LongReservation._CONSUMED


def reserve_long_slot() -> LongReservation:
    """Atomically admit a FUTURE run_long submission (codex round 11,
    finding 1): the same shutdown-gate + backlog-bound checks as run_long,
    the same counter increment, under the same lock — but performed BEFORE
    the caller's irreversible DB mutations. Raises EngineBusyError when
    saturated or once shutdown stopped admission, so callers answer a clean
    503 with the DB untouched. The returned reservation is a context
    manager; exiting it releases the slot unless run_long consumed it."""
    global _long_admitted
    with _executor_lock:
        _check_admission("long")
        if _long_admitted >= LONG_ADMISSION_LIMIT:
            raise EngineBusyError(
                f"engine busy — long-ops pool saturated "
                f"({LONG_ADMISSION_LIMIT} calls in flight), retry shortly"
            )
        _long_admitted += 1
        return LongReservation()


async def run_long(fn, /, *args, reservation: LongReservation | None = None,
                   **kwargs):
    """Run a LONG blocking engine call (non-streaming completion / mutating
    RPC) on the bounded long-ops pool. Drop-in replacement for
    ``asyncio.to_thread`` at those call sites.

    Finding 5 (codex round 3): bounded ADMISSION (``LONG_ADMISSION_LIMIT``)
    — streaming generation never uses run_long, so a backlog past workers +
    16 is stale work by construction and fails fast with EngineBusyError
    (the server's 503 backstop maps it). No end-to-end deadline: a long op
    that started may legitimately run for minutes.

    Finding 1: the shutdown-admission check, the backlog bound, the pool
    submission and the pending-registry insertion are ONE critical section
    under the module lock — quiesce() takes the same lock, so an admitted
    future is always visible to its snapshot and a post-quiesce submission
    is always rejected.

    Codex round 11, finding 1: ``reservation`` — a slot pre-acquired via
    ``reserve_long_slot()`` BEFORE the caller's DB mutations. When passed,
    run_long does NOT re-admit (the slot is already counted — re-checking
    the backlog bound would reject the endpoint against its own
    reservation): it re-checks only the shutdown gate (a mutation must
    still never land behind the shutdown flush; on rejection the
    reservation stays HELD so the caller's context manager releases it),
    submits, and CONSUMES the reservation — from then on the future's done
    callback releases the slot exactly like an unreserved admission.
    ``reservation`` is consumed here and never forwarded to ``fn``."""
    global _long_admitted
    call = functools.partial(fn, *args, **kwargs)
    with _executor_lock:
        _check_admission("long")
        if reservation is None:
            if _long_admitted >= LONG_ADMISSION_LIMIT:
                raise EngineBusyError(
                    f"engine busy — long-ops pool saturated "
                    f"({LONG_ADMISSION_LIMIT} calls in flight), retry shortly"
                )
            fut = get_long_executor().submit(call)
            _long_admitted += 1
        else:
            # Misuse guard BEFORE the submit; consume AFTER it, so a submit
            # failure (torn-down pool) leaves the reservation HELD and the
            # caller's context manager releases the slot.
            reservation._require_held()
            fut = get_long_executor().submit(call)
            reservation._consume()
        _pending_futures.add(fut)
    fut.add_done_callback(_release_long_slot)
    fut.add_done_callback(_untrack)
    # Codex round 7, finding 3 audit: run_long deliberately has NO
    # cancelled()-means-quiesce conversion — BOTH a quiesce-cancel of a
    # queued future and an awaiting-task cancellation surface here as
    # CancelledError (task cancellation propagates; a quiesce-cancelled
    # mutation aborts its request task during shutdown, which callers treat
    # as a dropped request — the pinned contract of
    # test_quiesce_cancels_queued_work_before_flush). Only run_critical
    # converts, because its callers MUST emit a terminal event afterwards.
    return await asyncio.wrap_future(fut)


async def run_read(fn, /, *args, **kwargs):
    """Run a BOUNDED engine read on the reserved read pool so it always gets
    a worker even when every long-ops worker is occupied. Drop-in
    replacement for ``asyncio.to_thread`` at those call sites.

    F2 residual (codex round 2): bounded ADMISSION + an end-to-end deadline
    from submission. Over-limit submissions and deadline expiries both raise
    ``EngineBusyError`` (callers already degrade — busy placeholder / 503).
    A read whose deadline expires while still QUEUED is cancelled and never
    runs; one already executing keeps its worker + admission slot until it
    returns (threads cannot be interrupted), but its result is discarded.

    Finding 1 (codex round 3): admission check + submit + registry insert
    are atomic under the module lock (see run_long)."""
    global _read_admitted
    call = functools.partial(fn, *args, **kwargs)
    with _executor_lock:
        _check_admission("read")
        if _read_admitted >= READ_ADMISSION_LIMIT:
            raise EngineBusyError(
                f"engine busy — read pool saturated ({READ_ADMISSION_LIMIT} "
                f"reads in flight), retry shortly"
            )
        fut = get_read_executor().submit(call)
        _read_admitted += 1
        _pending_futures.add(fut)
    fut.add_done_callback(_release_read_slot)
    fut.add_done_callback(_untrack)
    # Codex round 7, finding 3 audit: like run_long, run_read has NO
    # cancelled()-means-quiesce conversion — cancellations propagate as
    # CancelledError (see run_long's note); only the DEADLINE path below
    # converts, and it cancels the future itself (never a task cancel).
    #
    # Codex round 11, finding 2 audit — does this converter share the
    # Task.uncancel() ambiguity? No. The busy-conversion is keyed on
    # TimeoutError from asyncio's timeout machinery, whose expiry decision
    # comes from its OWN state (the deadline callback fired), not from
    # inspecting a cancelled future or task.cancelling(): an external task
    # cancellation before expiry re-raises CancelledError untouched
    # (uncancel or not). The one residual — an external cancel+uncancel
    # racing an ACTUAL expiry can surface as TimeoutError — is acceptable
    # by construction: the deadline genuinely elapsed (busy is the correct
    # answer) and a task that uncancel()led has explicitly declared it is
    # continuing, so absorbing that cancel is its stated intent. No poison
    # needed here: OUR watchdog cancels only the POOL future, never the
    # awaiting task, so provenance stays unambiguous.
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(fut), timeout=READ_DEADLINE_S
        )
    except asyncio.TimeoutError:
        # Deadline measured from SUBMISSION expired (queue wait counts).
        # cancel() succeeds for a not-yet-started pool future — the done
        # callback then frees its admission slot and it NEVER runs late.
        fut.cancel()
        raise EngineBusyError(
            f"engine busy — read not served within {READ_DEADLINE_S:.0f}s "
            f"of submission (queue wait + execution), retry shortly"
        ) from None


async def run_critical(fn, /, *args, **kwargs):
    """Run a MANDATORY post-stream engine commit (``update_session_messages``
    after a completed stream) on the reserved critical worker.

    Codex round 5, finding 1: streaming generation bypasses run_long but
    HOLDS the engine lock, so 20 concurrent non-streaming/mutating calls can
    saturate the long pool while a stream runs. The stream's post-stream
    commit used to go through the saturated run_long: EngineBusyError
    escaped MID-STREAM (tokens already sent — no 503 possible), the
    terminal done/[DONE] event was never emitted, and the freshly installed
    session cache was never marked dirty. The commit is not droppable work:
    it is lock-free and microsecond-scale engine-side (touch + mark-dirty;
    in process mode one bounded RPC on the child's serial loop).

    Contract (deliberately different from run_long):

    - NO admission limit: the op is microseconds, so backlog depth is
      bounded by stream-finish frequency, not by op duration.
    - NOT rejected by ``stop_admission``: during shutdown quiesce a stream
      that is just finishing still needs its commit, and the final engine
      flush runs AFTER quiesce — a commit admitted in that window lands its
      dirty-mark in time to be flushed. Allowing critical submissions until
      the pools actually shut down is therefore the safer contract (the
      alternative silently drops the last turn of every stream that
      finishes during a graceful shutdown).
    - REFUSED once ``shutdown_pools()`` ran (``_pools_shut_down``): the
      final flush already happened, so a later commit could never be
      persisted — and lazily resurrecting a pool at teardown time would
      leak threads into interpreter exit.
    - TRACKED in the pending registry, so quiesce sees in-flight commits
      and bounded-waits them like everything else. quiesce MAY reject a
      still-queued critical commit (benign: the engine-side dirty-mark at
      session install — finding 1a — already made persistence independent
      of this call) — since round 11 by POISONING it, never by cancelling
      it (see the poison-instead-of-cancel note at _CriticalPoisonCell):
      the wrapper raises the internal sentinel inside the pool and the
      awaiter converts it to EngineBusyError, so callers keep a single,
      already-handled failure shape and the stream's terminal event stays
      unconditional. A poisoned QUEUED commit learns its rejection when
      the worker reaches it — normally immediately (the op ahead of it is
      microsecond-scale); behind a pathologically hung head-of-line commit
      the awaiter instead ends via its own task cancellation at
      server-loop teardown, which now propagates with correct provenance.
    - HEAD-OF-LINE tradeoff (codex round 7 NIT, accepted): the critical
      lane is ONE worker, so a commit that blocks (a process-mode RPC
      bounded-waits up to 600s against a hung child) delays every LATER
      stream's post-stream commit — and therefore its terminal done/[DONE]
      event — behind it. Accepted as availability degradation, not a
      correctness risk: the engine-side dirty-mark at session install
      (finding 1a) already guarantees persistence independent of this
      call, and a hung child worker trips the liveness watchdog anyway.
    """
    call = functools.partial(fn, *args, **kwargs)
    cell = _CriticalPoisonCell()

    def _critical_wrapper():
        # POISON check FIRST (codex round 11, finding 2) — atomically with
        # the started-mark, so a shutdown path can never poison a commit
        # that already began and never miscount one that has not.
        with _executor_lock:
            if cell.poisoned:
                raise _ShutdownRejected(
                    "critical engine commit rejected by shutdown quiesce/"
                    "pool teardown (poisoned while still queued)"
                )
            cell.started = True
        return call()

    with _executor_lock:
        if _pools_shut_down:
            raise EngineBusyError(
                "server shut down — critical engine commit rejected "
                "(pools already torn down; state already flushed)"
            )
        fut = get_critical_executor().submit(_critical_wrapper)
        setattr(fut, _CRITICAL_CELL_ATTR, cell)
        _pending_futures.add(fut)
    fut.add_done_callback(_untrack)
    # Codex round 11, finding 2 (POISON-instead-of-cancel): CancelledError
    # needs NO handler here anymore. Shutdown never communicates rejection
    # through Future.cancel() on this lane (quiesce/shutdown_pools poison
    # the cell instead, and the critical pool is torn down WITHOUT
    # cancel_futures), so a CancelledError reaching this awaiter is ALWAYS
    # awaiter/task-originated — client disconnect, request-task shutdown,
    # and Task.cancel()+Task.uncancel() patterns included — and simply
    # propagates. The round-7 tag and round-10 ownership heuristics are
    # gone with the ambiguity they tried to patch: task.cancelling() is
    # provably insufficient (uncancel() drops it to 0 while the scheduled
    # CancelledError still delivers — codex round 11 repro).
    try:
        return await asyncio.wrap_future(fut)
    except _ShutdownRejected as exc:
        # Shutdown rejected the queued commit BEFORE it started: degrade to
        # the EngineBusyError shape every caller already logs-and-continues
        # on, so the stream's terminal event still goes out.
        raise EngineBusyError(str(exc)) from None


# --- server shutdown lifecycle (codex batch-3 review, rounds 2+3) ----------


def stop_admission():
    """Step 1 of server shutdown: reject every NEW run_long / run_read
    submission fail-fast (EngineBusyError — callers already handle the
    503-style degrade) so no new engine work slips in behind the shutdown
    flush. Taken under the admission lock (finding 1) so the flip can never
    interleave with an in-flight admission section. Idempotent."""
    global _admission_stopped
    with _executor_lock:
        _admission_stopped = True


def quiesce(timeout_s: float = 5.0) -> dict:
    """Steps 1-3 of server shutdown — MUST run BEFORE the engine flush:

    1. stop admission (new submissions fail fast);
    2. cancel every QUEUED (not-yet-started) future — a queued mutation must
       never execute after the flush persisted engine state;
    3. bounded wait (``timeout_s``) for RUNNING calls to finish so the flush
       sees a quiet engine (a straggler blocked on the engine lock is left
       to the ENGINE-SIDE shutdown gate — begin_shutdown() makes its
       eventual lock acquisition a mutation-free no-op — and pool teardown
       to the shutdown_pools() backstop).

    Finding 1: the admission-stop flip and the pending snapshot happen under
    the SAME lock the admission sections hold — after this lock is taken,
    no new future can be admitted and none can be missing from the snapshot.

    Codex round 11, finding 2 (POISON-instead-of-cancel): CRITICAL-lane
    futures are never cancelled — a cancelled Future is indistinguishable
    from an awaiter-driven task cancellation (the round-7 tags and round-10
    ownership checks both failed to fully recover provenance; Task.uncancel
    defeats task.cancelling()). Their poison cells are flipped instead
    (under this same lock, atomically against the wrapper's started-mark):
    the wrapper raises the internal sentinel INSIDE the pool and the
    awaiter degrades it to EngineBusyError. LONG/READ futures keep the
    plain .cancel(): their awaiters deliberately propagate CancelledError
    (a quiesce-cancelled mutation aborts its request task during shutdown —
    the pinned contract of test_quiesce_cancels_queued_work_before_flush),
    so no provenance question arises there.

    Idempotent (safe from both the FastAPI shutdown hook and atexit).
    Returns counts for logging/tests:
    {"cancelled": n, "poisoned_critical": p, "still_running": m} —
    "cancelled" counts cancelled long/read futures; "poisoned_critical"
    counts queued critical commits rejected via poison (they complete —
    by raising — once the worker reaches them, so the bounded wait below
    covers them too).
    """
    global _admission_stopped
    with _executor_lock:
        _admission_stopped = True
        pending = list(_pending_futures)
        # Deadlock safety of cancelling under the RLock (verified against
        # CPython 3.12): Future.cancel() on a queued future flips state
        # under the FUTURE's own condition, then runs done callbacks INLINE
        # on this thread. Our callbacks (_untrack, _release_*_slot) take
        # _executor_lock — an RLock this thread already holds → reentrant,
        # tiny, non-blocking; wrap_future's chained _call_set_state only
        # does loop.call_soon_threadsafe and never touches _executor_lock.
        # Cancelling an already-cancelled future returns True without
        # re-running callbacks. (_untrack may mutate _pending_futures
        # during the loop — `pending` is a snapshot list, so that is safe.)
        cancelled = 0
        poisoned = 0
        for fut in pending:
            cell = getattr(fut, _CRITICAL_CELL_ATTR, None)
            if cell is not None:
                # Critical lane: poison, never cancel. Only a commit that
                # has NOT started (and is not already done — e.g. cancelled
                # by its own awaiting task) is rejected; a running one is
                # left to the bounded wait. `not cell.poisoned` keeps a
                # second quiesce (idempotency) from double-counting.
                if not cell.started and not cell.poisoned and not fut.done():
                    cell.poisoned = True
                    poisoned += 1
                continue
            if fut.cancel():
                cancelled += 1
    not_done = [fut for fut in pending if not fut.done()]
    if not_done:
        futures_wait(not_done, timeout=timeout_s)
    still_running = sum(1 for fut in pending if not fut.done())
    return {
        "cancelled": cancelled,
        "poisoned_critical": poisoned,
        "still_running": still_running,
    }


def shutdown_pools():
    """Final backstop (step 5): tear all pools down without blocking —
    ``cancel_futures=True`` clears anything still queued. Runs AFTER the
    engine flush so pool teardown can never race it. Also closes the
    CRITICAL lane (``_pools_shut_down``) — critical commits are allowed
    through stop_admission, but past this point the final flush has run and
    nothing may resurrect a pool. Idempotent."""
    global _long_executor, _read_executor, _critical_executor, _pools_shut_down
    with _executor_lock:
        _pools_shut_down = True
        long_pool, _long_executor = _long_executor, None
        read_pool, _read_executor = _read_executor, None
        critical_pool, _critical_executor = _critical_executor, None
        # Codex round 11, finding 2: same poison-vs-cancel split as quiesce.
        # Critical-lane futures are POISONED (their awaiters must receive
        # an ordinary exception, never an ambiguous cancel); long/read
        # futures are cancelled outright inside this lock section (see
        # quiesce for the inline-done-callback / RLock-reentrancy safety
        # argument) — their pool.shutdown(cancel_futures=True) below stays
        # the queue-drain backstop.
        for fut in list(_pending_futures):
            cell = getattr(fut, _CRITICAL_CELL_ATTR, None)
            if cell is not None:
                if not cell.started and not fut.done():
                    cell.poisoned = True
                continue
            fut.cancel()
    for pool in (long_pool, read_pool):
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    if critical_pool is not None:
        # The CRITICAL pool is drained WITHOUT cancel_futures: a queued
        # poisoned wrapper must still RUN (only to raise the sentinel
        # immediately on the 1-worker lane — trivial) because that raise is
        # exactly how its awaiter learns the rejection; cancel_futures=True
        # would resurrect the cancelled-future ambiguity this design
        # removes. If the worker is wedged on a hung head-of-line commit,
        # a queued poisoned awaiter instead ends via its own request-task
        # cancellation at server-loop teardown — correct provenance.
        critical_pool.shutdown(wait=False, cancel_futures=False)


def _reset_for_tests():
    """TEST-ONLY: restore pristine module state (pools torn down, admission
    re-opened, registries cleared) so shutdown-path tests cannot leak their
    global state into the rest of the suite."""
    global _admission_stopped, _long_admitted, _read_admitted
    global _pools_shut_down
    shutdown_pools()
    with _executor_lock:
        _admission_stopped = False
        _pools_shut_down = False
        _long_admitted = 0
        _read_admitted = 0
        _pending_futures.clear()


def _atexit_backstop():  # pragma: no cover — interpreter-exit path
    """Idempotent re-run of the shutdown ordering for non-server embedders
    (the FastAPI hook already ran it on a normal server stop)."""
    try:
        quiesce()
        shutdown_pools()
    except Exception:  # noqa: BLE001 — never block interpreter exit
        pass


atexit.register(_atexit_backstop)
