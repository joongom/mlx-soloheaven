"""Parent-side async FIFO admission gate for GPU generation (bultagi Batch C).

Actual model inference is concurrency-1 already: the in-process engine holds
``MLXEngine._global_gpu_lock`` for a whole generation, and process mode
serializes in the child's single main loop. What was MISSING was an ordered,
BOUNDED front door: the streaming API path did NOT enter executors admission —
it blocked on the engine lock through a 0.1s polling loop (``threading.Lock``
gives NO FIFO guarantee, and there was no ceiling on the number of waiters).

This module is that front door. It lives entirely in the FastAPI PARENT
(async, single event-loop thread), so it fronts in-process AND process-mode
generation identically (the fronting is at the API layer, above the engine
abstraction — ``EngineProcessProxy.generate_stream_batches_async`` /
``.complete`` are gated too). It imports NO mlx (only ``asyncio`` +
``EngineBusyError`` from the mlx-free ``engine.types``), so the process-mode
parent stays mlx-free.

CONTRACT (bultagi, confirmed):

- inference concurrency EXACTLY 1 — one RUNNING slot, matching the engine;
- multiple requests are ordered FIFO;
- a configurable MAX QUEUE LENGTH bounds ``waiting + running`` — over it, a
  new acquire raises ``EngineBusyError(REASON_QUEUE_FULL)`` IMMEDIATELY (no
  await), which the app maps to 429 queue_full + Retry-After;
- while the server is shutting down, a new acquire raises
  ``EngineBusyError(REASON_ENGINE_NOT_READY)`` -> 503, and every QUEUED waiter
  is failed the same way (never left hung); the RUNNING one is left to finish;
- a client that disconnects while QUEUED is cancelled: its waiter is removed
  from the FIFO deque, the counter decremented, and it NEVER starts generating
  (the running-task cancel policy is the existing cooperative U13, unchanged).

DESIGN — an EXPLICIT ordered structure (a deque of ``asyncio.Future`` tickets),
NOT ``asyncio.Semaphore`` internals, so FIFO + cancellation stay deterministic
and unit-testable without MLX. The wake / hand-off / cancel algorithm mirrors
CPython's battle-tested ``asyncio.Lock`` (a single logical slot handed to the
FIFO head, with the same cancelled-while-handed-off re-wake), extended with:
  1. a bounded ``running + waiting`` admission check (429 before enqueue),
  2. a shutdown flag that rejects new acquires and fails all queued waiters
     (503), and
  3. a live ``queue_length`` (``running + waiting``) that /ready and (Batch D)
     /metrics read.

ATOMICITY: every method runs on the single event-loop thread; the bounded
admission check and the enqueue happen with NO ``await`` between them, so they
are atomic. The waiting counter is incremented exactly once on enqueue and
decremented exactly once in a ``finally`` on ANY exit (woken, cancelled,
errored, shut-down) — no leak.

RELATIONSHIP TO THE ENGINE LOCK: the gate does NOT replace the engine's GPU
lock. The lock also guards internal / non-API ops (compaction rebuild, branch,
truncate) and is the true concurrency-1 safety; with the gate admitting one
generation at a time it is simply uncontended on the API generation path now.
The gate is the FIFO + bounding layer on top.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque

from mlx_soloheaven.engine.types import EngineBusyError

logger = logging.getLogger(__name__)

# Default MAX QUEUE LENGTH (waiting + running). 32 is a sensible single-user
# ceiling: with concurrency-1 generation, a backlog of ~31 waiters already
# means the newest request would wait behind tens of full generations (each
# seconds-to-minutes) — long past any client's patience — so rejecting past
# it with a fast 429 is strictly better than an unbounded queue that would
# accumulate stale, already-abandoned work. Overridable via
# ``--max-queue-length`` / ``SOLOHEAVEN_MAX_QUEUE_LENGTH``.
DEFAULT_MAX_QUEUE_LENGTH = 32


class Lease:
    """Opaque ownership token for one admission (codex finding 6).

    ``acquire()`` mints a fresh Lease and, when this caller actually takes the
    single RUNNING slot, records it as the gate's ``_owner``. ``release(lease)``
    is a NO-OP unless ``lease`` is the current owner — so a double release, a
    stale release (the slot has since been handed to a successor), or a foreign
    release (a request that never held the slot) cannot clear a live owner's
    slot or wake the wrong waiter. Identity-compared; carries no state, so it is
    cheap and cannot be forged into equality with another lease."""

    __slots__ = ("_seq",)

    _counter = 0

    def __init__(self) -> None:
        # A monotonically increasing id purely for debug logging — ownership is
        # decided by object identity (``is``), never by this value.
        Lease._counter += 1
        self._seq = Lease._counter

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"Lease#{self._seq}"


class ClientDisconnected(Exception):
    """Raised by ``acquire_or_disconnect`` when the client's ASGI connection
    reports ``http.disconnect`` while the request was still QUEUED on the gate
    (codex finding 2). The handler catches it, does NOT proceed to any DB
    mutation or engine entry, and returns a closed/empty response — the client
    is already gone. The gate ticket is unlinked before this is raised, so the
    queue continues correctly for everyone else."""


class InferenceGate:
    """A single-slot FIFO admission gate with a bounded waiter queue.

    All state is touched only from the event-loop thread (async endpoints,
    the streaming generators' finally, the /ready probe, the server shutdown
    hook) — asyncio's single-threaded execution is the mutual exclusion, so
    no lock is needed and the check-then-enqueue stays atomic as long as no
    ``await`` sits between them (it does not).
    """

    __slots__ = ("_max", "_running", "_waiting", "_waiters", "_shutdown", "_owner")

    def __init__(self, max_queue_length: int = DEFAULT_MAX_QUEUE_LENGTH):
        self._max = max(1, int(max_queue_length))
        # The single RUNNING slot (0 or 1) — mirrors the engine's concurrency-1.
        self._running = 0
        # Live waiter count == number of tickets currently in ``_waiters``.
        # Incremented on enqueue, decremented once in the acquire finally.
        self._waiting = 0
        self._waiters: deque[asyncio.Future] = deque()
        self._shutdown = False
        # The Lease currently holding the RUNNING slot (finding 6). None when
        # the slot is free. release(lease) only clears this when lease IS it.
        self._owner: Lease | None = None

    # --- configuration / introspection -----------------------------------

    def set_max_queue_length(self, max_queue_length: int) -> None:
        """Update the bound (server startup wires the config value here)."""
        self._max = max(1, int(max_queue_length))

    @property
    def max_queue_length(self) -> int:
        return self._max

    def queue_length(self) -> int:
        """Live ``running + waiting`` — the source /ready reports and Batch D
        /metrics will read. Never blocks; a pure read of two ints."""
        return self._running + self._waiting

    def running(self) -> int:
        return self._running

    def stats(self) -> dict:
        return {
            "running": self._running,
            "waiting": self._waiting,
            "queue_length": self._running + self._waiting,
            "max_queue_length": self._max,
            "shutdown": self._shutdown,
        }

    # --- admission --------------------------------------------------------

    async def acquire(self) -> Lease:
        """Acquire the single RUNNING slot, FIFO, and return an ownership Lease.

        Returns once this caller HOLDS the slot — the caller MUST pair it with
        exactly one ``release(lease)`` on the RETURNED lease (use
        ``inference_slot`` / the response-lifecycle wrapper so the release is
        unconditional). The lease makes release ownership-safe (finding 6).

        Raises IMMEDIATELY (no await) with:
          - ``EngineBusyError(REASON_ENGINE_NOT_READY)`` if the gate is shut
            down (-> 503), or
          - ``EngineBusyError(REASON_QUEUE_FULL)`` if ``running + waiting`` is
            already at the bound (-> 429).
        Otherwise, if the slot is busy, enqueues a FIFO ticket and awaits it.
        A cancellation while queued (client disconnect) removes the ticket,
        decrements the counter, and re-wakes the next FIFO HEAD if the slot is
        free — this caller NEVER proceeds to generation.
        """
        if self._shutdown:
            raise EngineBusyError(
                "server shutting down — inference queue closed",
                reason=EngineBusyError.REASON_ENGINE_NOT_READY,
            )

        # The ownership token for THIS admission. Minted up front so the woken
        # path can record it as the owner atomically when it takes the slot.
        lease = Lease()

        # Fast path: slot free AND no one ahead -> take it with no suspension.
        if self._running == 0 and self._waiting == 0:
            self._running = 1
            self._owner = lease
            return lease

        # Bounded admission (running + waiting). Synchronous — the check and
        # the enqueue below have NO await between them, so they are atomic.
        if self._running + self._waiting >= self._max:
            raise EngineBusyError(
                f"inference queue full ({self._running + self._waiting} of "
                f"{self._max} in flight), retry shortly",
                reason=EngineBusyError.REASON_QUEUE_FULL,
            )

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        self._waiting += 1
        try:
            try:
                await fut
            finally:
                # Unlink exactly once, whatever the exit (woken / cancelled /
                # shutdown / error). Decrements the counter iff still present.
                self._unlink(fut)
        except asyncio.CancelledError:
            # Cancelled/disconnected while queued OR just after being handed the
            # slot. Finding 1: this coroutine is responsible for unlinking
            # itself (done above) AND, if it is now at/ahead of a FREE slot,
            # waking the next FIFO HEAD — but ONLY the head, and only if that
            # head is not already a live acquirer being handed the slot.
            # ``_wake_up_first`` inspects the head alone (mirrors CPython
            # asyncio.Lock): if the new head future is already resolved (a live
            # acquirer was handed the slot in the same turn) it does nothing, so
            # the slot is never handed to two successors.
            if self._running == 0:
                self._wake_up_first()
            raise

        # Woken with a result. Shutdown wakes queued waiters to REJECT them
        # (fail-all on shutdown) — distinguish that from a real hand-off.
        if self._shutdown:
            if self._running == 0:
                self._wake_up_first()
            raise EngineBusyError(
                "server shutting down — inference queue closed",
                reason=EngineBusyError.REASON_ENGINE_NOT_READY,
            )
        # A waiter is woken (via _wake_up_first) ONLY when _running == 0 and it
        # is the FIFO HEAD, so taking the slot here can never overlap another
        # holder. Record ownership atomically (no await before the return).
        self._running = 1
        self._owner = lease
        return lease

    def release(self, lease: Lease | None = None) -> None:
        """Release the RUNNING slot IFF ``lease`` is the current owner, then
        hand the slot to the FIFO head.

        Ownership-safe (finding 6): a double release, a stale release (the slot
        has already been handed to a successor who now owns a DIFFERENT lease),
        or a foreign/no-lease release is a NO-OP — it never clears a live
        owner's slot nor wakes a waiter behind the owner's back. Always safe to
        call from a finally / a response lifecycle.
        """
        if lease is None or lease is not self._owner:
            # Stale / double / foreign / no-lease release — reject silently.
            logger.debug(
                "inference gate: ignoring release(%r); current owner=%r",
                lease, self._owner,
            )
            return
        self._owner = None
        self._running = 0
        # Hand the freed slot to the head of the FIFO. The woken waiter marks
        # _running = 1 + records its own lease when it resumes, so the short
        # _running == 0 window cannot be stolen by a fast-path acquire: a live
        # waiter makes _waiting > 0, which fails the fast-path guard.
        self._wake_up_first()

    # --- shutdown ---------------------------------------------------------

    def shutdown(self) -> None:
        """Close the gate: reject every NEW acquire (503) and FAIL every
        currently QUEUED waiter (503) so none hangs forever. The RUNNING
        generation is left to finish (its release then wakes nothing new —
        every waiter is already resolved). Idempotent."""
        self._shutdown = True
        for fut in list(self._waiters):
            if not fut.done():
                # Wake it; on resume it sees _shutdown and raises 503.
                fut.set_result(True)

    # --- internals --------------------------------------------------------

    def _wake_up_first(self) -> None:
        """Wake the FIFO HEAD ticket only — mirrors CPython
        ``asyncio.Lock._wake_up_first`` (finding 1).

        CRITICAL: inspect ONLY the head. If the head future is already done
        (its coroutine was cancelled but has not yet resumed to unlink itself,
        OR it was already handed the slot by a prior wake), do NOTHING. The
        old implementation scanned PAST done tickets to the first live one,
        which — combined with a cancelled head's own re-wake on resume — could
        hand the single slot to TWO successors at once (double admission). By
        touching only the head, a cancelled head is left to unlink itself and
        wake the NEXT head when it resumes, and a head already resolved by a
        live acquirer is never woken a second time."""
        if not self._waiters:
            return
        fut = self._waiters[0]
        if not fut.done():
            fut.set_result(True)

    def _unlink(self, fut: asyncio.Future) -> None:
        """Remove a ticket from the deque + decrement the counter, exactly
        once. Defensive against a double unlink (never double-decrements)."""
        try:
            self._waiters.remove(fut)
        except ValueError:
            return
        self._waiting -= 1


# --- module-global singleton + accessors -----------------------------------
#
# One gate for the whole server: generation concurrency is a single global
# resource (the one GPU), so a single global FIFO/bound is correct even with
# multiple model engines registered.

_GATE = InferenceGate(DEFAULT_MAX_QUEUE_LENGTH)


def get_inference_gate() -> InferenceGate:
    """The process-wide inference gate."""
    return _GATE


def configure(max_queue_length: int) -> None:
    """Wire the configured bound into the live gate (server startup)."""
    _GATE.set_max_queue_length(max_queue_length)


def current_queue_length() -> int:
    """Live ``running + waiting`` for /ready (and Batch D /metrics)."""
    return _GATE.queue_length()


def shutdown() -> None:
    """Close the gate during server shutdown (reject new + fail queued)."""
    _GATE.shutdown()


def _reset_for_tests() -> None:
    """TEST-ONLY: replace the singleton with a pristine gate so a test that
    manipulates queue state (or a streaming handler test that returns an
    unconsumed StreamingResponse, leaving the fast-path slot held) cannot leak
    into the rest of the suite."""
    global _GATE
    _GATE = InferenceGate(DEFAULT_MAX_QUEUE_LENGTH)


@contextlib.asynccontextmanager
async def inference_slot(gate: InferenceGate | None = None):
    """Acquire the gate for the scope of an ``async with`` and release on exit.

    Used by the NON-STREAMING generation paths (openai_compat ``_sync_completion``
    submission, chat ``_sync_chat`` submission) where acquire, run and release
    all live in the one request coroutine. The 429/503 from ``acquire`` propagate
    as real HTTP statuses (the app-level EngineBusyError handler maps them).

    The lease is threaded through so the release is ownership-safe (finding 6):
    a stray double release from anywhere else cannot clear THIS scope's slot."""
    g = gate if gate is not None else _GATE
    lease = await g.acquire()
    try:
        yield lease
    finally:
        g.release(lease)


async def _watch_disconnect(receive) -> None:
    """Resolve when the ASGI connection reports ``http.disconnect``.

    ``receive`` is the raw ASGI receive callable (Starlette ``Request.receive``).
    By the time a POST handler runs, FastAPI has already consumed the request
    body to build the pydantic model, so the only remaining event on the
    channel is the disconnect — awaiting it is a clean, deterministic watcher
    (no polling). Touches ONLY asyncio, keeping this module starlette/mlx-free."""
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return


async def acquire_or_disconnect(gate: InferenceGate, receive) -> Lease:
    """FIFO-acquire the gate slot, but ABORT if the client disconnects while
    QUEUED (codex finding 2).

    A real queued HTTP disconnect does NOT cancel the handler task (uvicorn
    only wakes ``receive`` with ``http.disconnect``; Starlette starts listening
    for it only AFTER the response is returned), so a bare ``await
    gate.acquire()`` would promote a dead client and let it proceed to DB
    mutation / engine entry. This races the acquire against a disconnect watcher
    on the ASGI receive channel.

    Returns the ``Lease`` on success. Raises ``ClientDisconnected`` if the
    client is gone (the gate ticket is unlinked first, so the queue continues
    for everyone else) — the caller must return WITHOUT entering generation.
    429/503 from ``acquire`` propagate unchanged. ``receive is None`` (direct,
    non-ASGI callers such as unit tests that hand the handler a pydantic model
    only) falls back to a plain acquire."""
    if receive is None:
        return await gate.acquire()

    acquire_task = asyncio.ensure_future(gate.acquire())
    disconnect_task = asyncio.ensure_future(_watch_disconnect(receive))
    try:
        await asyncio.wait(
            {acquire_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        # The handler task itself was cancelled (e.g. a task-cancel disconnect
        # or server teardown). Tear both children down; if the slot was already
        # acquired, hand it back so it is never stranded.
        acquire_task.cancel()
        disconnect_task.cancel()
        await asyncio.gather(
            acquire_task, disconnect_task, return_exceptions=True
        )
        if not acquire_task.cancelled() and acquire_task.exception() is None:
            gate.release(acquire_task.result())
        raise

    disconnect_hit = (
        disconnect_task.done()
        and not disconnect_task.cancelled()
        and disconnect_task.exception() is None
    )

    if acquire_task.done() and not acquire_task.cancelled():
        exc = acquire_task.exception()
        if exc is not None:
            # 429 / 503 raised synchronously by acquire — propagate it.
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            raise exc
        # We hold a real lease.
        lease = acquire_task.result()
        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)
        if disconnect_hit:
            # Won the slot but the client already left in the same turn — return
            # the slot to the queue and abort.
            gate.release(lease)
            raise ClientDisconnected()
        return lease

    # Disconnect won while the acquire was still queued: cancel the ticket
    # (its CancelledError path unlinks it and wakes the next head), then abort.
    acquire_task.cancel()
    await asyncio.gather(acquire_task, return_exceptions=True)
    if not acquire_task.cancelled() and acquire_task.exception() is None:
        # Extremely tight race: the ticket was promoted in the same turn the
        # cancel landed, so we actually own the slot — give it back.
        gate.release(acquire_task.result())
    raise ClientDisconnected()
