"""Response-lifecycle ownership of the inference-gate slot for streaming
generation (codex findings 3 + 4).

The streaming handlers acquire the single generation slot BEFORE returning a
``StreamingResponse`` (so queue-full is a real 429, not an un-sendable
mid-stream error). The slot must then be released EXACTLY once when the stream
truly ends — and the release must be robust against the two disconnect races
Starlette (ASGI spec_version 2.3, task-group body streaming) exposes:

  - FINDING 3 (pre-body disconnect): if the connection is already gone when the
    handler returns, Starlette can cancel body streaming BEFORE the body
    generator's first ``__anext__``. An unstarted async generator never enters
    its ``try/finally``, so a release that lived only in the body generator's
    finally would NEVER run — the slot leaks and the server wedges at
    concurrency-1.
  - FINDING 4 (disconnect during send): when the outer stream is cancelled at a
    chunk ``send``, the body generator is closed. Releasing the slot in that
    close WITHOUT first closing the INNER generation stream admits the next
    request before the prior turn's U13 cancel / C1 commit-or-invalidate
    teardown has run.

The fix is a ``StreamingResponse`` subclass that owns the lease at the RESPONSE
level: its ``__call__`` wraps ``super().__call__`` in a try/finally that (1)
closes the inner generation stream FIRST — driving its full U13/C1 teardown —
and only THEN (2) releases the slot. The body iterator ALSO releases on normal
completion (so callers that consume ``response.body_iterator`` directly, e.g.
hermetic tests, still release); every release is ownership-safe via the lease
(``InferenceGate.release`` is a no-op for a stale/double/foreign lease), so
doing it in both places is safe.

This module lives in the API layer, NOT in ``inference_queue`` — the gate stays
mlx-free / starlette-free (asyncio + EngineBusyError only)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncIterator

from starlette.responses import Response, StreamingResponse

from mlx_soloheaven.inference_queue import InferenceGate, Lease

logger = logging.getLogger(__name__)


class SlotStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` that owns an inference-gate ``Lease`` and closes
    the inner generation stream before releasing it, on EVERY exit path.

    Order on any teardown (normal end / pre-body cancel / mid-send cancel):
      1. close the inner generation generator (``inner.aclose()``) so its
         cancel_event fires and the C1 commit-or-invalidate + U13 teardown run
         to completion (finding 4);
      2. THEN release the slot (finding 3 — guaranteed even if the body
         generator never started).
    """

    def __init__(
        self,
        gate: InferenceGate,
        lease: Lease,
        inner: AsyncIterator[str],
        **kwargs,
    ) -> None:
        self._gate = gate
        self._lease = lease
        self._inner = inner
        self._released = False
        # The body iterator wraps ``inner`` so a NORMAL end (or a direct
        # ``body_iterator`` consumer) also runs the close-then-release path.
        super().__init__(self._slot_body(inner), **kwargs)

    # Ceiling (seconds) on how long teardown waits for the engine C1 epilogue
    # while the outer task is being cancelled. C1 is a fast in-memory reconcile
    # (in-proc) or a liveness-bounded child done-ack (process mode); this only
    # caps a wedged engine so a cancellation cannot hang teardown forever. On
    # expiry the slot is released fail-closed (C1 may be incomplete) and logged.
    _TEARDOWN_C1_TIMEOUT_S = 30.0

    async def _teardown(self) -> None:
        """Close the inner streaming chain (cascading the engine's C1
        commit-or-invalidate teardown to completion) THEN release the slot —
        cancellation-safe and EXACTLY once.

        Finding 1: ``self._inner.aclose()`` cascades GeneratorExit down every
        wrapper layer (each closes its OWN inner in a finally) to the engine
        generator, whose GeneratorExit rescue runs C1. In BOTH engine modes the
        engine generator's aclose does not return until C1 has genuinely
        completed (in-proc awaits the worker's done sentinel; process mode awaits
        the child's done ack). So the release below happens strictly AFTER the
        previous turn's engine teardown truly finished.

        Finding 2: the close is SHIELDED (run as its own task so an outer-task
        cancellation of our await cannot abort the close itself). If the outer
        task IS cancelled, the shielded close keeps running and is re-awaited
        (bounded) so we never release mid-C1; the release lives in an OUTERMOST
        finally so it runs exactly once regardless of how the close exits
        (normal / cancelled / errored), and a CancelledError is re-raised AFTER
        the release so the cancellation is not swallowed. On the rare unavoidable
        abort (loop tearing the shield down, or the bound elapsed) the slot is
        still released fail-closed and it is logged that C1 may be incomplete.

        Idempotent: safe to call from both the body finally and __call__."""
        if self._released:
            return
        cancelled_exc: asyncio.CancelledError | None = None
        # Run the (cascading) close as its OWN task so a cancellation of the
        # task awaiting us does not abort C1 — the task keeps running and we
        # re-await it below. A never-started / already-closed generator makes
        # the aclose a harmless immediate no-op.
        close_task = asyncio.ensure_future(self._inner.aclose())
        try:
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                # The outer task was cancelled while the shielded close ran; the
                # engine C1 is still going. Wait (bounded) for it to finish so we
                # do not release mid-commit, then re-raise after the release.
                cancelled_exc = exc
                if not close_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(close_task),
                            timeout=self._TEARDOWN_C1_TIMEOUT_S,
                        )
                    except asyncio.CancelledError:
                        logger.warning(
                            "inference gate: teardown re-cancelled while awaiting "
                            "engine C1 — releasing fail-closed (C1 may be "
                            "incomplete)"
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "inference gate: engine C1 teardown exceeded %.0fs — "
                            "releasing fail-closed (C1 may be incomplete)",
                            self._TEARDOWN_C1_TIMEOUT_S,
                        )
            except Exception:  # noqa: BLE001 — a close failure must not skip release
                logger.exception(
                    "inference gate: inner stream close raised during teardown; "
                    "releasing the slot (lease-guarded, so this stays safe)"
                )
        finally:
            # Release LAST, after the engine teardown, so the next request is
            # admitted only once this turn is fully wound down. Lease-guarded ==
            # idempotent, and this finally guarantees it runs exactly once.
            self._released = True
            self._gate.release(self._lease)
        if cancelled_exc is not None:
            raise cancelled_exc

    async def _slot_body(self, inner: AsyncIterator[str]) -> AsyncIterator[str]:
        """Body iterator: stream ``inner``; on ANY exit close inner then release
        (finding 4 ordering). Covers normal completion and direct
        ``body_iterator`` consumption; the never-started case is covered by
        ``__call__`` below (this generator's finally does not run if it never
        starts)."""
        try:
            async for chunk in inner:
                yield chunk
        finally:
            await self._teardown()

    async def __call__(self, scope, receive, send) -> None:
        """Guarantee release even when Starlette cancels body streaming BEFORE
        the body generator starts (finding 3) or mid-send (finding 4).

        Under uvicorn's ASGI spec_version 2.3, ``super().__call__`` runs body
        streaming + disconnect listening as competing tasks inside an anyio
        cancel scope; a disconnect cancels streaming WITHIN that scope and
        ``super().__call__`` returns NORMALLY (the CancelledError does not
        escape to here), so this finally runs cleanly. If the whole app task is
        cancelled instead, the release (sync, in the innermost finally) still
        runs regardless."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._released:
                # Body generator never started (finding 3) or was cancelled
                # mid-send without running its own finally (finding 4): close
                # both layers (order preserved by _teardown), then release.
                try:
                    with contextlib.suppress(Exception):
                        await self.body_iterator.aclose()
                finally:
                    if not self._released:
                        await self._teardown()


def closed_stream_response(media_type: str = "text/event-stream") -> StreamingResponse:
    """A trivially-closed streaming response for a client that already
    disconnected while queued (finding 2): no generation ran, nothing is
    persisted, and the body is empty (the client is gone, so it is never read).
    """

    async def _empty() -> AsyncIterator[str]:
        return
        yield  # pragma: no cover - makes this an async generator

    return StreamingResponse(_empty(), media_type=media_type)


def closed_completion_response() -> Response:
    """A trivially-closed NON-streaming response for a client that already
    disconnected while queued (finding 3): no generation ran, nothing is
    persisted, and the body is empty (the client is gone, so it is never read).
    204 No Content keeps it unambiguous that there is no completion to read."""
    return Response(status_code=204)
