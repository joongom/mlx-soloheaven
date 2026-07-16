"""CHILD process entrypoint for process-mode generation (Stage 1).

Spawned once at server startup. Constructs a real ``MLXEngine`` with
``execution_mode="main_thread"`` on THIS process's MAIN thread, loads the
model, and serves ``generate`` commands synchronously — iterating
``engine.generate_stream(...)`` inline (same batching/flush rules as
``MLXEngine.generate_stream_batches_async``) and streaming ``batch`` /
``final`` / ``done`` / ``error`` frames back over the response Pipe.

A small daemon CONTROL thread blocks on the ctrl pipe and sets the active
request's ``threading.Event`` on cancel — the ONLY background thread, and it
never touches MLX tensors (it only sets a Python event).

mlx is imported lazily here (inside ``worker_main``) so the module can be
imported in the parent for symbol access without pulling in MLX.
"""

import logging
import signal
import threading
import time
import traceback
from collections import OrderedDict

from mlx_soloheaven.engine import process_protocol as proto

logger = logging.getLogger(__name__)

# Periodic idle flush (process-mode stand-in for thread mode's GPU-keepalive
# idle flush, which never runs in main_thread mode): the main loop waits on
# the cmd pipe with a bounded poll() instead of a blocking recv(), and flushes
# dirty sessions once the pipe has been quiet for IDLE_FLUSH_AFTER_S.
IDLE_FLUSH_POLL_S = 5.0
IDLE_FLUSH_AFTER_S = 60.0

# GPU keepalive fallback interval (seconds). The authoritative value is
# MLXEngine.GPU_KEEPALIVE_INTERVAL (thread mode's constant) read off the
# engine at loop start; this fallback only covers engine stand-ins in tests.
GPU_KEEPALIVE_INTERVAL_FALLBACK_S = 1.0

# Log the first keepalive-touch failure with a traceback, then go quiet:
# a persistently failing ping would otherwise spam the log every interval.
_keepalive_error_logged = False

# U13: bound on remembered PRE-ACTIVATION cancels (cancel ops that arrive
# before their request activates — e.g. the client disconnected while its
# generate command was still queued behind a long generation). Insertion-
# ordered; the oldest entry is dropped past the cap. 256 is far beyond any
# realistic queue depth on a single-worker engine.
PENDING_CANCEL_MAX = 256

# F3: bound on remembered RPC tombstones (rpc_cancel ops for parent-side
# timed-out RPCs whose commands are still queued behind a generation).
# Mirrors the U13 pending-cancel pattern: insertion-ordered, oldest dropped
# past the cap; rids are uuid4-unique, so a stale tombstone can never match
# a future request. F5 residual (codex round 3, finding 3): every tombstone
# is PENDING-ACK — "cancelled" is only sent when the MAIN loop actually
# consumes the tombstone and skips dispatch; an eviction sends "unknown"
# for the evicted rid (the bound just revoked the skip promise, and the
# parent must never believe "cancelled" for an RPC that may still execute).
CANCELLED_RPC_MAX = 256

# F5 (batch-3 round 2): bound on remembered COMPLETED rpc rids — consulted
# by the ctrl loop so an rpc_cancel that crossed the RPC's reply on the wire
# acks "unknown" (it RAN) instead of "cancelled" (it did not). Same bounded
# insertion-ordered pattern as the two sets above.
COMPLETED_RPC_MAX = 256

# U17: ONLY these ops reset the idle-flush timer (``last_activity``).
# Read-only RPCs (stats / health / list / overview — anything an admin UI or
# monitor polls periodically) must NOT starve the 60s idle flush: a dashboard
# polling every 30s would otherwise keep dirty sessions unflushed forever.
# Future ops default to NON-resetting unless added here (generation-shaped =
# it generates tokens or mutates session/cache state that the flush cadence
# should trail).
GENERATION_OPS = frozenset({"generate", "generate_scalar"})
# F5 (batch-3 round 2): mutating-RPC IDEMPOTENCY — consulted when a
# parent-side timeout leaves an operation OUTCOME-UNKNOWN (see the
# rpc_cancel ack contract in process_protocol / the ctrl loop below).
# "Idempotent" = safe for the caller to blindly retry after an unknown
# outcome (re-running with the same args converges to the same state):
#
#   complete                — NOT idempotent (generates tokens; a retry
#                             re-generates and advances session state again)
#   compact_session         — idempotent (rebuilds to the given messages)
#   truncate_session        — idempotent (same target count -> same state)
#   prepare_regenerate      — NOT idempotent (each call strips another
#                             assistant turn via truncate_session)
#   branch_from_turn        — idempotent (same args overwrite the same branch)
#   update_session_messages — idempotent (touch + mark-dirty only)
#   delete_session          — idempotent (delete-after-delete is a no-op)
#   clear_caches / reset    — idempotent (converge to the empty state)
ACTIVITY_RPC_METHODS = frozenset({
    "complete",
    "compact_session",
    "truncate_session",
    "prepare_regenerate",
    "branch_from_turn",
    "update_session_messages",
    "delete_session",
    "clear_caches",
    "reset",
})


def _resets_activity(cmd) -> bool:
    """U17: True iff this command is generation-shaped and should reset the
    idle-flush timer. Read-only RPCs and unknown ops do not."""
    if not isinstance(cmd, dict):
        return False
    op = cmd.get("op")
    if op in GENERATION_OPS:
        return True
    if op == "rpc":
        return cmd.get("method") in ACTIVITY_RPC_METHODS
    return False


class _GracefulShutdown(BaseException):
    """Sentinel raised by the worker's SIGTERM/SIGINT handler to unwind the
    main loop. Derives from BaseException (NOT Exception) so the per-request
    ``except Exception`` blocks in the loop can never swallow it — it always
    reaches the loop's own handler, whose ``finally`` runs the shutdown
    flush."""


class _SendLockedConn:
    """resp-pipe wrapper serializing ``send()`` across threads.

    F5 (batch-3 round 2): the ctrl thread now sends ``rpc_cancel_ack``
    frames on the resp pipe while the MAIN loop may be sending
    batch/final/rpc_result frames — ``multiprocessing.Connection.send`` is
    not documented thread-safe, so both writers go through one lock. Only
    ``send`` is exposed: the worker never recv()s on the resp pipe."""

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, obj):
        with self._lock:
            self._conn.send(obj)


def _flush_engine_for_shutdown(engine, shutdown_flag=None):
    """Persist all dirty sessions before the child exits.

    SERIALIZATION: called only from the worker MAIN loop (shutdown op /
    sentinel unwind / pipe EOF), and that loop runs every generation to
    completion inline before recv()ing the next command — so this can never
    be concurrent with an active generation. The engine lock is still taken
    (bounded) inside _flush_all_on_shutdown for defense in depth: if the
    SIGTERM sentinel unwound a generation mid-flight, the lock is released
    by generate_stream's ``with self._lock`` on the way out, before we get
    here.

    SIGNAL-SAFE: ``shutdown_flag`` (the worker's shutdown Event) is set as
    the FIRST statement, unconditionally. On non-signal exit paths
    ('shutdown' op, None cmd, pipe EOF, unexpected loop error) the flag was
    never set by the signal handler — without this, a SIGTERM landing while
    this flush runs (e.g. the parent's close() join timeout firing
    terminate()) would look like a FIRST signal and raise the sentinel,
    aborting _flush_all_on_shutdown AFTER it already drained the dirty set;
    the atexit backstop would then no-op and the sessions would be lost.
    With the flag set, any signal arriving during the final flush takes the
    handler's 'already unwinding → return silently' path.

    FAIL-CLOSED: a failed flush logs and continues — it must never block or
    corrupt shutdown. Idempotent with the engine's own atexit backstop (the
    first flush drains the dirty set).
    """
    if shutdown_flag is not None:
        shutdown_flag.set()
    try:
        engine._flush_all_on_shutdown()
    except Exception:  # noqa: BLE001
        logger.exception("[child] shutdown flush failed (continuing)")


def _idle_flush(engine):
    """Periodic dirty-session flush on the child MAIN thread.

    Runs only between commands (the loop is the sole generation driver), so
    the engine lock should be free; acquire it non-blocking and bail rather
    than ever stalling the command loop. Fail-closed: errors are logged,
    never raised.
    """
    try:
        if not engine._lock.acquire(blocking=False):
            return
        try:
            engine._flush_dirty_sessions()
        finally:
            engine._lock.release()
    except Exception:  # noqa: BLE001
        logger.exception("[child] idle flush failed (ignored)")


def _gpu_keepalive_touch(engine, shutdown_flag=None):
    """GPU keepalive touch on the child MAIN thread (process-mode analog of
    thread mode's keepalive thread, which load_model never starts in
    main_thread mode — see MLXEngine._start_gpu_keepalive).

    Runs only from the main loop's poll-timeout branch, so it can never be
    concurrent with a generation (the loop drives every generation inline to
    completion before the next poll). The NON-BLOCKING lock acquire is
    defense in depth: if the lock is somehow held, skip — NEVER wait behind
    a generation. Skipped outright while shutdown is unwinding
    (``shutdown_flag`` set by the SIGTERM sentinel / final flush).

    Fail-closed: the first ping failure is logged with a traceback, later
    ones are suppressed, and the loop never dies.

    Returns True if the touch was ATTEMPTED (success or logged failure —
    the caller resets its interval timer either way, so a broken ping is
    retried at the normal cadence, not in a tight loop) and False if it was
    SKIPPED (shutdown unwinding / lock held).
    """
    global _keepalive_error_logged
    if shutdown_flag is not None and shutdown_flag.is_set():
        return False
    if not engine._lock.acquire(blocking=False):
        return False
    try:
        try:
            engine._gpu_keepalive_ping()
        except Exception:  # noqa: BLE001
            if not _keepalive_error_logged:
                _keepalive_error_logged = True
                logger.exception(
                    "[child] GPU keepalive touch failed "
                    "(logged once; further failures suppressed)"
                )
        return True
    finally:
        engine._lock.release()


def worker_main(cfg_dict, cmd_conn, resp_conn, ctrl_conn):
    """Child main. Runs on the child process's MAIN thread.

    cfg_dict : dict produced by ``_config_to_dict`` in process_client.
    cmd_conn : recv generate commands (PARENT -> CHILD).
    resp_conn: send ready/batch/final/done/error (CHILD -> PARENT).
    ctrl_conn: recv cancel frames (PARENT -> CHILD), read by daemon thread.
    """
    # Configure logging in the CHILD: it is a spawned process and does NOT
    # inherit the parent's logging.basicConfig. Without this the engine's
    # [Generate]/TTFT/tps INFO lines are dropped. stdout/stderr are inherited
    # from the parent (redirected to the server log file) so these surface
    # there for `tail -f` monitoring.
    cfg_verbose = bool(cfg_dict.get("verbose", False))
    logging.basicConfig(
        level=logging.DEBUG if cfg_verbose else logging.INFO,
        format="%(asctime)s [child:%(name)s] %(levelname)s: %(message)s",
    )

    # Heavy imports happen in the child only.
    from mlx_soloheaven.config import Config, ModelConfig  # noqa: F401
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    cfg = _dict_to_config(cfg_dict)

    # F5 (round 2): the ctrl thread sends cancel-ack frames on the resp
    # pipe concurrently with the main loop — serialize every send.
    resp_conn = _SendLockedConn(resp_conn)

    # Build + load the engine on THIS (main) thread.
    engine = MLXEngine(cfg, execution_mode="main_thread")
    engine.load_model()

    # Per-request cancel registry: request_id -> threading.Event.
    # Mutated by the main loop (add/remove) and read by the ctrl thread.
    active: dict[str, threading.Event] = {}
    active_lock = threading.Lock()
    # U13: cancels that arrived BEFORE their request activated (queued
    # request, client already gone). Consulted (and consumed) at request
    # activation so the child never runs a full generation for a
    # disconnected client. Bounded + insertion-ordered; entries are dropped
    # when consumed, when their request completes, or when the cap evicts
    # the oldest. Guarded by active_lock (same registry).
    pending_cancels: "OrderedDict[str, None]" = OrderedDict()
    # F3: tombstones for parent-side TIMED-OUT RPCs still queued on the cmd
    # pipe. Consulted (and consumed) right before an rpc op would execute —
    # the parent's reply slot is gone and late replies are dropped at its
    # reader, so executing the stale request would only delay the next
    # generation / idle flush / GPU keepalive. Guarded by active_lock.
    cancelled_rpcs: "OrderedDict[str, None]" = OrderedDict()
    # F5 (round 2): rids of RPCs the main loop has STARTED executing —
    # added in the SAME critical section that consumes the tombstone,
    # removed on completion. Lets the ctrl loop ack an rpc_cancel with the
    # truthful outcome ("already_started") instead of silently tombstoning
    # work already in flight. Guarded by active_lock.
    started_rpcs: set = set()
    # F5 (round 2): recently COMPLETED rpc rids (bounded, insertion-ordered)
    # so a cancel that crossed the RPC's reply on the wire acks "unknown"
    # (it ran) rather than "cancelled" (it did not). Guarded by active_lock.
    completed_rpcs: "OrderedDict[str, None]" = OrderedDict()

    def _ctrl_loop():
        # Daemon thread: block on cancel frames, set the matching event.
        # Never touches MLX — only flips a Python threading.Event / records
        # a tombstone id.
        while True:
            try:
                frame = ctrl_conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(frame, dict):
                continue
            op = frame.get("op")
            if op == "cancel":
                rid = frame.get("id")
                with active_lock:
                    ev = active.get(rid)
                    if ev is None and isinstance(rid, str):
                        # U13: request not active yet — remember the cancel
                        # so activation consumes it instead of generating.
                        pending_cancels[rid] = None
                        pending_cancels.move_to_end(rid)
                        while len(pending_cancels) > PENDING_CANCEL_MAX:
                            pending_cancels.popitem(last=False)
                if ev is not None:
                    ev.set()
            elif op == "rpc_cancel":
                # F3: parent-side bounded RPC wait expired — tombstone the
                # rid so the queued command is skipped, never executed.
                # F5 (round 2): a cancel CANNOT stop an RPC that already
                # started (or finished) — ack the DEFINITIVE outcome so the
                # parent never believes such an operation didn't run:
                #   started        -> "already_started" (runs to completion)
                #   completed      -> "unknown" (it ran; reply crossed the
                #                     cancel on the wire)
                #   still queued / -> PENDING-ACK tombstone, NO ack yet
                #   never seen        (codex round 3, finding 3): "cancelled"
                #                     is a promise the bounded tombstone set
                #                     cannot keep on its own — an evicted
                #                     tombstone's RPC EXECUTES. The ack is
                #                     therefore emitted by whoever settles
                #                     the tombstone's fate: the MAIN loop
                #                     sends "cancelled" when it consumes the
                #                     tombstone and skips dispatch; the
                #                     eviction below sends "unknown" for the
                #                     evicted rid (we no longer know).
                rid = frame.get("id")
                if isinstance(rid, str):
                    outcome = None
                    evicted: list = []
                    with active_lock:
                        if rid in started_rpcs:
                            outcome = "already_started"
                        elif rid in completed_rpcs:
                            outcome = "unknown"
                        else:
                            cancelled_rpcs[rid] = None
                            cancelled_rpcs.move_to_end(rid)
                            while len(cancelled_rpcs) > CANCELLED_RPC_MAX:
                                ev_rid, _ = cancelled_rpcs.popitem(last=False)
                                evicted.append(ev_rid)
                    # Sends happen OUTSIDE active_lock: pipe backpressure
                    # must never couple into the registry lock the main
                    # loop's dispatch path takes.
                    for ev_rid in evicted:
                        try:
                            resp_conn.send(
                                proto.make_rpc_cancel_ack(ev_rid, "unknown")
                            )
                        except Exception:  # noqa: BLE001 — best-effort ack
                            pass
                    if outcome is not None:
                        try:
                            resp_conn.send(
                                proto.make_rpc_cancel_ack(rid, outcome)
                            )
                        except Exception:  # noqa: BLE001 — best-effort ack
                            pass

    threading.Thread(target=_ctrl_loop, daemon=True, name="proc-ctrl").start()

    # --- graceful-shutdown signal wiring -----------------------------------
    # engine.load_model() registered direct-flush SIGTERM/SIGINT handlers
    # (MLXEngine._register_shutdown_flush). In the CHILD we REPLACE them with
    # a flag+sentinel pair: NO MLX work may run inside a signal handler here,
    # because the handler executes on this same main thread and could
    # interrupt an in-flight generation that holds the non-reentrant engine
    # lock (flushing from the handler would self-deadlock or touch tensors
    # mid-mutation). The handler only sets a flag and raises the sentinel;
    # the actual flush runs in the loop's ``finally`` below, on the main
    # thread, after the generation unwound and released the lock. The atexit
    # hook the engine registered stays as a last-resort backstop (safe: the
    # process is single-threaded-for-MLX at interpreter exit, and the flush
    # is idempotent).
    shutdown_requested = threading.Event()

    def _graceful_signal_handler(signum, frame):
        if shutdown_requested.is_set():
            # Second signal while already unwinding/flushing (e.g. parent
            # terminate() after its join timeout): do NOT raise again — it
            # would abort the in-progress flush. Let the first unwind finish.
            return
        shutdown_requested.set()
        raise _GracefulShutdown(signum)

    try:
        signal.signal(signal.SIGTERM, _graceful_signal_handler)
        signal.signal(signal.SIGINT, _graceful_signal_handler)
    except ValueError:  # pragma: no cover — worker_main runs on the main thread
        logger.warning("[child] signal handlers not installed (non-main thread)")

    # Signal readiness with model metadata the parent proxy exposes. The cfg
    # snapshot carries the read-only fields the API reads (default_*, thinking,
    # token IDs, budgets, cache_dir) AFTER the child's load-time detection
    # (e.g. think_end_token auto-detect, enable_thinking auto-disable) so the
    # parent's cfg view matches the child's authoritative config.
    resp_conn.send(proto.make_ready(
        model_id=engine.model_id,
        model_family=engine.model_family,
        enable_thinking=engine.cfg.enable_thinking,
        think_end_token=engine.cfg.think_end_token,
        cfg=_cfg_snapshot(engine.cfg),
    ))

    # Coalescing config (mirrors generate_stream_batches_async).
    coalesce_n = getattr(cfg, "stream_coalesce_n", 4)
    coalesce_ms = getattr(cfg, "stream_coalesce_ms", 30)
    coalescing = coalesce_n > 1

    # GPU keepalive (process mode): the engine skips its keepalive THREAD in
    # main_thread mode (no background MLX access), but between requests this
    # main loop idles in poll() — and this IS the child's main thread, so
    # the poll-timeout branch below may touch MLX safely. When enabled, the
    # poll granularity shrinks to the thread-mode keepalive interval so the
    # touch cadence matches thread mode; the idle flush and shutdown paths
    # are unaffected (the flush trigger is wall-clock based on
    # ``last_activity``, and a shorter poll only makes the loop MORE
    # responsive to 'shutdown' ops / signals).
    keepalive_enabled = bool(getattr(engine.cfg, "gpu_keepalive", False))
    keepalive_interval = float(getattr(
        engine, "GPU_KEEPALIVE_INTERVAL", GPU_KEEPALIVE_INTERVAL_FALLBACK_S
    ))
    poll_s = (
        min(IDLE_FLUSH_POLL_S, keepalive_interval)
        if keepalive_enabled else IDLE_FLUSH_POLL_S
    )
    last_keepalive = time.monotonic()

    # MAIN LOOP. Ops are strictly SERIAL: each generation runs to completion
    # inline before the next recv(), so a 'shutdown' op (or the sentinel /
    # pipe EOF) is never processed concurrently with an active generation —
    # the shutdown flush in the ``finally`` below therefore never races MLX
    # work. The only other threads are proc-ctrl (no MLX) and the engine's
    # disk-index helpers (none in main_thread mode).
    last_activity = time.monotonic()

    def _flush_if_idle_due():
        """Evaluate the U17 idle-flush deadline and flush when it elapsed.

        Codex round 11, finding 3: this used to run ONLY in the
        poll-timeout branch below. Reads correctly do NOT reset
        ``last_activity``, but sustained read-only RPC traffic arriving
        more often than the 5s poll timeout meant poll() always returned
        data and the deadline was never even CHECKED — dirty sessions aged
        unbounded under a busy dashboard. The command path now calls this
        after handling each NON-activity command (never after
        generation-shaped work, which just reset the timer).

        GPU keepalive deliberately stays OUT of this helper (poll-timeout
        branch only): the keepalive is a latency nicety — it keeps Metal
        warm across true idle gaps so the next generation starts fast —
        not a durability invariant like the flush. Under sustained read
        traffic a missed touch risks only a slightly slower first token
        later, while running MLX work inline here would add tail latency
        to every read reply it piggybacked on.
        """
        nonlocal last_activity
        if time.monotonic() - last_activity >= IDLE_FLUSH_AFTER_S:
            _idle_flush(engine)
            last_activity = time.monotonic()

    try:
        while not shutdown_requested.is_set():
            try:
                # Bounded wait instead of a blocking recv so the loop can run
                # the periodic idle flush (and GPU keepalive) while the
                # server sits quiet.
                if not cmd_conn.poll(poll_s):
                    now = time.monotonic()
                    # Idle flush FIRST — the keepalive touch must never
                    # starve or delay it.
                    _flush_if_idle_due()
                    if (
                        keepalive_enabled
                        and now - last_keepalive >= keepalive_interval
                        and _gpu_keepalive_touch(engine, shutdown_requested)
                    ):
                        last_keepalive = time.monotonic()
                    continue
                cmd = cmd_conn.recv()
            except (EOFError, OSError):
                break
            # U17: only GENERATION-SHAPED ops reset the idle-flush timer —
            # periodic read-only admin/health RPC traffic must not starve
            # the 60s flush window (an explicit allowlist; see
            # GENERATION_OPS / ACTIVITY_RPC_METHODS). Codex round 11,
            # finding 3: non-activity commands instead CHECK the deadline
            # after they are handled (each `_flush_if_idle_due()` call at
            # the non-activity continue sites below).
            cmd_resets_activity = _resets_activity(cmd)
            if cmd_resets_activity:
                last_activity = time.monotonic()
            if cmd is None:
                break
            if not isinstance(cmd, dict):
                _flush_if_idle_due()  # junk frame: still non-activity traffic
                continue
            op = cmd.get("op")
            if op == "shutdown":
                # Graceful stop requested by the parent (proxy.close()).
                # Break — the ``finally`` below flushes dirty sessions to
                # disk BEFORE the process exits.
                break

            # --- generic synchronous RPC (Stage 2) ---
            if op == "rpc":
                rid = cmd.get("id", "")
                # F3: consume a tombstone for a parent-side TIMED-OUT RPC —
                # its reply slot is gone (a late reply would drop at the
                # reader), so executing it would only delay the next
                # generation / idle flush / keepalive. Skip entirely.
                # F5 (round 2): mark STARTED in the SAME critical section
                # that consumed the tombstone — from this point on an
                # rpc_cancel acks "already_started" instead of tombstoning
                # an execution it can no longer stop.
                with active_lock:
                    rpc_stale = cancelled_rpcs.pop(rid, "__miss__") is None
                    if not rpc_stale and isinstance(rid, str):
                        started_rpcs.add(rid)
                if rpc_stale:
                    logger.info(
                        f"[child] rpc id={rid} cancelled (parent-side timeout)"
                        f" — skipping execution"
                    )
                    # F5 residual (codex round 3, finding 3): the tombstone
                    # was CONSUMED and dispatch skipped — only now is the
                    # "cancelled" promise true by construction, so only now
                    # is the ack sent (the ctrl thread registered the
                    # tombstone pending-ack, without acking).
                    try:
                        resp_conn.send(
                            proto.make_rpc_cancel_ack(rid, "cancelled")
                        )
                    except Exception:  # noqa: BLE001 — best-effort ack
                        pass
                    # Round 11, finding 3: a skipped (tombstoned) RPC is
                    # still inbound traffic — evaluate the flush deadline
                    # unless the method was activity-shaped.
                    if not cmd_resets_activity:
                        _flush_if_idle_due()
                    continue
                method = cmd.get("method", "")
                args = cmd.get("args", []) or []
                kwargs = cmd.get("kwargs", {}) or {}
                try:
                    _run_rpc(engine, rid, method, args, kwargs, resp_conn)
                except Exception as e:  # noqa: BLE001
                    resp_conn.send(proto.make_error(rid, str(e), traceback.format_exc()))
                finally:
                    # F5: retire the started mark and remember the
                    # COMPLETION (bounded) so a late rpc_cancel acks
                    # "unknown", never "cancelled". A tombstone here should
                    # be impossible (the ctrl thread sees the rid in
                    # started_rpcs for the whole execution and acks
                    # "already_started" instead of tombstoning), but a
                    # pending-ack tombstone must never vanish silently
                    # (finding 3) — if one is somehow found, the RPC RAN, so
                    # the definitive ack is "unknown".
                    with active_lock:
                        tomb_raced = (
                            cancelled_rpcs.pop(rid, "__miss__") is None
                        )
                        started_rpcs.discard(rid)
                        if isinstance(rid, str):
                            completed_rpcs[rid] = None
                            completed_rpcs.move_to_end(rid)
                            while len(completed_rpcs) > COMPLETED_RPC_MAX:
                                completed_rpcs.popitem(last=False)
                    if tomb_raced:
                        try:
                            resp_conn.send(
                                proto.make_rpc_cancel_ack(rid, "unknown")
                            )
                        except Exception:  # noqa: BLE001 — best-effort ack
                            pass
                # Round 11, finding 3: evaluate the flush deadline AFTER
                # handling a read/metadata RPC (never after activity-shaped
                # methods — they just reset the timer and their flush
                # cadence should trail them by the full quiet window). The
                # check runs after the reply went out, so it never delays
                # the response it rode in on.
                if not cmd_resets_activity:
                    _flush_if_idle_due()
                continue

            if op not in ("generate", "generate_scalar"):
                # Unknown op — report loudly against its id if present.
                resp_conn.send(proto.make_error(
                    cmd.get("id", ""), f"unknown op {op!r}", ""
                ))
                # Round 11, finding 3: unknown ops never reset the timer —
                # they are non-activity traffic like reads.
                _flush_if_idle_due()
                continue

            rid = cmd["id"]
            payload = cmd.get("payload", {})
            messages = payload.get("messages", [])
            params = dict(payload.get("params", {}))
            scalar = (op == "generate_scalar")

            cancel_event = threading.Event()
            with active_lock:
                # U13: consume a PRE-ACTIVATION cancel — the client
                # disconnected while this request was still queued. Skip the
                # generation entirely (the parent already tore its stream
                # routing down, so these terminal frames are dropped as late
                # frames — sent anyway to keep the frame contract uniform).
                pre_cancelled = pending_cancels.pop(rid, "__miss__") is None
                if not pre_cancelled:
                    active[rid] = cancel_event
            if pre_cancelled:
                logger.info(
                    f"[child] rid={rid} cancelled before activation — "
                    f"skipping generation"
                )
                resp_conn.send(proto.make_final(rid, None))
                resp_conn.send(proto.make_done(rid))
                continue

            try:
                _run_generate(
                    engine, rid, messages, params, cancel_event, resp_conn,
                    coalesce_n, coalesce_ms, coalescing, scalar=scalar,
                )
            except Exception as e:  # noqa: BLE001
                resp_conn.send(proto.make_error(rid, str(e), traceback.format_exc()))
            finally:
                with active_lock:
                    active.pop(rid, None)
                    # A cancel that raced this request's completion (arrived
                    # after the active entry was consulted but before this
                    # cleanup) must not linger as a pending entry.
                    pending_cancels.pop(rid, None)
    except _GracefulShutdown:
        logger.info("[child] SIGTERM/SIGINT received — graceful shutdown")
    finally:
        # Runs for EVERY exit path: 'shutdown' op, None/pipe EOF (parent
        # died), the SIGTERM/SIGINT sentinel, or an unexpected error.
        # Fail-closed inside (_flush_engine_for_shutdown never raises), and
        # it sets shutdown_requested FIRST so a signal landing mid-flush
        # (parent terminate() after its join timeout) is treated as
        # 'already unwinding' instead of raising the sentinel and aborting
        # the flush after the dirty set was drained.
        _flush_engine_for_shutdown(engine, shutdown_requested)


def _run_rpc(engine, rid, method, args, kwargs, resp_conn):
    """Dispatch a generic engine method call on the child main thread.

    Looks up ``getattr(engine, method)``, invokes it with the (already
    serialized-as-plain) args/kwargs, runs it to completion if it returns a
    coroutine, normalizes engine result objects to plain dicts, and sends an
    ``rpc_result`` frame. Exceptions surface as ``error`` frames.
    """
    import asyncio as _asyncio

    fn = getattr(engine, method, None)
    if fn is None or not callable(fn):
        resp_conn.send(proto.make_error(
            rid, f"engine has no callable method {method!r}", ""
        ))
        return

    # response_format crosses the wire as a plain dict (the parent serialized
    # its pydantic model); rehydrate to the engine-compatible duck-typed view.
    if "response_format" in kwargs:
        kwargs = dict(kwargs)
        kwargs["response_format"] = proto.deserialize_response_format(
            kwargs.get("response_format")
        )

    result = fn(*args, **kwargs)
    if _asyncio.iscoroutine(result):
        result = _asyncio.run(result)

    resp_conn.send(proto.make_rpc_result(rid, _serialize_rpc_result(result)))


def _serialize_rpc_result(result):
    """Normalize an engine method's return value to a plain, picklable value.

    CompletionResult / GenerationResult expose ``to_dict``. Everything else
    the wired methods return is already plain (dict / list / bool / None / str
    / int), so it passes through unchanged.
    """
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return {"__type__": type(result).__name__, "value": result.to_dict()}
    return result


def _cfg_snapshot(cfg) -> dict:
    """Serializable read-only snapshot of the cfg fields the API reads.

    Sent in the ``ready`` frame so the parent proxy's ``cfg`` view reflects the
    child's authoritative config after load-time detection (think_end_token,
    enable_thinking auto-disable, etc.)."""
    fields = (
        "default_temperature", "default_top_p", "default_min_p", "default_top_k",
        "default_repetition_penalty", "default_max_tokens", "thinking_budget",
        "enable_thinking", "think_end_token", "think_start_token",
        "memory_budget_gb", "disk_budget_gb", "mlx_cache_limit_gb",
        "cache_dir", "model_path",
    )
    return {f: getattr(cfg, f, None) for f in fields}


def _run_generate(engine, rid, messages, params, cancel_event, resp_conn,
                  coalesce_n, coalesce_ms, coalescing, scalar=False):
    """Iterate engine.generate_stream synchronously on the main thread,
    applying the same flush rules as generate_stream_batches_async and
    sending batch / final / done frames on resp_conn.

    ``scalar=True`` disables coalescing so every GenerationResult is posted as
    its own 1-item batch — mirrors generate_stream_async (used by the summary
    streamer / compaction)."""
    if scalar:
        coalescing = False
    # Rehydrate response_format into an engine-compatible duck-typed view.
    response_format = proto.deserialize_response_format(params.get("response_format"))

    gen = engine.generate_stream(
        messages,
        max_tokens=params.get("max_tokens"),
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        min_p=params.get("min_p"),
        top_k=params.get("top_k"),
        repetition_penalty=params.get("repetition_penalty"),
        session_id=params.get("session_id"),
        tools=params.get("tools"),
        cancel_event=cancel_event,
        thinking=params.get("thinking"),
        thinking_budget=params.get("thinking_budget"),
        response_format=response_format,
        stop=params.get("stop"),
    )

    batch: list[dict] = []
    last_flush = time.perf_counter()
    first_content_seen = False

    def _flush_batch():
        nonlocal batch, last_flush
        if batch:
            resp_conn.send(proto.make_batch(rid, batch))
            batch = []
            last_flush = time.perf_counter()

    final_sent = False
    for result in gen:
        if cancel_event.is_set():
            _flush_batch()
            break

        is_content = (result.status is None and result.finish_reason is None)
        flush_now = (
            result.status == "generating"
            or result.finish_reason is not None
            or (is_content and not first_content_seen)
            or not coalescing
        )
        if is_content:
            first_content_seen = True

        if result.finish_reason is not None:
            # Terminal frame: flush pending, then emit as a final frame.
            _flush_batch()
            resp_conn.send(proto.make_final(rid, result.to_dict()))
            final_sent = True
            continue

        if flush_now:
            _flush_batch()
            resp_conn.send(proto.make_batch(rid, [result.to_dict()]))
            last_flush = time.perf_counter()
            continue

        batch.append(result.to_dict())
        now = time.perf_counter()
        if len(batch) >= coalesce_n or (now - last_flush) * 1000 >= coalesce_ms:
            _flush_batch()

    _flush_batch()
    if not final_sent:
        resp_conn.send(proto.make_final(rid, None))
    resp_conn.send(proto.make_done(rid))


def _dict_to_config(cfg_dict):
    """Reconstruct a Config from the plain dict shipped by the parent."""
    from mlx_soloheaven.config import Config
    cfg = Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
