"""PARENT-side proxy for process-mode generation (Stage 2).

``EngineProcessProxy`` looks (enough) like an ``MLXEngine`` for EVERY endpoint:
the ``/v1/chat/completions`` streaming + non-streaming paths, the web-UI chat
endpoints, session lifecycle (list/get/delete/branch/truncate/regenerate),
compaction (summary streaming + compact_session), and admin cache
overview/reset. It exposes:

  - ``generate_stream_batches_async`` (batched streaming) and
    ``generate_stream_async`` (scalar streaming, for the compaction summarizer).
  - model metadata (``model_id``, ``model_family``, ``enable_thinking``,
    ``think_end_token``) populated from the child's ``ready`` frame.
  - ``cfg`` — a real Config passed in by the server, refreshed from the child's
    ``ready`` cfg snapshot so default_*/thinking/token-id reads match the child.
  - ``cache_manager`` — a tiny parent-side shim whose ``stats()`` forwards to the
    child's cache_overview RPC (chat.py /api/cache/stats reads this directly).
  - generic engine methods (``complete``, ``compact_session``,
    ``list_sessions``, ``session_stats``, ``get_session``, ``delete_session``,
    ``base_cache_stats``, ``branch_from_turn``, ``prepare_regenerate``,
    ``truncate_session``, ``update_session_messages``, ``cache_overview``,
    ``clear_caches``) — each forwards to the child via a generic synchronous RPC
    and blocks on the result. These callers invoke the methods synchronously
    (not awaited), so the proxy methods are synchronous too; the brief block on
    a fast metadata RPC matches the in-process engine's existing behavior.

The parent imports NO mlx / mlx_engine — all MLX work stays in the child.

Architecture: one CHILD process spawned with the spawn context; three Pipes
(cmd / resp / ctrl). One daemon READER thread does ``resp_conn.recv()`` and
routes frames: streaming frames (batch/final/done/error) go into per-request
asyncio.Queues; ``rpc_result``/``error`` for an RPC id resolve a threading
future registered in ``_rpc_results``.
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
import uuid
from dataclasses import asdict
from types import SimpleNamespace

from mlx_soloheaven.engine import process_protocol as proto
from mlx_soloheaven.engine.types import EngineBusyError, GenerationResult

logger = logging.getLogger(__name__)


def _config_to_dict(cfg) -> dict:
    """Flatten a Config dataclass to a plain dict for spawn-pickling.

    Config is a dataclass but holds a list of ModelConfig dataclasses in
    ``.models``; asdict() recurses those into plain dicts. The child only
    needs the scalar fields (it reconstructs a single-model Config), so we
    drop the nested ``models`` list to keep the payload trivially picklable.
    """
    d = asdict(cfg)
    d.pop("models", None)
    return d


class _RpcError(RuntimeError):
    """Raised parent-side when the child reports an error for an RPC call."""


class EngineRestartingError(RuntimeError):
    """The process-mode child worker is dead, respawning, or shut down.

    Raised FAST on new generate/RPC calls while the engine is unavailable so
    the API layer can surface HTTP 503 ("engine restarting, retry shortly")
    instead of handing out a 200 with a dead stream — the production incident
    this fixes: the child aborted mid-prefill (Metal GPU Timeout, libc++abi
    abort — uncatchable in-process) and the parent kept serving for 1.5 days.
    Also raised by in-flight RPC waits / streaming generators when the child
    dies underneath them (death error frames carry ``engine_dead=True``)."""


class _CacheManagerShim:
    """Parent-side stand-in for ``engine.cache_manager``.

    chat.py's /api/cache/stats reads ``engine.cache_manager.stats()`` directly.
    In process mode the real cache_manager lives in the child, so this shim
    forwards ``stats()`` to the child's ``cache_overview`` RPC and projects the
    cache_manager sub-dict (byte-identical shape to CacheManager.stats())."""

    def __init__(self, proxy):
        self._proxy = proxy

    def stats(self) -> dict:
        overview = self._proxy.cache_overview()
        return overview.get("cache_manager", {})


class EngineProcessProxy:
    """Parent-side stand-in for an MLXEngine backed by a child process."""

    def __init__(self, cfg):
        self.cfg = cfg
        # spawn context: clean interpreter in the child (required so the child
        # builds its own MLX state on its own main thread).
        self._ctx = mp.get_context("spawn")

        # Pipes + process are created PER SPAWN (see _spawn) so a respawn gets
        # fresh channels — None until start().
        self._cmd_parent = None
        self._resp_parent = None
        self._ctrl_parent = None
        self._proc = None

        # Model metadata, populated from the child's `ready` frame.
        self.model_id = ""
        self.model_family = "chatml"
        self.enable_thinking = bool(getattr(cfg, "enable_thinking", True))
        self.think_end_token = int(getattr(cfg, "think_end_token", -1))
        # Set from the child's `ready` frame: whether the model uses a
        # translation-schema chat template (translategemma) whose user content
        # must stay a structured list (see MLXEngine._is_translation_template).
        self._is_translation_template = False
        # Max context length (tokens) reported by the child's ready frame;
        # None until the child is ready / when its config doesn't declare it.
        # Exposed by GET /ready without an RPC (read straight off the proxy).
        self.max_context_length: int | None = None

        # Parent-side cache_manager shim (chat.py reads .cache_manager.stats()).
        self.cache_manager = _CacheManagerShim(self)

        # Per-request STREAMING routing. The reader thread routes batch/final/
        # done/error frames into these queues keyed by request id. Each queue is
        # created on the request's own event loop; routing uses
        # call_soon_threadsafe.
        self._queues: dict[str, asyncio.Queue] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._routing_lock = threading.Lock()

        # Per-request RPC results. Synchronous engine-method calls register a
        # threading.Event-backed slot keyed by request id; the reader thread
        # fills it on rpc_result / error and sets the event. The cmd pipe is
        # serialized with a send lock so a streaming send and an RPC send from
        # different threads never interleave a single frame.
        self._rpc_results: dict[str, dict] = {}
        self._rpc_lock = threading.Lock()
        self._cmd_send_lock = threading.Lock()
        self._ctrl_send_lock = threading.Lock()

        # Stage-1 single-flight tracking (loud-ish; not a hard gate).
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self._ready_event = threading.Event()
        self._reader_thread = None
        self._closed = False

        # --- liveness / auto-respawn state --------------------------------
        # _dead: the reader thread saw EOF/OSError on the resp pipe — the
        #   child DIED under us (uncatchable aborts like a Metal GPU Timeout
        #   libc++abi abort land here). Pending RPCs/streams are failed and a
        #   respawn is kicked; new calls fail fast until it completes.
        # _respawning: a single respawn loop is in flight. The False->True
        #   transition under _state_lock is the serialization gate (only one
        #   respawn at a time).
        # _permanently_dead: respawn attempts exhausted — the proxy stays
        #   dead and rejects everything until the server is restarted.
        self._state_lock = threading.Lock()
        self._dead = False
        self._dead_reason = ""
        self._respawning = False
        self._permanently_dead = False
        self._respawn_attempts = 0
        self._respawn_max_attempts = 3
        self._respawn_backoff_s = 5.0
        self._ready_timeout_s = 600.0
        # True once ANY child reached `ready`. Before that, a reader EOF is a
        # STARTUP failure owned by start()/_respawn_loop (they raise/retry) —
        # death handling + auto-respawn only apply to a once-healthy engine.
        self._ever_ready = False
        # Per-spawn GENERATION token. Each _spawn() increments it under
        # _state_lock and hands the new value (plus the new Process object) to
        # that spawn's reader thread. Death/ready callbacks carrying a STALE
        # generation are ignored: a terminated replacement's delayed EOF must
        # never mark a NEWER healthy child dead (late-EOF-kills-healthy-child
        # bug), and its late `ready` frame must never satisfy a newer spawn's
        # ready event.
        self._spawn_gen = 0
        # Death RECORDED (not acted on) for the CURRENT generation while
        # _dead was still True: a respawned replacement that sent `ready`
        # and then died BEFORE the respawn loop cleared _dead/_respawning.
        # The early "already dead" return in _handle_child_death must not
        # swallow that death — the respawn loop consumes this record (under
        # _state_lock) right before declaring alive and counts the attempt
        # as FAILED instead.
        self._gen_death_recorded = None
        self._gen_death_reason = ""

    # --- lifecycle --------------------------------------------------------

    def start(self, ready_timeout: float = 600.0):
        """Spawn the child, start the reader thread, block until `ready`."""
        self._ready_timeout_s = ready_timeout
        self._spawn(ready_timeout)
        logger.info(
            f"[ProcessProxy] child ready: model_id={self.model_id} "
            f"family={self.model_family} thinking={self.enable_thinking}"
        )

    def _spawn(self, ready_timeout: float):
        """Create fresh pipes, spawn the child, start a reader thread bound
        to the NEW resp pipe, and block until the child's `ready` frame.

        Shared by the initial start() and the auto-respawn path (model reload
        takes ~1-3 min either way). Raises RuntimeError when the child does
        not become ready in time; the half-spawned child is terminated."""
        from mlx_soloheaven.engine.process_worker import worker_main

        self._cmd_parent, cmd_child = self._ctx.Pipe()
        self._resp_parent, resp_child = self._ctx.Pipe()
        self._ctrl_parent, ctrl_child = self._ctx.Pipe()

        # New spawn generation: reader callbacks (death / ready) from any
        # OLDER spawn become stale the moment this increments. The ready
        # event is OWNED by this generation: it is created fresh per spawn
        # and assigned together with the generation bump in ONE _state_lock
        # block. Assigning the event BEFORE the bump (the old order) left a
        # window where a stale reader — already validated against the OLD
        # generation — could set the NEW spawn's event and release this
        # spawn's ready wait before its child was actually ready.
        # self._ready_event remains an alias for the CURRENT generation's
        # event (direct/test callers of _apply_ready fall back to it).
        ready_event = threading.Event()
        with self._state_lock:
            self._spawn_gen += 1
            spawn_gen = self._spawn_gen
            self._ready_event = ready_event

        self._proc = self._ctx.Process(
            target=worker_main,
            args=(_config_to_dict(self.cfg), cmd_child, resp_child, ctrl_child),
            name="mlx-engine-child",
            daemon=True,
        )
        self._proc.start()
        # Close the PARENT's copies of the child-side pipe ends. The child got
        # its own duplicated handles at spawn; keeping ours open would hold a
        # writer on the resp pipe forever, so the reader thread would NEVER
        # see EOF when the child dies — exactly the production incident where
        # the aborted child left the parent serving dead streams for days.
        for conn in (cmd_child, resp_child, ctrl_child):
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._resp_parent, spawn_gen, self._proc, ready_event),
            daemon=True, name="proc-reader",
        )
        self._reader_thread.start()

        # Wait on the LOCAL event owned by THIS spawn — never the mutable
        # self._ready_event alias, which a newer concurrent _spawn may have
        # replaced by the time this wait runs.
        if not ready_event.wait(timeout=ready_timeout):
            try:
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                "process-mode child did not become ready within "
                f"{ready_timeout}s (model load failed?)"
            )

    def close(self, join_timeout: float = 10.0):
        """Graceful child stop. Idempotent.

        Sends the 'shutdown' op and WAITS (bounded) so the child's main loop
        can flush dirty sessions to disk before exiting — measured DISK SAVE
        cost is ~0.01s per session, so seconds-scale is generous. A wedged
        child (e.g. mid-generation past the timeout) is terminate()d; its
        SIGTERM handler still attempts the same graceful flush on the way
        out, so we join briefly once more to let that finish.

        RESPAWN INTERPLAY: setting ``_closed`` FIRST makes every liveness
        path stand down — the reader thread's EOF (caused by this shutdown)
        is treated as the expected end of life instead of a death,
        _kick_respawn refuses to start, and an already-running respawn loop
        aborts (or stops the fresh child it just spawned). A closed proxy
        NEVER respawns."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            with self._cmd_send_lock:
                self._cmd_parent.send({"op": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            self._proc.join(timeout=join_timeout)
            if self._proc.is_alive():
                logger.warning(
                    f"[ProcessProxy] child did not exit within {join_timeout}s "
                    f"after 'shutdown' — sending SIGTERM (child's handler "
                    f"still flushes)"
                )
                self._proc.terminate()
                self._proc.join(timeout=join_timeout)

    # --- reader thread ----------------------------------------------------

    def _reader_loop(self, resp_conn=None, gen=None, proc=None,
                     ready_event=None):
        """Drain resp pipe; route streaming frames into per-request asyncio
        queues and RPC results into their threading-future slots.

        ``gen`` is this reader's spawn generation, ``proc`` its Process
        object, and ``ready_event`` its generation-OWNED ready event (all
        captured at spawn time, so death reporting never reads the exitcode
        of a NEWER child through self._proc and a ready frame never sets a
        NEWER spawn's event through the mutable self._ready_event alias).
        ``gen=None`` (direct test drives) skips the staleness check.

        On EOF/OSError the child is GONE (its pipe ends closed): hand off to
        _handle_child_death, which marks the proxy dead, fails every pending
        RPC / live stream, and kicks the auto-respawn — unless close() owns
        this shutdown or this reader's spawn generation is stale."""
        if resp_conn is None:
            resp_conn = self._resp_parent
        reason = "resp pipe closed"
        while True:
            try:
                frame = resp_conn.recv()
            except EOFError:
                reason = "resp pipe EOF (child worker exited)"
                break
            except OSError as e:
                reason = f"resp pipe OSError: {e}"
                break
            if not isinstance(frame, dict):
                continue

            ftype = frame.get("type")
            if ftype == "ready":
                self._apply_ready(frame, gen=gen, ready_event=ready_event)
                continue

            rid = frame.get("id")
            if rid is None:
                continue

            if ftype == "rpc_cancel_ack":
                # F5 (round 2): the worker's definitive fate for a timed-out
                # RPC this parent already gave up on (its slot is gone).
                # Log the truth — "already_started"/"unknown" mean the
                # operation ran (or is running) despite the timeout report.
                outcome = frame.get("outcome", "unknown")
                if outcome == "cancelled":
                    logger.info(
                        f"[ProcessProxy] rpc_cancel ack rid={rid}: "
                        f"outcome='cancelled' — the child skipped it"
                    )
                else:
                    logger.warning(
                        f"[ProcessProxy] rpc_cancel ack rid={rid}: "
                        f"outcome={outcome!r} — the timed-out RPC executed "
                        f"(or was already executing) child-side"
                    )
                continue

            # RPC result / error for a pending synchronous call.
            with self._rpc_lock:
                slot = self._rpc_results.get(rid)
            if slot is not None and ftype in ("rpc_result", "error"):
                slot["frame"] = frame
                slot["event"].set()
                continue

            # Otherwise it's a streaming frame — route to the request's queue.
            with self._routing_lock:
                q = self._queues.get(rid)
                loop = self._loops.get(rid)
            if q is None or loop is None:
                # Late frame for a finished/cancelled request — drop.
                continue
            try:
                loop.call_soon_threadsafe(q.put_nowait, frame)
            except RuntimeError:
                # Loop closed — request gone.
                pass

        self._handle_child_death(reason, gen=gen, proc=proc)

    # --- death detection / auto-respawn -----------------------------------

    def _death_frame(self, rid, reason: str) -> dict:
        """Error frame injected parent-side when the child dies. The
        ``engine_dead`` marker makes _rpc/_stream raise EngineRestartingError
        (503-mappable) instead of a generic RuntimeError."""
        return {
            "id": rid,
            "type": "error",
            "error": f"engine child worker died: {reason}",
            "traceback": "",
            "engine_dead": True,
        }

    def _handle_child_death(self, reason: str, gen=None, proc=None):
        """Mark the proxy dead, fail every in-flight RPC/stream, kick respawn.

        Only the first caller of a death episode does the work (guarded by
        the _dead transition). A CLOSED proxy is left alone — close() owns
        that shutdown and the proxy must never be resurrected.

        STALE GENERATIONS: ``gen`` is the calling reader's spawn generation.
        A callback whose gen != the CURRENT _spawn_gen belongs to an OBSOLETE
        child (e.g. a respawn attempt that timed out and was terminated,
        whose reader delivers its EOF only after a newer replacement became
        ready) — acting on it would kill the healthy replacement, fail its
        RPCs/streams, and kick a useless respawn. Ignore it. ``gen=None``
        (direct/test callers) skips the check. The exitcode is reported from
        the CAPTURED ``proc`` (that spawn's own Process), never self._proc,
        which may already point at a newer child.

        SESSIONS: the KV caches living in the dead child are gone. That is
        acceptable — the worker's 60s idle flush persists dirty sessions to
        disk and the respawned child reloads them lazily on the next request,
        so at most the last <60s of un-flushed conversation state has to be
        re-prefilled."""
        if self._closed:
            logger.info(f"[ProcessProxy] reader exiting after close(): {reason}")
            return
        if not self._ever_ready:
            # Startup spawn never became ready: start() raises to the caller
            # and a respawn attempt's own failure is handled by _respawn_loop
            # (this reader belongs to a child terminated on ready-timeout).
            logger.info(
                f"[ProcessProxy] reader exiting before first ready: {reason}"
            )
            return
        with self._state_lock:
            if gen is not None and gen != self._spawn_gen:
                logger.info(
                    f"[ProcessProxy] ignoring death callback from obsolete "
                    f"spawn gen={gen} (current gen={self._spawn_gen}): {reason}"
                )
                return
            first_death = not self._dead
            if first_death:
                self._dead = True
                self._dead_reason = reason
            else:
                # CURRENT-generation death while _dead is STILL True: the
                # respawned replacement sent `ready` and then died before
                # the respawn loop cleared _dead/_respawning. Swallowing it
                # here (the old early return) let the loop clear _dead and
                # declare a DEAD child alive — with its reader already gone,
                # nothing would ever notice. RECORD the death instead; the
                # loop checks this record (under _state_lock) right before
                # declaring alive and counts that attempt as FAILED. A death
                # arriving in the window AFTER the loop clears _dead needs
                # no record: _dead is False by then and the gen matches, so
                # it takes the normal first-death path above.
                recorded_gen = self._spawn_gen if gen is None else gen
                self._gen_death_recorded = recorded_gen
                self._gen_death_reason = reason

        exitcode = None
        if proc is None:
            proc = self._proc
        if proc is not None:
            try:
                exitcode = proc.exitcode
            except Exception:  # noqa: BLE001
                pass

        # Fail ALL pending RPC slots so their waiters unblock immediately
        # instead of burning their full timeout against a dead child.
        with self._rpc_lock:
            rpc_slots = list(self._rpc_results.items())
        for rid, slot in rpc_slots:
            slot["frame"] = self._death_frame(rid, reason)
            slot["event"].set()

        # Fail ALL live streaming queues so waiting generators terminate with
        # a clear message instead of heartbeating a dead stream forever.
        with self._routing_lock:
            stream_targets = [
                (rid, self._queues.get(rid), self._loops.get(rid))
                for rid in list(self._queues.keys())
            ]
        for rid, q, loop in stream_targets:
            if q is None or loop is None:
                continue
            try:
                loop.call_soon_threadsafe(q.put_nowait, self._death_frame(rid, reason))
            except RuntimeError:
                # Loop closed — request gone anyway.
                pass

        if first_death:
            logger.critical(
                f"[ProcessProxy] child worker DIED ({reason}; "
                f"exitcode={exitcode}) — failed {len(rpc_slots)} pending "
                f"RPC(s) and {len(stream_targets)} live stream(s); "
                f"auto-respawn starting"
            )
            self._kick_respawn()
        else:
            # Recorded death: the running respawn loop owns retry/backoff —
            # kicking here would be refused by the _respawning gate anyway.
            logger.critical(
                f"[ProcessProxy] replacement child (gen={recorded_gen}) died "
                f"after ready but before respawn completed ({reason}; "
                f"exitcode={exitcode}) — death recorded; failed "
                f"{len(rpc_slots)} pending RPC(s) and {len(stream_targets)} "
                f"live stream(s); the respawn loop will count this attempt "
                f"as FAILED"
            )

    def _kick_respawn(self):
        """Start the (single) respawn loop if the proxy is dead and nothing
        is already respawning. Called on death detection AND lazily from
        ensure_available() so a respawn thread that died unexpectedly still
        gets retried by the next request — until attempts are exhausted."""
        with self._state_lock:
            if (self._closed or self._permanently_dead or self._respawning
                    or not self._dead):
                return
            self._respawning = True
        threading.Thread(
            target=self._respawn_loop, daemon=True, name="proc-respawn"
        ).start()

    def _respawn_loop(self):
        """Serialized respawn driver (the False->True ``_respawning``
        transition in _kick_respawn is the gate — one loop at a time).

        Bounded attempts with linear backoff; each attempt reuses the exact
        spawn/ready path start() uses, so a successful respawn means the
        model is fully reloaded (~1-3 min) and the ready cfg snapshot was
        re-applied. While this runs, new calls fail fast (503 at the API).
        After the FINAL failure the proxy goes permanently dead — operator
        restart required — and keeps rejecting with EngineRestartingError."""
        try:
            for attempt in range(1, self._respawn_max_attempts + 1):
                if self._closed:
                    logger.info("[ProcessProxy] respawn aborted — proxy closed")
                    return
                self._respawn_attempts = attempt
                logger.warning(
                    f"[ProcessProxy] respawning child worker (attempt "
                    f"{attempt}/{self._respawn_max_attempts}) — model reload "
                    f"may take minutes; requests fail fast (503) meanwhile"
                )
                if not self._reap_dead_child():
                    # The stuck old worker refused to die (terminate+kill
                    # both failed) — spawning now would put TWO live workers
                    # on the GPU. Refuse this attempt; retry after backoff.
                    logger.critical(
                        f"[ProcessProxy] respawn attempt {attempt} refused — "
                        f"old child could not be reaped"
                    )
                    if attempt < self._respawn_max_attempts:
                        time.sleep(self._respawn_backoff_s * attempt)
                    continue
                try:
                    self._spawn(self._ready_timeout_s)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        f"[ProcessProxy] respawn attempt {attempt} failed"
                    )
                    if attempt < self._respawn_max_attempts:
                        time.sleep(self._respawn_backoff_s * attempt)
                    continue
                if self._closed:
                    # close() raced the respawn — a closed proxy must never
                    # come back alive; stop the fresh child gracefully.
                    logger.info(
                        "[ProcessProxy] respawn finished after close() — "
                        "stopping the fresh child"
                    )
                    self._shutdown_spawned_child()
                    return
                with self._state_lock:
                    # The replacement may have died AFTER sending `ready` but
                    # BEFORE this state-clear: its reader's death callback ran
                    # while _dead was still True and RECORDED the death (see
                    # _handle_child_death) instead of being swallowed. Consume
                    # the record for the generation just spawned BEFORE
                    # declaring alive — clearing _dead over a dead child (its
                    # reader already gone) would leave the proxy claiming
                    # alive forever. A death arriving AFTER this block still
                    # takes the normal path: _dead is False by then and its
                    # gen matches, so _handle_child_death marks dead and
                    # kicks a fresh respawn.
                    if self._gen_death_recorded == self._spawn_gen:
                        death_reason = self._gen_death_reason
                        self._dead_reason = death_reason
                    else:
                        death_reason = None
                        self._dead = False
                        self._dead_reason = ""
                        self._respawning = False
                    # Consumed either way (a non-matching record is stale —
                    # it belongs to an older, already-superseded generation).
                    self._gen_death_recorded = None
                    self._gen_death_reason = ""
                if death_reason is not None:
                    logger.critical(
                        f"[ProcessProxy] respawn attempt {attempt} failed — "
                        f"replacement child died right after ready "
                        f"({death_reason})"
                    )
                    if attempt < self._respawn_max_attempts:
                        time.sleep(self._respawn_backoff_s * attempt)
                    continue
                logger.warning(
                    f"[ProcessProxy] child worker respawned and READY "
                    f"(attempt {attempt}) — resuming service"
                )
                return
            with self._state_lock:
                self._permanently_dead = True
            logger.critical(
                f"[ProcessProxy] child worker respawn FAILED after "
                f"{self._respawn_max_attempts} attempts — engine is "
                f"PERMANENTLY dead; all requests will be rejected (503) "
                f"until the server is restarted"
            )
        finally:
            # Exception safety: whatever path exits, the gate reopens (a
            # permanently-dead proxy never re-enters via _kick_respawn).
            with self._state_lock:
                self._respawning = False

    def _reap_dead_child(self) -> bool:
        """Reap the dead child process object before spawning a fresh one.

        Returns True when the old child is confirmed gone (or there was
        none). ESCALATION: terminate() (SIGTERM) + bounded join; if the
        process is STILL alive, kill() (SIGKILL) + a second bounded join. If
        it survives even that, return False — the caller must NOT spawn a
        replacement (a stuck-but-alive old worker plus a fresh one would
        double the GPU/memory residency); the respawn loop counts it as a
        failed attempt and retries after backoff."""
        proc = self._proc
        if proc is None:
            return True
        try:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                logger.warning(
                    "[ProcessProxy] old child still alive after terminate()+"
                    "join — escalating to kill()"
                )
                proc.kill()
                proc.join(timeout=5.0)
            if proc.is_alive():
                logger.critical(
                    "[ProcessProxy] old child STILL alive after terminate()+"
                    "kill() — refusing to spawn a replacement (double GPU "
                    "residency); staying dead until a later attempt reaps it"
                )
                return False
        except Exception:  # noqa: BLE001
            pass
        self._proc = None
        return True

    def _shutdown_spawned_child(self):
        """Stop a freshly-respawned child when close() raced the respawn —
        same graceful shutdown-op + bounded-join semantics as close(),
        INCLUDING the second bounded join after terminate(): the child's
        SIGTERM handler still flushes dirty sessions to disk on the way out,
        and cutting the parent's shutdown short would truncate that flush."""
        try:
            with self._cmd_send_lock:
                self._cmd_parent.send({"op": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        proc = self._proc
        if proc is not None:
            try:
                proc.join(timeout=10.0)
                if proc.is_alive():
                    logger.warning(
                        "[ProcessProxy] fresh child did not exit after "
                        "'shutdown' — sending SIGTERM (child's handler still "
                        "flushes) and waiting for the flush"
                    )
                    proc.terminate()
                    proc.join(timeout=10.0)
            except Exception:  # noqa: BLE001
                pass

    def ensure_available(self):
        """Fail FAST when the child is dead / respawning / closed.

        Called at the top of every generate/RPC entry point AND by the API
        layer before starting an SSE response, so clients get an immediate
        HTTP 503 ("engine restarting, retry shortly") instead of a 200 with
        a dead stream. Lazily kicks the respawn loop when needed."""
        if self._closed:
            raise EngineRestartingError("engine is shut down")
        if self._permanently_dead:
            raise EngineRestartingError(
                "engine child worker is permanently dead (respawn attempts "
                "exhausted) — server restart required"
            )
        if self._respawning:
            raise EngineRestartingError(
                "engine is restarting (child worker respawning), retry shortly"
            )
        if self._dead:
            self._kick_respawn()
            raise EngineRestartingError(
                f"engine child worker died ({self._dead_reason}) — respawn "
                f"in progress, retry shortly"
            )

    def liveness(self) -> dict:
        """Admin-visible liveness snapshot (alive / respawning / attempts)."""
        with self._state_lock:
            return {
                "alive": not (
                    self._dead or self._permanently_dead or self._closed
                ),
                "respawning": self._respawning,
                "respawn_attempts": self._respawn_attempts,
                "max_respawn_attempts": self._respawn_max_attempts,
                "permanently_dead": self._permanently_dead,
                "closed": self._closed,
                "dead_reason": self._dead_reason or None,
            }

    def _apply_ready(self, frame: dict, gen=None, ready_event=None):
        # STALE GENERATIONS: a `ready` from an obsolete spawn (e.g. a
        # ready-timeout-terminated replacement that got its frame out just
        # before dying) must not overwrite the current child's metadata or
        # spuriously set a NEWER spawn's ready event. gen=None (tests /
        # direct callers) skips the check.
        #
        # Validation, metadata publication, and the ready-event set happen
        # in ONE _state_lock critical section: releasing the lock between
        # validation and the set (the old shape) let a stale reader —
        # validated while its generation was still current — race _spawn's
        # gen bump and set the NEW generation's event through the mutable
        # self._ready_event alias, releasing the new spawn's ready wait
        # before its child was actually ready. The event set here is the
        # reader's CAPTURED, generation-owned event; direct callers without
        # one fall back to the current alias (read under the same lock).
        with self._state_lock:
            if gen is not None and gen != self._spawn_gen:
                logger.info(
                    f"[ProcessProxy] ignoring ready frame from obsolete "
                    f"spawn gen={gen} (current gen={self._spawn_gen})"
                )
                return
            self.model_id = frame.get("model_id", "")
            self.model_family = frame.get("model_family", "chatml")
            self.enable_thinking = bool(frame.get("enable_thinking", True))
            self.think_end_token = int(frame.get("think_end_token", -1))
            self._is_translation_template = bool(
                frame.get("is_translation_template", False)
            )
            ctx = frame.get("context_length")
            self.max_context_length = int(ctx) if isinstance(ctx, int) else None
            # Refresh cfg view from the child's authoritative post-load
            # snapshot so the API's default_*/thinking/token-id reads match
            # the child.
            snap = frame.get("cfg") or {}
            for k, v in snap.items():
                if v is not None:
                    try:
                        setattr(self.cfg, k, v)
                    except Exception:  # noqa: BLE001
                        pass
            # Keep these two consistent even if not in the snapshot.
            try:
                self.cfg.enable_thinking = self.enable_thinking
                self.cfg.think_end_token = self.think_end_token
            except Exception:  # noqa: BLE001
                pass
            # Death handling / auto-respawn arm only after the first ready.
            self._ever_ready = True
            if ready_event is None:
                ready_event = self._ready_event
            ready_event.set()

    # --- generic synchronous RPC -----------------------------------------

    # U14: bounded wait for READ-ONLY RPCs. The child services RPCs only
    # BETWEEN generations, so a stats/list call issued mid-generation would
    # otherwise block for the generation's full duration. On expiry the read
    # raises EngineBusyError and the caller degrades (busy placeholder /
    # 503) — mirrors the in-process engine's _read_locked bound.
    RPC_READ_TIMEOUT_S = 10.0

    # Batch C round 3, finding 1(b): ceiling (seconds) on how long a cancelled
    # streaming request waits for the child's terminal ``done`` ack before its
    # aclose returns (and the gate lease is released). The child now closes its
    # generate_stream — running the C1 teardown — BEFORE emitting ``done``, so
    # awaiting ``done`` means the child turn is wound down. Bounded + fail-closed
    # against a wedged/dead child so the release never hangs forever.
    CANCEL_ACK_TIMEOUT_S = 30.0

    def _rpc_read(self, method: str, *args, **kwargs):
        """Read-only RPC: bounded wait + EngineBusyError on expiry."""
        return self._rpc(
            method, *args,
            timeout=self.RPC_READ_TIMEOUT_S, busy_on_timeout=True, **kwargs,
        )

    def _rpc(self, method: str, *args, timeout: float = 600.0,
             busy_on_timeout: bool = False, **kwargs):
        """Forward a synchronous engine-method call to the child and block on
        the result. Returns the deserialized result; raises _RpcError on a
        child-side exception (EngineBusyError instead when
        ``busy_on_timeout`` and the wait expired — the child is busy
        generating, not broken), EngineRestartingError when the child is
        dead / respawning (fail fast — API maps it to 503).

        BLOCKING: this wait blocks its calling THREAD. Async endpoints must
        never call it on the event loop thread — the API layer wraps every
        proxy RPC call site in the dedicated ``executors`` pools
        (``run_long``/``run_read``/``run_critical``, U14) so one admin/
        metadata RPC during a long generation cannot freeze every in-flight
        SSE stream."""
        self.ensure_available()
        rid = uuid.uuid4().hex
        ev = threading.Event()
        with self._rpc_lock:
            self._rpc_results[rid] = {"event": ev, "frame": None}
        try:
            try:
                with self._cmd_send_lock:
                    self._cmd_parent.send(
                        proto.make_rpc(rid, method, list(args), kwargs)
                    )
            except OSError as e:
                # Child died between ensure_available and the send — the
                # reader's EOF will run the full death handling.
                raise EngineRestartingError(
                    f"engine cmd pipe write failed ({e}) — child worker dead?"
                ) from e
            # Sliced wait: the death sweep sets our event with an engine_dead
            # frame, but a death landing in the narrow window BEFORE this slot
            # registered would leave us blind — the dead-flag check each slice
            # is the backstop (1s worst-case instead of the full timeout).
            # Slices shrink near the deadline so sub-second timeouts expire
            # on time.
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # F3: the command may still sit QUEUED child-side (the
                    # child serves RPCs only between generations). Removing
                    # only the parent slot would leave the child to execute
                    # every stale request after the generation ends —
                    # delaying the next generation / idle flush / GPU
                    # keepalive (its late reply is dropped at the reader
                    # regardless). Send a tombstone on the ctrl pipe so the
                    # worker SKIPS execution. Best-effort: a dead pipe just
                    # means the child is gone anyway.
                    try:
                        with self._ctrl_send_lock:
                            self._ctrl_parent.send(proto.make_rpc_cancel(rid))
                    except Exception:  # noqa: BLE001
                        pass
                    if busy_on_timeout:
                        # Read-only RPC: a busy child is the diagnosis; the
                        # tombstone guarantees the stale read never runs.
                        # Engine is UP but generating -> 429 queue_full (batch
                        # B) if this propagates to the app handler.
                        raise EngineBusyError(
                            f"engine busy (generation in flight) — RPC "
                            f"{method!r} not served within {timeout}s, "
                            f"retry shortly",
                            reason=EngineBusyError.REASON_QUEUE_FULL,
                        )
                    # F5 (round 2): a MUTATING RPC that timed out is
                    # OUTCOME-UNKNOWN — the tombstone only stops a
                    # still-QUEUED command; the child may be executing (or
                    # have executed) this one. Never imply it didn't run;
                    # the worker's best-effort cancel-ack (logged by the
                    # reader) records the definitive outcome when it
                    # arrives. See the idempotency table next to
                    # ACTIVITY_RPC_METHODS in process_worker for which
                    # methods are safe to blindly retry.
                    logger.warning(
                        f"[ProcessProxy] RPC {method!r} timed out after "
                        f"{timeout}s — OUTCOME UNKNOWN: the operation may "
                        f"still have executed child-side (rpc_cancel sent; "
                        f"awaiting best-effort ack)"
                    )
                    raise _RpcError(
                        f"process-mode RPC {method!r} timed out after "
                        f"{timeout}s — outcome unknown (may have executed)"
                    )
                if ev.wait(timeout=min(1.0, remaining)):
                    break
                if self._dead or self._permanently_dead or self._closed:
                    raise EngineRestartingError(
                        f"engine child worker died during RPC {method!r} "
                        f"({self._dead_reason or 'closed'})"
                    )
            with self._rpc_lock:
                slot = self._rpc_results.get(rid)
            frame = (slot or {}).get("frame") or {}
        finally:
            with self._rpc_lock:
                self._rpc_results.pop(rid, None)

        if frame.get("type") == "error":
            err = frame.get("error", "unknown")
            tb = frame.get("traceback", "")
            if frame.get("engine_dead"):
                logger.error(f"[ProcessProxy] RPC {method!r} aborted: {err}")
                raise EngineRestartingError(
                    f"process-mode RPC {method!r} failed: {err}"
                )
            logger.error(f"[ProcessProxy] RPC {method!r} error: {err}\n{tb}")
            raise _RpcError(f"process-mode RPC {method!r} error: {err}")
        return _deserialize_rpc_result(frame.get("result"))

    # --- generation (streaming) ------------------------------------------

    async def generate_stream_batches_async(self, messages, **params):
        """Send a generate command; yield list[GenerationResult] batches.

        Batch C round 4, finding 1: HOIST ``_stream`` into a local and close it
        in a ``finally``. A plain ``async for batch in self._stream(...): yield``
        never closes the inner generator, so an ``aclose()`` of THIS public proxy
        generator — a client disconnect mid-send OR the response wrapper closing
        it after the child emits its terminal batch on normal completion — would
        leave ``_stream`` suspended and skip its cancel handler entirely. With the
        hoist, GeneratorExit CASCADES into ``_stream``'s
        ``except (CancelledError, GeneratorExit)`` (~1028), which sends the child
        cancel and awaits ``_await_child_cancel_ack`` (the child's C1 teardown)
        BEFORE this aclose returns and the gate lease is released."""
        stream = self._stream(messages, params, scalar=False)
        try:
            async for batch in stream:
                yield batch
        finally:
            await stream.aclose()

    async def generate_stream_async(self, messages, **params):
        """Send a scalar generate command; yield individual GenerationResult.

        Mirrors MLXEngine.generate_stream_async — used by the compaction summary
        streamer (CompactionEngine.generate_summary_stream).

        Batch C round 4, finding 1: same HOIST + ``finally: await stream.aclose()``
        as generate_stream_batches_async so closing this public generator cascades
        GeneratorExit into ``_stream``'s cancel handler (child cancel + C1 ack)
        before the aclose returns and the lease releases."""
        stream = self._stream(messages, params, scalar=True)
        try:
            async for batch in stream:
                for item in batch:
                    yield item
        finally:
            await stream.aclose()

    async def _stream(self, messages, params, scalar: bool):
        """Shared streaming driver for batched + scalar paths. Yields
        list[GenerationResult] batches (scalar path yields 1-item batches).
        Fails FAST with EngineRestartingError while the child is dead /
        respawning, and terminates with the same error if the child dies
        mid-generation (death error frame from _handle_child_death)."""
        self.ensure_available()
        loop = asyncio.get_event_loop()
        rid = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()

        with self._routing_lock:
            self._queues[rid] = q
            self._loops[rid] = loop
        with self._inflight_lock:
            self._inflight += 1

        # Serialize response_format (pydantic -> dict) for the wire.
        rf = params.get("response_format")
        wire_params = {
            "session_id": params.get("session_id"),
            "max_tokens": params.get("max_tokens"),
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "min_p": params.get("min_p"),
            "top_k": params.get("top_k"),
            "repetition_penalty": params.get("repetition_penalty"),
            "tools": params.get("tools"),
            "thinking": params.get("thinking"),
            "thinking_budget": params.get("thinking_budget"),
            "response_format": proto.serialize_response_format(rf),
            # U24: OpenAI stop sequences (plain str/list — wire-safe as-is).
            "stop": params.get("stop"),
        }

        make_cmd = proto.make_generate_scalar if scalar else proto.make_generate

        cancel_sent = False
        try:
            try:
                with self._cmd_send_lock:
                    self._cmd_parent.send(make_cmd(rid, messages, wire_params))
            except OSError as e:
                # Child died between ensure_available and the send — the
                # reader's EOF will run the full death handling.
                raise EngineRestartingError(
                    f"engine cmd pipe write failed ({e}) — child worker dead?"
                ) from e

            while True:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Backstop for the narrow window where the death sweep ran
                    # BEFORE this request's queue registered (no error frame
                    # will ever arrive) — never heartbeat a dead child.
                    if self._dead or self._permanently_dead or self._closed:
                        raise EngineRestartingError(
                            "engine child worker died mid-generation "
                            f"({self._dead_reason or 'closed'})"
                        )
                    # Keepalive batch during prompt processing (matches engine).
                    yield [GenerationResult(text="")]
                    continue

                ftype = frame.get("type")
                if ftype == "batch":
                    items = [GenerationResult.from_dict(d) for d in frame.get("items", [])]
                    if items:
                        yield items
                elif ftype == "final":
                    res = frame.get("result")
                    if res is not None:
                        yield [GenerationResult.from_dict(res)]
                elif ftype == "error":
                    err = frame.get("error", "unknown")
                    tb = frame.get("traceback", "")
                    if frame.get("engine_dead"):
                        # Death error frame injected by _handle_child_death —
                        # terminate the waiting generator with a clear,
                        # 503-mappable error (SSE paths emit an error event).
                        logger.error(
                            f"[ProcessProxy] rid={rid} aborted — {err}"
                        )
                        raise EngineRestartingError(
                            f"engine child worker died mid-generation: {err}"
                        )
                    logger.error(f"[ProcessProxy] child error rid={rid}: {err}\n{tb}")
                    raise RuntimeError(f"process-mode generation error: {err}")
                elif ftype == "done":
                    break
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnect — tell the child to cancel this request.
            if not cancel_sent:
                try:
                    with self._ctrl_send_lock:
                        self._ctrl_parent.send(proto.make_cancel(rid))
                    cancel_sent = True
                except Exception:  # noqa: BLE001
                    pass
            logger.info(
                f"[ProcessProxy] rid={rid} cancelled — sent cancel to child"
            )
            # Finding 1(b): do NOT return from this aclose until the child has
            # acked with its terminal ``done``/``error`` frame — the child
            # closes generate_stream (C1 teardown) BEFORE emitting it, so the
            # gate lease (released above this aclose by the response wrapper) is
            # not freed until the child turn is wound down. Bounded + fail-closed.
            await self._await_child_cancel_ack(rid, q)
            raise
        finally:
            with self._routing_lock:
                self._queues.pop(rid, None)
                self._loops.pop(rid, None)
            with self._inflight_lock:
                self._inflight -= 1

    async def _await_child_cancel_ack(self, rid, q) -> None:
        """After sending a cancel, wait for the child's terminal
        ``done``/``error`` frame on this request's queue (finding 1(b)).

        The child closes its ``generate_stream`` — running the C1 teardown —
        BEFORE emitting the terminal frame, so returning here means the child
        turn is genuinely wound down. Any in-flight ``batch``/``final`` frames
        are drained and ignored. Bounded + fail-closed: a dead child (liveness
        flags) or the timeout returns immediately so the caller's aclose (and
        the gate release above it) never hangs."""
        deadline = time.monotonic() + self.CANCEL_ACK_TIMEOUT_S
        while True:
            if self._dead or self._permanently_dead or self._closed:
                # Child is gone — no ack will ever come; the death sweep already
                # tore this request's routing down. Release fail-closed.
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"[ProcessProxy] rid={rid} cancel ack not received within "
                    f"{self.CANCEL_ACK_TIMEOUT_S}s — proceeding (child C1 may be "
                    f"incomplete)"
                )
                return
            try:
                frame = await asyncio.wait_for(q.get(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue  # re-check liveness + deadline; the child is winding down
            ftype = frame.get("type") if isinstance(frame, dict) else None
            if ftype in ("done", "error"):
                return
            # batch/final frames arriving after the cancel are ignored — we are
            # only waiting for the terminal ack.

    def is_busy(self) -> bool:
        with self._inflight_lock:
            return self._inflight > 0

    # --- non-streaming completion ----------------------------------------

    def complete(self, messages, **kwargs):
        """Non-streaming completion. Returns a CompletionResult (rehydrated
        from the child's serialized dict).

        ``response_format`` is a pydantic model on this side; serialize it to a
        plain dict for the wire (the worker rehydrates a duck-typed view, same
        as the streaming path) so we never pickle pydantic across the Pipe."""
        if "response_format" in kwargs:
            kwargs["response_format"] = proto.serialize_response_format(
                kwargs.get("response_format")
            )
        return self._rpc("complete", messages, **kwargs)

    # --- session lifecycle (synchronous RPCs) ----------------------------

    def update_session_messages(self, session_id, messages):
        # Real RPC now (Stage 1 was a no-op). The child's engine intentionally
        # ignores the caller messages (its internal thinking-bearing messages
        # are authoritative for cache matching) — it only touches + marks the
        # session dirty. Forwarding lets the child persist its KV cache.
        return self._rpc("update_session_messages", session_id, messages)

    def compact_session(self, session_id, messages):
        return self._rpc("compact_session", session_id, messages)

    def truncate_session(self, session_id, target_msg_count):
        return self._rpc("truncate_session", session_id, target_msg_count)

    def prepare_regenerate(self, session_id):
        return self._rpc("prepare_regenerate", session_id)

    def branch_from_turn(self, source_session_id, new_session_id, branch_turn,
                         branch_messages=None):
        return self._rpc(
            "branch_from_turn", source_session_id, new_session_id, branch_turn,
            branch_messages=branch_messages,
        )

    def get_session(self, session_id):
        # Read-only: bounded wait, EngineBusyError while generating (U14).
        return self._rpc_read("get_session", session_id)

    def delete_session(self, session_id):
        return self._rpc("delete_session", session_id)

    def list_sessions(self):
        # Read-only: bounded wait, EngineBusyError while generating (U14).
        return self._rpc_read("list_sessions")

    def session_stats(self):
        # Read-only: bounded wait, EngineBusyError while generating (U14).
        return self._rpc_read("session_stats")

    def base_cache_stats(self):
        # Read-only: bounded wait, EngineBusyError while generating (U14).
        return self._rpc_read("base_cache_stats")

    # --- tokenization + readiness ----------------------------------------

    def tokenize(self, content, add_special=False, with_pieces=False):
        """POST /tokenize in process mode: forward to the child via the
        generic RPC (the worker dispatches getattr(engine, "tokenize")). The
        tokenizer lives in the child, so this must round-trip. Read-only:
        bounded wait + EngineBusyError while the child is mid-generation (U14),
        EngineRestartingError (fail fast -> 503) when the child is dead."""
        return self._rpc_read(
            "tokenize", content,
            add_special=add_special, with_pieces=with_pieces,
        )

    def is_ready(self) -> bool:
        """GET /ready in process mode. Ready == the child reached its first
        `ready` frame and is not dead / permanently dead / closed / respawning.
        Reads ONLY proxy-side liveness state (no RPC, no GPU lock), so /ready
        stays non-blocking and independent of queue saturation."""
        with self._state_lock:
            return (
                self._ever_ready
                and not self._dead
                and not self._permanently_dead
                and not self._closed
                and not self._respawning
            )

    def active_request_count(self) -> int:
        """Best-effort in-flight streaming request count for GET /ready
        (Batch C adds a real queue). Non-blocking read of the inflight
        counter."""
        with self._inflight_lock:
            return self._inflight

    # --- admin cache overview / reset (synchronous RPCs) -----------------

    def cache_overview(self):
        # Read-only: bounded wait, EngineBusyError while generating (U14).
        return self._rpc_read("cache_overview")

    def clear_caches(self):
        return self._rpc("clear_caches")

    def reset(self):
        return self._rpc("reset")


def _deserialize_rpc_result(value):
    """Rehydrate the value produced by ``_serialize_rpc_result`` in the worker.

    Wrapped engine objects ({"__type__","value"}) become their dataclass; all
    other values pass through unchanged."""
    if isinstance(value, dict) and "__type__" in value and "value" in value:
        t = value["__type__"]
        payload = value["value"]
        if t == "CompletionResult":
            from mlx_soloheaven.engine.types import CompletionResult
            return CompletionResult.from_dict(payload)
        if t == "GenerationResult":
            return GenerationResult.from_dict(payload)
        return payload
    return value
