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
import uuid
from dataclasses import asdict
from types import SimpleNamespace

from mlx_soloheaven.engine import process_protocol as proto
from mlx_soloheaven.engine.types import GenerationResult

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
        self._cmd_parent, self._cmd_child = self._ctx.Pipe()
        self._resp_parent, self._resp_child = self._ctx.Pipe()
        self._ctrl_parent, self._ctrl_child = self._ctx.Pipe()

        self._proc = None

        # Model metadata, populated from the child's `ready` frame.
        self.model_id = ""
        self.model_family = "chatml"
        self.enable_thinking = bool(getattr(cfg, "enable_thinking", True))
        self.think_end_token = int(getattr(cfg, "think_end_token", -1))

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

    # --- lifecycle --------------------------------------------------------

    def start(self, ready_timeout: float = 600.0):
        """Spawn the child, start the reader thread, block until `ready`."""
        from mlx_soloheaven.engine.process_worker import worker_main

        self._proc = self._ctx.Process(
            target=worker_main,
            args=(_config_to_dict(self.cfg), self._cmd_child,
                  self._resp_child, self._ctrl_child),
            name="mlx-engine-child",
            daemon=True,
        )
        self._proc.start()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="proc-reader"
        )
        self._reader_thread.start()

        if not self._ready_event.wait(timeout=ready_timeout):
            raise RuntimeError(
                "process-mode child did not become ready within "
                f"{ready_timeout}s (model load failed?)"
            )
        logger.info(
            f"[ProcessProxy] child ready: model_id={self.model_id} "
            f"family={self.model_family} thinking={self.enable_thinking}"
        )

    def close(self, join_timeout: float = 10.0):
        """Graceful child stop. Idempotent.

        Sends the 'shutdown' op and WAITS (bounded) so the child's main loop
        can flush dirty sessions to disk before exiting — measured DISK SAVE
        cost is ~0.01s per session, so seconds-scale is generous. A wedged
        child (e.g. mid-generation past the timeout) is terminate()d; its
        SIGTERM handler still attempts the same graceful flush on the way
        out, so we join briefly once more to let that finish."""
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

    def _reader_loop(self):
        """Drain resp pipe; route streaming frames into per-request asyncio
        queues and RPC results into their threading-future slots."""
        while True:
            try:
                frame = self._resp_parent.recv()
            except (EOFError, OSError):
                break
            if not isinstance(frame, dict):
                continue

            ftype = frame.get("type")
            if ftype == "ready":
                self._apply_ready(frame)
                continue

            rid = frame.get("id")
            if rid is None:
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

    def _apply_ready(self, frame: dict):
        self.model_id = frame.get("model_id", "")
        self.model_family = frame.get("model_family", "chatml")
        self.enable_thinking = bool(frame.get("enable_thinking", True))
        self.think_end_token = int(frame.get("think_end_token", -1))
        # Refresh cfg view from the child's authoritative post-load snapshot so
        # the API's default_*/thinking/token-id reads match the child.
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
        self._ready_event.set()

    # --- generic synchronous RPC -----------------------------------------

    def _rpc(self, method: str, *args, timeout: float = 600.0, **kwargs):
        """Forward a synchronous engine-method call to the child and block on
        the result. Returns the deserialized result; raises _RpcError on a
        child-side exception."""
        rid = uuid.uuid4().hex
        ev = threading.Event()
        with self._rpc_lock:
            self._rpc_results[rid] = {"event": ev, "frame": None}
        try:
            with self._cmd_send_lock:
                self._cmd_parent.send(proto.make_rpc(rid, method, list(args), kwargs))
            if not ev.wait(timeout=timeout):
                raise _RpcError(
                    f"process-mode RPC {method!r} timed out after {timeout}s"
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
            logger.error(f"[ProcessProxy] RPC {method!r} error: {err}\n{tb}")
            raise _RpcError(f"process-mode RPC {method!r} error: {err}")
        return _deserialize_rpc_result(frame.get("result"))

    # --- generation (streaming) ------------------------------------------

    async def generate_stream_batches_async(self, messages, **params):
        """Send a generate command; yield list[GenerationResult] batches."""
        async for batch in self._stream(messages, params, scalar=False):
            yield batch

    async def generate_stream_async(self, messages, **params):
        """Send a scalar generate command; yield individual GenerationResult.

        Mirrors MLXEngine.generate_stream_async — used by the compaction summary
        streamer (CompactionEngine.generate_summary_stream)."""
        async for batch in self._stream(messages, params, scalar=True):
            for item in batch:
                yield item

    async def _stream(self, messages, params, scalar: bool):
        """Shared streaming driver for batched + scalar paths. Yields
        list[GenerationResult] batches (scalar path yields 1-item batches)."""
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
        }

        make_cmd = proto.make_generate_scalar if scalar else proto.make_generate

        cancel_sent = False
        try:
            with self._cmd_send_lock:
                self._cmd_parent.send(make_cmd(rid, messages, wire_params))

            while True:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
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
            raise
        finally:
            with self._routing_lock:
                self._queues.pop(rid, None)
                self._loops.pop(rid, None)
            with self._inflight_lock:
                self._inflight -= 1

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
        return self._rpc("get_session", session_id)

    def delete_session(self, session_id):
        return self._rpc("delete_session", session_id)

    def list_sessions(self):
        return self._rpc("list_sessions")

    def session_stats(self):
        return self._rpc("session_stats")

    def base_cache_stats(self):
        return self._rpc("base_cache_stats")

    # --- admin cache overview / reset (synchronous RPCs) -----------------

    def cache_overview(self):
        return self._rpc("cache_overview")

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
