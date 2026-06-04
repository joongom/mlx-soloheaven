"""PARENT-side proxy for process-mode generation (Stage 1).

``EngineProcessProxy`` looks (enough) like an ``MLXEngine`` for the
``/v1/chat/completions`` streaming path: it exposes
``generate_stream_batches_async``, ``model_id``, ``model_family``,
``enable_thinking``, ``think_end_token``, ``cfg`` (for ``cfg.enable_thinking``),
and ``update_session_messages`` (no-op — the child owns sessions). All other
engine methods the OTHER endpoints call raise NotImplementedError so failures
are loud rather than silently wrong.

Architecture: one CHILD process spawned with the spawn context; three Pipes
(cmd / resp / ctrl). One daemon READER thread does ``resp_conn.recv()`` and
routes frames by request id into a per-request asyncio.Queue via
``loop.call_soon_threadsafe``.
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import uuid
from dataclasses import asdict

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


class _NotSupported:
    """Mixin providing loud stubs for engine methods not wired in stage 1."""

    @staticmethod
    def _nope(*_a, **_k):
        raise NotImplementedError(
            "not supported in process mode (stage 1)"
        )


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

        # Per-request routing. The reader thread routes frames into these
        # queues keyed by request id. Each queue is created on the request's
        # own event loop; routing uses call_soon_threadsafe.
        self._queues: dict[str, asyncio.Queue] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._routing_lock = threading.Lock()

        # Stage-1 single-flight tracking (loud-ish; not a hard gate).
        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self._ready_event = threading.Event()
        self._reader_thread = None

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

    def close(self):
        try:
            self._cmd_parent.send({"op": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None:
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()

    # --- reader thread ----------------------------------------------------

    def _reader_loop(self):
        """Drain resp pipe; route frames into per-request asyncio queues."""
        while True:
            try:
                frame = self._resp_parent.recv()
            except (EOFError, OSError):
                break
            if not isinstance(frame, dict):
                continue

            ftype = frame.get("type")
            if ftype == "ready":
                self.model_id = frame.get("model_id", "")
                self.model_family = frame.get("model_family", "chatml")
                self.enable_thinking = bool(frame.get("enable_thinking", True))
                self.think_end_token = int(frame.get("think_end_token", -1))
                # Keep cfg.enable_thinking consistent (openai_compat reads it).
                try:
                    self.cfg.enable_thinking = self.enable_thinking
                    self.cfg.think_end_token = self.think_end_token
                except Exception:  # noqa: BLE001
                    pass
                self._ready_event.set()
                continue

            rid = frame.get("id")
            if rid is None:
                continue
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

    # --- generation -------------------------------------------------------

    async def generate_stream_batches_async(self, messages, **params):
        """Send a generate command; yield list[GenerationResult] batches.

        Mirrors MLXEngine.generate_stream_batches_async's contract: each
        yielded item is a list of GenerationResult. The terminal finish
        result arrives as a `final` frame and is re-yielded as a 1-item batch
        so the openai_compat consumer sees `result.finish_reason`.
        """
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

        cancel_sent = False
        try:
            self._cmd_parent.send(proto.make_generate(rid, messages, wire_params))

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

    # --- session/admin methods used by OTHER endpoints (loud stubs) -------

    def update_session_messages(self, *_a, **_k):
        # No-op: the child owns session state. The parent has no session to
        # touch; the child already persists on its own. Keeping this a no-op
        # (rather than raising) lets the /v1/chat/completions handler finish
        # cleanly without special-casing process mode.
        return None

    def session_stats(self) -> dict:
        return {"active_sessions": 0, "sessions": {}, "mode": "process"}

    def complete(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def compact_session(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def list_sessions(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def get_session(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def delete_session(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def base_cache_stats(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def summarize(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def compact(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def generate_summary_stream(self, *a, **k):
        _NotSupported._nope(*a, **k)

    def generate_stream_async(self, *a, **k):
        _NotSupported._nope(*a, **k)
