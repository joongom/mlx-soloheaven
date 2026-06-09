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
import threading
import time
import traceback

from mlx_soloheaven.engine import process_protocol as proto


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

    # Build + load the engine on THIS (main) thread.
    engine = MLXEngine(cfg, execution_mode="main_thread")
    engine.load_model()

    # Per-request cancel registry: request_id -> threading.Event.
    # Mutated by the main loop (add/remove) and read by the ctrl thread.
    active: dict[str, threading.Event] = {}
    active_lock = threading.Lock()

    def _ctrl_loop():
        # Daemon thread: block on cancel frames, set the matching event.
        # Never touches MLX — only flips a Python threading.Event.
        while True:
            try:
                frame = ctrl_conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(frame, dict):
                continue
            if frame.get("op") == "cancel":
                rid = frame.get("id")
                with active_lock:
                    ev = active.get(rid)
                if ev is not None:
                    ev.set()

    threading.Thread(target=_ctrl_loop, daemon=True, name="proc-ctrl").start()

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

    while True:
        try:
            cmd = cmd_conn.recv()
        except (EOFError, OSError):
            break
        if cmd is None:
            break
        if not isinstance(cmd, dict):
            continue
        op = cmd.get("op")
        if op == "shutdown":
            break

        # --- generic synchronous RPC (Stage 2) ---
        if op == "rpc":
            rid = cmd.get("id", "")
            method = cmd.get("method", "")
            args = cmd.get("args", []) or []
            kwargs = cmd.get("kwargs", {}) or {}
            try:
                _run_rpc(engine, rid, method, args, kwargs, resp_conn)
            except Exception as e:  # noqa: BLE001
                resp_conn.send(proto.make_error(rid, str(e), traceback.format_exc()))
            continue

        if op not in ("generate", "generate_scalar"):
            # Unknown op — report loudly against its id if present.
            resp_conn.send(proto.make_error(
                cmd.get("id", ""), f"unknown op {op!r}", ""
            ))
            continue

        rid = cmd["id"]
        payload = cmd.get("payload", {})
        messages = payload.get("messages", [])
        params = dict(payload.get("params", {}))
        scalar = (op == "generate_scalar")

        cancel_event = threading.Event()
        with active_lock:
            active[rid] = cancel_event

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
