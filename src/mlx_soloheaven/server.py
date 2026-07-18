"""FastAPI application factory and server entry point."""

import logging
import os
import socket
import sys
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mlx_soloheaven.config import Config

if TYPE_CHECKING:
    # Type-only: never imported at runtime so the parent process stays
    # mlx-free in `--engine-mode process`.
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("soloheaven")


def engine_unavailable_response(exc: Exception):
    """503 engine_not_ready body for a dead/respawning process-mode engine.

    Module-level (and dependency-light) so it is hermetically testable and
    reusable by any router. The SSE paths can't use this (headers already
    sent) — they emit an in-band error frame instead (see openai_compat/chat).

    Batch B: the canonical {message, type, code} envelope, NO ``detail:
    str(exc)`` (the exception is logged by the handler, never returned in the
    body), and the ``engine_not_ready`` vocabulary (renamed from the old
    ``engine_restarting``). ``exc`` is accepted for signature compatibility
    with existing callers/tests but is deliberately not echoed."""
    from mlx_soloheaven.api.errors import engine_not_ready_response

    return engine_not_ready_response()


def register_engine_error_handlers(app: FastAPI):
    """Map EngineRestartingError / EngineBusyError -> the Batch B HTTP status
    contract for all non-streaming endpoints.

    The process-mode proxy fails FAST with EngineRestartingError while its
    child worker is dead or respawning (Metal GPU aborts kill the child
    uncatchably; model reload takes ~1-3 min). process_client is mlx-free, so
    importing it here keeps the parent clean under --engine-mode process.

    EngineBusyError (codex batch-3, round 2): most read paths already catch
    it locally and answer a "busy" placeholder; this app-level handler is
    the backstop for the paths that don't — notably run_long/run_read
    submissions rejected by the executors module's admission/shutdown gate.

    Batch B status contract (downstream-confirmed):
      - EngineRestartingError (dead/respawning/reloading/shutdown child)
        -> 503 engine_not_ready (+ Retry-After 30).
      - EngineBusyError with reason=QUEUE_FULL (admission/pool saturation,
        read-lock/read-RPC busy) -> 429 queue_full (+ Retry-After 1).
      - EngineBusyError with reason=ENGINE_NOT_READY (admission stopped /
        shutting down / pools torn down) -> 503 engine_not_ready.
    NO ``detail: str(exc)`` in any body; the exception is logged only."""
    from mlx_soloheaven.api.errors import (
        engine_not_ready_response,
        queue_full_response,
    )
    from mlx_soloheaven.engine.process_client import EngineRestartingError
    from mlx_soloheaven.engine.types import EngineBusyError

    @app.exception_handler(EngineRestartingError)
    async def _engine_restarting_handler(request, exc):
        logger.warning(
            f"[503] {request.method} {request.url.path} — engine unavailable: {exc}"
        )
        return engine_not_ready_response()

    @app.exception_handler(EngineBusyError)
    async def _engine_busy_handler(request, exc):
        reason = getattr(exc, "reason", EngineBusyError.REASON_ENGINE_NOT_READY)
        if reason == EngineBusyError.REASON_QUEUE_FULL:
            logger.warning(
                f"[429] {request.method} {request.url.path} — queue full: {exc}"
            )
            return queue_full_response()
        logger.warning(
            f"[503] {request.method} {request.url.path} — engine not ready: {exc}"
        )
        return engine_not_ready_response()


_KNOWN_LOC_FIELD_NAMES: frozenset[str] | None = None


def _known_loc_field_names() -> frozenset[str]:
    """The set of loc string components that are SAFE to log verbatim: the
    request-location markers FastAPI prepends ("body"/"query"/...) plus every
    field NAME/alias declared on our request pydantic models.

    Batch D log-leak scrub (finding 2): a pydantic ``loc`` tuple can contain
    ATTACKER-CONTROLLED dict KEYS — e.g. ``MessageToolCall.function`` is
    ``dict[str, str]`` (schemas.py), so a body ``{"function":{"SUPERSECRET":…}}``
    puts ``SUPERSECRET`` into loc. Any loc string NOT in this known set is an
    arbitrary user key and gets redacted before logging. Derived from the models
    themselves so it can never drift out of date. Cached (the model set is static
    once imported).

    Finding 5 (Batch D round 2): the set is scanned from EVERY api module's
    pydantic models, not just ``api.schemas`` — the request bodies for
    chat/tokenize/compaction/settings (``CompactRequest``, ``TokenizeRequest``,
    ``SessionSettings``, ``CreateSessionRequest``, …) live in their own modules,
    so a schemas-only scan OVER-redacted their legitimate fields (``system_prompt``,
    ``add_special``, ``with_pieces``, ``keep_recent_turns``,
    ``context_window_limit``, …). Scanning the whole api package keeps those
    field names diagnostic while still redacting genuinely-unknown string
    components (arbitrary user dict keys)."""
    global _KNOWN_LOC_FIELD_NAMES
    if _KNOWN_LOC_FIELD_NAMES is not None:
        return _KNOWN_LOC_FIELD_NAMES
    import importlib
    import pkgutil

    from pydantic import BaseModel

    # Location markers FastAPI/pydantic prepend, plus pydantic's root marker.
    names: set[str] = {"body", "query", "path", "header", "cookie", "__root__"}
    # Nit (Batch D round 3): DISCOVER every module in the api package GENERICALLY
    # (pkgutil) rather than a hardcoded module list — so a NEW api module that
    # declares a request model has its field names covered automatically and can
    # never drift into over-redaction of its legitimate fields. This is safe and
    # cheap: all api modules are mlx-free and already imported at server startup,
    # so each import here is a sys.modules cache hit; discovery / import failures
    # are ignored (they just yield a smaller allowlist — fail-safe, never raises).
    # Memoized (computed once) via ``_KNOWN_LOC_FIELD_NAMES``.
    try:
        import mlx_soloheaven.api as api_pkg

        module_names = [
            info.name
            for info in pkgutil.iter_modules(
                api_pkg.__path__, api_pkg.__name__ + "."
            )
        ]
    except Exception:  # noqa: BLE001 — a discovery failure must never break scrub.
        module_names = []
    for mod_name in module_names:
        try:
            module = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 — a missing/optional module must never
            # break loc sanitization (it just yields a smaller allowlist).
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                for fname, field in obj.model_fields.items():
                    names.add(fname)
                    alias = getattr(field, "alias", None)
                    if alias:
                        names.add(alias)
    _KNOWN_LOC_FIELD_NAMES = frozenset(names)
    return _KNOWN_LOC_FIELD_NAMES


def _sanitize_loc(loc) -> list:
    """Redact user-controlled components of a pydantic error ``loc`` for logging.

    Keeps int indices (array positions) and KNOWN schema field-name/alias
    components (which field failed — the diagnostic value); replaces every other
    string component (an arbitrary dict key a caller can name freely) with a
    ``"<redacted>"`` marker so a prompt/secret smuggled as a JSON key can never
    reach the operational log (and thus the admin log SSE). Never raises.

    Finding 5 (Batch D round 2): ``loc`` is NORMALIZED to an iterable first — the
    "never raises" contract was false for a non-iterable ``loc`` (a bare int/str
    reached ``for comp in loc`` and a non-string/non-int raised ``TypeError``).
    A non-iterable (or a bare string, which would otherwise iterate per-char) is
    wrapped into a single-element list so it is sanitized as one component.

    Nit (Batch D round 3): the normalization must NOT gate on truthiness — a bare
    ``loc == 0`` (a legitimate index-0 array position) is falsy, so the old
    ``if not loc: return []`` dropped it entirely. Normalize by TYPE instead so
    ``_sanitize_loc(0)`` -> ``[0]`` (int indices, including 0, survive) while
    ``None`` / an unexpected object is wrapped and safely redacted, and an empty
    list/tuple still yields ``[]`` (the loop produces nothing)."""
    # Normalize a non-iterable (or a bare str, which would iterate per-CHARACTER)
    # into a one-element sequence so a component is treated as a whole. Gate on
    # TYPE, never truthiness, so a falsy-but-valid index (0) is preserved.
    if isinstance(loc, (str, bytes)) or not isinstance(loc, (list, tuple)):
        loc = [loc]
    known = _known_loc_field_names()
    out: list = []
    for comp in loc:
        if isinstance(comp, bool):
            # bool is an int subclass but never a valid loc index — redact.
            out.append("<redacted>")
        elif isinstance(comp, int):
            out.append(comp)
        elif isinstance(comp, str) and comp in known:
            out.append(comp)
        else:
            out.append("<redacted>")
    return out


def register_validation_error_handler(app: FastAPI):
    """Normalize FastAPI's default 422 body to the canonical envelope (Batch B).

    FastAPI's built-in RequestValidationError handler returns
    ``{"detail": [<per-field errors>]}`` — a different shape from every other
    error path AND one that echoes field-level input back to the caller. This
    replaces it with the canonical ``{error:{message,type,code}}`` envelope
    (type invalid_request_error, code invalid_request) and a GENERIC message
    that does NOT echo the request body or ``exc.errors()``. Batch D log-leak
    scrub: the server-side LOG now records only each error's ``type`` + ``loc``
    (never the raw body nor pydantic's input-echoing ``input``/``ctx`` fields),
    so a prompt/secret in an invalid request cannot leak into the operational
    log (or the admin log SSE fed from it). Module-level and dependency-light so
    it is hermetically testable."""
    from fastapi.exceptions import RequestValidationError

    from mlx_soloheaven.api.errors import (
        CODE_INVALID_REQUEST,
        TYPE_INVALID_REQUEST,
        error_response,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc):
        # Batch D log-leak scrub: NEVER log the raw request body — it can carry
        # prompts / PII / secrets, and every log record is broadcast to the admin
        # log SSE. pydantic's ``exc.errors()`` ALSO echoes the offending input
        # (the ``input`` field, and ``ctx`` can embed values), so log only the
        # ``type`` + a SANITIZED ``loc`` of each error — enough to debug which
        # field was invalid, with no input values. Finding 2: ``loc`` itself can
        # embed attacker-controlled dict KEYS (``MessageToolCall.function`` is
        # ``dict[str, str]``), so ``_sanitize_loc`` redacts any component that is
        # not an int index or a known schema field name.
        try:
            safe_errors = [
                {"type": e.get("type"), "loc": _sanitize_loc(e.get("loc"))}
                for e in exc.errors()
            ]
        except Exception:  # noqa: BLE001 — never let logging break the handler
            safe_errors = []
        logger.error(
            f"[422] {request.method} {request.url.path} | "
            f"errors={safe_errors}"
        )
        return error_response(
            422,
            "request validation failed: one or more fields are invalid",
            TYPE_INVALID_REQUEST,
            CODE_INVALID_REQUEST,
        )


def register_uncaught_error_handler(app: FastAPI):
    """Ensure an UNCAUGHT exception yields the canonical 500 envelope AND carries
    ``X-Request-ID`` (Batch D, finding 6).

    Starlette's outermost ``ServerErrorMiddleware`` generates a 500 for any
    exception that escapes the user middleware — OUTSIDE ``RequestIDMiddleware``'s
    send-wrapper — so an uncaught 500 would otherwise ship WITHOUT the correlation
    header. Registering an ``Exception`` handler routes those 500s through here:
    it logs the exception (never echoing it in the body — Batch B), returns the
    canonical ``server_error`` envelope, and stamps ``X-Request-ID`` read from the
    ASGI scope (the ContextVar is already reset by the time this outer handler
    runs, but the middleware also stashed the id on the shared scope)."""
    from mlx_soloheaven.api.errors import server_error_response
    from mlx_soloheaven.api.request_id import request_id_from_scope

    @app.exception_handler(Exception)
    async def _uncaught_exception_handler(request, exc):
        logger.exception(
            f"[500] {request.method} {request.url.path} — uncaught exception"
        )
        response = server_error_response()
        rid = request_id_from_scope(request.scope)
        if rid:
            response.headers["X-Request-ID"] = rid
        return response


def build_per_model_config(cfg: Config, mcfg, *, engine_mode: str | None = None) -> Config:
    """Per-model Config: the model's own settings + every GLOBAL server flag
    an engine reads off its cfg.

    U27: the process-mode and in-process multi-model builders used to be two
    hand-maintained copies of this kwargs list, and the in-process (--models)
    copy dropped ``stream_coalesce_n``/``stream_coalesce_ms`` — the
    --stream-coalesce-n/ms flags silently reset to the dataclass defaults
    (4/30) for every per-model engine. One shared, module-level (hermetically
    testable) builder makes that class of drift impossible. ``engine_mode``
    is the only caller-divergent field (process mode pins "process"; the
    in-process path keeps the dataclass default, matching its historical
    construction)."""
    kwargs = dict(
        model_path=mcfg.model_path,
        host=cfg.host,
        port=cfg.port,
        default_temperature=mcfg.default_temperature,
        default_top_p=mcfg.default_top_p,
        default_min_p=mcfg.default_min_p,
        default_top_k=mcfg.default_top_k,
        # Carry the CLI-pinned marker so each per-model engine knows which
        # sampling fields must NOT be overridden by generation_config.json.
        cli_set_sampling=mcfg.cli_set_sampling,
        default_repetition_penalty=mcfg.default_repetition_penalty,
        default_max_tokens=mcfg.default_max_tokens,
        thinking_budget=mcfg.thinking_budget,
        enable_thinking=mcfg.enable_thinking,
        memory_budget_gb=cfg.memory_budget_gb,
        disk_budget_gb=cfg.disk_budget_gb,
        # Carry the MLX Metal buffer-reuse pool cap so each engine honours
        # --mlx-cache-limit-gb / SOLOHEAVEN_MLX_CACHE_LIMIT_GB instead of
        # silently falling back to the dataclass default.
        mlx_cache_limit_gb=cfg.mlx_cache_limit_gb,
        data_dir=cfg.data_dir,
        verbose=cfg.verbose,
        gpu_keepalive=cfg.gpu_keepalive,
        # Carry over tuning flags from top-level cfg
        kv_bits=cfg.kv_bits,
        kv_group_size=cfg.kv_group_size,
        quantized_kv_start=cfg.quantized_kv_start,
        prefill_step_size=cfg.prefill_step_size,
        pld_enabled=cfg.pld_enabled,
        pld_num_draft_tokens=cfg.pld_num_draft_tokens,
        pld_ngram_k=cfg.pld_ngram_k,
        # Speculative-decoding drafter (MTP / DFlash) — must be propagated to
        # the per-model Config or `MLXEngine.load_model` silently skips
        # drafter wiring. Without this, `--draft-model` at the CLI never
        # reaches the engine and MTP is a no-op.
        draft_model=cfg.draft_model,
        draft_kind=cfg.draft_kind,
        draft_block_size=cfg.draft_block_size,
        # U27: stream-coalescing knobs — the engine reads them off ITS OWN
        # cfg in _run's post-loop (and the process worker off its snapshot).
        stream_coalesce_n=cfg.stream_coalesce_n,
        stream_coalesce_ms=cfg.stream_coalesce_ms,
        # Carry the backend selection so each per-model engine's load_model
        # gate routes mlx-lm vs mlx-vlm correctly (else it defaults to "auto").
        backend=mcfg.backend,
    )
    if engine_mode is not None:
        kwargs["engine_mode"] = engine_mode
    return Config(**kwargs)


def create_app(cfg: Config) -> FastAPI:
    """Build the FastAPI application with all routes and middleware."""
    # NOTE: do NOT import mlx_engine (or anything pulling in mlx.core/mlx_vlm)
    # at parent-process scope. In `--engine-mode process`, MLX must initialize
    # only in the child. The in-process path imports MLXEngine lazily, inside
    # its construction branch below.
    from mlx_soloheaven.storage import database as db
    from mlx_soloheaven.api import (
        openai_compat, chat, admin, settings, compaction, tokenize, metrics,
    )
    from mlx_soloheaven.api.request_id import RequestIDMiddleware
    from mlx_soloheaven import inference_queue

    # Batch C: wire the configured bound into the parent-side FIFO inference
    # gate that fronts all GPU generation (streaming + non-streaming completions,
    # web-chat generation). mlx-free — the parent stays mlx-free in process mode.
    inference_queue.configure(cfg.max_queue_length)

    # Set engine logger level based on verbose flag
    engine_logger = logging.getLogger("mlx_soloheaven.engine.mlx_engine")
    if not cfg.verbose:
        engine_logger.setLevel(logging.INFO)  # verbose=False: DEBUG hidden (default)
    else:
        engine_logger.setLevel(logging.DEBUG)  # verbose=True: show all

    app = FastAPI(
        title="MLX SoloHeaven",
        description="Single-user LLM inference server with KV cache optimization",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Batch D (observability): assign/propagate X-Request-ID and expose it via a
    # ContextVar for handler/log correlation. Pure-ASGI, mlx-free — no per-request
    # task overhead, and it stamps the header on JSON and streaming responses
    # alike.
    app.add_middleware(RequestIDMiddleware)

    # Batch B: normalize FastAPI's default 422 body to the canonical envelope
    # (no request-body / field-error echo in the response; logged only).
    register_validation_error_handler(app)

    # Dead/respawning process-mode child -> 503 with a JSON error body.
    register_engine_error_handlers(app)

    # Batch D (finding 6): an UNCAUGHT exception -> canonical 500 envelope that
    # still carries X-Request-ID (ServerErrorMiddleware generates 500s outside
    # the request-id send-wrapper).
    register_uncaught_error_handler(app)

    # Build per-model configs and engines. Annotated as the union of the
    # in-process engine and its process-mode proxy; kept as a string so this
    # module never imports MLXEngine at runtime.
    engines: "dict[str, MLXEngine]" = {}

    # Opt-in process mode (Stage 1): run a SINGLE model's generation in a
    # separate child process on its main thread. Requires --model (single),
    # NOT --models. Falls back to inprocess otherwise (with a warning).
    use_process_mode = getattr(cfg, "engine_mode", "inprocess") == "process"
    if use_process_mode and cfg.models and len(cfg.models) == 1:
        from mlx_soloheaven.engine.process_client import EngineProcessProxy

        mcfg = cfg.models[0]
        model_cfg = build_per_model_config(cfg, mcfg, engine_mode="process")
        engines[mcfg.model_id] = EngineProcessProxy(model_cfg)  # type: ignore[assignment]
        logger.info(
            f"[engine-mode] PROCESS mode enabled for single model "
            f"{mcfg.model_id} (separate child process, main-thread generation)"
        )
    elif use_process_mode:
        logger.warning(
            "[engine-mode] process mode requested but requires a single "
            "--model (got --models or none) — falling back to inprocess."
        )
        use_process_mode = False

    if engines:
        pass  # process-mode proxy already built above
    elif cfg.models:
        # In-process path: import MLXEngine lazily HERE (only reached when NOT
        # in process mode) so the parent stays mlx-free under --engine-mode
        # process.
        from mlx_soloheaven.engine.mlx_engine import MLXEngine
        for mcfg in cfg.models:
            # Create a Config per model with shared server settings (U27: one
            # shared builder — global flags can no longer be dropped from one
            # of the two hand-maintained copies).
            model_cfg = build_per_model_config(cfg, mcfg)
            engine = MLXEngine(model_cfg)
            engines[mcfg.model_id] = engine
    else:
        # In-process single-engine path — same lazy import rationale as above.
        from mlx_soloheaven.engine.mlx_engine import MLXEngine
        engine = MLXEngine(cfg)
        engines["default"] = engine

    # Default engine = first loaded
    default_engine: "MLXEngine" = None  # type: ignore

    @app.on_event("startup")
    async def startup():
        nonlocal default_engine

        db.set_db_path(cfg.db_path)
        await db.init_db()
        logger.info(f"Database initialized: {cfg.db_path}")

        for model_id, engine in engines.items():
            # Process-mode proxy: spawn child + block until ready (it has
            # .start() and no .load_model()). In-process engine: load weights
            # on this thread.
            if hasattr(engine, "load_model"):
                engine.load_model()
            else:
                engine.start()
            logger.info(f"Model ready: {model_id} -> {engine.model_id}")

        default_engine = list(engines.values())[0]

        # Set engine registry for API routers
        openai_compat.set_engines(engines, default_engine)
        chat.set_engines(engines, default_engine)
        admin.set_engines(engines, default_engine)
        tokenize.set_engines(engines, default_engine)
        compaction.set_engine(default_engine)
        admin.install_log_handler()

        logger.info(f"Server ready on http://{cfg.host}:{cfg.port}")
        logger.info(f"  Web UI:     http://{cfg.host}:{cfg.port}/")
        logger.info(f"  Admin:      http://{cfg.host}:{cfg.port}/admin")
        logger.info(f"  OpenAI API: http://{cfg.host}:{cfg.port}/v1/chat/completions")
        logger.info(f"  Settings:   http://{cfg.host}:{cfg.port}/api/sessions/{{id}}/settings")
        logger.info(f"  Compaction: http://{cfg.host}:{cfg.port}/api/sessions/{{id}}/compact")
        logger.info(f"  Models:     {list(engines.keys())}")
        logger.info(
            f"  Cache budget: {cfg.memory_budget_gb}GB memory, {cfg.disk_budget_gb}GB disk"
        )

    @app.on_event("shutdown")
    async def shutdown():
        """Persist session state on a normal server stop (uvicorn lifespan
        shutdown — runs on SIGTERM/SIGINT graceful stop).

        OWNERSHIP CONTRACT: Uvicorn owns process lifetime in server mode.
        The engine-level SIGINT/SIGTERM handler installed by
        MLXEngine._register_shutdown_flush only PREPENDS a flush and then
        CHAINS to the previously-installed (Uvicorn) handler — so Uvicorn's
        graceful shutdown still runs and THIS hook is reliably reached for
        in-process/main_thread engines too. The flush here is then an
        idempotent no-op if the chained handler already drained the dirty
        set, or a real RETRY if that earlier flush timed out on the engine
        lock and re-marked the ids dirty.

        Process mode (production default): proxy.close() sends the 'shutdown'
        op and waits (bounded) while the CHILD flushes dirty sessions to disk
        before exiting. Without this hook nothing ever told the child to stop
        gracefully — the daemon child was simply terminated at parent exit and
        process mode never persisted anything on a normal stop.

        In-process mode: flush directly here while the engine's executor is
        still alive; the engine's own atexit/SIGTERM registration remains the
        backstop (the flush re-reads the live dirty set on every call).

        EXECUTOR LIFECYCLE (codex batch-3 review, rounds 2+3). Ordering:

        1. executor quiesce — admission stops (new submissions fail fast,
           503-style), queued not-yet-started work is CANCELLED, and
           running calls get a bounded grace, all atomically against new
           admissions (round 3, finding 1).
        2. the engine-side mutation gate closes on every IN-PROCESS engine
           (round 3, finding 2). The quiesce's bounded wait cannot stop a
           worker that already STARTED but is BLOCKED on the engine lock
           behind a generation; such a straggler would otherwise acquire
           the lock AFTER the flush below, mutate session state, mark it
           dirty — and nothing would flush again. With the gate set, every
           post-flush mutating lock acquisition aborts with EngineBusyError
           WITHOUT mutating. The gate is set for ALL engines BEFORE the
           first flush (the flush is classmethod-wide). The flush itself is
           exempt from the gate (it must run); the process-mode proxy has
           no gate (the child's serial main loop + its own shutdown flush
           make the ordering safe child-side). begin_shutdown also fires
           the engine's cooperative cancel event, aborting an in-flight
           compaction/rebuild prefill between chunks (round 5, finding 3a).
        3. bounded drain of ALREADY-ENTERED mutations (round 5, finding
           3a): an op inside its _mutate_locked section when the gate
           closed can outlive quiesce and time out the flush's bounded
           lock acquire, then publish + mark dirty AFTER the final flush.
           wait_mutations_settled() waits (bounded) for in-flight
           mutations to drain; any straggler that remains is covered
           deterministically by the engine-side self-flush-on-exit — its
           _mutate_locked exit path flushes its own dirty sessions before
           releasing the lock, so nothing dirty survives un-persisted.
        4. the engine flush / proxy close.
        5. backstop pool teardown LAST.

        Every step is idempotent (the executors module's atexit backstop
        may re-run the same sequence).

        Fail-closed: a failed flush logs and continues — it must never block
        or corrupt server shutdown.
        """
        from mlx_soloheaven import executors, inference_queue

        # Batch C: close the parent-side FIFO inference gate FIRST — reject
        # every NEW generation acquire (503 engine_not_ready) and fail every
        # QUEUED waiter the same way, so no queued generation hangs or slips
        # into run_long/the engine behind the shutdown flush. The RUNNING
        # generation is left to finish (its slot release wakes nothing new).
        # Pure asyncio (no threads/pools), so it cannot deadlock with the
        # executors quiesce ordering below.
        try:
            inference_queue.shutdown()
        except Exception:  # noqa: BLE001 — never block server shutdown
            logger.exception("[Shutdown] inference-queue close failed (continuing)")

        try:
            drained = executors.quiesce()
            if (
                drained.get("cancelled")
                or drained.get("poisoned_critical")
                or drained.get("still_running")
            ):
                logger.info(f"[Shutdown] executors quiesced: {drained}")
        except Exception:  # noqa: BLE001 — never block server shutdown
            logger.exception("[Shutdown] executor quiesce failed (continuing)")

        # Close the engine-side mutation gate on EVERY in-process engine
        # BEFORE the first flush (see the ordering rationale above).
        for model_id, engine in engines.items():
            try:
                if hasattr(engine, "begin_shutdown"):
                    engine.begin_shutdown()
            except Exception:  # noqa: BLE001 — never block server shutdown
                logger.exception(
                    f"[Shutdown] engine {model_id} shutdown gate failed "
                    f"(continuing)"
                )

        # Bounded drain of already-entered mutations BEFORE the flush
        # (round 5, finding 3a). Stragglers past the bound self-flush on
        # exit — see the ordering rationale above. (Process-mode proxies
        # have no such method: the child's serial loop cannot interleave a
        # mutation with its own shutdown flush.)
        for model_id, engine in engines.items():
            try:
                if hasattr(engine, "wait_mutations_settled"):
                    remaining = engine.wait_mutations_settled(5.0)
                    if remaining:
                        logger.warning(
                            f"[Shutdown] engine {model_id}: {remaining} "
                            f"mutation(s) still in flight after bounded "
                            f"drain — relying on self-flush-on-exit"
                        )
            except Exception:  # noqa: BLE001 — never block server shutdown
                logger.exception(
                    f"[Shutdown] engine {model_id} mutation drain failed "
                    f"(continuing)"
                )

        for model_id, engine in engines.items():
            try:
                if hasattr(engine, "load_model"):
                    # In-process MLXEngine (classmethod via the instance's
                    # type — keeps this module mlx-import-free at top level).
                    type(engine)._flush_all_on_shutdown()
                else:
                    # Process-mode proxy: graceful child stop (child flushes).
                    engine.close()
                logger.info(f"[Shutdown] engine {model_id} stopped cleanly")
            except Exception:  # noqa: BLE001 — never block server shutdown
                logger.exception(
                    f"[Shutdown] engine {model_id} flush failed (continuing)"
                )

        # Backstop pool teardown AFTER the flush (idempotent).
        try:
            executors.shutdown_pools()
        except Exception:  # noqa: BLE001 — never block server shutdown
            logger.exception("[Shutdown] executor pool teardown failed")

    @app.get("/health")
    async def health():
        # U14: session_stats is a synchronous engine RPC (process mode: the
        # child serves it only between generations; in-process: bounded read
        # lock). Run it off the event loop and degrade to a "busy"/"error"
        # marker so /health ALWAYS answers — even mid-generation or with a
        # dead child. F2: on the RESERVED reads executor (not to_thread's
        # shared default pool), so the bounded read always gets a worker
        # even when concurrent generations/mutating RPCs occupy every
        # long-ops worker — otherwise the 10s bound never even starts.
        from mlx_soloheaven.engine.types import EngineBusyError
        from mlx_soloheaven.executors import run_read

        models = {}
        for model_id, engine in engines.items():
            try:
                sessions = await run_read(engine.session_stats)
            except EngineBusyError:
                sessions = {"busy": True}
            except Exception:  # noqa: BLE001 — dead child: health stays up
                sessions = {"error": True}
            models[model_id] = {
                "model_id": engine.model_id,
                "sessions": sessions,
            }
        return {"status": "ok", "models": models}

    app.include_router(openai_compat.router)
    app.include_router(chat.router)
    app.include_router(settings.router)
    app.include_router(compaction.router)
    app.include_router(admin.router)
    app.include_router(tokenize.router)
    # Batch D (observability): GET /metrics (Prometheus text). A read endpoint —
    # NOT fronted by the inference gate, so scraping never queues behind a
    # generation holding the single GPU slot.
    app.include_router(metrics.router)

    # Serve web UI static files (must be last — catches all unmatched routes)
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    if os.path.isdir(web_dir):
        from fastapi.responses import FileResponse

        @app.get("/admin")
        async def admin_page():
            return FileResponse(os.path.join(web_dir, "admin.html"))

        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


def _check_port(host: str, port: int):
    """Exit early if port is already in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
    except OSError:
        logger.error(f"Port {port} is already in use. Stop the existing server first.")
        sys.exit(1)
    finally:
        sock.close()


def run_server(cfg: Config):
    """Start the uvicorn server."""
    import uvicorn

    _check_port(cfg.host, cfg.port)

    # Create the app directly and run it — avoids __getattr__ / lazy import issues
    app = create_app(cfg)

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level="info",
        loop="asyncio",
    )
