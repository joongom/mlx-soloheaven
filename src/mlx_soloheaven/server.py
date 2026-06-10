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


def create_app(cfg: Config) -> FastAPI:
    """Build the FastAPI application with all routes and middleware."""
    # NOTE: do NOT import mlx_engine (or anything pulling in mlx.core/mlx_vlm)
    # at parent-process scope. In `--engine-mode process`, MLX must initialize
    # only in the child. The in-process path imports MLXEngine lazily, inside
    # its construction branch below.
    from mlx_soloheaven.storage import database as db
    from mlx_soloheaven.api import openai_compat, chat, admin, settings, compaction

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

    # Log validation errors with request body for debugging
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc):
        body = None
        try:
            body = await request.body()
            body = body.decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        logger.error(
            f"[422] {request.method} {request.url.path} | "
            f"errors={exc.errors()} | body={body}"
        )
        return _JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

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
        model_cfg = Config(
            model_path=mcfg.model_path,
            host=cfg.host,
            port=cfg.port,
            default_temperature=mcfg.default_temperature,
            default_top_p=mcfg.default_top_p,
            default_min_p=mcfg.default_min_p,
            default_top_k=mcfg.default_top_k,
            # Carry the CLI-pinned marker so the child engine knows which
            # sampling fields must NOT be overridden by generation_config.json.
            cli_set_sampling=mcfg.cli_set_sampling,
            default_repetition_penalty=mcfg.default_repetition_penalty,
            default_max_tokens=mcfg.default_max_tokens,
            thinking_budget=mcfg.thinking_budget,
            enable_thinking=mcfg.enable_thinking,
            memory_budget_gb=cfg.memory_budget_gb,
            disk_budget_gb=cfg.disk_budget_gb,
            # Carry the MLX Metal buffer-reuse pool cap; process mode is the
            # default for a single --model, so without this the child engine
            # silently falls back to the dataclass default and ignores
            # --mlx-cache-limit-gb / SOLOHEAVEN_MLX_CACHE_LIMIT_GB.
            mlx_cache_limit_gb=cfg.mlx_cache_limit_gb,
            data_dir=cfg.data_dir,
            verbose=cfg.verbose,
            gpu_keepalive=cfg.gpu_keepalive,
            kv_bits=cfg.kv_bits,
            kv_group_size=cfg.kv_group_size,
            quantized_kv_start=cfg.quantized_kv_start,
            prefill_step_size=cfg.prefill_step_size,
            pld_enabled=cfg.pld_enabled,
            pld_num_draft_tokens=cfg.pld_num_draft_tokens,
            pld_ngram_k=cfg.pld_ngram_k,
            draft_model=cfg.draft_model,
            draft_kind=cfg.draft_kind,
            draft_block_size=cfg.draft_block_size,
            stream_coalesce_n=cfg.stream_coalesce_n,
            stream_coalesce_ms=cfg.stream_coalesce_ms,
            engine_mode="process",
            # Carry the backend selection so the child engine's load_model gate
            # routes mlx-lm vs mlx-vlm correctly (else it defaults to "auto").
            backend=mcfg.backend,
        )
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
            # Create a Config per model with shared server settings
            model_cfg = Config(
                model_path=mcfg.model_path,
                host=cfg.host,
                port=cfg.port,
                default_temperature=mcfg.default_temperature,
                default_top_p=mcfg.default_top_p,
                default_min_p=mcfg.default_min_p,
                default_top_k=mcfg.default_top_k,
                # Carry the CLI-pinned marker so each per-model engine knows
                # which sampling fields must NOT be overridden by
                # generation_config.json.
                cli_set_sampling=mcfg.cli_set_sampling,
                default_repetition_penalty=mcfg.default_repetition_penalty,
                default_max_tokens=mcfg.default_max_tokens,
                thinking_budget=mcfg.thinking_budget,
                enable_thinking=mcfg.enable_thinking,
                memory_budget_gb=cfg.memory_budget_gb,
                disk_budget_gb=cfg.disk_budget_gb,
                # Carry the MLX Metal buffer-reuse pool cap so each per-model
                # engine honours --mlx-cache-limit-gb instead of the dataclass
                # default.
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
                # Speculative-decoding drafter (MTP / DFlash) — must be
                # propagated to the per-model Config or `MLXEngine.load_model`
                # silently skips drafter wiring. Without this, `--draft-model`
                # at the CLI never reaches the engine and MTP is a no-op.
                draft_model=cfg.draft_model,
                draft_kind=cfg.draft_kind,
                draft_block_size=cfg.draft_block_size,
                # Carry the backend selection so each per-model engine's
                # load_model gate routes mlx-lm vs mlx-vlm correctly.
                backend=mcfg.backend,
            )
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

        Fail-closed: a failed flush logs and continues — it must never block
        or corrupt server shutdown.
        """
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

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "models": {
                model_id: {
                    "model_id": engine.model_id,
                    "sessions": engine.session_stats(),
                }
                for model_id, engine in engines.items()
            },
        }

    app.include_router(openai_compat.router)
    app.include_router(chat.router)
    app.include_router(settings.router)
    app.include_router(compaction.router)
    app.include_router(admin.router)

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
