"""FastAPI application factory and server entry point."""

import logging
import os
import socket
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mlx_soloheaven.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("soloheaven")


def create_app(cfg: Config) -> FastAPI:
    """Build the FastAPI application with all routes and middleware."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine
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

    # Build per-model configs and engines
    engines: dict[str, MLXEngine] = {}

    if cfg.models:
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
                default_repetition_penalty=mcfg.default_repetition_penalty,
                default_max_tokens=mcfg.default_max_tokens,
                thinking_budget=mcfg.thinking_budget,
                enable_thinking=mcfg.enable_thinking,
                memory_budget_gb=cfg.memory_budget_gb,
                disk_budget_gb=cfg.disk_budget_gb,
                data_dir=cfg.data_dir,
                verbose=cfg.verbose,
                gpu_keepalive=cfg.gpu_keepalive,
            )
            engine = MLXEngine(model_cfg)
            engines[mcfg.model_id] = engine
    else:
        engine = MLXEngine(cfg)
        engines["default"] = engine

    # Default engine = first loaded
    default_engine: MLXEngine = None  # type: ignore

    @app.on_event("startup")
    async def startup():
        nonlocal default_engine

        db.set_db_path(cfg.db_path)
        await db.init_db()
        logger.info(f"Database initialized: {cfg.db_path}")

        for model_id, engine in engines.items():
            engine.load_model()
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
