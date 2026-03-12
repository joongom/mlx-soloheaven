"""Command-line interface for MLX SoloHeaven."""

import argparse
import os
import sys


def _env(key: str, default: str | None = None) -> str | None:
    """Read from environment with SOLOHEAVEN_ prefix."""
    return os.environ.get(f"SOLOHEAVEN_{key}", default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mlx-soloheaven",
        description="Single-user LLM inference server with KV cache optimization for Apple Silicon",
    )

    p.add_argument(
        "--model", "-m",
        default=_env("MODEL"),
        help="Path to MLX model directory (env: SOLOHEAVEN_MODEL)",
    )
    # SOLOHEAVEN_MODELS: comma-separated paths (e.g., "/path/model1,/path/model2")
    _models_env = _env("MODELS", "").strip()
    p.add_argument(
        "--models",
        nargs="+",
        default=_models_env.split(",") if _models_env else None,
        help="Multiple models: 'path' or 'alias=path' (env: SOLOHEAVEN_MODELS, comma-separated)",
    )
    p.add_argument(
        "--host",
        default=_env("HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=int(_env("PORT", "8000")),
        help="Listen port (default: 8000)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(_env("TEMPERATURE", "0.6")),
        help="Default sampling temperature (default: 0.6)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=int(_env("MAX_TOKENS", "32768")),
        help="Default max generation tokens (default: 32768)",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=int(_env("THINKING_BUDGET", "8192")),
        help="Max thinking tokens before forcing </think> (default: 8192, 0=unlimited)",
    )
    p.add_argument(
        "--memory-budget-gb",
        type=float,
        default=float(_env("MEMORY_BUDGET_GB", "200")),
        help="In-memory KV cache budget in GB (default: 200)",
    )
    p.add_argument(
        "--disk-budget-gb",
        type=float,
        default=float(_env("DISK_BUDGET_GB", "1000")),
        help="On-disk KV cache budget in GB (default: 1000)",
    )
    p.add_argument(
        "--data-dir",
        default=_env("DATA_DIR", "./data"),
        help="Directory for SQLite DB and KV cache files (default: ./data)",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode (no <think> tags in prompt)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=_env("VERBOSE", "").lower() in ("1", "true", "yes"),
        help="Enable verbose logging (env: SOLOHEAVEN_VERBOSE)",
    )
    p.add_argument(
        "--gpu-keepalive",
        action="store_true",
        default=_env("GPU_KEEPALIVE", "").lower() in ("1", "true", "yes"),
        help="Keep Metal GPU warm to avoid idle penalty (env: SOLOHEAVEN_GPU_KEEPALIVE)",
    )

    args = p.parse_args(argv)

    if not args.model and not args.models:
        p.error(
            "Model path is required. Set --model or --models, or SOLOHEAVEN_MODEL environment variable."
        )

    return args


def main(argv: list[str] | None = None):
    # Load .env file if present (before parsing args so env vars are available)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = parse_args(argv)

    from mlx_soloheaven.config import Config
    cfg = Config.from_args(args)

    from mlx_soloheaven.server import run_server
    run_server(cfg)
