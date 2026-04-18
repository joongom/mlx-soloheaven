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
        "--top-p",
        type=float,
        default=float(_env("TOP_P", "1.0")),
        help="Default nucleus sampling top-p (default: 1.0, disabled)",
    )
    p.add_argument(
        "--min-p",
        type=float,
        default=float(_env("MIN_P", "0.0")),
        help="Default min-p sampling threshold (default: 0.0, disabled)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(_env("TOP_K", "0")),
        help="Default top-k sampling (default: 0, disabled)",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(_env("REPETITION_PENALTY", "1.0")),
        help="Default repetition penalty (default: 1.0, disabled)",
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
        default=float(_env("DISK_BUDGET_GB", "100")),
        help="On-disk KV cache budget in GB (default: 100)",
    )
    p.add_argument(
        "--max-checkpoints",
        type=int,
        default=int(_env("MAX_CHECKPOINTS", "50")),
        help="Max DeltaNet checkpoints per session for branching (default: 50, 0=unlimited)",
    )
    p.add_argument(
        "--data-dir",
        default=_env("DATA_DIR", "./data"),
        help="Directory for SQLite DB and KV cache files (default: ./data)",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode globally (default: on). Per-model: use :no_think_tag suffix",
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
    p.add_argument(
        "--kv-bits",
        type=int,
        default=int(_env("KV_BITS", "0")),
        choices=[0, 4, 8],
        help="KV cache quantization bits (0=disabled, 4 or 8). mlx-lm path only. (env: SOLOHEAVEN_KV_BITS)",
    )
    p.add_argument(
        "--kv-group-size",
        type=int,
        default=int(_env("KV_GROUP_SIZE", "64")),
        help="KV quantization group size (default: 64)",
    )
    p.add_argument(
        "--quantized-kv-start",
        type=int,
        default=int(_env("QUANTIZED_KV_START", "0")),
        help="Token offset at which KV quantization starts (0=from beginning)",
    )
    p.add_argument(
        "--prefill-step-size",
        type=int,
        default=int(_env("PREFILL_STEP_SIZE", "2048")),
        help="Prefill chunk size (default: 2048). Try 4096/8192 for long-prompt speedup (env: SOLOHEAVEN_PREFILL_STEP_SIZE)",
    )
    p.add_argument("--pld", dest="pld_enabled", action="store_true",
                   default=_env("PLD", "").lower() in ("1", "true", "yes"),
                   help="Enable Prompt Lookup Decoding for speculative generation (env: SOLOHEAVEN_PLD)")
    p.add_argument("--pld-num-draft", type=int, default=int(_env("PLD_NUM_DRAFT", "10")),
                   help="Max draft tokens per step in PLD (default: 10)")
    p.add_argument("--pld-ngram-k", type=int, default=int(_env("PLD_NGRAM_K", "3")),
                   help="N-gram size for PLD matching (default: 3)")

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
