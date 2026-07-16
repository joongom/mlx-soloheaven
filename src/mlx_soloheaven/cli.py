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
        "--backend",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default=_env("BACKEND", "auto"),
        help="Inference backend: 'auto' (default; mlx-lm-first BY SUPPORT — "
             "mlx-lm whenever it supports the model_type, incl. gemma4; falls "
             "to mlx-vlm only for types mlx-lm cannot load), 'mlx-lm' (force "
             "mlx-lm), or 'mlx-vlm' (force mlx-vlm; REQUIRED for the MTP "
             "--draft-model speculative stack). (env: SOLOHEAVEN_BACKEND)",
    )
    # Sampling defaults use a None sentinel (NOT a hardcoded number) so that an
    # UNSET flag is distinguishable from an explicitly-set one. None means
    # "fall through to the model's generation_config.json (then to the built-in
    # dataclass fallback)"; a concrete value (CLI flag or SOLOHEAVEN_* env)
    # pins the field so generation_config can't override it. Guard the env
    # coercion: float()/int() must only run when the env value is present,
    # else float(None) raises at parse time. argparse's type= applies only to
    # values actually given on argv, so a None default passes through untouched.
    _temp_env = _env("TEMPERATURE")
    p.add_argument(
        "--temperature",
        type=float,
        default=float(_temp_env) if _temp_env is not None else None,
        help="Default sampling temperature (unset: use model generation_config, "
             "then fallback 0.6)",
    )
    _top_p_env = _env("TOP_P")
    p.add_argument(
        "--top-p",
        type=float,
        default=float(_top_p_env) if _top_p_env is not None else None,
        help="Default nucleus sampling top-p (unset: use model generation_config, "
             "then fallback 1.0=disabled)",
    )
    _min_p_env = _env("MIN_P")
    p.add_argument(
        "--min-p",
        type=float,
        default=float(_min_p_env) if _min_p_env is not None else None,
        help="Default min-p sampling threshold (unset: use model generation_config, "
             "then fallback 0.0=disabled)",
    )
    _top_k_env = _env("TOP_K")
    p.add_argument(
        "--top-k",
        type=int,
        default=int(_top_k_env) if _top_k_env is not None else None,
        help="Default top-k sampling (unset: use model generation_config, "
             "then fallback 0=disabled)",
    )
    # U23: same None sentinel as the sampling flags above — a hardcoded
    # argparse default (previously 1.0) is ALWAYS supplied by argparse, so it
    # unconditionally clobbered the Config dataclass default (1.05, the gemma4
    # anti-repetition FIX 2), making that default dead code. Precedence with
    # the sentinel: CLI/env explicitly set > dataclass default (1.05). The
    # generation_config.json auto-apply is INTENTIONALLY not a layer for this
    # field (see _load_generation_config_sampling: HF's value — often
    # 1.0/absent — would silently disable the gemma4 mitigation).
    _rep_env = _env("REPETITION_PENALTY")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(_rep_env) if _rep_env is not None else None,
        help="Default repetition penalty (unset: config default 1.05; "
             "1.0 disables). Sane range: 1.0-1.5.",
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
        "--stream-coalesce-n",
        type=int,
        default=int(_env("STREAM_COALESCE_N", "4")),
        help="Max tokens batched per SSE frame (default: 4; <=1 disables coalescing). "
             "(env: SOLOHEAVEN_STREAM_COALESCE_N)",
    )
    p.add_argument(
        "--stream-coalesce-ms",
        type=int,
        default=int(_env("STREAM_COALESCE_MS", "30")),
        help="Max ms to hold a partial batch before flushing (default: 30). "
             "(env: SOLOHEAVEN_STREAM_COALESCE_MS)",
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
        "--mlx-cache-limit-gb",
        type=float,
        default=float(_env("MLX_CACHE_LIMIT_GB", "4")),
        help="Upper bound for MLX's Metal buffer-reuse pool in GB, applied at "
             "startup via mx.set_cache_limit (default: 4). "
             "(env: SOLOHEAVEN_MLX_CACHE_LIMIT_GB)",
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
    # Speculative decoding via a drafter model (MTP / DFlash), dispatched by
    # the drafter's model_type.
    p.add_argument(
        "--draft-model",
        default=_env("DRAFT_MODEL"),
        help="Path to drafter MLX model directory (enables MTP/DFlash "
             "speculative decoding). qwen3_5_mtp heads (Qwen3.5/3.6 MTP) run "
             "NATIVELY on the default mlx-lm backend; gemma4_assistant MTP / "
             "DFlash drafters require `--backend mlx-vlm`. Without a drafter "
             "the mlx-lm backend can use --pld instead. "
             "(env: SOLOHEAVEN_DRAFT_MODEL)",
    )
    p.add_argument(
        "--draft-kind",
        choices=["mtp", "dflash"],
        default=_env("DRAFT_KIND"),
        help="mlx-vlm drafter kind override: 'mtp' (Gemma 4) or 'dflash' "
             "(Qwen3). Default: auto-detect from drafter's model_type "
             "(qwen3_5_mtp heads ignore this and resolve to 'qwen_mtp').",
    )
    p.add_argument(
        "--draft-block-size",
        type=int,
        default=int(_env("DRAFT_BLOCK_SIZE", "0")) or None,
        help="Drafter block size (default: use drafter config).",
    )

    p.add_argument(
        "--engine-mode",
        choices=["inprocess", "process"],
        default=_env("ENGINE_MODE", "process"),
        help="Engine execution mode: 'process' (default; run generation in a "
             "separate child process on its main thread for higher decode tok/s; "
             "single --model only, --models auto-falls-back to inprocess) or "
             "'inprocess'. (env: SOLOHEAVEN_ENGINE_MODE)",
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

    # Drafter (MTP/DFlash) requires single-model path; reject multi-model + drafter combos.
    if (args.models or []) and (
        getattr(args, "draft_model", None)
        or getattr(args, "draft_kind", None)
        or getattr(args, "draft_block_size", None)
    ):
        p.error(
            "--draft-model is not supported with --models (multi-model); use --model for single-model + drafter"
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
