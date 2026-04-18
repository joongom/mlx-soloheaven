"""Configuration dataclass — populated from CLI args or environment."""

import os
from argparse import Namespace
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Per-model configuration."""
    model_path: str
    alias: str = ""  # Short name for routing (e.g., "qwen3.5-122b")

    # Generation defaults (per-model override)
    default_temperature: float = 0.6
    default_top_p: float = 1.0
    default_min_p: float = 0.0
    default_top_k: int = 0
    default_repetition_penalty: float = 1.0
    default_max_tokens: int = 32768
    thinking_budget: int = 8192
    enable_thinking: bool = True

    # Token IDs — auto-detected from tokenizer on model load
    think_end_token: int = -1
    think_start_token: int = -1
    im_end_token: int = -1

    # KV cache quantization (mlx-lm path only; 0=disabled, 4 or 8 recommended)
    kv_bits: int = 0
    kv_group_size: int = 64
    quantized_kv_start: int = 0  # token offset at which quantization kicks in

    # Prefill chunk size (larger = faster prefill at cost of peak memory)
    # mlx-lm default is 2048; try 4096/8192 for long-prompt speedup
    prefill_step_size: int = 2048

    # PLD (Prompt Lookup Decoding): speculative decoding via n-gram prompt lookup
    pld_enabled: bool = False              # Off by default
    pld_num_draft_tokens: int = 10         # Max draft tokens per step
    pld_ngram_k: int = 3                   # N-gram size for matching

    @property
    def model_id(self) -> str:
        """Model ID derived from directory name."""
        return self.alias or os.path.basename(self.model_path.rstrip("/"))


@dataclass
class Config:
    # Models — list of model configs
    models: list[ModelConfig] = field(default_factory=list)

    # Legacy single-model support
    model_path: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Generation defaults (used as fallback for models without override)
    default_temperature: float = 0.6
    default_top_p: float = 1.0
    default_min_p: float = 0.0
    default_top_k: int = 0
    default_repetition_penalty: float = 1.0
    default_max_tokens: int = 32768
    thinking_budget: int = 8192
    enable_thinking: bool = True

    # Token IDs — auto-detected from tokenizer on model load
    think_end_token: int = -1
    think_start_token: int = -1
    im_end_token: int = -1

    # KV cache quantization (mlx-lm path only; 0=disabled, 4 or 8 recommended)
    kv_bits: int = 0
    kv_group_size: int = 64
    quantized_kv_start: int = 0

    # Prefill chunk size (mlx-lm default 2048; 4096/8192 can speed up long prompts)
    prefill_step_size: int = 2048

    # PLD (Prompt Lookup Decoding): speculative decoding via n-gram prompt lookup
    pld_enabled: bool = False              # Off by default
    pld_num_draft_tokens: int = 10         # Max draft tokens per step
    pld_ngram_k: int = 3                   # N-gram size for matching

    # Cache budgets (no time-based TTL — evict LRU when over budget)
    memory_budget_gb: float = 200.0
    disk_budget_gb: float = 100.0

    # Branching: max DeltaNet checkpoints per session (0=unlimited)
    max_checkpoints: int = 50

    # Paths
    data_dir: str = "./data"

    # Logging
    verbose: bool = False

    # GPU
    gpu_keepalive: bool = False

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_dir, "cache")

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "soloheaven.db")

    @classmethod
    def from_args(cls, args: Namespace) -> "Config":
        models = []
        if args.models:
            for spec in args.models:
                # Format: "path", "alias=path", or "path:opt1:opt2"
                # Options: :think (enable), :no_think_tag (disable)
                alias = ""
                enable_thinking = not args.no_thinking

                # Per-model thinking override
                if ":no_think_tag" in spec:
                    spec = spec.replace(":no_think_tag", "")
                    enable_thinking = False

                if "=" in spec:
                    alias, path = spec.split("=", 1)
                else:
                    path = spec

                models.append(ModelConfig(
                    model_path=path.strip(),
                    alias=alias.strip(),
                    default_temperature=args.temperature,
                    default_top_p=args.top_p,
                    default_min_p=args.min_p,
                    default_top_k=args.top_k,
                    default_repetition_penalty=args.repetition_penalty,
                    default_max_tokens=args.max_tokens,
                    thinking_budget=args.thinking_budget,
                    enable_thinking=enable_thinking,
                    pld_enabled=args.pld_enabled,
                    pld_num_draft_tokens=args.pld_num_draft,
                    pld_ngram_k=args.pld_ngram_k,
                ))
        elif args.model:
            models.append(ModelConfig(
                model_path=args.model,
                default_temperature=args.temperature,
                default_top_p=args.top_p,
                default_min_p=args.min_p,
                default_top_k=args.top_k,
                default_repetition_penalty=args.repetition_penalty,
                default_max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
                enable_thinking=not args.no_thinking,
                pld_enabled=args.pld_enabled,
                pld_num_draft_tokens=args.pld_num_draft,
                pld_ngram_k=args.pld_ngram_k,
            ))

        return cls(
            models=models,
            model_path=args.model or (models[0].model_path if models else ""),
            host=args.host,
            port=args.port,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            default_min_p=args.min_p,
            default_top_k=args.top_k,
            default_repetition_penalty=args.repetition_penalty,
            default_max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            enable_thinking=not args.no_thinking,
            memory_budget_gb=args.memory_budget_gb,
            disk_budget_gb=args.disk_budget_gb,
            max_checkpoints=args.max_checkpoints,
            data_dir=args.data_dir,
            verbose=args.verbose,
            gpu_keepalive=args.gpu_keepalive,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            prefill_step_size=args.prefill_step_size,
            pld_enabled=args.pld_enabled,
            pld_num_draft_tokens=args.pld_num_draft,
            pld_ngram_k=args.pld_ngram_k,
        )
