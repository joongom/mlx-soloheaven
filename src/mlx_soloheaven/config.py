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
    default_max_tokens: int = 32768
    thinking_budget: int = 8192
    enable_thinking: bool = True

    # Token IDs — auto-detected from tokenizer on model load
    think_end_token: int = -1
    think_start_token: int = -1
    im_end_token: int = -1

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
    default_max_tokens: int = 32768
    thinking_budget: int = 8192
    enable_thinking: bool = True

    # Token IDs — auto-detected from tokenizer on model load
    think_end_token: int = -1
    think_start_token: int = -1
    im_end_token: int = -1

    # Cache budgets (no time-based TTL — evict LRU when over budget)
    memory_budget_gb: float = 200.0
    disk_budget_gb: float = 1000.0

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
                # Options: no_thinking
                alias = ""
                enable_thinking = not args.no_thinking

                # Split options (after last colon that's not part of path)
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
                    default_max_tokens=args.max_tokens,
                    thinking_budget=args.thinking_budget,
                    enable_thinking=enable_thinking,
                ))
        elif args.model:
            models.append(ModelConfig(
                model_path=args.model,
                default_temperature=args.temperature,
                default_max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
                enable_thinking=not args.no_thinking,
            ))

        return cls(
            models=models,
            model_path=args.model or (models[0].model_path if models else ""),
            host=args.host,
            port=args.port,
            default_temperature=args.temperature,
            default_max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            enable_thinking=not args.no_thinking,
            memory_budget_gb=args.memory_budget_gb,
            disk_budget_gb=args.disk_budget_gb,
            data_dir=args.data_dir,
            verbose=args.verbose,
            gpu_keepalive=args.gpu_keepalive,
        )
