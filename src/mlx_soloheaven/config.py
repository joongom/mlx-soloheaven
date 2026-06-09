"""Configuration dataclass — populated from CLI args or environment."""

import os
from argparse import Namespace
from dataclasses import dataclass, field

# Valid inference backends. `--backend` argparse `choices` only validates a
# value passed explicitly on the CLI — it does NOT validate the argparse
# default, which here is sourced from the SOLOHEAVEN_BACKEND env var, nor any
# value set programmatically. So a typo in SOLOHEAVEN_BACKEND would silently
# flow through to the engine and behave like a forced backend. We validate
# here (in __post_init__) so an invalid value fails LOUDLY at startup.
_VALID_BACKENDS = frozenset({"auto", "mlx-lm", "mlx-vlm"})


def _normalize_backend(backend: str) -> str:
    """Lowercase and validate a backend value, raising loudly if invalid.

    Returns the normalized (lowercased) value. Raises ``ValueError`` for any
    value not in {auto, mlx-lm, mlx-vlm} — this is the single guard that
    catches bad env (SOLOHEAVEN_BACKEND) or programmatic values that bypass
    argparse `choices`.
    """
    normalized = (backend or "auto").lower()
    if normalized not in _VALID_BACKENDS:
        raise ValueError(
            f"invalid backend {backend!r}; "
            f"choose one of auto/mlx-lm/mlx-vlm"
        )
    return normalized


@dataclass
class ModelConfig:
    """Per-model configuration."""
    model_path: str
    alias: str = ""  # Short name for routing (e.g., "qwen3.5-122b")

    # Inference backend: "auto" (text -> mlx-lm, multimodal -> mlx-vlm),
    # "mlx-lm" (force text backend), or "mlx-vlm" (force mlx-vlm; required for
    # the MTP --draft-model speculative stack). See MLXEngine.load_model gate.
    backend: str = "auto"

    # Generation defaults (per-model override)
    default_temperature: float = 0.6
    default_top_p: float = 1.0
    default_min_p: float = 0.0
    default_top_k: int = 0
    # Names of sampling fields ({"temperature","top_p","min_p","top_k"}) that
    # were EXPLICITLY set on the CLI/env (non-None). The engine uses this at
    # load time to decide precedence: a CLI-pinned field is NOT overridden by
    # generation_config.json; an unpinned field IS. Empty => nothing pinned.
    cli_set_sampling: frozenset[str] = field(default_factory=frozenset)
    # FIX 2: enabled by default (1.05) to suppress gemma4's long-session
    # closing-paragraph repetition loop. Overridable via --repetition-penalty
    # / REPETITION_PENALTY. 1.0 disables (RepetitionPenaltyProcessor no-op).
    default_repetition_penalty: float = 1.05
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

    def __post_init__(self) -> None:
        # Normalize + validate the backend so a bad env/programmatic value
        # (which bypasses argparse `choices`) fails loudly here rather than
        # silently behaving like a forced backend downstream.
        self.backend = _normalize_backend(self.backend)

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

    # Engine execution mode: "inprocess" (default, current F3 behavior) or
    # "process" (opt-in: run generation in a separate child process on its
    # MAIN thread for ~90 tok/s at temp>0). Process mode requires a single
    # --model (not --models).
    engine_mode: str = "process"

    # Inference backend: "auto" (text -> mlx-lm, multimodal -> mlx-vlm),
    # "mlx-lm" (force text backend), or "mlx-vlm" (force mlx-vlm; required for
    # the MTP --draft-model speculative stack). See MLXEngine.load_model gate.
    backend: str = "auto"

    # Generation defaults (used as fallback for models without override)
    default_temperature: float = 0.6
    default_top_p: float = 1.0
    default_min_p: float = 0.0
    default_top_k: int = 0
    # Names of sampling fields ({"temperature","top_p","min_p","top_k"}) that
    # were EXPLICITLY set on the CLI/env (non-None). See ModelConfig for the
    # precedence semantics the engine applies with this marker.
    cli_set_sampling: frozenset[str] = field(default_factory=frozenset)
    # FIX 2: enabled by default (1.05) to suppress gemma4's long-session
    # closing-paragraph repetition loop. Overridable via --repetition-penalty
    # / REPETITION_PENALTY. 1.0 disables (RepetitionPenaltyProcessor no-op).
    default_repetition_penalty: float = 1.05
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

    # Speculative decoding via the mlx-vlm drafter (MTP / DFlash). Requires the
    # mlx-vlm backend: set `backend="mlx-vlm"` (CLI `--backend mlx-vlm`). After
    # the mlx-lm-first migration the drafter is an explicit mlx-vlm opt-in; on
    # the default mlx-lm backend use PLD (pld_enabled) for speculative decoding.
    draft_model: str | None = None
    draft_kind: str | None = None
    draft_block_size: int | None = None

    # Streaming coalescing: batch per-token SSE/JSON frames to recover decode
    # throughput lost to per-token cross-thread + asyncio.Queue + GIL overhead.
    # stream_coalesce_n <= 1 DISABLES coalescing (scalar behavior, for A/B).
    stream_coalesce_n: int = 4
    stream_coalesce_ms: int = 30

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

    def __post_init__(self) -> None:
        # Normalize + validate the backend so a bad env/programmatic value
        # (which bypasses argparse `choices`) fails loudly here rather than
        # silently behaving like a forced backend downstream. ModelConfig
        # validates its own `backend` field in its own __post_init__.
        self.backend = _normalize_backend(self.backend)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_dir, "cache")

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "soloheaven.db")

    @classmethod
    def from_args(cls, args: Namespace) -> "Config":
        # Sampling sentinel resolution: an UNSET CLI flag arrives as None, which
        # must leave the dataclass default in place (and be recorded as "not
        # CLI-pinned" so generation_config.json may later fill it). Only fields
        # that are non-None were explicitly set by the user.
        _sampling_args = (
            ("temperature", "default_temperature", args.temperature),
            ("top_p", "default_top_p", args.top_p),
            ("min_p", "default_min_p", args.min_p),
            ("top_k", "default_top_k", args.top_k),
        )
        cli_set_sampling = frozenset(
            name for name, _field, val in _sampling_args if val is not None
        )
        # kwargs for the dataclass: include default_* ONLY when the arg was set,
        # so an unset flag preserves the dataclass fallback (0.6/1.0/0.0/0).
        sampling_kwargs = {
            field_name: val
            for _name, field_name, val in _sampling_args
            if val is not None
        }

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
                    backend=getattr(args, "backend", "auto"),
                    # Unset sampling flags fall through to the dataclass default
                    # here AND leave cli_set_sampling empty for them, so each
                    # model's generation_config.json can fill them at load time.
                    **sampling_kwargs,
                    cli_set_sampling=cli_set_sampling,
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
                backend=getattr(args, "backend", "auto"),
                **sampling_kwargs,
                cli_set_sampling=cli_set_sampling,
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
            engine_mode=getattr(args, "engine_mode", "process"),
            backend=getattr(args, "backend", "auto"),
            **sampling_kwargs,
            cli_set_sampling=cli_set_sampling,
            default_repetition_penalty=args.repetition_penalty,
            default_max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            enable_thinking=not args.no_thinking,
            stream_coalesce_n=args.stream_coalesce_n,
            stream_coalesce_ms=args.stream_coalesce_ms,
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
            draft_model=getattr(args, "draft_model", None),
            draft_kind=getattr(args, "draft_kind", None),
            draft_block_size=getattr(args, "draft_block_size", None),
        )
