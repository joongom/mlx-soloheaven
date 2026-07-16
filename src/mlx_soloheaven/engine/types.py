"""Shared engine result dataclasses + (de)serialization helpers.

These live in their own module (with NO mlx / mlx-vlm imports) so they can
be imported in both the FastAPI PARENT process and the generation CHILD
process without dragging in heavy MLX dependencies. ``mlx_engine`` re-exports
``GenerationResult`` / ``CompletionResult`` from here so existing
``from .mlx_engine import GenerationResult`` imports keep working unchanged.
"""

from dataclasses import dataclass, asdict
from typing import Optional


class EngineBusyError(RuntimeError):
    """A READ-ONLY engine query could not be served within its bounded wait
    because a generation currently owns the engine (in-process mode: the
    global engine lock; process mode: the child worker services RPCs only
    BETWEEN generations).

    Raised instead of hanging for the full generation so API callers can
    DEGRADE (admin/stats endpoints answer with a "busy" placeholder, session
    reads answer 503 "retry shortly") — U14/U15. Lives here (no mlx imports)
    so the parent-process proxy, the in-process engine, and the API layer all
    share one exception type."""


class GenerationCancelled(Exception):
    """Raised INSIDE chunked prefill (between chunks — engine
    ``_prefill_cache``, mlx-lm's ``prompt_progress_callback`` hook, the
    ``qwen_mtp`` prefill loops, PLD ``_prefill``) when the request's
    ``cancel_event`` is set (U13).

    The engine's stream loop converts it into the NORMAL cancel path, whose
    reconcile applies the C1 fail-closed rules: a HIT session's partially
    prefilled suffix is rolled back (verified per-layer trim) or the session
    cache is invalidated; a MISS's partially-filled fresh cache is simply
    discarded (never persisted). Deliberately an ``Exception`` (not
    BaseException): it must be catchable by the engine loop and must never
    masquerade as a shutdown sentinel."""


@dataclass
class GenerationResult:
    """A single token/chunk from generation."""
    text: str = ""
    token: int = 0
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    status: Optional[str] = None  # "generating" when lock acquired
    cache_info: Optional[dict] = None  # cache hit/miss details

    def to_dict(self) -> dict:
        """Serialize to a plain dict for IPC (picklable, JSON-compatible)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationResult":
        """Rehydrate from an IPC dict. Ignores unknown keys defensively."""
        return cls(
            text=d.get("text", ""),
            token=d.get("token", 0),
            finish_reason=d.get("finish_reason"),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            prompt_tps=d.get("prompt_tps", 0.0),
            generation_tps=d.get("generation_tps", 0.0),
            status=d.get("status"),
            cache_info=d.get("cache_info"),
        )


@dataclass
class CompletionResult:
    """Full completion result after generation finishes."""
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    cache_info: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CompletionResult":
        return cls(
            content=d.get("content"),
            thinking=d.get("thinking"),
            tool_calls=d.get("tool_calls"),
            finish_reason=d.get("finish_reason", "stop"),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            prompt_tps=d.get("prompt_tps", 0.0),
            generation_tps=d.get("generation_tps", 0.0),
            cache_info=d.get("cache_info"),
        )
