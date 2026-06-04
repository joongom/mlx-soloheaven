"""
MLX Engine — model loading and generation with KV cache reuse.

mlx-vlm-first; mlx-lm legacy fallback for text-only model_types not in
mlx-vlm's registry. The mlx-vlm path is the canonical (and drafter-ready)
generation surface; the mlx-lm branch remains load-bearing for text-only
model_types such as Qwen3.5/3.6 dense, MiniMax-M2.5, GLM-4.7, and GLM-5.1
that mlx-vlm does not currently register under `mlx_vlm.models.<type>`.

Session-based KV cache management is built on mlx-vlm's PromptCacheState
which does prefix-matching on token IDs:
- Engine-internal messages store full assistant content (including thinking)
- apply_chat_template(tokenize=True) produces tokens matching stored IDs
- PromptCacheState.find_prefix_length() reuses the common prefix
- Only new user-message tokens are processed each turn

Cross-session sharing:
- Base cache pool stores system-prompt KV snapshots
- New sessions are seeded from the base cache via PromptCacheState
"""

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, Optional

import mlx.core as mx
from mlx_lm import load as lm_load
from mlx_lm import stream_generate as lm_stream_generate
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import (
    stream_generate as vlm_stream_generate,
    PromptCacheState,
)

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.thinking import ThinkingBudgetProcessor, RepetitionPenaltyProcessor
from mlx_soloheaven.engine.tool_parser import (
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
)
from mlx_soloheaven.cache.manager import CacheManager

logger = logging.getLogger(__name__)

# Heuristic threshold for the per-request drafter low-acceptance WARNING.
# At block_size=3 the max possible mean_accepted is 2.0 (every block fully
# accepted). A mean below 0.5 means more than 75% of drafted tokens are
# being rejected — at that rate the drafter is typically net negative.
_DRAFTER_LOW_ACCEPT_THRESHOLD = 0.5


def _pld_response_adapter(pld_iter, tokenizer):
    """Adapt pld_generate_step's (token, logprobs, from_draft) tuples to
    mimic lm_stream_generate's GenerationResponse objects.

    - Uses mlx-lm's StreamingDetokenizer (buffers partial UTF-8 byte
      sequences so multi-byte characters like CJK aren't emitted as
      replacement chars \ufffd between tokens).
    - Performs EOS detection — pld_generate_step doesn't stop on its own
      (mlx-lm's stream_generate normally handles that). EOS ids include
      the wrapper's list + HF tokenizer's (list or single).
    """
    import time as _time
    from types import SimpleNamespace

    # Collect EOS token IDs from both mlx-lm wrapper and HF tokenizer
    eos_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        eos_ids.update(tokenizer.eos_token_ids)
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    eid = getattr(inner, "eos_token_id", None)
    if eid is not None:
        if isinstance(eid, (list, tuple, set)):
            eos_ids.update(eid)
        else:
            eos_ids.add(eid)
    # GLM-family and other models expose multi-EOS via generation_config
    gc = getattr(inner, "generation_config", None)
    if gc is not None:
        gc_eos = getattr(gc, "eos_token_id", None)
        if gc_eos is not None:
            if isinstance(gc_eos, (list, tuple, set)):
                eos_ids.update(gc_eos)
            else:
                eos_ids.add(gc_eos)

    # Use mlx-lm's StreamingDetokenizer to buffer partial UTF-8 bytes
    # across tokens (mirrors stream_generate's behavior).
    detok = None
    make_detok = getattr(tokenizer, "detokenizer", None)
    if make_detok is not None:
        # TokenizerWrapper.detokenizer is a property that returns a fresh
        # detokenizer every access. Call .reset() and use it directly.
        try:
            detok = tokenizer.detokenizer
            detok.reset()
        except Exception:
            detok = None

    t_first = None
    count = 0
    from_draft_count = 0
    last_segment = ""
    for token, _logprobs, from_draft in pld_iter:
        count += 1
        if from_draft:
            from_draft_count += 1
        if t_first is None:
            t_first = _time.perf_counter()
        now = _time.perf_counter()
        tps = count / (now - t_first) if t_first and now > t_first else 0.0

        # Stop on EOS BEFORE emitting the EOS token's (often empty) text
        if token in eos_ids:
            # Flush any remaining buffered segment first
            if detok is not None:
                try:
                    detok.finalize()
                    remaining = detok.last_segment
                    if remaining:
                        yield SimpleNamespace(
                            text=remaining, token=token,
                            prompt_tps=0.0, generation_tps=tps,
                            from_draft=from_draft,
                        )
                except Exception:
                    pass
            break

        if detok is not None:
            try:
                detok.add_token(token)
                text = detok.last_segment
            except Exception:
                # Fallback: decode single token (may yield replacement chars)
                text = tokenizer.decode([token])
        else:
            text = tokenizer.decode([token])

        if not text:
            # Partial UTF-8 — buffered, wait for next token
            continue

        yield SimpleNamespace(
            text=text, token=token,
            prompt_tps=0.0, generation_tps=tps,
            from_draft=from_draft,
        )

    if count > 0:
        logger.info(
            f"[PLD] accepted {from_draft_count}/{count} draft tokens "
            f"({100*from_draft_count/count:.1f}% acceptance rate)"
        )


# GenerationResult / CompletionResult moved to engine/types.py (no-mlx module
# so they can be imported in the process-mode parent + child). Re-exported
# here so existing `from .mlx_engine import GenerationResult` imports work.
from mlx_soloheaven.engine.types import GenerationResult, CompletionResult


@dataclass
class SessionState:
    """Tracks a conversation session's KV cache and message history.

    Engine-internal messages include full assistant content (thinking + content)
    so that apply_chat_template(tokenize=True) produces tokens matching the
    stored PromptCacheState.token_ids for prefix-matching cache reuse.
    """
    cache_state: PromptCacheState  # mlx-vlm native: KV cache + token history
    messages: list[dict]  # messages WITH thinking in assistant content
    last_used: float = field(default_factory=time.time)
    total_cache_tokens: int = 0

    # Cache build time from last truncate/rebuild (seconds, consumed once)
    pending_build_time: float = 0.0

    # Cumulative drafter acceptance stats across all requests in this session.
    # None until the first drafter-enabled request completes. Shape:
    # {"requests": N, "total_rounds": R, "total_accepted": A}
    drafter_stats: dict | None = None

    def touch(self):
        self.last_used = time.time()


def _detect_token_id(tokenizer, text: str) -> int:
    """Auto-detect a special token's ID from the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()
    if text in vocab:
        return vocab[text]
    # Try encoding as fallback
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return -1


# --- mlx-vlm 0.5.0 Gemma 4 MTP wrap-around patches ---
#
# Two coordinated bugs surface once a ``RotatingKVCache`` ring buffer
# wraps (offset > max_size) while the Gemma 4 MTP drafter is running:
#
#   B1. ``Gemma4TextModel.__call__`` writes the sliding-attention layer's
#       rotated (non-temporal) keys/values directly into ``shared_kv_sink``.
#       The drafter's SWA mask (mlx_vlm/speculative/drafters/gemma4_assistant/
#       masks.py) assumes ``k_idx = arange(kv_len)`` is temporal — after wrap
#       that assumption is violated and the drafter attends to the wrong
#       keys. Symptom: drafter mean_accepted collapses (~1.17 → ~0.26).
#
#   B2. ``mlx_vlm.generate._mtp_rounds`` reads ``kv_offset = int(prompt_cache[0]
#       .offset)`` which is the *logical-cumulative* token count. After wrap
#       this exceeds ``max_size`` and the drafter computes a wrong query
#       offset for its SWA mask. The offset is read at TWO sites (entry,
#       and inside the verify/rollback loop). The earlier save/restore
#       patch only covered the entry read — the second read inside the
#       loop ran with the un-clamped offset and silently masked-out the
#       entire in-window key range (q_idx - k_idx >= window), leaving the
#       drafter blind → garbage tokens → no EOS → infinite max_tokens loop.
#       Current fix: replace ``_mtp_rounds`` with a corrected clone that
#       applies ``min(offset, max_size)`` at BOTH read sites for
#       RotatingKVCache (clone-replace, not save/restore).
#
# NOTE: B3 was previously installed to skip ``c.trim`` on a wrapped
# RotatingKVCache (``is_trimmable() == False``), but RCA-2 (2026-05-13)
# determined that ``c.trim(n)`` is unconditionally safe (``offset -= n;
# _idx -= n``); skipping it leaves rejected speculative K/V slots in the
# ring buffer and contaminates target attention, causing post-wrap MTP
# output to degrade into an infinite repetition loop. Upstream
# ``rollback_speculative_cache`` is now used unchanged.
#
# Idempotency: ``_MTP_PATCHES_INSTALLED`` guards re-application. Each
# patch records the original on the module so re-running the helper
# (e.g. across worker restarts in tests) does not double-wrap.
_MTP_PATCHES_INSTALLED = False

# PERF: when ``_run_vlm`` knows wrap is NOT imminent for the current
# request (via ``_will_wrap_during_generate``), it flips this flag so
# the B1/B2-v2 monkey-patches become near-noop pass-throughs and avoid
# their per-call guard work on the hot speculative-decoding verify path.
# Flag is single-worker safe — VLM executor pins all calls to one thread.
_HOT_PATH_FAST = False


def _clamped_kv_offset(prompt_cache):
    """Return ``prompt_cache[0].offset`` clamped to ``max_size`` for a
    wrapped ``RotatingKVCache``.

    This is the B2 fix: after the ring buffer wraps, the raw
    ``offset`` is the logical cumulative token count and exceeds the
    physical ``max_size``. The Gemma 4 drafter's SWA mask derives the
    query-row index from this value and masks out keys whose
    ``q_idx - k_idx >= window`` — with an un-clamped ``q_idx`` (e.g.
    1100 when ``max_size`` is 1024) every in-window key is rejected.
    Clamping to ``max_size`` keeps ``q_idx`` inside the window so the
    distance check admits the correct K range.

    Non-RotatingKVCache entries (KVCache, etc.) return their raw
    offset unchanged — only the ring-buffer cache exhibits this
    ``offset > max_size`` divergence.
    """
    c = prompt_cache[0]
    off = int(getattr(c, "offset", 0) or 0)
    if type(c).__name__ == "RotatingKVCache":
        max_size = int(getattr(c, "max_size", 0) or 0)
        if max_size > 0 and off > max_size:
            return max_size
    return off


def _install_mtp_wrap_patches() -> bool:
    """Patch the three mlx-vlm 0.5.0 Gemma 4 MTP wrap-around bugs.

    Returns True if patches were applied this call, False if already
    installed.  Safe to call multiple times.
    """
    global _MTP_PATCHES_INSTALLED
    if _MTP_PATCHES_INSTALLED:
        return False

    try:
        import mlx_vlm.models.gemma4.language as _g4lang
        import mlx_vlm.generate as _mvgen_attr  # noqa: F401 — populate sys.modules
        _mvgen = sys.modules["mlx_vlm.generate"]
    except ImportError:
        # mlx-vlm not installed / Gemma 4 module unavailable — nothing to
        # patch. Mark installed so we don't retry on every worker init.
        _MTP_PATCHES_INSTALLED = True
        return False

    # ----- B1: temporal-order shared_kv writes -----
    # Wrap ``Gemma4TextModel.__call__`` so any (keys, values) that land in
    # ``shared_kv_sink`` are converted to temporal order when the
    # corresponding cache layer is a RotatingKVCache that has wrapped.
    _orig_textmodel_call = _g4lang.Gemma4TextModel.__call__

    # PERF: per-cache-list memoization of (rotating_layer_indices,
    # layer_types). Hot non-MTP forwards early-return BEFORE this map
    # is ever touched. MTP forwards (shared_kv_sink != None) iterate
    # only the rotating layer indices instead of all 60.
    # Keyed by id(cache); bounded to 32 entries.
    _rot_idx_cache: dict[int, tuple[tuple[int, ...], tuple[str | None, ...]]] = {}

    def _patched_textmodel_call(self, *args, **kwargs):  # noqa: D401
        # PERF: fast-path bypass — when ``_run_vlm`` has determined wrap
        # is not imminent for the in-flight request, the entire
        # temporal-order rewrite is irrelevant. Skip all guard work and
        # just delegate to the original method.
        if _HOT_PATH_FAST:
            return _orig_textmodel_call(self, *args, **kwargs)
        shared_kv_sink = kwargs.get("shared_kv_sink")
        cache = kwargs.get("cache")
        out = _orig_textmodel_call(self, *args, **kwargs)
        # PERF: non-MTP path (shared_kv_sink is None) early-returns
        # BEFORE any per-layer work. This is the common (baseline) case.
        if shared_kv_sink is None or not cache:
            return out
        try:
            cache_key = id(cache)
            entry = _rot_idx_cache.get(cache_key)
            if entry is None:
                layers = getattr(self, "layers", [])
                indices: list[int] = []
                ltypes: list[str | None] = []
                for idx, c in enumerate(cache):
                    if (
                        c is not None
                        and idx < len(layers)
                        and type(c).__name__ == "RotatingKVCache"
                        and hasattr(c, "_temporal_order")
                    ):
                        indices.append(idx)
                        ltypes.append(getattr(layers[idx], "layer_type", None))
                entry = (tuple(indices), tuple(ltypes))
                if len(_rot_idx_cache) >= 32:
                    _rot_idx_cache.pop(next(iter(_rot_idx_cache)))
                _rot_idx_cache[cache_key] = entry
            indices, ltypes = entry
            for pos, idx in enumerate(indices):
                c = cache[idx]
                if c is None:
                    continue
                max_size = getattr(c, "max_size", 0)
                offset = getattr(c, "offset", 0)
                if offset <= max_size:
                    continue
                lt = ltypes[pos]
                if lt is None or lt not in shared_kv_sink:
                    continue
                kv = shared_kv_sink[lt]
                if not (isinstance(kv, tuple) and len(kv) == 2):
                    continue
                K, V = kv
                shared_kv_sink[lt] = (c._temporal_order(K), c._temporal_order(V))
        except Exception:  # noqa: BLE001 — patch must never break inference
            logger.exception("[MTP-Patch B1] temporal-order rewrite failed")
        return out

    _patched_textmodel_call._mtp_wrap_patch = True  # marker for tests
    _g4lang.Gemma4TextModel.__call__ = _patched_textmodel_call

    # ----- B2: clamp kv_offset on wrapped RotatingKVCache -----
    # Clone-replace: re-implement _mtp_rounds with the offset clamp
    # applied at BOTH read sites. The old save/restore patch only
    # covered the entry read; the second read inside the while-loop
    # (after verify+rollback, before the next set_shared_kv) still
    # received the un-clamped offset and silently broke the drafter's
    # SWA mask → infinite-loop until max_tokens.
    _orig_mtp_rounds = getattr(_mvgen, "_mtp_rounds", None)
    _speculative_walk = getattr(_mvgen, "_speculative_walk", None)
    _generation_stream = getattr(_mvgen, "generation_stream", None)
    if (
        _orig_mtp_rounds is not None
        and _speculative_walk is not None
        and _generation_stream is not None
    ):
        def _patched_mtp_rounds_v2(
            model,
            draft_model,
            prompt_cache,
            hidden,
            shared_kv_states,
            *,
            first_bonus,
            max_tokens,
            sampler,
            draft_block_size=None,
            token_dtype=mx.int32,
        ):
            # PERF: fast-path bypass when wrap is not imminent — delegate
            # straight to the upstream _mtp_rounds. The clamp is only
            # needed when a wrap has occurred (offset > max_size).
            if _HOT_PATH_FAST:
                return _orig_mtp_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    shared_kv_states,
                    first_bonus=first_bonus,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=token_dtype,
                )

            # --- Clone of mlx_vlm.generate._mtp_rounds with B2 clamp. ---
            lm = (
                model.language_model
                if hasattr(model, "language_model")
                else model
            )
            if not hasattr(lm, "rollback_speculative_cache"):
                raise RuntimeError(
                    f"{type(lm).__name__} does not implement "
                    "rollback_speculative_cache. MTP speculative decoding "
                    "currently only supports gemma4."
                )

            block_total = (
                draft_block_size
                if draft_block_size is not None
                else int(draft_model.config.block_size)
            )
            draft_model.reset(model)

            if hidden.shape[1] > 1:
                hidden = hidden[:, -1:, :]

            # B2 fix (read #1): entry-time clamped offset.
            kv_offset = _clamped_kv_offset(prompt_cache)
            draft_model.set_shared_kv(shared_kv_states, kv_offset)

            b = first_bonus
            emitted = 1  # caller already yielded the first bonus

            while emitted < max_tokens:
                bs = min(block_total, max_tokens - emitted + 1)
                if bs <= 1:
                    break

                draft_tokens = draft_model.draft_block(
                    b, hidden, None, bs, sampler, token_dtype
                )
                mx.async_eval(draft_tokens)

                with mx.stream(_generation_stream):
                    verify_input = mx.concatenate(
                        [mx.array([[b]], dtype=token_dtype), draft_tokens],
                        axis=1,
                    )
                    verify_out = lm(
                        verify_input,
                        cache=prompt_cache,
                        return_hidden=True,
                        return_shared_kv=True,
                    )
                    hidden_full = verify_out.hidden_states[-1]
                    target_tokens = sampler(verify_out.logits)
                mx.async_eval(target_tokens, hidden_full)

                accepted, new_tokens = _speculative_walk(
                    draft_tokens, target_tokens, max_tokens - emitted
                )
                draft_model.accept_lens.append(accepted)

                for tok in new_tokens:
                    yield tok, None
                    emitted += 1
                    if emitted >= max_tokens:
                        return

                hidden = hidden_full[:, accepted : accepted + 1, :]
                b = new_tokens[-1] if new_tokens else b

                if accepted < bs - 1:
                    with mx.stream(_generation_stream):
                        lm.rollback_speculative_cache(
                            prompt_cache, None, accepted, bs
                        )

                rejected = bs - (accepted + 1)
                next_shared_kv = {}
                for k, kv in verify_out.shared_kv_states.items():
                    K, V = kv
                    valid = K.shape[-2] - rejected
                    if valid <= 0 or valid >= K.shape[-2]:
                        next_shared_kv[k] = (
                            (K, V)
                            if valid >= K.shape[-2]
                            else (K[..., :1, :], V[..., :1, :])
                        )
                    else:
                        next_shared_kv[k] = (
                            K[..., :valid, :],
                            V[..., :valid, :],
                        )
                # B2 fix (read #2): inner-loop clamped offset. Without
                # this, after wrap the drafter's SWA mask receives
                # q_idx > max_size and rejects every in-window key.
                kv_offset = _clamped_kv_offset(prompt_cache)
                draft_model.set_shared_kv(next_shared_kv, kv_offset)

                if emitted % 256 == 0:
                    mx.clear_cache()

        _patched_mtp_rounds_v2._mtp_wrap_patch = True
        _mvgen._mtp_rounds = _patched_mtp_rounds_v2
        logger.debug("[MTP-Patch B2-v2] applied clone-replace path")
    elif _orig_mtp_rounds is not None:
        # Required symbol missing in this mlx-vlm version — skip rather
        # than crash. The original (unclamped-second-read) function
        # stays in place; tests covering B2 will reflect this regime.
        logger.warning(
            "[MTP-Patch B2] required symbols missing "
            "(_speculative_walk/generation_stream); B2 clone-replace skipped"
        )

    # ----- B3 REMOVED (RCA-2, 2026-05-13) -----
    # Previously we monkey-patched ``LanguageModel.rollback_speculative_cache``
    # to skip ``c.trim`` when ``c.is_trimmable()`` was False (i.e. after
    # a RotatingKVCache wrap). Upstream's ``RotatingKVCache.trim(n)`` is
    # unconditionally safe — it just does ``offset -= n; _idx -= n`` —
    # so skipping it leaves rejected speculative K/V slots resident in
    # the ring buffer, contaminating target attention and degrading
    # post-wrap output into an infinite repetition loop. Upstream
    # ``rollback_speculative_cache`` is now used unchanged.

    _MTP_PATCHES_INSTALLED = True
    logger.info(
        "[MTP-Patch] installed wrap-around patches for "
        "mlx_vlm.models.gemma4 (B1+B2-v2)"
    )
    return True


def _maybe_load_drafter(
    draft_model_path: str | None = None,
    kind: str | None = None,
):
    """Load a speculative drafter; returns (model, kind) or (None, None)."""
    if not draft_model_path:
        return (None, None)
    try:
        from mlx_vlm.speculative import load_drafter
    except ImportError as e:
        raise RuntimeError(
            "mlx_vlm.speculative is unavailable; --draft-model requires "
            "mlx-vlm >= 0.5.0"
        ) from e
    t0 = time.perf_counter()
    model, resolved_kind = load_drafter(draft_model_path, kind=kind)
    elapsed = time.perf_counter() - t0
    block_size = None
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None:
        block_size = getattr(cfg_obj, "block_size", None)
    logger.info(
        f"[Drafter] loaded {draft_model_path} kind={resolved_kind} "
        f"block_size={block_size} in {elapsed:.1f}s"
    )
    return (model, resolved_kind)


@dataclass
class BaseCacheEntry:
    """A cached KV state for a shared system prompt prefix."""
    system_hash: str
    cache: list  # mlx-lm prompt_cache snapshot (at end of system tokens)
    tokens: list[int]  # tokenized system message
    token_count: int
    created: float = field(default_factory=time.time)
    hit_count: int = 0


# Precompiled regexes used by message-normalization / cache-match hot path.
# Patterns + flags are byte-identical to the original inline re.* calls.
_NORMALIZE_RE_IMAGE_REMOVED = re.compile(r"\s*\[image data removed", re.IGNORECASE)
_NORMALIZE_RE_TODAYS_DATE = re.compile(
    r"Today's date:\s*\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4}"
)
_NORMALIZE_RE_SYSTEM_REMINDER = re.compile(
    r"\n?<system-reminder>.*?</system-reminder>", re.DOTALL
)
_NORMALIZE_RE_THINK_PREFIX = re.compile(r"^<think>\n?")
_NORMALIZE_RE_CHANNEL_THOUGHT_PREFIX = re.compile(r"^<\|channel>thought\n?")
_NORMALIZE_RE_THOUGHT_PREFIX = re.compile(r"^thought\n")
_NORMALIZE_RE_TOOL_CALL_CHANNEL = re.compile(
    r"<\|?tool_call>.*?<\|?tool_call\|?>", re.DOTALL
)
_NORMALIZE_RE_TOOL_CALL_XML = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_NORMALIZE_RE_TOOL_CALL_SPLIT = re.compile(r"\n*<tool_call>")


class MLXEngine:
    """MLX model engine with session-based KV cache reuse."""

    # Metal GPU goes cold after ~5s idle, causing ~2s TTFT penalty.
    # Keep GPU warm with periodic small computation.
    GPU_KEEPALIVE_INTERVAL = 1.0  # Short interval to prevent deep Metal sleep

    # Shared GPU lock across all engines (Metal can't handle concurrent command encoders)
    _global_gpu_lock = threading.Lock()
    _global_keepalive_started = False
    _global_last_gpu_activity = time.time()
    _global_keepalive_stop = threading.Event()
    _all_engines: list["MLXEngine"] = []

    def __init__(self, cfg: Config, execution_mode: str = "worker"):
        # execution_mode:
        #   "worker"      — DEFAULT, unchanged F3 behavior: a dedicated
        #                   persistent ThreadPoolExecutor worker thread owns
        #                   the mlx-vlm `generation_stream` + model + drafter.
        #   "main_thread" — process-mode (Stage 1): the engine is constructed
        #                   inside a dedicated CHILD process and ALL mlx work
        #                   (load + generation + disk save + MTP patches) runs
        #                   inline on the calling (== child's main) thread. No
        #                   `_vlm_executor` is created, and the GPU-keepalive
        #                   background thread is never started (no background
        #                   thread may touch MLX cache tensors in this mode).
        self.execution_mode = execution_mode
        self.cfg = cfg
        self._vlm_model = None
        self._language_model = None
        self._processor = None
        self.tokenizer = None
        self._lock = MLXEngine._global_gpu_lock  # shared lock
        self.cache_manager = CacheManager(
            memory_budget_gb=cfg.memory_budget_gb,
            disk_budget_gb=cfg.disk_budget_gb,
            cache_dir=cfg.cache_dir,
        )
        self.model_id = ""

        # Session-based cache: session_id -> SessionState
        self._sessions: dict[str, SessionState] = {}

        # Base cache pool: system_hash -> BaseCacheEntry
        self._base_caches: dict[str, BaseCacheEntry] = {}

        # Dirty session tracking for idle-time disk save
        self._dirty_sessions: set[str] = set()
        self._dirty_lock = threading.Lock()

        # PERF: deferred drafter-stats finalize. Set by ``_run_vlm`` when a
        # drafter is active so the per-token generator wrapper can be
        # removed from the hot path. ``_run``'s post-loop block invokes
        # this exactly once and resets it. VLM executor is single-worker,
        # so a per-engine stash is safe.
        self._pending_drafter_finalize = None

        # F3 architecture: pin ALL mlx-vlm calls to a single persistent worker
        # thread. mlx-vlm 0.5.0's module-global `generation_stream` is a
        # ThreadLocalStream created on the importing thread; if generation
        # runs on a different (or short-lived) thread, MLX raises
        # `RuntimeError: There is no Stream(gpu, N) in current thread.`
        # By dedicating one long-lived worker we register the stream ONCE
        # and every subsequent call inherits a consistent slot.
        # main_thread mode: do NOT create the worker executor. Generation,
        # load, and MTP-patch install all happen inline on this (main) thread.
        if self.execution_mode == "main_thread":
            self._vlm_executor = None
            self._vlm_worker_ready_event = threading.Event()
            self._vlm_worker_ready_event.set()
            MLXEngine._all_engines.append(self)
            return

        self._vlm_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-vlm-worker"
        )
        self._vlm_worker_ready_event = threading.Event()

        def _vlm_worker_init():
            # Runs on the dedicated worker thread.
            #
            # Install a worker-thread-local `generation_stream` on
            # `mlx_vlm.generate` and warm the per-thread slot. mlx-vlm
            # 0.5.0's module-global `generation_stream` is a
            # ThreadLocalStream created on the importing thread; if
            # generation runs on a different thread, MLX raises
            # `RuntimeError: There is no Stream(gpu, N) in current thread.`
            # Pinning load + inference to this worker (see F3-LOAD in
            # `load_model`) eliminates the hand-off; replacing the stream
            # here ensures every mlx-vlm call inside this worker uses a
            # slot we registered.
            try:
                import mlx_vlm.generate  # noqa: F401 — populate sys.modules
                _mvg = sys.modules["mlx_vlm.generate"]
                _old = getattr(_mvg, "generation_stream", None)
                _new = mx.new_thread_local_stream(mx.default_device())
                _mvg.generation_stream = _new

                # Pin a per-thread slot for `_new` on THIS thread.
                with mx.stream(_new):
                    _probe = mx.array([1.0]) * 1.0
                    mx.eval(_probe)

                # Install the Gemma 4 MTP wrap-around patches once mlx-vlm
                # is loaded. Idempotency-guarded so multiple engines / re-
                # inits don't double-wrap.
                _install_mtp_wrap_patches()

                logger.debug(
                    f"[F3-INIT] thread={threading.current_thread().name} "
                    f"old_stream={_old!r} new_stream={_new!r}"
                )
            finally:
                self._vlm_worker_ready_event.set()

        self._vlm_executor.submit(_vlm_worker_init)
        # Block until the worker has installed its stream — keeps the
        # one-time init synchronous from the engine's perspective.
        self._vlm_worker_ready_event.wait(timeout=10)

        MLXEngine._all_engines.append(self)

    def close(self):
        """Shut down the dedicated mlx-vlm worker. Idempotent."""
        ex = getattr(self, "_vlm_executor", None)
        if ex is not None:
            ex.shutdown(wait=True)
            self._vlm_executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — destructor must not raise
            pass

    def load_model(self):
        logger.info(f"Loading model: {self.cfg.model_path}")
        t0 = time.perf_counter()

        # main_thread mode: install the mlx-vlm thread-local generation_stream
        # + MTP wrap patches on THIS (main) thread before loading weights, so
        # that load + generation share a consistent per-thread MLX stream slot
        # (the same invariant the F3 worker provides on its dedicated thread).
        if self.execution_mode == "main_thread":
            try:
                import mlx_vlm.generate  # noqa: F401 — populate sys.modules
                _mvg = sys.modules["mlx_vlm.generate"]
                _new = mx.new_thread_local_stream(mx.default_device())
                _mvg.generation_stream = _new
                with mx.stream(_new):
                    _probe = mx.array([1.0]) * 1.0
                    mx.eval(_probe)
                _install_mtp_wrap_patches()
                logger.debug(
                    f"[MAIN-THREAD-INIT] thread={threading.current_thread().name} "
                    f"new_stream={_new!r}"
                )
            except Exception:  # noqa: BLE001 — mirror worker init best-effort
                logger.exception("[MAIN-THREAD-INIT] stream/patch install failed")

        # Detect model type from config.json
        model_config = {}
        config_path = os.path.join(self.cfg.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_config = json.load(f)
        self._model_type = model_config.get("model_type", "")

        # Check if mlx-vlm supports this model type BEFORE loading weights.
        # (mlx-vlm's load loads the entire safetensors before checking model type,
        #  which wastes memory for huge models like GLM-5.1 at 378GB.)
        #
        # F3-LOAD: load the VLM model on the dedicated worker thread, NOT
        # on whatever thread happens to call `load_model()`. mlx-vlm's
        # `load_model` calls `mx.eval(model.parameters())` to force the
        # weights off the lazy-load path; that eval runs on the thread
        # that invokes `vlm_load`, binding the weights' computation
        # graph to that thread's stream slots. If inference later runs
        # on a DIFFERENT thread (our worker), some lazy state — most
        # notably KV-cache buffers and any post-load model state — ends
        # up referencing stream slots the inference thread cannot
        # resolve, raising `Stream(gpu, N) in current thread`. Pinning
        # load + inference to the same worker eliminates the hand-off.
        self._use_vlm = False
        vlm_supported = self._vlm_supports(self._model_type)
        if vlm_supported:
            try:
                def _vlm_load_on_worker():
                    return vlm_load(self.cfg.model_path)

                if self.execution_mode == "main_thread":
                    self._vlm_model, self._processor = _vlm_load_on_worker()
                else:
                    self._vlm_model, self._processor = self._vlm_executor.submit(
                        _vlm_load_on_worker
                    ).result()
                self._language_model = getattr(
                    self._vlm_model, "language_model", self._vlm_model
                )
                self.tokenizer = getattr(self._processor, "tokenizer", self._processor)
                self._use_vlm = True
                logger.info("Loaded via mlx-vlm (on worker thread)")
            except Exception as e:
                logger.info(f"mlx-vlm load failed ({e}), falling back to mlx-lm")

        if not self._use_vlm:
            model, tokenizer = lm_load(self.cfg.model_path)
            self._vlm_model = model
            self._language_model = model
            self._processor = None
            self.tokenizer = tokenizer
            logger.info("Loaded via mlx-lm")

        elapsed = time.perf_counter() - t0

        # Derive model ID from directory name
        self.model_id = os.path.basename(self.cfg.model_path.rstrip("/"))
        logger.info(f"Model loaded in {elapsed:.1f}s — {self.model_id}")

        # Detect model family
        self.model_family = self._detect_model_family()
        logger.info(f"Model family: {self.model_family}")

        # Auto-detect thinking end token (needed for SSE thinking_done signal)
        self._detect_special_tokens()

        # Detect cache layer types (for logging/diagnostics)
        test_cache = make_prompt_cache(self._language_model)
        self._has_rotating_cache = any(
            type(c).__name__ == "RotatingKVCache" for c in test_cache
        )
        self._sliding_window_size = 0
        if self._has_rotating_cache:
            for c in test_cache:
                if type(c).__name__ == "RotatingKVCache" and hasattr(c, "max_size"):
                    self._sliding_window_size = c.max_size
                    break
        del test_cache
        if self._has_rotating_cache:
            logger.info(
                f"[{self.model_id}] RotatingKVCache detected — "
                f"sliding_window={self._sliding_window_size}"
            )

        if self.cfg.think_end_token < 0 and self.cfg.enable_thinking:
            self.cfg.enable_thinking = False
            logger.info(
                f"[{self.model_id}] think_end token not found — auto-disabled thinking"
            )

        logger.debug(
            f"[{self.model_id}] enable_thinking={self.cfg.enable_thinking}"
        )

        if self.cfg.prefill_step_size != 2048:
            logger.info(
                f"[{self.model_id}] Prefill step size: {self.cfg.prefill_step_size} "
                f"(default 2048)"
            )
        if self.cfg.kv_bits and not self._use_vlm:
            logger.info(
                f"[{self.model_id}] KV cache quantization: "
                f"bits={self.cfg.kv_bits}, group_size={self.cfg.kv_group_size}, "
                f"start={self.cfg.quantized_kv_start}"
            )
        elif self.cfg.kv_bits and self._use_vlm:
            logger.warning(
                f"[{self.model_id}] --kv-bits ignored (mlx-vlm path — quantization "
                f"not supported; only mlx-lm fallback models support this)"
            )

        if self.cfg.pld_enabled:
            if self._use_vlm:
                logger.warning(
                    f"[{self.model_id}] --pld ignored (mlx-vlm path — PLD "
                    f"not supported; falling back to standard VLM generation)"
                )
            else:
                logger.info(
                    f"[{self.model_id}] PLD enabled: "
                    f"num_draft={self.cfg.pld_num_draft_tokens}, k={self.cfg.pld_ngram_k}"
                )

        # Drafter weight is cached on ``self._drafter`` for per-request reuse;
        # mlx-lm legacy path has no drafter integration so we refuse loudly.
        #
        # F3-LOAD: the drafter is loaded **on the dedicated VLM worker
        # thread**. mlx-vlm 0.5.0 schedules drafter ops via the same
        # `mx.async_eval(...)` outside-of-with-block calls in
        # `generate_step` that previously raised the user's:
        #     RuntimeError: There is no Stream(gpu, N) in current thread.
        # When `--draft-model` is OFF, mlx-vlm's chunked-prefill block
        # does `mx.eval([c.state for c in prompt_cache])` *inside* the
        # `with mx.stream(generation_stream):` context — that incidental
        # eval keeps any lazy drafter state thread-portable. Drafter ON
        # disables chunked-prefill (`prefill_step_size = None` at
        # generate.py:1223), so any drafter weight evaluated on a
        # different thread from the inference loop's worker fails.
        # Loading on the worker eliminates the cross-thread hand-off.
        self._drafter = None
        self._draft_kind = None
        if self.cfg.draft_model:
            if not self._use_vlm:
                raise RuntimeError(
                    f"--draft-model is not supported on the mlx-lm legacy "
                    f"path (model_type={self._model_type!r}). Speculative "
                    f"decoding (MTP / DFlash) requires the mlx-vlm backend; "
                    f"this model_type is not in mlx-vlm's registry. Drop "
                    f"--draft-model or pick an mlx-vlm-supported target."
                )

            def _load_drafter_on_worker():
                return _maybe_load_drafter(
                    self.cfg.draft_model,
                    kind=self.cfg.draft_kind,
                )

            if self.execution_mode == "main_thread":
                self._drafter, self._draft_kind = _load_drafter_on_worker()
            else:
                self._drafter, self._draft_kind = self._vlm_executor.submit(
                    _load_drafter_on_worker
                ).result()

        # Set wired limit once at startup
        if mx.metal.is_available():
            max_rec = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(max_rec)
            logger.debug(f"Metal wired limit set to {max_rec / 1e9:.1f}GB")

            # Patch wired_limit in mlx_vlm: keep synchronize but skip set/reset cycle.
            # The set/reset cycle degrades Metal TTFT on repeated calls.
            import mlx_vlm.generate as vlm_gen_module

            @contextlib.contextmanager
            def _stable_wired_limit(model, streams=None):
                try:
                    yield
                finally:
                    if streams:
                        for s in streams:
                            mx.synchronize(s)
                    else:
                        mx.synchronize()

            vlm_gen_module.wired_limit = _stable_wired_limit
            logger.debug("Patched wired_limit: stable (set once at startup)")

        self._build_disk_index()
        self._touch_gpu()  # Mark GPU active after model load
        if self.cfg.gpu_keepalive and self.execution_mode != "main_thread":
            self._start_gpu_keepalive()
            logger.info(f"[{self.model_id}] GPU keepalive enabled (interval={self.GPU_KEEPALIVE_INTERVAL}s)")
        elif self.cfg.gpu_keepalive and self.execution_mode == "main_thread":
            # No background thread may touch MLX cache tensors in main-thread
            # mode (codex constraint). The keepalive flush is skipped entirely.
            logger.info(
                f"[{self.model_id}] GPU keepalive DISABLED (main_thread mode — "
                f"no background MLX access permitted)"
            )

    # --- Model detection helpers ---

    @staticmethod
    def _vlm_supports(model_type: str) -> bool:
        """Check if mlx-vlm has a model module for this model_type.

        Done BEFORE calling vlm_load to avoid loading huge weights
        just to fail on the model-type check (mlx-vlm currently loads
        all safetensors before checking model support).
        """
        if not model_type:
            return False
        try:
            import importlib
            importlib.import_module(f"mlx_vlm.models.{model_type}")
            return True
        except ImportError:
            return False

    def _detect_model_family(self) -> str:
        """Detect model family from model_type in config.json."""
        mt = self._model_type.lower()
        if "gemma4" in mt:
            return "gemma4"
        if "glm" in mt:
            return "glm"
        # Default: ChatML family (Qwen, MiniMax, etc.)
        return "chatml"

    def _detect_special_tokens(self):
        """Detect thinking end token for SSE thinking_done signal."""
        if self.model_family == "gemma4":
            self.cfg.think_end_token = _detect_token_id(self.tokenizer, "<channel|>")
        else:
            # ChatML and GLM both use </think>
            if self.cfg.think_end_token < 0:
                self.cfg.think_end_token = _detect_token_id(self.tokenizer, "</think>")

        logger.info(
            f"[{self.model_id}] model_family={self.model_family} | "
            f"think_end_token={self.cfg.think_end_token}"
        )

    # --- GPU keepalive ---

    def _start_gpu_keepalive(self):
        """Start background thread that keeps Metal GPU warm (once globally)."""
        if not mx.metal.is_available():
            return
        if MLXEngine._global_keepalive_started:
            return
        MLXEngine._global_keepalive_started = True

        self._keepalive_ping_count = 0

        def _keepalive_loop():
            logger.debug("[GPU Keepalive] Started (interval=%.1fs)", self.GPU_KEEPALIVE_INTERVAL)
            while not MLXEngine._global_keepalive_stop.wait(self.GPU_KEEPALIVE_INTERVAL):
                idle = time.time() - MLXEngine._global_last_gpu_activity
                if idle >= self.GPU_KEEPALIVE_INTERVAL:
                    if self._lock.acquire(blocking=False):
                        try:
                            t0 = time.perf_counter()
                            a = mx.random.normal((32, 32))
                            b = a @ a
                            mx.eval(b)
                            elapsed = (time.perf_counter() - t0) * 1000
                            self._keepalive_ping_count += 1
                            if self._keepalive_ping_count % 100 == 1 or elapsed > 100:
                                logger.info(
                                    f"[GPU Keepalive] ping #{self._keepalive_ping_count} "
                                    f"idle={idle:.1f}s, elapsed={elapsed:.0f}ms"
                                )
                            # Flush dirty sessions while GPU is idle and we hold the lock
                            for engine in MLXEngine._all_engines:
                                engine._flush_dirty_sessions()
                        except Exception as e:
                            logger.warning(f"[GPU Keepalive] ping failed: {e}")
                        finally:
                            self._lock.release()

        self._keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
        self._keepalive_thread.start()

        # Register shutdown handler to stop keepalive cleanly
        import atexit
        import signal

        _shutdown_done = False

        def _shutdown(*args):
            nonlocal _shutdown_done
            if _shutdown_done:
                return
            _shutdown_done = True
            MLXEngine._global_keepalive_stop.set()
            logger.info("[Shutdown] Flushing dirty sessions...")
            MLXEngine._flush_all_on_shutdown()
            logger.info("[Shutdown] Complete")

        atexit.register(_shutdown)
        # Handle SIGINT/SIGTERM so Ctrl+C stops keepalive immediately
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _signal_handler(signum, frame):
            _shutdown()
            # Restore default handler so next Ctrl+C forces quit
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    def _touch_gpu(self):
        """Mark GPU as recently active (resets keepalive timer)."""
        MLXEngine._global_last_gpu_activity = time.time()

    # --- Disk cache persistence ---

    def _session_cache_path(self, session_id: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"session_{session_id}.safetensors")

    def _save_session_to_disk(self, session_id: str, session: SessionState):
        """Save session's KV cache + token history to disk. Caller MUST hold _lock.

        Returns True on success, False if save is not possible (e.g. empty arrays).
        """
        if session.cache_state is None or session.cache_state.cache is None:
            return True
        t0 = time.perf_counter()
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = self._session_cache_path(session_id)
        metadata = {
            "session_id": session_id,
            "messages": json.dumps(session.messages, ensure_ascii=False),
            "total_cache_tokens": str(session.total_cache_tokens),
            "last_used": str(session.last_used),
            "token_ids": json.dumps(session.cache_state.token_ids or []),
        }
        def _do_save():
            # WHY: VLM KV-cache tensors are lazy and bound to the
            # _vlm_executor worker thread's generation_stream (post-gen
            # _eval_cache is skipped on the VLM path for perf, see F3 below).
            # Materializing AND serializing must therefore happen on that
            # same worker thread, or mx.eval raises "There is no Stream(gpu, 1)
            # in current thread." _eval_cache covers keys/values AND ArraysCache
            # .state arrays (DeltaNet recurrent state). On the legacy mlx-lm
            # path this runs inline on the request/flush thread (default stream).
            MLXEngine._eval_cache(session.cache_state.cache)
            save_prompt_cache(path, session.cache_state.cache, metadata=metadata)

        try:
            if getattr(self, "execution_mode", "worker") == "main_thread":
                # main_thread mode: model + cache tensors are bound to THIS
                # thread's stream slot. Materialize + serialize inline.
                _do_save()
            elif getattr(self, "_use_vlm", False) and getattr(self, "_vlm_executor", None) is not None:
                try:
                    fut = self._vlm_executor.submit(_do_save)
                except RuntimeError:
                    # Executor already shut down (can happen during
                    # _flush_all_on_shutdown). Best-effort inline save.
                    _do_save()
                else:
                    # Exceptions raised inside _do_save propagate out of
                    # .result(), so the surrounding except still classifies
                    # them (empty-array permanent skip vs unexpected re-raise).
                    fut.result(timeout=60)
            else:
                _do_save()
        except Exception as e:
            if "empty array" in str(e).lower() or "cannot serialize" in str(e).lower():
                # Some models (GLM MoE) have empty arrays that safetensors can't handle
                logger.info(
                    f"[KV Cache] session={session_id} | DISK SAVE SKIP | "
                    f"cache not serializable: {e}"
                )
                return False  # permanent failure, don't retry
            raise  # re-raise unexpected errors

        elapsed = time.perf_counter() - t0
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.add(session_id)
        fsize = os.path.getsize(path) / 1e6
        logger.info(
            f"[KV Cache] session={session_id} | DISK SAVE | "
            f"{session.total_cache_tokens} tokens, {len(session.messages)} msgs, "
            f"{fsize:.1f}MB, {elapsed:.2f}s"
        )
        # LRU eviction: keep total disk usage under budget
        self._evict_disk_sessions_if_needed(protect_session_id=session_id)
        return True

    def _evict_disk_sessions_if_needed(self, protect_session_id: str | None = None):
        """Scan cache_dir and delete oldest session files if disk usage exceeds budget.

        Protects:
        - The session we just saved (protect_session_id)
        - Any session currently in self._sessions (in-memory, active)
        """
        budget_bytes = int(self.cfg.disk_budget_gb * 1e9)
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        # Gather session file info: path, size, mtime, session_id
        entries = []
        total_size = 0
        for fname in os.listdir(cache_dir):
            if not fname.startswith("session_") or not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(cache_dir, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            total_size += st.st_size
            # Extract session_id: session_<id>.safetensors or session_<id>_ckpt.safetensors
            sid_part = fname[len("session_"):-len(".safetensors")]
            sid = sid_part[:-len("_ckpt")] if sid_part.endswith("_ckpt") else sid_part
            entries.append((st.st_mtime, st.st_size, fpath, sid))

        if total_size <= budget_bytes:
            return

        # Sort oldest first
        entries.sort(key=lambda e: e[0])
        protected = set(self._sessions.keys())
        if protect_session_id:
            protected.add(protect_session_id)

        deleted = 0
        freed = 0
        for mtime, size, fpath, sid in entries:
            if total_size <= budget_bytes:
                break
            if sid in protected:
                continue
            try:
                os.remove(fpath)
                total_size -= size
                freed += size
                deleted += 1
                if hasattr(self, "_disk_session_ids"):
                    self._disk_session_ids.discard(sid)
            except OSError as e:
                logger.debug(f"[Disk LRU] failed to delete {fpath}: {e}")

        if deleted:
            logger.info(
                f"[Disk LRU] evicted {deleted} files, freed {freed/1e9:.2f} GB "
                f"(total now {total_size/1e9:.2f}/{budget_bytes/1e9:.2f} GB)"
            )

    def _mark_dirty(self, session_id: str):
        """Mark a session for disk save on next idle cycle."""
        with self._dirty_lock:
            self._dirty_sessions.add(session_id)

    def _flush_dirty_sessions(self):
        """Flush all dirty sessions to disk. Caller MUST hold _lock."""
        with self._dirty_lock:
            to_save = self._dirty_sessions.copy()
            self._dirty_sessions.clear()

        if not to_save:
            return

        saved = 0
        for sid in to_save:
            session = self._sessions.get(sid)
            if session is None:
                continue
            try:
                success = self._save_session_to_disk(sid, session)
                if success:
                    saved += 1
                # If success=False (permanent failure like empty arrays), don't retry
            except Exception as e:
                logger.error(f"[KV Cache] session={sid} | FLUSH SAVE FAILED | {e}")
                with self._dirty_lock:
                    if sid in self._sessions:
                        self._dirty_sessions.add(sid)

        if saved:
            logger.info(f"[Idle Flush] saved {saved}/{len(to_save)} dirty sessions")

    @classmethod
    def _flush_all_on_shutdown(cls):
        """Save all dirty sessions across all engines on shutdown."""
        for engine in cls._all_engines:
            with engine._dirty_lock:
                to_save = engine._dirty_sessions.copy()
                engine._dirty_sessions.clear()
            if not to_save:
                continue
            logger.info(f"[Shutdown] Flushing {len(to_save)} dirty sessions for {engine.model_id}")
            with engine._lock:
                for sid in to_save:
                    session = engine._sessions.get(sid)
                    if session:
                        try:
                            engine._save_session_to_disk(sid, session)
                        except Exception as e:
                            logger.error(f"[Shutdown] Failed to save session {sid}: {e}")

    def _load_session_from_disk(self, session_id: str) -> Optional[SessionState]:
        """Load session's KV cache + token history from disk."""
        path = self._session_cache_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            t0 = time.perf_counter()
            cache, metadata = load_prompt_cache(path, return_metadata=True)
            messages = json.loads(metadata.get("messages", "[]"))
            total_tokens = int(metadata.get("total_cache_tokens", "0"))
            last_used = float(metadata.get("last_used", "0"))
            token_ids = json.loads(metadata.get("token_ids", "[]"))

            # Verify loaded cache matches model structure
            model_cache = make_prompt_cache(self._language_model)
            if len(cache) != len(model_cache):
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                    f"layer count mismatch: {len(cache)} vs {len(model_cache)}"
                )
                return None

            type_ok = all(
                type(c).__name__ == type(m).__name__
                for c, m in zip(cache, model_cache)
            )
            if not type_ok:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | cache type mismatch"
                )
                return None

            loaded_offset = self._get_cache_offset(cache)
            elapsed = time.perf_counter() - t0

            # Reconstruct PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = cache
            cache_state.token_ids = token_ids if token_ids else None

            session = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=loaded_offset,
                last_used=last_used,
            )

            fsize = os.path.getsize(path) / 1e6
            logger.info(
                f"[KV Cache] session={session_id} | DISK LOAD | "
                f"{loaded_offset} tokens, {len(messages)} msgs, "
                f"{fsize:.1f}MB, {elapsed:.2f}s"
            )
            return session
        except Exception as e:
            logger.error(f"[KV Cache] session={session_id} | DISK LOAD FAILED | {e}")
            return None

    def _build_disk_index(self):
        """Scan cache_dir for saved session caches."""
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        self._disk_session_ids: set[str] = set()
        count = 0
        for fname in os.listdir(cache_dir):
            if fname.startswith("session_") and fname.endswith(".safetensors") and "_ckpt" not in fname:
                sid = fname[len("session_"):-len(".safetensors")]
                self._disk_session_ids.add(sid)
                count += 1

        if count:
            logger.debug(f"[KV Cache] Disk index: {count} saved session caches")
        else:
            logger.debug("[KV Cache] Disk index: no saved session caches")

    def _has_disk_cache(self, session_id: str) -> bool:
        return hasattr(self, "_disk_session_ids") and session_id in self._disk_session_ids

    # --- Base cache pool ---

    @staticmethod
    def _system_hash(messages: list[dict], tools: list | None = None) -> str | None:
        """Hash the first system message (+ tools) for base cache lookup.

        Returns a hash even when there's no explicit system message — uses empty
        string as content. This supports models where the template auto-generates
        a system prefix (e.g. Gemma 4).
        """
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content", "")
        else:
            # No explicit system message — use empty content as hash key
            # (template may still generate a system prefix)
            content = ""
        h = hashlib.sha256(content.encode())
        if tools:
            h.update(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    def _find_base_cache(self, messages: list[dict], tools: list | None = None) -> BaseCacheEntry | None:
        """Find a matching base cache for the given messages' system prompt."""
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return self._base_caches[h]
        return None

    def _extract_system_tokens(
        self, messages: list[dict], full_tokens: list[int],
        tools: list | None = None, thinking: bool | None = None,
    ) -> list[int] | None:
        """Extract system prompt tokens from the full tokenized messages.

        Tokenizes [system + dummy user] then subtracts a [dummy user only]
        tokenization to get pure system tokens. Verifies they are a prefix
        of full_tokens.
        """
        has_system = messages and messages[0].get("role") in ("system", "developer")
        if not has_system and not self._has_rotating_cache:
            return None
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return None  # already registered
        try:
            enable_thinking = thinking if thinking is not None else self.cfg.enable_thinking
            if has_system:
                system_with_dummy = [messages[0], {"role": "user", "content": "hi"}]
            else:
                system_with_dummy = [{"role": "user", "content": "hi"}]
            # Tokenize system + dummy user
            full_with_dummy = self._tokenize_prompt(
                system_with_dummy, tools=tools, thinking=enable_thinking,
            )
            # Tokenize just dummy user (to strip)
            dummy_only = self._tokenize_prompt(
                [{"role": "user", "content": "hi"}], tools=tools, thinking=enable_thinking,
            )
            # System tokens = full_with_dummy minus the dummy user suffix
            # Heuristic: find where full_with_dummy diverges from dummy_only (from end)
            # Simpler: system tokens are the leading tokens that differ from dummy_only
            if has_system and len(full_with_dummy) > len(dummy_only):
                system_tokens = full_with_dummy[: len(full_with_dummy) - len(dummy_only)]
                # For models with auto-system prefix, adjust
                if not system_tokens:
                    return None
                # Verify prefix of full tokens
                if full_tokens[:len(system_tokens)] == system_tokens:
                    return system_tokens
            elif not has_system and len(full_with_dummy) > 0:
                # Models like Gemma 4 may auto-generate system prefix
                # The full_with_dummy == dummy_only in this case, check if there's
                # a shared prefix with the full conversation tokens
                return None
        except Exception as e:
            logger.warning(f"[Base Cache] Failed to extract system tokens: {e}")
        return None

    def _register_base_cache(
        self, messages: list[dict], cache: list, system_tokens: list[int], tools: list | None = None
    ):
        """Register a base cache from the system prompt portion of a processed cache."""
        h = self._system_hash(messages, tools=tools)
        if not h or h in self._base_caches:
            return
        # Deep copy the cache at current state (after system prompt processing)
        import copy
        base_snapshot = copy.deepcopy(cache)
        self._eval_cache(base_snapshot)
        entry = BaseCacheEntry(
            system_hash=h,
            cache=base_snapshot,
            tokens=system_tokens,
            token_count=len(system_tokens),
        )
        self._base_caches[h] = entry
        logger.debug(
            f"[Base Cache] REGISTERED | hash={h} | {len(system_tokens)} tokens | "
            f"pool_size={len(self._base_caches)}"
        )

    def _clone_base_cache(self, base: BaseCacheEntry) -> list:
        """Clone a base cache for a new session."""
        import copy
        cloned = copy.deepcopy(base.cache)
        # Force evaluation of cloned arrays to avoid lazy-eval aliasing issues
        self._eval_cache(cloned)
        base.hit_count += 1
        logger.debug(
            f"[Base Cache] CLONE | hash={base.system_hash} | "
            f"{base.token_count} tokens | hits={base.hit_count}"
        )
        return cloned

    def base_cache_stats(self) -> list[dict]:
        """Return stats for all base caches."""
        return [
            {
                "system_hash": e.system_hash,
                "token_count": e.token_count,
                "hit_count": e.hit_count,
                "created": e.created,
            }
            for e in self._base_caches.values()
        ]

    def _will_wrap_during_generate(self, prompt_token_ids, cache_state) -> bool:
        """True iff serving this request would cross the RotatingKVCache
        sliding-window boundary (ring-buffer wrap).

        Used as a Layer-A safety net in ``_run_vlm`` to bypass the
        speculative drafter on requests where the mlx-vlm Gemma 4 MTP
        wrap-around bugs (B1/B2-v2) would otherwise tank acceptance.

        ``cache_state.cache[0].offset`` is the logical-cumulative token
        count; ``len(prompt_token_ids)`` is the bytes-to-process
        post-prefix-trim. Their sum compared against
        ``self._sliding_window_size`` predicts whether the next
        ``update_and_fetch`` calls will exceed the ring capacity.
        """
        if not getattr(self, "_has_rotating_cache", False):
            return False
        win = getattr(self, "_sliding_window_size", 0) or 0
        if win <= 0:
            return False
        cur_offset = 0
        if cache_state is not None:
            cache = getattr(cache_state, "cache", None)
            if cache:
                first = cache[0]
                cur_offset = int(getattr(first, "offset", 0) or 0)
        return (cur_offset + len(prompt_token_ids or [])) >= win

    @staticmethod
    def _safe_to_reuse_cache(cache_state, prompt_token_ids=None) -> bool:
        """Return whether the prior-turn KV cache may be reused for the new
        prompt, given the new prompt's token ids.

        The danger is a RotatingKVCache (sliding-window attention, e.g.
        gemma4's 50 sliding layers) whose internal ring buffer has wrapped
        (offset > max_size). Once wrapped the physical buffer holds only the
        most-recent ``max_size`` tokens — it no longer corresponds to a
        contiguous *prefix* of the logical history, so prefix-trim based
        reuse on a DIVERGENT prompt (branch/edit past the wrap) would
        silently mis-align KV entries with the new prompt's tokens.

        However, for a STRICT APPEND — where the entire cached logical
        history is a prefix of the new prompt — mlx-vlm processes only the
        suffix against the wrapped cache, and RotatingKVCache._update_concat
        temporal-orders + trims-to-window + appends correctly. There is no
        mis-aligned slice because the logical prefix length (>> the physical
        buffer) drives the trim. So append-only wrapped reuse is SAFE; only
        genuine divergence past the wrap must cold-fill.

        Reference: `mlx_lm/models/cache.py::RotatingKVCache` exposes
        `.offset` (cumulative tokens seen) and `.max_size` (ring capacity).
        Non-rotating caches (KVCache, ArraysCache, ...) and non-wrapped
        rotating caches are always safe.

        ``prompt_token_ids`` defaults to None: callers that cannot supply it
        get the conservative pre-existing behavior (wrapped → cold-fill).

        Empty / None cache lists return True (nothing to gate).
        """
        if cache_state is None:
            return True
        cache = getattr(cache_state, "cache", None)
        if not cache:
            return True

        has_wrapped_rotating = False
        for c in cache:
            if type(c).__name__ != "RotatingKVCache":
                continue
            max_size = getattr(c, "max_size", None)
            offset = getattr(c, "offset", None)
            if max_size is None or offset is None:
                return False
            if offset > max_size:
                has_wrapped_rotating = True

        if not has_wrapped_rotating:
            return True  # non-wrapped (or no rotating cache) — unchanged behavior

        # Wrapped: only safe to reuse when the ENTIRE cached logical history is a
        # strict prefix of the new prompt (pure append). Any divergence/branch must
        # cold-fill — the rotating ring holds the old tail, not the new prefix's window.
        cached_ids = getattr(cache_state, "token_ids", None)
        if not cached_ids or prompt_token_ids is None:
            return False
        prefix_len = cache_state.find_prefix_length(prompt_token_ids)
        if prefix_len != len(cached_ids) or prefix_len >= len(prompt_token_ids):
            return False
        # Defensive: RoPE continuation needs cache.offset == logical history length.
        for c in cache:
            offset = getattr(c, "offset", None)
            if offset is not None and int(offset) != prefix_len:
                return False
        return True

    @staticmethod
    def _get_cache_offset(cache: list) -> int:
        """Get the total number of tokens processed by this cache.

        Prefers KVCache (full attention, accurate cumulative offset) over
        RotatingKVCache (offset is cumulative but size() caps at max_size).
        """
        # First pass: look for unbounded KVCache (full attention layers)
        for c in cache:
            if type(c).__name__ == "KVCache" and hasattr(c, "offset"):
                return c.offset
        # Fallback: any cache with offset (RotatingKVCache, ArraysCache, etc.)
        for c in cache:
            if hasattr(c, "offset"):
                return c.offset
        return 0

    @staticmethod
    def _eval_cache(cache: list):
        """Force evaluation of all lazy cache tensors."""
        arrays = []
        for c in cache:
            if hasattr(c, "keys") and c.keys is not None:
                items = c.keys if isinstance(c.keys, list) else [c.keys]
                arrays.extend(a for a in items if isinstance(a, mx.array))
            if hasattr(c, "values") and c.values is not None:
                items = c.values if isinstance(c.values, list) else [c.values]
                arrays.extend(a for a in items if isinstance(a, mx.array))
            if hasattr(c, "state") and c.state is not None:
                items = c.state if isinstance(c.state, list) else [c.state]
                arrays.extend(a for a in items if isinstance(a, mx.array))
        if arrays:
            mx.eval(*arrays)

    _PREFILL_STEP = 512

    def _prefill_cache(self, cache: list, tokens: list[int]):
        """Process tokens through the language model to populate a KV cache."""
        arr = mx.array(tokens)
        for i in range(0, len(tokens), self._PREFILL_STEP):
            chunk = arr[i : i + self._PREFILL_STEP]
            self._language_model(chunk[None], cache=cache)
        self._eval_cache(cache)

    @staticmethod
    def _is_compacted_tool(s_content: str, i_content: str) -> bool:
        """Check if either side is a compacted/cleared tool result placeholder."""
        for c in (s_content, i_content):
            if c.startswith("[") and ("cleared]" in c or "compacted:" in c):
                return True
        return False

    @staticmethod
    def _flatten_multipart(content) -> str:
        """Flatten OpenAI multi-part content to a plain string.

        Drops image/video parts and client-inserted
        "[image data removed ...]" placeholders so that a turn with an
        image blob and a subsequent turn where the client replaced the
        blob with a placeholder normalize to the same text.
        """
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype and ptype != "text":
                continue  # image, image_url, video, etc.
            txt = p.get("text", "") or ""
            if _NORMALIZE_RE_IMAGE_REMOVED.match(txt):
                continue
            parts.append(txt)
        return "\n".join(parts)

    @staticmethod
    def _normalize_for_match(content, role: str) -> str:
        """Normalize message content for comparison."""
        content = MLXEngine._flatten_multipart(content)
        if role == "system":
            # Normalize dynamic date (e.g. "Today's date: Tue Mar 10 2026" → placeholder)
            content = _NORMALIZE_RE_TODAYS_DATE.sub(
                "Today's date: __DATE__",
                content,
            )
        # Strip <system-reminder>...</system-reminder> tags injected dynamically by clients
        content = _NORMALIZE_RE_SYSTEM_REMINDER.sub(
            "",
            content,
        )
        # Strip thinking and tool calls from assistant messages for comparison.
        # Only the actual text content matters for cache matching.
        if role == "assistant":
            # Strip thinking blocks
            if "<channel|>" in content:
                content = content[content.rindex("<channel|>") + len("<channel|>"):]
            elif "</think>" in content:
                content = content[content.rindex("</think>") + len("</think>"):]
            else:
                content = _NORMALIZE_RE_THINK_PREFIX.sub("", content)
                content = _NORMALIZE_RE_CHANNEL_THOUGHT_PREFIX.sub("", content)
                content = _NORMALIZE_RE_THOUGHT_PREFIX.sub("", content)
            # Strip tool call blocks (both ChatML and Gemma 4 formats)
            content = _NORMALIZE_RE_TOOL_CALL_CHANNEL.sub("", content)
            content = _NORMALIZE_RE_TOOL_CALL_XML.sub("", content)
        return content.strip()

    def _messages_match(self, stored: list[dict], incoming: list[dict]) -> bool:
        """Check if incoming messages start with the stored conversation."""
        if len(incoming) < len(stored):
            logger.debug(
                f"[Match] FAIL: incoming({len(incoming)}) < stored({len(stored)})"
            )
            return False
        for i, s_msg in enumerate(stored):
            i_msg = incoming[i]
            if s_msg.get("role") != i_msg.get("role"):
                logger.info(
                    f"[Match] FAIL at msg[{i}]: role {s_msg.get('role')!r} != {i_msg.get('role')!r}"
                )
                return False
            s_content = self._flatten_multipart(s_msg.get("content"))
            i_content = self._flatten_multipart(i_msg.get("content"))
            role = s_msg.get("role", "")

            # Assistant tool_call messages: OpenCode strips <tool_call> from content
            # and moves it to tool_calls field. Handle all cases:
            # 1. stored="<tool_call>..." vs incoming="" (pure tool call)
            # 2. stored="text\n\n<tool_call>..." vs incoming="text" (text + tool call)
            if role == "assistant" and s_content != i_content:
                s_stripped = _NORMALIZE_RE_TOOL_CALL_SPLIT.split(s_content, maxsplit=1)[0].rstrip()
                i_stripped = _NORMALIZE_RE_TOOL_CALL_SPLIT.split(i_content, maxsplit=1)[0].rstrip()
                if s_stripped == i_stripped:
                    logger.debug(
                        f"[Match] msg[{i}] assistant tool_call content mismatch ignored "
                        f"(stored_len={len(s_content)}, incoming_len={len(i_content)})"
                    )
                    continue

            # Normalize and compare
            s_norm = self._normalize_for_match(s_content, role)
            i_norm = self._normalize_for_match(i_content, role)
            if s_norm != i_norm:
                # Tool content compacted/cleared by client (either direction)
                # KV cache still valid — the tokens were already processed.
                if role == "tool" and self._is_compacted_tool(s_content, i_content):
                    logger.debug(
                        f"[Match] msg[{i}] tool content compacted — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Last stored assistant message: tolerate content difference.
                # Client may have received truncated/reformatted response
                # (disconnect, streaming, client-side processing).
                # KV cache is still valid because it was saved from full generation.
                if role == "assistant" and i == len(stored) - 1:
                    logger.debug(
                        f"[Match] msg[{i}] assistant content mismatch at last stored msg — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Client may reconstruct assistant content as "{thinking}\n\n{final}"
                # without <think> tags. Stored normalized is just "{final}"; incoming
                # normalized is "{thinking}\n\n{final}". KV cache reflects what the
                # model actually processed (stored), so if the final answer matches
                # as a suffix of the incoming, the cache is still valid.
                if role == "assistant" and s_norm and len(s_norm) >= 8 and (
                    i_norm.endswith(s_norm) or s_norm.endswith(i_norm)
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant content suffix match — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Assistant tool_call turn: stored was (optional <think>...</think>
                # + <tool_call>...</tool_call>), which normalizes to empty because
                # both blocks are stripped. OpenAI-format clients move the tool
                # call to tool_calls[] and may reconstruct the thinking as plain
                # text in content (no <think> tags). KV cache is still valid —
                # it reflects the tokens the model actually emitted.
                if (
                    role == "assistant"
                    and not s_norm
                    and i_msg.get("tool_calls")
                    and ("<tool_call>" in s_content or s_msg.get("tool_calls"))
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant tool_call turn — "
                        f"accepting reconstructed content "
                        f"(stored_len={len(s_content)}, incoming_len={len(i_content)})"
                    )
                    continue

                # Find exact divergence point
                diff_pos = next(
                    (j for j in range(min(len(s_norm), len(i_norm))) if s_norm[j] != i_norm[j]),
                    min(len(s_norm), len(i_norm)),
                )
                s_ctx = s_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                i_ctx = i_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                logger.info(
                    f"[Match] FAIL at msg[{i}] role={role}: "
                    f"stored_len={len(s_content)} vs incoming_len={len(i_content)} | "
                    f"diff at char {diff_pos} | "
                    f"stored=...{s_ctx!r}... | incoming=...{i_ctx!r}..."
                )
                return False
        return True

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages for chat template: fix roles, flatten content."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            if role == "developer":
                role = "system"
            m = {"role": role}
            if msg.get("content") is not None:
                content = msg["content"]
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(part["text"])
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join(parts)
                m["content"] = content
            if msg.get("tool_calls"):
                normalized_tcs = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc_copy, dict) and "function" in tc_copy:
                        fn = dict(tc_copy["function"])
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except (json.JSONDecodeError, ValueError):
                                fn["arguments"] = {}
                        tc_copy["function"] = fn
                    normalized_tcs.append(tc_copy)
                m["tool_calls"] = normalized_tcs
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            formatted.append(m)
        return formatted

    def _build_prompt_text(
        self, messages: list[dict], thinking: bool = True, tools: list | None = None,
    ) -> str:
        """Build prompt text from messages using chat template (tokenize=False)."""
        formatted = self._format_messages(messages)
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": thinking,
        }
        if tools:
            kwargs["tools"] = [
                t.model_dump() if hasattr(t, "model_dump") else t for t in tools
            ]
        return self.tokenizer.apply_chat_template(formatted, **kwargs)

    def _tokenize_prompt(
        self, messages: list[dict], thinking: bool = True, tools: list | None = None,
    ) -> list[int]:
        """Tokenize messages using chat template (tokenize=True)."""
        formatted = self._format_messages(messages)
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": thinking,
        }
        if tools:
            kwargs["tools"] = [
                t.model_dump() if hasattr(t, "model_dump") else t for t in tools
            ]
        result = self.tokenizer.apply_chat_template(formatted, **kwargs)
        # Some tokenizers return BatchEncoding instead of list[int]
        if hasattr(result, "input_ids"):
            return list(result.input_ids)
        return list(result)

    def _suffix_tokens(
        self, new_messages: list[dict], thinking: bool = True,
    ) -> list[int]:
        """Compute suffix tokens for new messages to append to stored token_ids.

        This avoids full re-tokenization (which breaks special token round-trip)
        by directly encoding only the new message suffix in model-specific format.
        """
        if self.model_family == "gemma4":
            return self._suffix_tokens_gemma4(new_messages, thinking)
        if self.model_family == "glm":
            return self._suffix_tokens_glm(new_messages, thinking)
        return self._suffix_tokens_chatml(new_messages, thinking)

    def _suffix_tokens_gemma4(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """Gemma 4 suffix: <turn|>\\n<|turn>user\\n{content}<turn|>\\n<|turn>model\\n"""
        parts = ["<turn|>"]
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue  # already in cache
            elif role == "tool":
                parts.append(
                    f"\n<|turn>user\n<|tool_response>\n"
                    f"response:{msg.get('name', '')}{{{content}}}\n"
                    f"<tool_response|><turn|>"
                )
            else:
                parts.append(f"\n<|turn>user\n{content}<turn|>")
        parts.append("\n<|turn>model\n")
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def _suffix_tokens_chatml(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """ChatML suffix: \\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"""
        parts = []
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue
            elif role == "tool":
                parts.append(
                    f"\n<|im_start|>user\n<tool_response>\n"
                    f"{content}\n</tool_response><|im_end|>"
                )
            else:
                parts.append(f"\n<|im_start|>user\n{content}<|im_end|>")
        gen_prompt = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        parts.append(gen_prompt)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def _suffix_tokens_glm(
        self, new_messages: list[dict], thinking: bool,
    ) -> list[int]:
        """GLM suffix: <|user|>{content}<|assistant|><think>"""
        parts = []
        for msg in new_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in content
                )
            if role == "assistant":
                continue
            elif role == "tool":
                parts.append(f"<|user|><tool_response>\n{content}\n</tool_response>")
            else:
                parts.append(f"<|user|>{content}")
        gen_prompt = "<|assistant|><think>" if thinking else "<|assistant|>"
        parts.append(gen_prompt)
        return self.tokenizer.encode("".join(parts), add_special_tokens=False)

    def generate_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        session_id: str | None = None,
        tools: list | None = None,
        cancel_event: threading.Event | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> Generator[GenerationResult, None, None]:
        """Generate with session-based KV cache reuse (holds lock)."""
        sid = session_id or "anon"
        t_wait = time.perf_counter()
        logger.debug(f"[Queue] session={sid} | waiting for lock | messages={len(messages)}")
        with self._lock:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            logger.debug(f"[Queue] session={sid} | lock acquired | waited={wait_ms:.0f}ms")
            yield GenerationResult(status="generating")
            yield from self._generate_locked(
                messages,
                max_tokens=max_tokens if max_tokens is not None else self.cfg.default_max_tokens,
                temperature=temperature if temperature is not None else self.cfg.default_temperature,
                top_p=top_p if top_p is not None else self.cfg.default_top_p,
                min_p=min_p if min_p is not None else self.cfg.default_min_p,
                top_k=top_k if top_k is not None else self.cfg.default_top_k,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cfg.default_repetition_penalty,
                session_id=session_id,
                tools=tools,
                cancel_event=cancel_event,
                thinking=thinking,
                thinking_budget=thinking_budget,
                response_format=response_format,
            )

    def _generate_locked(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        session_id: str | None,
        tools: list | None,
        cancel_event: threading.Event | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        response_format=None,
    ) -> Generator[GenerationResult, None, None]:
        """Core generation logic using mlx-vlm (must hold lock).

        Session messages include thinking in assistant content so that
        PromptCacheState prefix matching covers generated tokens from
        previous turns.
        """
        self._touch_gpu()

        has_tools = bool(tools)
        use_thinking = thinking if thinking is not None else self.cfg.enable_thinking

        if not session_id:
            session_id = "anon"
        session = self._sessions.get(session_id)

        # Try loading from disk if not in memory
        if not session and self._has_disk_cache(session_id):
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session

        cache_mode = "miss"

        # Determine prompt messages:
        # - On cache hit: use session's stored messages (with thinking) + new messages
        # - On cache miss: use incoming messages as-is
        prompt_messages = messages
        cache_state = PromptCacheState()

        if (
            session
            and session.cache_state is not None
            and session.cache_state.cache is not None
            and self._messages_match(session.messages, messages)
        ):
            new_messages = messages[len(session.messages):]
            if not new_messages:
                # Retry: discard cache, start fresh
                cache_mode = "retry"
                prompt_messages = messages
                cache_state = PromptCacheState()
                logger.info(
                    f"[KV Cache] session={session_id} | RETRY | "
                    f"discarding cache, re-processing {len(messages)} messages"
                )
            else:
                # Cache hit: extend stored token_ids with suffix for new messages.
                # This avoids full re-tokenization which breaks special token
                # round-trip (e.g. Gemma 4 <|channel>/<channel|>).
                cache_mode = "hit"
                cache_state = session.cache_state
                cached_tokens = session.total_cache_tokens
                suffix = self._suffix_tokens(new_messages, thinking=use_thinking)
                prompt_token_ids = list(cache_state.token_ids or []) + suffix
                logger.info(
                    f"[KV Cache] session={session_id} | HIT | "
                    f"reusing {cached_tokens} cached tokens + "
                    f"{len(suffix)} suffix tokens"
                )
        else:
            # Cache miss — seed from base cache if available
            prev_cached = session.total_cache_tokens if session else 0
            base = self._find_base_cache(messages, tools=tools)
            if base:
                cache_state.cache = self._clone_base_cache(base)
                cache_state.token_ids = list(base.tokens)
                cache_mode = "base_hit"
                logger.info(
                    f"[KV Cache] session={session_id} | BASE HIT | "
                    f"seeding {base.token_count} base tokens "
                    f"(was {prev_cached} cached)"
                )
            else:
                if prev_cached:
                    logger.info(
                        f"[KV Cache] session={session_id} | MISS | "
                        f"discarding {prev_cached} cached tokens, "
                        f"processing {len(messages)} messages"
                    )
                else:
                    logger.info(
                        f"[KV Cache] session={session_id} | MISS (new) | "
                        f"processing {len(messages)} messages"
                    )

        # Tokenize prompt — only needed for MISS/RETRY (HIT already computed above)
        if cache_mode != "hit":
            prompt_token_ids = self._tokenize_prompt(
                prompt_messages, thinking=use_thinking, tools=tools,
            )
        total_prompt_tokens = len(prompt_token_ids)

        # Determine how many tokens will actually be processed (for cache info)
        if cache_state.token_ids:
            reused = cache_state.find_prefix_length(prompt_token_ids)
        else:
            reused = 0
        new_token_count = total_prompt_tokens - reused

        # Build logits processors
        logits_processors = []
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyProcessor(penalty=repetition_penalty))
        budget = thinking_budget if thinking_budget is not None else self.cfg.thinking_budget
        if use_thinking and budget > 0 and self.cfg.think_end_token >= 0:
            think_start = _detect_token_id(
                self.tokenizer,
                "<|channel>" if self.model_family == "gemma4" else "<think>",
            )
            logits_processors.append(ThinkingBudgetProcessor(
                budget=budget,
                think_end_token=self.cfg.think_end_token,
                think_start_token=think_start,
                model_family=self.model_family,
            ))
        # Structured output (response_format) via FSM-based logits masking.
        # Works on both mlx-vlm and mlx-lm paths (same logits_processors contract).
        # PLD is incompatible: speculative decoding advances multiple tokens
        # per step, breaking the FSM's single-step advance assumption.
        structured_proc = None
        rf_type = getattr(response_format, "type", None) if response_format else None
        if rf_type in ("json_schema", "json_object") and self.cfg.pld_enabled and not self._use_vlm:
            logger.warning(
                f"[Structured] response_format={rf_type} disabled: PLD is active "
                f"(speculative decoding is incompatible with FSM-based constraints). "
                f"Disable PLD to use structured output."
            )
        elif rf_type in ("json_schema", "json_object"):
            try:
                from mlx_soloheaven.engine.structured import (
                    build_json_schema_processor,
                    build_json_object_processor,
                )
                if rf_type == "json_schema":
                    js = response_format.json_schema
                    if js is None or js.schema_ is None:
                        raise ValueError("response_format.json_schema.schema is required")
                    structured_proc = build_json_schema_processor(
                        js.schema_, self.tokenizer,
                        cache_key=f"{js.name or 'anon'}:{hash(json.dumps(js.schema_, sort_keys=True))}",
                    )
                    logger.info(f"[Structured] json_schema active (name={js.name})")
                else:  # json_object
                    structured_proc = build_json_object_processor(self.tokenizer)
                    logger.info(f"[Structured] json_object active")
            except Exception as e:
                logger.warning(f"[Structured] failed to build processor: {e} — ignoring")
                structured_proc = None
        if structured_proc is not None:
            logits_processors.append(structured_proc)

        # Build response cache info
        response_cache_info = {
            "cache_mode": cache_mode,
            "cached_tokens": reused,
            "new_tokens": new_token_count,
            "total_prompt_tokens": total_prompt_tokens,
        }
        if session and session.pending_build_time > 0:
            response_cache_info["build_time"] = round(session.pending_build_time, 2)
            session.pending_build_time = 0.0

        sampling_info = f"temp={temperature}"
        if top_p < 1.0:
            sampling_info += f", top_p={top_p}"
        if min_p > 0.0:
            sampling_info += f", min_p={min_p}"
        if top_k > 0:
            sampling_info += f", top_k={top_k}"
        if repetition_penalty != 1.0:
            sampling_info += f", rep_pen={repetition_penalty}"
        logger.info(
            f"[Generate] prompt={new_token_count} new tokens "
            f"(reused={reused}, total={total_prompt_tokens}), "
            f"max={max_tokens}, {sampling_info}, cache_mode={cache_mode}"
        )

        # Stream generate — mlx-vlm for VLM models, mlx-lm for text models
        # PERF: accumulate token texts in a list and ``"".join`` once after
        # the loop. The previous ``accumulated_text += text`` triggers a
        # fresh string allocation/copy each token — O(N^2) over the run.
        text_parts: list[str] = []
        accumulated_text = ""
        t_gen_start = time.perf_counter()
        t_first_token = None
        last_prompt_tps = 0.0
        last_gen_tps = 0.0
        gen_token_count = 0
        cancelled = False
        # Capture every yielded token so post-loop can extend
        # cache_state.token_ids = prompt_token_ids + generated_token_ids on BOTH
        # paths. mlx-vlm's stream_generate mutates cache_state.cache in place but
        # does not append generated IDs to cache_state.token_ids — without this
        # capture, subsequent turns silently miss the generated-token prefix.
        generated_token_ids: list[int] = []
        # Snapshot of the full prompt IDs (pre any in-branch slicing) used by
        # the unified post-generation cache_state.token_ids update for both paths.
        full_prompt_token_ids = list(prompt_token_ids)

        # Shared sampler — used by the mlx-lm legacy path; the mlx-vlm path
        # consumes top_p/min_p/top_k/temp directly so it ignores `sampler`.
        # Constructing it unconditionally costs ~one closure allocation and
        # keeps the dispatcher signature uniform across paths.
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

        # Dispatch to the per-backend runner. The runner returns the
        # streaming generator AND (legacy path only) the local
        # `prompt_cache` reference needed for the post-loop write-back.
        gen_iter, prompt_cache = self._run_generate(
            cache_state=cache_state,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            sampler=sampler,
            logits_processors=logits_processors,
            session_id=session_id,
            total_prompt_tokens=total_prompt_tokens,
        )

        # Configurable progress log interval (tokens between INFO snapshots)
        progress_interval = 50  # tokens — every ~2s at 25 TPS
        # PERF: hoist hot-path lookups out of the per-token loop.
        # ``logger.isEnabledFor`` is cheap but called per token, and
        # the f-string in ``logger.debug(...)`` is evaluated even when
        # DEBUG is disabled — gate it explicitly.
        _debug_enabled = logger.isEnabledFor(logging.DEBUG)
        _logger = logger
        # PERF: deferred drafter-stats finalize, invoked exactly once after
        # the stream loop exits (normal end OR cancellation break). Replaces
        # the per-token ``_stream_ctx_wrapper`` generator frame that used to
        # cost ~30 tps at high TPS on the VLM speculative-decoding hot path.
        _drafter_finalize = getattr(self, "_pending_drafter_finalize", None)
        self._pending_drafter_finalize = None
        for resp in gen_iter:
            if cancel_event is not None and cancel_event.is_set():
                # Report last token state when cancelled so we can see
                # where generation was when the client disconnected.
                tail = ("".join(text_parts))[-200:].replace('\n', '\\n')
                _logger.info(
                    f"[Generate] session={session_id} | CANCELLED at token {gen_token_count} | "
                    f"last_tps={last_gen_tps:.1f} | tail={tail!r}"
                )
                cancelled = True
                break

            gen_token_count += 1
            text = resp.text if hasattr(resp, "text") else ""
            tok_attr = getattr(resp, "token", None)
            token = tok_attr if tok_attr is not None else 0
            prompt_tps = getattr(resp, "prompt_tps", 0.0) or 0.0
            gen_tps = getattr(resp, "generation_tps", 0.0) or 0.0

            # PERF: append-to-list + post-loop join avoids the O(N^2)
            # cost of repeated ``str += text`` (each += allocates a new
            # string and copies the entire accumulated buffer).
            text_parts.append(text)
            # Track yielded token IDs so the post-loop update keeps
            # cache_state.token_ids == full_prompt_token_ids + generated_token_ids
            # on both paths. resp.token may be None on the synthetic terminal
            # frame; in that case the token has already been counted earlier.
            if tok_attr is not None:
                generated_token_ids.append(int(tok_attr))

            if t_first_token is None:
                t_first_token = time.perf_counter()
                last_prompt_tps = prompt_tps
                _logger.info(
                    f"[Generate] TTFT={round((t_first_token - t_gen_start)*1000)}ms"
                )

            # PERF: guard the per-token DEBUG f-string so it's not built
            # unless DEBUG logging is actually enabled. The default --verbose
            # off case skips the f-string entirely.
            if _debug_enabled:
                _logger.debug(
                    f"[Token] session={session_id} | n={gen_token_count} id={token} text={text!r}"
                )

            # Periodic INFO snapshot (every 50 tokens) so we can see progress
            # when verbose is off
            if gen_token_count % progress_interval == 0:
                tail = ("".join(text_parts[-40:]))[-120:].replace('\n', '\\n')
                _logger.info(
                    f"[Generate] session={session_id} | "
                    f"tokens={gen_token_count} | tps={gen_tps:.1f} | "
                    f"tail={tail!r}"
                )

            last_gen_tps = gen_tps

            yield GenerationResult(
                text=text,
                token=token,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=gen_token_count,
                prompt_tps=prompt_tps,
                generation_tps=gen_tps,
            )

        # PERF: single join at end of loop — replaces O(N^2) accumulation.
        accumulated_text = "".join(text_parts)
        # PERF: drafter-stats finalize (post-stream, exactly once). Captures
        # all drafter.accept_lens accumulated during _mtp_rounds → log + per-
        # session bookkeeping. Replaces the per-token generator wrapper.
        if _drafter_finalize is not None:
            try:
                _drafter_finalize()
            except Exception:  # noqa: BLE001 — finalize must never break inference
                logger.exception("[Drafter] post-stream finalize failed")
        self._touch_gpu()

        # Unified post-generation cache_state update (both VLM and legacy
        # mlx-lm paths). mlx-vlm's stream_generate mutates cache_state.cache
        # in place but never appends generated IDs to cache_state.token_ids;
        # the legacy branch needs an explicit write-back of `prompt_cache` too.
        # Both paths converge on:
        #     cache_state.cache     = <post-generation cache>
        #     cache_state.token_ids = full_prompt_token_ids + generated_token_ids
        # Doing this even on cancellation keeps cache_state.token_ids consistent
        # with the in-place mutated cache_state.cache (the VLM path may have
        # already prefilled / advanced the cache before the cancel was observed).
        # Legacy mlx-lm path returns a local `prompt_cache` reference that
        # must be written back; VLM path mutates cache_state.cache in place
        # and `prompt_cache` is None — the conditional below is the single
        # remaining `self._use_vlm` branch in this method's post-loop and
        # is kept because the two paths legitimately diverge in WHERE the
        # post-generation KV cache lives.
        if prompt_cache is not None:
            cache_state.cache = prompt_cache
        cache_state.token_ids = full_prompt_token_ids + list(generated_token_ids)

        # WHY: F3 pins every mlx-vlm call to one persistent _vlm_executor
        # worker thread (model + drafter + generation_stream all bound to
        # that thread). Under F3 there is no cross-thread lazy-array
        # hazard — the post-gen _eval_cache here is redundant and the
        # forced full materialization adds measurable per-request overhead
        # (large KV tensor sync). We now skip it on the VLM path and only
        # keep it for the legacy mlx-lm path as a defensive no-op (mlx-lm
        # runs on the request thread; this is a cheap final eval).
        if (not self._use_vlm) and cache_state.cache is not None:
            try:
                self._eval_cache(cache_state.cache)
            except Exception as _eval_err:  # noqa: BLE001 — best-effort
                logger.warning(
                    f"[KV Cache] session={session_id} | "
                    f"post-gen _eval_cache failed: {_eval_err!r}"
                )

        # Log generated text for debugging
        if accumulated_text:
            preview = accumulated_text[:200].replace('\n', '\\n')
            logger.debug(
                f"[Generate] session={session_id} | "
                f"tokens={gen_token_count} | cancelled={cancelled} | "
                f"text={preview!r}"
            )

        if cancelled:
            return

        # Guard: detect empty response (no content after thinking)
        if accumulated_text and session_id:
            _, content = split_thinking_and_content(
                accumulated_text, model_family=self.model_family,
            )
            if not content or not content.strip():
                logger.warning(
                    f"[KV Cache] session={session_id} | SKIP SAVE | "
                    f"empty response ({gen_token_count} tokens, no content)"
                )
                yield GenerationResult(
                    text="",
                    finish_reason="stop",
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=gen_token_count,
                )
                return

        # Parse tool_calls once — used both for session persistence and
        # for the terminal GenerationResult's finish_reason.
        parsed_tool_calls: list[dict] = []
        if has_tools and accumulated_text:
            _, parsed_tool_calls = parse_tool_calls(
                accumulated_text, model_family=self.model_family,
            )

        # Save session
        if session_id:
            new_offset = self._get_cache_offset(cache_state.cache) if cache_state.cache else 0
            # Fallback: some models (GLM MoE) don't expose offset in cache objects
            if new_offset == 0 and cache_state.token_ids:
                new_offset = len(cache_state.token_ids)
            prev_offset = session.total_cache_tokens if session else 0

            # Build full assistant content for engine-internal messages
            # This includes thinking so next turn's suffix extends correctly
            full_assistant_content = self._make_full_assistant_content(
                accumulated_text, use_thinking,
            )

            if parsed_tool_calls:
                # Strip the tool_call XML from stored content so the template
                # doesn't double-render (content + tool_calls both emit XML).
                start_tag, _ = get_tool_markers(self.model_family)
                tc_idx = full_assistant_content.find(start_tag)
                if tc_idx >= 0:
                    full_assistant_content = full_assistant_content[:tc_idx].rstrip()

            # On HIT: extend session.messages with new incoming + assistant
            # On MISS: use incoming messages + assistant
            if cache_mode == "hit" and session:
                base_messages = list(session.messages) + new_messages
            else:
                base_messages = list(messages)
            assistant_msg: dict = {
                "role": "assistant",
                "content": full_assistant_content,
            }
            if parsed_tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in parsed_tool_calls
                ]
            updated_messages = base_messages + [assistant_msg]

            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=updated_messages,
                total_cache_tokens=new_offset,
            )

            logger.debug(
                f"[KV Cache] session={session_id} | SAVED | "
                f"offset: {prev_offset} -> {new_offset} tokens "
                f"(+{new_offset - prev_offset})"
            )

        # Auto-register base cache on miss
        if cache_mode in ("miss", "retry") and messages:
            self._maybe_register_base_cache(
                messages, prompt_token_ids, tools=tools, thinking=use_thinking,
            )

        # Determine finish reason (parsed_tool_calls computed above)
        finish_reason = "tool_calls" if parsed_tool_calls else "stop"

        yield GenerationResult(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=gen_token_count,
            prompt_tps=last_prompt_tps,
            generation_tps=last_gen_tps,
            cache_info=response_cache_info,
        )

    def _run_generate(
        self,
        *,
        cache_state,
        prompt_token_ids,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        sampler,
        logits_processors,
        session_id,
        total_prompt_tokens,
    ):
        """Backend dispatcher. Returns `(gen_iter, prompt_cache_or_none)`.

        - mlx-vlm path → `(iter, None)` — KV cache is mutated in place on
          `cache_state.cache` by mlx-vlm's stream_generate; no write-back
          required by the caller.
        - mlx-lm legacy path → `(iter, prompt_cache)` — caller must write
          `prompt_cache` back to `cache_state.cache` after the stream loop
          finishes (mlx-lm doesn't know about cache_state).

        This method is the SINGLE `self._use_vlm` branch point for stream
        construction. Other `self._use_vlm` references gate orthogonal
        concerns (load path, kv_bits warning, structured-output incompat
        warning) and remain in place.
        """
        if self._use_vlm:
            return (
                self._run_vlm(
                    cache_state=cache_state,
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    logits_processors=logits_processors,
                    session_id=session_id,
                    total_prompt_tokens=total_prompt_tokens,
                ),
                None,
            )
        return self._run_lm_legacy(
            cache_state=cache_state,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

    def _run_vlm(
        self,
        *,
        cache_state,
        prompt_token_ids,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        logits_processors,
        session_id,
        total_prompt_tokens,
    ):
        """mlx-vlm streaming path. Mutates `cache_state` in place.

        Applies the RotatingKVCache wrap-around safety gate before
        delegating to `vlm_stream_generate`.
        """
        # PLD is mlx-lm legacy only; fail loud rather than silently fall back.
        if self.cfg.pld_enabled:
            raise RuntimeError(
                "PLD is mlx-lm legacy only; cannot enable on a "
                "VLM-supported model_type"
            )

        # Safety gate: drop wrapped RotatingKVCache before reuse.
        if not self._safe_to_reuse_cache(cache_state, prompt_token_ids):
            logger.warning(
                f"[KV Cache] session={session_id} | "
                f"COLD-FILL (RotatingKVCache wrapped) — dropping cache and "
                f"re-prefilling full prompt ({total_prompt_tokens} tokens)"
            )
            cache_state.cache = None
            cache_state.token_ids = None

        input_ids = mx.array([prompt_token_ids])

        # Only inject drafter kwargs when a drafter is loaded — keeps the
        # non-drafter call shape byte-equal to baseline.
        gen_kwargs = dict(
            input_ids=input_ids,
            prompt_cache_state=cache_state,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            prefill_step_size=self.cfg.prefill_step_size,
            logits_processors=logits_processors if logits_processors else None,
        )
        drafter = getattr(self, "_drafter", None)
        # Layer-A safety net: if the next generation will cross the
        # RotatingKVCache sliding-window boundary (ring buffer wrap),
        # skip the drafter for this request only. Even with the B-series
        # monkey-patches active, we keep this guard so a future mlx-vlm
        # version that re-introduces the bug — or any unforeseen edge
        # case in the patch — cannot regress drafter acceptance to ~0.
        # We mutate the LOCAL ``drafter`` variable only; ``self._drafter``
        # stays loaded and the next non-wrapping request resumes MTP.
        wrap_imminent = self._will_wrap_during_generate(
            prompt_token_ids, cache_state
        )
        if drafter is not None and wrap_imminent:
            logger.warning(
                f"[Drafter] session={session_id} skip: RotatingKVCache wrap "
                f"imminent (sliding_window={self._sliding_window_size}) — "
                f"drafter bypassed for this request only"
            )
            drafter = None
        # PERF: flip the module-level fast-path flag so the B1/B2-v2
        # monkey-patches become near-noop pass-throughs when wrap CANNOT
        # happen for this entire request (including max_tokens worth of
        # generation). This eliminates per-token guard work (dict lookups,
        # layer iteration) on the speculative-decoding verify hot path.
        # The patches still run their full check whenever wrap is even
        # remotely possible (correctness-critical regime).
        win = int(getattr(self, "_sliding_window_size", 0) or 0)
        has_rot = bool(getattr(self, "_has_rotating_cache", False))
        cur_offset = 0
        if cache_state is not None:
            _cache = getattr(cache_state, "cache", None)
            if _cache:
                cur_offset = int(getattr(_cache[0], "offset", 0) or 0)
        wrap_possible = (
            has_rot
            and win > 0
            and (cur_offset + len(prompt_token_ids or []) + int(max_tokens or 0)) >= win
        )
        global _HOT_PATH_FAST
        _HOT_PATH_FAST = not wrap_possible
        if drafter is not None:
            gen_kwargs["draft_model"] = drafter
            gen_kwargs["draft_kind"] = getattr(self, "_draft_kind", None)
            if self.cfg.draft_block_size:
                gen_kwargs["draft_block_size"] = self.cfg.draft_block_size
            # Reset acceptance bookkeeping per request so the post-stream
            # log line below reports this request's stats only.
            if hasattr(drafter, "accept_lens"):
                drafter.accept_lens = []

        # F3: generation_stream is installed ONCE on the dedicated
        # _vlm_executor worker thread during engine __init__. Per-call
        # log stays at DEBUG so verbose logs don't drown in stream-id
        # lines; raise to INFO if a future cross-thread bug recurs.
        logger.debug(
            f"[VLM] thread={threading.current_thread().name} "
            f"gen_stream_id={id(sys.modules['mlx_vlm.generate'].generation_stream)}"
        )

        stream_iter = vlm_stream_generate(
            self._vlm_model,
            self._processor,
            "",  # prompt text ignored when input_ids is provided
            **gen_kwargs,
        )

        # PERF: ALWAYS return the raw stream_iter (no per-token wrapper).
        # The drafter-stats wrapper used to wrap every yielded token in an
        # extra Python generator frame (~30 tps cost at high TPS). The
        # stats are pure post-stream bookkeeping — accumulate them ONCE
        # after the stream is exhausted via a deferred finalize stash on
        # ``self``. The VLM executor is single-worker so a per-engine stash
        # is safe across the in-flight request.
        if drafter is None:
            self._pending_drafter_finalize = None
            return stream_iter

        # Defer finalize to a single post-stream call. Capture only the
        # locals needed; no per-token cost.
        _drafter_ref = drafter
        _sid = session_id

        def _finalize() -> None:
            lens = list(getattr(_drafter_ref, "accept_lens", []) or [])
            if lens:
                n_rounds = len(lens)
                total_accepted = sum(lens)
                mean_accepted = total_accepted / n_rounds
                max_a = max(lens)
                logger.debug(
                    f"[Drafter] session={_sid} rounds={n_rounds} "
                    f"mean_accepted={mean_accepted:.2f} max_accepted={max_a}"
                )
                if (
                    mean_accepted < _DRAFTER_LOW_ACCEPT_THRESHOLD
                    and n_rounds >= 10
                ):
                    logger.warning(
                        f"[Drafter] session={_sid} low acceptance: "
                        f"mean_accepted={mean_accepted:.2f} over {n_rounds} rounds — "
                        f"drafter may be net negative; consider --draft-block-size 2 "
                        f"or different drafter weights"
                    )
                session = self._sessions.get(_sid)
                if session is not None:
                    stats = session.drafter_stats or {
                        "requests": 0,
                        "total_rounds": 0,
                        "total_accepted": 0,
                    }
                    stats["requests"] += 1
                    stats["total_rounds"] += n_rounds
                    stats["total_accepted"] += total_accepted
                    session.drafter_stats = stats
            else:
                logger.debug(
                    f"[Drafter] session={_sid} no rounds recorded "
                    f"(drafter present but accept_lens empty)"
                )

        self._pending_drafter_finalize = _finalize
        return stream_iter

    def _run_lm_legacy(
        self,
        *,
        cache_state,
        prompt_token_ids,
        max_tokens,
        sampler,
        logits_processors,
    ):
        """mlx-lm legacy path. Manages a local `prompt_cache` because
        mlx-lm mutates the cache list in place during prefix-trim +
        stream_generate; the caller writes it back to `cache_state.cache`
        after the stream loop completes.

        Returns `(gen_iter, prompt_cache)`.
        """
        # Drafter is only wired on the mlx-vlm path; fail loud if reached here.
        if getattr(self, "_drafter", None) is not None:
            raise RuntimeError(
                "--draft-model not supported on mlx-lm legacy path; "
                "this model_type has no mlx-vlm support yet"
            )
        prompt_cache = cache_state.cache
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(self._language_model)
        else:
            # Trim cache to match prefix
            stored_ids = cache_state.token_ids or []
            prefix_len = 0
            for j in range(min(len(stored_ids), len(prompt_token_ids))):
                if stored_ids[j] != prompt_token_ids[j]:
                    break
                prefix_len = j + 1
            # Trim KV cache to prefix length
            for c in prompt_cache:
                if hasattr(c, "keys") and c.keys is not None:
                    cached_len = c.keys.shape[2] if len(c.keys.shape) > 2 else 0
                    if cached_len > prefix_len:
                        c.keys = c.keys[..., :prefix_len, :]
                        c.values = c.values[..., :prefix_len, :]
                        if hasattr(c, "offset"):
                            c.offset = prefix_len
            # Only feed tokens after prefix
            prompt_token_ids = prompt_token_ids[prefix_len:]

        lm_kwargs = {
            "max_tokens": max_tokens,
            "sampler": sampler,
            "prompt_cache": prompt_cache,
            "prefill_step_size": self.cfg.prefill_step_size,
            "logits_processors": logits_processors if logits_processors else None,
        }
        if self.cfg.kv_bits:
            lm_kwargs["kv_bits"] = self.cfg.kv_bits
            lm_kwargs["kv_group_size"] = self.cfg.kv_group_size
            lm_kwargs["quantized_kv_start"] = self.cfg.quantized_kv_start

        # PLD requires trimmable cache (for rollback on rejection).
        # Models with ArraysCache layers (Qwen3.5 DeltaNet, etc.) are NOT
        # trimmable — fall back to regular lm_stream_generate.
        use_pld = self.cfg.pld_enabled
        if use_pld:
            from mlx_lm.models.cache import can_trim_prompt_cache
            if not can_trim_prompt_cache(prompt_cache):
                if not getattr(self, "_pld_incompat_warned", False):
                    logger.warning(
                        f"[{self.model_id}] PLD disabled: model uses "
                        f"non-trimmable cache (e.g. ArraysCache/DeltaNet). "
                        f"Falling back to standard generation."
                    )
                    self._pld_incompat_warned = True
                use_pld = False

        if use_pld:
            from mlx_soloheaven.engine.pld import pld_generate_step
            gen_iter = _pld_response_adapter(
                pld_generate_step(
                    prompt=mx.array(prompt_token_ids),
                    model=self._language_model,
                    num_draft_tokens=self.cfg.pld_num_draft_tokens,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                    prompt_cache=prompt_cache,
                    prefill_step_size=self.cfg.prefill_step_size,
                    kv_bits=self.cfg.kv_bits if self.cfg.kv_bits else None,
                    kv_group_size=self.cfg.kv_group_size,
                    quantized_kv_start=self.cfg.quantized_kv_start,
                    ngram_k=self.cfg.pld_ngram_k,
                ),
                tokenizer=self.tokenizer,
            )
        else:
            gen_iter = lm_stream_generate(
                self._language_model,
                self.tokenizer,
                prompt=prompt_token_ids,
                **lm_kwargs,
            )
        return gen_iter, prompt_cache

    def _make_full_assistant_content(
        self, accumulated_text: str, thinking_enabled: bool,
    ) -> str:
        """Build full assistant content for engine-internal messages.

        Includes thinking markers so that suffix token computation works
        correctly on subsequent turns.

        ChatML/GLM: prompt suffix includes '<think>\\n' (or '<think>'), so
        accumulated_text starts after it. Prepend to get the complete content.

        Gemma 4: model generates thinking markers itself (e.g. '<|channel>thought\\n'),
        so accumulated_text already includes them.
        """
        if self.model_family == "gemma4":
            return accumulated_text
        # ChatML and GLM both use <think> prefix
        if thinking_enabled:
            prefix = "<think>" if self.model_family == "glm" else "<think>\n"
            return prefix + accumulated_text
        return accumulated_text

    def _maybe_register_base_cache(
        self,
        messages: list[dict],
        prompt_tokens: list[int],
        tools: list | None = None,
        thinking: bool = True,
    ):
        """Register a base cache for the system prompt if not already cached."""
        has_system_or_rotating = (
            (messages and messages[0].get("role") in ("system", "developer"))
            or self._has_rotating_cache
        )
        if not has_system_or_rotating:
            return
        sys_hash = self._system_hash(messages, tools=tools)
        if not sys_hash or sys_hash in self._base_caches:
            return
        system_tokens = self._extract_system_tokens(
            messages, prompt_tokens, tools=tools, thinking=thinking,
        )
        if system_tokens and len(system_tokens) < len(prompt_tokens):
            try:
                base_cache = make_prompt_cache(self._language_model)
                self._prefill_cache(base_cache, system_tokens)
                self._register_base_cache(
                    messages, base_cache, system_tokens, tools=tools,
                )
            except Exception as e:
                logger.warning(f"[Base Cache] registration failed: {e}")

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        tools: list | None = None,
        session_id: str | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> CompletionResult:
        """Non-streaming completion.

        VLM path is routed through the dedicated `_vlm_executor` worker so
        the module-global `mlx_vlm.generate.generation_stream` (a
        ThreadLocalStream installed once during engine init) stays on the
        same thread for both streaming and non-streaming requests. On the
        user's M3 Ultra-class hardware, calling `vlm_stream_generate` from
        the FastAPI event-loop thread surfaced `RuntimeError: There is no
        Stream(gpu, N) in current thread.` whenever lazy KV-cache arrays
        produced by an earlier worker-thread call were touched on the loop
        thread. Pinning both paths to the same worker eliminates the
        cross-thread evaluation entirely.
        """
        result = CompletionResult()

        def _drive() -> list[str]:
            # TODO(P3): _drive here and _run in generate_stream_async share
            # the same generate_stream-driving shape; consolidate into a
            # single helper once the API surface stabilises (>20 lines refactor).
            chunks: list[str] = []
            for chunk in self.generate_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                session_id=session_id,
                tools=tools,
                thinking=thinking,
                thinking_budget=thinking_budget,
                response_format=response_format,
            ):
                if chunk.text:
                    chunks.append(chunk.text)
                if chunk.finish_reason:
                    result.finish_reason = chunk.finish_reason
                    result.prompt_tokens = chunk.prompt_tokens
                    result.completion_tokens = chunk.completion_tokens
                    result.prompt_tps = chunk.prompt_tps
                    result.generation_tps = chunk.generation_tps
                    result.cache_info = chunk.cache_info
            return chunks

        if self._use_vlm and getattr(self, "_vlm_executor", None) is not None:
            fut = self._vlm_executor.submit(_drive)
            all_text = fut.result()
        else:
            all_text = _drive()

        full_text = "".join(all_text)
        thinking, content = split_thinking_and_content(full_text, model_family=self.model_family)
        result.thinking = thinking

        if tools:
            text_part, tool_calls = parse_tool_calls(content, model_family=self.model_family)
            if tool_calls:
                result.tool_calls = tool_calls
                result.content = text_part if text_part else None
                result.finish_reason = "tool_calls"
            else:
                result.content = content
        else:
            result.content = content

        return result

    def update_session_messages(self, session_id: str, messages: list[dict]):
        """Touch session and mark dirty after external caller finalizes messages.

        Note: The engine manages its own internal messages (with thinking) in
        _generate_locked. This method exists for the API layer to signal that
        generation is complete and the session should be persisted.
        The incoming messages parameter is ignored — internal messages are
        authoritative for cache matching.
        """
        session = self._sessions.get(session_id)
        if session:
            session.touch()
            logger.info(
                f"[Session] {session_id} | messages finalized | "
                f"{len(session.messages)} msgs, {session.total_cache_tokens} cached tokens"
            )
            self._mark_dirty(session_id)

    def compact_session(self, session_id: str, messages: list[dict]) -> dict:
        """Replace a session's messages and rebuild KV cache from scratch.

        Used when client compresses/summarizes conversation context.
        """
        with self._lock:
            self._touch_gpu()
            t0 = time.perf_counter()

            prompt_tokens = self._tokenize_prompt(messages)

            # Try base cache first
            base = self._find_base_cache(messages)
            base_tokens_used = 0
            prompt_cache = None
            if base and len(prompt_tokens) >= base.token_count:
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    feed_tokens = prompt_tokens[base.token_count:]
                    base_tokens_used = base.token_count
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self._language_model)
                feed_tokens = prompt_tokens

            if feed_tokens:
                self._prefill_cache(prompt_cache, feed_tokens)
            self._eval_cache(prompt_cache)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            # Build PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = prompt_cache
            cache_state.token_ids = prompt_tokens

            prev = self._sessions.get(session_id)
            prev_tokens = prev.total_cache_tokens if prev else 0
            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=new_offset,
            )

            logger.info(
                f"[Compact] session={session_id} | "
                f"{prev_tokens} -> {new_offset} tokens | "
                f"base={base_tokens_used} | "
                f"processed={len(feed_tokens)} tokens | "
                f"{elapsed:.2f}s"
            )

            # Auto-register base cache
            self._maybe_register_base_cache(messages, prompt_tokens)

            self._mark_dirty(session_id)

            return {
                "session_id": session_id,
                "status": "ok",
                "cached_tokens": new_offset,
                "previous_tokens": prev_tokens,
                "base_tokens": base_tokens_used,
                "processing_time_ms": round(elapsed * 1000),
            }

    def list_sessions(self) -> list[dict]:
        """List all active sessions."""
        result = []
        for sid, s in self._sessions.items():
            entry = {
                "session_id": sid,
                "messages": len(s.messages),
                "cache_tokens": s.total_cache_tokens,
                "last_used": s.last_used,
            }
            if s.drafter_stats is not None:
                entry["drafter_stats"] = s.drafter_stats
            result.append(entry)
        return sorted(result, key=lambda x: x["last_used"], reverse=True)

    def get_session(self, session_id: str) -> dict | None:
        """Get details for a specific session."""
        s = self._sessions.get(session_id)
        if not s:
            return None
        info = {
            "session_id": session_id,
            "messages": len(s.messages),
            "cache_tokens": s.total_cache_tokens,
            "last_used": s.last_used,
        }
        if s.drafter_stats is not None:
            info["drafter_stats"] = s.drafter_stats
        return info

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its cache."""
        with self._dirty_lock:
            self._dirty_sessions.discard(session_id)
        if session_id in self._sessions:
            del self._sessions[session_id]
        path = self._session_cache_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.discard(session_id)
        logger.info(f"[Session] DELETED | session={session_id}")
        return True

    async def generate_stream_async(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        session_id: str | None = None,
        tools: list | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> AsyncGenerator[GenerationResult, None]:
        """Async wrapper for generate_stream. Supports client disconnect cancellation."""
        loop = asyncio.get_event_loop()
        q: asyncio.Queue[GenerationResult | None | Exception] = asyncio.Queue()
        cancel_event = threading.Event()

        def _run():
            try:
                for result in self.generate_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    session_id=session_id,
                    tools=tools,
                    cancel_event=cancel_event,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    response_format=response_format,
                ):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, result)
            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        # F3: pin to the dedicated mlx-vlm worker thread so mlx-vlm's
        # module-global generation_stream (a ThreadLocalStream) stays
        # consistent across all turns. The future is discarded — results
        # flow via the existing asyncio.Queue plumbing. mlx-lm legacy
        # has no thread-local stream constraint and uses a one-shot
        # daemon thread to avoid contention with the VLM worker.
        if self._use_vlm and getattr(self, "execution_mode", "worker") != "main_thread":
            self._vlm_executor.submit(_run)
        else:
            threading.Thread(target=_run, daemon=True).start()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Yield empty result as keepalive during prompt processing
                    yield GenerationResult(text="")
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except (asyncio.CancelledError, GeneratorExit) as exc:
            cancel_event.set()
            # INFO-level so we always see disconnects (debugging client timeouts etc.)
            logger.info(
                f"[Stream] session={session_id} | client disconnected "
                f"({type(exc).__name__}) — cancelling generation"
            )
            raise

    async def generate_stream_batches_async(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        session_id: str | None = None,
        tools: list | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        response_format=None,
    ) -> AsyncGenerator[list[GenerationResult], None]:
        """Batched async wrapper for generate_stream.

        Same contract as generate_stream_async but yields *batches* (lists) of
        GenerationResult instead of one result at a time. Coalescing only
        changes BATCHING — the concatenation of all results across all batches
        is byte-identical to the scalar generate_stream_async ordering.

        Flush rules in the worker:
        - status="generating", finish_reason set, and the FIRST content token
          flush immediately as their own single-item batch (preserves the
          control-signal semantics + TTFT).
        - other normal content tokens accumulate; flushed when the batch reaches
          ``stream_coalesce_n`` OR ``stream_coalesce_ms`` has elapsed since the
          last flush.
        - cancellation, exceptions, and end-of-stream flush any pending batch
          first, then signal completion.

        ``stream_coalesce_n <= 1`` disables coalescing: every result is posted
        as a 1-item batch (uniform batch interface, scalar timing).
        """
        loop = asyncio.get_event_loop()
        q: asyncio.Queue[tuple[GenerationResult, ...] | None | Exception] = asyncio.Queue()
        cancel_event = threading.Event()

        coalesce_n = getattr(self.cfg, "stream_coalesce_n", 4)
        coalesce_ms = getattr(self.cfg, "stream_coalesce_ms", 30)
        # n <= 1 disables coalescing (1-item batches).
        coalescing = coalesce_n > 1

        def _run():
            batch: list[GenerationResult] = []
            last_flush = time.perf_counter()
            first_content_seen = False

            def _flush_batch():
                # Post a NEW tuple each time — never enqueue a list we then
                # mutate/clear. Caller resets ``batch`` + ``last_flush`` after.
                if batch:
                    loop.call_soon_threadsafe(q.put_nowait, tuple(batch))

            try:
                for result in self.generate_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    session_id=session_id,
                    tools=tools,
                    cancel_event=cancel_event,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    response_format=response_format,
                ):
                    if cancel_event.is_set():
                        # Flush whatever's batched, then stop.
                        _flush_batch()
                        batch = []
                        break

                    is_content = (
                        result.status is None and result.finish_reason is None
                    )
                    # FLUSH-IMMEDIATELY conditions:
                    #  - control signals (status / finish_reason)
                    #  - the very first content token (preserve TTFT)
                    #  - coalescing disabled (n <= 1): every result on its own
                    flush_now = (
                        result.status == "generating"
                        or result.finish_reason is not None
                        or (is_content and not first_content_seen)
                        or not coalescing
                    )
                    if is_content:
                        first_content_seen = True

                    if flush_now:
                        # Post any pending batch first (preserve order), then
                        # this result as its own single-item batch.
                        _flush_batch()
                        batch = []
                        loop.call_soon_threadsafe(q.put_nowait, (result,))
                        last_flush = time.perf_counter()
                        continue

                    batch.append(result)
                    now = time.perf_counter()
                    if len(batch) >= coalesce_n or (now - last_flush) * 1000 >= coalesce_ms:
                        loop.call_soon_threadsafe(q.put_nowait, tuple(batch))
                        batch = []
                        last_flush = now
            except Exception as e:
                if not cancel_event.is_set():
                    # Flush any pending normal tokens before the error so their
                    # content is not lost, then post the exception.
                    _flush_batch()
                    batch = []
                    loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                _flush_batch()
                loop.call_soon_threadsafe(q.put_nowait, None)

        # F3: same worker-submit logic as generate_stream_async — VLM is pinned
        # to the persistent _vlm_executor worker; legacy mlx-lm uses a one-shot
        # daemon thread.
        if self._use_vlm and getattr(self, "execution_mode", "worker") != "main_thread":
            self._vlm_executor.submit(_run)
        else:
            threading.Thread(target=_run, daemon=True).start()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Keepalive batch (single empty result) during prompt processing
                    yield [GenerationResult(text="")]
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                # ``item`` is a batch (tuple/list). Drain any immediately
                # available batches to collapse backlog while preserving order,
                # then yield each batch in order.
                pending: list[tuple[GenerationResult, ...]] = [item]
                while True:
                    try:
                        nxt = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        # End sentinel arrived in the backlog — yield what we
                        # have, then stop after this drain.
                        for b in pending:
                            yield list(b)
                        return
                    if isinstance(nxt, Exception):
                        # Yield content gathered so far, then raise.
                        for b in pending:
                            yield list(b)
                        raise nxt
                    pending.append(nxt)
                for b in pending:
                    yield list(b)
        except (asyncio.CancelledError, GeneratorExit) as exc:
            cancel_event.set()
            # INFO-level so we always see disconnects (debugging client timeouts etc.)
            logger.info(
                f"[Stream] session={session_id} | client disconnected "
                f"({type(exc).__name__}) — cancelling generation (batched)"
            )
            raise

    # --- Truncation & Regeneration ---

    def branch_from_turn(
        self,
        source_session_id: str,
        new_session_id: str,
        branch_turn: int,
        branch_messages: list[dict] | None = None,
    ) -> dict:
        """Branch a new session by building cache from scratch."""
        source = self._sessions.get(source_session_id)
        if not source and self._has_disk_cache(source_session_id):
            source = self._load_session_from_disk(source_session_id)
            if source:
                self._sessions[source_session_id] = source

        if source:
            engine_messages = source.messages[:branch_turn]
        elif branch_messages:
            engine_messages = branch_messages
        else:
            return {"error": "source session not found and no messages provided"}

        return self._rebuild_session(new_session_id, engine_messages)

    def prepare_regenerate(self, session_id: str) -> dict:
        """Remove last assistant message and restore cache."""
        session = self._sessions.get(session_id)
        if not session or len(session.messages) < 2:
            return {"error": "nothing to regenerate"}

        last_msg = session.messages[-1]
        if last_msg.get("role") != "assistant":
            return {"error": "last message is not assistant"}

        restore_to = len(session.messages) - 2  # before user msg
        result = self.truncate_session(session_id, restore_to)
        if result.get("status") == "ok":
            result["turn"] = restore_to
        return result

    def truncate_session(self, session_id: str, target_msg_count: int) -> dict:
        """Truncate session to target_msg_count messages, rebuilding cache."""
        session = self._sessions.get(session_id)
        if not session and self._has_disk_cache(session_id):
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session
        if not session:
            return {"error": "session not found"}
        if target_msg_count >= len(session.messages):
            return {"error": "nothing to truncate"}

        restore_messages = session.messages[:target_msg_count]
        return self._rebuild_session(session_id, restore_messages)

    def _rebuild_session(self, session_id: str, messages: list[dict]) -> dict:
        """Build a fresh KV cache for the given messages."""
        with self._lock:
            self._touch_gpu()
            t0 = time.perf_counter()

            prompt_tokens = self._tokenize_prompt(messages)

            # Try base cache first
            prompt_cache = None
            base = self._find_base_cache(messages)
            feed_tokens = prompt_tokens
            if base and len(prompt_tokens) >= base.token_count:
                if prompt_tokens[:base.token_count] == base.tokens:
                    prompt_cache = self._clone_base_cache(base)
                    feed_tokens = prompt_tokens[base.token_count:]
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self._language_model)

            if feed_tokens:
                self._prefill_cache(prompt_cache, feed_tokens)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            cache_state = PromptCacheState()
            cache_state.cache = prompt_cache
            cache_state.token_ids = prompt_tokens

            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=new_offset,
                pending_build_time=elapsed,
            )
            self._mark_dirty(session_id)

        logger.info(
            f"[Rebuild] session={session_id} | "
            f"{new_offset} tokens, msgs={len(messages)}, {elapsed:.2f}s"
        )
        return {
            "status": "ok",
            "method": "build",
            "build_time": round(elapsed, 2),
            "cached_tokens": new_offset,
            "messages": len(messages),
        }

    def session_stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": {
                sid: {
                    "messages": len(s.messages),
                    "cache_tokens": s.total_cache_tokens,
                    "last_used": s.last_used,
                }
                for sid, s in self._sessions.items()
            },
        }

    # --- Admin: cache overview + reset (process-mode-safe) ---------------
    #
    # These wrap the previously-direct admin.py accesses to ``_sessions`` /
    # ``_base_caches`` / ``cache_manager`` so the admin endpoints can read +
    # mutate cache state through the engine API. In process mode the proxy
    # RPCs these to the child (the authoritative cache owner); in-process they
    # operate on local state directly — both return plain JSON-serializable
    # dicts.

    def cache_overview(self) -> dict:
        """Serializable per-engine cache overview (memory sessions, base
        caches, cache-manager stats, disk files). Used by /api/admin/cache
        and /api/cache/stats. Reads in-memory + disk state directly."""
        sessions = []
        total_memory_bytes = 0
        for sid, s in self._sessions.items():
            cache = s.cache_state.cache if s.cache_state else None
            cache_size = self.cache_manager._estimate_cache_size(cache) if cache else 0
            total_memory_bytes += cache_size
            sessions.append({
                "session_id": sid,
                "messages": len(s.messages),
                "cache_tokens": s.total_cache_tokens,
                "cache_size_mb": round(cache_size / 1e6, 1),
                "last_used": s.last_used,
                "age_s": round(time.time() - s.last_used, 0),
            })
        sessions.sort(key=lambda x: x["last_used"], reverse=True)

        base_caches = [
            {
                "hash": h,
                "token_count": bc.token_count,
                "hit_count": bc.hit_count,
                "created": bc.created,
            }
            for h, bc in self._base_caches.items()
        ]

        disk_files = []
        total_disk_bytes = 0
        cache_dir = self.cfg.cache_dir
        if os.path.isdir(cache_dir):
            for fname in sorted(os.listdir(cache_dir)):
                if fname.endswith(".safetensors"):
                    fpath = os.path.join(cache_dir, fname)
                    try:
                        fsize = os.path.getsize(fpath)
                    except OSError:
                        continue
                    total_disk_bytes += fsize
                    disk_files.append({
                        "file": fname,
                        "size_mb": round(fsize / 1e6, 1),
                    })

        return {
            "model_id": self.model_id,
            "enable_thinking": self.cfg.enable_thinking,
            "sessions": sessions,
            "session_count": len(sessions),
            "base_caches": base_caches,
            "cache_manager": self.cache_manager.stats(),
            "disk_files": disk_files,
            "memory_bytes": total_memory_bytes,
            "disk_bytes": total_disk_bytes,
            "cache_dir": cache_dir,
        }

    def clear_caches(self) -> dict:
        """Clear all KV caches (memory sessions + base caches + cache_manager +
        disk files). Returns counts cleared. Used by /api/admin/cache/reset."""
        cleared = {"memory_sessions": 0, "disk_files": 0, "base_caches": 0}

        with self._dirty_lock:
            self._dirty_sessions.clear()

        cleared["memory_sessions"] += len(self._sessions)
        self._sessions.clear()

        cleared["base_caches"] += len(self._base_caches)
        self._base_caches.clear()

        self.cache_manager.memory_caches.clear()
        self.cache_manager.disk_index.clear()

        cache_dir = self.cfg.cache_dir
        if os.path.isdir(cache_dir):
            for fname in os.listdir(cache_dir):
                if fname.endswith(".safetensors"):
                    try:
                        os.remove(os.path.join(cache_dir, fname))
                        cleared["disk_files"] += 1
                    except OSError:
                        pass

        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.clear()

        return cleared

    # Alias: codex spec mentions reset(...); admin uses clear_caches semantics.
    def reset(self) -> dict:
        return self.clear_caches()
