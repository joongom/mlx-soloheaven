"""
MLX Engine — model loading and generation with KV cache reuse.

mlx-lm-first for text; mlx-vlm for the MTP/vision opt-in or types mlx-lm
cannot load. soloheaven is a TEXT-only server, so under `--backend auto`
the choice is mlx-lm-first BY SUPPORT, NOT by multimodal-ness: a
`vision_config`/`audio_config`/`image_token_index` in config.json does
NOT force mlx-vlm. gemma4 (a VLM family whose config always carries
`vision_config`) loads via mlx-lm because mlx-lm supports `gemma4` and
its output is byte-identical to LM Studio's. mlx-vlm is used only when
`--backend mlx-vlm` is passed explicitly, or (under auto) for a
model_type that mlx-lm lacks. The mlx-vlm path remains the canonical
(and drafter-ready / MTP) generation surface — it is the only backend
that supports the `--draft-model` speculative MTP/DFlash stack — and
stays load-bearing for that opt-in. The backend gate lives in
`_select_backend` and keys off `cfg.backend` ∈ {auto, mlx-lm, mlx-vlm}
plus `_mlx_lm_supports()` (auto fall-to-vlm only for unsupported types).

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
import bisect
import contextlib
import copy
import enum
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, Optional

import mlx.core as mx
from mlx_lm import load as lm_load
from mlx_lm import stream_generate as lm_stream_generate
from mlx_lm.models.cache import (
    make_prompt_cache,
    save_prompt_cache,
    load_prompt_cache,
    trim_prompt_cache,
)
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import (
    stream_generate as vlm_stream_generate,
    PromptCacheState,
)

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.thinking import (
    ForceDeferralGate,
    ThinkingBudgetProcessor,
    RepetitionPenaltyProcessor,
    initial_think_state,
    advance_think_state,
    force_end_from_state,
)
from mlx_soloheaven.engine.tool_parser import (
    _parse_glm_tool_calls,
    _partial_marker_tail,
    content_segments,
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

# U26 round 2 (codex F5a): cap on the per-session drafter-stats registry.
# The registry is pruned by delete_session/clear_caches, but sessions deleted
# through OTHER lifecycles (or never deleted at all) used to accumulate one
# entry per unique session id for the life of the process. LRU-bounded:
# evicting an old entry is SAFE because the stats are purely advisory
# (admin/list_sessions display) — losing one merely restarts that session's
# cumulative counters. Each entry is a 3-int dict, so 512 entries is a few
# tens of KB.
_DRAFTER_STATS_MAX = 512


# ---------------------------------------------------------------------------
# Eager (thread-safe) sampler.
#
# ROOT CAUSE this replaces: ``mlx_lm.sample_utils.make_sampler`` builds its
# categorical draw and its top_p/min_p/top_k filters with
# ``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``.
# That random-state-bound compiled graph is the ONLY thing that advances the
# global PRNG between decode steps, and it only advances on the thread it was
# first bound to. SoloHeaven runs per-request generation on a non-main
# (daemon) worker thread, so on that thread ``mx.random.state`` FREEZES: every
# token samples with the same key -> the same token repeats -> degenerate
# repetition loops. (Verified: main thread 30 draws ~12 unique tokens; daemon
# thread = 1 unique token.) Greedy (temp==0) is unaffected because it
# short-circuits to ``argmax`` with no RNG.
#
# FIX (mirrors LM Studio's bundled ``mlx_engine.utils.sampling.create_sampler``):
# reimplement the filters locally with plain ``@mx.compile`` (NO random-state
# binding) and do the FINAL categorical draw EAGERLY via
# ``mx.random.categorical``. The eager draw advances the global PRNG correctly
# on whatever thread it runs on. The filter MATH/op-order is identical to
# upstream mlx_lm (top_p -> min_p -> top_k); only the RNG threading changes, so
# the output distribution matches upstream.
#
# NOTE: do NOT call ``mx.random.seed`` inside a worker thread — it raises
# "no Stream(gpu,0) in current thread". The eager categorical advances state
# on its own without seeding.
# ---------------------------------------------------------------------------


@mx.compile
def _eager_apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """Keep only the top_k tokens by probability (mirror of mlx_lm.apply_top_k,
    but without the ``inputs/outputs=mx.random.state`` binding)."""
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )


@mx.compile
def _eager_apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """Min-p filter (mirror of mlx_lm.apply_min_p, no random-state binding)."""
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is "
            f"{min_tokens_to_keep}"
        )

    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            mx.array(False, tokens_to_remove.dtype),
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


@mx.compile
def _eager_apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """Top-p (nucleus) filter (mirror of mlx_lm.apply_top_p, no random-state
    binding)."""
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


def _make_eager_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
):
    """Thread-safe drop-in replacement for ``mlx_lm.sample_utils.make_sampler``.

    Filter math and op-order (top_p -> min_p -> top_k) are identical to
    upstream mlx_lm; the only difference is that nothing binds
    ``mx.random.state`` into a compiled graph, so the PRNG advances correctly
    on non-main (daemon) worker threads. ``temp==0`` short-circuits to
    ``argmax`` (byte-identical to upstream / LM Studio greedy).
    """
    if temp == 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    inv_temp = 1.0 / temp

    def sampler(logprobs):
        if top_p > 0 and top_p < 1.0:
            logprobs = _eager_apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = _eager_apply_min_p(logprobs, min_p, min_tokens_to_keep)
        # U20: cap top_k to the vocab defensively — _eager_apply_top_k raises
        # for top_k >= vocab_size (mid-generation ValueError on the plain
        # decode path). top_k >= vocab keeps every token, i.e. the filter is
        # a no-op, so skip it instead of erroring.
        if 0 < top_k < logprobs.shape[-1]:
            logprobs = _eager_apply_top_k(logprobs, top_k)
        # EAGER final draw — advances mx.random.state on the calling thread.
        return mx.random.categorical(logprobs * inv_temp)

    return sampler


def _collect_eos_ids(tokenizer) -> set[int]:
    """Collect end-of-sequence token IDs from every surface a model exposes
    them on: the mlx-lm wrapper's ``eos_token_ids`` list, the inner HF
    tokenizer's ``eos_token_id`` (scalar or list), and
    ``generation_config.eos_token_id`` (GLM-family multi-EOS). Returns an
    empty set when nothing is exposed."""
    eos_ids: set[int] = set()
    if getattr(tokenizer, "eos_token_ids", None):
        eos_ids.update(int(i) for i in tokenizer.eos_token_ids)
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    eid = getattr(inner, "eos_token_id", None)
    if eid is not None:
        if isinstance(eid, (list, tuple, set)):
            eos_ids.update(int(i) for i in eid)
        else:
            eos_ids.add(int(eid))
    gc = getattr(inner, "generation_config", None)
    if gc is not None:
        gc_eos = getattr(gc, "eos_token_id", None)
        if gc_eos is not None:
            if isinstance(gc_eos, (list, tuple, set)):
                eos_ids.update(int(i) for i in gc_eos)
            else:
                eos_ids.add(int(gc_eos))
    return eos_ids


def _pld_response_adapter(pld_iter, tokenizer, label: str = "PLD",
                          close_reason=None):
    """Adapt (token, logprobs, from_draft) tuples (pld_generate_step or
    qwen_mtp_generate_step — same contract) to mimic lm_stream_generate's
    GenerationResponse objects. ``label`` tags the acceptance log line.

    CLOSE-REASON CONTRACT (codex round 7, finding 1): this adapter owns the
    EOS knowledge, so IT decides whether closing the inner runner tears
    down a COMPLETE turn or cancels a live one. Once the EOS frame has been
    yielded — whether the teardown is the natural break-on-resumption after
    that yield OR a GeneratorExit received while suspended AT the EOS yield
    (engine observed the EOS downstream, client disconnected, engine closed
    us) — the close is NATURAL: the shared ``close_reason`` cell (see
    ``qwen_mtp.GeneratorCloseReason``) is marked before the inner runner is
    closed, so its finalize forwards run regardless of a late-set
    cancel_event. Pre-fix the inner runner was only closed by GC after this
    frame died and inferred "cancel" from the live event: finalization was
    skipped, the cache landed one token short of the recorded ids, and the
    persisted complete message silently corrupted the next HIT's suffix. A
    close BEFORE the EOS frame leaves the cell unmarked (mid-stream
    disconnect keeps the runner's cancel inference + skip semantics). The
    inner runner is also closed EXPLICITLY in the finally — deterministic
    finalize ordering instead of GC timing (works for close_reason=None
    callers like the PLD path too; PLD's finally-rewind is close-reason
    independent, so it needs no cell).

    - Uses mlx-lm's StreamingDetokenizer (buffers partial UTF-8 byte
      sequences so multi-byte characters like CJK aren't emitted as
      replacement chars \ufffd between tokens).
    - Performs EOS detection — pld_generate_step doesn't stop on its own
      (mlx-lm's stream_generate normally handles that). EOS ids include
      the wrapper's list + HF tokenizer's (list or single).
    """
    import time as _time
    from types import SimpleNamespace

    # Collect EOS token IDs from the mlx-lm wrapper, HF tokenizer, and
    # generation_config (GLM-family multi-EOS) — shared helper.
    eos_ids: set[int] = _collect_eos_ids(tokenizer)

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
    # Codex round 7, finding 1: True once the terminal EOS frame is about to
    # be yielded. Set BEFORE that yield so a GeneratorExit received while
    # suspended AT the EOS yield already counts — the turn's EOS was
    # produced, so any close from that point on is teardown of a COMPLETE
    # turn (see the finally below).
    eos_yielded = False
    try:
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
                # Flush any remaining buffered segment, then ALWAYS emit a
                # terminal frame carrying the EOS token id — mirroring
                # mlx-lm's stream_generate, whose final GenerationResponse
                # carries token=eos. The engine's post-loop records that id
                # into cache_state.token_ids; the QwenMTP finalize forwards
                # the stop token through the target, so stored ids MUST
                # include it for offset == len(token_ids) to hold
                # (multi-turn cache reuse). Paths whose cache does NOT
                # contain the stop (e.g. PLD) are reconciled by the
                # post-loop offset truncation as before.
                remaining = ""
                if detok is not None:
                    try:
                        detok.finalize()
                        remaining = detok.last_segment or ""
                    except Exception:
                        remaining = ""
                eos_yielded = True
                yield SimpleNamespace(
                    text=remaining, token=token,
                    prompt_tps=0.0, generation_tps=tps,
                    from_draft=from_draft,
                    # U7: mirror mlx-lm's stream_generate terminal frame —
                    # a natural EOS reports finish_reason="stop".
                    finish_reason="stop",
                )
                break

            if detok is not None:
                try:
                    detok.add_token(token)
                    text = detok.last_segment
                except Exception:
                    # Fallback: decode single token (may yield replacement
                    # chars)
                    text = tokenizer.decode([token])
            else:
                text = tokenizer.decode([token])

            # ACCOUNTING CONTRACT: yield a frame for EVERY token pulled from
            # the runner — INCLUDING tokens whose detok segment is empty
            # (partial UTF-8 bytes, or a BPE space held back until the next
            # token) — exactly mirroring mlx-lm's stream_generate, which
            # yields a GenerationResponse per token regardless of segment
            # emptiness. The engine's post-loop records resp.token of every
            # frame into cache_state.token_ids, and the QwenMTP runner
            # commits every DELIVERED token into the target cache (finalize
            # forwards the pending one), so dropping a frame here leaks a
            # committed-but-unrecorded token: at reconcile the cache lands
            # AHEAD of len(token_ids) and the hybrid (untrimmable
            # ArraysCache) branch can only INVALIDATE — every server turn
            # cold-fills. (Production signature: "cache ahead of recorded
            # ids by N"; the leaked ids decoded to BPE-held ' ' tokens
            # inside markdown thinking lists.) The buffered text is not
            # lost — the detokenizer attaches it to a later frame's segment.
            yield SimpleNamespace(
                text=text, token=token,
                prompt_tps=0.0, generation_tps=tps,
                from_draft=from_draft,
            )
        else:
            # U8/U7: the inner runner EXHAUSTED without EOS — max_tokens.
            # The runner already ran its own finalize (natural return), but
            # the DETOKENIZER still buffers tail text (partial UTF-8 bytes /
            # BPE-held whitespace) that mlx-lm's stream_generate would flush
            # via detokenizer.finalize(). Flush it here as a TEXT-ONLY frame
            # (token=None — every generated token id was already yielded on
            # its own frame and recorded by the engine's post-loop, so the
            # cache/token-id accounting contract is untouched), tagged
            # finish_reason="length" (U7). No frame when nothing is buffered
            # — the engine's max_tokens heuristic labels the bare-exhaustion
            # case, and an early corruption/cancel termination must not be
            # mislabeled here.
            remaining = ""
            if detok is not None:
                try:
                    detok.finalize()
                    remaining = detok.last_segment or ""
                except Exception:
                    remaining = ""
            if remaining:
                yield SimpleNamespace(
                    text=remaining, token=None,
                    prompt_tps=0.0,
                    generation_tps=(
                        count / (_time.perf_counter() - t_first)
                        if t_first and _time.perf_counter() > t_first
                        else 0.0
                    ),
                    from_draft=False,
                    finish_reason="length",
                )
    finally:
        # Codex round 7, finding 1: signal the close reason, then close the
        # INNER runner explicitly. Runs on EVERY teardown — the natural
        # break-on-resumption after the EOS yield, a GeneratorExit thrown
        # while suspended at any yield (including AT the EOS yield), and
        # inner-runner exhaustion/exceptions (close is then a no-op). If
        # the EOS frame was already yielded the turn is COMPLETE: mark the
        # cell NATURAL so the runner's finalize runs regardless of a
        # late-set cancel_event; otherwise leave the cell unmarked so a
        # mid-stream close keeps the runner's cancel inference (event set
        # -> skip finalize) intact. Explicit close replaces the old
        # GC-at-frame-death close: finalize now runs deterministically
        # BEFORE the engine's post-loop reconcile reads cache offsets.
        if eos_yielded and close_reason is not None:
            try:
                close_reason.mark_natural()
            except Exception:  # noqa: BLE001 — teardown must never mask the exit
                logger.exception(f"[{label}] close_reason.mark_natural failed")
        _inner_close = getattr(pld_iter, "close", None)
        if _inner_close is not None:
            try:
                _inner_close()
            except Exception:  # noqa: BLE001 — teardown must never mask the exit
                logger.exception(f"[{label}] inner runner close() failed")

    if count > 0:
        logger.info(
            f"[{label}] accepted {from_draft_count}/{count} draft tokens "
            f"({100*from_draft_count/count:.1f}% acceptance rate)"
        )


# GenerationResult / CompletionResult moved to engine/types.py (no-mlx module
# so they can be imported in the process-mode parent + child). Re-exported
# here so existing `from .mlx_engine import GenerationResult` imports work.
from mlx_soloheaven.engine.types import (
    CompletionResult,
    EngineBusyError,
    GenerationCancelled,
    GenerationResult,
)


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

    # --- Prompt contract (U3/U21) -------------------------------------------
    # Everything that shapes the tokenized prompt PREFIX besides the messages
    # themselves. ``tools`` is the CANONICAL serialization (plain dicts via
    # model_dump — JSON round-trippable for disk persistence) of the tool
    # schema the session's cache was built with; ``thinking`` is the
    # enable_thinking flag in effect. ``prompt_fingerprint`` is the
    # _prompt_fingerprint hash over both — compared on every HIT so a client
    # changing tools (or the thinking flag) mid-session takes an honest MISS
    # rebuild instead of silently answering with the stale schema, and
    # threaded through every rebuild path (compact / truncate / regenerate /
    # branch) so rebuilds re-tokenize WITH the session's tools. All three
    # round-trip disk save/load; ``None`` fingerprint marks a legacy /
    # pre-upgrade session (see the HIT gate's legacy rule).
    tools: list | None = None
    thinking: bool = True
    prompt_fingerprint: str | None = None

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


def _detect_token_ids(tokenizer, text: str) -> list[int]:
    """Tokenize ``text`` to its full token-id SEQUENCE (no special tokens).

    Unlike ``_detect_token_id`` (single-token only), this returns the multi-token
    encoding used by the BARE ``thought\\n`` opener detection (gemma4 sliding-
    window variant): past the 1024 window the ``<|channel>`` token is gone and the
    model emits ``thought\\n`` as plain text, which encodes to >1 token. Returns
    ``[]`` if the text encodes empty (defensive — bare detection then stays off).
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


class TurnCloseResult(enum.Enum):
    """Tri-state outcome of ``MLXEngine._try_close_interrupted_turn`` (C1).

    A plain bool conflated two very different "no close happened" shapes:
    "close UNNECESSARY" (gemma4/glm — the next-turn suffix leads with the
    closer, committing unterminated is template-valid) versus "close
    UNAVAILABLE" (ChatML where the closer could not be forwarded/verified —
    committing there hands the next HIT a cache whose suffix splice assumes
    a ``<|im_end|>`` that is NOT in the KV: template corruption).

    NOT_REQUIRED — the cached turn needs no template closer (gemma4/glm
        templates delimit turns themselves, or the recorded tail already IS
        an end-of-turn token — natural termination verified): commit as-is.
    CLOSED — the end-of-turn token was forwarded through the target and
        verified against every offset-bearing layer: commit.
    FAILED — a close is REQUIRED but could not be performed or verified
        (ChatML on the mlx-vlm path, head-bearing MTP cache, no detectable
        end-of-turn token, non-callable target, forward failure, or a
        post-forward offset mismatch). The session cache has been
        INVALIDATED fail-closed: do NOT commit.
    """

    NOT_REQUIRED = "not_required"
    CLOSED = "closed"
    FAILED = "failed"


def _load_generation_config_sampling(model_path: str) -> dict:
    """Read sampling DEFAULTS from ``<model_path>/generation_config.json``.

    Returns a dict containing ONLY the keys among
    ``{temperature, top_p, top_k, min_p}`` that are present AND of the expected
    numeric type. Anything else (missing file, invalid JSON, missing keys,
    wrong types) yields ``{}`` for that key — never raises.

    Notes:
      - ``repetition_penalty`` is intentionally NOT read here: soloheaven
        defaults it to 1.05 (gemma4 anti-repetition FIX 2) and HF's value
        (often 1.0/absent) would silently disable that mitigation.
      - HF stores ``top_k=0`` / ``top_p=1.0`` to mean "disabled", which already
        matches soloheaven's disabled sentinels, so no remapping is needed.
    """
    out: dict = {}
    if not model_path:
        return out
    path = os.path.join(model_path, "generation_config.json")
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001 — malformed/unreadable -> configless
        return out
    if not isinstance(data, dict):
        return out
    # Only accept the right numeric type. bool is a subclass of int, so reject
    # it explicitly; reject ints-where-float-is-fine is fine (we coerce floats).
    for key in ("temperature", "top_p", "min_p"):
        val = data.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[key] = float(val)
    top_k = data.get("top_k")
    if not isinstance(top_k, bool) and isinstance(top_k, int):
        out["top_k"] = top_k
    return out


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

# FIX 2: per-request stash for wiring logits_processors (RepetitionPenalty +
# ThinkingBudget) into the MTP clone ``_patched_mtp_rounds_v2``. Upstream
# ``generate_step`` builds ``processors`` locally and threads them into its
# non-speculative ``_step`` ONLY — the speculative ``_mtp_rounds`` path is
# called with a BARE ``sampler`` and never sees the processors, so MTP-decoded
# tokens were never repetition-penalised (gemma4 long-session closing-para
# loop). We cannot change how upstream invokes the clone, so ``_run_vlm``
# stashes the processors + the prompt-token-history seed here just before
# calling ``vlm_stream_generate``; the clone reads them and applies the
# processors itself (mirroring upstream ``_step``'s tokens/logits contract).
# Single-worker safe for the same reason as ``_HOT_PATH_FAST`` (VLM executor
# pins all calls to one thread). Cleared after each request by ``_run_vlm``.
#   _MTP_LOGITS_PROCESSORS: list[Callable[[mx.array, mx.array], mx.array]] | None
#   _MTP_TOKEN_HISTORY_SEED: list[int] | None  (prompt token ids)
_MTP_LOGITS_PROCESSORS = None
_MTP_TOKEN_HISTORY_SEED = None

# RUNAWAY-THINKING FIX (2026-06): the thinking-budget cap was only enforced in
# the post-wrap ``_plain_step`` (gated behind SOLOHEAVEN_MTP_WRAP_GATE, default
# OFF), so during normal MTP speculative decoding the cap NEVER fired and a
# thinking model could stay inside its ``<|channel>``…``<channel|>`` block until
# max_tokens (observed: 21,600+ tokens of empty thinking). The stateful
# ``ThinkingBudgetProcessor`` cannot be applied per verify-position (it
# over-counts rejected drafts), so the clone instead enforces the budget with a
# HISTORY-DERIVED helper (see thinking.advance_think_state/force_end_from_state)
# driven by ACTUALLY-EMITTED tokens. These three globals thread the budget +
# tokens into the clone alongside the existing stash; set in ``_run_vlm`` only
# when use_thinking + budget>0 + think_end_token>=0 (same gate as the
# ThinkingBudgetProcessor wiring), cleared in the same finally. When unset /
# budget<=0 this is a complete no-op (greedy-exact, current behaviour).
#   _MTP_THINK_BUDGET: int | None
#   _MTP_THINK_END_TOKEN: int | None
#   _MTP_THINK_START_TOKEN: int | None
#   _MTP_THINK_FAMILY: str | None  ("gemma4" / "chatml" — drives start-state)
#
# BARE-OPENER FIX (2026-06, Option A): past the 1024 sliding window the gemma4
# ``<|channel>`` think_start prime falls out of the window and the model emits a
# BARE ``thought\n`` opener with NO ``<|channel>`` token — so the token-id based
# open above never fires and the budget runs to max_tokens (observed: 8192 =
# max_tokens at 50K context). We thread the token-id SEQUENCE of the bare
# ``thought\n`` opener so ``advance_think_state`` can recognise it at GENERATION
# START ONLY (mirrors ThinkingRouter FIX 4 — a literal ``thought\n`` mid-content
# does NOT falsely open thinking). Empty/None => bare detection off => no-op.
#   _MTP_THINK_BARE_OPEN_TOKENS: list[int] | None
#
# U22 round 2 (codex batch-5 F4a): the tokenizer backing the request's
# ThinkingBudgetProcessor, threaded so the clone's history-derived force site
# applies the SAME bounded UTF-8 boundary deferral (ForceDeferralGate) as the
# plain path — the clone used to force ``think_end`` mid multi-byte character
# (trailing U+FFFD in reasoning_content on the gemma4 MTP path). None =>
# immediate force (historical behaviour).
#   _MTP_THINK_TOKENIZER: tokenizer | None
_MTP_THINK_BUDGET = None
_MTP_THINK_END_TOKEN = None
_MTP_THINK_START_TOKEN = None
_MTP_THINK_FAMILY = None
_MTP_THINK_BARE_OPEN_TOKENS = None
_MTP_THINK_TOKENIZER = None

# Post-wrap drafter gate (plain-decode fallback). SUPERSEDED by the B4
# RoPE-frame fix, which restores post-wrap drafter acceptance (~1.1) so the
# drafter stays net-positive past the wrap — gating it off would now throw
# away that speedup. Kept OFF by default as a safety fallback; set
# SOLOHEAVEN_MTP_WRAP_GATE=1 to force plain decode past the wrap (e.g. if a
# future drafter/model regresses post-wrap acceptance).
_MTP_WRAP_GATE = os.environ.get("SOLOHEAVEN_MTP_WRAP_GATE", "0") != "0"


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


def _rotating_wrapped(prompt_cache) -> bool:
    """True iff the model's sliding-window (RotatingKVCache) ring buffer has
    wrapped — i.e. cumulative ``offset`` has reached the physical
    ``max_size``.

    Past this point the Gemma 4 MTP drafter's acceptance collapses
    (~1.2 → ~0.25, MEASURED) because its SWA mask can no longer align with
    the rotated ring buffer, making speculative decoding NET-NEGATIVE vs
    plain decode (measured: 60 vs 83 tok/s by 2500 tokens on M5 Max). Used
    by the patched ``_mtp_rounds`` to switch to plain single-token decode
    for the remainder of a long generation once the ring wraps, while
    keeping the drafter's ~+40% pre-wrap win on the first ~``max_size``
    tokens. ``offset`` is a plain Python int (no mx sync) so this check is
    free on the per-round hot path.
    """
    # Scan for the first RotatingKVCache (layer 0 for Gemma 4, but be
    # robust to layouts where a non-sliding layer comes first).
    for c in prompt_cache:
        if type(c).__name__ == "RotatingKVCache":
            ms = int(getattr(c, "max_size", 0) or 0)
            off = int(getattr(c, "offset", 0) or 0)
            return ms > 0 and off >= ms
    return False


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

    # ----- B4: RoPE-frame fix — absolute-position drafter mask -----
    # The original drafter SWA mask returns a correct (None) bias only while
    # ``query_offset + query_len <= kv_len + window``; past that (true offset
    # >= ~2*window) it builds an explicit mask using PHYSICAL key indices and
    # masks out valid keys. More importantly, pairing the mask with B2's
    # clamped offset (below) keeps the mask happy but pins the query's RoPE
    # phase to max_size while the shared keys keep their TRUE absolute RoPE
    # positions [N-window..N-1] — shifting the query/key relative phase by
    # -(N-max_size) and corrupting drafter attention for BOTH sliding and
    # full layers. MEASURED: acceptance collapses ~1.1 -> ~0.2 right at the
    # wrap on real content. Fix (validated by A/B, codex-reviewed): feed the
    # TRUE offset (see the clone below) and build the sliding mask from the
    # keys' ABSOLUTE positions so the window is correct at any offset; full
    # layers then need no mask. Recovers post-wrap acceptance to ~1.1.
    try:
        import mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant as _g4a
        _orig_make_drafter_masks = _g4a.make_drafter_masks

        def _abs_make_drafter_masks(
            shared_kv_states, query_len, query_offset, sliding_window,
            dtype=mx.float32,
        ):
            # Unbatched (B=1) server path: scalar query_offset. Delegate the
            # rare batched/padded path to upstream for safety.
            if isinstance(query_offset, int):
                qo = query_offset
            elif hasattr(query_offset, "ndim") and query_offset.ndim == 0:
                qo = int(query_offset.item())
            else:
                return _orig_make_drafter_masks(
                    shared_kv_states, query_len, query_offset,
                    sliding_window, dtype,
                )
            masks = {}
            for lt, kv in shared_kv_states.items():
                kv_len = int(kv[0].shape[-2])
                if lt == "sliding_attention":
                    valid = min(qo, kv_len)
                    # keys occupy absolute positions [qo-valid, qo-valid+kv_len)
                    k_abs = (qo - valid) + mx.arange(kv_len)
                    q_abs = mx.arange(qo, qo + query_len)
                    dist = q_abs[:, None] - k_abs[None, :]
                    inside = (
                        (mx.arange(kv_len) < valid)
                        & (dist > -sliding_window)
                        & (dist < sliding_window)
                    )
                    masks[lt] = mx.where(
                        inside,
                        mx.array(0.0, dtype=dtype),
                        mx.array(-mx.inf, dtype=dtype),
                    )[None, None, :, :]
                else:
                    # full attention: with the true offset every real key is
                    # in range, so no mask is needed (matches upstream None).
                    masks[lt] = None
            return masks

        _abs_make_drafter_masks._mtp_wrap_patch = True
        _g4a.make_drafter_masks = _abs_make_drafter_masks
        logger.debug("[MTP-Patch B4] installed absolute-position drafter mask")
    except Exception:  # noqa: BLE001 — patch must never break inference
        logger.exception("[MTP-Patch B4] failed to install absolute-position mask")

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
            # NOTE (2026-06: FIX 1): the previous ``if _HOT_PATH_FAST:
            # return _orig_mtp_rounds(...)`` fast-path delegation was
            # REMOVED. Delegating to upstream for short generations
            # (prompt+max_tokens < sliding_window → wrap_possible=False →
            # _HOT_PATH_FAST=True) produced broken 2-token early-EOS output
            # because upstream lacks this clone's ``finally`` rollback of
            # rejected speculative draft K/V (see the ``finally`` below).
            # The clone is exact-greedy-equivalent to upstream pre-wrap
            # (true offset identical, B4 mask all-allowed/None pre-wrap), so
            # always running it is the minimal correct fix. The B1
            # ``_HOT_PATH_FAST`` no-op at the TextModel ``__call__`` patch is
            # left intact — it is a valid perf no-op pre-wrap.

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

            # B4 (supersedes B2): feed the TRUE offset, NOT the clamp, so the
            # drafter's query RoPE phase matches the shared keys' absolute
            # positions. The absolute-position mask (installed above) keeps
            # the sliding window correct without clamping.
            kv_offset = int(prompt_cache[0].offset)
            draft_model.set_shared_kv(shared_kv_states, kv_offset)

            b = first_bonus
            emitted = 1  # caller already yielded the first bonus

            # ----- FIX 2: logits_processors (RepetitionPenalty + ThinkingBudget) -----
            # Upstream calls this clone with a BARE sampler, so processors built
            # in generate_stream never reached MTP-decoded tokens (the active
            # decode path) — nothing suppressed gemma4's long-session repetition
            # loop. Read the per-request processors + prompt-token seed stashed
            # by _run_vlm and apply them here, mirroring upstream
            # generate_step._step's contract: each processor takes
            # (running_token_history, logits) → processed logits, BEFORE sampling.
            #
            # We split by statefulness because speculative decoding samples a
            # BLOCK of positions at once:
            #   * _block_procs = RepetitionPenaltyProcessor — stateless (derives
            #     everything from the ``tokens`` arg). Safe to apply per-position
            #     in the verify block with the correct cumulative history; this
            #     is THE repetition fix and applies on BOTH the speculative and
            #     post-wrap plain paths.
            #   * _plain_procs = structured-FSM — mutates internal state ONCE
            #     PER CALL, so calling it per verify-position (bs calls, only
            #     accepted+1 emitted) would over-count. Applied ONLY in the
            #     single-token _plain_step (post-wrap path) where call==emit.
            #     (Structured outputs disable the drafter entirely upstream of
            #     here, so in practice this list is empty on the MTP path.)
            #
            # RUNAWAY-THINKING FIX: ThinkingBudgetProcessor is STATEFUL too, but
            # the cap MUST fire during normal MTP speculative decoding (the
            # default path) — leaving it for _plain_step (gated OFF by default)
            # let thinking models run to max_tokens inside <|channel>…<channel|>.
            # Applying the stateful processor per verify-position over-counts
            # rejected drafts, so the clone DROPS ThinkingBudgetProcessor from
            # the per-position lists and instead enforces the budget from the
            # EMITTED-token history via a small incremental state (in_thinking,
            # thinking_count) folded over actually-emitted tokens (see the
            # _think_* setup below + force_end_from_state). This fires at the
            # bonus sample, every block-verify position, and _plain_step.
            # With no processors (penalty==1.0, no budget) this is an exact
            # no-op — pre-/post-wrap output stays greedy-exact.
            _all_procs = list(_MTP_LOGITS_PROCESSORS or [])
            _block_procs = [
                p for p in _all_procs if isinstance(p, RepetitionPenaltyProcessor)
            ]
            # Drop ThinkingBudgetProcessor from BOTH per-position lists — the
            # history-derived enforcement below replaces it for the MTP path
            # (avoids the stateful over-count). Keep everything else (e.g. the
            # structured-FSM processor) in _plain_procs for _plain_step.
            _plain_procs = [
                p
                for p in _all_procs
                if not isinstance(
                    p, (RepetitionPenaltyProcessor, ThinkingBudgetProcessor)
                )
            ]

            # ----- RUNAWAY-THINKING FIX: history-derived thinking-budget -----
            # Threaded from _run_vlm via the dedicated stash (set only when
            # use_thinking + budget>0 + think_end_token>=0). budget<=0 / unset
            # => _think_budget<=0 => force_end_from_state always False => no-op.
            _think_budget = int(_MTP_THINK_BUDGET or 0)
            _think_end_tok = (
                int(_MTP_THINK_END_TOKEN) if _MTP_THINK_END_TOKEN is not None else -1
            )
            _think_start_tok = (
                int(_MTP_THINK_START_TOKEN)
                if _MTP_THINK_START_TOKEN is not None
                else -1
            )
            _think_family = _MTP_THINK_FAMILY or "chatml"
            # BARE-OPENER FIX: token-id sequence of the gemma4 long-context bare
            # ``thought\n`` opener (no ``<|channel>``). Empty/None => bare path off.
            _think_bare_open = tuple(_MTP_THINK_BARE_OPEN_TOKENS or ())
            _think_active = _think_budget > 0 and _think_end_tok >= 0

            # U22 round 2 (codex F4a): bounded UTF-8 boundary deferral at
            # THIS force site, mirroring ThinkingBudgetProcessor. The gate
            # defers the forced close (skips the logit override) while the
            # emitted byte tail ends mid multi-byte character, up to 4
            # consults, then forces regardless. ``_think_tail`` tracks the
            # EMITTED token ids (python ints, bounded window) so the byte
            # check never syncs an mx array; per verify-position the pending
            # draft prefix is appended by the caller (extra_tail). No
            # tokenizer stashed => gate None => immediate force (historical).
            _think_gate = (
                ForceDeferralGate(_MTP_THINK_TOKENIZER)
                if (_think_active and _MTP_THINK_TOKENIZER is not None)
                else None
            )
            _think_tail = None

            _seed = _MTP_TOKEN_HISTORY_SEED or []
            # Running token history as a 1-D mx.array (processor contract).
            # Seeded with the prompt ids + the first bonus (already yielded by
            # the caller). Tracked whenever ANY processor is active.
            if _all_procs:
                _hist = mx.array(list(_seed) + [int(b)], dtype=token_dtype)
            else:
                _hist = None

            # U22 round 2: emitted-token tail for the deferral gate's byte
            # check — seeded like the thinking state (prompt tail + the
            # already-yielded first bonus). Bounded deque: O(1) appends, and
            # the byte check only ever inspects the last few ids.
            if _think_gate is not None:
                _think_tail = deque(
                    [int(t) for t in list(_seed)[-16:]], maxlen=16
                )
                _think_tail.append(int(b))

            # Incremental thinking state derived ONLY from emitted tokens. Seed
            # by folding over the prompt seed + the already-emitted first bonus,
            # so the state matches ThinkingBudgetProcessor at the same point.
            #
            # BARE-OPENER PROMPT/GENERATION DOMAIN SPLIT (codex HIGH fix): the
            # bare ``thought\n`` opener is a GENERATION-START phenomenon. The
            # prompt seed (``_MTP_TOKEN_HISTORY_SEED`` = the FULL prompt) must NOT
            # exercise the bare matcher — its first (normal) token mismatches the
            # bare opener and would latch ``content_seen=True``, so the GENERATED
            # bare opener could never fire and the budget never caps (the original
            # long-context runaway). Fix: fold the PROMPT SEED with the bare
            # matcher OFF (empty tuple) — prompt tokens still correctly track
            # ``in_thinking`` via the full ``<|channel>`` markers (e.g. a prior
            # turn's complete ``<|channel>...<channel|>`` block in history) WITHOUT
            # latching the bare ``content_seen``. THEN, at the generation boundary
            # (right before folding the first GENERATED token, the bonus ``b``),
            # reset the bare sub-state (``bare_idx=0, content_seen=False``) iff
            # gemma4 and currently OUTSIDE thinking, so a generated bare opener can
            # open even though the prompt preceded it. From ``b`` onward the bare
            # matcher is ON. ChatML / mid-prompt-thinking (in_thinking True at the
            # boundary) keep ``content_seen`` True — no gen-start bare detection.
            if _think_active:
                _think_state = initial_think_state(_think_family)
                # Prompt domain: bare matcher OFF (see comment above).
                for _t in _seed:
                    _think_state = advance_think_state(
                        _think_state,
                        int(_t),
                        think_start_token=_think_start_tok,
                        think_end_token=_think_end_tok,
                        bare_open_tokens=(),
                    )
                # Generation boundary: reset the bare sub-state so the FIRST
                # generated token can begin a bare-opener match. Only when gemma4
                # and outside thinking (in_thinking False); inside thinking there
                # is no gen-start bare opener to detect.
                _gen_in_thinking = _think_state[0]
                if (
                    _think_family == "gemma4"
                    and _think_bare_open
                    and not _gen_in_thinking
                ):
                    _think_state = (
                        _think_state[0],
                        _think_state[1],
                        0,      # bare_idx
                        False,  # content_seen
                    )
                # Generation domain: bare matcher ON. Fold the already-emitted
                # first bonus ``b`` so the state matches the same point as the
                # stateful processor (which starts matching at the first sampled
                # token).
                _think_state = advance_think_state(
                    _think_state,
                    int(b),
                    think_start_token=_think_start_tok,
                    think_end_token=_think_end_tok,
                    bare_open_tokens=_think_bare_open,
                )
            else:
                _think_state = initial_think_state(_think_family)

            def _advance_think(tok):
                # O(1) emitted-token state advance (no-op when inactive).
                nonlocal _think_state
                if _think_active:
                    _think_state = advance_think_state(
                        _think_state,
                        int(tok),
                        think_start_token=_think_start_tok,
                        think_end_token=_think_end_tok,
                        bare_open_tokens=_think_bare_open,
                    )
                    if _think_tail is not None:
                        # U22 round 2: track the emitted tail for the byte
                        # check; a closed block re-arms the deferral budget
                        # (parity with ThinkingBudgetProcessor's close reset).
                        _think_tail.append(int(tok))
                        if int(tok) == _think_end_tok and _think_gate is not None:
                            _think_gate.rearm()

            # Batch-5 round 6 (codex round-5 P2): preview-local deferral EPOCH
            # for the speculative verify loop. Sequential semantics grant a
            # REOPENED thinking block a fresh deferral allowance (the gate
            # re-arms on think_end), but the round-4 previews consulted
            # ``self.deferrals + sum(defer_events)`` for the WHOLE block — a
            # mid-block provisional think_end left the pre-close usage counted
            # against post-reopen positions, hard-forcing the close mid
            # multi-byte character ('�</think>') in the reopened cycle. The
            # verify loop snapshots the committed count into
            # ``_pos_defer_base`` at block start, counts provisional grants in
            # ``_pos_defer_pending``, and RESETS BOTH to 0 whenever the
            # provisional accepted prefix folds a think_end — mirroring,
            # preview-side only, the rearm the emitted-fold replay performs
            # for committed state. A provisionally folded think_end that is
            # ultimately REJECTED never touches the gate (previews are pure);
            # the post-walk replay recomputes committed truth and the next
            # block re-snapshots the true carried-over count.
            _pos_defer_base = 0
            _pos_defer_pending = 0

            def _force_think(logits, state, extra_tail=(), defer_events=None):
                nonlocal _pos_defer_pending
                # If the budget is reached and we're still inside thinking, force
                # the think_end token (logit -> 1e9) so it is sampled next —
                # exactly what ThinkingBudgetProcessor does. ``state`` is the
                # thinking state DERIVED from the cumulative history up to (but
                # excluding) the position being sampled. Returns ``logits``.
                # U22 round 2: before forcing, consult the deferral gate — if
                # the byte stream at this position (emitted tail + the pending
                # draft prefix ``extra_tail``) ends mid multi-byte character,
                # SKIP the override this round (bounded: the gate hard-forces
                # after 4 skips). Identical semantics to the plain path's
                # _apply_forced_close.
                # Batch-5 round 4 (codex finding 1): at PROVISIONAL block-verify
                # positions acceptance is not known yet, so the gate must not be
                # mutated — pass ``defer_events`` (one bool appended per call:
                # True == deferral granted) to PREVIEW the decision instead;
                # earlier provisional grants of the same block count toward the
                # bound via ``pending``. The caller replays ``commit_deferral``
                # after the accept/reject walk for the EMITTED positions only,
                # so rejected drafts never consume the shared 4-deferral budget.
                # ``defer_events=None`` (plain step / call==emit sites) keeps
                # the historical consult-and-consume behaviour.
                deferred = False
                if _think_active and force_end_from_state(state, _think_budget):
                    if _think_gate is not None:
                        tail = list(_think_tail) + [int(t) for t in extra_tail]
                        if defer_events is None:
                            deferred = _think_gate.should_defer(tail)
                        else:
                            # Round 6: preview against the EPOCH accounting
                            # (base + pending since the last provisional
                            # think_end), not the whole-block sum — see the
                            # _pos_defer_base/_pos_defer_pending comment.
                            deferred = _think_gate.should_defer_preview(
                                tail,
                                pending=_pos_defer_pending,
                                base=_pos_defer_base,
                            )
                    if not deferred:
                        logits[:, _think_end_tok] = 1e9
                if defer_events is not None:
                    defer_events.append(deferred)
                    if deferred:
                        _pos_defer_pending += 1
                return logits

            def _apply_procs(procs, logits, hist):
                # logits: (1, vocab). hist: 1-D mx.array of prior tokens.
                # Mirrors upstream _step: run each processor in order.
                for processor in procs:
                    logits = processor(hist, logits)
                return logits

            # Non-budget processors for the plain path, in logits_processors
            # order (rep-penalty first, then any structured-FSM). The
            # thinking-budget is intentionally EXCLUDED here and enforced via
            # _force_think (history-derived) instead, so it is not
            # double-enforced. Empty => exact no-op.
            _plain_all_procs = _block_procs + _plain_procs
            _plain_needs_procs = bool(_plain_all_procs) or _think_active

            # Plain (non-speculative) single-token decode step used by the
            # post-wrap gate below. Omits return_hidden/return_shared_kv so
            # it runs at full plain-decode speed and bypasses the B1
            # temporal-order patch (which keys on shared_kv_sink). FIX 2 +
            # RUNAWAY-THINKING FIX: applies the non-budget processors
            # (rep-penalty + structured) against the running history — this is
            # single-token so they are call==emit correct — then forces
            # think_end from the emitted-token state (call==emit, so the
            # incremental state stays exact), then appends the sampled token.
            def _plain_step(yarr):
                nonlocal _hist
                with mx.stream(_generation_stream):
                    logits = lm(yarr, cache=prompt_cache).logits
                    if _plain_needs_procs:
                        # logits here is (1, 1, vocab) — squeeze the time axis
                        # to (1, vocab) for the processor contract, then sample.
                        li = _apply_procs(_plain_all_procs, logits[:, -1, :], _hist)
                        li = _force_think(li, _think_state)
                        y = sampler(li)
                    else:
                        y = sampler(logits)
                if _all_procs:
                    _hist = mx.concat([_hist, y.reshape(-1).astype(token_dtype)])
                if _think_active:
                    # .item() forces a sync; only pay it when the budget is
                    # actually active (otherwise this path stays pipelined).
                    _advance_think(int(y.reshape(-1)[0].item()))
                return y

            while emitted < max_tokens:
                # PERF: dynamic wrap gate. Once the RotatingKVCache ring has
                # wrapped, the Gemma 4 drafter's acceptance collapses
                # (~1.2 → ~0.25, MEASURED) and speculative decoding goes
                # net-negative. Switch to plain decode for the rest of the
                # stream: this keeps the drafter's pre-wrap win (the first
                # ~max_size tokens) and FLOORS throughput at plain-decode
                # speed afterwards instead of the net-negative tail
                # (measured: 60→84 tok/s at 2500 tokens). Output is
                # unchanged — speculative decoding is exact-greedy, so
                # pre-wrap MTP == greedy and post-wrap plain == greedy.
                if _MTP_WRAP_GATE and _rotating_wrapped(prompt_cache):
                    logger.info(
                        f"[MTP] ring wrapped at emitted={emitted} — drafter "
                        "disabled, plain decode for remainder"
                    )
                    # Pipelined plain decode: dispatch the NEXT forward
                    # before materializing the current token so the GPU
                    # never idles on the per-token sync (mirrors mlx-vlm's
                    # non-speculative generate_step loop).
                    y = _plain_step(mx.array([[b]], dtype=token_dtype))
                    mx.async_eval(y)
                    mx.eval(y)
                    while emitted < max_tokens:
                        next_y = _plain_step(y.reshape(1, 1))
                        mx.async_eval(next_y)
                        yield int(y.reshape(-1)[0].item()), None
                        emitted += 1
                        if emitted % 256 == 0:
                            mx.clear_cache()
                        y = next_y
                    return

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
                    if _block_procs or _think_active:
                        # FIX 2 (block-aware repetition penalty) + RUNAWAY-
                        # THINKING FIX (history-derived thinking budget):
                        # verify_out.logits is (1, bs, vocab) — position i
                        # predicts the token following the prefix
                        # [b, draft_0..draft_{i-1}]. Per position we (a) apply the
                        # (stateless) rep-penalty processor with the CORRECT
                        # cumulative token history so the target's choice is
                        # repetition-penalised exactly as plain decode would be,
                        # and (b) force the think_end token when the budget is
                        # reached for THAT position's cumulative thinking state.
                        # Both are derived from the per-position prefix, so
                        # accept/reject stays consistent with penalised+capped
                        # greedy decode (exact for temp==0; temp>0 follows the
                        # same naive-equality scheme upstream uses). The stateful
                        # FSM processor is still NOT applied here — see the
                        # _plain_procs note.
                        vl = verify_out.logits
                        _bs = int(vl.shape[1])
                        # Draft tokens proposed for positions 1..bs-1 (position 0
                        # is conditioned only on [b]). Materialise once for the
                        # prefix-history build.
                        _draft_list = draft_tokens.reshape(-1).tolist()
                        _per_pos = []
                        # rep-penalty needs the mx.array history; build it only
                        # when rep-penalty is active to avoid the per-position
                        # concat cost on the think-only path.
                        _pos_hist = _hist if _block_procs else None
                        # Position 0 is conditioned on the prefix ending at b,
                        # which is already folded into _think_state.
                        _pos_think = _think_state
                        # Batch-5 round 4 (codex finding 1): per-position gate
                        # decisions are PREVIEWED into this event list (one
                        # bool per position) and committed after the walk for
                        # emitted positions only — rejected positions must not
                        # consume the shared UTF-8 deferral budget.
                        _pos_defer_events = (
                            [] if _think_gate is not None else None
                        )
                        # Round 6: fresh preview epoch per verify block — the
                        # committed count is re-snapshotted here (commits only
                        # happen in the post-walk replay, never mid-block, so
                        # the snapshot equals the live count for the whole
                        # per-position loop until a provisional think_end
                        # resets it).
                        _pos_defer_base = (
                            _think_gate.deferrals
                            if _think_gate is not None
                            else 0
                        )
                        _pos_defer_pending = 0
                        for i in range(_bs):
                            li = vl[:, i, :]
                            if _block_procs:
                                li = _apply_procs(_block_procs, li, _pos_hist)
                            # U22 round 2: position i's byte tail includes the
                            # draft tokens accepted-so-far in this block
                            # (positions 0..i-1) — thread them to the gate.
                            li = _force_think(
                                li,
                                _pos_think,
                                extra_tail=_draft_list[:i],
                                defer_events=_pos_defer_events,
                            )
                            _per_pos.append(sampler(li))
                            # Extend per-position context with the DRAFT token at
                            # position i for the NEXT position (mirrors the
                            # autoregressive prefix the target attends to). The
                            # walk below decides which of these are actually kept.
                            if i < len(_draft_list):
                                _dtok = int(_draft_list[i])
                                if _block_procs:
                                    _pos_hist = mx.concat([
                                        _pos_hist,
                                        mx.array([_dtok], dtype=token_dtype),
                                    ])
                                if _think_active:
                                    # Round 6: a provisionally folded
                                    # think_end closes the block for
                                    # SUBSEQUENT previews — reset the preview
                                    # epoch so the reopened cycle sees a fresh
                                    # deferral allowance (preview-side parity
                                    # with the emitted-fold rearm; committed
                                    # state is untouched here).
                                    if (
                                        _think_gate is not None
                                        and _dtok == _think_end_tok
                                    ):
                                        _pos_defer_base = 0
                                        _pos_defer_pending = 0
                                    _pos_think = advance_think_state(
                                        _pos_think,
                                        _dtok,
                                        think_start_token=_think_start_tok,
                                        think_end_token=_think_end_tok,
                                        bare_open_tokens=_think_bare_open,
                                    )
                        target_tokens = mx.stack(
                            [t.reshape(-1) for t in _per_pos], axis=1
                        )
                    else:
                        target_tokens = sampler(verify_out.logits)
                mx.async_eval(target_tokens, hidden_full)

                accepted, new_tokens = _speculative_walk(
                    draft_tokens, target_tokens, max_tokens - emitted
                )
                draft_model.accept_lens.append(accepted)
                # FIX 2: extend the running token history with the tokens
                # ACTUALLY emitted this block so subsequent positions/steps see
                # the correct repetition context. new_tokens = accepted drafts +
                # one target bonus (see _speculative_walk). Tracked whenever ANY
                # processor is active (the plain path past the wrap reads _hist).
                if _all_procs and new_tokens:
                    _hist = mx.concat([
                        _hist,
                        mx.array(new_tokens, dtype=token_dtype),
                    ])
                # RUNAWAY-THINKING FIX: advance the emitted-token thinking state
                # over the tokens ACTUALLY emitted this block (accepted drafts +
                # target bonus), so the next block's position-0 force decision
                # uses the true thinking count — no over-count from rejected
                # drafts (those were only tried per-position, never folded in).
                # Batch-5 round 4 (codex finding 1): replay the gate commits for
                # the emitted positions here — position i's previewed deferral
                # is committed right before its token is folded, exactly the
                # sequential (plain-path) order, so a think_end in new_tokens
                # re-arms AFTER the commits that preceded it. Positions past
                # len(new_tokens) (rejected drafts) are dropped: their previews
                # never touched the gate.
                if _think_active and new_tokens:
                    for _idx, _ntok in enumerate(new_tokens):
                        if (
                            _pos_defer_events is not None
                            and _idx < len(_pos_defer_events)
                            and _pos_defer_events[_idx]
                        ):
                            _think_gate.commit_deferral()
                        _advance_think(_ntok)

                try:
                    for tok in new_tokens:
                        yield tok, None
                        emitted += 1
                        if emitted >= max_tokens:
                            return
                finally:
                    # Roll back rejected speculative drafts even when the
                    # generator is CLOSED mid-block — stream_generate breaks
                    # on EOS while _mtp_rounds is suspended at the last
                    # `yield`, so the post-loop rollback never runs and this
                    # block's rejected draft K/V is stranded in the cache
                    # (cache.offset runs +rejected ahead of the recorded
                    # tokens → offset > len(token_ids) → next turn's
                    # wrapped-cache reuse COLD-FILLs, multi-second TTFT). The
                    # finally guarantees the trim runs on normal completion,
                    # `return` at max_tokens, AND GeneratorExit. (mlx-lm and
                    # PLD already rewind in a finally; the MTP clone didn't.)
                    if accepted < bs - 1:
                        try:
                            with mx.stream(_generation_stream):
                                lm.rollback_speculative_cache(
                                    prompt_cache, None, accepted, bs
                                )
                        except Exception:  # noqa: BLE001
                            logger.exception("[MTP] finally rollback failed")

                hidden = hidden_full[:, accepted : accepted + 1, :]
                b = new_tokens[-1] if new_tokens else b

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
                # B4 (supersedes B2): true offset (see read #1). The
                # absolute-position drafter mask keeps the window correct
                # without clamping the RoPE phase.
                kv_offset = int(prompt_cache[0].offset)
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
    # U2: byte-accounted LRU bookkeeping. ``size_bytes`` is measured once at
    # registration (the snapshot is immutable afterwards — clones deep-copy);
    # ``last_used`` is touched on every clone so the eviction sweep
    # (_evict_active_sessions_if_needed) can drop base caches LRU-first when
    # the shared memory_budget_gb is exceeded.
    size_bytes: int = 0
    last_used: float = field(default_factory=time.time)
    # MTP-finalized layout marker (qwen_mtp servers only): the cache holds
    # n_target + n_head entries with the head trailing by the lazy last
    # slot (head_offset == token_count - 1), and mtp_resume_hidden is the
    # boundary hidden h_{N-1} (1, 1, H) that validate_mtp_cache_reuse
    # requires to commit the head's last-slot pair at resume. Plain bases
    # keep the historical 40-entry shape (False/None).
    mtp_layout: bool = False
    mtp_resume_hidden: object | None = None


# Precompiled regexes used by message-normalization / cache-match hot path.
# Patterns + flags are byte-identical to the original inline re.* calls.
_NORMALIZE_RE_IMAGE_REMOVED = re.compile(r"\s*\[image data removed", re.IGNORECASE)
_NORMALIZE_RE_TODAYS_DATE = re.compile(
    r"Today's date:\s*\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4}"
)
_NORMALIZE_RE_SYSTEM_REMINDER = re.compile(
    r"\n?<system-reminder>.*?</system-reminder>", re.DOTALL
)
# (round 9, finding 2: the gemma4 prefix regexes that lived here were
# replaced by the router-authoritative _content_channel_union reduction in
# _normalize_for_match. Codex round 13, finding 1: the bare ``<think>``
# opener-strip regex is retired too — an unclosed raw is kept RAW.)
# Codex round 13, finding 2 (companion): the gemma4 channel markers are
# EXACTLY "<|tool_call>" / "<tool_call|>" (tool_parser._TOOL_MARKERS). The
# old optional-pipe pattern (<\|?tool_call>…<\|?tool_call\|?>) also matched
# a bare chatml "<tool_call>" opener as BOTH start and end, so on a turn
# with two chatml call blocks it consumed "block1 … <tool_call>" — from the
# first opener through the SECOND — and left the second block's body behind
# as phantom content. The removed prefix shortcut had masked this (it
# continued before normalization ran); with the real content comparison in
# effect the pattern must match only the genuine gemma4 markers.
_NORMALIZE_RE_TOOL_CALL_CHANNEL = re.compile(
    r"<\|tool_call>.*?<tool_call\|>", re.DOTALL
)
_NORMALIZE_RE_TOOL_CALL_XML = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
# (codex round 13, finding 2: the first-<tool_call> prefix-split regex is
# retired with the _messages_match shortcut it served.)


def _content_channel_union(
    text: str, model_family: str, thinking_active: bool = True,
) -> str:
    """U12 round 2: the CONTENT channel of raw assistant text, as the
    concatenation of ALL content segments (``content_segments`` — the
    positional twin of the streaming router, which is the authority).

    Tool-call XML inside a thinking segment is the model REHEARSING a call,
    not making one — the match helpers and the stored-content stripping
    below must never see it. The round-1 reduction (suffix after the LAST
    close marker) broke gemma4 multi-cycle output (thought → content →
    thought → content): a legitimate call in an EARLIER content cycle was
    treated as thinking territory, so the session stored BOTH the raw XML
    and the structured tool_calls, the match helpers ignored the call, and
    a rebuild could double-render it.

    Codex round 3, finding 4: ``thinking_active`` is the router-active
    state the text streamed/was stored under — with thinking DISABLED the
    router passes EVERYTHING through as content, so a literal ``</think>``
    in the text is a quote, never a channel boundary (gemma4 segmentation
    stays marker-driven regardless; see content_segments)."""
    return "".join(
        text[s:e]
        for s, e in content_segments(text, model_family, thinking_active)
    )


def _strip_content_channel_tool_xml(
    text: str, model_family: str, thinking_active: bool = True,
) -> str:
    """U12 round 2: remove PARSEABLE tool-call XML from every CONTENT
    segment of raw assistant text, preserving thinking segments byte-for-
    byte (rehearsals stay intact) and — per the U11 per-block contract —
    retaining unparseable blocks and all ordinary content text in their
    original order. Used by the session save so the stored content never
    duplicates the structured ``tool_calls`` entry (template double-render),
    across ALL content cycles, not just the one after the last close
    marker. The result is right-stripped (historical save behavior).

    Codex round 3, finding 3: the per-segment parse runs in SEGMENT MODE
    (``rstrip_content=False``) — a segment's trailing whitespace is INTERNAL
    whitespace of the reassembled string (it sits before the next thinking
    marker), and the historical per-call right-strip silently deleted it.
    Only the final reconstructed string gets the one historical right-strip.
    Finding 4: ``thinking_active`` threads the router-active state into the
    channel segmentation (see _content_channel_union)."""
    out: list[str] = []
    pos = 0
    for s, e in content_segments(text, model_family, thinking_active):
        out.append(text[pos:s])
        seg_content, _ = parse_tool_calls(
            text[s:e], model_family=model_family, rstrip_content=False,
        )
        out.append(seg_content)
        pos = e
    out.append(text[pos:])
    return "".join(out).rstrip()


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
        # Engine-side shutdown gate (codex batch-3 round 3, finding 2): set
        # by begin_shutdown() right before the server-shutdown flush. Checked
        # IMMEDIATELY AFTER acquiring the engine lock on every mutating path
        # (_mutate_locked chokepoint + _acquire_lock_cancellable), so a
        # straggler that outlived the executors' bounded quiesce wait while
        # BLOCKED on the engine lock becomes a mutation-free no-op instead of
        # dirtying state after the flush. Plain bool (GIL-atomic).
        self._shutting_down = False
        # Codex round 5, finding 3a: mutations that ENTERED their critical
        # section before the gate closed (a minutes-long compaction prefill)
        # outlive the executors' bounded quiesce AND the flush's bounded lock
        # acquire, then publish + mark dirty AFTER the final flush.
        # ``_mutations_in_flight`` counts ops currently inside
        # ``_mutate_locked`` (incremented/decremented under the engine lock;
        # read lock-free — plain int, GIL-atomic) so the server shutdown can
        # bounded-wait for them, and ``_shutdown_cancel_event`` is set by
        # ``begin_shutdown`` as a cooperative cancellation source for the
        # compaction/rebuild prefills (aborting a compaction at shutdown is
        # safe — it reruns after restart).
        self._mutations_in_flight = 0
        self._shutdown_cancel_event = threading.Event()
        self.cache_manager = CacheManager(
            memory_budget_gb=cfg.memory_budget_gb,
            disk_budget_gb=cfg.disk_budget_gb,
            cache_dir=cfg.cache_dir,
        )
        self.model_id = ""

        # Session-based cache: session_id -> SessionState
        self._sessions: dict[str, SessionState] = {}

        # PROVENANCE of anon session ids: ids the ENGINE itself minted for
        # session-less requests ("anon-<hex>" from
        # _resolve_anon_session_id_locked, plus the legacy "anon" fallback in
        # _generate_locked). The anon prefix-scan only considers sessions whose
        # id is in this set — NOT name-based matching — so an EXPLICIT session
        # a client happened to key as user="anon-..." can never be selected /
        # mutated by an anonymous request. Mutated only under self._lock.
        self._anon_minted_ids: set[str] = set()

        # Base cache pool: system_hash -> BaseCacheEntry
        self._base_caches: dict[str, BaseCacheEntry] = {}

        # U26: cumulative per-session drafter acceptance stats, keyed OUTSIDE
        # SessionState. Every turn installs a brand-new SessionState (the
        # post-generation save, the interrupted-turn commit, compact/rebuild),
        # so an accumulator kept on the state object was reset each turn —
        # and the post-stream finalize runs BEFORE the install, so a brand-new
        # session's first turn was dropped entirely. This registry is the
        # single source of truth; installs stamp the CURRENT dict onto the new
        # SessionState (same object — in-place finalize updates stay visible
        # to the admin readers that surface s.drafter_stats). Pruned on
        # delete_session/clear_caches; entries survive active-LRU eviction so
        # a disk-reloaded session reclaims its stats. Mutated under self._lock.
        # U26 round 2 (F5a): LRU-ordered and bounded at _DRAFTER_STATS_MAX
        # (see _accumulate_drafter_stats).
        self._session_drafter_stats: "OrderedDict[str, dict]" = OrderedDict()

        # Dirty session tracking for idle-time disk save
        self._dirty_sessions: set[str] = set()
        self._dirty_lock = threading.Lock()

        # Sessions with an in-flight generation. Such sessions MUST NOT be
        # evicted (their KV cache is being mutated in place). Generations are
        # serialized by the shared GPU lock, but a session can be marked busy
        # while another concurrent caller (e.g. an admin RPC) inspects state,
        # so guard with a dedicated lock.
        self._busy_sessions: set[str] = set()
        self._busy_lock = threading.Lock()

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
        # Identity of the single _vlm_executor worker thread. Recorded so that
        # code already executing ON the worker (e.g. post-generation eviction
        # driven inside generate_stream's finally) can detect re-entrancy and
        # run inline instead of self-submitting + blocking on .result(), which
        # would deadlock the only worker against itself. See _save_session_to_disk.
        self._vlm_worker_ident: int | None = None

        def _vlm_worker_init():
            # Runs on the dedicated worker thread.
            self._vlm_worker_ident = threading.get_ident()
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
        #
        # mlx-lm-first gate: under `auto`, route to mlx-lm whenever mlx-lm
        # supports the model_type (gemma4 included — vision_config does NOT
        # force vlm; soloheaven is text-only). mlx-vlm is used only for an
        # explicit `--backend mlx-vlm` opt-in (e.g. the MTP drafter stack) or,
        # under auto, for a model_type mlx-lm cannot load. When a vlm backend
        # is requested/needed but the model_type isn't in mlx-vlm's registry,
        # we warn and fall through to the mlx-lm branch below.
        self._use_vlm = False
        vlm_supported = self._select_backend(model_config)
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

        # Sampling defaults precedence: CLI-pinned flag > generation_config.json
        # > built-in dataclass fallback (see _apply_generation_config_sampling).
        try:
            self._apply_generation_config_sampling()
        except Exception:  # noqa: BLE001 — never let genconfig parsing break load
            logger.exception(
                f"[{self.model_id}] generation_config sampling apply failed "
                f"(keeping current defaults)"
            )

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
        # FIX-7: kv_bits>0 on a rotating-cache model crashes MID-GENERATION
        # (RotatingKVCache.to_quantized raises NotImplementedError once
        # quantized_kv_start is reached; mlx-lm's maybe_quantize_kv_cache
        # only gates on hasattr, which RotatingKVCache satisfies). Reject
        # loudly at load instead.
        self._reject_kv_bits_on_rotating_cache()
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

        # Drafter weight is cached on ``self._drafter`` for per-request reuse.
        # Dispatch by the DRAFTER's model_type:
        #   * qwen3_5_mtp     -> native mlx-lm MTP head (engine/qwen_mtp.py);
        #                        loaded inline on this thread, _draft_kind
        #                        "qwen_mtp" routes _run_lm_legacy through
        #                        qwen_mtp_generate_step.
        #   * everything else -> mlx-vlm drafter stack (gemma4_assistant MTP /
        #                        DFlash), --backend mlx-vlm required; the
        #                        mlx-lm path refuses loudly.
        #
        # F3-LOAD (vlm drafters only): the drafter is loaded **on the dedicated VLM worker
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
        self._mtp_block_size = None
        if self.cfg.draft_model:
            from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

            _draft_type = qwen_mtp_mod.read_model_type(self.cfg.draft_model)
            if _draft_type == qwen_mtp_mod.QWEN_MTP_MODEL_TYPE:
                # Qwen3.5/3.6 MTP head — runs NATIVELY on the mlx-lm path
                # (no mlx-vlm; the installed mlx-vlm has no qwen3_5_mtp
                # drafter). Loaded inline like the target weights: lm-path
                # generation runs on the calling thread (process mode owns
                # the child main thread), so there is no worker hand-off.
                if self._use_vlm:
                    raise RuntimeError(
                        "qwen3_5_mtp MTP head (--draft-model) runs natively "
                        "on the mlx-lm path; remove --backend mlx-vlm (the "
                        "installed mlx-vlm has no qwen3_5_mtp drafter)."
                    )
                t0_drafter = time.perf_counter()
                _head, _info = qwen_mtp_mod.load_qwen_mtp_head(
                    self.cfg.draft_model
                )
                # Fail-closed at load: the head borrows the target's
                # embeddings + lm_head, and the manual pre-final-norm layer
                # loop must reproduce model() logits exactly.
                qwen_mtp_mod.assert_head_target_compat(
                    _info, self._language_model
                )
                qwen_mtp_mod.verify_layer_loop_parity(self._language_model)
                self._drafter = _head
                self._draft_kind = "qwen_mtp"
                self._mtp_block_size = int(
                    self.cfg.draft_block_size or _info["block_size"] or 3
                )
                logger.info(
                    f"[Drafter] loaded {self.cfg.draft_model} kind=qwen_mtp "
                    f"block_size={self._mtp_block_size} "
                    f"num_head_layers={_info['num_layers']} "
                    f"weights={_info['num_weights']} (strict) in "
                    f"{time.perf_counter() - t0_drafter:.1f}s — "
                    f"MTP speculative decoding active on the mlx-lm path"
                )
            elif not self._use_vlm:
                raise RuntimeError(
                    f"MTP drafter (--draft-model) with drafter model_type="
                    f"{_draft_type!r} requires --backend mlx-vlm "
                    f"(gemma4_assistant MTP/DFlash run on mlx-vlm). Only "
                    f"qwen3_5_mtp heads run natively on the default mlx-lm "
                    f"backend; alternatively use --pld for speculative "
                    f"decoding. (target model_type={self._model_type!r})"
                )
            else:

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

            # Bound the Metal buffer-reuse pool. The plain decode path never
            # clears this pool, so without a cap it can grow to tens of GB and
            # OOM the Mac. set_cache_limit returns the PREVIOUS limit. Process-
            # mode safe: runs in whichever process owns the model. Overridable
            # via Config.mlx_cache_limit_gb (CLI --mlx-cache-limit-gb / env
            # SOLOHEAVEN_MLX_CACHE_LIMIT_GB); <=0 disables the cap.
            limit_gb = getattr(self.cfg, "mlx_cache_limit_gb", 4.0)
            if limit_gb and limit_gb > 0 and hasattr(mx, "set_cache_limit"):
                prev = mx.set_cache_limit(int(limit_gb * 1e9))
                logger.info(
                    f"Metal cache limit set to {limit_gb:.1f}GB "
                    f"(was {prev / 1e9:.1f}GB)"
                )

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
        # Persist-on-stop is MODE-AGNOSTIC: register the shutdown flush for
        # every execution mode. It was previously registered only inside
        # _start_gpu_keepalive, which is skipped in main_thread/process mode
        # (and when --gpu-keepalive is off) — so a normal server stop never
        # flushed dirty sessions to disk in the production-default process
        # mode. Keepalive below remains thread-mode-only.
        self._register_shutdown_flush()
        if self.cfg.gpu_keepalive and self.execution_mode != "main_thread":
            self._start_gpu_keepalive()
            logger.info(f"[{self.model_id}] GPU keepalive enabled (interval={self.GPU_KEEPALIVE_INTERVAL}s)")
        elif self.cfg.gpu_keepalive and self.execution_mode == "main_thread":
            # No background thread may touch MLX cache tensors in main-thread
            # mode (codex constraint), so the keepalive THREAD is never
            # started here. In PROCESS mode this engine lives inside the
            # child whose main loop (process_worker.worker_main) runs the
            # same GPU touch (_gpu_keepalive_ping) from its poll-timeout
            # branch — that loop IS the child's main thread, so it may touch
            # MLX safely. For a bare in-process main_thread engine (no
            # worker loop) keepalive stays off entirely: there is no safe
            # idle hook on the main thread to piggyback the touch on.
            logger.info(
                f"[{self.model_id}] GPU keepalive: no background thread in "
                f"main_thread mode — the process-mode worker loop provides "
                f"the periodic GPU touch (see process_worker)"
            )

    def _apply_generation_config_sampling(self) -> dict:
        """Populate self.cfg.default_* sampling fields from the model's
        ``generation_config.json``, honouring CLI precedence.

        Precedence (this method only decides what self.cfg.default_* HOLDS at
        load time; the per-request and CLI-vs-default layers are elsewhere):
            CLI-pinned flag (cfg.cli_set_sampling) > generation_config.json >
            built-in dataclass fallback (already in place).

        For each of temperature/top_p/min_p/top_k PRESENT in the model's
        generation_config.json, write it into self.cfg.default_* UNLESS the
        field was explicitly pinned on the CLI. Fields neither CLI-pinned nor in
        generation_config keep the dataclass fallback — so a model with no
        generation_config (or no sampling fields) is byte-for-byte unchanged.

        Returns the dict of values actually applied (for tests/diagnostics).
        """
        cli_pinned = getattr(self.cfg, "cli_set_sampling", frozenset()) or frozenset()
        gen_sampling = _load_generation_config_sampling(self.cfg.model_path)
        applied: dict = {}
        for name in ("temperature", "top_p", "min_p", "top_k"):
            if name in cli_pinned:
                continue  # CLI flag wins — do not override
            if name in gen_sampling:
                setattr(self.cfg, f"default_{name}", gen_sampling[name])
                applied[name] = gen_sampling[name]

        sources = []
        if cli_pinned:
            sources.append("CLI=" + ",".join(sorted(cli_pinned)))
        if applied:
            sources.append(
                "generation_config="
                + ",".join(f"{k}={v}" for k, v in applied.items())
            )
        fallback = [
            n for n in ("temperature", "top_p", "min_p", "top_k")
            if n not in cli_pinned and n not in gen_sampling
        ]
        if fallback:
            sources.append("fallback=" + ",".join(fallback))
        logger.info(
            f"[{getattr(self, 'model_id', '?')}] sampling defaults — "
            + ("; ".join(sources) if sources else "fallback (all)")
            + f" -> temp={self.cfg.default_temperature}, "
            f"top_p={self.cfg.default_top_p}, "
            f"min_p={self.cfg.default_min_p}, "
            f"top_k={self.cfg.default_top_k}"
        )
        return applied

    # --- Model detection helpers ---

    def _select_backend(self, model_config: dict) -> bool:
        """Decide whether load_model should load via mlx-vlm vs mlx-lm.

        Single source of truth for the mlx-lm-first backend gate (PR1). The
        criterion under `backend=auto` is **mlx-lm-first BY SUPPORT, not by
        multimodal-ness**: soloheaven is a TEXT-only server, so a config that
        merely carries `vision_config`/`audio_config`/`image_token_index` does
        NOT force mlx-vlm. (Gemma 4 is a VLM family whose config ALWAYS has
        `vision_config`, yet `mlx_lm.load()` loads its text checkpoint and its
        greedy output is byte-identical to LM Studio's — so it must route to
        mlx-lm.) mlx-vlm is reserved for the MTP/vision opt-in (`--backend
        mlx-vlm`) or for model types mlx-lm simply cannot load.

        Decision:
            backend ∈ {auto, mlx-lm, mlx-vlm} (already validated/lowercased).
            if   backend == "mlx-vlm": want_vlm = True   (explicit opt-in)
            elif backend == "mlx-lm":  want_vlm = False  (force mlx-lm)
            else (auto):               want_vlm = not _mlx_lm_supports(type)
        When a vlm backend is *requested* (`want_vlm`) but the model_type isn't
        in mlx-vlm's registry, we warn and report mlx-lm so the caller falls
        through to the mlx-lm branch.

        Returns True iff load_model should load via mlx-vlm; False -> mlx-lm.

        load_model and tests/test_backend_selection.py both call this method
        so the gate decision is never duplicated.

        (`model_config` is retained for the INFO log / signature stability; it
        no longer influences the choice — vision_config does not force vlm.)
        """
        backend = (getattr(self.cfg, "backend", "auto") or "auto").lower()
        # Defense-in-depth: Config validates this at startup, but a programmatic
        # cfg.backend that bypassed that path must NOT silently fall through as
        # "not vlm" (which would behave like a forced mlx-lm). Fail loudly.
        if backend not in ("auto", "mlx-lm", "mlx-vlm"):
            raise ValueError(
                f"invalid backend {backend!r}; "
                f"choose one of auto/mlx-lm/mlx-vlm"
            )
        if backend == "mlx-vlm":
            want_vlm = True   # explicit opt-in: MTP / vision
        elif backend == "mlx-lm":
            want_vlm = False  # force mlx-lm
        else:  # auto: mlx-lm-first; only fall to vlm for types mlx-lm lacks
            want_vlm = not self._mlx_lm_supports(self._model_type)
        vlm_supported = want_vlm and self._vlm_supports(self._model_type)
        if want_vlm and not vlm_supported:
            logger.warning(
                f"[{self._model_type or 'unknown'}] backend={backend!r} "
                f"requested/needed mlx-vlm but mlx-vlm lacks this model_type — "
                f"loading via mlx-lm instead."
            )
        return vlm_supported

    @staticmethod
    def _mlx_lm_supports(model_type: str) -> bool:
        """Check if mlx-lm can load this model_type.

        Mirrors `_vlm_supports`, but probes the mlx-lm registry the way
        `mlx_lm.utils._get_classes` resolves a model module: the raw
        `model_type` is first run through `MODEL_REMAPPING` (e.g.
        `mistral`->`llama`), then `mlx_lm.models.{resolved}` is imported. So
        gemma4 -> `mlx_lm.models.gemma4`, mistral -> `mlx_lm.models.llama`,
        etc. A type mlx-lm has no module for (after remapping) returns False
        and, under `backend=auto`, falls through to the mlx-vlm path.

        Done BEFORE loading any weights (analogous to `_vlm_supports`).
        Defensive: returns False on ImportError / any resolution failure.
        """
        if not model_type:
            return False
        try:
            import importlib

            try:
                from mlx_lm.utils import MODEL_REMAPPING
                resolved = MODEL_REMAPPING.get(model_type, model_type)
            except ImportError:
                # MODEL_REMAPPING absent in this mlx_lm build: probe raw type.
                resolved = model_type
            importlib.import_module(f"mlx_lm.models.{resolved}")
            return True
        except ImportError:
            return False

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
                            self._gpu_keepalive_ping()
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
        # NOTE: the shutdown flush (atexit + SIGINT/SIGTERM) used to be
        # registered HERE, which silently skipped it for main_thread/process
        # mode and for --gpu-keepalive off. It now lives in
        # _register_shutdown_flush(), called unconditionally from load_model.

    # --- Shutdown flush registration (mode-agnostic) -----------------------

    # Class-level once-guards: registration and the flush-run marker. Exposed
    # as class attrs (not closure state) so tests can reset/inspect them and
    # so the process worker can reuse the same handler.
    _shutdown_flush_registered = False
    _shutdown_flush_fn = None  # the registered flush callable (for tests/worker)

    def _register_shutdown_flush(self):
        """Register atexit + SIGINT/SIGTERM handlers that persist dirty
        sessions on a normal stop. MODE-AGNOSTIC and idempotent (class-level
        once); called from load_model for every execution mode.

        OWNERSHIP CONTRACT: in server mode UVICORN OWNS PROCESS LIFETIME —
        its SIGINT/SIGTERM handlers drive the graceful stop (serve loop
        winds down → lifespan shutdown → server.py's shutdown hook flushes
        again, idempotently). Our handler only PREPENDS a flush and then
        CHAINS to whatever handler was installed before us; it must never
        force process termination itself when someone else manages it. Only
        when the previous disposition was the OS default (bare scripts,
        tests) do we re-deliver the signal with SIG_DFL so the process still
        dies of it (correct exit status for supervisors).

        Per-mode safety contract:
          - thread mode ("worker"): generation runs on the _vlm_executor
            thread, so a main-thread handler can safely block (bounded) on
            the engine lock until an in-flight generation finishes.
          - main_thread mode (non-process): that mode's contract is
            main-thread-only MLX, and Python signal handlers run on the main
            thread, so flushing inline from the handler is allowed. If the
            signal interrupted an in-flight generation (which holds the
            NON-reentrant engine lock on this same thread), the bounded
            acquire in _flush_all_on_shutdown skips the flush, re-marks the
            ids dirty, and the retry happens downstream: the chained Uvicorn
            handler unwinds the generation (releasing the lock), then the
            lifespan shutdown hook and/or the atexit backstop re-run the
            flush against the LIVE dirty set.
          - process mode: the CHILD's worker loop REPLACES these signal
            handlers with a flag+sentinel pair (see process_worker.py) so
            the flush runs from the loop's finally, never inside a signal
            handler; the atexit hook registered here remains as a
            last-resort backstop (idempotent — the first flush drains the
            dirty set, so a second run is a no-op).
        """
        if MLXEngine._shutdown_flush_registered:
            return
        MLXEngine._shutdown_flush_registered = True

        import atexit
        import signal

        def _shutdown(*args):
            # NO once-guard here, deliberately: _flush_all_on_shutdown
            # re-reads the LIVE dirty set on every call, so a repeat run is
            # a cheap no-op when the first flush drained it — and a REAL
            # retry when a timed-out bounded lock acquire re-marked ids
            # dirty. The atexit backstop below relies on that retry.
            MLXEngine._global_keepalive_stop.set()
            logger.info("[Shutdown] Flushing dirty sessions...")
            try:
                MLXEngine._flush_all_on_shutdown()
            except Exception as e:  # noqa: BLE001 — never block/corrupt shutdown
                logger.error(f"[Shutdown] flush failed (continuing): {e}")
            logger.info("[Shutdown] Complete")

        MLXEngine._shutdown_flush_fn = _shutdown
        # atexit is the UNIVERSAL backstop: it runs on every interpreter
        # exit (normal stop, Uvicorn graceful shutdown, sys.exit) and
        # retries anything a timed-out flush re-marked dirty.
        atexit.register(_shutdown)

        # Capture the handlers installed BEFORE us (Uvicorn's, when
        # load_model runs inside server startup) so we can chain instead of
        # clobbering them.
        prev_handlers: dict = {}

        def _signal_handler(signum, frame):
            _shutdown()
            prev = prev_handlers.get(signum, signal.SIG_DFL)
            if callable(prev):
                # Chain: Uvicorn (or another wrapper) installed a handler
                # before us — hand the signal over so its graceful shutdown
                # still runs (lifespan shutdown → server.py hook → second
                # flush is an idempotent no-op / retry).
                prev(signum, frame)
                return
            if prev is signal.SIG_IGN:
                # The signal was being ignored before we registered — keep
                # ignoring it after our flush.
                return
            # SIG_DFL (or None — handler installed by non-Python code, which
            # we cannot invoke): restore the default disposition and
            # re-deliver so the process still dies of the signal.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        try:
            for _sig in (signal.SIGINT, signal.SIGTERM):
                prev_handlers[_sig] = signal.getsignal(_sig)
                signal.signal(_sig, _signal_handler)
        except ValueError:
            # signal.signal only works on the main thread (e.g. load_model
            # driven from a worker thread in tests) — atexit-only fallback.
            logger.debug(
                "[Shutdown] signal handlers not installed (non-main thread); "
                "atexit flush still registered"
            )

    def _touch_gpu(self):
        """Mark GPU as recently active (resets keepalive timer)."""
        MLXEngine._global_last_gpu_activity = time.time()

    def _gpu_keepalive_ping(self):
        """Tiny GPU op that keeps Metal clocked up (avoids the idle
        downclock that roughly doubles the next request's TTFT).

        This is the CORE OP of the thread-mode keepalive loop
        (_start_gpu_keepalive), factored out so the PROCESS-mode child's
        main loop (process_worker._gpu_keepalive_touch) can run the
        identical touch from its poll-timeout branch — that loop IS the
        child's main thread, so the main-thread-only MLX contract holds.

        CALLER MUST HOLD self._lock (non-blocking acquire recommended) so
        the ping can never overlap a generation.
        """
        a = mx.random.normal((32, 32))
        b = a @ a
        mx.eval(b)

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
        _sess_tools = getattr(session, "tools", None)
        metadata = {
            "session_id": session_id,
            "messages": json.dumps(session.messages, ensure_ascii=False),
            "total_cache_tokens": str(session.total_cache_tokens),
            "last_used": str(session.last_used),
            "token_ids": json.dumps(session.cache_state.token_ids or []),
            # Prompt contract (U3/U21) — must survive a restart so rebuilds
            # keep the tool schema and the HIT gate can verify it.
            "tools": json.dumps(_sess_tools, ensure_ascii=False) if _sess_tools else "",
            "thinking": "1" if getattr(session, "thinking", True) else "0",
            "prompt_fingerprint": getattr(session, "prompt_fingerprint", None) or "",
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
                if threading.get_ident() == getattr(self, "_vlm_worker_ident", None):
                    # RE-ENTRANCY: we are ALREADY running on the single
                    # _vlm_executor worker thread (e.g. post-generation eviction
                    # driven inside generate_stream's finally, which is itself
                    # invoked on the worker). Submitting to the same one-worker
                    # pool and blocking on fut.result() would deadlock against
                    # ourselves for the full 60s timeout. The cache tensors are
                    # already bound to THIS thread's generation_stream, so just
                    # materialize + serialize inline.
                    _do_save()
                else:
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

    # --- Active-session memory bounding (LRU eviction) ----------------------

    def _session_cache_bytes(self, session: "SessionState") -> int:
        """Estimate the resident KV bytes held by a single session."""
        cache = session.cache_state.cache if session.cache_state else None
        if cache is None:
            return 0
        return self.cache_manager._estimate_cache_size(cache)

    def _active_sessions_memory_gb(self) -> float:
        """Total resident KV memory (GB) across all in-memory sessions.

        This is the number that memory_budget_gb is supposed to bound but did
        NOT before: idle-flush saves dirty sessions to disk yet leaves their KV
        caches resident in self._sessions, so without active eviction they
        accumulate until the Mac OOMs.
        """
        return sum(
            self._session_cache_bytes(s) for s in self._sessions.values()
        ) / 1e9

    def _base_caches_memory_gb(self) -> float:
        """Total resident bytes (GB) held by the base-cache pool (U2).

        Sizes are measured once at registration (entries are immutable
        snapshots), so this is a cheap sum, not a re-walk."""
        base_caches = getattr(self, "_base_caches", None)
        if not base_caches:
            return 0.0
        return sum(e.size_bytes for e in base_caches.values()) / 1e9

    def _evict_base_caches_lru(
        self, over_gb_fn, *, mru_allowance_gb: "float | None" = None,
    ) -> tuple[int, int]:
        """Evict base caches LRU-first while ``over_gb_fn()`` is True (U2).

        MRU protection is CONDITIONAL (round 2, codex F3): the single
        most-recently-used entry is kept while it can actually FIT the
        budget — base caches are auto re-registered with a FULL secondary
        system-prompt prefill on the next MISS, so evicting an entry the
        active workload is using would thrash (register -> evict ->
        re-prefill every turn). But an UNCONDITIONAL keep-MRU defeats the
        budget entirely: a single base entry larger than the budget left
        ZERO eviction candidates, and the next clone could OOM.
        ``mru_allowance_gb`` is the budget headroom left for the MRU entry
        after everything this sweep can never reclaim (the protected/MRU
        session + the prefix pool — see the caller); if we are STILL over
        budget after the LRU pass and the MRU entry exceeds that allowance,
        it is evicted too (loud WARNING). ``mru_allowance_gb=None`` keeps
        the historical unconditional protection (direct callers without
        budget context). Base caches are memory-only derived state (never
        persisted), so eviction is a plain drop.

        Caller MUST hold ``self._lock`` — the same lock that guards
        registration (_register_base_cache runs under _generate_locked /
        _mutate_locked), so eviction never races a half-built entry.

        Returns (evicted_count, freed_bytes)."""
        base_caches = getattr(self, "_base_caches", None)
        if not base_caches:
            return 0, 0
        # LRU first; the last (MRU) entry is protected during this pass.
        by_recency = sorted(base_caches.items(), key=lambda kv: kv[1].last_used)
        lru_hashes = [h for h, _ in by_recency[:-1]]
        mru_hash = by_recency[-1][0]
        evicted = 0
        freed_bytes = 0
        for h in lru_hashes:
            if not over_gb_fn():
                break
            entry = base_caches.pop(h, None)
            if entry is None:
                continue
            evicted += 1
            freed_bytes += entry.size_bytes
            logger.info(
                f"[Base Cache] EVICTED (LRU, over budget) | hash={h} | "
                f"{entry.token_count} tokens | {entry.size_bytes / 1e6:.1f}MB "
                f"| hits={entry.hit_count}"
            )
        # Round 2 (codex F3): waive the MRU anti-thrash protection when the
        # entry provably cannot fit — still over budget after shedding every
        # other entry, and the MRU alone exceeds the remaining allowance.
        # When it FITS, MRU stays protected (anti-thrash intent preserved).
        if (
            mru_allowance_gb is not None
            and mru_hash in base_caches
            and over_gb_fn()
        ):
            entry = base_caches[mru_hash]
            entry_gb = entry.size_bytes / 1e9
            if entry_gb > mru_allowance_gb:
                base_caches.pop(mru_hash, None)
                evicted += 1
                freed_bytes += entry.size_bytes
                logger.warning(
                    f"[Base Cache] EVICTED MRU entry (budget cannot fit it) | "
                    f"hash={mru_hash} | {entry.token_count} tokens | "
                    f"{entry.size_bytes / 1e6:.1f}MB "
                    f"({entry_gb:.2f} GB > allowance "
                    f"{max(mru_allowance_gb, 0.0):.2f} GB) | anti-thrash "
                    f"protection waived — it will re-prefill on the next MISS; "
                    f"raise memory_budget_gb to keep it resident"
                )
        return evicted, freed_bytes

    def _mark_session_busy(self, session_id: str | None):
        if not session_id:
            return
        lock = getattr(self, "_busy_lock", None)
        busy = getattr(self, "_busy_sessions", None)
        if lock is None or busy is None:
            return
        with lock:
            busy.add(session_id)

    def _unmark_session_busy(self, session_id: str | None):
        if not session_id:
            return
        lock = getattr(self, "_busy_lock", None)
        busy = getattr(self, "_busy_sessions", None)
        if lock is None or busy is None:
            return
        with lock:
            busy.discard(session_id)

    # --- Drafter acceptance stats (U26) --------------------------------------

    def _accumulate_drafter_stats(
        self, session_id: str | None, n_rounds: int, total_accepted: int,
    ) -> None:
        """Fold one request's drafter acceptance numbers into the session's
        cumulative stats (U26). Keyed in ``_session_drafter_stats`` so the
        accumulation survives the per-turn SessionState reinstall; the live
        SessionState (when present) is pointed at the SAME dict so admin
        readers (list_sessions/get_session) surface it without change.
        Caller holds the engine lock (post-stream finalize / generation).

        Round 2 (codex F5a): the registry is a BOUNDED LRU
        (_DRAFTER_STATS_MAX) — deletes through non-engine lifecycles (or no
        delete at all) no longer grow it for the life of the process. Safe
        to evict: the stats are advisory display data (see the constant)."""
        if not session_id:
            return
        registry = getattr(self, "_session_drafter_stats", None)
        if registry is None or not isinstance(registry, OrderedDict):
            # Fresh (or legacy plain-dict — e.g. shell engines in tests):
            # promote to the LRU-ordered form preserving existing entries.
            registry = self._session_drafter_stats = OrderedDict(registry or {})
        stats = registry.setdefault(session_id, {
            "requests": 0,
            "total_rounds": 0,
            "total_accepted": 0,
        })
        registry.move_to_end(session_id)
        while len(registry) > _DRAFTER_STATS_MAX:
            old_sid, _ = registry.popitem(last=False)
            logger.debug(
                f"[Drafter] stats registry over cap "
                f"({_DRAFTER_STATS_MAX}) — evicted LRU entry "
                f"session={old_sid} (advisory stats only)"
            )
        stats["requests"] += 1
        stats["total_rounds"] += n_rounds
        stats["total_accepted"] += total_accepted
        session = self._sessions.get(session_id)
        if session is not None:
            session.drafter_stats = stats

    def _drafter_stats_for(self, session_id: str | None) -> dict | None:
        """The session's cumulative drafter stats dict (or None). Used by the
        SessionState install sites to carry the accumulator across the
        per-turn reinstall (U26)."""
        if not session_id:
            return None
        registry = getattr(self, "_session_drafter_stats", None)
        if not registry:
            return None
        return registry.get(session_id)

    def _evict_active_sessions_if_needed(self, protect_session_id: str | None = None):
        """Best-effort bound on total resident KV memory toward memory_budget_gb.

        Active per-session KV caches (self._sessions) are the dominant memory
        consumer but were previously unbounded — only delete_session removed
        them. This evicts the LEAST-RECENTLY-USED idle session when the active
        KV total PLUS the separate LRU prefix-reuse pool PLUS the base-cache
        pool (U2) exceeds the budget: the session is persisted to disk (so its
        next request transparently reloads it) and then dropped from
        self._sessions so MLX frees the buffers. Base caches are shed FIRST
        (LRU, CONDITIONAL keep-MRU — see _evict_base_caches_lru; round 2,
        codex F3: the MRU base entry is evicted too when it cannot fit the
        budget allowance left after the unevictable sessions): they are
        memory-only derived state, auto re-registered on the next MISS.

        Never evicts:
          - a session with an in-flight generation (_busy_sessions),
          - the just-used session (protect_session_id),
          - the single most-recently-used session (always keep one resident).

        The bound is BEST-EFFORT, not hard: because the protected/MRU/last
        session is always preserved, the sweep can legitimately exit still over
        budget when that one un-evictable session alone exceeds the budget. In
        that case it logs a WARNING and admin status reports
        ``budget_unmet=True`` (see status_dict's ``memory`` block) — the caller
        should raise memory_budget_gb or use fewer concurrent long sessions.

        Caller MUST hold ``self._lock`` (it persists caches + mutates
        self._sessions, exactly like _flush_dirty_sessions).
        """
        # Defensive: a partially-constructed shell engine (e.g. unit tests that
        # build via __new__ and only set cfg + _lock) has no session/cache
        # machinery — there is nothing to evict.
        if getattr(self, "_sessions", None) is None or getattr(self, "cache_manager", None) is None:
            return

        budget_gb = float(getattr(self.cfg, "memory_budget_gb", 0) or 0)
        if budget_gb <= 0:
            return

        pool_gb = self.cache_manager._memory_usage_gb()

        # U2: base caches are charged against the SAME memory_budget_gb as
        # the active sessions + the prefix-reuse pool — they are the third
        # (previously unaccounted, unbounded) resident-KV consumer.
        def _total_gb() -> float:
            return (
                self._active_sessions_memory_gb()
                + pool_gb
                + self._base_caches_memory_gb()
            )

        if _total_gb() <= budget_gb:
            return

        # U2: shed base caches FIRST (LRU, conditional keep-MRU) — they are
        # memory-only derived state, auto re-registered from the next MISS's
        # prefill, while evicting a session costs a disk save + reload. Only
        # if the total is still over budget does the session sweep below run.
        #
        # Round 2 (codex F3): the MRU base entry's protection is conditional
        # on it FITTING. Its allowance = budget minus everything this sweep
        # can never reclaim: the prefix pool plus the session(s) the session
        # sweep below always keeps resident (the protected just-used session
        # and the MRU session — often the same one). A base entry larger
        # than that allowance would leave the sweep permanently over budget
        # with zero candidates (the original U2 hole: an 80GB lone base
        # cache under a 64GB budget was never evictable), so it is evicted
        # too — see _evict_base_caches_lru.
        unevictable_gb = pool_gb
        if self._sessions:
            protected_sids = {
                max(
                    self._sessions.items(), key=lambda kv: kv[1].last_used
                )[0]
            }
            if protect_session_id and protect_session_id in self._sessions:
                protected_sids.add(protect_session_id)
            unevictable_gb += sum(
                self._session_cache_bytes(self._sessions[sid])
                for sid in protected_sids
            ) / 1e9
        base_evicted, base_freed = self._evict_base_caches_lru(
            lambda: _total_gb() > budget_gb,
            mru_allowance_gb=budget_gb - unevictable_gb,
        )
        if base_evicted:
            logger.info(
                f"[Base Cache] evicted {base_evicted} entr"
                f"{'y' if base_evicted == 1 else 'ies'}, freed "
                f"~{base_freed / 1e9:.2f} GB (over memory budget)"
            )
        if _total_gb() <= budget_gb:
            if base_evicted:
                # Base-cache shedding alone brought us under budget — release
                # the now-unreferenced Metal buffers before returning.
                try:
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                except Exception:  # noqa: BLE001
                    pass
            return

        with self._busy_lock:
            busy = set(self._busy_sessions)

        # Candidates = evictable session IDs, LRU first (oldest last_used
        # first). Hold IDs only (not the SessionState objects) so that once a
        # session is popped from _sessions nothing in this sweep keeps a strong
        # reference to its KV cache — otherwise mx.clear_cache() below could not
        # free the buffers while this list is still alive.
        candidate_ids = [
            sid
            for sid, _ in sorted(
                self._sessions.items(), key=lambda kv: kv[1].last_used
            )
        ]
        # The single most-recently-used session is always kept resident.
        keep_mru = candidate_ids[-1] if candidate_ids else None

        evicted = 0
        freed_bytes = 0
        for sid in candidate_ids:
            if _total_gb() <= budget_gb:
                break
            # len() check after each eviction so we never drop the last one.
            if len(self._sessions) <= 1:
                break
            if sid in busy:
                continue
            if protect_session_id and sid == protect_session_id:
                continue
            if sid == keep_mru:
                continue

            session = self._sessions.get(sid)
            if session is None:
                continue
            sess_bytes = self._session_cache_bytes(session)
            try:
                # Persist so the next request for this session reloads it from
                # disk (see _generate_locked / truncate / branch disk-cache
                # consult paths). save returning False = permanently unsaveable
                # cache (e.g. GLM empty arrays) — it would never round-trip, so
                # there is nothing on disk to reload either way.
                saved = self._save_session_to_disk(sid, session)
                save_error: Exception | None = None
            except Exception as e:  # noqa: BLE001
                # TRANSIENT failure (timeout, unexpected error). Do NOT drop the
                # session: dropping it here would lose the KV cache with no disk
                # copy to reload from, forcing a full from-scratch rebuild on the
                # client's next request. Keep it resident and skip this sweep.
                saved = False
                save_error = e

            if save_error is not None:
                logger.warning(
                    f"[Active LRU] session={sid} | SAVE FAILED, keeping resident "
                    f"(no durable copy — would be a lossy rebuild) | {save_error}"
                )
                continue

            # Persist-then-reload is only durable when the save actually
            # succeeded. A returning-False save is a PERMANENT unsaveable cache:
            # there is no disk copy to reload, but the session can rebuild from
            # its `messages`, so evicting it to reclaim RAM is acceptable. Any
            # other failure was already handled (continue) above.
            self._sessions.pop(sid, None)
            with self._dirty_lock:
                self._dirty_sessions.discard(sid)
            # Anon-provenance hygiene: best-effort discard so the set tracks
            # live sessions (a stale id would be harmless — the prefix scan
            # iterates self._sessions — this just bounds growth).
            anon_ids = getattr(self, "_anon_minted_ids", None)
            if anon_ids is not None:
                anon_ids.discard(sid)
            evicted += 1
            freed_bytes += sess_bytes
            if not saved:
                logger.warning(
                    f"[Active LRU] session={sid} | evicted WITHOUT disk save "
                    f"(cache not serializable) — will rebuild from messages"
                )
            # Drop our only remaining strong ref to the evicted session (and its
            # KV cache) so the buffers are unreferenced before clear_cache().
            session = None

        # The loop variable can still pin the last-inspected (possibly evicted)
        # session — drop it before asking MLX to release buffers.
        session = None

        if evicted:
            # Encourage MLX to release the now-unreferenced Metal buffers.
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass
            logger.info(
                f"[Active LRU] evicted {evicted} idle session(s), "
                f"freed ~{freed_bytes / 1e9:.2f} GB | "
                f"active KV now {self._active_sessions_memory_gb():.2f} GB "
                f"+ pool {pool_gb:.2f} GB "
                f"+ base {self._base_caches_memory_gb():.2f} GB "
                f"/ budget {budget_gb:.1f} GB"
            )

        # Best-effort, not a hard cap: the protected / MRU / last-remaining
        # session (and the MRU base cache) is never evicted. If that
        # un-evictable residue alone still exceeds the budget, surface it
        # loudly rather than silently lying.
        if _total_gb() > budget_gb:
            logger.warning(
                f"[Active LRU] still OVER budget after sweep: resident "
                f"{_total_gb():.2f} GB > budget {budget_gb:.1f} GB "
                f"(protected/MRU/last session + MRU base cache are "
                f"un-evictable) — raise memory_budget_gb or reduce "
                f"concurrent long sessions"
            )

    def _mark_dirty(self, session_id: str):
        """Mark a session for disk save on next idle cycle."""
        with self._dirty_lock:
            self._dirty_sessions.add(session_id)

    def _flush_dirty_sessions(self):
        """Flush all dirty sessions to disk. Caller MUST hold _lock.

        U16 — POP-ONE-SAVE-ONE (chosen over drain-then-iterate): the process
        worker's SIGTERM handler raises a ``BaseException`` sentinel
        (``_GracefulShutdown``) that can land while this runs from the idle
        flush. The old shape drained the WHOLE dirty set up front and only
        ``except Exception`` re-marked — a BaseException unwinding mid-loop
        lost every drained-but-unsaved id, so the shutdown flush saw an empty
        set and the sessions were silently dropped. Now each id is popped
        individually right before its save, and a BaseException re-marks the
        single in-flight id before re-raising — an interrupt therefore loses
        NOTHING (already-saved ids need no retry; unprocessed ids were never
        popped; the in-flight one is re-marked and the shutdown flush's own
        pass saves it).

        ``processed`` guards against re-popping an id this call already
        handled: a failing save re-marks its id for the NEXT flush cycle,
        which must not turn into an infinite retry loop here.

        F1 (codex batch-3 review, round 2): the ENTIRE pop→save→handle-error
        body sits inside ONE ``try``/``finally``. ``settled`` flips to True
        only once the in-flight id needs NO restore (saved, permanently
        unsaveable, evicted, or the ordinary-failure handler already made
        its own re-mark decision); the ``finally`` re-marks otherwise. The
        earlier shape used a sibling ``except BaseException`` re-mark — but
        a BaseException (SIGTERM sentinel) raised INSIDE the
        exception-handling path itself (during logging / just before the
        re-mark) escapes sibling excepts, so the popped id was lost. The
        ``finally`` re-mark covers every unwind path, and it is a bare
        ``set.add`` — cheap, idempotent, exception-safe. sid selection stays
        OUTSIDE the guard: an interrupt there finds the id still un-popped
        (nothing to restore). BaseException still propagates to abort the
        loop (never swallow the sentinel).
        """
        processed: set[str] = set()
        attempted = 0
        saved = 0
        while True:
            with self._dirty_lock:
                remaining = self._dirty_sessions - processed
                if not remaining:
                    break
                sid = next(iter(remaining))
            # True once this id needs NO finally re-mark (see F1 above).
            settled = False
            try:
                with self._dirty_lock:
                    self._dirty_sessions.discard(sid)
                processed.add(sid)
                session = self._sessions.get(sid)
                if session is None:
                    # Evicted since it was marked: drop, never re-mark.
                    settled = True
                    continue
                attempted += 1
                success = self._save_session_to_disk(sid, session)
                if success:
                    saved += 1
                # If success=False (permanent failure like empty arrays),
                # don't retry — either way the id is settled.
                settled = True
            except Exception as e:
                logger.error(f"[KV Cache] session={sid} | FLUSH SAVE FAILED | {e}")
                # Terminate this call's retry loop even if the failure landed
                # before processed.add ran (idempotent when it already did).
                processed.add(sid)
                with self._dirty_lock:
                    if sid in self._sessions:
                        self._dirty_sessions.add(sid)
                settled = True
            finally:
                # U16/F1: anything still un-settled here is a BaseException
                # unwind (shutdown sentinel / KeyboardInterrupt) — from the
                # pop→save gap, the save itself, or the except-block above.
                # Re-mark the in-flight id so the shutdown flush retries it;
                # the unwind then proceeds untouched. Re-adding an id the
                # interrupt caught BEFORE the pop is a harmless no-op.
                if not settled:
                    with self._dirty_lock:
                        self._dirty_sessions.add(sid)

        if saved:
            logger.info(f"[Idle Flush] saved {saved}/{attempted} dirty sessions")

    @classmethod
    def _flush_all_on_shutdown(cls, lock_timeout: float = 10.0):
        """Save all dirty sessions across all engines on shutdown.

        IDEMPOTENT: an engine whose dirty set is empty is skipped with a
        cheap PEEK (no lock acquire), so a second call (worker-loop flush
        followed by the atexit backstop, or atexit after a signal handler)
        is a cheap no-op.

        LOCK-FIRST ORDER (codex round 7, finding 2b): the engine lock is
        acquired BEFORE anything is drained from the dirty set. The old
        shape drained a snapshot first and RE-MARKED it when the bounded
        lock acquire timed out — that re-mark could land AFTER a concurrent
        straggler's self-flush (which holds the engine lock) had already
        drained the set, so the re-marked ids were stranded dirty forever
        (the self-flush was the last flush that could ever run). With the
        lock acquired first, a timeout drained NOTHING: the ids simply stay
        marked, and the lock-holding straggler's self-flush (which rescans
        — finding 2c) saves them before releasing the lock.

        FAIL-CLOSED: the lock acquire stays BOUNDED. In main_thread mode a
        signal handler runs on the same thread as an in-flight generation
        that holds the non-reentrant lock — a blocking acquire there would
        self-deadlock shutdown forever. On timeout the engine is skipped
        with a warning (nothing drained, nothing lost). Per-session and
        per-engine save failures are logged and never propagate.

        Draining goes through ``_flush_dirty_sessions``, which keeps the
        U16/R2-F1 pop-one-save-one + settled-flag semantics intact: each id
        is popped individually right before its save, a BaseException
        unwind re-marks exactly the in-flight id via the ``finally``
        re-mark, and the loop keeps draining ids marked while it runs
        (bounded by ``processed`` — each id at most once per call).
        """
        for engine in cls._all_engines:
            try:
                # Cheap idempotency peek — never touches the engine lock.
                with engine._dirty_lock:
                    n_dirty = len(engine._dirty_sessions)
                if not n_dirty:
                    continue
                logger.info(
                    f"[Shutdown] Flushing {n_dirty} dirty sessions for "
                    f"{engine.model_id}"
                )
                # Finding 2b: LOCK FIRST — only a holder of the engine lock
                # may drain the dirty set during shutdown.
                if not engine._lock.acquire(timeout=lock_timeout):
                    logger.warning(
                        f"[Shutdown] engine lock not acquired within "
                        f"{lock_timeout}s (in-flight generation?) — skipping "
                        f"flush for {engine.model_id} (nothing drained; the "
                        f"ids stay marked for the lock holder's self-flush)"
                    )
                    continue
                try:
                    engine._flush_dirty_sessions()
                finally:
                    engine._lock.release()
            except Exception as e:  # noqa: BLE001 — never block shutdown
                logger.error(
                    f"[Shutdown] flush failed for engine "
                    f"{getattr(engine, 'model_id', '?')}: {e}"
                )

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
            # Prompt contract (U3/U21). Legacy files (pre-fingerprint) load
            # as tools=None / fingerprint=None — the HIT gate then takes ONE
            # unconditional cold rebuild that stamps the fingerprint (F5:
            # never a lenient HIT; the legacy contract is unknowable).
            _tools_meta = metadata.get("tools", "")
            sess_tools = json.loads(_tools_meta) if _tools_meta else None
            sess_thinking = metadata.get("thinking", "1") != "0"
            sess_fp = metadata.get("prompt_fingerprint") or None

            # Verify loaded cache matches model structure (leading slice).
            # MTP-finalized sessions (qwen_mtp) persist n_target + n_head
            # entries, but the finalize hidden the MTP gate would need is an
            # in-memory stash that is never written to disk — MTP reuse is
            # impossible after a restart regardless. Plain reuse of the full
            # token history is what matters: accept the load and STRIP the
            # extra trailing (head) entries, so the next turn plans
            # REUSE_FALLBACK_PLAIN over the whole history instead of
            # cold-filling. Stripping is gated to EXACTLY the qwen_mtp
            # finalized layout (engine mtp-capable + n_extra == head layer
            # count + trailing entries are KVCache heads at
            # len(token_ids) - 1); any other oversized layout is rejected.
            model_cache = make_prompt_cache(self._language_model)
            n_model = len(model_cache)
            if len(cache) < n_model:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                    f"layer count mismatch: {len(cache)} vs {n_model}"
                )
                return None

            type_ok = all(
                type(c).__name__ == type(m).__name__
                for c, m in zip(cache[:n_model], model_cache)
            )
            if not type_ok:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | cache type mismatch"
                )
                return None

            if len(cache) > n_model:
                # Fail-closed strip — tightened contract: extra trailing
                # entries are ONLY strippable when they are exactly THIS
                # server's qwen_mtp finalized layout, i.e. ALL of:
                #   (a) the engine is qwen-mtp-capable right now
                #       (_mtp_base_caches_active: mlx-lm backend, qwen_mtp
                #       drafter loaded, no --kv-bits),
                #   (b) n_extra == the drafter's head layer count (the same
                #       count make_head_cache was sized with at save time),
                #   (c) every trailing extra is a head entry of the exact
                #       type make_head_cache produces (KVCache), sitting at
                #       the finalized lazy-last-slot offset
                #       len(token_ids) - 1 (head trails the target by one),
                #   (d) after dropping them, EVERY offset-bearing target
                #       layer sits exactly at len(token_ids).
                # Anything else — a foreign cache with extra target layers,
                # a same-type larger layout, a plain/non-MTP server reading
                # a head-carrying file — is a layout we don't understand:
                # reject the whole load (as before the strip feature).
                from mlx_soloheaven.engine.pld import _layer_offsets
                from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
                n_extra = len(cache) - n_model
                if not self._mtp_base_caches_active():
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"{n_extra} extra trailing entries but this engine is "
                        f"not qwen-mtp-capable — refusing to strip an unknown "
                        f"layout"
                    )
                    return None
                _n_head = max(1, len(getattr(self._drafter, "layers", [])) or 1)
                if n_extra != _n_head:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"{n_extra} extra trailing entries != drafter head "
                        f"layer count {_n_head} — not the MTP-finalized layout"
                    )
                    return None
                _head_type = type(qwen_mtp_mod.make_head_cache(1)[0])
                _head_off = len(token_ids) - 1
                _bad_head = [
                    (n_model + i, type(c).__name__, getattr(c, "offset", None))
                    for i, c in enumerate(cache[n_model:])
                    if type(c) is not _head_type
                    or getattr(c, "offset", None) != _head_off
                ]
                if _bad_head:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"trailing entries are not finalized MTP head entries "
                        f"(need {_head_type.__name__} at offset {_head_off}): "
                        f"{_bad_head[:8]}"
                    )
                    return None
                cache = cache[:n_model]
                _bad = [
                    (i, off)
                    for i, off in enumerate(_layer_offsets(cache))
                    if off is not None and off != len(token_ids)
                ]
                if _bad:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"stripped {n_extra} trailing entries but target "
                        f"offsets != {len(token_ids)} stored ids: {_bad[:8]}"
                    )
                    return None
                logger.info(
                    f"[KV Cache] session={session_id} | DISK LOAD | stripped "
                    f"{n_extra} trailing MTP head entries ({n_model}-layer "
                    f"target) — session continues via plain-fallback reuse"
                )

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
                tools=sess_tools,
                thinking=sess_thinking,
                prompt_fingerprint=sess_fp,
                # U26: a disk-reloaded session reclaims its in-memory drafter
                # stats (the registry outlives active-LRU eviction).
                drafter_stats=self._drafter_stats_for(session_id),
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

    # --- Prompt contract (U3/U21) ----------------------------------------

    @staticmethod
    def _canonical_tools(tools: list | None) -> list | None:
        """Canonical (JSON-serializable) form of a request's tool schema.

        Pydantic models are dumped to plain dicts so the result can be
        stored on SessionState, persisted in safetensors metadata, and
        hashed deterministically. ``None``/empty → ``None`` (toolless)."""
        if not tools:
            return None
        return [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]

    @staticmethod
    def _prompt_fingerprint(tools_canonical: list | None, thinking: bool) -> str:
        """Hash of everything that alters the tokenized prompt prefix
        OUTSIDE the messages: the canonical tool schema + the thinking flag.
        Compared on every HIT (U21) — a mismatch means the cached prefix was
        built under a different contract and reuse would silently keep the
        stale schema in context, so the turn takes an honest MISS instead."""
        payload = json.dumps(
            {"tools": tools_canonical, "thinking": bool(thinking)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

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
        self, messages: list[dict], cache: list, system_tokens: list[int],
        tools: list | None = None, mtp_resume_hidden=None,
    ):
        """Register a base cache from the system prompt portion of a processed cache."""
        h = self._system_hash(messages, tools=tools)
        # Existing entries are kept (skip, never overwrite) — a stale plain
        # 40-entry base under an MTP server still seeds sessions correctly:
        # the MTP gate takes the plain-decode fallback instead of
        # cold-filling, so reuse is preserved either way.
        if not h or h in self._base_caches:
            return
        # Deep copy the cache at current state (after system prompt processing)
        import copy
        base_snapshot = copy.deepcopy(cache)
        self._eval_cache(base_snapshot)
        # U2: measure the entry once at registration — the same helper that
        # byte-counts session caches for the memory budget. The MTP boundary
        # hidden is tiny (1,1,H) but counted for honesty. getattr guard:
        # partially-constructed shell engines (unit tests via __new__) have no
        # cache_manager — size 0 there, never a registration failure.
        _cm = getattr(self, "cache_manager", None)
        size_bytes = _cm._estimate_cache_size(base_snapshot) if _cm else 0
        hidden_nbytes = getattr(mtp_resume_hidden, "nbytes", 0)
        if isinstance(hidden_nbytes, int):
            size_bytes += hidden_nbytes
        entry = BaseCacheEntry(
            system_hash=h,
            cache=base_snapshot,
            tokens=system_tokens,
            token_count=len(system_tokens),
            mtp_layout=mtp_resume_hidden is not None,
            mtp_resume_hidden=mtp_resume_hidden,
            size_bytes=size_bytes,
        )
        self._base_caches[h] = entry
        logger.debug(
            f"[Base Cache] REGISTERED | hash={h} | {len(system_tokens)} tokens | "
            f"{size_bytes / 1e6:.1f}MB | mtp={entry.mtp_layout} | "
            f"pool_size={len(self._base_caches)}"
        )

    def _clone_base_cache(self, base: BaseCacheEntry) -> list:
        """Clone a base cache for a new session."""
        import copy
        cloned = copy.deepcopy(base.cache)
        # Force evaluation of cloned arrays to avoid lazy-eval aliasing issues
        self._eval_cache(cloned)
        base.hit_count += 1
        # U2: LRU touch — every actual use (clone) marks recency for the
        # budget-driven base-cache eviction sweep.
        base.last_used = time.time()
        logger.debug(
            f"[Base Cache] CLONE | hash={base.system_hash} | "
            f"{base.token_count} tokens | hits={base.hit_count}"
        )
        return cloned

    def base_cache_stats(self) -> list[dict]:
        """Return stats for all base caches.

        U15: reads ``_base_caches`` under the engine lock (bounded — raises
        EngineBusyError while a generation holds it) so a concurrent
        registration/clear never surfaces a half-built entry."""
        with self._read_locked("base cache stats"):
            return [
                {
                    "system_hash": e.system_hash,
                    "token_count": e.token_count,
                    "hit_count": e.hit_count,
                    "created": e.created,
                    "mtp": e.mtp_layout,
                    # U2: byte accounting + LRU recency.
                    "size_mb": round(e.size_bytes / 1e6, 1),
                    "last_used": e.last_used,
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
        (offset >= max_size, the is_trimmable() boundary). Once wrapped the
        physical buffer holds only the
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
            # >= (not >): is_trimmable() is already False at offset ==
            # max_size, so prefix-trim reuse is impossible there too
            # (matches pld._wrapped_rotating_layers' boundary).
            if offset >= max_size:
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
        # Defensive: RoPE continuation needs cache.offset == logical history
        # length. token_ids is reconciled to cache.offset at save time (see
        # the post-generation update), so these match for a strict append;
        # any residual mismatch (e.g. a cache that advanced beyond the
        # recorded ids) is genuinely unsafe and must cold-fill.
        for c in cache:
            offset = getattr(c, "offset", None)
            if offset is not None and int(offset) != prefix_len:
                logger.warning(
                    f"[KV Cache] offset/ids mismatch ({type(c).__name__}."
                    f"offset={int(offset)} != prefix_len={prefix_len}) — cold-fill"
                )
                return False
        return True

    def _reject_kv_bits_on_rotating_cache(self) -> None:
        """FIX-7: refuse ``--kv-bits`` for models whose cache contains
        RotatingKVCache (sliding-window) layers.

        ``mlx_lm.models.cache.RotatingKVCache.to_quantized`` raises
        ``NotImplementedError`` ("RotatingKVCache Quantization NYI"), and
        ``maybe_quantize_kv_cache`` gates only on ``hasattr(c, "to_quantized")``
        — which RotatingKVCache satisfies — so kv_bits>0 would crash
        MID-GENERATION as soon as ``quantized_kv_start`` is reached, instead
        of failing at startup. (The session prefix-reuse path would also
        break: ``QuantizedKVCache.keys`` is a list of arrays, not a tensor.)
        """
        if not self.cfg.kv_bits or self._use_vlm:
            return
        if getattr(self, "_has_rotating_cache", False):
            raise ValueError(
                f"[{self.model_id}] --kv-bits={self.cfg.kv_bits} is not "
                f"supported for this model: its cache contains "
                f"RotatingKVCache (sliding-window attention, window="
                f"{getattr(self, '_sliding_window_size', '?')}) layers, and "
                f"mlx-lm's RotatingKVCache does not implement KV quantization "
                f"(to_quantized raises NotImplementedError). Remove --kv-bits "
                f"(use bf16 KV cache) for this model."
            )

    @staticmethod
    def _flatten_cache_layers(cache) -> list:
        """Flatten CacheList-like containers into their leaf cache objects.

        Codex round 3, finding 1: GLM-5.1 / DeepSeek-V3.2 (MLA + DSA
        indexer) wrap TWO KVCaches per layer in a top-level ``CacheList``
        that exposes NO ``offset`` of its own — offset discovery that only
        inspects top-level attributes reads 0 for the whole cache. Containers
        expose their children via ``.caches`` (the established mlx-lm idiom —
        the same attribute cache/manager.py's size estimator recurses
        through); anything without it is a leaf and is returned as-is."""
        flat: list = []
        for c in cache or []:
            sub = getattr(c, "caches", None)
            if sub is not None:
                flat.extend(MLXEngine._flatten_cache_layers(sub))
            else:
                flat.append(c)
        return flat

    @staticmethod
    def _leaf_trimmable(c) -> bool:
        """Codex round 5, finding 2: POSITIVE trimmability for one flattened
        cache leaf. Method presence alone is insufficient — RotatingKVCache
        exposes ``trim()`` while ``is_trimmable()`` is False once the ring
        has wrapped (a trim decrements offsets so the post-trim per-layer
        verification passes, but the evicted window entries cannot be
        restored — the rewound cache is semantically corrupt). Honor the
        semantic gate when the leaf provides one; a probe failure is
        unverifiable → not trimmable (fail-closed).

        NOTE: this gates the POST-STREAM reconcile trim-back only. The
        mlx-vlm speculative ROLLBACK path (rejected draft tokens trimmed in
        the same step they were appended) deliberately keeps upstream's
        unconditional trim — see the B3/RCA-2 note in the MTP patch module.
        """
        if not hasattr(c, "trim"):
            return False
        probe = getattr(c, "is_trimmable", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:  # noqa: BLE001 — unverifiable → fail-closed
                return False
        return True

    @staticmethod
    def _get_cache_offset(cache: list) -> int:
        """Get the total number of tokens processed by this cache.

        Prefers KVCache (full attention, accurate cumulative offset) over
        RotatingKVCache (offset is cumulative but size() caps at max_size).

        Codex round 3, finding 1a: discovery recurses through CacheList-like
        containers (GLM MoE/DSA layouts) — the leaf KVCaches carry the real
        offsets even when the top-level entries expose none.
        """
        layers = MLXEngine._flatten_cache_layers(cache)
        # First pass: look for unbounded KVCache (full attention layers)
        for c in layers:
            if type(c).__name__ == "KVCache" and hasattr(c, "offset"):
                return c.offset
        # Fallback: any cache with offset (RotatingKVCache, ArraysCache, etc.)
        for c in layers:
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

    def _prefill_cache(self, cache: list, tokens: list[int], cancel_event=None):
        """Process tokens through the language model to populate a KV cache.

        U13: ``cancel_event`` is checked BETWEEN chunks — a client disconnect
        during a long prefill aborts within one chunk instead of burning
        minutes of GPU. Raises ``GenerationCancelled``; the caller owns the
        partially-filled cache's fate (a fresh cache is simply discarded).

        F4 (codex batch-3 review, round 2): checked once more AFTER the
        final chunk — the between-chunk checks alone let a disconnect
        DURING the last chunk slip through to the caller's follow-up work
        (cache eval here, base-cache registration upstream)."""
        arr = mx.array(tokens)
        for i in range(0, len(tokens), self._PREFILL_STEP):
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled(
                    f"prefill cancelled at {i}/{len(tokens)} tokens"
                )
            chunk = arr[i : i + self._PREFILL_STEP]
            self._language_model(chunk[None], cache=cache)
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled(
                f"prefill cancelled after final chunk ({len(tokens)} tokens)"
            )
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
    def _normalize_for_match(
        content, role: str, model_family: str = "chatml",
        thinking_active: bool = True,
    ) -> str:
        """Normalize message content for comparison.

        Codex round 9, finding 2: the assistant thinking-channel reduction
        follows the authoritative router machinery under the message's
        ``thinking_active`` contract (threaded from _messages_match — the
        round-3/8 threading covered the tool-call helpers but left this one
        flag-blind and suffix-only):

        - gemma4: the content channel is the MULTI-CYCLE union of all
          content segments (_content_channel_union). The old
          last-``<channel|>`` slice matched suffix-only (multi-cycle stored
          'thought → important → thought → done' wrong-HIT against a bare
          incoming 'done'), and the unconditional bare ``thought\\n`` strip
          let a thinking-DISABLED stored turn genuinely starting with those
          words ('thought\\nSECRET<channel|>answer' — all content under the
          router's enable_thinking gate) wrong-HIT against 'answer'.
        - chatml/glm, thinking active: FIRST-close reduction under the
          router contract (codex round 11, finding 1 — the router never
          re-enters thinking, tool_parser.content_segments). A raw side
          (leading ``<think>`` — the engine always stores the opener, see
          _make_full_assistant_content) reduces to the text after the
          FIRST ``</think>``; any later ``</think>`` is a literal quote
          INSIDE the content channel. An already-split side (no leading
          opener) is kept WHOLE — the old LAST-close (rindex) reduction
          collapsed distinct content channels onto their final suffix, so
          a forged plain resend of that suffix wrong-HIT even on the
          strict anon path. The degenerate UNCLOSED shape (leading opener,
          no close) is kept RAW — codex round 13, finding 1; see the
          inline comment.
        - chatml/glm, thinking DISABLED: pure pass-through — a literal
          ``</think>`` in content is a quote, never a boundary (round 3,
          finding 4 semantics, previously missing here).
        """
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
            # Strip thinking blocks (channel reduction per family/contract —
            # see the docstring).
            if model_family == "gemma4":
                content = _content_channel_union(
                    content, model_family, thinking_active,
                )
            elif thinking_active:
                # Codex round 11, finding 1: FIRST-close channel reduction.
                # The router contract (content_segments /
                # split_thinking_and_content) makes the FIRST ``</think>``
                # authoritative — chatml/glm never re-enter thinking, so any
                # later ``</think>`` is a literal quote INSIDE the content
                # channel (both families share the <think> markers). Shape
                # alignment:
                # - leading ``<think>``: raw wire text (the engine always
                #   stores the opener — _make_full_assistant_content);
                #   content is everything after the FIRST close, quotes
                #   included. No close at all → kept RAW (codex round 13,
                #   finding 1; see below).
                # - no leading opener: the text IS a content channel — a
                #   literal ``</think>`` in it is a quote, never a boundary.
                #   Kept WHOLE: shapes that cannot be aligned one-to-one
                #   against a stored raw (e.g. a bare leading ``</think>``
                #   with no open) must never normalize onto a plain resend
                #   — an honest MISS, not a forgeable suffix HIT (the old
                #   rindex reduction collapsed distinct content channels
                #   onto their final suffix).
                body = content.lstrip()
                if body.startswith("<think>"):
                    close = body.find("</think>", len("<think>"))
                    if close == -1:
                        # Codex round 13, finding 1: an UNCLOSED raw (leading
                        # opener, no close) is kept RAW — returned verbatim
                        # (outer whitespace trim only), skipping every
                        # reduction including the tool-call strips below.
                        # Under an active chatml/glm contract the router
                        # routes an unclosed stream ENTIRELY to reasoning
                        # with EMPTY content, so the legacy bare-opener strip
                        # normalized stored '<think>SECRET' equal to a plain
                        # incoming 'SECRET' — a forgeable suffix even on the
                        # strict/anon path. Normalizing to '' (the router-
                        # faithful content channel) would instead collide
                        # with genuinely-empty assistant content — another
                        # many-to-one. The legitimate interrupted-resend
                        # equivalence is handled by _interrupted_resend_equiv
                        # (which splits the channel itself under exact
                        # rules), never by plain equality: an unclosed-
                        # thinking turn must normalize equal to nothing but
                        # its byte-identical self.
                        return content.strip()
                    content = body[close + len("</think>"):]
            # Strip tool call blocks (both ChatML and Gemma 4 formats)
            content = _NORMALIZE_RE_TOOL_CALL_CHANNEL.sub("", content)
            content = _NORMALIZE_RE_TOOL_CALL_XML.sub("", content)
        return content.strip()

    @staticmethod
    def _canonicalize_calls(calls: list) -> list:
        """Canonical ``[(name, canonical-args-json), ...]`` form of a list
        of OpenAI-shaped tool-call dicts. JSON-string arguments are parsed
        so a re-serialized resend (key order / whitespace) still compares
        equal. Call ids are deliberately EXCLUDED: XML-parsed calls get
        fresh random ids, and the tool ROLE's tool_call_id comparison
        already pins the id chain; name + canonical arguments are what make
        two calls "the same call" for cache validity."""
        out: list = []
        for tc in calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            out.append((
                fn.get("name") or "",
                json.dumps(args, sort_keys=True, ensure_ascii=False, default=str),
            ))
        return out

    @staticmethod
    def _content_xml_calls_for_match(
        raw_content: str, model_family: str, thinking_active: bool = True,
    ) -> list:
        """Canonical calls parsed from tool-call XML embedded in RAW content
        (legacy engine-stored turns kept the template XML in content) —
        empty list when no start marker is present or nothing parses.

        This is THE content-side parse chain, shared by
        _tool_calls_for_match (legacy fallback), _has_residual_tool_markers
        (N3) and _structured_content_call_conflict (round 4): the family
        parser first, then the N2 bare-name GLM canonicalization for a
        COMPLETE block the family parser could not read
        (``<tool_call>name[<arg_key>..</arg_value>]</tool_call>`` — a shape
        some clients replay verbatim), so a legitimate bare-name turn
        compares STRUCTURALLY instead of hitting the one-sided FAIL. A
        PARTIAL block (no end marker after the start) stays unparseable on
        purpose: against a side with >=1 canonical call it is mismatch
        evidence, not indeterminate.

        Round 5: matching requires CLOSED blocks. The GLM parser
        intentionally tolerates a MISSING ``</tool_call>`` closer (the
        ``\\Z`` alternate — generation-time robustness for a stream cut
        mid-block), so an UNCLOSED trailing GLM block canonicalized exactly
        like the closed one — equal to the structured call it mirrors — and
        sailed past the one-sided (N2), residual-marker (N3) AND
        structured-content conflict (round 4) checks straight into the
        marker-strip/wildcard acceptance, despite the tokenized prompts
        differing. The parser's leniency is deliberately untouched (do not
        change generation-time parsing); at MATCH time the lenient parse is
        DISCOUNTED: at most one call can come from an unclosed block (only
        the final ``\\Z``-terminated match) and it is always the LAST parsed
        call, so when more calls parsed than CLOSED
        ``<tool_call>...</tool_call>`` spans exist the trailing call is
        dropped. The dropped call then surfaces through the existing checks
        (one-sided / residual marker / self-conflict). A spurious drop can
        only reject (honest MISS), never produce a wrong HIT.

        U12: only the CONTENT channel is scanned — tool-call XML inside a
        thinking segment is a rehearsal the engine no longer records, so at
        match time it must not count as an extractable call (or a residual
        marker) either; otherwise every turn after a rehearsal would FAIL
        the one-sided check against the client's clean resend. Round 2: the
        channel reduction is the UNION of all content segments
        (_content_channel_union) — gemma4 multi-cycle output keeps a
        legitimate call in an EARLIER content cycle visible here, while
        thinking segments stay excluded. Round 3 (finding 4):
        ``thinking_active`` is the router-active state the message's turn
        ran under (the session's stored thinking flag) — with thinking
        disabled a literal ``</think>`` in content is a quote, not a
        boundary, so a call BEFORE it stays extractable."""
        raw_content = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, end_tag = get_tool_markers(model_family)
        if not raw_content or start_tag not in raw_content:
            return []
        _, calls = parse_tool_calls(raw_content, model_family=model_family)
        if (
            not calls
            and start_tag == "<tool_call>"
            and end_tag in raw_content[raw_content.find(start_tag):]
        ):
            _, calls = _parse_glm_tool_calls(raw_content)
        if calls and start_tag == "<tool_call>":
            # Round 5 closed-block discount (see docstring). The closed-span
            # count uses the same non-greedy left-to-right scan as the GLM
            # block pattern, so an orphan closer BEFORE the first start
            # marker cannot masquerade as a closer for a trailing unclosed
            # block (a naive closers-vs-starts count would miss that).
            closed = sum(
                1 for _ in _NORMALIZE_RE_TOOL_CALL_XML.finditer(raw_content)
            )
            if len(calls) > closed:
                calls = calls[:-1]
        return MLXEngine._canonicalize_calls(calls)

    @staticmethod
    def _tool_calls_for_match(
        msg: dict, raw_content: str, model_family: str,
        thinking_active: bool = True,
    ) -> list | None:
        """F3: canonical, comparable form of an assistant message's tool
        calls — ``[(name, canonical-args-json), ...]`` or ``None`` when no
        call is extractable.

        Sources, in order:
        - the structured ``tool_calls`` field (OpenAI clients / the engine's
          normal save);
        - tool-call XML embedded in the RAW content (legacy engine-stored
          turns kept the template XML in content) via
          _content_xml_calls_for_match.

        When the structured field is present it WINS and content XML is not
        consulted here — _structured_content_call_conflict (round 4) is the
        companion check that rejects a message whose content XML DISAGREES
        with its structured list."""
        out = MLXEngine._canonicalize_calls(msg.get("tool_calls") or [])
        if out:
            return out
        # Fall back to tool-call XML in the raw content (legacy stored shape).
        return (
            MLXEngine._content_xml_calls_for_match(
                raw_content, model_family, thinking_active,
            )
            or None
        )

    @staticmethod
    def _structured_content_call_conflict(
        msg: dict, raw_content: str, model_family: str,
        thinking_active: bool = True,
    ) -> bool:
        """Round 4: True when a message carries BOTH tool-call
        representations — a structured ``tool_calls`` field AND tool-call
        start markers in its RAW content — and they do NOT agree
        canonically.

        _tool_calls_for_match returns the STRUCTURED list as soon as it is
        non-empty, ignoring content entirely, so a side holding the same
        structured call but content that ALSO embeds a fully parseable
        ``<tool_call>`` block for a DIFFERENT call (e.g. delete_files)
        compared canonically EQUAL to a side with just the structured call;
        _has_residual_tool_markers stayed False (every marker parses) and
        the marker-strip content shortcut in _messages_match then accepted
        the message on both the strict and the lenient path — despite the
        tokenized prompts differing.

        When both representations exist they must agree: the canonical
        SEQUENCE of content-XML calls must equal the canonical structured
        list. A start marker whose block fails to parse counts as
        disagreement (same rule as the one-sided garbled block, N2). A
        self-consistent turn — the same call(s) rendered both ways, or
        either representation alone — never trips this; a false positive
        (a literal marker inside prose next to a structured call) degrades
        to an honest MISS, never a wrong HIT."""
        structured = MLXEngine._canonicalize_calls(msg.get("tool_calls") or [])
        if not structured:
            return False
        # U12: a marker inside a thinking segment is a rehearsal, not a
        # conflicting representation — only content-channel XML must agree
        # with the structured list. Round 2: the channel is the UNION of all
        # content segments (multi-cycle aware); the RAW content is handed to
        # _content_xml_calls_for_match, which applies the same reduction
        # itself (the reduction is not idempotent for chatml text quoting a
        # literal ``</think>`` in its content, so it must run exactly once).
        content_union = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, _ = get_tool_markers(model_family)
        if not content_union or start_tag not in content_union:
            return False
        return (
            MLXEngine._content_xml_calls_for_match(
                raw_content or "", model_family, thinking_active,
            )
            != structured
        )

    @staticmethod
    def _has_residual_tool_markers(
        raw_content: str, model_family: str, thinking_active: bool = True,
    ) -> bool:
        """Round 3 (N3): True when ``raw_content`` holds tool-call START
        markers the family parse chain does NOT consume into canonical calls
        (a partial/garbled block trailing — or preceding — valid blocks).

        _tool_calls_for_match returns only the calls that PARSE, so a side
        with one valid block plus a residual ``<tool_call>`` fragment compares
        canonically EQUAL to a side with just the valid block — and the
        marker-strip content shortcut in _messages_match then accepts the
        message on both the strict and the lenient path. Residual markers are
        mismatch evidence (same rule as the one-sided garbled block, N2).

        Detection mirrors _tool_calls_for_match's exact parse chain (family
        parser, then the N2 bare-name GLM fallback) and compares the number
        of canonical calls against the number of start-marker occurrences.
        Counting the WHOLE content is consistent with existing semantics: the
        family parsers scan the full text (a marker inside a code fence /
        quote is already treated as tool-call territory by parse_tool_calls),
        and each parsed block consumes exactly one start marker. A false
        positive (e.g. a parameter VALUE containing the literal marker)
        degrades to an honest MISS, never a wrong HIT — and only when the two
        sides' raw contents already differ (byte-identical sides skip this
        check entirely).

        U12: markers inside thinking segments are rehearsals, not
        residuals — only the content channel is counted. Round 2: the
        channel is the UNION of all content segments (mirrors the reduction
        in _content_xml_calls_for_match, so parsed-call and marker counts
        stay aligned; the RAW content is handed down so the reduction runs
        exactly once)."""
        content_union = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, _ = get_tool_markers(model_family)
        if not content_union or start_tag not in content_union:
            return False
        calls = MLXEngine._content_xml_calls_for_match(
            raw_content or "", model_family, thinking_active,
        )
        return len(calls) < content_union.count(start_tag)

    def _interrupted_resend_equiv(
        self, s_content: str, i_norm: str, *, exact: bool = False,
        thinking_active: bool = True,
    ) -> bool:
        """U1 narrow equivalence for a C1-committed INTERRUPTED assistant turn.

        The stored message (marked ``interrupted=True`` by
        _commit_interrupted_hit_turn) holds the engine-side content INCLUDING
        thinking; the client resends only what it received on the wire —
        the content channel (thinking stripped), possibly TRUNCATED at the
        cancel point, or the thinking reconstructed as plain text.

        ``exact=False`` (EXPLICIT session ids only — F2): accept exactly
        these shapes and nothing else:

        - incoming (normalized) is a PREFIX of the stored CONTENT channel
          (includes the empty resend — a cancel before any content frame);
        - incoming (normalized) is a PREFIX of
          ``{stored thinking}\\n\\n{stored content}`` — the client replayed
          the (possibly truncated) thinking as plain content.

        ``exact=True`` (the ANON resolver / strict path — F2): prefix
        equivalence is FORGEABLE for session-less requests (an empty resend
        would match every interrupted turn, and any shared prefix could
        select another conversation's session), so anon resolution requires
        EXACT content-channel equality after thinking-strip normalization —
        an empty resend matches ONLY an empty stored content channel.

        Arbitrary divergence — anything that is not (a prefix of, or under
        ``exact`` equal to) what was actually streamed — does NOT match
        (honest MISS / new anon session), unlike the old
        last-stored-assistant wildcard this replaces.

        ``thinking_active`` (codex round 9, finding 2): the stored turn's
        thinking contract, threaded into the split AND the normalization —
        the default-active split treated a thinking-DISABLED gemma4 turn's
        bare ``thought\\n`` content as a reasoning span, so its content
        channel shrank to the post-``<channel|>`` suffix and a forged bare
        resend passed even the EXACT (anon/strict) rule."""
        s_content = s_content or ""
        family = getattr(self, "model_family", "chatml")
        started = s_content.lstrip().startswith("<think>")
        if family == "gemma4" or (thinking_active and started):
            # gemma4 keeps its positional-router split; a chatml/glm stored
            # raw always carries the leading opener
            # (_make_full_assistant_content), and the split's Case 1 is
            # already FIRST-close (non-greedy) — any later ``</think>`` in
            # the resulting content is a literal quote it preserves.
            thinking, content = split_thinking_and_content(
                s_content,
                model_family=family,
                started_in_thinking=started,
                thinking_active=thinking_active,
            )
        else:
            # Codex round 11, finding 1: no leading opener (or thinking
            # disabled) — the stored text IS the content channel and a
            # literal ``</think>`` in it is a quote, never a boundary. The
            # old unconditional split reduced it at the close, so a forged
            # resend of the post-quote suffix passed even the EXACT
            # (anon/strict) rule. Kept whole → honest MISS instead.
            thinking, content = None, s_content
        c_norm = self._normalize_for_match(
            content or "", "assistant", family, thinking_active,
        )
        if exact:
            return c_norm == i_norm
        if c_norm.startswith(i_norm):
            return True
        if thinking:
            t_norm = thinking.strip()
            combo = f"{t_norm}\n\n{c_norm}".strip() if c_norm else t_norm
            if combo.startswith(i_norm):
                return True
        return False

    def _messages_match(
        self,
        stored: list[dict],
        incoming: list[dict],
        *,
        last_assistant_wildcard: bool = True,
        thinking_active: bool = True,
    ) -> bool:
        """Check if incoming messages start with the stored conversation.

        ``last_assistant_wildcard`` (U1/F2): when True (explicit-session HIT
        path — the historical behavior), a content mismatch on the LAST
        stored assistant message is tolerated wholesale (client may hold a
        truncated/reformatted view of the reply), interrupted turns get the
        NARROW prefix equivalence, and the thinking-prepend suffix leniency
        applies. The anon prefix RESOLVER passes False — STRICT mode: for
        session-less requests every one of those leniencies is a forgery
        vector (a DIFFERENT anonymous conversation — prefix-equal except an
        assistant turn — could hijack another conversation's session and
        cache), so anon resolution requires EXACT normalized assistant
        content; an interrupted turn matches only on exact content-channel
        equality (empty resend == empty stored content only) and the
        generic suffix equivalence is bypassed entirely.

        Structured fields (F3) are compared on BOTH paths whenever present:
        assistant ``tool_calls`` (canonically serialized name+arguments,
        incl. legacy XML-in-content stored turns) and the tool role's
        ``tool_call_id`` — two turns with identical (even empty) content but
        different calls are different conversations.

        Round-2 hardening (N1/N2):
        - N1: the compacted/cleared tool-result equivalence ("[cleared]" /
          "[compacted:…]" placeholders) applies to EXPLICIT sessions only —
          in strict (anon) mode a placeholder would match ANY stored tool
          result, so exact tool-result content is required.
        - N2: when tool calls are extractable on exactly ONE side, the
          message FAILS on both paths — merely containing a start marker no
          longer defers to the content rules (the marker-strip shortcut
          could accept a valid stored call against a partial/different
          block). Complete bare-name blocks are canonicalized first (see
          _tool_calls_for_match), so only genuine parse failures reject.

        Round-3 hardening (N3): RESIDUAL unparsed tool-call start markers
        (a valid call followed by a partial/different block — canonically
        equal to the valid call alone) are mismatch evidence when the two
        sides' raw contents differ: FAIL on both paths (see
        _has_residual_tool_markers).

        Round-4 hardening: a message carrying BOTH representations —
        structured ``tool_calls`` AND tool-call XML in content — must agree
        with itself; a conflicting side (the structured comparison prefers
        the field, so content XML for a DIFFERENT call was invisible) FAILS
        on both paths (see _structured_content_call_conflict).

        Codex round 3, finding 4: ``thinking_active`` is the session's
        stored thinking contract — threaded into the content-channel
        reduction of every helper below so a literal ``</think>`` in a
        NON-thinking turn's content is a quote, never a channel boundary
        that hides tool-call XML from matching. Callers pass the STORED
        session's flag (the HIT gate only reaches here after the
        fingerprint check, so it equals the request's flag there; the anon
        resolver passes each candidate's own flag)."""
        if len(incoming) < len(stored):
            logger.debug(
                f"[Match] FAIL: incoming({len(incoming)}) < stored({len(stored)})"
            )
            return False
        for i, s_msg in enumerate(stored):
            i_msg = incoming[i]
            if s_msg.get("role") != i_msg.get("role"):
                # DEBUG (not INFO): _messages_match runs PER CANDIDATE session
                # in the anon prefix scan, so per-candidate FAILs are expected
                # noise; the aggregate "no prefix match among N sessions" line
                # stays at INFO.
                logger.debug(
                    f"[Match] FAIL at msg[{i}]: role {s_msg.get('role')!r} != {i_msg.get('role')!r}"
                )
                return False
            s_content = self._flatten_multipart(s_msg.get("content"))
            i_content = self._flatten_multipart(i_msg.get("content"))
            role = s_msg.get("role", "")

            # F3: structured fields participate in matching on ALL paths —
            # content comparison alone lets two assistant turns with equal
            # (often empty) content but DIFFERENT tool calls match, poisoning
            # context across conversations (A's cached call + B's tool
            # result). Compared via MLXEngine directly so lightweight test
            # stubs without the helper bound still work.
            family = getattr(self, "model_family", "chatml")
            if role == "assistant":
                s_calls = MLXEngine._tool_calls_for_match(
                    s_msg, s_content, family, thinking_active,
                )
                i_calls = MLXEngine._tool_calls_for_match(
                    i_msg, i_content, family, thinking_active,
                )
                if s_calls is not None and i_calls is not None:
                    if s_calls != i_calls:
                        logger.debug(
                            f"[Match] FAIL at msg[{i}]: assistant tool_calls "
                            f"differ ({s_calls} != {i_calls})"
                        )
                        return False
                elif s_calls != i_calls:
                    # Exactly one side carries extractable calls (a non-None
                    # result always holds >=1 canonical call) — FAIL on BOTH
                    # paths (N2). This is mismatch evidence, never
                    # indeterminate:
                    # - the bare side shows NO tool-call evidence at all (no
                    #   structured field, no start marker) — structurally
                    #   different conversations (the historical rule); or
                    # - it carries a start marker whose block FAILS to parse
                    #   (even after the bare-name canonicalization fallback
                    #   in _tool_calls_for_match) against >=1 canonical call
                    #   on the other side. Pre-fix, merely containing the
                    #   marker fell through and the marker-strip shortcut
                    #   below accepted a VALID stored call against an
                    #   arbitrary partial/different <tool_call> block — a
                    #   session-forgery vector on the STRICT (anon) path and
                    #   cross-conversation poisoning on the lenient one. A
                    #   legitimate resend that still fails to canonicalize
                    #   degrades to an honest MISS, never a wrong HIT.
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: tool_calls extractable on "
                        f"one side only (stored="
                        f"{'yes' if s_calls is not None else 'no'}, incoming="
                        f"{'yes' if i_calls is not None else 'no'})"
                    )
                    return False
                # N3 (round 3): RESIDUAL unparsed tool-call start markers.
                # _tool_calls_for_match returns only the calls that PARSE, so
                # a side with one VALID call plus a trailing partial/different
                # <tool_call> fragment compared canonically EQUAL to a side
                # with just the valid call — and the marker-strip shortcut
                # below then accepted on BOTH the strict and the lenient
                # path. A residual marker on a side whose raw content differs
                # from the other side is mismatch evidence (same rule as the
                # one-sided garbled block above): FAIL on both paths.
                # Byte-identical contents carry no divergence to hide, so
                # verbatim replays of a degenerate turn still match.
                if s_content != i_content and (
                    MLXEngine._has_residual_tool_markers(
                        s_content, family, thinking_active,
                    )
                    or MLXEngine._has_residual_tool_markers(
                        i_content, family, thinking_active,
                    )
                ):
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: residual unparsed "
                        f"tool-call marker(s) on a differing side "
                        f"(stored_len={len(s_content)}, "
                        f"incoming_len={len(i_content)})"
                    )
                    return False
                # Round 4: a side carrying BOTH representations (structured
                # tool_calls AND tool-call XML in content) must agree with
                # ITSELF. The canonical comparison above prefers the
                # structured field, so a side whose content ALSO holds a
                # fully parseable block for a DIFFERENT call (no residual —
                # every marker parses) compared EQUAL, and the marker-strip
                # shortcut below then accepted it on the strict AND the
                # lenient path despite the tokenized prompts differing. A
                # self-conflicting side is mismatch evidence on BOTH paths;
                # content XML that canonically AGREES with the structured
                # list (the same call rendered both ways) never trips this.
                if (
                    MLXEngine._structured_content_call_conflict(
                        s_msg, s_content, family, thinking_active,
                    )
                    or MLXEngine._structured_content_call_conflict(
                        i_msg, i_content, family, thinking_active,
                    )
                ):
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: structured tool_calls "
                        f"conflict with tool-call XML in content on at "
                        f"least one side (stored_len={len(s_content)}, "
                        f"incoming_len={len(i_content)})"
                    )
                    return False
            elif role == "tool":
                s_tcid = s_msg.get("tool_call_id") or ""
                i_tcid = i_msg.get("tool_call_id") or ""
                if (s_tcid or i_tcid) and s_tcid != i_tcid:
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: tool_call_id "
                        f"{s_tcid!r} != {i_tcid!r}"
                    )
                    return False

            # Codex round 13, finding 2: the historical first-<tool_call>
            # PREFIX SHORTCUT that lived here (initial commit — OpenCode
            # strips tool-call XML from content and moves it to the
            # structured tool_calls field, so stored
            # "text\n\n<tool_call>..." had to match incoming "text")
            # compared ONLY the text before the FIRST start marker and then
            # accepted the whole message on BOTH paths. Everything after
            # that marker was invisible: divergent content AFTER the call
            # blocks ("...ALPHA" vs "...BETA"), and — because a rehearsal
            # call inside a thinking segment also carries the marker (U12:
            # not extractable, so the structural comparison above sees
            # None on both sides) — the entire post-thinking content
            # channel, a wrong-HIT vector even in strict/anon mode. It is
            # REMOVED: its two legitimate shapes are covered end-to-end by
            # the modern machinery — (a) call lists compare STRUCTURALLY
            # above (F3/N2, with N3/round-4 rejecting any unvalidated
            # marker on a differing side), and (b) _normalize_for_match
            # strips the validated closed blocks, so the SURROUNDING
            # content channel is byte-compared below; both must agree. The
            # streaming-truncated mid-call stored turn is not a lost
            # leniency either: the N3 residual check above already
            # rejected that shape before the shortcut could run (honest
            # MISS), on the strict and the lenient path alike.

            # Normalize and compare (round 9, finding 2: the channel
            # reduction inside runs under the same family/contract as the
            # tool-call helpers above).
            s_norm = self._normalize_for_match(
                s_content, role, family, thinking_active,
            )
            i_norm = self._normalize_for_match(
                i_content, role, family, thinking_active,
            )
            if s_norm != i_norm:
                # Tool content compacted/cleared by client (either direction)
                # KV cache still valid — the tokens were already processed.
                # N1: EXPLICIT sessions only (last_assistant_wildcard=True).
                # On the STRICT anon path a "[cleared]"/"[compacted:…]"
                # placeholder is equivalent to EVERY stored tool result, so
                # with this leniency active two anonymous conversations
                # sharing a call chain (same tool_call_ids) could resolve
                # onto each other's session — anon resolution requires EXACT
                # tool-result content.
                if (
                    role == "tool"
                    and last_assistant_wildcard
                    and self._is_compacted_tool(s_content, i_content)
                ):
                    logger.debug(
                        f"[Match] msg[{i}] tool content compacted — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # U1/F2: last-stored-assistant tolerance is CONDITIONAL now.
                # - C1-committed interrupted turns (any position): the NARROW
                #   prefix equivalence (thinking-channel stripping /
                #   wire-truncation shapes) is reserved for EXPLICIT session
                #   ids (last_assistant_wildcard=True). The anon resolver
                #   (False) requires EXACT content-channel equality — prefix
                #   shapes are forgeable for session-less requests (an empty
                #   resend would match every interrupted turn).
                # - Otherwise, the historical last-stored-assistant wildcard
                #   applies ONLY when the caller allows it (explicit-session
                #   HIT keeps current behavior; the anon resolver disables it
                #   — the hijack vector).
                if role == "assistant":
                    if s_msg.get("interrupted"):
                        if self._interrupted_resend_equiv(
                            s_content, i_norm,
                            exact=not last_assistant_wildcard,
                            thinking_active=thinking_active,
                        ):
                            logger.debug(
                                f"[Match] msg[{i}] interrupted-turn "
                                f"{'narrow' if last_assistant_wildcard else 'exact'} "
                                f"equivalence — accepting (stored="
                                f"{len(s_content)}, incoming={len(i_content)})"
                            )
                            continue
                        # Narrow/exact rule failed: fall through to the
                        # generic leniencies below, then the FAIL log — an
                        # interrupted turn never gets the wildcard.
                    elif i == len(stored) - 1 and last_assistant_wildcard:
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
                # F2: explicit sessions only — for the anon resolver this
                # generic suffix equivalence accepts non-prefix values and is
                # therefore forgeable; strict mode bypasses it.
                if role == "assistant" and last_assistant_wildcard and (
                    s_norm and len(s_norm) >= 8
                ) and (
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
                # (Safe on BOTH paths since F3: when calls are extractable on
                # both sides the structured comparison above has already
                # verified they are the SAME calls — this leniency only
                # bridges the content-channel reconstruction.)
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
                # DEBUG (not INFO): per-candidate diagnostic — see the role
                # FAIL above; aggregate summaries stay at INFO.
                logger.debug(
                    f"[Match] FAIL at msg[{i}] role={role}: "
                    f"stored_len={len(s_content)} vs incoming_len={len(i_content)} | "
                    f"diff at char {diff_pos} | "
                    f"stored=...{s_ctx!r}... | incoming=...{i_ctx!r}..."
                )
                return False
        return True

    def _resolve_anon_session_id_locked(
        self, messages: list[dict], prompt_fingerprint: str | None = None,
    ) -> str:
        """Resolve a session-less request onto a concrete per-conversation id.

        Session-less requests (no OpenAI ``user`` field — e.g. OpenCode and
        most agents) used to collapse onto the single shared ``"anon"`` key, so
        any two interleaved conversations (agent A vs agent B, or a client
        changing its system prompt mid-flow) thrashed the one slot: every
        request from the *other* conversation was a MISS → full cold-fill.

        Instead, scan the resident sessions for one whose stored
        ``session.messages`` is a (proper or exact) prefix of the incoming
        messages, reusing the same ``_messages_match`` logic the HIT path
        uses — but with ``last_assistant_wildcard=False`` (U1): the HIT
        path's last-stored-assistant content wildcard would let a DIFFERENT
        anonymous conversation, prefix-equal except its last assistant turn,
        hijack this conversation's session + cache (cross-conversation
        pollution). Candidate selection is therefore STRICT content
        matching; the only divergence tolerated is the narrow
        interrupted-turn equivalence (see _interrupted_resend_equiv), which
        applies identically on the HIT path, so selection and the subsequent
        cache decision still cannot disagree. Pick the LONGEST matched
        message prefix; tie-break by most-recently-used. An EXACT match
        (stored == incoming) is selected too — _generate_locked then takes
        its existing "retry" path (same messages re-sent → discard cache,
        re-process), unchanged.

        ``prompt_fingerprint`` (U21 interplay): when both sides carry a
        fingerprint, a mismatching candidate is skipped — two conversations
        with identical message prefixes but different tool schemas are
        DIFFERENT conversations and must not share a session slot. A
        ``None`` on either side (legacy session / direct caller) skips the
        filter; the HIT gate still applies the contract check downstream.

        If nothing matches, mint a NEW unique ``anon-<8 hex>`` id so each
        session-less conversation owns its own cache entry. These sessions
        live in ``self._sessions`` like any other, so the active-session LRU
        eviction (memory_budget_gb) bounds them; busy-protection applies
        because the caller busy-marks the id returned HERE.

        Scope notes:
        - Linear scan is fine: the session count is small and LRU-bounded by
          _evict_active_sessions_if_needed; _messages_match early-outs on
          length/role before any content comparison.
        - In-memory candidates only. Enumerating disk-cached sessions would
          need a safetensors metadata read PER candidate PER request just to
          compare messages — an evicted anon conversation simply degrades to
          one cold-fill (exactly today's behavior) and is resident again
          afterwards.
        - The legacy ``"anon"`` slot participates only when the ENGINE's own
          fallback minted it (see _generate_locked): provenance, not name. A
          pre-existing disk "anon" cache from before this feature simply stops
          being reused by anonymous requests — one cold-fill, no migration.

        Caller MUST hold ``self._lock`` (``self._sessions`` and
        ``self._anon_minted_ids`` are mutated under it) and must busy-mark the
        RETURNED id, not "anon".
        """
        # Defensive: partially-constructed shell engines (unit tests build via
        # __new__) may lack _sessions / _anon_minted_ids — same pattern as the
        # eviction sweep.
        sessions = getattr(self, "_sessions", None) or {}
        minted = getattr(self, "_anon_minted_ids", None)
        if minted is None:
            minted = set()
        best_sid: str | None = None
        best_len = -1
        best_used = -1.0
        for sid, session in sessions.items():
            # PROVENANCE-based isolation: only sessions the engine itself
            # minted for session-less requests are candidates. A name check
            # (sid.startswith("anon-")) is NOT enough — ChatCompletionRequest
            # .user is unrestricted and passed verbatim as session_id, so a
            # client could create an EXPLICIT session keyed "anon-aaaaaaaa";
            # it must never be selected — and then mutated — by an anonymous
            # request, even on a perfect message-prefix match. Stale ids in
            # the set (session since evicted/deleted) are harmless: the scan
            # iterates self._sessions and merely checks membership; removal
            # points additionally discard ids best-effort (see
            # _evict_active_sessions_if_needed / delete_session /
            # clear_caches).
            if sid not in minted:
                continue
            stored = session.messages
            # Empty stored messages would prefix-match ANYTHING — skip.
            if not stored or len(stored) > len(messages):
                continue
            # U21: different tool contract → different conversation, even on
            # a perfect message-prefix match (None on either side = legacy /
            # unknown → filter skipped, HIT gate decides downstream).
            _sess_fp = getattr(session, "prompt_fingerprint", None)
            if (
                prompt_fingerprint is not None
                and _sess_fp is not None
                and _sess_fp != prompt_fingerprint
            ):
                continue
            if not self._messages_match(
                stored, messages, last_assistant_wildcard=False,
                # Round 3, finding 4: the candidate's own stored thinking
                # contract drives its content-channel reduction.
                thinking_active=getattr(session, "thinking", True),
            ):
                continue
            n = len(stored)
            if n > best_len or (n == best_len and session.last_used > best_used):
                best_sid = sid
                best_len = n
                best_used = session.last_used
        if best_sid is not None:
            logger.info(
                f"[KV Cache] anon request matched session={best_sid} "
                f"(prefix {best_len}/{len(messages)} msgs)"
            )
            return best_sid
        new_sid = f"anon-{uuid.uuid4().hex[:8]}"
        # Record provenance AT MINT TIME so the next anonymous request can
        # find this session (shell engines without the set just lose tracking,
        # which only means no reuse — never cross-session mutation).
        if getattr(self, "_anon_minted_ids", None) is not None:
            self._anon_minted_ids.add(new_sid)
        logger.info(
            f"[KV Cache] anon request: no prefix match among "
            f"{len(sessions)} sessions — new session={new_sid}"
        )
        return new_sid

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

    def _suffix_blocking_assistants(self, new_messages: list[dict]) -> int:
        """U4 gate: count assistant messages in ``new_messages`` that the
        suffix path cannot represent — ANY non-resident assistant blocks.

        An assistant message past the stored prefix is by definition NOT
        cache-resident (the stored messages are the authoritative record of
        what was generated into the KV — the shape arises from crash
        recovery, where the disk-persisted session lags the conversation the
        client resends). Skipping it — the builders' historical behavior —
        silently drops the model's own prior reply from its context.

        Policy (F4, per review): on EVERY template a non-resident assistant
        turn routes to divergence → honest MISS (full re-tokenization). A
        manual splice is NOT token-exact against apply_chat_template — e.g.
        the Qwen3.6 ChatML template emits a '<think>\\n\\n</think>\\n\\n'
        prefix inside past assistant turns when thinking is disabled, and
        trims message content, while a naive
        '\\n<|im_start|>assistant\\n{content}<|im_end|>' splice preserves
        boundary whitespace — a cache-poisoning risk (spliced tokens != what
        full tokenization would produce, silently corrupting every later
        turn built on the cache). A cold-fill is cheap and always correct.

        NOTE for a future splice attempt: it must be proven by a REAL-TOKEN
        differential test — token ids of (cached-prefix + spliced suffix)
        compared against tokenizer.apply_chat_template over the full
        message list, on the actual installed template(s) — not by a
        fabricated encoded-string assertion.
        """
        return sum(1 for m in new_messages if m.get("role") == "assistant")

    def _suffix_tokens(
        self, new_messages: list[dict], thinking: bool = True,
    ) -> list[int]:
        """Compute suffix tokens for new messages to append to stored token_ids.

        This avoids full re-tokenization (which breaks special token round-trip)
        by directly encoding only the new message suffix in model-specific format.

        Assistant messages never appear in ``new_messages`` here: a
        non-resident assistant (crash-recovery resend) is routed to an
        honest MISS by the U4 gate in _generate_locked
        (_suffix_blocking_assistants) on ALL templates — a manual splice is
        not token-exact against apply_chat_template (F4).
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
                # U4: only reachable for cache-resident turns — the engine's
                # _suffix_blocking_assistants gate routes any NON-resident
                # assistant (crash-recovery resend) to an honest MISS before
                # this builder runs, because splicing model turns through
                # gemma4's special-token round-trip is template-risky.
                continue
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
        """ChatML suffix: \\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

        U4/F4: only reachable for cache-resident turns — a NON-resident
        assistant (crash-recovery resend) was routed to an honest MISS by
        the _suffix_blocking_assistants gate. A manual assistant splice is
        NOT token-exact vs apply_chat_template (e.g. Qwen3.6 renders a
        '<think>\\n\\n</think>\\n\\n' prefix into past assistant turns when
        thinking is disabled, and trims content) — see the gate's docstring
        for the differential-test requirement before ever splicing here."""
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
                # U4: only reachable for cache-resident turns — non-resident
                # assistants were routed to an honest MISS by the
                # _suffix_blocking_assistants gate (GLM's <think> handling
                # makes spliced model turns template-risky).
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
        stop: str | list[str] | None = None,
    ) -> Generator[GenerationResult, None, None]:
        """Generate with session-based KV cache reuse (holds lock)."""
        # U24: normalize OpenAI's ``stop`` (string or array) to a list of
        # non-empty sequences once, up front (empty -> None: no scanning).
        if isinstance(stop, str):
            stop = [stop]
        stop_sequences = [s for s in (stop or []) if s] or None
        t_wait = time.perf_counter()
        logger.debug(
            f"[Queue] session={session_id or 'anon?'} | waiting for lock | "
            f"messages={len(messages)}"
        )
        # U13: the lock wait is CANCEL-AWARE — a client that disconnects
        # while queued behind another generation must not start its own.
        # threading.Lock has no wait-with-predicate, so poll with a bounded
        # acquire and check the event between attempts.
        if not self._acquire_lock_cancellable(cancel_event):
            logger.info(
                f"[Queue] session={session_id or 'anon?'} | cancelled while "
                f"waiting for engine lock — request dropped (no generation)"
            )
            return
        try:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            # Resolve session-less requests to a concrete anon-* session id
            # ONCE, inside the lock and BEFORE busy-marking. The resolved id is
            # what gets busy-protected, stored in self._sessions, logged, and
            # evicted/saved — resolving later (inside _generate_locked) would
            # busy-mark the wrong key and let the post-generation eviction
            # sweep evict the cache mid-use. Explicit session_ids keep exact-key
            # behavior unchanged (no prefix scanning). The resolver gets the
            # request's prompt-contract fingerprint (U21) so two anon
            # conversations with identical message prefixes but different
            # tool schemas never share a slot.
            # U9: mirror _generate_locked's structured-output thinking
            # suppression so the anon resolver fingerprints the CONTRACT the
            # generation will actually stamp (thinking=False for
            # json_schema/json_object requests).
            _fp_thinking = (
                thinking if thinking is not None else self.cfg.enable_thinking
            )
            if _fp_thinking and getattr(response_format, "type", None) in (
                "json_schema", "json_object",
            ):
                _fp_thinking = False
            sid = session_id or self._resolve_anon_session_id_locked(
                messages,
                prompt_fingerprint=self._prompt_fingerprint(
                    self._canonical_tools(tools),
                    _fp_thinking,
                ),
            )
            logger.debug(f"[Queue] session={sid} | lock acquired | waited={wait_ms:.0f}ms")
            yield GenerationResult(status="generating")
            # Mark the session in-flight so the post-generation eviction sweep
            # (and any concurrent admin inspection) never evicts a cache that is
            # being mutated. Unmarked + eviction run in finally so cancellation
            # / errors still clean up and bound memory.
            self._mark_session_busy(sid)
            try:
                yield from self._generate_locked(
                    messages,
                    max_tokens=max_tokens if max_tokens is not None else self.cfg.default_max_tokens,
                    temperature=temperature if temperature is not None else self.cfg.default_temperature,
                    top_p=top_p if top_p is not None else self.cfg.default_top_p,
                    min_p=min_p if min_p is not None else self.cfg.default_min_p,
                    top_k=top_k if top_k is not None else self.cfg.default_top_k,
                    repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cfg.default_repetition_penalty,
                    session_id=sid,
                    tools=tools,
                    cancel_event=cancel_event,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    response_format=response_format,
                    stop_sequences=stop_sequences,
                )
            finally:
                # Clear busy BEFORE eviction so this session is a normal LRU
                # candidate; protect_session_id keeps the just-used one resident.
                self._unmark_session_busy(sid)
                try:
                    self._evict_active_sessions_if_needed(protect_session_id=sid)
                except Exception as e:  # noqa: BLE001 — never break generation
                    logger.error(f"[Active LRU] post-generation eviction failed: {e}")
        finally:
            self._lock.release()

    # U13: bounded-acquire poll interval while waiting for the engine lock
    # with a live cancel_event (worst-case cancel latency in the queue).
    _LOCK_CANCEL_POLL_S = 0.1

    def _acquire_lock_cancellable(self, cancel_event) -> bool:
        """Acquire the (non-reentrant, global) engine lock; with a
        ``cancel_event``, poll bounded acquires and give up as soon as the
        event is set. Returns True when the lock is HELD by the caller,
        False when the wait was cancelled (lock NOT held).

        Shutdown gate (codex round 3, finding 2): the gate is checked
        IMMEDIATELY AFTER a successful acquire — a generation that was
        queued behind the engine lock when the server began shutting down
        must not start (it would advance session state + mark it dirty
        AFTER the shutdown flush). Generations that already HOLD the lock
        are untouched: the gate only stops NEW acquisitions."""
        if cancel_event is None:
            self._lock.acquire()
        else:
            while True:
                if cancel_event.is_set():
                    return False
                if self._lock.acquire(timeout=self._LOCK_CANCEL_POLL_S):
                    break
        try:
            self._check_shutdown_gate("generation")
        except BaseException:
            self._lock.release()
            raise
        return True

    # U15/U14: bounded lock wait for READ-ONLY session/cache queries
    # (list/stats/overview). A generation can hold the engine lock for
    # minutes — admin/health reads must not hang that long, but they must
    # not read _sessions mid-mutation either. On timeout they raise
    # EngineBusyError and the API layer degrades (busy placeholder / 503).
    _READ_LOCK_TIMEOUT_S = 10.0

    @contextlib.contextmanager
    def _read_locked(self, what: str):
        """Bounded-acquire context manager for read-only query paths.

        Raises ``EngineBusyError`` when the engine lock cannot be acquired
        within ``_READ_LOCK_TIMEOUT_S`` (generation in flight). MUTATING
        admin paths (delete/clear) take the lock UNBOUNDED instead —
        correctness over latency, and U14's ``asyncio.to_thread`` wrappers
        keep the event loop responsive while they wait.

        Deliberately NOT shutdown-gated (finding 2): reads mutate nothing,
        so a post-flush read is harmless — and /health should stay live
        through a graceful shutdown."""
        if not self._lock.acquire(timeout=self._READ_LOCK_TIMEOUT_S):
            raise EngineBusyError(
                f"engine busy (generation in flight) — {what} not served "
                f"within {self._READ_LOCK_TIMEOUT_S:.0f}s, retry shortly"
            )
        try:
            yield
        finally:
            self._lock.release()

    # --- engine-side shutdown gate (codex batch-3 round 3, finding 2) -----
    #
    # Future.cancel() cannot stop an executor worker that has STARTED but is
    # BLOCKED on the engine lock behind a generation. The executors module's
    # quiesce() therefore returns after its bounded wait with such work
    # still "running"; the server then flushes; and a queued mutation
    # (compact/truncate/branch/delete/clear/generation) could acquire the
    # engine lock AFTER the flush, mutate state, mark it dirty — and nothing
    # would ever flush again. The gate closes that: begin_shutdown() is
    # called by the server shutdown hook BEFORE the flush, and every
    # mutating lock acquisition checks it IMMEDIATELY AFTER acquiring —
    # aborting with EngineBusyError WITHOUT mutating. Chokepoints, not
    # per-method sprinkles:
    #   * _mutate_locked      — the lock-taking mutating session/cache ops
    #                           (compact_session, delete_session,
    #                           clear_caches/reset — and, codex round 5
    #                           finding 3b, the WHOLE truncate/branch/
    #                           regenerate wrappers: their prelude disk
    #                           reload + _sessions publication is inside
    #                           the gate, not only the inner rebuild);
    #   * _acquire_lock_cancellable — generation's lock acquire (a queued
    #                           generation must not START post-flush; ones
    #                           already holding the lock are untouched —
    #                           Uvicorn's graceful shutdown owns live
    #                           connections).
    # Already-ENTERED mutations (codex round 5, finding 3a) are handled by
    # _mutate_locked's in-flight counter + self-flush-on-exit, and the
    # begin_shutdown cancel event aborts compaction/rebuild prefills
    # cooperatively — see _mutate_locked / _self_flush_on_shutdown_exit.
    # EXEMPT by design: the shutdown flush itself (_flush_all_on_shutdown /
    # _flush_dirty_sessions acquire the lock directly — the flush MUST run),
    # and _read_locked (reads mutate nothing; /health stays live).
    # update_session_messages (lock-free touch+mark-dirty) needs no gate:
    # it never blocks on the engine lock, and since round 5 finding 1a the
    # generation marks its session dirty engine-side at install — a
    # post-flush touch adds no un-flushed state beyond LRU recency.

    def begin_shutdown(self):
        """Flip the engine into shutdown mode BEFORE the shutdown flush:
        every subsequent mutating engine-lock acquisition aborts with
        ``EngineBusyError`` without mutating (see the block comment above).
        Codex round 5, finding 3a: also sets ``_shutdown_cancel_event`` so
        an ALREADY-ENTERED compaction/rebuild aborts its prefill between
        chunks (cooperative cancel) instead of delaying shutdown by
        minutes. Idempotent; never blocks. (getattr: shell engines built
        via ``__new__`` in tests may not have run ``__init__``.)"""
        if not getattr(self, "_shutting_down", False):
            logger.info(
                f"[Shutdown] engine {getattr(self, 'model_id', '') or '?'} — "
                f"mutation gate closed (stragglers become no-ops)"
            )
        self._shutting_down = True
        cancel_ev = getattr(self, "_shutdown_cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.set()

    def _check_shutdown_gate(self, what: str):
        """Raise ``EngineBusyError`` (503-mappable, like every busy path)
        once ``begin_shutdown()`` ran. Callers MUST invoke this immediately
        after acquiring the engine lock and before any mutation."""
        if getattr(self, "_shutting_down", False):
            raise EngineBusyError(
                f"engine shutting down — {what} rejected (state already "
                f"flushed; no further mutations accepted)"
            )

    @contextlib.contextmanager
    def _mutate_locked(self, what: str):
        """UNBOUNDED engine-lock acquire for the mutating session/cache ops
        (correctness over latency — same policy as before), with the
        shutdown gate checked IMMEDIATELY AFTER the acquire: a straggler
        that wins the lock after the shutdown flush aborts here with
        ``EngineBusyError`` before touching any state.

        Codex round 5, finding 3a — already-ENTERED mutations: an op that
        passed the gate before shutdown began can outlive the executors'
        bounded quiesce (minutes-long compaction prefill), make the final
        flush's bounded lock acquire time out, and publish + mark dirty
        AFTER that flush. Two additions close the persistence hole:

        * every entered op is counted in ``_mutations_in_flight``
          (incremented under the engine lock, after the gate check) so the
          server shutdown can bounded-wait for in-flight mutations to
          drain before flushing;
        * SELF-FLUSH-ON-EXIT: if the gate closed while this op was inside
          its critical section, the exit path — still holding the engine
          lock — synchronously flushes THIS engine's dirty sessions before
          releasing. "Publish after the final flush" thereby becomes
          "publish, then immediately flush yourself": nothing dirty
          survives un-persisted, without unbounded server-side waits. Runs
          on EVERY unwind (success, exception, cooperative prefill
          cancel); gate-REJECTED ops never enter, never self-flush."""
        self._lock.acquire()
        entered = False
        try:
            self._check_shutdown_gate(what)
            # getattr: shell engines built via __new__ in tests.
            self._mutations_in_flight = (
                getattr(self, "_mutations_in_flight", 0) + 1
            )
            entered = True
            yield
        finally:
            if entered:
                try:
                    if getattr(self, "_shutting_down", False):
                        self._self_flush_on_shutdown_exit(what)
                finally:
                    # Codex round 7, finding 2a: decrement AFTER the
                    # self-flush fully completes. The old order decremented
                    # first, so wait_mutations_settled() could report zero
                    # while this mutation still held the engine lock doing
                    # the self-flush's disk IO (a VLM save can block for up
                    # to a minute) — the server-side "mutations settled"
                    # signal then let the final flush race a mid-save
                    # straggler. The inner finally keeps the counter exact
                    # even if the self-flush itself raises (it never
                    # should — it swallows internally).
                    self._mutations_in_flight -= 1
            self._lock.release()

    # Codex round 7, finding 2c: upper bound on self-flush rescan passes.
    # Concurrent re-marks during shutdown are FINITE — the gate rejects new
    # mutations, the final flush no longer re-marks (finding 2b: it drains
    # only under the engine lock, which WE hold here), and only lock-free
    # markers (update_session_messages' touch) can still add ids — so a
    # handful of passes reaches an observed-empty scan (best-effort: a
    # touch landing after the last scan stays unseen — acknowledged-benign,
    # see _self_flush_on_shutdown_exit). The bound exists purely to
    # guarantee termination against a misbehaving marker; failed saves are
    # excluded from later passes (re-marked once, never retried here), so a
    # permanent save failure cannot livelock the exit path either.
    _SELF_FLUSH_MAX_PASSES = 8

    def _self_flush_on_shutdown_exit(self, what: str):
        """Finding 3a straggler backstop (caller HOLDS the engine lock):
        the shutdown gate closed while a mutation was inside its critical
        section, so the final flush either already ran (bounded lock
        acquire timed out against us — with finding 2b it drained NOTHING,
        so the ids are still marked) or is blocked waiting on our lock.
        Drain + save the dirty set NOW, before releasing the lock: state
        published by this mutation is persisted deterministically rather
        than relying on an atexit backstop racing an external kill
        deadline. Failures re-mark and never propagate (they must not mask
        the mutation's own outcome).

        Codex round 7, finding 2c: RESCAN until a pass observes an empty
        dirty set (bounded — see _SELF_FLUSH_MAX_PASSES). A single drain
        could race a lock-free dirty marker: an id published into the set
        AFTER the drain but BEFORE this method returned was stranded dirty
        forever (this is the LAST flush that can ever see it — the final
        flush has already run or will skip on our lock). BEST-EFFORT, not
        a guarantee the set is empty at return (codex round 9 nit): a
        lock-free ``update_session_messages`` recency touch can re-mark an
        id right AFTER the final scan, and nothing re-checks past it.
        Acknowledged-benign: substantive generation state is already
        marked dirty at session INSTALLATION and thus drained by an
        earlier pass — a post-scan touch can strand only recency
        metadata."""
        try:
            failed: set[str] = set()
            for _pass in range(self._SELF_FLUSH_MAX_PASSES):
                with self._dirty_lock:
                    to_save = set(self._dirty_sessions) - failed
                    self._dirty_sessions -= to_save
                if not to_save:
                    return
                logger.warning(
                    f"[Shutdown] {what}: mutation finished after the "
                    f"shutdown gate closed — self-flushing {len(to_save)} "
                    f"dirty session(s) before releasing the engine lock "
                    f"(pass {_pass + 1})"
                )
                for sid in to_save:
                    session = self._sessions.get(sid)
                    if session is None:
                        continue
                    try:
                        self._save_session_to_disk(sid, session)
                    except Exception as e:  # noqa: BLE001 — keep saving the rest
                        logger.error(
                            f"[Shutdown] self-flush save failed for {sid}: {e}"
                        )
                        failed.add(sid)
                        with self._dirty_lock:
                            self._dirty_sessions.add(sid)
            with self._dirty_lock:
                leftover = len(self._dirty_sessions - failed)
            if leftover:
                logger.error(
                    f"[Shutdown] self-flush rescan bound "
                    f"({self._SELF_FLUSH_MAX_PASSES} passes) reached with "
                    f"{leftover} session(s) still dirty — a marker kept "
                    f"re-dirtying during shutdown"
                )
        except Exception:  # noqa: BLE001 — never mask the mutation's outcome
            logger.exception("[Shutdown] self-flush-on-exit failed")

    def wait_mutations_settled(self, timeout_s: float = 5.0) -> int:
        """Bounded wait for in-flight ``_mutate_locked`` sections to drain
        (codex round 5, finding 3a — server shutdown step between
        ``begin_shutdown`` and the final flush). Returns the number still
        in flight at the deadline (0 = settled); any remainder is covered
        by the stragglers' self-flush-on-exit."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            remaining = int(getattr(self, "_mutations_in_flight", 0))
            if remaining <= 0:
                return 0
            if time.monotonic() >= deadline:
                return remaining
            time.sleep(0.05)

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
        stop_sequences: list[str] | None = None,
    ) -> Generator[GenerationResult, None, None]:
        """Core generation logic using mlx-vlm (must hold lock).

        Session messages include thinking in assistant content so that
        PromptCacheState prefix matching covers generated tokens from
        previous turns.
        """
        self._touch_gpu()

        has_tools = bool(tools)
        use_thinking = thinking if thinking is not None else self.cfg.enable_thinking
        # U9: structured output (json_schema/json_object) suppresses thinking
        # for THIS request. The FSM processor masks logits to the JSON grammar
        # from the first generated token, so pairing it with the chatml/glm
        # <think> prompt opener produced an unclosed thought block whose
        # "reasoning" was the entire JSON answer (content empty; a thinking-
        # budget-forced </think> would corrupt the constrained JSON). With
        # thinking off, prompt/template/FSM/router all agree: the output IS
        # the content. Applied before the prompt-contract fingerprint below,
        # so the session is stamped with the contract actually rendered.
        if use_thinking and getattr(response_format, "type", None) in (
            "json_schema", "json_object",
        ):
            logger.info(
                f"[Structured] thinking suppressed for this request: "
                f"response_format={getattr(response_format, 'type', None)!r} "
                f"constrains output to JSON from the first token"
            )
            use_thinking = False

        # Defensive fallback only: generate_stream always passes a resolved id
        # (explicit session_id, or an anon-* id from
        # _resolve_anon_session_id_locked). This catches direct/legacy callers.
        # The legacy "anon" slot is registered as engine-minted (provenance)
        # so the anon prefix-scan can keep treating it as a candidate. Caveat:
        # a client explicitly sending user="anon" would share this slot with
        # direct/legacy callers — acceptable on a single-user server, and
        # strictly no worse than the old shared-"anon" behavior.
        if not session_id:
            session_id = "anon"
            if getattr(self, "_anon_minted_ids", None) is not None:
                self._anon_minted_ids.add("anon")
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
        # C1 FIX bookkeeping: on a cache HIT, ``cache_state`` below becomes an
        # ALIAS of session.cache_state, so generation advances the session's
        # LIVE cache in place. These two record what the HIT turn started from
        # so the interrupted-turn commit (client cancel / empty-response early
        # returns) can keep session.messages ↔ cache_state.token_ids in sync,
        # or roll the prefilled suffix back when nothing was generated.
        new_messages: list[dict] = []
        _hit_prior_len = 0

        # --- Cache-reuse contract (U1/U21/U4, one consistent gate) ---------
        # A stored session may be reused only when ALL of:
        #   1. its cache is live;
        #   2. its prompt contract (tools + thinking fingerprint, U21)
        #      matches the request — a mismatch means the cached prefix was
        #      rendered under a different tool schema, so reuse would keep
        #      answering with the STALE schema: honest MISS instead;
        #   3. its stored messages prefix-match the incoming ones (U1 rules
        #      inside _messages_match); and
        #   4. the message suffix past the stored prefix is representable by
        #      the suffix builder (U4): an assistant message in new_messages
        #      is NOT cache-resident (stored messages are the authoritative
        #      record of what the KV contains — e.g. crash recovery where
        #      disk lagged the conversation), and manual assistant splices
        #      are not token-exact vs apply_chat_template (F4) → the shape
        #      is a divergence → honest MISS on ALL templates.
        _tools_canonical = self._canonical_tools(tools)
        _incoming_fp = self._prompt_fingerprint(_tools_canonical, use_thinking)
        _reusable = (
            session is not None
            and session.cache_state is not None
            and session.cache_state.cache is not None
        )
        if _reusable:
            _stored_fp = getattr(session, "prompt_fingerprint", None)
            # F5: a legacy session (fp=None — pre-fingerprint disk file or
            # hand-built state) was built under an UNKNOWN contract: it may
            # carry tools or a different thinking flag in its cached prefix,
            # so ANY reuse (even toolless) would be fail-open. It takes ONE
            # unconditional cold rebuild — an honest MISS whose save stamps
            # the request's fingerprint — never a lenient HIT.
            _contract_ok = _stored_fp is not None and _stored_fp == _incoming_fp
            if not _contract_ok:
                logger.info(
                    f"[KV Cache] session={session_id} | prompt contract "
                    f"{'unknown (legacy, no fingerprint)' if _stored_fp is None else 'changed'} "
                    f"({_stored_fp or 'legacy'} -> {_incoming_fp}) — treating "
                    f"as divergence (honest MISS rebuild stamps the contract)"
                )
                _reusable = False
        if _reusable:
            # Round 3, finding 4: the fingerprint gate above already
            # verified stored thinking == request thinking; use_thinking is
            # therefore the contract BOTH sides' turns ran under.
            _reusable = self._messages_match(
                session.messages, messages, thinking_active=use_thinking,
            )
        if _reusable:
            new_messages = messages[len(session.messages):]
            _blocked_asst = self._suffix_blocking_assistants(new_messages)
            if _blocked_asst:
                logger.info(
                    f"[KV Cache] session={session_id} | suffix contains "
                    f"{_blocked_asst} non-cache-resident assistant turn(s) "
                    f"this template cannot splice — treating as divergence "
                    f"(honest MISS)"
                )
                _reusable = False
                new_messages = []

        if _reusable:
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
                _hit_prior_len = len(cache_state.token_ids or [])
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
                if base.mtp_resume_hidden is not None:
                    # MTP-finalized base (head entries trail by the lazy
                    # last slot): hand the boundary hidden + its offset tag
                    # to the gate so the seeded clone passes
                    # validate_mtp_cache_reuse instead of cold-filling.
                    # mx.arrays are immutable — sharing one hidden across
                    # clones is safe (the stash consume only drops the
                    # cache_state reference, never the pool's).
                    cache_state.mtp_last_hidden = base.mtp_resume_hidden
                    cache_state.mtp_hidden_offset = base.token_count
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
        # U20: defensive clamp — a non-positive repetition penalty divides
        # positive logits by zero (inf/NaN cascade in the penalty processor).
        # The API boundary 400s these; any internal caller that slips one
        # through gets a no-op penalty instead of corrupted logits.
        # Round 2 (codex F6): NaN passes every comparison (``nan <= 0`` is
        # False) — treat any non-finite value as invalid too.
        if repetition_penalty is not None and (
            not math.isfinite(repetition_penalty) or repetition_penalty <= 0
        ):
            logger.warning(
                f"[Generate] repetition_penalty={repetition_penalty} is "
                f"invalid (must be a finite number > 0) — clamping to 1.0 "
                f"(disabled)"
            )
            repetition_penalty = 1.0
        logits_processors = []
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyProcessor(penalty=repetition_penalty))
        budget = thinking_budget if thinking_budget is not None else self.cfg.thinking_budget
        if use_thinking and budget > 0 and self.cfg.think_end_token >= 0:
            think_start = _detect_token_id(
                self.tokenizer,
                "<|channel>" if self.model_family == "gemma4" else "<think>",
            )
            # BARE-OPENER FIX (gemma4 only): past the 1024 sliding window the
            # ``<|channel>`` prime falls out and the model opens thinking with a
            # bare ``thought\n`` (no ``<|channel>`` token). Tokenize that opener
            # once here so the budget recognises it at generation start (mirrors
            # ThinkingRouter's bare-opener handling). Empty/non-gemma4 => off.
            bare_open_tokens = (
                _detect_token_ids(self.tokenizer, "thought\n")
                if self.model_family == "gemma4"
                else []
            )
            logits_processors.append(ThinkingBudgetProcessor(
                budget=budget,
                think_end_token=self.cfg.think_end_token,
                think_start_token=think_start,
                model_family=self.model_family,
                bare_open_tokens=bare_open_tokens,
                # U22: enables the bounded UTF-8 boundary deferral — never
                # force the close mid multi-byte character (U+FFFD would land
                # at the end of reasoning_content).
                tokenizer=self.tokenizer,
            ))
        # Structured output (response_format) via FSM-based logits masking.
        # Works on both mlx-vlm and mlx-lm paths (same logits_processors contract).
        # PLD/speculative decoding is incompatible with the FSM's single-step
        # advance assumption — but that is resolved PER REQUEST at the final
        # use_pld decision (_run_lm_legacy / _run_vlm's drafter disable):
        # structured output wins and speculation is disabled for the request.
        # The FSM processor is therefore ALWAYS built here; disabling it based
        # on cfg.pld_enabled alone used to silently produce UNCONSTRAINED
        # output whenever PLD would not actually run (e.g. wrapped-cache
        # gemma4 sessions where use_pld ends up False).
        structured_proc = None
        rf_type = getattr(response_format, "type", None) if response_format else None
        if rf_type in ("json_schema", "json_object"):
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
        # Use the local EAGER sampler instead of mlx_lm.make_sampler: the
        # latter's compiled categorical/filters bind mx.random.state and freeze
        # the PRNG on non-main worker threads, causing repetition loops. See
        # _make_eager_sampler above for the full root-cause writeup.
        sampler = _make_eager_sampler(
            temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
        )

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
            response_format=response_format,
            cancel_event=cancel_event,
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
        global _MTP_LOGITS_PROCESSORS, _MTP_TOKEN_HISTORY_SEED
        global _MTP_THINK_BUDGET, _MTP_THINK_END_TOKEN, _MTP_THINK_START_TOKEN
        global _MTP_THINK_FAMILY, _MTP_THINK_BARE_OPEN_TOKENS
        global _MTP_THINK_TOKENIZER
        # ------------------------------------------------------------------
        # A3 (GeneratorExit safety): the post-stream reconcile + the HIT-turn
        # commit must NOT live only on the straight-line path after the loop.
        # A client disconnect can close THIS generator at the streaming yield
        # — GeneratorExit then propagates from that yield and everything
        # after the loop is SKIPPED, resurrecting the exact C1 desync
        # (session.messages stale while the HIT session's LIVE cache was
        # already advanced in place). Both production drivers close the
        # engine generator directly, racing the in-loop cancel check:
        #   * in-process mode: generate_stream_async / _batches_async `_run`
        #     threads break out of their for-loop on cancel_event and DROP
        #     the generator — CPython refcounting closes it at the suspended
        #     yield (GeneratorExit) before the engine loop ever observes the
        #     event;
        #   * process mode: the worker's _run_generate does the same
        #     cancel_event break before pulling the next frame (the parent's
        #     _drain_engine_generator name refers to draining the PROXY; the
        #     child-side break is what closes the engine generator).
        # The reconcile below is idempotent and yield-free so the
        # GeneratorExit handler can drive it; the rescue commits the
        # interrupted turn exactly like a cancel (the inner finally has
        # already settled the backend stream) and ANY failure inside the
        # rescue invalidates the session cache fail-closed.
        _turn_committed = False    # a consistent terminal was reached
        _stream_reconciled = False  # post-stream reconcile ran (idempotence)
        # U6: True when the runner signalled fail-closed cache corruption
        # (PLD rewind failure / MTPCacheCorruption). The QwenMTP runner now
        # TERMINATES its stream at the corruption point, so this drives the
        # terminal frame's finish_reason ("error") — the client sees an
        # explicit abnormal finish instead of a silent truncation, retries,
        # and the next turn takes an honest MISS (the corruption callback
        # already invalidated the session cache).
        _cache_corrupted = False
        # U7: terminal-reason bookkeeping. ``_runner_finish`` captures the
        # backend runner's own terminal signal when it provides one (mlx-lm's
        # stream_generate and the PLD/MTP response adapter tag their final
        # frame "stop"/"length"). The mlx-vlm path yields no finish_reason —
        # the post-loop falls back to a max_tokens/EOS heuristic using
        # ``_eos_ids``.
        _runner_finish: Optional[str] = None
        _eos_ids: set[int] = set()
        try:
            _eos_ids = _collect_eos_ids(self.tokenizer)
        except Exception:  # noqa: BLE001 — vocab probing must never break generation
            pass
        # U24: stop-sequence scan state. ``_stop_pending`` holds back the
        # longest suffix of emitted-so-far text that could still begin a stop
        # sequence (a stop can split across token boundaries); text is only
        # released downstream once it provably cannot be part of a match.
        # Scanning happens on the RAW generated text (thinking included) —
        # matching vLLM/LM-Studio semantics for stop strings on reasoning
        # models.
        #
        # U24 round 2 (codex finding 1) — commit-or-invalidate on a stop hit,
        # mirroring the C1 design. The round-1 behavior recorded the withheld
        # stop tokens in generated_token_ids/the KV while session.messages
        # stored the TRUNCATED text: the next HIT spliced its suffix onto the
        # untruncated ids, so the model context silently kept the stop text
        # the client never saw (contract violation — stored messages must
        # describe the cached ids). Now, on a stop hit:
        #   (a) the visible text ends EXACTLY on a token boundary (the match
        #       starts at the first character of a token's decoded segment):
        #       generated_token_ids is trimmed back to that boundary and the
        #       post-stream reconcile trims the cache the same way (its
        #       existing all-layers-trimmable check + per-layer verification
        #       — the C1 machinery). messages ↔ ids ↔ cache all agree.
        #   (b) the match starts mid-token (the stop shares a token with
        #       visible text) OR any cache layer is untrimmable (the
        #       production Qwen hybrid's ArraysCache layers, MTP head-bearing
        #       layouts): the truncated text is still committed to
        #       session.messages but the session cache is INVALIDATED
        #       fail-closed — the next turn takes an honest MISS/cold-fill.
        # TRADEOFF (documented on the invalidation site below): on the hybrid
        # production model a stop hit costs the session cache. ``stop`` is
        # off by default and rare; correctness wins over the lost reuse.
        #
        # ``_stop_tok_bounds[i]`` = cumulative scanned-text length after the
        # frame that recorded generated_token_ids[i] (text-only frames extend
        # the LAST entry — their bytes belong to already-recorded tokens, so
        # a match inside them can never be boundary-aligned). ``_stop_match_abs``
        # is the absolute scanned-text position where the accepted stop match
        # starts; ``_stop_keep_tokens`` is the boundary-aligned kept-token
        # count (None = case (b): not aligned).
        _stop_hit = False
        _stop_pending = ""
        _stop_flush_text = ""
        _stop_total_len = 0
        _stop_tok_bounds: list[int] = []
        # NIT (codex round 3): ``_stop_tok_bounds`` used to grow one int per
        # generated token for the whole stream (~1.2MB at 32k tokens) while
        # boundary resolution only ever needs the RECENT window a future
        # match could still start in: a match starts at or after the pending
        # buffer's absolute start (``_stop_total_len - len(_stop_pending)``,
        # monotonically non-decreasing), padded by the longest stop for
        # slack. Old entries are pruned in batches; ``_stop_bounds_dropped``
        # counts the dropped leading entries so resolved indices stay
        # ABSOLUTE token counts.
        _stop_bounds_dropped = 0
        _stop_max_len = max((len(s) for s in stop_sequences or ()), default=0)
        _STOP_BOUNDS_PRUNE_AT = 4096
        _stop_match_abs = -1
        _stop_keep_tokens: Optional[int] = None
        # Codex round 7, finding 2 (U24 cancellation): set when a
        # cancellation/teardown reconciled a NON-EMPTY _stop_pending buffer —
        # bytes withheld from the wire whose token ids were already recorded
        # (and forwarded through the KV). The reconcile epilogue then applies
        # the same commit-or-invalidate contract as a stop hit: survive only
        # on a positively verified boundary trim, else invalidate.
        _cancel_pending_hidden = False

        def _resolve_stop_boundary():
            """Finding 1, case (a): map the accepted stop-match start onto a
            generated-token boundary and trim generated_token_ids to it.

            The visible text is boundary-aligned iff SOME kept-token count n
            has cumulative decoded length exactly _stop_match_abs; pick the
            SMALLEST such n (a trailing token whose decoded segment is empty
            holds bytes of the stop text — trim it too, fail-closed).
            Trimming the recorded ids HERE — at the hit site, before any
            reconcile (normal post-loop OR the GeneratorExit rescue) — makes
            the reconcile see the cache AHEAD of the recorded ids and run its
            existing C1 trim-back: all-layers-trimmable check, per-layer
            offset verification, and fail-closed invalidation on any hybrid /
            MTP-head-bearing / partially-trimmed cache. ``None`` (not
            boundary-aligned) is case (b): the reconcile invalidates instead.
            """
            nonlocal _stop_keep_tokens
            keep: Optional[int] = None
            if _stop_match_abs == 0:
                keep = 0
            else:
                # NIT (round 3): the bounds list is pruned from the front;
                # _stop_bounds_dropped restores the ABSOLUTE token index. A
                # match can never start inside the pruned region (entries are
                # only dropped once they fall behind the pending buffer's
                # start minus the longest-stop slack).
                _i = bisect.bisect_left(_stop_tok_bounds, _stop_match_abs)
                if (
                    _i < len(_stop_tok_bounds)
                    and _stop_tok_bounds[_i] == _stop_match_abs
                ):
                    keep = _stop_bounds_dropped + _i + 1
            if keep is not None and keep > len(generated_token_ids):
                # Defensive: a boundary beyond the recorded ids cannot be
                # trimmed to — treat as unaligned (case b, invalidation).
                keep = None
            _stop_keep_tokens = keep
            if keep is not None:
                del generated_token_ids[keep:]

        def _reconcile_stream_end():
            """Post-stream reconcile: join text, drafter finalize, write back
            / trim / invalidate the cache so cache offsets == token_ids.
            Idempotent (guarded by _stream_reconciled) and yield-free — safe
            to run from the normal post-loop path AND from the GeneratorExit
            rescue handler."""
            nonlocal accumulated_text, _stream_reconciled, _cache_corrupted
            nonlocal _stop_pending, _stop_match_abs, _cancel_pending_hidden
            if _stream_reconciled:
                return
            _stream_reconciled = True
            # PERF: single join at end of loop — replaces O(N^2) accumulation.
            accumulated_text = "".join(text_parts)
            # Codex round 7, finding 2 (U24 cancellation): a cancellation or
            # stream teardown can reach this reconcile with _stop_pending
            # still NON-EMPTY (the normal post-loop flush is guarded by
            # ``not cancelled``, and the GeneratorExit rescue never runs it).
            # Those bytes were withheld from the wire, but their token ids
            # are ALREADY in generated_token_ids and forwarded through the
            # KV — committing accumulated_text without them would leave a
            # live cache holding hidden context ('answer ST' in the KV vs
            # 'answer ' in messages) that the next HIT silently resumes
            # from. Fail-closed, mirroring the stop-hit-site trim: when the
            # pending text starts EXACTLY at a generated-token boundary,
            # trim the recorded ids back to that boundary HERE — the
            # cache-ahead trim-back below then rewinds the cache and
            # verifies every flattened leaf (the C1 machinery); a mid-token
            # pending start (or any trim/verify shortfall) is caught by the
            # epilogue at the end of this reconcile, which INVALIDATES the
            # session cache. The committed message stays byte-what-the-
            # client-received either way (pending text is never appended).
            if _stop_pending:
                _cancel_pending_hidden = True
                _stop_match_abs = _stop_total_len - len(_stop_pending)
                _resolve_stop_boundary()
                _stop_pending = ""
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
            # FIX-2 (PLD fail-closed rewind): if pld_generate_step signalled a
            # trim failure mid-stream, the local prompt_cache contains ghost
            # tokens (RoPE/offset desync) — INVALIDATE the session cache instead
            # of writing it back; the next turn cold-fills from scratch.
            _pld_cache_invalid = bool(getattr(self, "_pld_cache_invalid", False))
            self._pld_cache_invalid = False
            if _pld_cache_invalid:
                _cache_corrupted = True
                logger.error(
                    f"[KV Cache] session={session_id} | PLD cache rewind failed "
                    f"mid-stream — INVALIDATING session cache (next turn "
                    f"cold-fills)"
                )
                cache_state.cache = None
                cache_state.token_ids = None
            elif prompt_cache is not None:
                cache_state.cache = prompt_cache
            # Reconcile token_ids to the cache's TRUE offset. Speculative paths
            # can drift from cache.offset (PLD never forwards the final token;
            # the gemma4 vlm MTP terminating block forwards past the recorded
            # tail; cancellation drops in-flight tokens). The QwenMTP path now
            # finalizes to EXACT equality — its finalize forwards the pending
            # stop token through the target (mirroring plain mlx-lm's lookahead)
            # and gen_iter.close() above runs that finalize deterministically —
            # so on that path this is a no-op except after cancellation. If
            # token_ids != cache logical length, next turn's reuse cold-fills
            # (offset != prefix_len → full re-prefill, multi-second TTFT). Trim
            # any un-forwarded tail so token_ids EXACTLY matches the cache
            # content; the dropped 1–2 tokens are reprocessed as part of the
            # next turn's suffix (cheap).
            # (Skipped entirely when the PLD corruption signal invalidated the
            # cache above — there is nothing consistent to reconcile against.)
            _actual_token_ids = full_prompt_token_ids + list(generated_token_ids)
            _cache_off = (
                self._get_cache_offset(cache_state.cache) if cache_state.cache else None
            )
            if not _pld_cache_invalid:
                cache_state.token_ids = _actual_token_ids
            if _cache_off is not None and _cache_off > 0:
                if _cache_off < len(_actual_token_ids):
                    # Cache BEHIND recorded (a yielded token not yet forwarded —
                    # bonus lag / final token). Drop the un-forwarded tail so
                    # token_ids matches the cache content exactly.
                    cache_state.token_ids = _actual_token_ids[:_cache_off]
                elif _cache_off > len(_actual_token_ids) and cache_state.cache:
                    # Cache AHEAD of recorded (speculative tail forwarded past
                    # the last RECORDED token — e.g. the terminating block
                    # forwards b but its only output is the stop token, which
                    # stream_generate drops before the engine records it; or a
                    # cancelled QwenMTP stream whose finalize forwarded tokens
                    # the cancel-break dropped). Trim the cache back so
                    # cache.offset == len(token_ids); the few trimmed positions
                    # are reprocessed in next turn's suffix. Without this,
                    # offset>prefix_len → wrapped reuse COLD-FILLs.
                    # FAIL-CLOSED guard: trimming requires EVERY layer to be
                    # trimmable. A hybrid cache (qwen3.5: ArraysCache recurrent
                    # layers have no .trim) would desync — KV layers rewound,
                    # recurrent state still containing the trimmed tokens (ghost
                    # tokens on the next forward) — so INVALIDATE instead.
                    _over = _cache_off - len(_actual_token_ids)
                    # Codex round 3, finding 1: trimmability + per-layer
                    # verification run on the FLATTENED layers so a
                    # CacheList container (GLM MoE/DSA) is judged by its
                    # leaf KVCaches, not by the offset-less wrapper. The
                    # trim() calls themselves stay on the top-level entries
                    # (CacheList.trim delegates to every child).
                    # Codex round 5, finding 2: trimmability is the SEMANTIC
                    # gate (_leaf_trimmable), not mere trim()-presence — a
                    # wrapped RotatingKVCache decrements offsets on trim (so
                    # the post-trim verification below would pass) while the
                    # evicted ring entries are unrecoverable; it must take
                    # the invalidate path instead.
                    _flat_layers = self._flatten_cache_layers(cache_state.cache)
                    if all(self._leaf_trimmable(_c) for _c in _flat_layers):
                        # FAIL-CLOSED trim-back (mirrors the 8491f1d prefix-trim
                        # post-condition): a trim() exception OR any offset-
                        # bearing layer NOT landing exactly on len(token_ids)
                        # leaves a PARTIALLY-trimmed cache (some layers rewound,
                        # others not — e.g. a wrapped RotatingKVCache whose trim
                        # silently no-ops, or a mid-list layer that raised) while
                        # token_ids above was already reconciled to the shorter
                        # history. That desynced cache must never survive into
                        # next turn's reuse — verify every layer and INVALIDATE
                        # on any shortfall. (The qwen MTP head entry trails the
                        # target by one BY DESIGN after finalize, so a head-
                        # bearing cache that lands here also invalidates — the
                        # post-trim stash offset tag is stale anyway, so MTP
                        # resume was already off; a cold-fill is the only safe
                        # outcome.)
                        from mlx_soloheaven.engine.pld import _layer_offsets
                        _trim_exc = False
                        for _c in cache_state.cache:
                            try:
                                _c.trim(_over)
                            except Exception:  # noqa: BLE001
                                _trim_exc = True
                                logger.exception(
                                    "[KV Cache] offset>len cache trim failed"
                                )
                        _bad_layers = [
                            (i, off)
                            for i, off in enumerate(_layer_offsets(_flat_layers))
                            if off is not None and off != len(_actual_token_ids)
                        ]
                        if _trim_exc or _bad_layers:
                            logger.warning(
                                f"[KV Cache] session={session_id} | post-stream "
                                f"trim-back to {len(_actual_token_ids)} failed "
                                f"(exception={_trim_exc}, layers off target: "
                                f"{_bad_layers[:8]}) — INVALIDATING session "
                                f"cache (next turn cold-fills)"
                            )
                            cache_state.cache = None
                            cache_state.token_ids = None
                            # The MTP finalize-hidden stash is offset-tagged
                            # against the now-discarded cache — clear it so a
                            # later turn can never pair a stale hidden with a
                            # rebuilt cache.
                            cache_state.mtp_last_hidden = None
                            cache_state.mtp_hidden_offset = None
                    else:
                        logger.warning(
                            f"[KV Cache] session={session_id} | cache ahead of "
                            f"recorded ids by {_over} but the cache has "
                            f"untrimmable layers (recurrent state, or a "
                            f"wrapped RotatingKVCache whose is_trimmable() "
                            f"is False) — INVALIDATING session cache (next "
                            f"turn cold-fills)"
                        )
                        cache_state.cache = None
                        cache_state.token_ids = None
                        # Same stash hygiene as the verified-trim branch above.
                        cache_state.mtp_last_hidden = None
                        cache_state.mtp_hidden_offset = None

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

            # U24 round 2 (finding 1, case b): the stop match starts
            # MID-TOKEN — no token-boundary trim can remove the stop text
            # from the KV, so the advanced cache must not stay alive:
            # session.messages commits the truncated text (normal save /
            # interrupted-turn Path 0) while the cache is invalidated
            # fail-closed → the next turn takes an honest MISS/cold-fill.
            # This sits at the END of the reconcile so it runs AFTER the
            # runner close has settled (MTP finalize done — the settle
            # contract) on BOTH drivers (normal post-loop and the
            # GeneratorExit rescue), and _invalidate_cache_state clears the
            # MTP finalize-hidden stash so no on_finalize resume state
            # survives for the invalidated session.
            # TRADEOFF: on the production Qwen hybrid (30 untrimmable
            # ArraysCache layers) even a boundary-aligned stop hit already
            # invalidates via the trim-back's trimmability check above — a
            # stop hit costs the session cache there. ``stop`` is off by
            # default and rare; a silent " STOP" resurrected into the next
            # HIT's model context is worse than one cold-fill.
            # Codex round 3, finding 1b — FAIL-CLOSED backstop: a stop hit
            # must never leave an UNVERIFIABLE cache alive. The mid-token
            # case (_stop_keep_tokens is None) always invalidates as before;
            # additionally, a boundary-aligned hit only keeps the cache when
            # its alignment is POSITIVELY established.
            # Codex round 5, finding 2: alignment is verified PER FLATTENED
            # LEAF, not on the first readable offset — with leaves
            # [N, N+1] and N == len(trimmed ids), the scalar read reported
            # N (ahead/trim branch skipped, backstop accepted) while the
            # second leaf stayed AHEAD with the withheld stop tokens
            # inside a 'reusable' cache; the same hole covered an
            # offset-opaque leaf sitting next to one aligned visible leaf.
            # Survival now requires EVERY leaf's offset to be readable AND
            # to land exactly on the trimmed recorded ids; ANY opaque or
            # divergent leaf invalidates.
            # Codex round 7, finding 2: the SAME backstop covers a
            # cancellation/teardown that reconciled a withheld _stop_pending
            # buffer (see the pending resolution at the top of this
            # reconcile) — the hidden bytes' ids must be verifiably trimmed
            # out of the cache, or the cache must not survive.
            if _stop_hit or _cancel_pending_hidden:
                from mlx_soloheaven.engine.pld import _layer_offsets
                _stop_leaf_offs = (
                    _layer_offsets(
                        self._flatten_cache_layers(cache_state.cache)
                    )
                    if cache_state.cache is not None else []
                )
                _recorded_len = len(cache_state.token_ids or [])
                _stop_cache_verified = (
                    _stop_keep_tokens is not None
                    and cache_state.cache is not None
                    and cache_state.token_ids is not None
                    and _recorded_len > 0
                    and bool(_stop_leaf_offs)
                    and all(
                        off is not None and off == _recorded_len
                        for off in _stop_leaf_offs
                    )
                )
                if not _stop_cache_verified and cache_state.cache is not None:
                    _stop_why = (
                        "stop sequence hit"
                        if _stop_hit
                        else "cancellation with withheld stop-scan text"
                    )
                    logger.warning(
                        f"[KV Cache] session={session_id} | {_stop_why} "
                        f"but cache alignment could not be positively "
                        f"verified on every leaf "
                        f"(keep_tokens={_stop_keep_tokens}, "
                        f"match at scan pos {_stop_match_abs}, "
                        f"leaf offsets={_stop_leaf_offs[:8]}, recorded="
                        f"{_recorded_len}) — INVALIDATING "
                        f"session cache (messages keep the truncated text; "
                        f"next turn cold-fills)"
                    )
                    self._invalidate_cache_state(cache_state)

        def _rescue_uncommitted_turn(reason: str, *, commit: bool) -> None:
            """Finalization guard for a stream that never reached a
            consistent terminal (GeneratorExit at the streaming yield, or an
            exception escaping the loop). Idempotent via _turn_committed —
            the normal path commits exactly once and sets the flag before
            its terminal yields, so a GeneratorExit AT a terminal yield is a
            no-op here. Only HIT turns need rescue: non-HIT modes use a
            fresh/cloned cache_state that is simply discarded, leaving the
            stored session untouched and self-consistent.

            ``commit=True`` (GeneratorExit — a clean client disconnect, the
            same shape as a cancel): close the backend stream (idempotent;
            the enclosing finally re-runs it) so the QwenMTP finalize has
            settled BEFORE offsets are read, then reconcile and commit the
            interrupted turn via _commit_interrupted_hit_turn.
            ``commit=False`` (arbitrary exception — generation state is
            mid-frame and unverifiable): invalidate fail-closed instead of
            committing. Any failure inside the rescue itself also
            invalidates fail-closed."""
            if _turn_committed:
                return
            if cache_mode != "hit" or session is None:
                return
            try:
                if commit:
                    try:
                        _close = getattr(gen_iter, "close", None)
                        if _close is not None:
                            _close()
                    except Exception:  # noqa: BLE001 — teardown must never mask the rescue
                        logger.exception(
                            "[Generate] rescue gen_iter.close() failed"
                        )
                    _reconcile_stream_end()
                    self._commit_interrupted_hit_turn(
                        session_id=session_id,
                        session=session,
                        cache_state=cache_state,
                        new_messages=new_messages,
                        accumulated_text=accumulated_text,
                        use_thinking=use_thinking,
                        hit_prior_len=_hit_prior_len,
                        prompt_len=len(full_prompt_token_ids),
                        reason=reason,
                        tools_canonical=_tools_canonical,
                        prompt_fingerprint=_incoming_fp,
                    )
                else:
                    logger.warning(
                        f"[KV Cache] session={session_id} | {reason} before "
                        f"the turn was committed — INVALIDATING session "
                        f"cache fail-closed (next turn cold-fills)"
                    )
                    self._invalidate_cache_state(cache_state)
            except BaseException:  # noqa: BLE001 — rescue must never mask the exit
                logger.exception(
                    f"[KV Cache] session={session_id} | stream-teardown "
                    f"rescue failed — INVALIDATING session cache"
                )
                self._invalidate_cache_state(cache_state)

        try:
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

                text = resp.text if hasattr(resp, "text") else ""
                tok_attr = getattr(resp, "token", None)
                token = tok_attr if tok_attr is not None else 0
                prompt_tps = getattr(resp, "prompt_tps", 0.0) or 0.0
                gen_tps = getattr(resp, "generation_tps", 0.0) or 0.0
                # U7: capture the runner's terminal signal (mlx-lm final
                # frame / adapter EOS + exhaustion frames). Never forwarded
                # on the per-token GenerationResults below — only the
                # engine's own terminal frame carries a finish_reason.
                _fr = getattr(resp, "finish_reason", None)
                if _fr in ("stop", "length"):
                    _runner_finish = _fr

                # U8: count/record GENERATED TOKENS only. A text-only frame
                # (token=None — the adapter's detokenizer tail flush, or a
                # synthetic terminal) carries text whose token ids were
                # already recorded on their own frames.
                if tok_attr is not None:
                    gen_token_count += 1
                    generated_token_ids.append(int(tok_attr))
                elif not text:
                    # Signal-only frame (e.g. the adapter's exhaustion frame
                    # with no buffered tail): the finish signal was captured
                    # above; nothing to convey outward.
                    continue

                # U24: stop-sequence scan with holdback. The pending buffer
                # is bounded by max(len(stop)) once released text is emitted;
                # on a match the emitted text ends BEFORE the stop sequence
                # and the loop terminates after this frame.
                if stop_sequences:
                    _stop_total_len += len(text)
                    if tok_attr is not None:
                        _stop_tok_bounds.append(_stop_total_len)
                    elif _stop_tok_bounds:
                        # Text-only frame (detokenizer tail flush): its bytes
                        # come from already-recorded tokens — extend the last
                        # token's segment so a match inside it can never be
                        # mistaken for boundary-aligned (fail-closed).
                        _stop_tok_bounds[-1] = _stop_total_len
                    # NIT (round 3): batched front-prune of bounds that no
                    # future match can land on. Threshold = start of the
                    # pending buffer minus the longest-stop slack (matches
                    # start at/after the pending start; the slack is pure
                    # belt-and-braces). The NEWEST entry always survives so
                    # a text-only frame keeps extending the last segment.
                    if len(_stop_tok_bounds) >= _STOP_BOUNDS_PRUNE_AT:
                        _thresh = (
                            _stop_total_len - len(_stop_pending)
                            - len(text) - _stop_max_len
                        )
                        _cut = bisect.bisect_left(_stop_tok_bounds, _thresh)
                        _cut = min(_cut, len(_stop_tok_bounds) - 1)
                        if _cut > 0:
                            del _stop_tok_bounds[:_cut]
                            _stop_bounds_dropped += _cut
                if stop_sequences and text:
                    _stop_pending += text
                    _hit_idx = None
                    for _s in stop_sequences:
                        _j = _stop_pending.find(_s)
                        if _j != -1 and (_hit_idx is None or _j < _hit_idx):
                            _hit_idx = _j
                    # U24 round 2 (codex finding 2): deterministic
                    # earliest-START semantics independent of chunking. A
                    # completed match at position P is only accepted when NO
                    # other stop has a viable partial match starting BEFORE P
                    # that extends to the end of the buffer (repro: stops
                    # ["b","aba"] — frame ["aba"] vs frames ["ab","a"] must
                    # both yield ""). While such a partial is alive, HOLD:
                    # it either completes (its earlier start wins on a later
                    # rescan of the same buffer) or dies (the rescan then
                    # accepts the held completed match at P). At stream end
                    # the post-loop flush resolves a still-held buffer the
                    # same way (partials can never complete there).
                    # _partial_marker_tail returns the LONGEST buffer suffix
                    # that is a proper prefix of any stop == the EARLIEST
                    # viable partial start.
                    _keep = _partial_marker_tail(
                        _stop_pending, tuple(stop_sequences)
                    )
                    if _hit_idx is not None and (
                        len(_stop_pending) - _keep >= _hit_idx
                    ):
                        text = _stop_pending[:_hit_idx]
                        _stop_match_abs = (
                            _stop_total_len - len(_stop_pending) + _hit_idx
                        )
                        _stop_pending = ""
                        _stop_hit = True
                        # Finding 1: resolve the token boundary + trim the
                        # recorded ids AT THE HIT SITE, so even a client
                        # disconnect at this frame's yield (GeneratorExit
                        # rescue) reconciles against the truncated ids.
                        _resolve_stop_boundary()
                    else:
                        text = _stop_pending[: len(_stop_pending) - _keep]
                        _stop_pending = _stop_pending[len(_stop_pending) - _keep:]

                # PERF: append-to-list + post-loop join avoids the O(N^2)
                # cost of repeated ``str += text`` (each += allocates a new
                # string and copies the entire accumulated buffer).
                text_parts.append(text)

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

                if _stop_hit:
                    # U24: a stop sequence completed in this frame's text.
                    # Break out — the finally below closes gen_iter (the GPU
                    # actually stops) and the post-loop reconcile applies the
                    # commit-or-invalidate contract: generated_token_ids was
                    # already trimmed to the visible-text token boundary at
                    # the hit site (case a — the reconcile trims the cache to
                    # match, all-layers-trimmable + verified), or the match
                    # is mid-token/untrimmable (case b — the reconcile
                    # invalidates the session cache fail-closed). Either way
                    # the stored messages never claim less than the KV holds.
                    _logger.info(
                        f"[Generate] session={session_id} | STOP sequence hit "
                        f"at token {gen_token_count} — terminating generation"
                    )
                    break

        except GeneratorExit:
            # A3: the response generator was closed at the streaming yield
            # (client disconnect / consumer break-and-drop — see the rescue
            # closure's driver notes). Commit the interrupted HIT turn like
            # a cancel, then let the exit propagate (never swallow it).
            _rescue_uncommitted_turn(
                "disconnected (stream torn down)", commit=True,
            )
            raise
        except GenerationCancelled as _prefill_cancel:
            # U13: cancel observed BETWEEN PREFILL CHUNKS (before any token
            # reached the client). Route into the NORMAL cancel path below:
            # the reconcile trims token_ids back to the cache's true offset
            # and the HIT commit applies the C1 fail-closed rules (verified
            # roll-back of the partially prefilled suffix, or invalidation);
            # a MISS's partially-filled FRESH cache is simply discarded
            # (never persisted). On the QwenMTP path the runner's own
            # corruption callback already invalidated the session cache
            # (target+head advanced mid-prefill cannot be verifiably
            # rewound) — the cancel commit's Path 0 then leaves the stale
            # messages harmless.
            logger.info(
                f"[Generate] session={session_id} | CANCELLED during prefill "
                f"({_prefill_cancel}) — aborting before first token"
            )
            cancelled = True
        except BaseException:
            # Fail-closed consistency guard: an exception escaping the loop
            # leaves the HIT session's in-place-advanced cache unreconciled
            # and mid-frame — invalidate rather than commit unverifiable
            # state, then re-raise.
            _rescue_uncommitted_turn("generation failed mid-stream", commit=False)
            raise
        finally:
            # Deterministically close the backend stream BEFORE the post-loop
            # reads cache offsets. The QwenMTP runner settles + finalizes its
            # caches in the generator's own ``finally``; on the cancel/EOS
            # break paths nothing else closes gen_iter until GC, so without
            # this the reconcile below could read MID-ROUND offsets and the
            # finalize would mutate the cache AFTER token_ids were already
            # reconciled (stale bookkeeping -> spurious cold-fills).
            try:
                _close = getattr(gen_iter, "close", None)
                if _close is not None:
                    _close()
            except Exception:  # noqa: BLE001 — teardown must never break the stream
                logger.exception("[Generate] gen_iter.close() failed")
            # CORRECTION 4: clear the per-request MTP stash once the stream
            # is fully driven — on normal completion, the cancel/EOS break
            # ABOVE, AND any exception / GeneratorExit propagating through a
            # yield inside the loop. Otherwise the drafter path leaves the
            # logits-processor instances + the FULL prompt token history
            # stashed on module globals across requests (stale-state /
            # memory / privacy risk) until the next drafter request
            # overwrites them. Single-worker assumption holds (one in-flight
            # VLM request at a time).
            _MTP_LOGITS_PROCESSORS = None
            _MTP_TOKEN_HISTORY_SEED = None
            # RUNAWAY-THINKING FIX: clear the thinking-budget stash in the SAME
            # finally so it never leaks across requests (matches the MTP
            # processor/seed clear above).
            _MTP_THINK_BUDGET = None
            _MTP_THINK_END_TOKEN = None
            _MTP_THINK_START_TOKEN = None
            _MTP_THINK_FAMILY = None
            _MTP_THINK_BARE_OPEN_TOKENS = None
            _MTP_THINK_TOKENIZER = None
        # U24: resolve the held-back scan buffer at stream end (exhaustion /
        # EOS reached with text still held). Finding 2: a COMPLETED match may
        # be sitting in the buffer, held only because an earlier viable
        # partial could still have won — at stream end partials can never
        # complete, so accept the EARLIEST completed match (identical output
        # for every chunk segmentation of the same text). Otherwise the held
        # tail is real output — fold it into the accumulated text BEFORE the
        # reconcile (session persistence) and remember it for a text-only
        # flush frame after the save (its token ids are already recorded).
        if _stop_pending and not cancelled and not _stop_hit:
            _hit_idx = None
            for _s in stop_sequences or ():
                _j = _stop_pending.find(_s)
                if _j != -1 and (_hit_idx is None or _j < _hit_idx):
                    _hit_idx = _j
            if _hit_idx is not None:
                _stop_hit = True
                _stop_match_abs = (
                    _stop_total_len - len(_stop_pending) + _hit_idx
                )
                _resolve_stop_boundary()
                _vis = _stop_pending[:_hit_idx]
                if _vis:
                    text_parts.append(_vis)
                    _stop_flush_text = _vis
            else:
                text_parts.append(_stop_pending)
                _stop_flush_text = _stop_pending
            _stop_pending = ""

        # Post-stream reconcile (idempotent, yield-free — shared with the
        # GeneratorExit rescue defined above the loop). Includes the stop-hit
        # commit-or-invalidate epilogue (finding 1).
        _reconcile_stream_end()

        # U7: max_tokens exhaustion detection. Prefer the runner's explicit
        # terminal signal; the mlx-vlm path reports none, so fall back to a
        # heuristic: the loop consumed max_tokens frames and the last
        # generated token is not an EOS (an EOS landing exactly on the limit
        # is still a natural stop).
        _length_hit = not cancelled and not _stop_hit and (
            _runner_finish == "length"
            or (
                _runner_finish is None
                and max_tokens > 0
                and gen_token_count >= max_tokens
                and bool(generated_token_ids)
                and int(generated_token_ids[-1]) not in _eos_ids
            )
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
            # C1 FIX (a): on a cache HIT, ``cache_state`` ALIASES the
            # session's live cache_state and the reconcile above has already
            # advanced its token_ids to prompt+generated — returning without
            # touching session.messages would desync them: the next
            # same-history request would match the STALE messages, take HIT,
            # and splice user_N a SECOND time after the dangling partial
            # output (silently corrupted model input, persistent in the
            # session KV). Commit or roll back so messages ↔ token_ids/KV
            # stay consistent. Non-HIT modes keep the old skip: their
            # cache_state is a fresh/cloned object that is simply discarded
            # and the session (if any) is untouched and still
            # self-consistent.
            if cache_mode == "hit" and session is not None:
                self._commit_interrupted_hit_turn(
                    session_id=session_id,
                    session=session,
                    cache_state=cache_state,
                    new_messages=new_messages,
                    accumulated_text=accumulated_text,
                    use_thinking=use_thinking,
                    hit_prior_len=_hit_prior_len,
                    prompt_len=len(full_prompt_token_ids),
                    reason="cancelled",
                    tools_canonical=_tools_canonical,
                    prompt_fingerprint=_incoming_fp,
                )
            # Idempotence marker: the cancel terminal is consistent (commit /
            # rollback / invalidate all reconcile messages ↔ token_ids).
            _turn_committed = True
            return

        # Guard: detect empty response (no content after thinking)
        if accumulated_text and session_id:
            # Round 3, finding 4: judge emptiness on the router-authoritative
            # content channel — with thinking DISABLED the whole text is
            # content (a literal </think> quote no longer empties it); with
            # thinking active the stream began inside the thought block.
            # Codex round 7, finding 3: thinking_active gates the gemma4
            # bare-opener recognition (a thinking=False turn genuinely
            # starting with 'thought\n' is content, not empty).
            _, content = split_thinking_and_content(
                accumulated_text, model_family=self.model_family,
                started_in_thinking=(
                    use_thinking and self.model_family != "gemma4"
                ),
                thinking_active=use_thinking,
            )
            if not use_thinking and self.model_family != "gemma4":
                content = accumulated_text
            if not content or not content.strip():
                if cache_mode == "hit" and session is not None:
                    # C1 FIX (b): the session's live cache already contains
                    # suffix(user_N) + the thinking-only output (in-place
                    # alias advance) — "SKIP SAVE" would leave
                    # session.messages claiming the turn never happened and
                    # the next request would splice user_N twice. Commit the
                    # turn instead. A2: do NOT assume the stream ended
                    # naturally — a thinking-only exhaustion at max_tokens
                    # has no stop token in the KV. The commit's close
                    # reconcile verifies the recorded tail against the
                    # end-of-turn ids: verified → commit as-is; otherwise it
                    # closes the turn (or invalidates fail-closed). The
                    # guard's original no-junk-persistence intent is
                    # preserved for non-HIT modes below, where the fresh
                    # cache_state is discarded and nothing desyncs.
                    logger.warning(
                        f"[KV Cache] session={session_id} | EMPTY RESPONSE "
                        f"on HIT ({gen_token_count} tokens, no content) — "
                        f"committing thinking-only turn for messages ↔ "
                        f"token_ids consistency"
                    )
                    self._commit_interrupted_hit_turn(
                        session_id=session_id,
                        session=session,
                        cache_state=cache_state,
                        new_messages=new_messages,
                        accumulated_text=accumulated_text,
                        use_thinking=use_thinking,
                        hit_prior_len=_hit_prior_len,
                        prompt_len=len(full_prompt_token_ids),
                        reason="empty response",
                        tools_canonical=_tools_canonical,
                        prompt_fingerprint=_incoming_fp,
                    )
                else:
                    logger.warning(
                        f"[KV Cache] session={session_id} | SKIP SAVE | "
                        f"empty response ({gen_token_count} tokens, no content)"
                    )
                # Idempotence marker BEFORE the terminal yield: a
                # GeneratorExit delivered at that yield must not re-commit.
                _turn_committed = True
                # U24: the held partial-match tail was already folded into
                # the persisted text — deliver it on the wire too.
                if _stop_flush_text:
                    yield GenerationResult(
                        text=_stop_flush_text,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=gen_token_count,
                    )
                yield GenerationResult(
                    text="",
                    # U6: a corruption-terminated stream is an abnormal end —
                    # surface it instead of claiming a clean stop. U7: a
                    # thinking-only max_tokens exhaustion is "length".
                    finish_reason=(
                        "error" if _cache_corrupted
                        else ("length" if _length_hit else "stop")
                    ),
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=gen_token_count,
                )
                return

        # Parse tool_calls once — used both for session persistence and
        # for the terminal GenerationResult's finish_reason.
        # U6/F1: NEVER on a corruption-terminated stream — the text is
        # truncated at an arbitrary point and a partial <tool_call> block
        # must not be surfaced (or persisted) as a real, executable call.
        # U12: parse the CONTENT CHANNEL only. A model "rehearsing" a tool
        # call inside its thinking region must not have the rehearsal
        # recorded/emitted as a real tool_calls entry — only text outside
        # the thinking region is tool-call territory (the streaming path
        # already enforces this via the ThinkingRouter, whose tool FSM only
        # consumes content segments).
        parsed_tool_calls: list[dict] = []
        if has_tools and accumulated_text and not _cache_corrupted:
            # Codex round 3, finding 4 — the content channel here must be
            # what the STREAMING router would have emitted (the authority):
            # - gemma4: the router-policy union of all content segments
            #   (orphan-close with no prior thought-open is CONTENT — the
            #   old extract-based split discarded everything before it,
            #   hiding a call the streaming FSM had already parsed);
            # - chatml/glm, thinking ACTIVE: the stream began inside the
            #   thought block — split with started_in_thinking=True (the
            #   degenerate no-</think> turn is ALL reasoning, rehearsals
            #   never parsed; unchanged from U12/FIX 1);
            # - chatml/glm, thinking DISABLED: the inactive router passes
            #   EVERYTHING through — the whole text is the content channel
            #   and a literal </think> in it is a quote, never a boundary
            #   that hides a call before it.
            if self.model_family == "gemma4":
                # Codex round 7, finding 3: thread the turn's thinking
                # contract — bare-opener recognition inside the segmentation
                # follows the router (full markers stay authoritative).
                _content_channel = _content_channel_union(
                    accumulated_text, "gemma4", use_thinking,
                )
            elif use_thinking:
                _, _content_channel = split_thinking_and_content(
                    accumulated_text,
                    model_family=self.model_family,
                    started_in_thinking=True,
                )
            else:
                _content_channel = accumulated_text
            _, parsed_tool_calls = parse_tool_calls(
                _content_channel, model_family=self.model_family,
            )

        # Save session
        # U6/F1: skip persistence entirely on a corruption-terminated stream
        # — the truncated text must not enter session.messages (the cache is
        # already invalidated, so the stale stored messages are harmless:
        # the next request's HIT condition requires a live cache → honest
        # MISS cold-fill).
        if session_id and not _cache_corrupted:
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
                # U12: strip from CONTENT segments only — a rehearsed marker
                # inside the (kept) thinking segments must never truncate the
                # stored thinking. Round 2: channel-aware across ALL content
                # segments — gemma4 multi-cycle output (thought → content
                # with a call → thought → content) keeps the earlier cycle's
                # call strippable (the round-1 last-close-marker reduction
                # missed it, storing BOTH the raw XML and the structured
                # tool_calls); parse-based per-block removal also stops
                # truncating unrelated post-call text.
                # Round 3, finding 4: thread the turn's thinking contract so
                # the strip's channel segmentation matches the batch parse
                # above (thinking disabled → the whole text is content and a
                # literal </think> never hides a strippable block).
                full_assistant_content = _strip_content_channel_tool_xml(
                    full_assistant_content, self.model_family,
                    thinking_active=use_thinking,
                )

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
                # U3/U21: stamp the prompt contract the cache was built /
                # extended under so every later HIT and rebuild path can
                # verify + re-render it.
                tools=_tools_canonical,
                thinking=use_thinking,
                prompt_fingerprint=_incoming_fp,
                # U26: carry the cumulative drafter stats — this install used
                # to reset them every turn (the finalize above wrote to the
                # OLD state object this line replaces).
                drafter_stats=self._drafter_stats_for(session_id),
            )
            # Codex round 5, finding 1a: mark dirty HERE, atomically with the
            # install (this thread holds the engine lock). Persistence used to
            # depend on the API layer's post-stream update_session_messages
            # call — if that call was dropped (saturated long pool, client
            # disconnect tearing the SSE generator down before its post-loop),
            # the freshly extended cache was never marked and could never be
            # flushed. The API call remains (session touch + the terminal-log
            # contract) but is bookkeeping only now.
            self._mark_dirty(session_id)

            logger.debug(
                f"[KV Cache] session={session_id} | SAVED | "
                f"offset: {prev_offset} -> {new_offset} tokens "
                f"(+{new_offset - prev_offset})"
            )

        # Idempotence marker: the normal save is done — a GeneratorExit at
        # the final yield below finds a consistent session.
        _turn_committed = True

        # Auto-register base cache on miss. F4: cancel_event threads into the
        # secondary system-prompt prefill so a disconnect aborts it too.
        if cache_mode in ("miss", "retry") and messages:
            self._maybe_register_base_cache(
                messages, prompt_token_ids, tools=tools, thinking=use_thinking,
                cancel_event=cancel_event,
            )

        # U24: deliver the held partial-match tail on the wire (it was
        # already folded into the persisted text before the reconcile). The
        # turn is committed, so a GeneratorExit at this yield is safe.
        if _stop_flush_text:
            yield GenerationResult(
                text=_stop_flush_text,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=gen_token_count,
            )

        # Determine finish reason (parsed_tool_calls computed above).
        # U6: a corruption-terminated stream (QwenMTP fail-closed / PLD
        # rewind failure) ends with finish_reason="error" — the session
        # cache is already invalidated, so the client's retry takes an
        # honest MISS; a partial <tool_call> in the truncated text must not
        # be surfaced as a real tool call either.
        # U7/U24 precedence: error > tool_calls > stop-sequence stop >
        # length > natural stop. A stop-sequence termination reports the
        # OpenAI "stop" reason; max_tokens exhaustion reports "length".
        if _cache_corrupted:
            finish_reason = "error"
        elif parsed_tool_calls:
            finish_reason = "tool_calls"
        elif _stop_hit:
            finish_reason = "stop"
        elif _length_hit:
            finish_reason = "length"
        else:
            finish_reason = "stop"

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
        response_format=None,
        cancel_event=None,
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

        U13: ``cancel_event`` makes the mlx-lm path's chunked prefill
        cancel-aware (progress-callback hook / qwen_mtp / PLD prefill
        loops). mlx-vlm's generate_step exposes NO prompt-progress hook, so
        the vlm path's prefill cancel stays token-granular (first check at
        the first yielded token).
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
                    response_format=response_format,
                ),
                None,
            )
        return self._run_lm_legacy(
            cache_state=cache_state,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            response_format=response_format,
            cancel_event=cancel_event,
        )

    def _sampling_vocab_size(self) -> "int | None":
        """Vocabulary size used to normalize ``top_k`` for the mlx-vlm
        backend (U20 round 2). Derived from the tokenizer (``len`` includes
        added tokens); the model's logits width is padded UP from this, so a
        ``top_k`` at or beyond the tokenizer vocab is keep-all regardless of
        padding. Returns None when no usable surface exists (clamp then
        passes the value through unchanged)."""
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return None
        hf = getattr(tok, "_tokenizer", tok)
        for probe in (hf, tok):
            try:
                n = len(probe)
                if n and n > 0:
                    return int(n)
            except Exception:  # noqa: BLE001 — try the next surface
                pass
            vs = getattr(probe, "vocab_size", None)
            if isinstance(vs, int) and vs > 0:
                return vs
        return None

    def _clamp_vlm_top_k(self, top_k) -> int:
        """U20 round 2 (codex F1): map an oversized ``top_k`` to upstream's
        keep-all sentinel (0 = filter disabled) BEFORE it reaches mlx-vlm.
        mlx_lm.sample_utils.apply_top_k raises for top_k >= vocab_size, and
        the raise would land mid-generation; top_k >= vocab keeps every
        token, i.e. it is semantically identical to the disabled filter."""
        if not top_k or top_k <= 0:
            return 0
        vocab = self._sampling_vocab_size()
        if vocab is not None and top_k >= vocab:
            logger.info(
                f"[Generate] top_k={top_k} >= vocab ({vocab}) — keep-all; "
                f"normalized to 0 (filter disabled) for the mlx-vlm sampler"
            )
            return 0
        return int(top_k)

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
        response_format=None,
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

        # U20 round 2 (codex F1): the API deliberately accepts an oversized
        # top_k (excess == keep-all, capped engine-side), but only the mlx-lm
        # eager sampler clamped it. mlx-vlm builds its sampler via
        # mlx_lm.sample_utils.make_sampler, whose apply_top_k RAISES for
        # top_k >= vocab_size MID-STREAM. Normalize here — the single point
        # where the vlm path's sampling kwargs are assembled — mapping the
        # keep-all excess to upstream's keep-all sentinel (top_k=0 disables
        # the filter entirely, exactly the keep-all semantics).
        top_k = self._clamp_vlm_top_k(top_k)

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
        # CORRECTION 3: structured output (response_format) requires the
        # FSM/structured-output logits processor to advance EXACTLY one token
        # per step. Speculative MTP decoding samples a BLOCK of positions at
        # once and the clone only applies stateless rep-penalty to the
        # speculative block (stateful FSM processors run only in the post-wrap
        # _plain_step), so pre-wrap speculative tokens would bypass the
        # constraint and a later _plain_step would see stale FSM state → the
        # structured schema could be violated. Simplest safe fix: DISABLE the
        # drafter for the whole request so generation uses the plain path where
        # the FSM processor (already in logits_processors) is applied correctly
        # on every token.
        if response_format is not None and drafter is not None:
            logger.info(
                f"[Drafter] session={session_id} disabled: response_format="
                f"{getattr(response_format, 'type', response_format)!r} "
                f"(structured output requires single-step FSM advance; "
                f"speculative decoding is incompatible)"
            )
            drafter = None
        # Layer-A safety net (SUPERSEDED by the B4 RoPE-frame fix): for a
        # request whose cache is ALREADY wrapped at the start (e.g. turn 3+
        # of a long chat, offset > sliding_window), this used to bypass the
        # drafter entirely — which on multi-turn chats dropped throughput to
        # plain-decode speed for every later turn. B4 makes the drafter
        # wrap-safe, so we keep it ON by default and only honour this
        # bypass when the explicit fallback (SOLOHEAVEN_MTP_WRAP_GATE=1) is
        # set. The LOCAL ``drafter`` is mutated; ``self._drafter`` stays
        # loaded for the next request.
        wrap_imminent = self._will_wrap_during_generate(
            prompt_token_ids, cache_state
        )
        if drafter is not None and wrap_imminent and _MTP_WRAP_GATE:
            logger.warning(
                f"[Drafter] session={session_id} skip: RotatingKVCache wrap "
                f"imminent (sliding_window={self._sliding_window_size}) — "
                f"drafter bypassed (SOLOHEAVEN_MTP_WRAP_GATE fallback)"
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
        global _HOT_PATH_FAST, _MTP_LOGITS_PROCESSORS, _MTP_TOKEN_HISTORY_SEED
        global _MTP_THINK_BUDGET, _MTP_THINK_END_TOKEN, _MTP_THINK_START_TOKEN
        global _MTP_THINK_FAMILY, _MTP_THINK_BARE_OPEN_TOKENS
        global _MTP_THINK_TOKENIZER
        _HOT_PATH_FAST = not wrap_possible
        # FIX 2: stash the per-request logits_processors + prompt-token-history
        # seed so the MTP clone (_patched_mtp_rounds_v2) can apply them — the
        # clone is the active decode path and upstream calls it with a BARE
        # sampler. Only meaningful on the drafter (MTP) path; cleared after the
        # stream is built on the non-drafter path. Single-worker safe (same
        # contract as _HOT_PATH_FAST). The seed is the FULL prompt so the
        # rep-penalty context matches upstream generate_step (which accumulates
        # the prompt tokens into its running ``tokens`` array during prefill).
        if drafter is not None:
            _MTP_LOGITS_PROCESSORS = (
                list(logits_processors) if logits_processors else None
            )
            # NOTE (codex risk #3 — INTENTIONAL): seed the rep-penalty history
            # with the FULL logical prompt (incl. cache-hit prior turns), NOT
            # upstream's suffix-trimmed tail. This is deliberate: it lets the
            # repetition penalty see prior-turn tokens and suppress cross-turn
            # repetition. Do NOT change to suffix-only.
            _MTP_TOKEN_HISTORY_SEED = list(prompt_token_ids)
            # RUNAWAY-THINKING FIX: thread the thinking-budget cap into the MTP
            # clone so it is enforced during NORMAL speculative decoding (the
            # default path), not only in the gated _plain_step. The budget +
            # token ids come from the ThinkingBudgetProcessor that
            # generate_stream already built and added to logits_processors under
            # the exact gate (use_thinking + budget>0 + think_end_token>=0) — we
            # read them off that instance so this stays a no-op whenever thinking
            # is off / budget<=0 (no processor present → stash stays None). The
            # stateful processor itself is filtered OUT of the clone's
            # per-position lists; the history-derived helper replaces it there.
            _tbp = next(
                (
                    p
                    for p in (logits_processors or [])
                    if isinstance(p, ThinkingBudgetProcessor)
                ),
                None,
            )
            if _tbp is not None and _tbp.budget > 0 and _tbp.think_end_token >= 0:
                _MTP_THINK_BUDGET = int(_tbp.budget)
                _MTP_THINK_END_TOKEN = int(_tbp.think_end_token)
                _MTP_THINK_START_TOKEN = int(_tbp.think_start_token)
                _MTP_THINK_FAMILY = _tbp.model_family
                # BARE-OPENER FIX: thread the gemma4 bare ``thought\n`` opener
                # token sequence so the MTP clone recognises it at gen-start too.
                _MTP_THINK_BARE_OPEN_TOKENS = (
                    list(_tbp.bare_open_tokens) or None
                )
                # U22 round 2 (codex F4a): thread the processor's tokenizer so
                # the clone's force site applies the same bounded UTF-8
                # boundary deferral (None keeps the immediate force).
                _MTP_THINK_TOKENIZER = _tbp._tokenizer
            else:
                _MTP_THINK_BUDGET = None
                _MTP_THINK_END_TOKEN = None
                _MTP_THINK_START_TOKEN = None
                _MTP_THINK_FAMILY = None
                _MTP_THINK_BARE_OPEN_TOKENS = None
                _MTP_THINK_TOKENIZER = None
            gen_kwargs["draft_model"] = drafter
            gen_kwargs["draft_kind"] = getattr(self, "_draft_kind", None)
            if self.cfg.draft_block_size:
                gen_kwargs["draft_block_size"] = self.cfg.draft_block_size
            # Reset acceptance bookkeeping per request so the post-stream
            # log line below reports this request's stats only.
            if hasattr(drafter, "accept_lens"):
                drafter.accept_lens = []
        else:
            _MTP_LOGITS_PROCESSORS = None
            _MTP_TOKEN_HISTORY_SEED = None
            _MTP_THINK_BUDGET = None
            _MTP_THINK_END_TOKEN = None
            _MTP_THINK_START_TOKEN = None
            _MTP_THINK_FAMILY = None
            _MTP_THINK_BARE_OPEN_TOKENS = None
            _MTP_THINK_TOKENIZER = None

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
                # U26: accumulate into the session-keyed registry — this
                # finalize runs BEFORE the post-generation SessionState
                # install, which used to replace the state object (and with
                # it the stats written here) every turn.
                self._accumulate_drafter_stats(_sid, n_rounds, total_accepted)
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
        response_format=None,
        cancel_event=None,
    ):
        """mlx-lm legacy path. Manages a local `prompt_cache` because
        mlx-lm mutates the cache list in place during prefix-trim +
        stream_generate; the caller writes it back to `cache_state.cache`
        after the stream loop completes.

        Returns `(gen_iter, prompt_cache)`.

        U13: ``cancel_event`` is threaded into every prefill loop this path
        can dispatch to (plain mlx-lm via ``prompt_progress_callback``,
        qwen_mtp, PLD) so a disconnect during a huge prefill aborts between
        chunks (``GenerationCancelled`` — the stream loop converts it into
        the normal cancel reconcile).
        """
        # Drafter dispatch by kind: qwen3_5_mtp heads ("qwen_mtp") run
        # NATIVELY on this mlx-lm path (engine/qwen_mtp.py); every other
        # drafter kind (gemma4_assistant MTP / DFlash) is mlx-vlm only —
        # fail loud if reached here.
        _drafter = getattr(self, "_drafter", None)
        use_mtp = _drafter is not None and (
            getattr(self, "_draft_kind", None) == "qwen_mtp"
        )
        # qwen-mtp-capable engines are the ONLY source of trailing head
        # cache entries / the finalize-hidden stash; remember capability
        # before per-request disables so the plain-dispatch hygiene
        # invariant below knows whether enforcement applies at all.
        _qwen_mtp_capable = use_mtp
        if _drafter is not None and not use_mtp:
            raise RuntimeError(
                "MTP drafter (--draft-model) requires --backend mlx-vlm for "
                f"this drafter kind ({getattr(self, '_draft_kind', None)!r}); "
                "only qwen3_5_mtp heads run natively on the mlx-lm backend. "
                "Alternatively use --pld for speculative decoding."
            )
        # FIX-6 mirror: structured output (response_format json_schema /
        # json_object) requires the FSM processor to advance EXACTLY one
        # token per emitted position — disable MTP for THIS request (the FSM
        # processor stays active; same rule as the PLD/vlm-drafter gates).
        _rf_type = (
            getattr(response_format, "type", None) if response_format else None
        )
        if use_mtp and _rf_type in ("json_schema", "json_object"):
            logger.info(
                f"[Structured] QwenMTP disabled for this request: "
                f"response_format={_rf_type!r} (single-step FSM advance)"
            )
            use_mtp = False
        # KV-quantized caches interact badly with the MTP trim/rollback
        # boundary (and bf16 KV measured optimal for this family anyway).
        if use_mtp and self.cfg.kv_bits:
            if not getattr(self, "_mtp_kv_bits_warned", False):
                self._mtp_kv_bits_warned = True
                logger.warning(
                    f"[{self.model_id}] QwenMTP disabled: --kv-bits="
                    f"{self.cfg.kv_bits} is incompatible with MTP rollback "
                    f"(keep the bf16 KV cache on this path)."
                )
            use_mtp = False
        if use_mtp and self.cfg.pld_enabled:
            logger.info(
                f"[{self.model_id}] both --draft-model (qwen_mtp) and --pld "
                f"set — MTP takes precedence; PLD skipped."
            )
        # FIX-2: per-request PLD cache-corruption flag. pld_generate_step's
        # fail-closed _rewind fires the callback when a trim silently no-ops;
        # the callback INVALIDATES the session cache on the spot (robust to
        # abandoned generators) and this flag tells the generate_stream
        # post-loop to skip the ghost-token cache write-back. Single
        # in-flight request per engine (same assumption as
        # _pending_drafter_finalize).
        self._pld_cache_invalid = False

        prompt_cache = cache_state.cache
        # FIX-3 (CRITICAL-2): same wrapped-RotatingKVCache reuse gate as the
        # vlm path (_run_vlm). Once a sliding-window ring buffer has wrapped,
        # its physical buffer no longer corresponds to a contiguous logical
        # PREFIX — prefix-trim reuse on a divergent prompt (branch/edit past
        # the wrap) would mis-align rotated ring slots with prefix positions.
        # Append-only reuse (no divergence) stays allowed, including
        # post-wrap append-only (verified correct).
        if prompt_cache is not None and not self._safe_to_reuse_cache(
            cache_state, prompt_token_ids
        ):
            logger.warning(
                f"[KV Cache] COLD-FILL (RotatingKVCache wrapped + prompt "
                f"divergence) — dropping cache and re-prefilling full prompt "
                f"({len(prompt_token_ids)} tokens)"
            )
            cache_state.cache = None
            cache_state.token_ids = None
            prompt_cache = None
        # QwenMTP cache-reuse gate (fail-closed, BEFORE the prefix-trim):
        # the MTP runner needs the 40 target entries + the head's KV entries
        # in the FINALIZED shape — every target offset == len(stored ids)
        # (stored ids include the natural stop token; finalize forwards it
        # through the target, matching the plain path's lookahead), head
        # offset == target - 1 (the head's last slot pairs with THIS turn's
        # first suffix token and is committed lazily at resume from the
        # stashed finalize hidden) — and a PURE-APPEND prompt (the 30
        # ArraysCache layers cannot trim, so divergence can never be
        # rewound). Divergence/target-desync misses -> COLD-FILL once;
        # afterwards the session carries the target+head layout and stays
        # MTP-reusable across appends. Head-side-only misses and head-less
        # layouts keep the cache and fall back to PLAIN decode (see
        # plan_mtp_cache_reuse and the branch comments below).
        _mtp_resume_hidden = None
        if use_mtp:
            from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
            _n_target = len(self._language_model.layers)
            _n_head = max(1, len(getattr(_drafter, "layers", [])) or 1)
            if prompt_cache is not None:
                # Consume the finalize-hidden stash up front (single-use): a
                # failed/aborted stream must never resurrect a stale one.
                _mtp_resume_hidden = getattr(cache_state, "mtp_last_hidden", None)
                _mtp_hidden_off = getattr(cache_state, "mtp_hidden_offset", None)
                cache_state.mtp_last_hidden = None
                cache_state.mtp_hidden_offset = None
                if (
                    _mtp_resume_hidden is not None
                    and _mtp_hidden_off != self._get_cache_offset(prompt_cache)
                ):
                    # Stash predates a cache mutation (e.g. an intervening
                    # non-MTP request advanced this session) — fail closed.
                    _mtp_resume_hidden = None
                _action, _why = qwen_mtp_mod.plan_mtp_cache_reuse(
                    prompt_cache,
                    cache_state.token_ids or [],
                    prompt_token_ids,
                    _n_target,
                    _n_head,
                    _mtp_resume_hidden,
                )
                if _action == qwen_mtp_mod.REUSE_FALLBACK_PLAIN:
                    # Head-less target-only layout (plain base cache, a
                    # disk-reloaded session — the loader strips the head —
                    # or a session written by a non-MTP server) whose target
                    # side is pure-append-consistent: keep the cache, decode
                    # plain. POLICY: falling back is ONE-WAY — the session
                    # stays a head-less plain session for its remaining
                    # append-only lifetime (every later turn re-plans
                    # FALLBACK_PLAIN; head entries are only rebuilt by a
                    # cold-fill, which we deliberately never force). Right
                    # trade on this target: Qwen3.6 A3B MoE measured MTP
                    # block1 speedup 1.019x ≈ a wash vs plain decode, so
                    # reusing the history always beats re-speculating it —
                    # NO cold-fill-to-restore-MTP threshold.
                    logger.info(
                        f"[QwenMTP] cache not MTP-reusable ({_why}) — "
                        f"plain-decode fallback (reusing "
                        f"{len(cache_state.token_ids or [])} tokens)"
                    )
                    use_mtp = False
                    _mtp_resume_hidden = None
                elif _action == qwen_mtp_mod.REUSE_FALLBACK_STRIP_HEAD:
                    # Head-side-only failure (stale head offset, missing or
                    # stale resume hidden) with a consistent target: keep
                    # the target entries, decode plain. The actual head
                    # strip happens in the plain-dispatch invariant right
                    # below (every non-MTP route runs it), IN PLACE —
                    # cache_state.cache aliases this list. Same one-way
                    # policy as FALLBACK_PLAIN above.
                    logger.info(
                        f"[QwenMTP] cache not MTP-reusable ({_why}) — "
                        f"head stripped, plain-decode fallback (reusing "
                        f"{len(cache_state.token_ids or [])} tokens)"
                    )
                    use_mtp = False
                    _mtp_resume_hidden = None
                elif _action != qwen_mtp_mod.REUSE_MTP:
                    # Divergence / desynced target offsets / nothing cached:
                    # the untrimmable ArraysCache layers can never rewind —
                    # fail closed (cold-fill, MTP stays on).
                    logger.info(
                        f"[QwenMTP] cache not MTP-reusable ({_why}) — "
                        f"COLD-FILL ({len(prompt_token_ids)} tokens)"
                    )
                    cache_state.cache = None
                    cache_state.token_ids = None
                    prompt_cache = None
                    _mtp_resume_hidden = None
        if _qwen_mtp_capable and not use_mtp:
            # INVARIANT (enforced on EVERY plain dispatch route of a
            # qwen-mtp-capable engine, BEFORE the prefix-trim and
            # generation): the cache handed to plain generation never
            # carries more entries than the target has layers, and the
            # single-use finalize-hidden stash never survives a plain turn
            # it didn't belong to. This covers MTP disables that fire
            # BEFORE the reuse gate (response_format FSM, --kv-bits) — which
            # would otherwise hand a 41-entry MTP-finalized session or a
            # base-seeded 41-entry clone to plain stream_generate intact —
            # as well as the gate's own fallback branches (where it
            # performs STRIP_HEAD's strip). Non-qwen-mtp engines have no
            # source of head entries (the disk loader strips them on load),
            # so enforcement is scoped to where the invariant can break.
            self._strip_mtp_head_for_plain_dispatch(cache_state, prompt_cache)
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
            # FIX-3: trim via the caches' own .trim() (keeps
            # RotatingKVCache._idx bookkeeping consistent and supports
            # QuantizedKVCache, whose .keys is a list — raw keys/values
            # slicing crashed there and silently desynced _idx). The logical
            # cached length comes from the cache offset, not buffer shapes.
            cached_len = self._get_cache_offset(prompt_cache)
            trim_needed = cached_len - prefix_len
            if trim_needed > 0:
                from mlx_soloheaven.engine.pld import _layer_offsets
                trimmed = trim_prompt_cache(prompt_cache, trim_needed)
                # Post-condition (cheap, offset reads only): EVERY
                # offset-bearing layer must land EXACTLY on prefix_len.
                # trim_prompt_cache's return is ONLY cache[0]'s trim count
                # ([c.trim(n) for c in cache][0]); if per-layer offsets had
                # already diverged, layer 0 can trim fully while another
                # layer under-trims (trim clamps to min(offset, n)) — the
                # shortfall would otherwise go undetected.
                _bad_layers = [
                    (i, off) for i, off in enumerate(_layer_offsets(prompt_cache))
                    if off is not None and off != prefix_len
                ]
                if trimmed != trim_needed or _bad_layers:
                    # Fail closed: an untrimmable layer (e.g. a wrapped
                    # RotatingKVCache that slipped past the gate) no-ops the
                    # whole trim, and a diverged layer under-trims — either
                    # way cold-fill rather than reuse a mis-aligned cache.
                    logger.warning(
                        f"[KV Cache] prefix trim failed (requested "
                        f"{trim_needed}, trimmed {trimmed}, layers not at "
                        f"prefix_len={prefix_len}: {_bad_layers[:8]}) — "
                        f"COLD-FILL"
                    )
                    cache_state.cache = None
                    cache_state.token_ids = None
                    prompt_cache = make_prompt_cache(self._language_model)
                    prefix_len = 0
            # Only feed tokens after prefix
            prompt_token_ids = prompt_token_ids[prefix_len:]

        # --- QwenMTP dispatch (native mlx-lm MTP speculative decoding) ---
        if use_mtp:
            # Fresh / cold-filled cache: append the head's KV entries after
            # the target's (PR #990 layout convention). The SAME list object
            # is written back to cache_state.cache post-stream, so the head
            # entries persist for append-only multi-turn reuse.
            if len(prompt_cache) == _n_target:
                prompt_cache.extend(qwen_mtp_mod.make_head_cache(_n_head))

            def _on_mtp_cache_corruption():
                # Same contract as _on_pld_cache_corruption: invalidate the
                # session cache IMMEDIATELY (abandoned generators never reach
                # the post-loop) and flag the post-loop to skip write-back.
                self._pld_cache_invalid = True
                cache_state.cache = None
                cache_state.token_ids = None
                cache_state.mtp_last_hidden = None
                cache_state.mtp_hidden_offset = None

            def _on_mtp_finalize(final_token, final_hidden):
                # Stash the finalize hidden (h at the LAST target position):
                # next turn's reuse gate requires it so the runner can
                # lazily commit the head's last slot pair
                # (final_hidden, first_suffix_token) — works for ANY
                # continuation, including the chat-template "\n" glue after
                # a natural stop token. Tagged with the cache offset so a
                # stash that predates any later cache mutation is detected
                # and dropped (fail-closed -> cold-fill). In-memory only —
                # never persisted: on disk reload the session loader strips
                # the trailing head entries, so a reloaded session reuses
                # its full history via the plain fallback
                # (REUSE_FALLBACK_PLAIN), not via MTP.
                if final_hidden is None:
                    cache_state.mtp_last_hidden = None
                    cache_state.mtp_hidden_offset = None
                else:
                    cache_state.mtp_last_hidden = final_hidden
                    cache_state.mtp_hidden_offset = self._get_cache_offset(
                        prompt_cache
                    )

            # Codex round 7, finding 1: shared close-reason cell between the
            # runner and its adapter. The adapter marks it NATURAL once the
            # EOS frame went out, so the teardown close (engine cancel-break
            # / disconnect rescue arriving AFTER the EOS was delivered) still
            # runs the finalize forwards — cache offset == recorded ids. The
            # engine's cancel-driven close of a MID-stream adapter leaves the
            # cell unmarked; the runner then keeps the round-6 inference
            # (GeneratorExit + event set -> cancel, finalize skipped).
            _mtp_close_reason = qwen_mtp_mod.GeneratorCloseReason()
            gen_iter = _pld_response_adapter(
                qwen_mtp_mod.qwen_mtp_generate_step(
                    prompt=mx.array(prompt_token_ids),
                    model=self._language_model,
                    head=_drafter,
                    block_size=getattr(self, "_mtp_block_size", None) or 3,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=(
                        logits_processors if logits_processors else None
                    ),
                    prompt_cache=prompt_cache,
                    n_target_layers=_n_target,
                    prefill_step_size=self.cfg.prefill_step_size,
                    on_cache_corruption=_on_mtp_cache_corruption,
                    on_finalize=_on_mtp_finalize,
                    resume_hidden=_mtp_resume_hidden,
                    cancel_event=cancel_event,
                    close_reason=_mtp_close_reason,
                ),
                tokenizer=self.tokenizer,
                label="QwenMTP",
                close_reason=_mtp_close_reason,
            )
            return gen_iter, prompt_cache

        lm_kwargs = {
            "max_tokens": max_tokens,
            "sampler": sampler,
            "prompt_cache": prompt_cache,
            "prefill_step_size": self.cfg.prefill_step_size,
            "logits_processors": logits_processors if logits_processors else None,
        }
        if cancel_event is not None:
            # U13: mlx-lm's generate_step invokes this between prefill
            # chunks — raising here aborts the prefill within one chunk of
            # the disconnect instead of finishing the whole prompt.
            def _abort_prefill_on_cancel(processed, total):
                if cancel_event.is_set():
                    raise GenerationCancelled(
                        f"prefill cancelled at {processed}/{total} prompt tokens"
                    )
            lm_kwargs["prompt_progress_callback"] = _abort_prefill_on_cancel
        if self.cfg.kv_bits:
            lm_kwargs["kv_bits"] = self.cfg.kv_bits
            lm_kwargs["kv_group_size"] = self.cfg.kv_group_size
            lm_kwargs["quantized_kv_start"] = self.cfg.quantized_kv_start

        # PLD requires trimmable cache (for rollback on rejection).
        use_pld = self.cfg.pld_enabled
        # FIX-6: structured output (response_format) requires the FSM
        # processor to advance EXACTLY one token per emitted position with
        # no speculative multi-token rounds — disable PLD for THIS request
        # and keep the FSM (mirrors the vlm path's drafter disable). This
        # decision lives here, AFTER cache reuse/cold-fill resolution, so it
        # reflects whether PLD would actually run.
        # Gate on the SAME types that actually build an FSM in generate_stream
        # (json_schema / json_object): {"type": "text"} and unknown types
        # build NO processor and must KEEP PLD.
        _rf_type = (
            getattr(response_format, "type", None) if response_format else None
        )
        if use_pld and _rf_type in ("json_schema", "json_object"):
            logger.info(
                f"[Structured] PLD disabled for this request: response_format="
                f"{_rf_type!r} (structured output requires single-step FSM "
                f"advance; the FSM processor stays active)"
            )
            use_pld = False
        if use_pld:
            from mlx_lm.models.cache import can_trim_prompt_cache
            if not can_trim_prompt_cache(prompt_cache):
                from mlx_soloheaven.engine.pld import _wrapped_rotating_layers
                # FIX-8: name the ACTUAL cause. For gemma4 the untrimmable
                # layer is a wrapped RotatingKVCache (sliding-window ring,
                # offset >= max_size — permanent for this session), NOT
                # ArraysCache/DeltaNet.
                if _wrapped_rotating_layers(prompt_cache):
                    logger.info(
                        f"[{self.model_id}] PLD disabled for this request: "
                        f"RotatingKVCache sliding-window has wrapped "
                        f"(offset >= max_size) — the cache is permanently "
                        f"untrimmable past the wrap. Falling back to "
                        f"standard generation."
                    )
                elif not getattr(self, "_pld_incompat_warned", False):
                    logger.warning(
                        f"[{self.model_id}] PLD disabled: model uses "
                        f"non-trimmable cache (e.g. ArraysCache/DeltaNet). "
                        f"Falling back to standard generation."
                    )
                    self._pld_incompat_warned = True
                use_pld = False

        if use_pld:
            from mlx_soloheaven.engine.pld import pld_generate_step

            def _on_pld_cache_corruption():
                # FIX-2: surfaced by pld_generate_step's fail-closed rewind.
                # Invalidate the session cache IMMEDIATELY (here, not only in
                # generate_stream's post-loop): an abandoned generator (client
                # disconnect that never drives the stream to completion) never
                # reaches the post-loop, and the ghost-token cache must not
                # survive into the next turn. The flag is still consumed by
                # the post-loop — when it DOES run — to skip the prompt_cache
                # write-back / token_ids reconcile and log the event.
                self._pld_cache_invalid = True
                cache_state.cache = None
                cache_state.token_ids = None

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
                    on_cache_corruption=_on_pld_cache_corruption,
                    cancel_event=cancel_event,
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

    @staticmethod
    def _invalidate_cache_state(cache_state) -> None:
        """Fail-closed session-cache invalidation: the next turn's HIT
        condition requires a live cache, so it takes the honest MISS /
        cold-fill path. The MTP finalize-hidden stash is cleared with the
        cache — a stale hidden must never pair with a rebuilt cache."""
        cache_state.cache = None
        cache_state.token_ids = None
        cache_state.mtp_last_hidden = None
        cache_state.mtp_hidden_offset = None

    # End-of-turn text used to template-close a client-cancelled assistant
    # turn on the ChatML family (Qwen etc.). gemma4/glm need no close: the
    # gemma4 next-turn suffix LEADS with the ``<turn|>`` closer (see
    # _suffix_tokens_gemma4) and GLM turns are delimited by the role markers
    # (<|user|>/<|assistant|>) themselves — neither template records a
    # per-turn terminator that the interrupted turn would be missing.
    _CHATML_TURN_END = "<|im_end|>"

    def _try_close_interrupted_turn(
        self, session_id: str, cache_state,
    ) -> TurnCloseResult:
        """Template-close reconcile for an interrupted assistant turn (C1).

        Forwards the end-of-turn token through the TARGET model (1-token
        forward — cheap) and appends it to token_ids, so the cached sequence
        is template-valid for the next turn — mirroring the natural-EOS path,
        where the stop token is recorded into token_ids and forwarded through
        the cache (mlx-lm lookahead / QwenMTP finalize).

        Tri-state per-path policy (see TurnCloseResult):
        - gemma4 / glm: NOT_REQUIRED — see _CHATML_TURN_END comment; the
          commit proceeds without a close and stays template-valid.
        - recorded tail already IS an end-of-turn token (tokenizer EOS ids /
          <|im_end|>): NOT_REQUIRED — the turn terminated naturally and the
          stop token is in the KV (recorded + forwarded by the runner). This
          is the A2 verification: an "empty response" (thinking-only) turn
          may equally be a max_tokens exhaustion, which does NOT end with a
          stop token and must be closed like a cancel.
        - chatml + mlx-lm + target-only cache: CLOSED (forward, then verify
          every offset-bearing layer landed on len(token_ids)).
        - mlx-vlm path: FAILED — F3 pins every mlx-vlm model call to the
          persistent _vlm_executor worker thread; a raw forward from this
          (request) thread would break that pinning, and committing the
          unterminated ChatML turn instead would hand the next HIT a cache
          whose suffix splice (_suffix_tokens_chatml) assumes a prior
          <|im_end|> that is NOT in the KV — template corruption.
        - MTP head-bearing (target+head) cache: FAILED — the head/lazy-slot
          bookkeeping is owned by the runner finalize, which
          gen_iter.close() already ran (never double-commit); a target-only
          forward here would desync head offset == target - 1. (Nearly
          unreachable: the cancel reconcile invalidates head-bearing caches
          whose target offsets cannot equal len(token_ids); a finalized
          natural end passes the tail check above instead.)
        - Detection failures (eot not in vocab, non-callable model): FAILED.
        - Failure DURING the forward, or a post-forward offset mismatch:
          FAILED (layers may be half-advanced).

        EVERY FAILED path invalidates the session cache before returning
        (fail-closed): a close was needed but cannot be performed/verified,
        so the only safe outcome is an honest MISS → cold-fill next turn.
        """
        if self.model_family in ("gemma4", "glm"):
            return TurnCloseResult.NOT_REQUIRED
        cache = cache_state.cache
        if cache is None or not cache_state.token_ids:
            # Defensive: callers gate on a live cache; anything else has
            # nothing verifiable to close — fail closed (invalidation of an
            # already-dead state is a no-op).
            self._invalidate_cache_state(cache_state)
            return TurnCloseResult.FAILED

        # A2 verification: natural termination leaves the stop token as the
        # recorded tail (the terminal frame carries token=eos and the runner
        # forwarded it — mlx-lm lookahead / QwenMTP finalize). If the tail
        # already IS an end-of-turn token, the cached sequence is
        # template-terminated: no close needed.
        eot_ids: set[int] = set()
        try:
            eot_ids |= _collect_eos_ids(self.tokenizer)
        except Exception:  # noqa: BLE001 — vocab probing must never raise
            pass
        try:
            eot = _detect_token_id(self.tokenizer, self._CHATML_TURN_END)
        except Exception:  # noqa: BLE001 — vocab probing must never raise
            eot = -1
        if eot >= 0:
            eot_ids.add(eot)
        if eot_ids and int(cache_state.token_ids[-1]) in eot_ids:
            return TurnCloseResult.NOT_REQUIRED

        # From here a close IS required — any unavailability is FAILED
        # (fail-closed invalidate), never an unterminated ChatML commit.
        def _close_unavailable(why: str) -> TurnCloseResult:
            logger.warning(
                f"[KV Cache] session={session_id} | interrupted turn needs "
                f"an end-of-turn close but {why} — INVALIDATING session "
                f"cache (fail-closed; an unterminated ChatML commit would "
                f"corrupt the next HIT's suffix splice)"
            )
            self._invalidate_cache_state(cache_state)
            return TurnCloseResult.FAILED

        if self._use_vlm:
            return _close_unavailable(
                "the mlx-vlm path cannot forward from this thread (F3 "
                "worker-thread pinning)"
            )
        model = self._language_model
        n_target = len(getattr(model, "layers", None) or [])
        if n_target and len(cache) > n_target:
            return _close_unavailable(
                "the cache carries trailing MTP head entries (a target-only "
                "forward would desync the head offset)"
            )
        if not callable(model):
            return _close_unavailable("the target model is not callable")
        if eot < 0:
            return _close_unavailable(
                f"no {self._CHATML_TURN_END!r} token in the vocab"
            )
        try:
            model(mx.array([[eot]]), cache=cache)
            self._eval_cache(cache)
        except Exception:  # noqa: BLE001 — fail closed: layers may be half-advanced
            logger.exception(
                f"[KV Cache] session={session_id} | end-of-turn close "
                f"forward failed — INVALIDATING session cache (next turn "
                f"cold-fills)"
            )
            self._invalidate_cache_state(cache_state)
            return TurnCloseResult.FAILED
        cache_state.token_ids = list(cache_state.token_ids) + [eot]
        from mlx_soloheaven.engine.pld import _layer_offsets
        expected = len(cache_state.token_ids)
        _bad_layers = [
            (i, off)
            for i, off in enumerate(_layer_offsets(cache))
            if off is not None and off != expected
        ]
        if _bad_layers:
            logger.warning(
                f"[KV Cache] session={session_id} | end-of-turn close left "
                f"layers off target (expected {expected}, layers: "
                f"{_bad_layers[:8]}) — INVALIDATING session cache"
            )
            self._invalidate_cache_state(cache_state)
            return TurnCloseResult.FAILED
        # The interrupted turn is closed; any finalize-hidden stash from it
        # is offset-stale now — clear rather than let the next-turn gate
        # discover it (single-use hygiene, same as the plain-dispatch strip).
        cache_state.mtp_last_hidden = None
        cache_state.mtp_hidden_offset = None
        return TurnCloseResult.CLOSED

    def _commit_interrupted_hit_turn(
        self,
        *,
        session_id: str,
        session: SessionState,
        cache_state,
        new_messages: list[dict],
        accumulated_text: str,
        use_thinking: bool,
        hit_prior_len: int,
        prompt_len: int,
        reason: str,
        tools_canonical: list | None = None,
        prompt_fingerprint: str | None = None,
    ) -> None:
        """C1 FIX: reconcile session bookkeeping after an early-returning HIT
        turn (client cancel / empty thinking-only response).

        On a cache HIT ``cache_state`` IS ``session.cache_state`` (alias) and
        generation advanced the session's live KV + token_ids IN PLACE; only
        the normal save at the end of _generate_locked keeps
        session.messages / total_cache_tokens in step. When an early return
        skips that save, session.messages claims the turn never happened
        while token_ids already contain suffix(user_N) + partial output — the
        next same-history request then matches the STALE messages, takes HIT,
        and splices user_N a SECOND time after the dangling partial output.
        This helper restores messages ↔ token_ids on every early-return
        shape:

        - cache already invalidated (AHEAD-reconcile on an untrimmable /
          hybrid cache, PLD/MTP corruption callback): nothing to commit —
          the stale messages are harmless because the next request's HIT
          condition requires a live cache (honest MISS → cold-fill).
        - no generated token recorded in the cache (cancel before any
          token): committing an (empty) assistant turn would be semantically
          wrong for the client's next request shape — ROLL BACK the
          prefilled suffix instead (verified per-layer trim back to the
          pre-turn history), or INVALIDATE if any layer cannot trim
          (fail-closed). session.messages, unchanged, describes exactly the
          rolled-back state.
        - partial (cancel / stream teardown) or thinking-only (empty
          response) output in the cache: reconcile the turn's template close
          FIRST (_try_close_interrupted_turn's tri-state — NOT_REQUIRED and
          CLOSED commit; FAILED already invalidated fail-closed, so nothing
          is committed), then COMMIT the assistant message the same way the
          normal save path does (full content incl. thinking via
          _make_full_assistant_content, session.messages + new_messages +
          assistant, total_cache_tokens from the cache offset) so the next
          request's existing _messages_match machinery decides reuse
          honestly — its last-assistant leniency accepts client-side
          truncation of the partial output; real divergence takes the
          existing cold-fill path. The close reconcile runs for EVERY
          committed shape, including the "empty response" one: a
          thinking-only stream may have ended at max_tokens exhaustion
          rather than EOS, and only the recorded-tail verification inside
          _try_close_interrupted_turn can tell them apart (A2).

        Tool-call XML in a cancelled partial output is stored RAW (no
        parsing / content stripping, unlike the normal save): the stored
        assistant content is engine-internal and only consulted by
        _messages_match — whose assistant leniency covers the difference —
        and a partial <tool_call> block must never be surfaced as a real
        tool call.
        """
        # Path 0 — already invalidated upstream: token_ids is gone too, so
        # there is nothing to keep consistent (fail-closed already happened).
        if cache_state.cache is None or not cache_state.token_ids:
            logger.info(
                f"[KV Cache] session={session_id} | {reason} on HIT | cache "
                f"already invalidated — messages left as-is (next turn "
                f"MISSes and cold-fills)"
            )
            return

        token_ids = list(cache_state.token_ids)

        # Path C — interrupted before ANY generated token was recorded: the
        # KV holds at most the prefilled suffix (the reconcile already
        # trimmed any un-recorded speculative tail back to the prompt).
        if len(token_ids) <= prompt_len:
            over = len(token_ids) - hit_prior_len
            if over <= 0:
                logger.info(
                    f"[KV Cache] session={session_id} | {reason} on HIT | "
                    f"cache still at pre-turn history ({len(token_ids)} "
                    f"tokens) — nothing to roll back"
                )
                return
            if all(hasattr(_c, "trim") for _c in cache_state.cache):
                from mlx_soloheaven.engine.pld import _layer_offsets
                _trim_exc = False
                for _c in cache_state.cache:
                    try:
                        _c.trim(over)
                    except Exception:  # noqa: BLE001
                        _trim_exc = True
                        logger.exception(
                            f"[KV Cache] session={session_id} | suffix "
                            f"rollback trim failed"
                        )
                _bad_layers = [
                    (i, off)
                    for i, off in enumerate(_layer_offsets(cache_state.cache))
                    if off is not None and off != hit_prior_len
                ]
                if _trim_exc or _bad_layers:
                    logger.warning(
                        f"[KV Cache] session={session_id} | {reason} before "
                        f"any generated token — suffix rollback to "
                        f"{hit_prior_len} failed (exception={_trim_exc}, "
                        f"layers off target: {_bad_layers[:8]}) — "
                        f"INVALIDATING session cache (next turn cold-fills)"
                    )
                    self._invalidate_cache_state(cache_state)
                    return
                cache_state.token_ids = token_ids[:hit_prior_len]
                # The whole turn was rolled back — any finalize stash from
                # it is bogus (a pre-turn stash cannot exist here: the MTP
                # gate consumes it and the plain-dispatch strip clears it).
                cache_state.mtp_last_hidden = None
                cache_state.mtp_hidden_offset = None
                session.total_cache_tokens = hit_prior_len
                session.touch()
                logger.info(
                    f"[KV Cache] session={session_id} | {reason} before any "
                    f"generated token — rolled the {over}-token suffix back "
                    f"(cache at {hit_prior_len} tokens, messages unchanged)"
                )
            else:
                logger.warning(
                    f"[KV Cache] session={session_id} | {reason} before any "
                    f"generated token but the cache has untrimmable "
                    f"(recurrent) layers — INVALIDATING session cache (next "
                    f"turn cold-fills)"
                )
                self._invalidate_cache_state(cache_state)
            return

        # Path A/B — partial (cancel / stream teardown) or thinking-only
        # (empty response) output is in the KV: reconcile the template close
        # (tri-state), then commit. FAILED means a close was REQUIRED but
        # unavailable/unverifiable — _try_close_interrupted_turn already
        # invalidated the cache fail-closed; the stale messages are harmless
        # without a live cache (the next request's HIT condition requires
        # one → honest MISS → cold-fill), exactly like Path 0.
        close = self._try_close_interrupted_turn(session_id, cache_state)
        if close is TurnCloseResult.FAILED or cache_state.cache is None:
            logger.info(
                f"[KV Cache] session={session_id} | {reason} on HIT | "
                f"turn close unavailable — session cache invalidated "
                f"fail-closed, messages left as-is (next turn cold-fills)"
            )
            return
        assistant_msg = {
            "role": "assistant",
            "content": self._make_full_assistant_content(
                accumulated_text, use_thinking,
            ),
            # U1 marker: this turn was committed by the interrupted-turn
            # path, so its stored content may legitimately differ from what
            # the client received (thinking channel + wire truncation).
            # _messages_match applies the NARROW _interrupted_resend_equiv
            # rule to marked messages — instead of the removed
            # last-stored-assistant wildcard. The marker round-trips disk
            # (messages are JSON metadata) and never leaks into prompts:
            # _format_messages copies only role/content/tool_calls/
            # tool_call_id, and suffix builders render INCOMING client
            # messages, never this stored dict.
            "interrupted": True,
        }
        new_offset = (
            self._get_cache_offset(cache_state.cache) if cache_state.cache else 0
        )
        # Fallback mirrors the normal save: some models (GLM MoE) don't
        # expose offset in cache objects.
        if new_offset == 0 and cache_state.token_ids:
            new_offset = len(cache_state.token_ids)
        self._sessions[session_id] = SessionState(
            cache_state=cache_state,
            messages=list(session.messages) + list(new_messages) + [assistant_msg],
            total_cache_tokens=new_offset,
            # F5: stamp the CURRENT request's contract (a HIT turn implies it
            # matched the session's — the U21 gate ran before the HIT), never
            # propagate a legacy None forward: an interrupted commit that
            # carried fp=None would re-open the legacy leniency indefinitely.
            tools=tools_canonical,
            thinking=use_thinking,
            prompt_fingerprint=(
                prompt_fingerprint
                if prompt_fingerprint is not None
                else self._prompt_fingerprint(tools_canonical, use_thinking)
            ),
            # U26: the interrupted commit is a reinstall too — keep the
            # session's cumulative drafter stats.
            drafter_stats=self._drafter_stats_for(session_id),
        )
        # Codex round 5, finding 1a: an interrupted commit is exactly the
        # case where the API layer NEVER reaches its post-stream
        # update_session_messages (the client is gone) — mark dirty at the
        # install or the committed turn is unflushable.
        self._mark_dirty(session_id)
        logger.info(
            f"[KV Cache] session={session_id} | {reason} on HIT | committed "
            f"interrupted turn ({len(new_messages)} new + 1 assistant msg, "
            f"offset {hit_prior_len} -> {new_offset})"
        )

    def _strip_mtp_head_for_plain_dispatch(self, cache_state, prompt_cache):
        """Hygiene invariant for every plain (non-MTP) dispatch route.

        Strips trailing MTP head entries IN PLACE (cache_state.cache aliases
        the list) when the cache carries more entries than the target has
        layers, and drops the single-use finalize-hidden stash. mlx-lm
        0.31.x would silently ignore the extra entries (zip(layers, cache)),
        so output stays correct without this — but the head entries would
        silently go stale (their offsets stop tracking the target's) and
        every downstream offset/trim post-condition assumes a target-only
        layout. Once stripped, the session continues plain for its
        append-only lifetime (see the FALLBACK_PLAIN policy in the MTP
        gate).

        Duck-typed: a model without an introspectable ``.layers`` (hermetic
        test stubs) skips the strip — every real mlx-lm model exposes it,
        and the MTP gate that creates head entries requires it anyway."""
        n_target = len(getattr(self._language_model, "layers", None) or [])
        if n_target and prompt_cache is not None and len(prompt_cache) > n_target:
            n_extra = len(prompt_cache) - n_target
            del prompt_cache[n_target:]
            logger.info(
                f"[QwenMTP] plain dispatch: stripped {n_extra} trailing MTP "
                f"head entries ({n_target}-layer target)"
            )
        # Single-use stash: must not survive a plain turn it didn't belong
        # to (the MTP gate consumes it; pre-gate disables skip the gate).
        cache_state.mtp_last_hidden = None
        cache_state.mtp_hidden_offset = None

    def _mtp_base_caches_active(self) -> bool:
        """True iff base caches must be built in the MTP-finalized layout —
        the qwen_mtp drafter actually runs on this server's decode path
        (mlx-lm backend, no --kv-bits, which disables MTP globally). All
        other configurations (plain, gemma4/mlx-vlm, PLD) keep the
        historical target-only layout untouched."""
        return (
            not self._use_vlm
            and getattr(self, "_drafter", None) is not None
            and getattr(self, "_draft_kind", None) == "qwen_mtp"
            and not self.cfg.kv_bits
        )

    def _maybe_register_base_cache(
        self,
        messages: list[dict],
        prompt_tokens: list[int],
        tools: list | None = None,
        thinking: bool = True,
        cancel_event: threading.Event | None = None,
    ):
        """Register a base cache for the system prompt if not already cached.

        F4 (codex batch-3 review): ``cancel_event`` threads through to the
        secondary system-prompt prefill (``mtp_prefill_base`` /
        ``_prefill_cache``) — a client disconnect during this potentially
        large prefill aborts between chunks instead of running to
        completion. Round 2 residual: the between-chunk checks miss a
        disconnect DURING the final chunk, so both prefill helpers check
        once more after their last chunk AND this method re-checks right
        before ``_register_base_cache`` — a dead request never proceeds to
        registration or any follow-up work. The base cache is an
        optimization, never worth blocking cancellation: on
        ``GenerationCancelled`` registration is simply skipped (the
        partially filled fresh cache is discarded)."""
        has_system_or_rotating = (
            (messages and messages[0].get("role") in ("system", "developer"))
            or self._has_rotating_cache
        )
        if not has_system_or_rotating:
            return
        sys_hash = self._system_hash(messages, tools=tools)
        # Skip if present, even when the pooled entry predates the MTP
        # layout (plain 40-entry under an MTP server): base_hit on such an
        # entry takes the gate's plain-decode fallback, so reuse still wins.
        if not sys_hash or sys_hash in self._base_caches:
            return
        system_tokens = self._extract_system_tokens(
            messages, prompt_tokens, tools=tools, thinking=thinking,
        )
        if system_tokens and len(system_tokens) < len(prompt_tokens):
            try:
                base_cache = make_prompt_cache(self._language_model)
                mtp_hidden = None
                if self._mtp_base_caches_active():
                    # MTP-finalized base: target + head entries prefilled
                    # over the system tokens with the head fed its
                    # (hidden_i, tok_{i+1}) pairs — same machinery as the
                    # runner's prompt prefill — so a seeded clone PASSES
                    # validate_mtp_cache_reuse (41-entry layout, head
                    # trailing by one, boundary hidden) instead of
                    # cold-filling the whole prompt.
                    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
                    _n_head = max(
                        1, len(getattr(self._drafter, "layers", [])) or 1
                    )
                    base_cache.extend(qwen_mtp_mod.make_head_cache(_n_head))
                    mtp_hidden = qwen_mtp_mod.mtp_prefill_base(
                        system_tokens,
                        model=self._language_model,
                        head=self._drafter,
                        prompt_cache=base_cache,
                        n_target_layers=len(self._language_model.layers),
                        prefill_step_size=self.cfg.prefill_step_size,
                        cancel_event=cancel_event,
                    )
                else:
                    self._prefill_cache(
                        base_cache, system_tokens, cancel_event=cancel_event,
                    )
                # F4 (round 2): a disconnect DURING the prefill's final
                # chunk slips past the between-chunk checks — verify once
                # more BEFORE registering, so a dead request does no
                # follow-up work (the partial fresh cache is discarded with
                # this frame). Covers the plain AND MTP branches uniformly.
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled(
                        "base cache prefill finished after client disconnect"
                    )
                self._register_base_cache(
                    messages, base_cache, system_tokens, tools=tools,
                    mtp_resume_hidden=mtp_hidden,
                )
            except GenerationCancelled:
                # F4/U13: client disconnected mid-prefill — skip registration
                # (the partial fresh cache is dropped with this frame). Must
                # precede the generic handler below and never propagate: the
                # base cache is best-effort.
                logger.debug(
                    "[Base Cache] registration skipped: cancelled mid-prefill"
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
        stop: str | list[str] | None = None,
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
                stop=stop,
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
        # U12/FIX 1 alignment: when thinking is active the chatml/glm stream
        # begins INSIDE the thought block (opener in the prompt suffix) — a
        # degenerate no-</think> output is ALL reasoning, and any tool-call
        # XML in it is a rehearsal that must not be parsed below. Mirrors
        # the streaming router and _generate_locked's parse. Structured
        # requests suppress thinking engine-side (U9), matching this flag.
        _use_thinking = thinking if thinking is not None else self.cfg.enable_thinking
        if _use_thinking and getattr(response_format, "type", None) in (
            "json_schema", "json_object",
        ):
            _use_thinking = False
        # Codex round 7, finding 3: thinking_active gates the gemma4
        # bare-opener recognition to the request's effective contract.
        thinking, content = split_thinking_and_content(
            full_text,
            model_family=self.model_family,
            started_in_thinking=_use_thinking and self.model_family != "gemma4",
            thinking_active=_use_thinking,
        )
        result.thinking = thinking
        # Codex round 3, finding 4: the channel handed to the tool parse (and
        # returned as content) follows the streaming router — with thinking
        # DISABLED on chatml/glm the whole output is content (a literal
        # </think> quote is not a boundary hiding a call before it). gemma4
        # keeps the split (its extract also feeds result.thinking); the
        # session-persistence parse above already converged its tool
        # extraction on the router-policy union.
        if not _use_thinking and self.model_family != "gemma4":
            content = full_text

        if result.finish_reason == "error":
            # U6/F1: corruption-terminated stream — the text is truncated at
            # an arbitrary point, so a partial <tool_call> block must never
            # be parsed into an executable tool call, and 'error' must not
            # be overwritten with 'tool_calls'. The API layer turns this
            # into an error response (it is not a valid OpenAI
            # finish_reason); content is kept for diagnostics only.
            result.content = content
        elif tools:
            # Round 3, finding 4 (gemma4): parse calls from the router-policy
            # content union — an orphan <channel|> with no prior thought-open
            # is content, so a call BEFORE it (which the streaming FSM
            # parsed) is not hidden by the extract-based split above.
            # Codex round 7, finding 3: the union threads the thinking
            # contract (bare-opener gate) like the split above.
            _parse_channel = (
                _content_channel_union(full_text, "gemma4", _use_thinking)
                if self.model_family == "gemma4" else content
            )
            text_part, tool_calls = parse_tool_calls(
                _parse_channel, model_family=self.model_family,
            )
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

        U3: the rebuild tokenizes WITH the session's stored prompt contract
        (tools + thinking) — a bare re-tokenization would silently drop the
        tool schema from the cached prefix, so every later HIT turn would
        answer without the tools in context.
        """
        with self._mutate_locked("compact_session"):
            self._touch_gpu()
            t0 = time.perf_counter()

            prev = self._sessions.get(session_id)
            # An evicted-but-persisted session still carries its contract on
            # disk — reload it (mirrors truncate_session) before rebuilding.
            if prev is None and self._has_disk_cache(session_id):
                prev = self._load_session_from_disk(session_id)
                if prev:
                    self._sessions[session_id] = prev
            sess_tools = getattr(prev, "tools", None) if prev else None
            sess_thinking = getattr(prev, "thinking", True) if prev else True

            prompt_tokens = self._tokenize_prompt(
                messages, thinking=sess_thinking, tools=sess_tools,
            )

            # Try base cache first
            base = self._find_base_cache(messages, tools=sess_tools)
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
                # Finding 3a (codex round 5): shutdown gate as cooperative
                # cancel — begin_shutdown() aborts this minutes-scale prefill
                # between chunks. Nothing new was published yet (the prelude
                # only re-published the disk-loaded PREV session, which equals
                # its disk copy), so aborting is mutation-free and the
                # compaction simply reruns after restart.
                self._prefill_cache(
                    prompt_cache, feed_tokens,
                    cancel_event=getattr(self, "_shutdown_cancel_event", None),
                )
            self._eval_cache(prompt_cache)

            new_offset = self._get_cache_offset(prompt_cache)
            elapsed = time.perf_counter() - t0

            # Build PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = prompt_cache
            cache_state.token_ids = prompt_tokens

            prev_tokens = prev.total_cache_tokens if prev else 0
            self._sessions[session_id] = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=new_offset,
                tools=sess_tools,
                thinking=sess_thinking,
                prompt_fingerprint=self._prompt_fingerprint(
                    sess_tools, sess_thinking,
                ),
                # U26: compaction rebuilds the cache, not the session identity
                # — keep the cumulative drafter stats.
                drafter_stats=self._drafter_stats_for(session_id),
            )

            logger.info(
                f"[Compact] session={session_id} | "
                f"{prev_tokens} -> {new_offset} tokens | "
                f"base={base_tokens_used} | "
                f"processed={len(feed_tokens)} tokens | "
                f"{elapsed:.2f}s"
            )

            # Auto-register base cache. Finding 3a: shutdown-gated cancel
            # threads into the secondary prefill too (registration is
            # best-effort and skips silently on cancel).
            self._maybe_register_base_cache(
                messages, prompt_tokens, tools=sess_tools, thinking=sess_thinking,
                cancel_event=getattr(self, "_shutdown_cancel_event", None),
            )

            self._mark_dirty(session_id)

            # New resident cache added — keep total under memory_budget_gb.
            self._evict_active_sessions_if_needed(protect_session_id=session_id)

            return {
                "session_id": session_id,
                "status": "ok",
                "cached_tokens": new_offset,
                "previous_tokens": prev_tokens,
                "base_tokens": base_tokens_used,
                "processing_time_ms": round(elapsed * 1000),
            }

    # Defensive bound on the safetensors JSON header read by the preflight's
    # metadata parse — a corrupt length prefix must never trigger a giant
    # allocation (real session-cache headers are tens of KB).
    _SAFETENSORS_HEADER_MAX_BYTES = 100 * 1024 * 1024

    @staticmethod
    def _read_safetensors_metadata(path: str) -> dict | None:
        """Read ONLY the ``__metadata__`` block of a safetensors file with
        plain Python file IO — the format is a little-endian 8-byte header
        length followed by a JSON header; tensor data is never touched and
        NO MLX call is involved (F5 round 2: the preflight must not create
        arrays on a non-owning thread). Returns None on any read/parse
        failure (fail-closed advisory)."""
        try:
            with open(path, "rb") as f:
                raw_len = f.read(8)
                if len(raw_len) != 8:
                    return None
                header_len = int.from_bytes(raw_len, "little")
                if not 0 < header_len <= MLXEngine._SAFETENSORS_HEADER_MAX_BYTES:
                    return None
                raw = f.read(header_len)
                if len(raw) != header_len:
                    # Short read == truncated file (codex round 3, finding
                    # 6): the available prefix may still parse as JSON, but
                    # the file is provably damaged — fail closed rather than
                    # report a disk hit from a header we only partially saw.
                    return None
                header = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(header, dict):
            return None
        meta = header.get("__metadata__")
        return meta if isinstance(meta, dict) else None

    def session_cache_preflight(self, session_id: str, messages: list[dict]) -> dict:
        """Web-chat cache preflight: ADVISORY report of whether this
        session's next turn will reuse its KV cache (and how). Returns
        ``{"cache_hit": bool, "cache_info": dict}`` — informational only
        (the generation itself re-resolves — and actually loads — the cache
        under its own lock, on its own thread).

        F5 (codex batch-3 review, round 2): METADATA-ONLY. The round-1
        shape ran ``_load_session_from_disk`` (mx.load + KV-cache object
        construction) on the ``engine-read`` executor thread and published
        the result into ``_sessions`` — generation later consumed those
        arrays on the engine's owning thread, violating the same-thread
        rule documented at ``_save_session_to_disk`` (VLM thread
        ownership) — and it held the engine lock across the whole disk
        load. Now:

        - the bounded ``_read_locked`` acquire (EngineBusyError on expiry,
          callers degrade like every other busy path) covers ONLY in-memory
          dict lookups + shallow snapshot copies — never disk IO, never MLX;
        - the disk fallback parses ONLY the safetensors JSON header
          (``_read_safetensors_metadata``, plain file IO) AFTER the lock is
          released — a stale answer is acceptable for an advisory preflight;
        - ``_sessions`` is never written; the authoritative disk load stays
          where it always ran: on the generation thread during the request.

        Callers run it off the event loop (reads executor); the process-mode
        proxy has no such method and callers report a neutral marker."""
        # Phase 1 — SHORT bounded lock: in-memory presence only (dict
        # lookups + shallow copies; the messages list is copied so the
        # match below never walks a list a generation is appending to).
        stored_messages: list | None = None
        cached_tokens = 0
        cache_live = False
        stored_thinking = True
        with self._read_locked("cache preflight"):
            session_state = self._sessions.get(session_id)
            if session_state is not None:
                stored_messages = list(session_state.messages)
                cached_tokens = session_state.total_cache_tokens
                cache_live = (
                    session_state.cache_state is not None
                    and session_state.cache_state.cache is not None
                )
                stored_thinking = bool(getattr(session_state, "thinking", True))
            check_disk = session_state is None and self._has_disk_cache(session_id)

        if session_state is not None:
            source = "memory"
        elif check_disk:
            # Phase 2 — disk fallback WITHOUT the engine lock: header-only
            # metadata parse (no mx.load, no cache objects, no _sessions
            # write).
            meta = self._read_safetensors_metadata(
                self._session_cache_path(session_id)
            )
            if meta is not None:
                try:
                    stored_messages = json.loads(meta.get("messages", "[]"))
                    cached_tokens = int(meta.get("total_cache_tokens", "0"))
                except (ValueError, TypeError):
                    stored_messages = None
            if not isinstance(stored_messages, list):
                return {
                    "cache_hit": False,
                    "cache_info": {
                        "type": "none",
                        "detail": (
                            "Disk cache present but metadata unreadable — "
                            "cache state unknown"
                        ),
                    },
                }
            # A persisted file implies a reloadable cache (the generation
            # thread performs — and verifies — the actual load).
            cache_live = True
            source = "disk"
        else:
            return {
                "cache_hit": False,
                "cache_info": {"type": "none", "detail": "New session"},
            }

        # Phase 3 — pure-Python match on the snapshots (no lock, no MLX).
        # Round 3, finding 4: advisory match runs under the stored session's
        # thinking contract (disk fallback keeps the default — informational
        # only; the generation re-resolves authoritatively).
        cache_hit = False
        if self._messages_match(
            stored_messages, messages, thinking_active=stored_thinking,
        ):
            cache_hit = cache_live
            new_msgs = messages[len(stored_messages):]
            suffix_desc = (
                f"{len(new_msgs)} new message(s)" if new_msgs else "retry"
            )
            cache_info = {
                "type": "kv_cache_hit" if cache_hit else "kv_cache_rebuild",
                "detail": (
                    f"KV Cache reuse ({source}): {cached_tokens} tokens "
                    f"cached, {suffix_desc}"
                    if cache_hit
                    else f"Rebuilding KV cache for {len(messages)} messages"
                ),
                "cached_tokens": cached_tokens,
                "stored_msgs": len(stored_messages),
                "source": source,
            }
        else:
            cache_info = {
                "type": "kv_cache_miss",
                "detail": (
                    f"Conversation changed, reprocessing "
                    f"{len(messages)} messages"
                ),
                "stored_msgs": len(stored_messages),
                "incoming_msgs": len(messages),
            }
        return {"cache_hit": cache_hit, "cache_info": cache_info}

    def list_sessions(self) -> list[dict]:
        """List all active sessions.

        U15: reads ``_sessions`` under the engine lock (bounded read
        acquire) — a generation mutates SessionState/_sessions in place, so
        an unlocked iteration could observe (or crash on) mid-mutation
        state."""
        with self._read_locked("session list"):
            result = []
            for sid, s in self._sessions.items():
                entry = {
                    "session_id": sid,
                    "messages": len(s.messages),
                    "cache_tokens": s.total_cache_tokens,
                    "last_used": s.last_used,
                }
                if s.drafter_stats is not None:
                    # U26 round 2 (codex F5b): SNAPSHOT the shared mutable
                    # stats dict — the reference escapes the lock, and an
                    # in-process JSON serialization after release could
                    # otherwise observe torn counter updates from the next
                    # generation's accumulate.
                    entry["drafter_stats"] = dict(s.drafter_stats)
                result.append(entry)
            return sorted(result, key=lambda x: x["last_used"], reverse=True)

    def get_session(self, session_id: str) -> dict | None:
        """Get details for a specific session. U15: locked read (bounded)."""
        with self._read_locked("session details"):
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
                # U26 round 2 (F5b): snapshot — see list_sessions.
                info["drafter_stats"] = dict(s.drafter_stats)
            return info

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its cache.

        U15: takes the engine lock UNBOUNDED (mutating admin op — it must
        not race a generation that is advancing this session's KV cache in
        place; the busy wait is bounded by that generation's length and
        U14's to_thread wrappers keep the event loop free meanwhile).
        NOT re-entrant: no engine-internal caller holds the lock here."""
        with self._mutate_locked("delete_session"):
            with self._dirty_lock:
                self._dirty_sessions.discard(session_id)
            if session_id in self._sessions:
                del self._sessions[session_id]
            # Anon-provenance hygiene (best-effort; stale ids are harmless).
            if hasattr(self, "_anon_minted_ids"):
                self._anon_minted_ids.discard(session_id)
            # U26: drop the session's cumulative drafter stats (bounds the
            # registry — deleted sessions never come back under this id).
            if hasattr(self, "_session_drafter_stats"):
                self._session_drafter_stats.pop(session_id, None)
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
        stop: str | list[str] | None = None,
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
                    stop=stop,
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
        stop: str | list[str] | None = None,
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
                    stop=stop,
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
    #
    # Codex round 5, finding 3b: these wrappers used to run their PRELUDE
    # (inspect _sessions, disk-load, PUBLISH the loaded session into
    # _sessions) BEFORE reaching the gated _rebuild_session — post-shutdown
    # calls mutated _sessions outside the gate. The gate now wraps the
    # ENTIRE operation: each public wrapper takes _mutate_locked at entry
    # and delegates to a lock-free ``*_locked`` body (the engine lock is
    # non-reentrant, so inner calls must never re-gate).

    def branch_from_turn(
        self,
        source_session_id: str,
        new_session_id: str,
        branch_turn: int,
        branch_messages: list[dict] | None = None,
    ) -> dict:
        """Branch a new session by building cache from scratch.

        Whole-op gate (finding 3b): the prelude's disk reload publishes the
        SOURCE session into ``_sessions`` — that publication must be inside
        the shutdown gate too, not only the inner rebuild."""
        with self._mutate_locked("branch_from_turn"):
            return self._branch_from_turn_locked(
                source_session_id, new_session_id, branch_turn,
                branch_messages,
            )

    def _branch_from_turn_locked(
        self,
        source_session_id: str,
        new_session_id: str,
        branch_turn: int,
        branch_messages: list[dict] | None = None,
    ) -> dict:
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

        # U3: the branch inherits the SOURCE session's prompt contract; a
        # message-only branch (no source) has no contract to inherit.
        return self._rebuild_session_locked(
            new_session_id,
            engine_messages,
            tools=getattr(source, "tools", None) if source else None,
            thinking=getattr(source, "thinking", True) if source else True,
        )

    def prepare_regenerate(self, session_id: str) -> dict:
        """Remove last assistant message and restore cache.

        Whole-op gate (finding 3b): the prelude's disk reload publishes
        into ``_sessions`` before the inner truncate — gate at entry."""
        with self._mutate_locked("prepare_regenerate"):
            session = self._sessions.get(session_id)
            # An active-LRU-evicted session is no longer in _sessions but
            # persisted to disk — reload it transparently (mirrors
            # truncate_session / branch_from_turn) before inspecting its
            # messages.
            if not session and self._has_disk_cache(session_id):
                session = self._load_session_from_disk(session_id)
                if session:
                    self._sessions[session_id] = session
            if not session or len(session.messages) < 2:
                return {"error": "nothing to regenerate"}

            last_msg = session.messages[-1]
            if last_msg.get("role") != "assistant":
                return {"error": "last message is not assistant"}

            restore_to = len(session.messages) - 2  # before user msg
            result = self._truncate_session_locked(session_id, restore_to)
            if result.get("status") == "ok":
                result["turn"] = restore_to
            return result

    def truncate_session(self, session_id: str, target_msg_count: int) -> dict:
        """Truncate session to target_msg_count messages, rebuilding cache.

        Whole-op gate (finding 3b): prelude disk reload + publication run
        inside the gate."""
        with self._mutate_locked("truncate_session"):
            return self._truncate_session_locked(session_id, target_msg_count)

    def _truncate_session_locked(
        self, session_id: str, target_msg_count: int,
    ) -> dict:
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
        # U3: rebuild under the session's own prompt contract.
        return self._rebuild_session_locked(
            session_id,
            restore_messages,
            tools=getattr(session, "tools", None),
            thinking=getattr(session, "thinking", True),
        )

    def _rebuild_session(
        self,
        session_id: str,
        messages: list[dict],
        tools: list | None = None,
        thinking: bool = True,
    ) -> dict:
        """Gated entry point kept for direct callers/tests; the public
        wrappers above gate at THEIR entry and call the lock-free body."""
        with self._mutate_locked("session rebuild (truncate/branch)"):
            return self._rebuild_session_locked(
                session_id, messages, tools=tools, thinking=thinking,
            )

    def _rebuild_session_locked(
        self,
        session_id: str,
        messages: list[dict],
        tools: list | None = None,
        thinking: bool = True,
    ) -> dict:
        """Build a fresh KV cache for the given messages (caller holds the
        engine lock via a ``_mutate_locked`` wrapper).

        U3: ``tools``/``thinking`` are the session's stored prompt contract
        (callers pass the source SessionState's fields) so the rebuilt
        prefix keeps the tool schema in context — a bare re-tokenization
        would silently drop it for every later HIT turn."""
        self._touch_gpu()
        t0 = time.perf_counter()

        prompt_tokens = self._tokenize_prompt(
            messages, thinking=thinking, tools=tools,
        )

        # Try base cache first
        prompt_cache = None
        base = self._find_base_cache(messages, tools=tools)
        feed_tokens = prompt_tokens
        if base and len(prompt_tokens) >= base.token_count:
            if prompt_tokens[:base.token_count] == base.tokens:
                prompt_cache = self._clone_base_cache(base)
                feed_tokens = prompt_tokens[base.token_count:]
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(self._language_model)

        if feed_tokens:
            # Finding 3a: the shutdown gate doubles as a cooperative cancel
            # source — begin_shutdown() aborts this prefill between chunks
            # (GenerationCancelled) instead of delaying shutdown by minutes.
            # Nothing was published yet, so aborting here is mutation-free.
            self._prefill_cache(
                prompt_cache, feed_tokens,
                cancel_event=getattr(self, "_shutdown_cancel_event", None),
            )

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
            tools=tools,
            thinking=thinking,
            prompt_fingerprint=self._prompt_fingerprint(tools, thinking),
            # U26: rebuild reinstalls the state — keep the cumulative
            # drafter stats.
            drafter_stats=self._drafter_stats_for(session_id),
        )
        self._mark_dirty(session_id)

        # New resident cache added — keep total under memory_budget_gb.
        self._evict_active_sessions_if_needed(protect_session_id=session_id)

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
        # U15: locked read (bounded) — same rationale as list_sessions.
        with self._read_locked("session stats"):
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
        and /api/cache/stats. Reads in-memory + disk state directly.

        U15: the whole snapshot is built under the engine lock (bounded read
        acquire — EngineBusyError while a generation runs) so the session /
        base-cache / memory numbers are mutually consistent and the size
        estimation never walks a cache list mid-mutation."""
        with self._read_locked("cache overview"):
            return self._cache_overview_locked()

    def _cache_overview_locked(self) -> dict:
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
                # U2: per-entry bytes + LRU recency for the admin overview.
                "size_mb": round(bc.size_bytes / 1e6, 1),
                "last_used": bc.last_used,
            }
            for h, bc in self._base_caches.items()
        ]
        total_base_bytes = sum(bc.size_bytes for bc in self._base_caches.values())

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

        # MLX process memory (Metal): live tensor working set + buffer-reuse
        # pool. These are the numbers that actually grow during an OOM, so
        # surface them next to the budget the user configured.
        try:
            mx_active_gb = round(mx.get_active_memory() / 1e9, 2)
        except Exception:  # noqa: BLE001
            mx_active_gb = None
        try:
            mx_cache_gb = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:  # noqa: BLE001
            mx_cache_gb = None

        active_sessions_kv_gb = round(total_memory_bytes / 1e9, 2)
        pool_kv_gb = round(self.cache_manager._memory_usage_gb(), 2)
        # U2: base caches are the third resident-KV consumer charged against
        # the shared budget (previously invisible + unbounded).
        base_kv_gb = round(total_base_bytes / 1e9, 2)
        budget_gb = float(getattr(self.cfg, "memory_budget_gb", 0) or 0)

        total_kv_gb = round(active_sessions_kv_gb + pool_kv_gb + base_kv_gb, 2)
        # The active-session LRU never evicts the protected / MRU / last-
        # remaining session, so the budget is BEST-EFFORT, not a hard cap. When
        # the lone un-evictable session alone exceeds the budget, the sweep
        # cannot bring us under it. Estimate that residue (largest single active
        # session KV + the always-present prefix pool) so admins can tell a
        # transient overage that the next sweep will fix apart from a structural
        # one that no eviction can fix.
        try:
            largest_session_gb = round(
                max(
                    (self._session_cache_bytes(s) for s in self._sessions.values()),
                    default=0,
                )
                / 1e9,
                2,
            )
        except Exception:  # noqa: BLE001
            largest_session_gb = 0.0
        # U2: the base-cache sweep keeps the single MRU entry resident, so it
        # is part of the un-evictable residue alongside the largest session +
        # the always-present prefix pool.
        try:
            mru_base = max(
                self._base_caches.values(),
                key=lambda bc: bc.last_used,
                default=None,
            )
            mru_base_gb = round(
                (mru_base.size_bytes if mru_base else 0) / 1e9, 2,
            )
        except Exception:  # noqa: BLE001
            mru_base_gb = 0.0
        irreducible_kv_gb = round(largest_session_gb + pool_kv_gb + mru_base_gb, 2)
        over_budget = total_kv_gb > budget_gb if budget_gb > 0 else False
        budget_unmet = over_budget and irreducible_kv_gb > budget_gb

        memory = {
            # Active per-session KV caches — the previously-unbounded consumer
            # that memory_budget_gb now bounds via active-session LRU eviction.
            "active_sessions_kv_gb": active_sessions_kv_gb,
            # Separate LRU prefix-reuse pool (cache_manager.memory_caches).
            "prefix_pool_kv_gb": pool_kv_gb,
            # U2: base-cache pool — byte-accounted + LRU-evicted against the
            # same budget as the sessions/pool above.
            "base_caches_kv_gb": base_kv_gb,
            "base_cache_count": len(self._base_caches),
            # What the eviction sweep compares against the budget.
            "total_kv_gb": total_kv_gb,
            "budget_gb": budget_gb,
            # memory_budget_gb is a BEST-EFFORT target, not a hard cap: the
            # protected/MRU/last session is never evicted.
            "budget_best_effort": True,
            # Currently above the target (a sweep may still bring this down).
            "over_budget": over_budget,
            # The target CANNOT be met by eviction: the lone un-evictable
            # session alone exceeds it. Admins should raise memory_budget_gb.
            "budget_unmet": budget_unmet,
            "irreducible_kv_gb": irreducible_kv_gb,
            # MLX/Metal process memory.
            "mx_active_gb": mx_active_gb,
            "mx_cache_gb": mx_cache_gb,
            "mlx_cache_limit_gb": float(getattr(self.cfg, "mlx_cache_limit_gb", 0) or 0),
        }

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
            "memory": memory,
            "cache_dir": cache_dir,
        }

    def clear_caches(self) -> dict:
        """Clear all KV caches (memory sessions + base caches + cache_manager +
        disk files). Returns counts cleared. Used by /api/admin/cache/reset.

        U15: takes the engine lock UNBOUNDED (mutating admin op — clearing
        ``_sessions``/``_base_caches`` under a running generation would pull
        the live cache out from under it). Not re-entrant: ``reset()`` is
        the only internal caller and holds no lock."""
        with self._mutate_locked("clear_caches"):
            cleared = {"memory_sessions": 0, "disk_files": 0, "base_caches": 0}

            with self._dirty_lock:
                self._dirty_sessions.clear()

            cleared["memory_sessions"] += len(self._sessions)
            self._sessions.clear()
            # Anon-provenance hygiene: no sessions remain → no minted ids remain.
            if hasattr(self, "_anon_minted_ids"):
                self._anon_minted_ids.clear()
            # U26: no sessions remain → no cumulative drafter stats remain.
            if hasattr(self, "_session_drafter_stats"):
                self._session_drafter_stats.clear()

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
