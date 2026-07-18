"""Neutral leaf module for the session-cache boundary.

Shared dataclasses, enums, precompiled regexes, and stateless helpers used
by both ``MLXEngine`` and ``SessionCacheMixin``. This module is a LEAF: it
may import only stdlib, ``mlx_vlm``, and ``tool_parser`` — NEVER
``mlx_engine`` — so the dependency arrow stays

    mlx_engine -> session_cache_mixin -> session_cache_types

and the import cycle is broken. All contents here are byte-identical moves
out of ``mlx_engine.py`` (behavior-preserving); ``mlx_engine`` re-imports
them so existing bare-name references and public test imports keep working.
"""

import enum
import re
import time
from dataclasses import dataclass, field

from mlx_vlm.generate import PromptCacheState

from mlx_soloheaven.engine.tool_parser import (
    content_segments,
    parse_tool_calls,
)


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
