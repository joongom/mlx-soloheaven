"""Structured output (JSON mode / JSON schema) via outlines-core FSM masking.

Implements OpenAI's `response_format` parameter as a logits-level constraint
without prompt engineering. Uses outlines-core's Rust FSM engine to produce
token masks that force the generated text to conform to a regex/JSON schema.

Usage:
    proc = build_json_schema_processor(schema, tokenizer)
    # proc is a callable compatible with mlx-lm's logits_processors kwarg
    for resp in stream_generate(model, tok, prompt, logits_processors=[proc]):
        ...
    if proc.guide.is_finished():
        # schema fully satisfied
        break
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
from collections import OrderedDict
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# Regex for a permissive JSON object (used for json_object mode)
_JSON_OBJECT_REGEX = (
    r'\{[\s\S]*\}'  # anything inside braces; outlines compiles more carefully
)


# Schema index cache: "<tokenizer_fp>:<schema-or-regex key>" -> Index.
# U19: the key is prefixed with a TOKENIZER FINGERPRINT — an FSM index bakes
# in the vocabulary it was compiled against, so a key without tokenizer
# identity let multi-model deployments (--models) reuse a stale index built
# for another model's vocab (token-id masks applied to the wrong vocabulary:
# silently corrupted structured output). Bounded LRU (the old dict grew
# without limit across schemas AND now across models); guarded by a lock —
# per-model engines compile from different worker threads.
_INDEX_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_INDEX_CACHE_MAX = 64
_INDEX_CACHE_LOCK = threading.Lock()

# Attribute used to memoize the fingerprint on the tokenizer instance itself
# (the full-vocab hash is computed at most once per tokenizer object).
_TOKENIZER_FP_ATTR = "_soloheaven_fsm_tokenizer_fp"


def _collect_fingerprint_eos_ids(tokenizer, hf_tok) -> list[int]:
    """EOS ids EXACTLY as ``_get_vocab`` collects them (wrapper
    ``eos_token_ids`` list + HF ``eos_token_id`` scalar-or-list). They are
    part of the compiled index's identity: ``_get_vocab`` EXCLUDES the EOS
    ids from the outlines Vocabulary, so the same vocab with different EOS
    config compiles a DIFFERENT index (U19 round 2, codex F2)."""
    eos_ids: set[int] = set()
    if hasattr(tokenizer, "eos_token_ids"):
        eos_ids.update(int(i) for i in (tokenizer.eos_token_ids or []))
    eos = getattr(hf_tok, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            eos_ids.update(int(i) for i in eos)
        else:
            eos_ids.add(int(eos))
    return sorted(eos_ids)


def _tokenizer_fingerprint(tokenizer) -> str:
    """Stable identity of a tokenizer's compiled-vocabulary inputs for the
    index-cache key.

    Hashes name_or_path + the EOS token ids + the full (token, id)
    vocabulary — the exact inputs the compiled FSM index depends on
    (``_get_vocab`` bakes the EOS set into the Vocabulary it builds).

    U19 round 2 (codex F2): the serialization is CANONICAL — every variable-
    length field is length-prefixed (fixed-width big-endian u32) and ids are
    fixed-width i64, so no two distinct vocabularies can concatenate to the
    same byte stream ({'a':0,'0b':1} vs {'a0':0,'b':1} collided under the old
    delimiter-free concat). The FULL sha256 hexdigest is kept (no 64-bit
    truncation). The full-vocab hash runs once per tokenizer INSTANCE
    (memoized on the object), i.e. once per model load, so the cost
    (~tens of ms) never lands on the per-request path.
    """
    fp = getattr(tokenizer, _TOKENIZER_FP_ATTR, None)
    if isinstance(fp, str) and fp:
        return fp
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    h = hashlib.sha256()
    name_b = str(getattr(hf_tok, "name_or_path", "")).encode("utf-8", "replace")
    h.update(struct.pack(">I", len(name_b)))
    h.update(name_b)
    eos_sorted = _collect_fingerprint_eos_ids(tokenizer, hf_tok)
    h.update(struct.pack(">I", len(eos_sorted)))
    for eid in eos_sorted:
        h.update(struct.pack(">q", eid))
    vocab = hf_tok.get_vocab()
    h.update(struct.pack(">I", len(vocab)))
    for tstr, tid in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        # surrogatepass: byte-level BPE vocab strings can contain lone
        # surrogates that strict utf-8 refuses to encode.
        tb = tstr.encode("utf-8", "surrogatepass")
        h.update(struct.pack(">I", len(tb)))
        h.update(tb)
        h.update(struct.pack(">q", int(tid)))
    fp = h.hexdigest()
    try:
        setattr(tokenizer, _TOKENIZER_FP_ATTR, fp)
    except Exception:  # noqa: BLE001 — slotted/frozen wrapper: just recompute
        pass
    return fp


def _index_cache_get(key: str) -> Any | None:
    with _INDEX_CACHE_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is not None:
            _INDEX_CACHE.move_to_end(key)
        return index


def _index_cache_put(key: str, index: Any) -> None:
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[key] = index
        _INDEX_CACHE.move_to_end(key)
        while len(_INDEX_CACHE) > _INDEX_CACHE_MAX:
            _INDEX_CACHE.popitem(last=False)


# U19 round 2 (codex F2, optional improvement): single-flight the compile.
# Two workers missing the same key used to both run the 10-200ms Index
# compile (harmless duplicate work, but wasteful). A per-key lock makes the
# second requester WAIT and then take the first's cached result. The lock
# registry is guarded by _INDEX_CACHE_LOCK and pruned after each compile —
# a late third requester simply mints a fresh lock and hits the cache on
# its re-check.
_INDEX_COMPILE_LOCKS: dict[str, threading.Lock] = {}


def _index_cache_get_or_compile(key: str, compile_fn) -> Any:
    """Return the cached index for ``key``, compiling (single-flight) on a
    miss. ``compile_fn`` is a zero-arg callable producing the index."""
    index = _index_cache_get(key)
    if index is not None:
        return index
    with _INDEX_CACHE_LOCK:
        lock = _INDEX_COMPILE_LOCKS.setdefault(key, threading.Lock())
    try:
        with lock:
            # Re-check under the per-key lock: a concurrent compiler may
            # have already populated the cache while we waited.
            index = _index_cache_get(key)
            if index is None:
                index = compile_fn()
                _index_cache_put(key, index)
            return index
    finally:
        with _INDEX_CACHE_LOCK:
            _INDEX_COMPILE_LOCKS.pop(key, None)


def _get_vocab(tokenizer) -> tuple[object, int]:
    """Build an outlines-core Vocabulary from an mlx-lm TokenizerWrapper.

    Returns (vocabulary, logits_vocab_size).
    """
    from outlines_core import Vocabulary

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    # Collect EOS ids from both mlx-lm wrapper and HF tokenizer — the SAME
    # collector feeds the fingerprint (U19 round 2: the compiled vocabulary
    # varies with these ids, so they are part of the cache-key identity).
    eos_ids = set(_collect_fingerprint_eos_ids(tokenizer, hf_tok))
    if not eos_ids:
        raise ValueError(
            "Cannot build vocabulary: tokenizer exposes no EOS token. "
            "Structured output requires a defined EOS."
        )
    primary_eos = min(eos_ids)

    vocab_dict: dict[str, list[int]] = {}
    raw_vocab = hf_tok.get_vocab()
    max_id = 0
    for tstr, tid in raw_vocab.items():
        if tid > max_id:
            max_id = tid
        if tid in eos_ids:
            continue
        vocab_dict.setdefault(tstr, []).append(tid)

    vocab = Vocabulary(primary_eos, vocab_dict)
    logits_vocab_size = max_id + 1  # upper bound on logits width
    return vocab, logits_vocab_size


def _compile_regex_index(regex: str, vocab) -> object:
    """Compile an outlines-core Index from a regex pattern."""
    from outlines_core import Index
    return Index(regex, vocab)


def _schema_to_regex(schema: dict | str) -> str:
    """Convert a JSON Schema to a regex pattern via outlines-core."""
    from outlines_core.json_schema import build_regex_from_schema
    schema_json = json.dumps(schema) if isinstance(schema, dict) else schema
    return build_regex_from_schema(schema_json)


class _LogitsProcessor:
    """FSM-based logits processor.

    NOTE: stateful — instantiate a fresh one per generation request.
    Not thread-safe; mlx-lm calls this serially per sequence.
    """

    def __init__(self, index, description: str = ""):
        from outlines_core import Guide
        self.guide = Guide(index)
        self.description = description
        self._initialized = False
        self._prev_len = 0

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        cur_len = int(tokens.size) if hasattr(tokens, "size") else len(tokens)
        if not self._initialized:
            # First call: tokens is the prompt. Don't advance the FSM —
            # the FSM tracks only *generated* tokens, and the prompt is
            # (by convention) not part of the schema output.
            self._initialized = True
            self._prev_len = cur_len
        else:
            # Advance FSM by one token (the last sampled token).
            if cur_len > self._prev_len and not self.guide.is_finished():
                try:
                    last_tok = int(tokens[-1].item()) if hasattr(tokens, "item") else int(tokens[-1])
                    self.guide.advance(last_tok)
                except Exception as e:
                    # Malformed FSM advance — log and continue (mask will lock
                    # to EOS-only state, letting generation terminate)
                    logger.debug(f"[structured] guide.advance failed: {e}")
            self._prev_len = cur_len

        vsize = logits.shape[-1]
        packed_size = (vsize + 31) // 32
        mask_packed = np.zeros(packed_size, dtype=np.int32)
        self.guide.write_mask_into(mask_packed.ctypes.data, packed_size, 4)
        # Unpack bits (little-endian bit order within each int32 word)
        bits = np.unpackbits(mask_packed.view(np.uint8), bitorder="little")[:vsize]
        mask_mx = mx.array(bits > 0)

        if len(logits.shape) == 2:
            mask_shape = mask_mx[None]
        else:
            mask_shape = mask_mx
        neg_inf = mx.array(-1e9, dtype=logits.dtype)
        return mx.where(mask_shape, logits, neg_inf)

    def is_finished(self) -> bool:
        return self.guide.is_finished()


def build_json_schema_processor(schema: dict | str, tokenizer, cache_key: str | None = None):
    """Build a logits processor that constrains output to a JSON schema.

    Args:
        schema: JSON schema (dict) or JSON string.
        tokenizer: mlx-lm TokenizerWrapper.
        cache_key: Optional cache key for the compiled FSM index. Useful when
            the same schema is reused across requests (compilation is the
            expensive step; Index compile is 10-200ms per schema).

    Returns:
        Callable `(tokens, logits) -> logits` with extra methods:
        - .guide.is_finished() — whether the FSM reached a terminal state
    """
    vocab, _ = _get_vocab(tokenizer)
    regex = _schema_to_regex(schema)

    # U19: tokenizer fingerprint in the key — never reuse an index compiled
    # against another model's vocabulary. Round 2: compile is single-flight
    # (simultaneous misses on the same key run ONE compile).
    key = f"{_tokenizer_fingerprint(tokenizer)}:{cache_key or regex}"
    index = _index_cache_get_or_compile(
        key, lambda: _compile_regex_index(regex, vocab)
    )

    return _LogitsProcessor(index, description="json_schema")


def build_json_object_processor(tokenizer):
    """Build a logits processor that constrains output to any valid JSON object.

    Uses the loose `{"type":"object"}` schema. For strict schemas, prefer
    `build_json_schema_processor()`.
    """
    return build_json_schema_processor(
        {"type": "object"},
        tokenizer,
        cache_key="__json_object__",
    )


def build_regex_processor(regex: str, tokenizer):
    """Build a logits processor from a raw regex pattern."""
    vocab, _ = _get_vocab(tokenizer)
    # U19: same tokenizer-scoped keying (and round-2 single-flight compile)
    # as build_json_schema_processor.
    key = f"{_tokenizer_fingerprint(tokenizer)}:{regex}"
    index = _index_cache_get_or_compile(
        key, lambda: _compile_regex_index(regex, vocab)
    )
    return _LogitsProcessor(index, description="regex")
