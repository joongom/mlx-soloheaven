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

import json
import logging
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# Regex for a permissive JSON object (used for json_object mode)
_JSON_OBJECT_REGEX = (
    r'\{[\s\S]*\}'  # anything inside braces; outlines compiles more carefully
)


# Schema index cache: hash(schema) -> (Index, vocab_size)
_INDEX_CACHE: dict[str, Any] = {}


def _get_vocab(tokenizer) -> tuple[object, int]:
    """Build an outlines-core Vocabulary from an mlx-lm TokenizerWrapper.

    Returns (vocabulary, logits_vocab_size).
    """
    from outlines_core import Vocabulary

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    # Collect EOS ids from both mlx-lm wrapper and HF tokenizer
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_ids"):
        eos_ids.update(tokenizer.eos_token_ids or [])
    eos = getattr(hf_tok, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            eos_ids.update(eos)
        else:
            eos_ids.add(eos)
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

    key = cache_key or regex
    index = _INDEX_CACHE.get(key)
    if index is None:
        index = _compile_regex_index(regex, vocab)
        _INDEX_CACHE[key] = index

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
    index = _INDEX_CACHE.get(regex)
    if index is None:
        index = _compile_regex_index(regex, vocab)
        _INDEX_CACHE[regex] = index
    return _LogitsProcessor(index, description="regex")
