"""Gemma4 RotatingKVCache / PLD cache-safety units (reviewed fixes).

Focused, NO-real-model coverage for the sliding-window safety work:

(a) FIX-1 headroom gate: a draft is only verified when every rotating layer
    stays trimmable AFTER the verify chunk (offset + y.size + want < max_size);
    once headroom is gone, speculation stays off for the rest of the
    generation (offsets are monotonic).
(b) FIX-2 fail-closed rewind: a lying cache (trim under-reports, or a layer
    silently under-trims while layer 0 trims fully — upstream
    ``trim_prompt_cache`` returns ONLY cache[0]'s count) fires the corruption
    signal, and the engine-level callback invalidates ``cache_state`` so the
    ghost-token cache is never written back.
(c) FIX-3 lm-path reuse gate: wrapped+divergent → cold-fill,
    wrapped+append-only → reuse (real ``mlx_lm`` cache classes with dummy
    arrays), including the ``offset == max_size`` boundary (is_trimmable is
    already False there — the ``>=`` consistency fix).
(d) FIX-6 structured-output gate: ``{"type": "text"}`` / unknown types build
    no FSM and must KEEP PLD; only json_schema / json_object disable it.
(e) FIX-7: kv_bits>0 + rotating-cache model raises at load with a clear
    message (RotatingKVCache.to_quantized is NotImplementedError upstream).

Mock patterns mirror tests/test_pld_prompt_only.py (mx-array mock models) and
tests/test_rotating_cache_prefix.py (_FakeCacheState + find_prefix_length).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    make_prompt_cache as lm_make_prompt_cache,
)

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine
from mlx_soloheaven.engine.pld import pld_generate_step


# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------


class _HonestCache:
    """Plain trimmable KV-cache mock (mirrors mlx_lm KVCache.trim clamping)."""

    def __init__(self):
        self.offset = 0
        self.state = mx.array([0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class _RotatingMockCache(_HonestCache):
    """RotatingKVCache-shaped mock: cumulative offset, ring capacity
    ``max_size``, ``is_trimmable() == offset < max_size`` (mirrors mlx_lm)."""

    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
        self.keep = 0
        self.trim_calls: List[int] = []

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        self.trim_calls.append(n)
        return super().trim(n)


class _UnderReportingCache(_HonestCache):
    """Lying cache: trim() always trims (and reports) one token SHORT."""

    def trim(self, n):
        m = min(self.offset, max(0, n - 1))
        self.offset -= m
        return m


class _SilentUnderTrimCache(_HonestCache):
    """Lying cache: trim() REPORTS a full trim but never moves its offset.

    Models the reviewer's scenario where per-layer offsets diverged: layer 0
    trims fully (and is the ONLY return value upstream trim_prompt_cache
    surfaces) while this layer under-trims undetected.
    """

    def trim(self, n):
        return n


class _RecordingConstModel(nn.Module):
    """Always argmaxes to token 0; records (pre_offset, chunk_len) per forward
    so tests can prove the headroom gate ran BEFORE the verify forward."""

    def __init__(self, vocab: int = 16):
        super().__init__()
        self.vocab = vocab
        self.layers = [object()]
        self.forwards: List[tuple] = []

    def __call__(self, x, cache=None):
        T = x.shape[1]
        self.forwards.append(
            (cache[0].offset if cache else -1, T)
        )
        row = mx.concatenate(
            [mx.array([[1e9]]), mx.zeros((1, self.vocab - 1))], axis=1
        )
        if cache is not None:
            for c in cache:
                c.offset += T
        return mx.broadcast_to(row[:, None, :], (1, T, self.vocab))


class _DivergentModel(nn.Module):
    """Deterministic next-token mapping that diverges from the PLD draft
    (mirrors pld.py self-test 4): prompt [1..8, 1..5] drafts [6,7,8], model
    truth is 5→6→10→11→12, so draft[1] (=7) is rejected → _rewind fires."""

    def __init__(self, vocab: int = 16):
        super().__init__()
        self.vocab = vocab
        self.layers = [object()]

    def _next(self, tok: int) -> int:
        mapping = {5: 6, 6: 10, 10: 11, 11: 12, 7: 8, 8: 9,
                   1: 2, 2: 3, 3: 4, 4: 5, 0: 1}
        return mapping.get(tok, (tok + 1) % self.vocab)

    def __call__(self, x, cache=None):
        T = x.shape[1]
        toks = x[0].tolist()
        rows = []
        for t in toks:
            row = mx.zeros((1, self.vocab))
            row[0, self._next(int(t))] = 1e9
            rows.append(row)
        out = mx.stack(rows, axis=1)
        if cache is not None:
            for c in cache:
                c.offset += T
        return out


_REJECTION_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5]
_REJECTION_OUTPUT = [6, 10, 11, 12]  # greedy ground truth for max_tokens=4


class _StubTokenizer:
    """Minimal tokenizer for _pld_response_adapter (no EOS, no detokenizer)."""

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class _FakeCacheState:
    """Minimal stand-in for mlx-vlm's PromptCacheState (copied pattern from
    tests/test_rotating_cache_prefix.py)."""

    def __init__(self, cache, token_ids):
        self.cache = cache
        self.token_ids = token_ids

    def find_prefix_length(self, new_ids: list) -> int:
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len


def _make_lm_engine(*, pld_enabled: bool) -> MLXEngine:
    """Bare mlx-lm-path engine (pattern from tests/test_pld_path_guard.py)."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.pld_enabled = pld_enabled
    eng._use_vlm = False
    eng.model_id = "stub"
    eng.tokenizer = _StubTokenizer()
    eng._language_model = SimpleNamespace(make_cache=lambda: [_HonestCache()])
    return eng


def _advance(layer, n: int):
    """Feed n dummy tokens through a real mlx_lm cache layer."""
    k = mx.zeros((1, 2, n, 4), dtype=mx.float16)
    layer.update_and_fetch(k, k)


# ---------------------------------------------------------------------------
# (a) FIX-1: headroom gate trips BEFORE the verify forward and stays off.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_size", [12, 13])
def test_headroom_gate_trips_before_verify_and_stays_off(max_size):
    """Trace (num_draft_tokens=1, prompt len 6, 1-token drafts, 2 tokens per
    accepted round): verify chunks land at offsets 0(+7), 7(+2), 9(+2); the
    next round's check offset + y.size + want >= max_size trips BEFORE any
    forward, and the rest of the generation is single-token only.

    max_size=13 exercises the exact ``>=`` boundary (11 + 2 == 13).
    """
    cache = [_RotatingMockCache(max_size=max_size)]
    model = _RecordingConstModel()
    prompt = mx.array([1, 2, 3, 0, 0, 0], dtype=mx.uint32)
    results = list(
        pld_generate_step(
            prompt, model, num_draft_tokens=1, max_tokens=8, ngram_k=3,
            prompt_cache=cache,
        )
    )
    out = [t for t, _lp, _fd in results]
    flags = [fd for _t, _lp, fd in results]
    assert out == [0] * 8, f"lossless output broken: {out}"

    # Speculation actually ran before the trip (the test is not vacuous)...
    multi = [(pre, T) for pre, T in model.forwards if T > 1]
    assert multi, "expected at least one speculative verify chunk"
    # ...and the gate ran BEFORE every verify forward: no multi-token chunk
    # may leave the rotating layer untrimmable (pre + T >= max_size would be
    # an un-rewindable partial rejection).
    for pre, T in multi:
        assert pre + T < max_size, (
            f"verify forward at offset {pre} (+{T}) crossed max_size="
            f"{max_size} — gate did not run before the forward"
        )

    # Stays off for the rest of the generation: after the LAST multi-token
    # forward, every forward is single-token, no token is from_draft, and
    # generation continued past the wrap without speculation.
    last_multi = max(i for i, (_p, T) in enumerate(model.forwards) if T > 1)
    assert all(T == 1 for _p, T in model.forwards[last_multi + 1:])
    assert any(flags), "no draft was ever accepted (test is vacuous)"
    assert flags[-2:] == [False, False], (
        f"speculation re-engaged after the wrap trip: {flags}"
    )
    assert cache[0].offset >= max_size, (
        "generation never reached the wrap — gate trip untested"
    )
    # No rejection ever happened (const model accepts its own drafts), so the
    # gate — not a failed trim — is what stopped speculation.
    assert cache[0].trim_calls == []


# ---------------------------------------------------------------------------
# (b) FIX-2: fail-closed rewind — lying caches fire the corruption signal,
#     and the engine invalidates cache_state.
# ---------------------------------------------------------------------------


def test_rewind_underreporting_trim_fires_corruption_signal():
    """cache[0].trim under-reports (and under-trims) → trimmed != requested →
    on_cache_corruption fires exactly once and output stays greedy-correct
    (speculation disabled, single-token fallback)."""
    cache = [_UnderReportingCache()]
    model = _DivergentModel()
    corrupted: List[bool] = []
    out = [
        t for t, _lp, _fd in pld_generate_step(
            mx.array(_REJECTION_PROMPT, dtype=mx.uint32), model,
            num_draft_tokens=3, max_tokens=4, ngram_k=3,
            prompt_cache=cache,
            on_cache_corruption=lambda: corrupted.append(True),
        )
    ]
    assert out == _REJECTION_OUTPUT, f"expected {_REJECTION_OUTPUT}, got {out}"
    assert len(corrupted) == 1, (
        f"corruption signal fired {len(corrupted)} times (expected 1)"
    )


def test_rewind_per_layer_undertrim_fires_corruption_signal():
    """Layer 0 trims fully (and is the ONLY return value upstream
    trim_prompt_cache surfaces); layer 1 reports success but never moves its
    offset. The per-layer post-condition must catch the shortfall."""
    honest, liar = _HonestCache(), _SilentUnderTrimCache()
    model = _DivergentModel()
    corrupted: List[bool] = []
    out = [
        t for t, _lp, _fd in pld_generate_step(
            mx.array(_REJECTION_PROMPT, dtype=mx.uint32), model,
            num_draft_tokens=3, max_tokens=4, ngram_k=3,
            prompt_cache=[honest, liar],
            on_cache_corruption=lambda: corrupted.append(True),
        )
    ]
    assert out == _REJECTION_OUTPUT
    assert len(corrupted) == 1, (
        "per-layer shortfall went undetected (trim_prompt_cache's return is "
        "only cache[0]'s count — layer 1 under-trimmed silently)"
    )
    # The two layers' offsets prove the divergence the check caught.
    assert liar.offset > honest.offset


def test_engine_invalidates_cache_state_on_pld_corruption():
    """End-to-end on the lm path: _run_lm_legacy wires on_cache_corruption →
    cache_state is invalidated IMMEDIATELY (cache None, token_ids None) and
    the post-loop skip flag is set, so the ghost-token cache is never
    written back."""
    eng = _make_lm_engine(pld_enabled=True)
    eng._language_model = _DivergentModel()
    honest, liar = _HonestCache(), _SilentUnderTrimCache()
    cache_state = SimpleNamespace(cache=[honest, liar], token_ids=None)

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=list(_REJECTION_PROMPT),
        max_tokens=4,
        sampler=None,
        logits_processors=None,
        response_format=None,
    )
    assert prompt_cache is not None
    results = list(gen_iter)  # drive the stream to completion
    assert [r.token for r in results] == _REJECTION_OUTPUT

    # Invalidation happened inside the corruption callback itself (robust to
    # abandoned generators that never reach generate_stream's post-loop).
    assert cache_state.cache is None, "corrupt cache survived in cache_state"
    assert cache_state.token_ids is None
    # And the post-loop skip flag is set for when the loop DOES complete.
    assert eng._pld_cache_invalid is True


# ---------------------------------------------------------------------------
# (c) FIX-3 + nit: lm-path reuse gate with REAL mlx_lm cache classes.
# ---------------------------------------------------------------------------


def _wrapped_rotating(feed: int, max_size: int = 8) -> RotatingKVCache:
    rot = RotatingKVCache(max_size)
    for _ in range(feed // max_size):
        _advance(rot, max_size)
    rem = feed % max_size
    if rem:
        _advance(rot, rem)
    assert rot.offset == feed
    return rot


def test_safe_to_reuse_wrapped_divergent_cold_fills_real_caches():
    rot = _wrapped_rotating(16)
    full = KVCache()
    _advance(full, 16)
    cached_ids = list(range(100, 116))
    cs = _FakeCacheState(cache=[full, rot], token_ids=cached_ids)
    divergent = cached_ids[:8] + [9999, 9998]
    assert MLXEngine._safe_to_reuse_cache(cs, divergent) is False


def test_safe_to_reuse_wrapped_append_only_reuses_real_caches():
    rot = _wrapped_rotating(16)
    full = KVCache()
    _advance(full, 16)
    cached_ids = list(range(100, 116))
    cs = _FakeCacheState(cache=[full, rot], token_ids=cached_ids)
    append_only = cached_ids + [9999, 9998]
    assert MLXEngine._safe_to_reuse_cache(cs, append_only) is True


def test_safe_to_reuse_offset_equals_max_size_counts_as_wrapped():
    """Nit fix (>= not >): at offset == max_size, is_trimmable() is already
    False, so the prefix-trim reuse path cannot rewind — divergence must
    cold-fill at the boundary too, while strict append remains allowed."""
    rot = _wrapped_rotating(8, max_size=8)  # offset == max_size exactly
    assert not rot.is_trimmable()  # documents why >= is the right boundary
    cached_ids = list(range(100, 108))
    cs = _FakeCacheState(cache=[rot], token_ids=cached_ids)

    divergent = cached_ids[:4] + [9999, 9998]
    assert MLXEngine._safe_to_reuse_cache(cs, divergent) is False

    append_only = cached_ids + [9999]
    assert MLXEngine._safe_to_reuse_cache(cs, append_only) is True


def test_lm_path_wrapped_divergent_drops_cache_and_cold_fills(monkeypatch):
    """_run_lm_legacy: wrapped + divergent prompt → cache_state dropped, a
    FRESH cache is built, and the FULL prompt is re-prefilled."""
    eng = _make_lm_engine(pld_enabled=False)
    fresh_cache = [_HonestCache()]
    eng._language_model = SimpleNamespace(make_cache=lambda: fresh_cache)

    calls = []

    def _fake_lm_stream(model, tokenizer, prompt=None, **kw):
        calls.append({"prompt": prompt, **kw})
        return iter(())

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", _fake_lm_stream)

    rot = _wrapped_rotating(16)
    cached_ids = list(range(100, 116))
    cs = _FakeCacheState(cache=[rot], token_ids=cached_ids)
    divergent = cached_ids[:8] + [9999, 9998]

    _gen, prompt_cache = eng._run_lm_legacy(
        cache_state=cs,
        prompt_token_ids=list(divergent),
        max_tokens=4,
        sampler=None,
        logits_processors=None,
        response_format=None,
    )
    assert cs.cache is None and cs.token_ids is None, "wrapped+divergent must drop"
    assert prompt_cache is fresh_cache, "cold-fill must build a FRESH cache"
    assert calls and calls[0]["prompt"] == divergent, "full prompt re-prefill"


def test_lm_path_wrapped_append_only_reuses_cache(monkeypatch):
    """_run_lm_legacy: wrapped + strict append → the SAME cache object is
    reused and only the suffix is fed (no trim, offsets untouched)."""
    eng = _make_lm_engine(pld_enabled=False)

    calls = []

    def _fake_lm_stream(model, tokenizer, prompt=None, **kw):
        calls.append({"prompt": prompt, **kw})
        return iter(())

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", _fake_lm_stream)

    rot = _wrapped_rotating(16)
    cached = [rot]
    cached_ids = list(range(100, 116))
    cs = _FakeCacheState(cache=cached, token_ids=cached_ids)
    append_only = cached_ids + [777, 778]

    _gen, prompt_cache = eng._run_lm_legacy(
        cache_state=cs,
        prompt_token_ids=list(append_only),
        max_tokens=4,
        sampler=None,
        logits_processors=None,
        response_format=None,
    )
    assert prompt_cache is cached, "append-only must reuse the existing cache"
    assert cs.cache is cached
    assert rot.offset == 16, "no trim may run on a pure append"
    assert calls and calls[0]["prompt"] == [777, 778], "suffix-only feed"


# ---------------------------------------------------------------------------
# (d) FIX-6: response_format type gating — text/unknown KEEP PLD,
#     json_schema/json_object disable it.
# ---------------------------------------------------------------------------


def _run_with_response_format(monkeypatch, response_format):
    """Drive _run_lm_legacy with pld_enabled=True and return which generation
    path was taken ('PLD' or 'LM')."""
    eng = _make_lm_engine(pld_enabled=True)
    cache_state = SimpleNamespace(cache=None, token_ids=None)

    monkeypatch.setattr(
        mlx_engine_module, "_pld_response_adapter",
        lambda pld_iter, tokenizer: ("PLD", pld_iter),
    )
    monkeypatch.setattr(
        mlx_engine_module, "lm_stream_generate",
        lambda *a, **kw: ("LM", None),
    )
    gen_iter, _cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        sampler=None,
        logits_processors=None,
        response_format=response_format,
    )
    return gen_iter[0]


def test_response_format_text_keeps_pld(monkeypatch):
    """{"type": "text"} builds NO FSM (generate_stream only builds one for
    json_schema/json_object) — PLD must stay enabled."""
    rf = SimpleNamespace(type="text")
    assert _run_with_response_format(monkeypatch, rf) == "PLD"


def test_response_format_none_keeps_pld(monkeypatch):
    assert _run_with_response_format(monkeypatch, None) == "PLD"


def test_response_format_unknown_type_keeps_pld(monkeypatch):
    rf = SimpleNamespace(type="xml")  # unknown: no FSM is ever built for it
    assert _run_with_response_format(monkeypatch, rf) == "PLD"


def test_response_format_json_schema_disables_pld(monkeypatch):
    rf = SimpleNamespace(type="json_schema")
    assert _run_with_response_format(monkeypatch, rf) == "LM"


def test_response_format_json_object_disables_pld(monkeypatch):
    rf = SimpleNamespace(type="json_object")
    assert _run_with_response_format(monkeypatch, rf) == "LM"


# ---------------------------------------------------------------------------
# (e) FIX-7: kv_bits>0 + rotating-cache model must raise at load.
# ---------------------------------------------------------------------------


def _make_kv_bits_engine(*, kv_bits: int, rotating: bool, use_vlm: bool = False):
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.kv_bits = kv_bits
    eng._use_vlm = use_vlm
    eng.model_id = "stub-gemma4"
    # Derive the flag EXACTLY the way load() does, over a mocked cache
    # construction that returns a real RotatingKVCache.
    layers = [RotatingKVCache(1024), KVCache()] if rotating else [KVCache()]
    stub_model = SimpleNamespace(make_cache=lambda: list(layers))
    test_cache = lm_make_prompt_cache(stub_model)
    eng._has_rotating_cache = any(
        type(c).__name__ == "RotatingKVCache" for c in test_cache
    )
    eng._sliding_window_size = 1024 if rotating else 0
    return eng


def test_kv_bits_with_rotating_cache_raises_with_clear_message():
    eng = _make_kv_bits_engine(kv_bits=8, rotating=True)
    with pytest.raises(ValueError) as exc:
        eng._reject_kv_bits_on_rotating_cache()
    msg = str(exc.value)
    assert "kv-bits" in msg
    assert "RotatingKVCache" in msg
    assert "1024" in msg  # names the sliding window so the user can act


def test_kv_bits_without_rotating_cache_is_allowed():
    eng = _make_kv_bits_engine(kv_bits=8, rotating=False)
    eng._reject_kv_bits_on_rotating_cache()  # must not raise


def test_kv_bits_zero_with_rotating_cache_is_allowed():
    eng = _make_kv_bits_engine(kv_bits=0, rotating=True)
    eng._reject_kv_bits_on_rotating_cache()  # must not raise


def test_kv_bits_on_vlm_path_is_not_rejected_here():
    """The vlm path ignores --kv-bits with a warning elsewhere; the loud
    reject is mlx-lm-path only."""
    eng = _make_kv_bits_engine(kv_bits=8, rotating=True, use_vlm=True)
    eng._reject_kv_bits_on_rotating_cache()  # must not raise
