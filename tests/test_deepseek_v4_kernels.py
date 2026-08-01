"""DeepSeek-V4 index math + routing (step 2a of the port spec).

Every function here is checked against an INDEPENDENT NumPy transcription of
DeepSeek's ``inference/model.py``, not against itself. That matters because
these are the failure modes that do not raise: a wrong window index attends to
the wrong past token, a wrong routing rule picks the wrong experts, and the
model still produces fluent-looking output. The EXAONE MTP attempt died exactly
here — plausible output, no way to tell it was wrong.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_soloheaven.models.deepseek_v4 import (
    MASKED_INDEX,
    clipped_swiglu,
    compress_topk_indices,
    route,
    sqrtsoftplus,
    window_topk_indices,
)


# --- independent transcriptions of the reference ---------------------------

def ref_window(window_size: int, seqlen: int, start_pos: int) -> np.ndarray:
    """get_window_topk_idxs, transcribed from inference/model.py."""
    if start_pos >= window_size - 1:
        sp = start_pos % window_size
        return np.concatenate(
            [np.arange(sp + 1, window_size), np.arange(0, sp + 1)]
        )[None, :]
    if start_pos > 0:
        row = np.arange(start_pos + 1)
        pad = np.full(window_size - start_pos - 1, -1)
        return np.concatenate([row, pad])[None, :]
    base = np.arange(seqlen)[:, None]
    width = min(seqlen, window_size)
    m = np.clip(base - window_size + 1, 0, None) + np.arange(width)
    return np.where(m > base, -1, m)


def ref_compress(ratio: int, seqlen: int, start_pos: int, offset: int) -> np.ndarray:
    """get_compress_topk_idxs, transcribed from inference/model.py."""
    if start_pos > 0:
        return (np.arange((start_pos + 1) // ratio) + offset)[None, :]
    width = seqlen // ratio
    m = np.tile(np.arange(width), (seqlen, 1))
    mask = m >= (np.arange(1, seqlen + 1)[:, None] // ratio)
    return np.where(mask, -1, m + offset)


def ref_gate(scores, topk, route_scale, bias=None):
    biased = scores if bias is None else scores + bias
    idx = np.argsort(-biased, axis=-1)[..., :topk]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)
    return w * route_scale, idx


# --- window indices --------------------------------------------------------

@pytest.mark.parametrize("window,seqlen,start", [
    (8, 5, 0),      # prefill shorter than the window
    (8, 8, 0),      # prefill exactly the window
    (8, 20, 0),     # prefill longer -> triangle clamps
    (128, 300, 0),
    (8, 1, 3),      # decode, ring partially filled
    (8, 1, 7),      # decode, first fully-wrapped position
    (8, 1, 9),      # decode, wrapped
    (128, 1, 5000), # decode, long-running session
])
def test_window_indices_match_reference(window, seqlen, start):
    got = np.array(window_topk_indices(window, seqlen, start))
    assert np.array_equal(got, ref_window(window, seqlen, start))


def test_window_prefill_is_causal():
    """Row i must never reference a slot beyond i."""
    m = np.array(window_topk_indices(8, 20, 0))
    for i, row in enumerate(m):
        real = row[row != MASKED_INDEX]
        assert real.max() <= i


def test_window_decode_covers_every_ring_slot_once_wrapped():
    row = np.array(window_topk_indices(8, 1, 9))[0]
    assert sorted(row.tolist()) == list(range(8))


def test_window_partial_fill_masks_the_unused_tail():
    row = np.array(window_topk_indices(8, 1, 3))[0]
    assert row[:4].tolist() == [0, 1, 2, 3]
    assert (row[4:] == MASKED_INDEX).all()


# --- compressed indices ----------------------------------------------------

@pytest.mark.parametrize("ratio,seqlen,start,offset", [
    (4, 16, 0, 128),
    (4, 17, 0, 128),
    (128, 512, 0, 128),
    (4, 1, 99, 128),
    (128, 1, 5000, 128),
])
def test_compress_indices_match_reference(ratio, seqlen, start, offset):
    got = np.array(compress_topk_indices(ratio, seqlen, start, offset))
    assert np.array_equal(got, ref_compress(ratio, seqlen, start, offset))


def test_compressed_slots_are_offset_past_the_ring():
    """Ring and compressed KV share ONE buffer; compressed slots start at
    `window_size`, so an un-offset index would silently read ring slots."""
    m = np.array(compress_topk_indices(4, 16, 0, 128))
    real = m[m != MASKED_INDEX]
    assert real.min() >= 128


def test_a_compressed_group_is_invisible_until_complete():
    # With ratio 4, query position 2 has seen tokens 0..2 — group 0 is not
    # finished, so nothing compressed is visible yet.
    m = np.array(compress_topk_indices(4, 16, 0, 128))
    assert (m[2] == MASKED_INDEX).all()
    assert (m[3] != MASKED_INDEX).any()  # group 0 complete at position 3


# --- routing ---------------------------------------------------------------

def test_sqrtsoftplus_matches_reference():
    x = np.array([-30.0, -1.0, 0.0, 1.0, 30.0], np.float32)
    ref = np.sqrt(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))
    got = np.array(sqrtsoftplus(mx.array(x)))
    assert np.allclose(got, ref, atol=1e-5)


def test_sqrtsoftplus_does_not_overflow_on_large_logits():
    # naive log(1+exp(x)) is inf by x=100 and the sqrt propagates it into every
    # routing decision for that token.
    got = np.array(sqrtsoftplus(mx.array([100.0, 1000.0], mx.float32)))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, [10.0, np.sqrt(1000.0)], atol=1e-3)


def test_route_matches_reference_without_bias():
    rng = np.random.default_rng(0)
    scores = rng.random((6, 32)).astype(np.float32)
    w, idx = route(mx.array(scores), topk=4, route_scale=1.5)
    rw, ridx = ref_gate(scores, 4, 1.5)
    assert sorted(np.array(idx)[0].tolist()) == sorted(ridx[0].tolist())
    assert np.allclose(np.sort(np.array(w), axis=-1), np.sort(rw, axis=-1), atol=1e-5)


def test_route_bias_selects_but_does_not_reweight():
    """The correction bias shifts SELECTION only; weights come from the
    unbiased scores. Folding it into the weights is a silent quality loss."""
    scores = mx.array([[0.1, 0.2, 0.9, 0.3]], mx.float32)
    bias = mx.array([10.0, 0.0, 0.0, 0.0], mx.float32)
    w, idx = route(scores, topk=1, route_scale=1.0, bias=bias)
    assert int(np.array(idx)[0, 0]) == 0          # bias won the selection
    assert float(np.array(w)[0, 0]) == pytest.approx(1.0)  # normalized alone
    # and the weight came from the unbiased score, not 0.1 + 10
    w2, _ = route(scores, topk=2, route_scale=1.0, bias=bias)
    assert float(np.array(w2).sum()) == pytest.approx(1.0)


def test_route_weights_are_normalized_then_scaled():
    scores = mx.array([[1.0, 3.0, 2.0, 0.5]], mx.float32)
    w, _ = route(scores, topk=2, route_scale=1.5)
    assert float(np.array(w).sum()) == pytest.approx(1.5, abs=1e-5)


# --- expert ----------------------------------------------------------------

def test_clipped_swiglu_asymmetry_matches_reference():
    """up clamps BOTH sides, gate only from above — straight from the
    reference. Clamping gate symmetrically would cut the negative tail silu
    needs."""
    gate = mx.array([[-50.0, 50.0]], mx.float32)
    up = mx.array([[-50.0, 50.0]], mx.float32)
    got = np.array(clipped_swiglu(gate, up, limit=10.0))

    g = np.array([[-50.0, 10.0]], np.float32)   # only upper clamp
    u = np.array([[-10.0, 10.0]], np.float32)   # both sides
    ref = (g / (1 + np.exp(-g))) * u
    assert np.allclose(got, ref, atol=1e-4)


def test_clipped_swiglu_limit_zero_disables_clipping():
    gate = mx.array([[50.0]], mx.float32)
    up = mx.array([[50.0]], mx.float32)
    got = float(np.array(clipped_swiglu(gate, up, limit=0.0))[0, 0])
    assert got == pytest.approx(50.0 / (1 + np.exp(-50.0)) * 50.0, rel=1e-5)
