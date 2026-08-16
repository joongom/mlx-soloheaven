"""DeepSeek-V4 modules (step 2b of the port spec).

Two kinds of coverage, both chosen for failure modes that do not raise:

* Kernel transcriptions — sparse attention with the sink term and the
  Sinkhorn split are checked against independent NumPy transcriptions of
  DeepSeek's ``inference/kernel.py``, not against themselves.
* Decode consistency — prefill(N) must equal prefill(k) + (N-k) decode steps,
  through the full model. This is the same invariant DeepSeek exposes as
  ``--decode-consistency``; it is the test that catches ring/compressor state
  machine bugs, which produce fluent-but-wrong output rather than errors.
"""

from __future__ import annotations

import json
import os
import struct

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_soloheaven.models.deepseek_v4 import (
    Compressor,
    CompressorState,
    Model,
    ModelArgs,
    apply_interleaved_rope,
    hc_split_sinkhorn,
    rope_cos_sin,
    sparse_attend,
    yarn_freqs,
)

# ---------------------------------------------------------------------------
# a tiny config exercising every layer kind: dense, ratio-4 (+Indexer, overlap
# compression), and a non-overlap compressed layer; layer 0 routes by hash.

TINY = ModelArgs.from_dict(
    {
        "model_type": "deepseek_v4",
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "head_dim": 8,
        "qk_rope_head_dim": 4,
        "q_lora_rank": 8,
        "o_lora_rank": 4,
        "o_groups": 2,
        "moe_intermediate_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 1.5,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "index_topk": 3,
        "sliding_window": 8,
        "num_hash_layers": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 5,
        "swiglu_limit": 10.0,
        "compress_rope_theta": 160000,
        "rope_theta": 10000,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "compress_ratios": [0, 4, 8],
    }
)


def randomize(module, seed=0):
    """Fill every parameter with small random values (ints -> valid ids)."""
    keys = [mx.random.key(seed + i) for i in range(2)]

    def fill(a):
        if mx.issubdtype(a.dtype, mx.integer):
            return mx.random.randint(0, 4, a.shape, dtype=a.dtype, key=keys[0])
        return mx.random.normal(a.shape, dtype=a.dtype, key=keys[1]) * 0.2

    module.update(tree_map(fill, module.parameters()))


# --- kernel transcriptions -------------------------------------------------


def ref_sparse_attn(q, kv, sink, idxs, scale):
    """Per-row plain-softmax transcription of sparse_attn_kernel: masked idxs
    contribute nothing; the sink joins the denominator as exp(sink - max)."""
    s_len, h, d = q.shape
    out = np.zeros_like(q)
    for si in range(s_len):
        rows = idxs[si]
        g = np.stack([kv[r] if r >= 0 else np.zeros(d, q.dtype) for r in rows])
        sc = (q[si] @ g.T) * scale
        sc[:, rows < 0] = -np.inf
        for hi in range(h):
            m = max(sc[hi].max(), sink[hi])
            p = np.exp(sc[hi] - m)
            out[si, hi] = (p @ g) / (p.sum() + np.exp(sink[hi] - m))
    return out


def test_sparse_attend_matches_reference_transcription():
    rng = np.random.default_rng(0)
    s_len, h, d, n = 5, 3, 8, 11
    q = rng.standard_normal((s_len, h, d)).astype(np.float32)
    kv = rng.standard_normal((n, d)).astype(np.float32)
    sink = rng.standard_normal(h).astype(np.float32)
    idxs = rng.integers(0, n, (s_len, 6)).astype(np.int32)
    idxs[0, 3:] = -1  # partially masked row
    got = sparse_attend(
        mx.array(q)[None], [(mx.array(kv)[None], mx.array(idxs)[None])],
        mx.array(sink), 0.35,
    )
    assert np.allclose(np.array(got)[0], ref_sparse_attn(q, kv, sink, idxs, 0.35), atol=1e-5)


def test_sparse_attend_concatenates_parts_like_one_buffer():
    """Two parts must equal a single part over the concatenated buffer with
    shifted indices — that is exactly how the decode path uses it."""
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((1, 4, 2, 8)).astype(np.float32))
    a = mx.array(rng.standard_normal((1, 6, 8)).astype(np.float32))
    b = mx.array(rng.standard_normal((1, 5, 8)).astype(np.float32))
    ia = mx.array(rng.integers(0, 6, (1, 4, 3)).astype(np.int32))
    ib = mx.array(rng.integers(0, 5, (1, 4, 2)).astype(np.int32))
    sink = mx.array(rng.standard_normal(2).astype(np.float32))
    two = sparse_attend(q, [(a, ia), (b, ib)], sink, 0.5)
    one = sparse_attend(
        q, [(mx.concatenate([a, b], axis=1), mx.concatenate([ia, ib + 6], axis=-1))],
        sink, 0.5,
    )
    assert np.allclose(np.array(two), np.array(one), atol=1e-6)


def ref_sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = 1 / (1 + np.exp(-(mixes[..., :hc] * scale[0] + base[:hc]))) + eps
    post = 2 / (1 + np.exp(-(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])))
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(
        *mixes.shape[:-1], hc, hc
    )
    e = np.exp(comb - comb.max(-1, keepdims=True))
    comb = e / e.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


def test_sinkhorn_split_matches_reference_transcription():
    rng = np.random.default_rng(2)
    hc = 4
    mixes = rng.standard_normal((3, (2 + hc) * hc)).astype(np.float32)
    scale = rng.standard_normal(3).astype(np.float32)
    base = rng.standard_normal((2 + hc) * hc).astype(np.float32)
    pre, post, comb = hc_split_sinkhorn(
        mx.array(mixes), mx.array(scale), mx.array(base), hc, 20, 1e-6
    )
    rp, rq, rc = ref_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    assert np.allclose(np.array(pre), rp, atol=1e-6)
    assert np.allclose(np.array(post), rq, atol=1e-6)
    assert np.allclose(np.array(comb), rc, atol=1e-5)
    # after 20 iterations comb is near doubly stochastic
    assert np.allclose(np.array(comb).sum(-1), 1.0, atol=1e-2)
    assert np.allclose(np.array(comb).sum(-2), 1.0, atol=1e-2)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
@pytest.mark.parametrize("two_parts", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_metal_decode_kernel_matches_eager(two_parts, masked):
    """The fused decode kernel must reproduce the eager path (which is itself
    NumPy-verified) — including masked slots, the sink, and the dense-layer
    single-part case."""
    from mlx_soloheaven.models.deepseek_v4 import (
        _sparse_attend_decode_metal,
        _sparse_attend_eager,
    )

    rng = np.random.default_rng(4)
    h, d = 4, 32
    q = mx.array(rng.standard_normal((1, 1, h, d)).astype(np.float32))
    sink = mx.array(rng.standard_normal(h).astype(np.float32))
    kv1 = mx.array(rng.standard_normal((1, 9, d)).astype(np.float32))
    i1 = rng.integers(0, 9, (1, 1, 7)).astype(np.int32)
    if masked:
        i1[0, 0, 2:4] = -1
    parts = [(kv1, mx.array(i1))]
    if two_parts:
        kv2 = mx.array(rng.standard_normal((1, 5, d)).astype(np.float32))
        i2 = rng.integers(0, 5, (1, 1, 3)).astype(np.int32)
        if masked:
            i2[0, 0, 0] = -1
        parts.append((kv2, mx.array(i2)))

    got = np.array(_sparse_attend_decode_metal(q, parts, sink, 0.4))
    ref = np.array(_sparse_attend_eager(q, parts, sink, 0.4))
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_moe_kernel_matches_dequantized_reference():
    """The fused batch-1 MoE kernels must reproduce the dequantized eager
    computation on a REAL 2-bit/gs64 quantized SwitchGLU — including the
    clipped-SwiGLU asymmetry, routing weights, and a masked (-1) slot."""
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from mlx_soloheaven.models.deepseek_v4 import (
        ClippedSwiGLU,
        _moe_kernel_usable,
        _moe_routed_kernel,
        clipped_swiglu,
    )

    d, inner, n_exp, limit = 128, 192, 8, 10.0
    glu = SwitchGLU(d, inner, n_exp, activation=ClippedSwiGLU(limit), bias=False)
    nn.quantize(glu, group_size=64, bits=2)
    assert _moe_kernel_usable(glu)

    rng = np.random.default_rng(6)
    x = mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)).astype(mx.bfloat16)
    idxs = mx.array([[[3, 5, 0, -1]]], dtype=mx.int32)  # incl. masked slot
    wts = mx.array([[[0.5, 0.3, 0.2, 0.4]]], dtype=mx.float32)

    got = np.array(_moe_routed_kernel(glu, x, idxs, wts, limit))

    def deq(m):
        return mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=2)

    xf = mx.array(np.array(x.astype(mx.float32)))
    ref = mx.zeros((d,))
    for s in range(4):
        e = int(np.array(idxs)[0, 0, s])
        if e < 0:
            continue
        g = (xf.reshape(-1) @ deq(glu.gate_proj)[e].T).astype(mx.float32)
        u = (xf.reshape(-1) @ deq(glu.up_proj)[e].T).astype(mx.float32)
        h = clipped_swiglu(g, u, limit)
        ref = ref + float(np.array(wts)[0, 0, s]) * (h @ deq(glu.down_proj)[e].T)
    assert np.allclose(got.ravel(), np.array(ref), atol=2e-2, rtol=2e-2), (
        np.abs(got.ravel() - np.array(ref)).max()
    )


# --- rope ------------------------------------------------------------------


def test_rope_matches_complex_reference_and_inverse_cancels():
    rng = np.random.default_rng(3)
    d, s_len = 8, 5
    freqs = yarn_freqs(d, 10000.0, 0, 1.0, 32.0, 1.0)
    cos, sin = rope_cos_sin(freqs, mx.arange(2, 2 + s_len))
    x = rng.standard_normal((s_len, d)).astype(np.float32)
    got = np.array(apply_interleaved_rope(mx.array(x), cos, sin))
    # reference convention: interleaved pairs as complex, times e^{i*pos*freq}
    z = x.reshape(s_len, d // 2, 2)[..., 0] + 1j * x.reshape(s_len, d // 2, 2)[..., 1]
    ang = np.arange(2, 2 + s_len)[:, None] * np.array(freqs)[None]
    ref = z * np.exp(1j * ang)
    ref = np.stack([ref.real, ref.imag], -1).reshape(s_len, d).astype(np.float32)
    assert np.allclose(got, ref, atol=1e-5)
    back = np.array(apply_interleaved_rope(mx.array(got), cos, sin, inverse=True))
    assert np.allclose(back, x, atol=1e-5)


def test_yarn_freqs_match_reference_transcription():
    dim, base, orig, factor, bf, bs = 64, 160000.0, 65536, 16.0, 32.0, 1.0
    import math

    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))

    def cdim(r):
        return dim * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(cdim(bf)), 0), min(math.ceil(cdim(bs)), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    ref = freqs / factor * (1 - smooth) + freqs * smooth
    got = np.array(yarn_freqs(dim, base, orig, factor, bf, bs))
    assert np.allclose(got, ref, rtol=1e-6)
    # dense layers pass original_seq_len=0 -> plain rope
    got_plain = np.array(yarn_freqs(dim, 10000.0, 0, factor, bf, bs))
    assert np.allclose(got_plain, 1.0 / (10000.0 ** (np.arange(0, dim, 2) / dim)), rtol=1e-6)


# --- compressor state machine ----------------------------------------------


@pytest.mark.parametrize("ratio", [4, 8])  # 4 = overlap path, 8 = plain
@pytest.mark.parametrize("split", [2, 13, 16])  # < ratio, mid-group, aligned
def test_compressor_prefill_equals_step_decode(ratio, split):
    total = 21
    comp = Compressor(dim=12, head_dim=8, ratio=ratio, rope_dim=4, eps=1e-6)
    randomize(comp, seed=ratio)
    x = mx.random.normal((1, total, 12), key=mx.random.key(9))

    full = CompressorState()
    comp(x, full, comp_freqs := yarn_freqs(4, 100.0, 0, 1.0, 32.0, 1.0), 0)

    part = CompressorState()
    comp(x[:, :split], part, comp_freqs, 0)
    for p in range(split, total):
        comp(x[:, p : p + 1], part, comp_freqs, p)

    assert full.n == total // ratio and part.n == full.n
    if full.n:
        assert np.allclose(
            np.array(part.valid()), np.array(full.valid()), atol=1e-5
        ), f"compressed caches diverge (ratio={ratio}, split={split})"


@pytest.mark.parametrize("ratio", [4, 8])
@pytest.mark.parametrize(
    "plan",
    [
        [2, 7, 5, 9],   # chunks landing mid-group everywhere
        [13, 8, 2],     # chunk completing a partial group then bulk
        [3, 1, 4, 15],  # single-token steps interleaved with chunks
        [8, 8, 7],      # aligned chunks (ratio 4 and 8 boundaries)
    ],
)
def test_compressor_chunked_equals_full_prefill(ratio, plan):
    """Continuation chunks (start > 0, seqlen > 1) — the regime the engine's
    prefill-step chunking produces and the reference never exercises — must
    reproduce the one-shot prefill exactly."""
    total = sum(plan)
    comp = Compressor(dim=12, head_dim=8, ratio=ratio, rope_dim=4, eps=1e-6)
    randomize(comp, seed=100 + ratio)
    x = mx.random.normal((1, total, 12), key=mx.random.key(10))
    freqs = yarn_freqs(4, 100.0, 0, 1.0, 32.0, 1.0)

    full = CompressorState()
    comp(x, full, freqs, 0)

    part = CompressorState()
    pos = 0
    for n in plan:
        comp(x[:, pos : pos + n], part, freqs, pos)
        pos += n

    assert part.n == full.n == total // ratio
    if full.n:
        assert np.allclose(np.array(part.valid()), np.array(full.valid()), atol=1e-5)


# --- full model ------------------------------------------------------------


def build_tiny():
    model = Model(TINY)
    randomize(model, seed=7)
    # tid2eid must hold valid expert ids
    model.layers[0].ffn.gate.tid2eid = mx.random.randint(
        0, TINY.n_routed_experts, (TINY.vocab_size, TINY.num_experts_per_tok),
        dtype=mx.int32, key=mx.random.key(11),
    )
    return model


TOKENS = [3, 17, 42, 9, 60, 1, 33, 25, 8, 50, 12, 44, 2, 19, 63, 7, 30, 11, 5, 58, 21, 40, 14]


@pytest.mark.parametrize("split", [2, 11])  # before/after windows+groups fill
def test_decode_consistency_full_model(split):
    """prefill(23) == prefill(split) + step decode — across the ring wrap
    (window 8), overlap groups (ratio 4), plain groups (ratio 8), the Indexer,
    and hash + score routing."""
    model = build_tiny()
    ids = mx.array([TOKENS])

    ref = model(ids)  # [1, T, V]

    cache = model.make_cache()
    out = [model(ids[:, :split], cache)]
    for p in range(split, len(TOKENS)):
        out.append(model(ids[:, p : p + 1], cache))
    got = mx.concatenate(out, axis=1)

    assert np.allclose(np.array(got), np.array(ref), atol=2e-4), (
        f"max abs diff {np.abs(np.array(got) - np.array(ref)).max()}"
    )


@pytest.mark.parametrize(
    "plan",
    [
        [2, 5, 9, 7],    # chunks crossing the window-8 wrap mid-chunk
        [11, 3, 9],      # first chunk longer than the window
        [6, 1, 1, 15],   # decode steps then a long continuation chunk
    ],
)
def test_chunked_prefill_consistency_full_model(plan):
    """prefill in engine-style chunks == one-shot prefill, through every layer
    kind. This is the exact call pattern mlx-lm's generate_step produces with
    --prefill-step-size, so it is the invariant serving depends on."""
    model = build_tiny()
    ids = mx.array([TOKENS])
    assert sum(plan) == len(TOKENS)

    ref = model(ids)

    cache = model.make_cache()
    out, pos = [], 0
    for n in plan:
        out.append(model(ids[:, pos : pos + n], cache))
        pos += n
    got = mx.concatenate(out, axis=1)

    assert np.allclose(np.array(got), np.array(ref), atol=2e-4), (
        f"max abs diff {np.abs(np.array(got) - np.array(ref)).max()}"
    )
    # The compiled path silently falls back to eager on a trace error — a
    # green consistency run must not hide that (it did once: a None cidx on
    # plain layers only broke at the first ratio-128 layer).
    import mlx_soloheaven.models.deepseek_v4 as v4

    assert not v4._COMPILED_DECODE_BROKEN, "compiled decode fell back to eager"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_decode_consistency_bf16():
    """The serving dtype. fp32-only tests missed a Metal bfloat implicit-
    conversion compile error (bfloat rejects float assignment); this runs the
    same consistency check at bf16 and refuses the silent eager fallback."""
    from mlx.utils import tree_map

    import mlx_soloheaven.models.deepseek_v4 as v4

    model = build_tiny()
    model.update(
        tree_map(
            lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a,
            model.parameters(),
        )
    )
    for layer in model.layers:  # fp32-by-design params stay fp32
        layer.attn.attn_sink = layer.attn.attn_sink.astype(mx.float32)
    ids = mx.array([TOKENS])
    ref = model(ids)
    cache = model.make_cache()
    out = [model(ids[:, :11], cache)]
    for p in range(11, len(TOKENS)):
        out.append(model(ids[:, p : p + 1], cache))
    got = mx.concatenate(out, axis=1)
    assert not v4._COMPILED_DECODE_BROKEN, "compiled decode fell back to eager"
    a = np.array(got.astype(mx.float32))
    b = np.array(ref.astype(mx.float32))
    assert np.allclose(a, b, atol=5e-2), np.abs(a - b).max()  # bf16 tolerance


def test_cache_state_round_trips_mid_session():
    """Snapshot cache state after some decode steps, restore into fresh caches,
    continue — logits must match an uninterrupted run exactly. This is the
    property the disk cache's state get/set relies on."""
    model = build_tiny()
    ids = mx.array([TOKENS])

    cache = model.make_cache()
    model(ids[:, :14], cache)
    states = [(c.state, c.meta_state) for c in cache]
    a = [model(ids[:, p : p + 1], cache) for p in range(14, 20)]

    fresh = model.make_cache()
    for c, (st, meta) in zip(fresh, states):
        c.state, c.meta_state = st, meta
    b = [model(ids[:, p : p + 1], fresh) for p in range(14, 20)]

    for x, y in zip(a, b):
        assert np.allclose(np.array(x), np.array(y), atol=1e-6)


def test_dense_layers_have_no_compressor_state():
    model = build_tiny()
    cache = model.make_cache()
    assert cache[0].comp is None and cache[0].idx is None  # dense
    assert cache[1].comp is not None and cache[1].idx is not None  # ratio 4
    assert cache[2].comp is not None and cache[2].idx is None  # ratio 8


# --- weight-shape parity against the real checkpoint -----------------------

CKPT = os.path.expanduser("~/.lmstudio/models/deepseek-ai/DeepSeek-V4-Flash-0731")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")),
    reason="0731 checkpoint not present",
)
def test_parameter_tree_matches_real_checkpoint():
    """Build the real-size model (lazy arrays — shapes only, nothing is
    evaluated) and require an exact 1:1 mapping to the checkpoint tensors after
    the documented transforms: fp8 ``.scale`` sidecars dropped, ``mtp.*``
    deferred, per-expert ``w1/w3/w2`` stacked into SwitchGLU, ``wo_a`` grouped,
    I8-packed fp4 unpacking to double the last dim."""
    with open(os.path.join(CKPT, "config.json")) as f:
        args = ModelArgs.from_dict(json.load(f))
    model = Model(args)
    ours = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}

    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    dtypes: dict[str, str] = {}
    for shard in {v for v in index["weight_map"].values()}:
        with open(os.path.join(CKPT, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for name, meta in json.loads(f.read(n)).items():
                if name != "__metadata__":
                    dtypes[name] = (meta["dtype"], meta["shape"])

    expected: dict[str, tuple] = {}
    stacked: dict[str, dict[int, tuple]] = {}
    for name, (dtype, shape) in dtypes.items():
        if name.startswith("mtp.") or name.endswith(".scale"):
            continue
        shape = tuple(shape)
        if dtype == "I8":  # fp4 packed pairwise along the last dim
            shape = (*shape[:-1], shape[-1] * 2)
        if ".ffn.experts." in name:
            prefix, rest = name.split(".experts.")
            eid, wname, _ = rest.split(".")
            proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[wname]
            stacked.setdefault(f"{prefix}.experts.{proj}.weight", {})[int(eid)] = shape
            continue
        if name.endswith("attn.wo_a.weight"):
            out, dim = shape
            g = args.o_groups
            shape = (g, out // g, dim)
        expected[name] = shape
    for name, per_expert in stacked.items():
        assert len(per_expert) == args.n_routed_experts
        assert len(set(per_expert.values())) == 1
        expected[name] = (args.n_routed_experts, *per_expert[0])

    missing = sorted(set(expected) - set(ours))
    extra = sorted(set(ours) - set(expected))
    assert not missing, f"model lacks checkpoint tensors: {missing[:8]}"
    assert not extra, f"model has tensors absent from checkpoint: {extra[:8]}"
    bad = {k: (ours[k], expected[k]) for k in expected if ours[k] != expected[k]}
    assert not bad, f"shape mismatches: {dict(list(bad.items())[:8])}"


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_moe_kernel_matches_dequantized_reference_at_any_supported_width(bits):
    """The fused MoE kernels derive their packing from the weights, so the
    gate must admit every width they can read — not just the 2-bit one the
    DeepSeek-V4 converter happens to emit. This is what would let another
    model's 8-bit experts use them."""
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from mlx_soloheaven.models.deepseek_v4 import (
        ClippedSwiGLU,
        _moe_kernel_usable,
        _moe_routed_kernel,
        clipped_swiglu,
    )

    d, inner, n_exp, limit = 128, 192, 8, 10.0
    glu = SwitchGLU(d, inner, n_exp, activation=ClippedSwiGLU(limit), bias=False)
    nn.quantize(glu, group_size=64, bits=bits)
    assert _moe_kernel_usable(glu), f"{bits}-bit experts rejected by the gate"

    rng = np.random.default_rng(6)
    x = mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)).astype(mx.bfloat16)
    idxs = mx.array([[[3, 5, 0, -1]]], dtype=mx.int32)
    wts = mx.array([[[0.5, 0.3, 0.2, 0.4]]], dtype=mx.float32)

    got = np.array(_moe_routed_kernel(glu, x, idxs, wts, limit))
    ref = mx.zeros((d,), dtype=mx.float32)
    for slot in range(idxs.shape[-1]):
        e = int(idxs[0, 0, slot])
        if e < 0:
            continue
        g = mx.dequantize(glu.gate_proj.weight[e], glu.gate_proj.scales[e],
                          glu.gate_proj.biases[e], group_size=64, bits=bits)
        u = mx.dequantize(glu.up_proj.weight[e], glu.up_proj.scales[e],
                          glu.up_proj.biases[e], group_size=64, bits=bits)
        dn = mx.dequantize(glu.down_proj.weight[e], glu.down_proj.scales[e],
                           glu.down_proj.biases[e], group_size=64, bits=bits)
        xf = x.reshape(-1).astype(mx.float32)
        h = clipped_swiglu((g @ xf).astype(mx.float32),
                           (u @ xf).astype(mx.float32), limit)
        ref = ref + float(wts[0, 0, slot]) * (dn @ h.astype(mx.float32))
    mx.eval(ref)
    assert np.abs(got.reshape(-1) - np.array(ref)).max() < 2e-2


def test_moe_kernel_gate_refuses_non_affine_quantization():
    """`_moe_kernel_usable` widened to every bit width the kernels can read;
    it must NOT widen to other MODES. mxfp4 stores a shared exponent instead
    of scale/bias, so the affine unpack would return plausible garbage with no
    error — and Qwen3.5-122B-A10B ships mxfp4."""
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from mlx_soloheaven.models.deepseek_v4 import ClippedSwiGLU, _moe_kernel_usable

    glu = SwitchGLU(128, 192, 8, activation=ClippedSwiGLU(10.0), bias=False)
    nn.quantize(glu, group_size=64, bits=8)
    assert _moe_kernel_usable(glu)

    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        proj.mode = "mxfp4"
    assert not _moe_kernel_usable(glu)


def test_a_retired_native_flag_name_warns_instead_of_being_silent(monkeypatch, caplog):
    """A stale export is otherwise invisible: the server starts, serves
    correct output, and is simply ~2x slower. That cost a whole session of
    'why is this 15 tok/s'."""
    import logging

    import mlx_soloheaven.models.deepseek_v4 as v4

    monkeypatch.setattr(v4, "_WARNED_STALE_FLAG", False)
    monkeypatch.delenv(v4._NATIVE_FLAG, raising=False)
    monkeypatch.setenv("SOLOHEAVEN_DSV4_NATIVE", "1")

    with caplog.at_level(logging.WARNING):
        assert v4._NATIVE_DECODE_ENABLED() is False
    assert "RENAMED" in caplog.text and v4._NATIVE_FLAG in caplog.text

    # warns once, not per decoded token
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        v4._NATIVE_DECODE_ENABLED()
    assert caplog.text == ""


def test_the_current_flag_name_enables_without_warning(monkeypatch, caplog):
    import logging

    import mlx_soloheaven.models.deepseek_v4 as v4

    monkeypatch.setattr(v4, "_WARNED_STALE_FLAG", False)
    monkeypatch.setattr(v4, "_NATIVE_DECODE_BROKEN", False)
    monkeypatch.setenv("SOLOHEAVEN_DSV4_NATIVE", "1")     # stale one also set
    monkeypatch.setenv(v4._NATIVE_FLAG, "1")
    with caplog.at_level(logging.WARNING):
        assert v4._NATIVE_DECODE_ENABLED() is True
    assert caplog.text == ""
