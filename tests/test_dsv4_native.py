"""Stage 3b runtime — the C encode loop and DLPack buffer table.

These verify the plumbing the native decode loop is built on: that a plan
encoded and committed through libdsv4enc reproduces mx.quantized_matmul
bit-for-bit, that per-token uniform writes take effect without re-encoding,
and that the C encode cost is in the budget (ladder step 1 wanted ~1 ms for
~1500 dispatches; python-ctypes was 16.6 ms — this must beat that decisively).

Skipped off Metal / when the dylib can't build.
"""

from __future__ import annotations

import ctypes
import struct
import time

import mlx.core as mx
import numpy as np
import pytest

metal = pytest.importorskip("mlx.core").metal
if not metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

try:
    from mlx_soloheaven.native import runtime as rt_mod
    _RT = rt_mod.Runtime()
except Exception as e:  # noqa: BLE001 — dylib/build/device issues -> skip
    pytest.skip(f"native runtime unavailable: {e}", allow_module_level=True)


def _q8(N, K):
    w = mx.random.normal((N, K)).astype(mx.bfloat16)  # bf16 scales (T-typed kernel)
    qw, sc, bi = mx.quantize(w, group_size=64, bits=8)
    return qw, sc, bi


def test_qmv_through_c_loop_matches_mlx():
    N, K = 512, 4096
    qw, sc, bi = _q8(N, K)
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    y = mx.zeros((1, N), dtype=mx.bfloat16)
    ref = mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=64, bits=8)
    mx.eval(qw, sc, bi, x, y, ref)
    mx.synchronize()

    keep = []
    bufs = []
    for a in (qw, sc, bi, x, y):
        ptr, off, cap = rt_mod.mtl_buffer(a)
        assert off == 0
        keep.append(cap)
        bufs.append(ptr)

    const = struct.pack("<ii", K, N)
    item = rt_mod.plan_qmv(_RT, (0, 1, 2, 3, 4), K, N, 0, 4)
    _RT.commit([item], bufs, const, wait=True)

    got = np.array(y.astype(mx.float32))
    exp = np.array(ref.astype(mx.float32))
    assert np.abs(got - exp).max() == 0.0, np.abs(got - exp).max()


def test_uniform_write_takes_effect_without_reencode():
    """The per-token contract: rewrite a uniform buffer's contents in place
    (unified memory), re-commit the SAME plan, get the new result — no
    re-encode, no new MLX op. Here the 'uniform' is x; two different x values
    must yield their two correct products through one plan object."""
    N, K = 256, 512
    qw, sc, bi = _q8(N, K)
    x = mx.zeros((1, K), dtype=mx.bfloat16)
    y = mx.zeros((1, N), dtype=mx.bfloat16)
    mx.eval(qw, sc, bi, x, y)
    mx.synchronize()

    keep, bufs = [], []
    for a in (qw, sc, bi, x, y):
        ptr, off, cap = rt_mod.mtl_buffer(a)
        keep.append(cap)
        bufs.append(ptr)
    x_contents = _RT.buffer_contents(bufs[3])

    const = struct.pack("<ii", K, N)
    item = rt_mod.plan_qmv(_RT, (0, 1, 2, 3, 4), K, N, 0, 4)

    rng = np.random.default_rng(0)
    for _ in range(3):
        xv = rng.standard_normal((K,)).astype(np.float32)
        # write bf16 x directly into the shared buffer
        xb = np.array(mx.array(xv).astype(mx.bfloat16).view(mx.uint16))
        ctypes.memmove(x_contents, xb.ctypes.data, xb.nbytes)
        _RT.commit([item], bufs, const, wait=True)
        ref = mx.quantized_matmul(
            mx.array(xv).astype(mx.bfloat16)[None], qw, sc, bi,
            transpose=True, group_size=64, bits=8,
        )
        mx.synchronize()
        got = np.array(y.astype(mx.float32))
        exp = np.array(ref.astype(mx.float32))
        assert np.abs(got - exp).max() == 0.0


def test_buffer_table_dedupes_and_stays_valid():
    """A shared array gets ONE slot; distinct arrays get distinct slots; and a
    qmv encoded against a table-built buffer list still gives diff 0.0 (the
    table produces the same pointers the direct path does)."""
    N, K = 256, 512
    qw, sc, bi = _q8(N, K)
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    y = mx.zeros((1, N), dtype=mx.bfloat16)
    mx.eval(qw, sc, bi, x, y)
    mx.synchronize()

    table = rt_mod.BufferTable()
    w_slot, s_slot, b_slot = table.add(qw), table.add(sc), table.add(bi)
    x_slot, y_slot = table.add(x), table.add(y)
    assert table.add(qw) == w_slot  # idempotent by identity
    assert len({w_slot, s_slot, b_slot, x_slot, y_slot}) == 5

    ref = mx.quantized_matmul(x, qw, sc, bi, transpose=True, group_size=64, bits=8)
    mx.eval(ref)
    mx.synchronize()
    const = struct.pack("<ii", K, N)
    item = rt_mod.plan_qmv(_RT, (w_slot, s_slot, b_slot, x_slot, y_slot), K, N, 0, 4)
    _RT.commit([item], table.ptrs, const, wait=True)
    got = np.array(y.astype(mx.float32))
    exp = np.array(ref.astype(mx.float32))
    assert np.abs(got - exp).max() == 0.0


def test_native_moe_w2_kernel_matches_mx_fast():
    """Our custom kernel, compiled from native/kernels.metal and dispatched by
    the C loop, must match the model's mx.fast.metal_kernel version of the same
    body (which is itself differential-tested against dequantized math). This
    is the last unproven plumbing: OUR kernels via explicit-signature source.
    """
    from mlx_soloheaven.models.deepseek_v4 import _get_moe_kernels

    n_act, d_model, d_inner, E = 4, 256, 192, 8
    dw_w = mx.random.normal((E, d_model, d_inner)).astype(mx.bfloat16)
    dqw, dsc, dbi = mx.quantize(dw_w, group_size=64, bits=2)
    h = mx.random.normal((n_act, d_inner)).astype(mx.float32)
    idxs = mx.array([3, 5, 0, -1], dtype=mx.int32)
    wts = mx.array([0.5, 0.3, 0.2, 0.4], dtype=mx.float32)

    # reference: the model's mx.fast K2
    _, k2 = _get_moe_kernels()
    params = mx.array([n_act, d_model, d_inner], dtype=mx.int32)
    (ref,) = k2(
        inputs=[h.reshape(-1), dqw, dsc, dbi, idxs, wts, params],
        grid=(((d_model + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(d_model,)],
        output_dtypes=[mx.float32],
    )
    y = mx.zeros((d_model,), dtype=mx.float32)
    mx.eval(dqw, dsc, dbi, h, idxs, wts, ref, y)
    mx.synchronize()

    load = rt_mod.load_custom_kernels
    load(_RT)
    table = rt_mod.BufferTable()
    slots = [table.add(a) for a in (h.reshape(-1), dqw, dsc, dbi, idxs, wts, y)]

    it = rt_mod._PlanItem()
    it.pso = _RT.pipeline("dsv4_moe_w2", custom=True)
    it.n_bufs = 6
    for i, (bid, slot) in enumerate(
        [(slots[0], 0), (slots[1], 1), (slots[2], 2), (slots[3], 3),
         (slots[4], 4), (slots[5], 5)]
    ):
        it.buf_ids[i], it.buf_slots[i], it.buf_offs[i] = bid, slot, 0
    # params (int32 x3) as static bytes at slot 6; y at slot 7
    it.buf_ids[6], it.buf_slots[6], it.buf_offs[6] = slots[6], 7, 0
    it.n_bufs = 7
    it.n_bytes = 1
    it.bytes_off[0], it.bytes_len[0], it.bytes_slot[0] = 0, 12, 6
    it.grid[:] = [(d_model + 7) // 8, 1, 1]
    it.group[:] = [256, 1, 1]

    const = struct.pack("<iii", n_act, d_model, d_inner)
    _RT.commit([it], table.ptrs, const, wait=True)

    got = np.array(y)
    exp = np.array(ref)
    assert np.abs(got - exp).max() < 1e-3, np.abs(got - exp).max()


def test_native_moe_w13_kernel_matches_mx_fast():
    """dsv4_moe_w13 (gate/up + clipped SwiGLU) native == mx.fast twin."""
    from mlx_soloheaven.models.deepseek_v4 import _get_moe_kernels

    n_act, d_model, d_inner, E, limit = 4, 256, 192, 8, 10.0
    gw_w = mx.random.normal((E, d_inner, d_model)).astype(mx.bfloat16)
    gqw, gsc, gbi = mx.quantize(gw_w, group_size=64, bits=2)
    uw_w = mx.random.normal((E, d_inner, d_model)).astype(mx.bfloat16)
    uqw, usc, ubi = mx.quantize(uw_w, group_size=64, bits=2)
    x = mx.random.normal((d_model,)).astype(mx.bfloat16)
    idxs = mx.array([3, 5, 0, -1], dtype=mx.int32)

    k1, _ = _get_moe_kernels()
    params = mx.array([n_act, d_model, d_inner], dtype=mx.int32)
    feps = mx.array([limit], dtype=mx.float32)
    (ref,) = k1(
        inputs=[x, gqw, gsc, gbi, uqw, usc, ubi, idxs, params, feps],
        grid=((((n_act * d_inner) + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_act * d_inner,)],
        output_dtypes=[mx.float32],
    )
    h = mx.zeros((n_act * d_inner,), dtype=mx.float32)
    mx.eval(gqw, gsc, gbi, uqw, usc, ubi, x, idxs, ref, h)
    mx.synchronize()

    rt_mod.load_custom_kernels(_RT)
    table = rt_mod.BufferTable()
    ins = [x, gqw, gsc, gbi, uqw, usc, ubi, idxs]
    slots = [table.add(a) for a in ins]
    h_slot = table.add(h)

    it = rt_mod._PlanItem()
    it.pso = _RT.pipeline("dsv4_moe_w13", custom=True)
    it.n_bufs = len(ins) + 1
    for i, s in enumerate(slots):
        it.buf_ids[i], it.buf_slots[i], it.buf_offs[i] = s, i, 0
    it.buf_ids[len(ins)], it.buf_slots[len(ins)], it.buf_offs[len(ins)] = h_slot, 10, 0
    it.n_bytes = 2
    it.bytes_off[0], it.bytes_len[0], it.bytes_slot[0] = 0, 12, 8   # params (3 int)
    it.bytes_off[1], it.bytes_len[1], it.bytes_slot[1] = 12, 4, 9   # feps (1 float)
    it.grid[:] = [(((n_act * d_inner) + 7) // 8), 1, 1]
    it.group[:] = [256, 1, 1]

    const = struct.pack("<iiif", n_act, d_model, d_inner, limit)
    _RT.commit([it], table.ptrs, const, wait=True)
    assert np.abs(np.array(h) - np.array(ref)).max() < 1e-3


def test_native_attn_core_matches_mx_fast():
    """dsv4_attn_core native (explicit signature, C-loop dispatch) == the
    mx.fast twin, on a plain-compressed decode shape. This is the largest and
    most branch-heavy kernel; matching it validates the generated-signature
    path for the whole family."""
    from mlx_soloheaven.native.kernels import BUFFER_SLOTS

    H, D, RD, WIN = 4, 64, 16, 8
    KC = 6  # comp part width (plain path)
    rng = np.random.default_rng(9)
    q = mx.array(rng.standard_normal((1, 1, H, D)).astype(np.float32)).astype(mx.bfloat16)
    kv = mx.array(rng.standard_normal((1, 1, D)).astype(np.float32)).astype(mx.bfloat16)
    ring = mx.array(rng.standard_normal((1, WIN, D)).astype(np.float32)).astype(mx.bfloat16)
    comp = mx.array(rng.standard_normal((1, KC, D)).astype(np.float32)).astype(mx.bfloat16)
    cidx = mx.zeros((1, 1, 1), dtype=mx.int32)  # PLAIN -> unread dummy
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    freqs = mx.array((1.0 / (10000.0 ** (np.arange(0, RD, 2) / RD))).astype(np.float32))
    offset, ncomp = 20, 3
    params = mx.array([D, RD, WIN, KC, 1], dtype=mx.int32)     # plain=1
    fscal = mx.array([D ** -0.5, 1e-6], dtype=mx.float32)
    ioff = mx.array([offset, ncomp], dtype=mx.int32)

    from mlx_soloheaven.models.deepseek_v4 import _get_attn_core_kernel

    out_ref, kv_ref = _get_attn_core_kernel()(
        inputs=[q.reshape(-1), kv.reshape(-1), ring.reshape(-1), comp.reshape(-1),
                cidx.reshape(-1), sink, freqs, params, fscal, ioff],
        template=[("T", mx.bfloat16)],
        grid=(H * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(H * D,), (D,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    out = mx.zeros((H * D,), dtype=mx.bfloat16)
    kvo = mx.zeros((D,), dtype=mx.bfloat16)
    mx.eval(q, kv, ring, comp, cidx, sink, freqs, out_ref, kv_ref, out, kvo)
    mx.synchronize()

    rt_mod.load_custom_kernels(_RT)
    table = rt_mod.BufferTable()
    named = dict(
        q=q.reshape(-1), kv=kv.reshape(-1), ring=ring.reshape(-1),
        comp=comp.reshape(-1), cidx=cidx.reshape(-1), sink=sink, freqs=freqs,
        out=out, kv_out=kvo,
    )
    slots = {nm: table.add(a) for nm, a in named.items()}
    smap = BUFFER_SLOTS["dsv4_attn_core"]

    it = rt_mod._PlanItem()
    it.pso = _RT.pipeline("dsv4_attn_core", custom=True)
    buf_items = list(slots.items())
    it.n_bufs = len(buf_items)
    for i, (nm, s) in enumerate(buf_items):
        it.buf_ids[i], it.buf_slots[i], it.buf_offs[i] = s, smap[nm], 0
    # params(int x5) at slot smap['params'], fscal(float x2), ioff(int x2)
    it.n_bytes = 3
    it.bytes_off[0], it.bytes_len[0], it.bytes_slot[0] = 0, 20, smap["params"]
    it.bytes_off[1], it.bytes_len[1], it.bytes_slot[1] = 20, 8, smap["fscal"]
    it.bytes_off[2], it.bytes_len[2], it.bytes_slot[2] = 28, 8, smap["ioff"]
    it.grid[:] = [H, 1, 1]
    it.group[:] = [128, 1, 1]

    const = struct.pack("<5i2f2i", D, RD, WIN, KC, 1, D ** -0.5, 1e-6, offset, ncomp)
    _RT.commit([it], table.ptrs, const, wait=True)

    a = np.array(out.astype(mx.float32))
    b = np.array(out_ref.astype(mx.float32))
    assert np.abs(a - b).max() < 1e-2, np.abs(a - b).max()
    ka = np.array(kvo.astype(mx.float32))
    kb = np.array(kv_ref.astype(mx.float32))
    assert np.abs(ka - kb).max() < 1e-2, np.abs(ka - kb).max()


def test_native_gate_and_rms_match_reference():
    """dsv4_gate_k (sqrtsoftplus scores + noaux_tc top-k, weights from unbiased
    scores) and dsv4_rms_k, both through the C loop, vs the model math. Data is
    seeded and margin-checked so no near-tie flips the top-k set (a real
    boundary flip is quantization noise, not a kernel bug)."""
    from mlx_soloheaven.models.deepseek_v4 import route, sqrtsoftplus
    from mlx_soloheaven.native.kernels import BUFFER_SLOTS

    n_exp, dim, topk, rscale = 32, 256, 6, 1.5
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((dim,)).astype(np.float32)).astype(mx.bfloat16)
    weight = mx.array(rng.standard_normal((n_exp, dim)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    bias = mx.array(rng.standard_normal(n_exp).astype(np.float32)).astype(mx.float32)
    scores_ref = sqrtsoftplus(x.astype(mx.float32)[None] @ weight.T.astype(mx.float32))
    wref, iref = route(scores_ref, topk, rscale, bias)
    biased = np.array(scores_ref)[0] + np.array(bias)
    order = np.argsort(-biased)
    assert biased[order[topk - 1]] - biased[order[topk]] > 1e-2, "seed hit a near-tie"

    rt_mod.load_custom_kernels(_RT)
    sc = mx.zeros((n_exp,), dtype=mx.float32)
    oi = mx.zeros((topk,), dtype=mx.int32)
    ow = mx.zeros((topk,), dtype=mx.float32)
    mx.eval(x, weight, bias, wref, iref, sc, oi, ow)
    mx.synchronize()
    table = rt_mod.BufferTable()
    gs = BUFFER_SLOTS["dsv4_gate_k"]
    s = {nm: table.add(a) for nm, a in
         [("x", x), ("weight", weight), ("bias", bias), ("scores", sc),
          ("out_idx", oi), ("out_w", ow)]}
    const = struct.pack("<iiif", n_exp, dim, topk, rscale)
    it = rt_mod.plan_item(
        _RT, "dsv4_gate_k", True,
        [(s["x"], gs["x"]), (s["weight"], gs["weight"]), (s["bias"], gs["bias"]),
         (s["scores"], gs["scores"]), (s["out_idx"], gs["out_idx"]),
         (s["out_w"], gs["out_w"])],
        [(0, 12, gs["params"]), (12, 4, gs["feps"])],
        (1, 1, 1), (256, 1, 1),
    )
    _RT.commit([it], table.ptrs, const, wait=True)

    got_set = sorted(np.array(oi).tolist())
    ref_set = sorted(np.array(iref)[0].tolist())
    assert got_set == ref_set, (got_set, ref_set)
    gw = dict(zip(np.array(oi).tolist(), np.array(ow).tolist()))
    rw = dict(zip(np.array(iref)[0].tolist(), np.array(wref)[0].tolist()))
    assert all(abs(gw[k] - rw[k]) < 1e-3 for k in rw)

    # rms
    w = mx.array(rng.standard_normal(dim).astype(np.float32)).astype(mx.bfloat16)
    y = mx.zeros((dim,), dtype=mx.bfloat16)
    xf = x.astype(mx.float32)
    yr = (xf * mx.rsqrt((xf ** 2).mean() + 1e-6) * w.astype(mx.float32))
    mx.eval(w, y, yr)
    mx.synchronize()
    t2 = rt_mod.BufferTable()
    rs = BUFFER_SLOTS["dsv4_rms_k"]
    r = {nm: t2.add(a) for nm, a in [("x", x), ("w", w), ("y", y)]}
    const2 = struct.pack("<if", dim, 1e-6)
    it2 = rt_mod.plan_item(
        _RT, "dsv4_rms_k", True,
        [(r["x"], rs["x"]), (r["w"], rs["w"]), (r["y"], rs["y"])],
        [(0, 4, rs["params"]), (4, 4, rs["feps"])],
        (1, 1, 1), (256, 1, 1),
    )
    _RT.commit([it2], t2.ptrs, const2, wait=True)
    # y is bf16 (serving dtype); values reach ~2.6 so bf16 rounding is ~1.5e-2
    assert np.abs(np.array(y.astype(mx.float32)) - np.array(yr)).max() < 3e-2


def test_native_hc_pre_matches_reference():
    """dsv4_hc_pre_k native (explicit signature, C-loop) == _hc_pre_math (the
    reference the mx.fast twin is diffed against elsewhere). Covers the rms,
    the mixes GEMV, the full 20-iteration Sinkhorn, and the pre-reduction."""
    from mlx_soloheaven.models.deepseek_v4 import _hc_pre_math
    from mlx_soloheaven.native.kernels import BUFFER_SLOTS

    hc, d, iters = 4, 256, 20
    h = mx.random.normal((1, 1, hc, d)).astype(mx.bfloat16)
    fn = mx.random.normal((24, hc * d)).astype(mx.float32)
    scale = mx.random.normal((3,)).astype(mx.float32)
    base = mx.random.normal((24,)).astype(mx.float32)
    yref, pref, cref = _hc_pre_math(h, fn, scale, base, hc, iters, 1e-6, 1e-6)
    y = mx.zeros((d,), dtype=mx.bfloat16)
    post = mx.zeros((hc,), dtype=mx.float32)
    comb = mx.zeros((hc * hc,), dtype=mx.float32)
    mx.eval(h, fn, scale, base, yref, pref, cref, y, post, comb)
    mx.synchronize()

    rt_mod.load_custom_kernels(_RT)
    table = rt_mod.BufferTable()
    sl = BUFFER_SLOTS["dsv4_hc_pre_k"]
    s = {nm: table.add(a) for nm, a in
         [("h", h.reshape(-1)), ("fn", fn), ("scale", scale), ("base", base),
          ("y", y), ("post", post), ("comb", comb)]}
    const = struct.pack("<2i2fi", hc, d, 1e-6, 1e-6, iters)
    it = rt_mod.plan_item(
        _RT, "dsv4_hc_pre_k", True,
        [(s["h"], sl["h"]), (s["fn"], sl["fn"]), (s["scale"], sl["scale"]),
         (s["base"], sl["base"]), (s["y"], sl["y"]), (s["post"], sl["post"]),
         (s["comb"], sl["comb"])],
        [(0, 8, sl["params"]), (8, 8, sl["feps"]), (16, 4, sl["iters"])],
        (1, 1, 1), (256, 1, 1),
    )
    _RT.commit([it], table.ptrs, const, wait=True)

    assert np.abs(np.array(y.astype(mx.float32))
                  - np.array(yref.reshape(-1).astype(mx.float32))).max() < 1e-2
    assert np.abs(np.array(post) - np.array(pref.reshape(-1))).max() < 1e-4
    assert np.abs(np.array(comb) - np.array(cref.reshape(-1))).max() < 1e-4


def test_native_moe_routed_chain_matches_mx_fast():
    """A TWO-item plan (K1 -> K2) sharing an intermediate h buffer, committed
    once, must reproduce the model's routed-expert output. This is the first
    real chained multi-kernel plan — the shape the full-layer replay is made
    of — proving an intermediate buffer written by one dispatch is read by the
    next within a single command buffer."""
    from mlx_soloheaven.models.deepseek_v4 import _moe_routed_kernel
    from mlx_soloheaven.native.kernels import BUFFER_SLOTS

    n_act, d_model, d_inner, E, limit = 6, 256, 192, 16, 10.0
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from mlx_soloheaven.models.deepseek_v4 import ClippedSwiGLU

    glu = SwitchGLU(d_model, d_inner, E, activation=ClippedSwiGLU(limit), bias=False)
    nn.quantize(glu, group_size=64, bits=2)
    # The native kernels are specialized to the DEPLOYED build's dtypes:
    # scales/biases are bf16 there (converter output), while nn.quantize here
    # yields fp32. Cast to match — this is exactly what the real weights are.
    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        proj.scales = proj.scales.astype(mx.bfloat16)
        proj.biases = proj.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 1, d_model)).astype(mx.bfloat16)
    idxs = mx.array([[[1, 7, 3, 12, 0, 9]]], dtype=mx.int32)
    wts = mx.array([[[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]]], dtype=mx.float32)
    ref = _moe_routed_kernel(glu, x, idxs, wts, limit)
    mx.eval(ref)
    mx.synchronize()

    rt_mod.load_custom_kernels(_RT)
    h = mx.zeros((n_act * d_inner,), dtype=mx.float32)
    y = mx.zeros((d_model,), dtype=mx.float32)
    idx1 = idxs.reshape(-1)
    wt1 = wts.reshape(-1)
    mx.eval(h, y, idx1, wt1)
    mx.synchronize()

    table = rt_mod.BufferTable()
    s = {
        "x": table.add(x.reshape(-1)),
        "gw": table.add(glu.gate_proj.weight), "gs_": table.add(glu.gate_proj.scales),
        "gb": table.add(glu.gate_proj.biases),
        "uw": table.add(glu.up_proj.weight), "us": table.add(glu.up_proj.scales),
        "ub": table.add(glu.up_proj.biases),
        "dw": table.add(glu.down_proj.weight), "ds_": table.add(glu.down_proj.scales),
        "db": table.add(glu.down_proj.biases),
        "idxs": table.add(idx1), "wts": table.add(wt1),
        "h": table.add(h), "y": table.add(y),
    }
    w13, w2 = BUFFER_SLOTS["dsv4_moe_w13"], BUFFER_SLOTS["dsv4_moe_w2"]
    const = struct.pack("<iiif", n_act, d_model, d_inner, limit)

    k1 = rt_mod.plan_item(
        _RT, "dsv4_moe_w13", True,
        [(s["x"], w13["x"]), (s["gw"], w13["gw"]), (s["gs_"], w13["gs_"]),
         (s["gb"], w13["gb"]), (s["uw"], w13["uw"]), (s["us"], w13["us"]),
         (s["ub"], w13["ub"]), (s["idxs"], w13["idxs"]), (s["h"], w13["h"])],
        [(0, 12, w13["params"]), (12, 4, w13["feps"])],
        ((n_act * d_inner + 7) // 8, 1, 1), (256, 1, 1),
    )
    k2 = rt_mod.plan_item(
        _RT, "dsv4_moe_w2", True,
        [(s["h"], w2["h"]), (s["dw"], w2["dw"]), (s["ds_"], w2["ds_"]),
         (s["db"], w2["db"]), (s["idxs"], w2["idxs"]), (s["wts"], w2["wts"]),
         (s["y"], w2["y"])],
        [(0, 12, w2["params"])],
        ((d_model + 7) // 8, 1, 1), (256, 1, 1),
    )
    _RT.commit([k1, k2], table.ptrs, const, wait=True)

    got = np.array(y)
    exp = np.array(ref.reshape(-1).astype(mx.float32))
    assert np.abs(got - exp).max() < 2e-2, np.abs(got - exp).max()


def test_c_encode_cost_is_in_budget():
    """CPU-side encode+commit of a 1500-dispatch plan (wait=False, so GPU
    time is excluded — this is the per-token CPU cost that must fit in the
    decode budget alongside async GPU work). python-ctypes was 16.6 ms of
    pure FFI (ladder step 1); the C loop must be a small fraction. < 4 ms is
    the regression bound; it measures ~1-2 ms."""
    N, K = 512, 4096
    qw, sc, bi = _q8(N, K)
    x = mx.random.normal((1, K)).astype(mx.bfloat16)
    y = mx.zeros((1, N), dtype=mx.bfloat16)
    mx.eval(qw, sc, bi, x, y)
    mx.synchronize()
    bufs, keep = [], []
    for a in (qw, sc, bi, x, y):
        ptr, off, cap = rt_mod.mtl_buffer(a)
        keep.append(cap)
        bufs.append(ptr)
    const = struct.pack("<ii", K, N)
    items = [rt_mod.plan_qmv(_RT, (0, 1, 2, 3, 4), K, N, 0, 4) for _ in range(1500)]

    _RT.commit(items, bufs, const, wait=False)  # warm
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        _RT.commit(items, bufs, const, wait=False)
        best = min(best, time.perf_counter() - t0)
    mx.synchronize()  # drain the queued commits before the test returns
    assert best < 4e-3, f"CPU encode+commit of 1500 dispatches took {best*1e3:.1f} ms"
