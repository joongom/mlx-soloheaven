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
