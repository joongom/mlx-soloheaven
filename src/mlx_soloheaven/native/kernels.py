"""Generate the native kernel source from the model's FROZEN kernel bodies.

Single source of truth: the kernel BODIES live in models/deepseek_v4.py and
are differential-tested there via mx.fast.metal_kernel. This module imports
those exact body strings and wraps each with an explicit ``[[buffer(i)]]``
signature so the external replay loop can dispatch them. Because the body is
imported verbatim, the native kernel and its mx.fast twin can never drift —
and tests/test_dsv4_native.py diffs them anyway as a belt-and-braces check.

The serving dtype is bf16, so the T-templated kernels (attention core,
compressor step) are specialized to ``bfloat`` here via ``typedef``.
"""

from __future__ import annotations

from mlx_soloheaven.models import deepseek_v4 as _m

# Per kernel: the body-holding attribute on the model module, and the buffer
# declarations IN ORDER (the [[buffer(i)]] index is the list position). Types
# match what mx.fast binds: quantized weights uint32, scales/biases bfloat,
# indices/params int, scalars float; T-typed activation buffers -> bfloat.
_B = "const device bfloat*"
_U = "const device uint32_t*"
_I = "const device int*"
_F = "const device float*"

_SPECS = {
    "sh_dsv4_moe_w13": (
        "_MOE_K1_SRC",
        [("x", _B), ("gw", _U), ("gs_", _B), ("gb", _B), ("uw", _U), ("us", _B),
         ("ub", _B), ("idxs", _I), ("params", _I), ("feps", _F),
         ("h", "device float*")],
    ),
    "sh_dsv4_moe_w2": (
        "_MOE_K2_SRC",
        [("h", _F), ("dw", _U), ("ds_", _B), ("db", _B), ("idxs", _I),
         ("wts", _F), ("params", _I), ("y", "device float*")],
    ),
    "sh_dsv4_attn_core": (
        "_ATTN_CORE_SRC",
        [("q", _B), ("kv", _B), ("ring", "device bfloat*"), ("comp", _B),
         ("cidx", _I),
         ("sink", _F), ("freqs", _F), ("params", _I), ("fscal", _F),
         ("ioff", _I), ("out", "device bfloat*"), ("kv_out", "device bfloat*")],
    ),
    "sh_dsv4_ring_store_k": (
        "_RING_STORE_SRC",
        [("src", _B), ("params", _I), ("ioff", _I), ("ring", "device bfloat*")],
    ),
    "sh_dsv4_wo_a_k": (
        "_WO_A_SRC",
        [("x", _B), ("weight", _U), ("scales", _B), ("biases", _B),
         ("params", _I), ("out", "device bfloat*")],
    ),
    "sh_dsv4_sh13_k": (
        "_SH13_SRC",
        [("x", _B), ("w1", _U), ("s1", _B), ("b1", _B),
         ("w3", _U), ("s3", _B), ("b3", _B),
         ("params", _I), ("feps", _F), ("out", "device bfloat*")],
    ),
    "sh_dsv4_swiglu_k": (
        "_SWIGLU_SRC",
        [("gate", _B), ("up", _B), ("params", _I), ("feps", _F),
         ("out", "device bfloat*")],
    ),
    "sh_dsv4_add_k": (
        "_ADD_SRC",
        [("a", _F), ("b", _B), ("params", _I), ("out", "device bfloat*")],
    ),
    "sh_dsv4_idx_score_k": (
        "_IDX_SCORE_SRC",
        [("q", _B), ("buf", _B), ("w", _B), ("freqs", _F), ("params", _I),
         ("fscal", _F), ("ioff", _I), ("scores", "device float*")],
    ),
    "sh_dsv4_idx_topk_k": (
        "_IDX_TOPK_SRC",
        [("scores", _F), ("params", _I), ("ioff", _I), ("out_idx", "device int*")],
    ),
    "sh_dsv4_embed_k": (
        "_EMBED_SRC",
        [("weight", _U), ("scales", _B), ("biases", _B), ("params", _I),
         ("ioff", _I), ("h", "device bfloat*")],
    ),
    "sh_dsv4_hc_head_k": (
        "_HC_HEAD_SRC",
        [("h", _B), ("fn", _F), ("scale", _F), ("base", _F), ("params", _I),
         ("feps", _F), ("y", "device bfloat*")],
    ),
    "sh_dsv4_gate_k": (
        "_GATE_SRC",
        [("x", _B), ("weight", _B), ("bias", _F), ("params", _I), ("feps", _F),
         ("scores", "device float*"), ("out_idx", "device int*"),
         ("out_w", "device float*")],
    ),
    "sh_dsv4_gate_hash_k": (
        "_GATE_HASH_SRC",
        [("x", _B), ("weight", _B), ("tid2eid", _I), ("params", _I), ("feps", _F),
         ("ioff", _I), ("out_idx", "device int*"), ("out_w", "device float*")],
    ),
    "sh_dsv4_gate_score_k": (
        "_GATE_SCORE_SRC",
        [("x", _B), ("weight", _B), ("params", _I), ("scores", "device float*")],
    ),
    "sh_dsv4_gate_topk_k": (
        "_GATE_TOPK_SRC",
        [("scores", _F), ("bias", _F), ("params", _I), ("feps", _F),
         ("out_idx", "device int*"), ("out_w", "device float*")],
    ),
    "sh_dsv4_rms2_k": (
        "_RMS2_SRC",
        [("xa", _B), ("wa", _B), ("xb", _B), ("wb", _B), ("params", _I),
         ("feps", _F), ("ya", "device bfloat*"), ("yb", "device bfloat*")],
    ),
    "sh_dsv4_rms_k": (
        "_RMS_SRC",
        [("x", _B), ("w", _B), ("params", _I), ("feps", _F),
         ("y", "device bfloat*")],
    ),
    "sh_dsv4_hc_mix_k": (
        "_HC_MIX_SRC",
        [("h", _B), ("fn", _F), ("params", _I), ("mixes", "device float*")],
    ),
    "sh_dsv4_hc_pre_k": (
        "_HC_PRE_SRC",
        [("h", _B), ("mixes", _F), ("scale", _F), ("base", _F), ("params", _I),
         ("feps", _F), ("iters", _I), ("y", "device bfloat*"),
         ("post", "device float*"), ("comb", "device float*")],
    ),
    "sh_dsv4_hc_post_k": (
        "_HC_POST_SRC",
        [("x", _B), ("residual", _B), ("post", _F), ("comb", _F), ("params", _I),
         ("y", "device bfloat*")],
    ),
    "sh_dsv4_hc_post2_k": (
        "_HC_POST2_SRC",
        [("a", _F), ("b", _B), ("residual", _B), ("post", _F), ("comb", _F),
         ("params", _I), ("y", "device bfloat*")],
    ),
    "sh_dsv4_comp_step": (
        "_COMP_STEP_SRC",
        [("kv_row", _B), ("sc_row", _B), ("kv_st", "device float*"),
         ("sc_st", "device float*"),
         ("ape", _F), ("nw", _B), ("freqs", _F), ("params", _I), ("feps", _F),
         ("ioff", _I), ("row_out", "device bfloat*"), ("old_row", _B)],
    ),
}

# Canonical thread-position params, declared with the SAME names the bodies
# use (the bodies reference e.g. `threadgroup_position_in_grid.x`).
_THREAD_PARAMS = [
    "uint3 thread_position_in_grid [[thread_position_in_grid]]",
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
    "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
    "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
]


def _wrap(name: str, body: str, bufs: list[tuple[str, str]]) -> str:
    args = [f"    {ty} {nm} [[buffer({i})]]" for i, (nm, ty) in enumerate(bufs)]
    args += [f"    {p}" for p in _THREAD_PARAMS]
    return (
        f"kernel void {name}(\n"
        + ",\n".join(args)
        + ")\n{\n    typedef bfloat T;\n"
        # attn_core folds the ring write in (Stage 4h); the mx.fast twin
        # defines this away because its `ring` is a const input.
        + "#define RING_WRITE(s, i, v) ring[(s) * D + (i)] = (v)\n"
        + body
        + "\n}\n"
    )


def build_source() -> str:
    """The complete native/kernels source, generated from the model bodies."""
    # The 2-bit unpack table is file-scope `constant`, so it is emitted ONCE
    # here; the mx.fast twins get the same text as their per-kernel header.
    parts = ["#include <metal_stdlib>", "using namespace metal;", "",
             _m._Q2_LUT_HEADER, ""]
    for name, (attr, bufs) in _SPECS.items():
        parts.append(_wrap(name, getattr(_m, attr), bufs))
    return "\n".join(parts)


#: buffer name -> [[buffer(i)]] index, per kernel — for plan builders so slot
#: numbers stay in one place.
BUFFER_SLOTS = {
    name: {nm: i for i, (nm, _) in enumerate(bufs)}
    for name, (_attr, bufs) in _SPECS.items()
}
