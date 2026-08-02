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
    "dsv4_moe_w13": (
        "_MOE_K1_SRC",
        [("x", _B), ("gw", _U), ("gs_", _B), ("gb", _B), ("uw", _U), ("us", _B),
         ("ub", _B), ("idxs", _I), ("params", _I), ("feps", _F),
         ("h", "device float*")],
    ),
    "dsv4_moe_w2": (
        "_MOE_K2_SRC",
        [("h", _F), ("dw", _U), ("ds_", _B), ("db", _B), ("idxs", _I),
         ("wts", _F), ("params", _I), ("y", "device float*")],
    ),
    "dsv4_attn_core": (
        "_ATTN_CORE_SRC",
        [("q", _B), ("kv", _B), ("ring", _B), ("comp", _B), ("cidx", _I),
         ("sink", _F), ("freqs", _F), ("params", _I), ("fscal", _F),
         ("ioff", _I), ("out", "device bfloat*"), ("kv_out", "device bfloat*")],
    ),
    "dsv4_hc_pre_k": (
        "_HC_PRE_SRC",
        [("h", _B), ("fn", _F), ("scale", _F), ("base", _F), ("params", _I),
         ("feps", _F), ("iters", _I), ("y", "device bfloat*"),
         ("post", "device float*"), ("comb", "device float*")],
    ),
    "dsv4_hc_post_k": (
        "_HC_POST_SRC",
        [("x", _B), ("residual", _B), ("post", _F), ("comb", _F), ("params", _I),
         ("y", "device bfloat*")],
    ),
    "dsv4_comp_step": (
        "_COMP_STEP_SRC",
        [("kv_row", _B), ("sc_row", _B), ("kv_st", _F), ("sc_st", _F),
         ("ape", _F), ("nw", _B), ("freqs", _F), ("params", _I), ("feps", _F),
         ("ioff", _I), ("kv_out", "device float*"), ("sc_out", "device float*"),
         ("row_out", "device bfloat*"), ("old_row", _B)],
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
        + body
        + "\n}\n"
    )


def build_source() -> str:
    """The complete native/kernels source, generated from the model bodies."""
    parts = ["#include <metal_stdlib>", "using namespace metal;", ""]
    for name, (attr, bufs) in _SPECS.items():
        parts.append(_wrap(name, getattr(_m, attr), bufs))
    return "\n".join(parts)


#: buffer name -> [[buffer(i)]] index, per kernel — for plan builders so slot
#: numbers stay in one place.
BUFFER_SLOTS = {
    name: {nm: i for i, (nm, _) in enumerate(bufs)}
    for name, (_attr, bufs) in _SPECS.items()
}
