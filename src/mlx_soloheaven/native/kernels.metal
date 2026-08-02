// dsv4 native decode kernels — explicit [[buffer(i)]] signatures for the
// external replay loop (Stage 3b). The BODIES are identical to the
// mx.fast.metal_kernel versions in models/deepseek_v4.py (which are
// differential-tested against dequantized reference math); only the
// signature differs, because mx.fast generates it and here we write it.
//
// The runtime compiles this via newLibraryWithSource. Kept in lockstep with
// the model's kernels; a mismatch is caught by tests/test_dsv4_native.py,
// which diffs the native kernel against the mx.fast one on identical inputs.

#include <metal_stdlib>
using namespace metal;

// MoE down-projection: y[d_model] = sum_e wts[e] * (down_e . h_e), 2-bit.
// grid: ceil(d_model/8) threadgroups x (256) ; one simdgroup per 8-block.
kernel void dsv4_moe_w2(
    const device float*    h        [[buffer(0)]],   // [n_act, d_inner]
    const device uint32_t* dw       [[buffer(1)]],   // [E, d_model, d_inner/16]
    const device bfloat*   ds_      [[buffer(2)]],   // [E, d_model, d_inner/64]
    const device bfloat*   db       [[buffer(3)]],
    const device int*      idxs     [[buffer(4)]],   // [n_act]
    const device float*    wts      [[buffer(5)]],   // [n_act]
    const device int*      params   [[buffer(6)]],   // n_act, d_model, d_inner
    device float*          y        [[buffer(7)]],   // [d_model]
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const int TG = 256;
    uint dim = tgid * 8 + sg_id;
    const int n_act = params[0];
    const int d_model = params[1];
    const int d_inner = params[2];
    const int words = d_inner / 16;
    const int wpg = 4;

    threadgroup float hs[2048];

    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) {
        int e = idxs[s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e >= 0) {
            for (int i = tid; i < d_inner; i += TG) hs[i] = h[s * d_inner + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e < 0 || dim >= (uint)d_model) continue;
        float we = wts[s];
        const uint base = ((uint)e * (uint)d_model + dim) * (uint)words;
        const uint sbase = ((uint)e * (uint)d_model + dim) * (uint)(d_inner / 64);
        float a = 0.0f;
        for (int w = lane; w < words; w += 32) {
            uint p = dw[base + w];
            uint g_ = w / wpg;
            float sc = float(ds_[sbase + g_]);
            float bi = float(db[sbase + g_]);
            threadgroup const float* hv = hs + w * 16;
            float aw = 0.0f, sw = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                float hj = hv[j];
                aw += float((p >> (2 * j)) & 3u) * hj;
                sw += hj;
            }
            a += aw * sc + sw * bi;
        }
        acc += we * a;
    }
    if (dim >= (uint)d_model) return;
    acc = simd_sum(acc);
    if (lane == 0) y[dim] = acc;
}
