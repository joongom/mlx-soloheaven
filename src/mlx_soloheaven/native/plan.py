"""Per-layer decode PLAN assembly for the replay runtime.

Turns a Block + scratch buffers into the native dispatch sequence the C encode
loop replays. Every builder here mirrors an individually diff-verified plan in
tests/test_dsv4_native.py — this is wiring, not new compute. Dense layers only
for now (ratio==0); compressed layers add the compressor/indexer dispatches.

Buffer/dtype contracts (from the deployed build, all diff-tested):
* quantized projections: 8-bit gs64, scales/biases bf16; routed experts 2-bit.
* norms and gate.weight are bf16.
* every projection K is %512, so affine_qmv_fast covers them.
"""

from __future__ import annotations

import struct

from mlx_soloheaven.native.kernels import BUFFER_SLOTS

_QMV = "affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0"


class ConstBlob:
    """Accumulates plan-static setBytes data; returns (offset, length)."""

    def __init__(self):
        self._b = bytearray()

    def add(self, fmt, *vals):
        off = len(self._b)
        self._b += struct.pack("<" + fmt, *vals)
        return off, len(self._b) - off

    def bytes(self):
        return bytes(self._b)


class Planner:
    """Holds the runtime, table, const blob and scratch, and builds items."""

    def __init__(self, rt, table, cb: ConstBlob, scratch: dict):
        self.rt = rt
        self.t = table
        self.cb = cb
        self.S = scratch  # name -> table slot (pre-registered scratch buffers)

    def _pi(self, kernel, custom, bufs, bytes_, grid, group, barrier=True):
        from mlx_soloheaven.native.runtime import plan_item
        return plan_item(self.rt, kernel, custom, bufs, bytes_, grid, group, barrier)

    def qmv(self, mod, x, y, K, N, w_off=0, s_off=0, b_off=0, xoff=0, yoff=0):
        ko, _ = self.cb.add("ii", K, N)
        return self._pi(
            _QMV, False,
            [(self.t.add(mod.weight), 0, w_off), (self.t.add(mod.scales), 1, s_off),
             (self.t.add(mod.biases), 2, b_off), (self.S[x], 3, xoff), (self.S[y], 4, yoff)],
            [(ko, 4, 5), (ko + 4, 4, 6)], (1, (N + 7) // 8, 1), (32, 2, 1))

    def rms(self, w, x, y, d, eps, xoff=0):
        o, _ = self.cb.add("if", d, eps)
        r = BUFFER_SLOTS["sh_dsv4_rms_k"]
        return self._pi(
            "sh_dsv4_rms_k", True,
            [(self.S[x], r["x"], xoff), (self.t.add(w), r["w"]), (self.S[y], r["y"])],
            [(o, 4, r["params"]), (o + 4, 4, r["feps"])], (1, 1, 1), (256, 1, 1))

    def hc_pre(self, h, fn, scale, base, x, post, comb, hc, hidden, iters, eps, hc_eps):
        # Two dispatches: dsv4_hc_mix_k parallelizes the ~1.5 MB fn.h GEMV over
        # (mix) threadgroups into the shared `hc_mixes` scratch, then hc_pre reads
        # those raw dots and does rms/sinkhorn/gates/output in one threadgroup.
        p, _ = self.cb.add("2i", hc, hidden)
        f, _ = self.cb.add("2f", eps, hc_eps)
        i, _ = self.cb.add("i", iters)
        mix = (2 + hc) * hc
        mk = BUFFER_SLOTS["sh_dsv4_hc_mix_k"]
        k = BUFFER_SLOTS["sh_dsv4_hc_pre_k"]
        return [
            self._pi(
                "sh_dsv4_hc_mix_k", True,
                [(self.S[h], mk["h"]), (self.t.add(fn), mk["fn"]),
                 (self.S["hc_mixes"], mk["mixes"])],
                [(p, 8, mk["params"])], (mix, 1, 1), (1024, 1, 1)),
            self._pi(
                "sh_dsv4_hc_pre_k", True,
                [(self.S[h], k["h"]), (self.S["hc_mixes"], k["mixes"]),
                 (self.t.add(scale), k["scale"]), (self.t.add(base), k["base"]),
                 (self.S[x], k["y"]), (self.S[post], k["post"]), (self.S[comb], k["comb"])],
                [(p, 8, k["params"]), (f, 8, k["feps"]), (i, 4, k["iters"])],
                (1, 1, 1), (1024, 1, 1)),
        ]

    def hc_post(self, x, residual, post, comb, y, hc, hidden):
        p, _ = self.cb.add("2i", hc, hidden)
        k = BUFFER_SLOTS["sh_dsv4_hc_post_k"]
        # grid = hc * NSPLIT(8): must match the kernel's compile-time NSPLIT.
        return self._pi(
            "sh_dsv4_hc_post_k", True,
            [(self.S[x], k["x"]), (self.S[residual], k["residual"]), (self.S[post], k["post"]),
             (self.S[comb], k["comb"]), (self.S[y], k["y"])],
            [(p, 8, k["params"])], (hc * 8, 1, 1), (256, 1, 1))

    def hc_post2(self, a, b, residual, post, comb, y, hc, hidden):
        """hc_post with x = a + b fused in (the MoE routed+shared add)."""
        p, _ = self.cb.add("2i", hc, hidden)
        k = BUFFER_SLOTS["sh_dsv4_hc_post2_k"]
        return self._pi(
            "sh_dsv4_hc_post2_k", True,
            [(self.S[a], k["a"]), (self.S[b], k["b"]), (self.S[residual], k["residual"]),
             (self.S[post], k["post"]), (self.S[comb], k["comb"]), (self.S[y], k["y"])],
            [(p, 8, k["params"])], (hc * 8, 1, 1), (256, 1, 1))


def plan_compressor(pl: Planner, comp, freqs, kv_src: str, sc_src: str,
                    cache: dict, ioff_off: int, n: int,
                    kv_off: int = 0, sc_off: int = 0) -> list:
    """Compressor decode step (dsv4_comp_step). ``kv_src``/``sc_src`` are the
    scratch slots holding comp.wkv/comp.wgate outputs; ``cache`` names the
    per-layer state slots (kv_st, sc_st, buf). ``n`` (Python) is the
    completed-group count so the buf write lands at row n; the driver
    re-encodes per token (cheap: 0.65 ms/1500), so baking n as a static byte
    offset is fine. State updates IN PLACE: only the fresh slot row is
    written (pooling redirects that row's reads to kv_row/sc_row), and the
    overlap-head shift runs behind the kernel's pooling barrier — the old
    full-state double-buffer copy cost 2.5 ms/token (Stage 3o)."""
    cs = BUFFER_SLOTS["sh_dsv4_comp_step"]
    d, ratio, coff = comp.head_dim, comp.ratio, comp.coff
    po, _ = pl.cb.add("4i", ratio, d, coff, comp.rope_dim)
    fo, _ = pl.cb.add("f", 1e-6)
    row_off = n * d * 2  # bf16 buf row stride
    return [pl._pi(
        "sh_dsv4_comp_step", True,
        [(pl.S[kv_src], cs["kv_row"], kv_off), (pl.S[sc_src], cs["sc_row"], sc_off),
         (pl.S[cache["kv_st"]], cs["kv_st"]), (pl.S[cache["sc_st"]], cs["sc_st"]),
         (pl.t.add(comp.ape), cs["ape"]), (pl.t.add(comp.norm.weight), cs["nw"]),
         (pl.t.add(freqs), cs["freqs"]),
         (pl.S[cache["buf"]], cs["row_out"], row_off), (pl.S[cache["buf"]], cs["old_row"], row_off)],
        [(po, 16, cs["params"]), (fo, 4, cs["feps"]), (ioff_off, 4, cs["ioff"])],
        (1, 1, 1), (1024, 1, 1))]


def plan_indexer(pl: Planner, idxr, freqs, xin: str, qr: str, idx_cache: dict,
                 ioff_off: int, n: int, ncomp: int,
                 iw_off: int, ickv_off: int, icwg_off: int) -> list:
    """DSA indexer: comp step + wq_b qmv + score/top-k -> cidx
    (scratch['cidx']). Its x-projections (weights_proj, its compressor's
    wkv/wgate) arrive PRE-COMPUTED in scratch['xall'] at byte offsets
    iw_off/ickv_off/icwg_off (the stacked x-projection — see plan_attention);
    only wq_b (whose input is qr, not x) dispatches here. idx_cache names the
    indexer's compressor state slots. cidx indexes the MAIN compressor's
    groups (aligned by position). ncomp = visible groups."""
    hd, rd, n_h = idxr.head_dim, idxr.rope_dim, idxr.n_heads
    q_lora = idxr.wq_b.scales.shape[1] * idxr.wq_b.group_size
    items = [
        pl.qmv(idxr.wq_b, qr, "iq", q_lora, n_h * hd),
    ]
    items += plan_compressor(pl, idxr.compressor, freqs, "xall", "xall",
                             idx_cache, ioff_off, n,
                             kv_off=ickv_off, sc_off=icwg_off)
    isk, itk = BUFFER_SLOTS["sh_dsv4_idx_score_k"], BUFFER_SLOTS["sh_dsv4_idx_topk_k"]
    cap = 256
    sp, _ = pl.cb.add("4i", n_h, hd, rd, cap)
    sf, _ = pl.cb.add("f", hd ** -0.5 * n_h ** -0.5)
    tp, _ = pl.cb.add("2i", cap, idxr.topk)
    items.append(pl._pi(
        "sh_dsv4_idx_score_k", True,
        [(pl.S["iq"], isk["q"]), (pl.S[idx_cache["i_buf"]], isk["buf"]),
         (pl.S["xall"], isk["w"], iw_off), (pl.t.add(freqs), isk["freqs"]),
         (pl.S["scores"], isk["scores"])],
        [(sp, 16, isk["params"]), (sf, 4, isk["fscal"]), (ioff_off, 8, isk["ioff"])],
        (cap, 1, 1), (256, 1, 1)))
    items.append(pl._pi(
        "sh_dsv4_idx_topk_k", True,
        [(pl.S["scores"], itk["scores"]), (pl.S["cidx"], itk["out_idx"])],
        [(tp, 8, itk["params"]), (ioff_off, 8, itk["ioff"])], (1, 1, 1), (256, 1, 1)))
    return items


def plan_attention(pl: Planner, attn, xin: str, ring: str, out: str,
                   ioff_off: int, comp_cache: dict | None = None, ncomp: int = 0,
                   n: int = 0, idx_cache: dict | None = None) -> list:
    """Attention on scratch[xin] (the attn_norm output) -> scratch[out].
    Dense (comp_cache is None) or plain-compressed (ratio-128: comp_cache
    names the compressor state slots, ncomp = visible compressed groups).
    Indexer (ratio-4) layers are not yet wired here."""
    import types

    a = attn
    H, D, g = a.n_heads, a.head_dim, a.n_groups
    NHD, gin, o_lora, win = H * D, (H * D) // g, a.wo_a.weight.shape[1], a.window
    hidden = a.wq_a.scales.shape[1] * a.wq_a.group_size
    q_lora = a.wq_b.scales.shape[1] * a.wq_b.group_size
    # ONE stacked qmv for every projection of x (wq_a, wkv, comp wkv/wgate,
    # indexer weights_proj + its comp wkv/wgate) into scratch['xall'];
    # consumers read their slice at a byte offset. Reuses the model's lazily
    # concatenated _x_stack (identical numerics, stable array identity for the
    # buffer table). Saves up to 6 dispatches/layer over separate qmv.
    st = a._x_stack()
    assert st[0] == "q", "native plan requires the quantized build"
    _, sqw, ssc, sbs, _gs, _bits, sizes = st
    shim = types.SimpleNamespace(weight=sqw, scales=ssc, biases=sbs)
    offs, acc = [], 0
    for sz in sizes:
        offs.append(acc * 2)  # bf16 bytes
        acc += sz
    items = [
        pl.qmv(shim, xin, "xall", hidden, acc),
        pl.rms(a.q_norm.weight, "xall", "qr", q_lora, a.eps, xoff=offs[0]),
        pl.qmv(a.wq_b, "qr", "q_raw", q_lora, NHD),
        pl.rms(a.kv_norm.weight, "xall", "kvn", D, a.eps, xoff=offs[1]),
    ]
    kc, plain = 0, 1
    comp_slot, cidx_slot = "dummy", "dummy_idx"
    if idx_cache is not None:               # ratio-4: indexer selects groups
        items += plan_indexer(pl, a.indexer, a._freqs, xin, "qr", idx_cache,
                              ioff_off, n, ncomp,
                              iw_off=offs[4], ickv_off=offs[5], icwg_off=offs[6])
        cidx_slot = "cidx"
        plain = 0
    if comp_cache is not None:
        items += plan_compressor(pl, a.compressor, a._freqs, "xall", "xall",
                                 comp_cache, ioff_off, n,
                                 kv_off=offs[2], sc_off=offs[3])
        comp_slot = comp_cache["buf"]
        # plain layers: kc = visible groups; indexer layers: kc = topk width.
        kc = a.indexer.topk if idx_cache is not None else ncomp
    ac = BUFFER_SLOTS["sh_dsv4_attn_core"]
    po, _ = pl.cb.add("5i", D, a.rope_dim, win, kc, plain)
    fo, _ = pl.cb.add("2f", a.scale, a.eps)
    items.append(pl._pi(
        "sh_dsv4_attn_core", True,
        [(pl.S["q_raw"], ac["q"]), (pl.S["kvn"], ac["kv"]), (pl.S[ring], ac["ring"]),
         (pl.S[comp_slot], ac["comp"]), (pl.S[cidx_slot], ac["cidx"]),
         (pl.t.add(a.attn_sink), ac["sink"]), (pl.t.add(a._freqs), ac["freqs"]),
         (pl.S["acore"], ac["out"]), (pl.S["kv_roped"], ac["kv_out"])],
        [(po, 20, ac["params"]), (fo, 8, ac["fscal"]), (ioff_off, 8, ac["ioff"])],
        (H, 1, 1), (512, 1, 1)))
    rp, _ = pl.cb.add("ii", D, win)
    # the ring write now happens inside attn_core (Stage 4h)
        # o_groups grouped 8-bit qmv as ONE dispatch (was g separate library qmv):
    # out[gi*o_lora+j] = deq(wo_a[gi,j]) . acore[gi*gin:], one simdgroup/row.
    wa = BUFFER_SLOTS["sh_dsv4_wo_a_k"]
    wo, _ = pl.cb.add("3i", g, gin, o_lora)
    items.append(pl._pi(
        "sh_dsv4_wo_a_k", True,
        [(pl.S["acore"], wa["x"]), (pl.t.add(a.wo_a.weight), wa["weight"]),
         (pl.t.add(a.wo_a.scales), wa["scales"]), (pl.t.add(a.wo_a.biases), wa["biases"]),
         (pl.S["o_lora"], wa["out"])],
        [(wo, 12, wa["params"])], ((g * o_lora + 7) // 8, 1, 1), (256, 1, 1)))
    items.append(pl.qmv(a.wo_b, "o_lora", out, g * o_lora, hidden))
    return items


def plan_moe(pl: Planner, ffn, xin: str, out: str, topk: int, rscale: float,
             limit: float, tok_off: int = 0) -> list:
    """MoE on scratch[xin] -> scratch[out]. scratch must hold: scores, idx, w,
    hexp, y_routed, sg, su, sh, shared. Score layers run the noaux_tc gate;
    hash layers (the first num_hash_layers) index experts by tid2eid[token]
    (token at ``tok_off`` in the const blob) with no top-k search or bias."""
    exp, sh = ffn.experts, ffn.shared_experts
    n_exp = ffn.gate.weight.shape[0]
    hidden = ffn.gate.weight.shape[1]
    inter = exp.gate_proj.weight.shape[1]
    items = []
    go, _ = pl.cb.add("iiif", n_exp, hidden, topk, rscale)
    if getattr(ffn.gate, "hash", False):
        gh = BUFFER_SLOTS["sh_dsv4_gate_hash_k"]
        items.append(pl._pi(
            "sh_dsv4_gate_hash_k", True,
            [(pl.S[xin], gh["x"]), (pl.t.add(ffn.gate.weight), gh["weight"]),
             (pl.t.add(ffn.gate.tid2eid), gh["tid2eid"]),
             (pl.S["idx"], gh["out_idx"]), (pl.S["w"], gh["out_w"])],
            [(go, 12, gh["params"]), (go + 12, 4, gh["feps"]), (tok_off, 4, gh["ioff"])],
            (1, 1, 1), (256, 1, 1)))
    else:
        # score every expert in parallel (grid = n_exp threadgroups so the chip
        # hides the weight-fetch latency), then a tiny top-k over the scores.
        gs = BUFFER_SLOTS["sh_dsv4_gate_score_k"]
        items.append(pl._pi(
            "sh_dsv4_gate_score_k", True,
            [(pl.S[xin], gs["x"]), (pl.t.add(ffn.gate.weight), gs["weight"]),
             (pl.S["scores"], gs["scores"])],
            [(go, 8, gs["params"])], (n_exp, 1, 1), (256, 1, 1)))
        gt = BUFFER_SLOTS["sh_dsv4_gate_topk_k"]
        items.append(pl._pi(
            "sh_dsv4_gate_topk_k", True,
            [(pl.S["scores"], gt["scores"]), (pl.t.add(ffn.gate.bias), gt["bias"]),
             (pl.S["idx"], gt["out_idx"]), (pl.S["w"], gt["out_w"])],
            [(go, 12, gt["params"]), (go + 12, 4, gt["feps"])], (1, 1, 1), (256, 1, 1)))
    mo, _ = pl.cb.add("iii", topk, hidden, inter)
    ml, _ = pl.cb.add("f", limit)
    w1 = BUFFER_SLOTS["sh_dsv4_moe_w13"]
    items.append(pl._pi(
        "sh_dsv4_moe_w13", True,
        [(pl.S[xin], w1["x"]), (pl.t.add(exp.gate_proj.weight), w1["gw"]),
         (pl.t.add(exp.gate_proj.scales), w1["gs_"]), (pl.t.add(exp.gate_proj.biases), w1["gb"]),
         (pl.t.add(exp.up_proj.weight), w1["uw"]), (pl.t.add(exp.up_proj.scales), w1["us"]),
         (pl.t.add(exp.up_proj.biases), w1["ub"]), (pl.S["idx"], w1["idxs"]), (pl.S["hexp"], w1["h"])],
        [(mo, 12, w1["params"]), (ml, 4, w1["feps"])], ((topk * inter + 7) // 8, 1, 1), (256, 1, 1)))
    w2 = BUFFER_SLOTS["sh_dsv4_moe_w2"]
    items.append(pl._pi(
        "sh_dsv4_moe_w2", True,
        [(pl.S["hexp"], w2["h"]), (pl.t.add(exp.down_proj.weight), w2["dw"]),
         (pl.t.add(exp.down_proj.scales), w2["ds_"]), (pl.t.add(exp.down_proj.biases), w2["db"]),
         (pl.S["idx"], w2["idxs"]), (pl.S["w"], w2["wts"]), (pl.S["y_routed"], w2["y"])],
        [(mo, 12, w2["params"])], ((hidden + 7) // 8, 1, 1), (256, 1, 1)))
    # shared expert w1 + w3 + clipped SwiGLU as ONE dispatch (was 2 library
    # qmv + the elementwise swiglu): one simdgroup per inter row does both
    # 8-bit dots and applies the activation in-register.
    k13 = BUFFER_SLOTS["sh_dsv4_sh13_k"]
    so, _ = pl.cb.add("2i", hidden, inter)
    sf, _ = pl.cb.add("f", limit)
    items.append(pl._pi(
        "sh_dsv4_sh13_k", True,
        [(pl.S[xin], k13["x"]),
         (pl.t.add(sh.w1.weight), k13["w1"]), (pl.t.add(sh.w1.scales), k13["s1"]),
         (pl.t.add(sh.w1.biases), k13["b1"]),
         (pl.t.add(sh.w3.weight), k13["w3"]), (pl.t.add(sh.w3.scales), k13["s3"]),
         (pl.t.add(sh.w3.biases), k13["b3"]),
         (pl.S["sh"], k13["out"])],
        [(so, 8, k13["params"]), (sf, 4, k13["feps"])],
        ((inter + 7) // 8, 1, 1), (256, 1, 1)))
    items.append(pl.qmv(sh.w2, "sh", "shared", inter, hidden))
    # No add here: the block's hc_post2 fuses y_routed + shared into its x.
    return items


def plan_embed(pl: Planner, embed, hout: str, hc: int, hidden: int,
               tok_off: int) -> list:
    """Dequantize the embedding row for the token in the uniform buffer (at
    byte offset ``tok_off``) into scratch[hout] as hc replicated streams."""
    ek = BUFFER_SLOTS["sh_dsv4_embed_k"]
    p, _ = pl.cb.add("2i", hidden, hc)
    return [pl._pi(
        "sh_dsv4_embed_k", True,
        [(pl.t.add(embed.weight), ek["weight"]), (pl.t.add(embed.scales), ek["scales"]),
         (pl.t.add(embed.biases), ek["biases"]), (pl.S[hout], ek["h"])],
        [(p, 8, ek["params"]), (tok_off, 4, ek["ioff"])], (hidden, 1, 1), (256, 1, 1))]


def plan_head(pl: Planner, model, hin: str, logits: str, hc: int, hidden: int,
              vocab: int) -> list:
    """hc-head reduce -> final RMSNorm -> head qmv, scratch[hin] -> scratch[logits]."""
    hk = BUFFER_SLOTS["sh_dsv4_hc_head_k"]
    p, _ = pl.cb.add("2i", hc, hidden)
    f, _ = pl.cb.add("2f", model.eps, model.hc_eps)
    items = [pl._pi(
        "sh_dsv4_hc_head_k", True,
        [(pl.S[hin], hk["h"]), (pl.t.add(model.hc_head_fn.reshape(-1)), hk["fn"]),
         (pl.t.add(model.hc_head_scale), hk["scale"]), (pl.t.add(model.hc_head_base), hk["base"]),
         (pl.S["headx"], hk["y"])],
        [(p, 8, hk["params"]), (f, 8, hk["feps"])], (1, 1, 1), (256, 1, 1))]
    items.append(pl.rms(model.norm.weight, "headx", "headn", hidden, model.eps))
    items.append(pl.qmv(model.head, "headn", logits, hidden, vocab))
    return items


def plan_block(pl: Planner, blk, hin: str, ring: str, hout: str, ioff_off: int,
               topk: int, rscale: float, limit: float, comp_cache: dict | None = None,
               idx_cache: dict | None = None, ncomp: int = 0, n: int = 0,
               tok_off: int = 0) -> list:
    """A full Block: hc_pre/attn_norm/attention/hc_post, then
    hc_pre/ffn_norm/MoE/hc_post. scratch[hin] in, scratch[hout] out. Dense when
    comp_cache is None; plain-compressed (comp_cache) or ratio-4 (comp_cache +
    idx_cache) otherwise — the driver passes the layer's cache slot names and
    its current group count n."""
    hc = blk.hc
    hidden = blk.attn.wq_a.scales.shape[1] * blk.attn.wq_a.group_size
    items = []
    items += pl.hc_pre(hin, blk.hc_attn_fn.reshape(-1), blk.hc_attn_scale,
                       blk.hc_attn_base, "hx", "post", "comb", hc, hidden,
                       blk.iters, blk.eps, blk.hc_eps)
    items.append(pl.rms(blk.attn_norm.weight, "hx", "xn", hidden, blk.eps))
    items += plan_attention(pl, blk.attn, "xn", ring, "attn_out", ioff_off,
                            comp_cache=comp_cache, ncomp=ncomp, n=n, idx_cache=idx_cache)
    items.append(pl.hc_post("attn_out", hin, "post", "comb", "h1", hc, hidden))
    items += pl.hc_pre("h1", blk.hc_ffn_fn.reshape(-1), blk.hc_ffn_scale,
                       blk.hc_ffn_base, "hx", "post", "comb", hc, hidden,
                       blk.iters, blk.eps, blk.hc_eps)
    items.append(pl.rms(blk.ffn_norm.weight, "hx", "xn", hidden, blk.eps))
    items += plan_moe(pl, blk.ffn, "xn", "moe_out", topk, rscale, limit, tok_off)
    items.append(pl.hc_post2("y_routed", "shared", "h1", "post", "comb", hout, hc, hidden))
    return items
