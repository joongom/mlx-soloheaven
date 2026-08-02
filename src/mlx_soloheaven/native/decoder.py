"""NativeDecoder — drives the external Metal replay loop for one decoded token.

Builds a session buffer table from a loaded DeepSeek-V4 Model (all weights +
per-layer ring/compressor/indexer cache buffers at fixed capacity) plus a
scratch pool, and per token re-encodes the full plan (embed + N blocks +
head) with the current offset and per-layer group counts baked in, commits,
and returns the logits. Re-encoding costs ~0.65 ms/1500 dispatches (measured),
so per-token-varying buffer offsets need no dynamic in-kernel indexing.

Correctness of the plan builders is verified in tests/test_dsv4_native.py
(per-layer-type plans + full dense-model logits). This class is the wiring
that runs them on a real model. Opt-in via SOLOHEAVEN_DSV4_NATIVE=1.
"""

from __future__ import annotations

import struct

import mlx.core as mx

from mlx_soloheaven.native import plan as P
from mlx_soloheaven.native import runtime as rt


class NativeDecoder:
    def __init__(self, model, max_context: int = 32768):
        self.model = model
        a = model.args
        self.rt = rt.Runtime()
        rt.load_custom_kernels(self.rt)
        self.hc, self.hidden, self.D = a.hc_mult, a.hidden_size, a.head_dim
        self.win, self.vocab = a.sliding_window, a.vocab_size
        self.topk, self.rscale, self.limit = (a.num_experts_per_tok,
                                              a.routed_scaling_factor, a.swiglu_limit)
        self.n_layers = a.num_hidden_layers
        self._cap = self._round_cap(max_context)
        self.offset = 0
        # Plan cache: the built dispatch list depends only on each layer's
        # completed-group count `n` (baked buffer row offsets) and the
        # compressor double-buffer parity `par` (which state buffer is
        # read/written). The token id and running offset are the ONLY
        # per-token-varying values, and both live at fixed positions in the
        # const blob (setBytes) — patched in place per token. `n` advances at
        # most every ratio tokens; `par` flips every token but has 2 states.
        # So we keep <=2 built plans for the current n and rebuild only when n
        # advances, turning the ~10 ms/token Python plan build into a patch.
        self._plan_cache: dict = {}
        self._plan_n: tuple | None = None
        self._tok_off = 0
        self._ioff_off = 0

        self.table = rt.BufferTable()
        self.S: dict[str, int] = {}
        self._arrays: dict[str, mx.array] = {}
        self._alloc_scratch()
        self._logits = self._reg("logits", mx.zeros((self.vocab,), mx.bfloat16))
        self._layers = [self._alloc_layer(i) for i in range(self.n_layers)]

    @staticmethod
    def _round_cap(ctx: int, ratio_min: int = 4, growth: int = 256) -> int:
        need = ctx // ratio_min
        return max(growth, ((need + growth - 1) // growth) * growth)

    def _reg(self, name: str, arr: mx.array) -> int:
        mx.eval(arr)
        slot = self.table.add(arr)
        self.S[name] = slot
        self._arrays[name] = arr
        return slot

    def _z(self, n, dt=mx.bfloat16):
        return mx.zeros((n,), dtype=dt)

    def _ninf(self, n):
        return mx.full((n,), -mx.inf, mx.float32)

    def _alloc_scratch(self):
        a = self.model.args
        hc, hidden, D = self.hc, self.hidden, self.D
        NHD = a.num_attention_heads * D
        q_lora, inter, ihd, i_nh = (a.q_lora_rank, a.moe_intermediate_size,
                                    a.index_head_dim, a.index_n_heads)
        s = dict(
            ha=self._z(hc * hidden), hb=self._z(hc * hidden), hx=self._z(hidden),
            post=self._z(hc, mx.float32), comb=self._z(hc * hc, mx.float32), xn=self._z(hidden),
            xp0=self._z(q_lora), qr=self._z(q_lora), q_raw=self._z(NHD), xp1=self._z(D),
            kvn=self._z(D), acore=self._z(NHD), kv_roped=self._z(D), o_lora=self._z(NHD),
            attn_out=self._z(hidden), h1=self._z(hc * hidden), scores=self._z(self._cap, mx.float32),
            idx=mx.zeros((self.topk,), mx.int32), w=self._z(self.topk, mx.float32),
            hexp=self._z(self.topk * inter, mx.float32), y_routed=self._z(hidden, mx.float32),
            sg=self._z(inter), su=self._z(inter), sh=self._z(inter), shared=self._z(hidden),
            moe_out=self._z(hidden), cwkv=self._z(2 * D), cwgate=self._z(2 * D),
            i_ckv=self._z(2 * ihd), i_cwg=self._z(2 * ihd), iw=self._z(i_nh),
            iq=self._z(i_nh * ihd), cidx=mx.zeros((512,), mx.int32),
            headx=self._z(hidden), headn=self._z(hidden), dummy=self._z(D),
            dummy_idx=mx.full((1,), -1, mx.int32),
        )
        for k, v in s.items():
            self._reg(k, v)

    def _alloc_layer(self, i: int) -> dict:
        a = self.model.args
        ratio = a.layer_compress_ratio(i)
        D, ihd, p = self.D, a.index_head_dim, f"L{i}_"
        self._reg(p + "ring", self._z(self.win * D))
        lay = {"ratio": ratio, "n": 0, "par": 0, "ring": p + "ring"}
        if ratio:
            # Compressor state is [coff*ratio, coff*head_dim] = [rows, cd]
            # (CompressorState.reset), so the buffer is rows*cd. The dsv4_comp_step
            # kernel indexes rows = coff*ratio rows; sizing at ratio*cd (missing
            # the coff row factor) under-allocates the ratio-4 (coff=2) state and
            # the kernel reads/writes 1 group past the end into adjacent buffers.
            coff, cd = (2, 2 * D) if ratio == 4 else (1, D)
            st = coff * ratio * cd
            self._reg(p + "kv_a", self._z(st, mx.float32))
            self._reg(p + "sc_a", self._ninf(st))
            self._reg(p + "kv_b", self._z(st, mx.float32))
            self._reg(p + "sc_b", self._z(st, mx.float32))
            self._reg(p + "buf", self._z(self._cap * D))
            lay["comp"] = p
            if ratio == 4:
                icoff, icd = 2, 2 * ihd
                ist = icoff * ratio * icd
                self._reg(p + "ikv_a", self._z(ist, mx.float32))
                self._reg(p + "isc_a", self._ninf(ist))
                self._reg(p + "ikv_b", self._z(ist, mx.float32))
                self._reg(p + "isc_b", self._z(ist, mx.float32))
                self._reg(p + "ibuf", self._z(self._cap * ihd))
                lay["idx"] = p
        return lay

    @staticmethod
    def _cache_dicts(lay: dict):
        """comp_cache / idx_cache with the double buffers oriented by parity:
        the kernel reads *_st (last token's output) and writes *_st2."""
        if "comp" not in lay:
            return None, None
        p, par = lay["comp"], lay["par"]
        a, b = ("a", "b") if par == 0 else ("b", "a")
        comp = {"kv_st": p + "kv_" + a, "sc_st": p + "sc_" + a,
                "kv_st2": p + "kv_" + b, "sc_st2": p + "sc_" + b, "buf": p + "buf"}
        idx = None
        if "idx" in lay:
            idx = {"kv_st": p + "ikv_" + a, "sc_st": p + "isc_" + a,
                   "kv_st2": p + "ikv_" + b, "sc_st2": p + "isc_" + b,
                   "buf": p + "ibuf", "i_buf": p + "ibuf"}
        return comp, idx

    def _build_plan(self, cb: P.ConstBlob, token: int):
        pl = P.Planner(self.rt, self.table, cb, self.S)
        tok_off, _ = cb.add("i", token)
        ioff_off, _ = cb.add("2i", self.offset, 0)
        self._tok_off, self._ioff_off = tok_off, ioff_off
        items = P.plan_embed(pl, self.model.embed, "ha", self.hc, self.hidden, tok_off)
        cur, nxt = "ha", "hb"
        for i, blk in enumerate(self.model.layers):
            lay = self._layers[i]
            comp, idx = self._cache_dicts(lay)
            items += P.plan_block(pl, blk, cur, lay["ring"], nxt, ioff_off,
                                  self.topk, self.rscale, self.limit,
                                  comp_cache=comp, idx_cache=idx,
                                  ncomp=lay["n"], n=lay["n"], tok_off=tok_off)
            cur, nxt = nxt, cur
        # after N layers the result is in `cur`; head reads it
        items += P.plan_head(pl, self.model, cur, "logits", self.hc, self.hidden, self.vocab)
        self._final = cur
        return items

    def decode(self, token: int) -> mx.array:
        """Replay one decode step for `token` at the current offset; returns
        the logits (bf16 [vocab]). Advances offset + per-layer state."""
        n_tuple = tuple(lay["n"] for lay in self._layers)
        if n_tuple != self._plan_n:
            self._plan_cache.clear()   # `n`-baked buffer offsets changed
            self._plan_n = n_tuple
        par_key = tuple(lay["par"] for lay in self._layers)
        cached = self._plan_cache.get(par_key)
        if cached is None:
            cb = P.ConstBlob()
            items = self._build_plan(cb, int(token))
            cached = (items, bytearray(cb.bytes()), self._tok_off, self._ioff_off)
            self._plan_cache[par_key] = cached
        items, blob, tok_off, ioff_off = cached
        struct.pack_into("<i", blob, tok_off, int(token))
        struct.pack_into("<i", blob, ioff_off, int(self.offset))
        self.rt.commit(items, self.table.ptrs, bytes(blob), wait=True)
        for lay in self._layers:
            if lay["ratio"]:
                if (self.offset + 1) % lay["ratio"] == 0:
                    lay["n"] += 1
                lay["par"] ^= 1
        self.offset += 1
        return self._arrays["logits"]

    def set_ring(self, layer: int, ring: mx.array) -> None:
        """Seed a layer's ring from an existing (prefilled) cache. Writes into
        the SAME MTLBuffer via its contents pointer — an mx in-place setitem
        (`dst[:] = ...`) allocates a NEW buffer, so the registered DLPack
        pointer would keep seeing the old one."""
        import ctypes

        name = self._layers[layer]["ring"]
        dst = self._arrays[name]
        src = ring.reshape(-1).astype(dst.dtype)
        mx.eval(src)
        mx.synchronize()
        src_bytes = bytes(memoryview(np_view(src)))
        ptr = self.rt.buffer_contents(self.table.ptrs[self.S[name]])
        ctypes.memmove(ptr, src_bytes, len(src_bytes))


def np_view(a: mx.array):
    """Raw little-endian bytes of an mx array (bf16 as uint16)."""
    import numpy as np

    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16))
    return np.array(a)
