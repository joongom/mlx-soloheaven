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

_shared_runtime = None


def shared_runtime():
    """One Runtime (with its compiled kernel PSOs) per process: rebinding a
    NativeDecoder to fresh cache arrays after every prefill must not pay the
    ~second of Metal source compilation again."""
    global _shared_runtime
    if _shared_runtime is None:
        r = rt.Runtime()
        rt.load_custom_kernels(r)
        _shared_runtime = r
    return _shared_runtime


class NativeDecoder:
    def __init__(self, model, max_context: int = 32768, barriers: bool = True,
                 cache=None, runtime=None):
        """``cache`` (a list of DeepSeekV4Cache) switches on BORROW mode: the
        per-layer ring/compressor/indexer state buffers are the cache's own
        arrays, registered zero-copy — every native write IS a cache write,
        so the engine's session machinery (state snapshots, prefix reuse of
        decoded tokens) reads current state with no sync step. The caller
        must not replace those arrays while this decoder is live; any prefill
        or state-restore does (mx setitem allocates new buffers), so the
        model integration rebinds — cheaply, when ``runtime`` (with its
        compiled kernels) is shared across rebuilds."""
        self.model = model
        self.cache = cache
        # barriers=False strips the per-dispatch buffer barrier — UNSAFE for
        # correctness (hazards), for throughput DIAGNOSIS only: it isolates how
        # much of the decode time is blanket-barrier serialization vs kernel
        # compute (see docs/benchmarks/deepseek-v4.md, real-model bench entry).
        self._barriers = barriers
        a = model.args
        if runtime is not None:
            self.rt = runtime
        else:
            self.rt = rt.Runtime()
            rt.load_custom_kernels(self.rt)
        self.hc, self.hidden, self.D = a.hc_mult, a.hidden_size, a.head_dim
        self.win, self.vocab = a.sliding_window, a.vocab_size
        self.topk, self.rscale, self.limit = (a.num_experts_per_tok,
                                              a.routed_scaling_factor, a.swiglu_limit)
        self.n_layers = a.num_hidden_layers
        self._cap = self._round_cap(max_context)
        self.offset = 0 if cache is None else int(cache[0].offset)
        # Plan cache: the built dispatch list depends only on each layer's
        # completed-group count `n` (baked buffer row offsets); compressor
        # state updates in place, so there is no buffer parity. The token id
        # and running offset are the ONLY per-token-varying values, and both
        # live at fixed positions in the const blob (setBytes) — patched in
        # place per token. `n` advances at most every ratio tokens, so we
        # keep ONE built plan for the current n and rebuild only when it
        # advances, turning the ~10 ms/token Python plan build into a patch.
        self._plan_cache: dict = {}
        self._plan_n: tuple | None = None
        self._tok_off = 0
        self._ioff_offs: list[int] = []

        self.table = rt.BufferTable()
        self.S: dict[str, int] = {}
        self._arrays: dict[str, mx.array] = {}
        self._alloc_scratch()
        self._logits = self._reg("logits", mx.zeros((self.vocab,), mx.bfloat16))
        self._layers = [self._alloc_layer(i) for i in range(self.n_layers)]
        if cache is not None:
            # Prefill wrote these buffers on MLX's queue; our queue must not
            # read them until those writes have landed.
            mx.synchronize()

    def _visible(self, lay: dict) -> int:
        """Compressed groups this token may attend to = completed groups plus
        the one this token itself completes (the reference's post-step `cn`)."""
        r = lay["ratio"]
        if not r:
            return 0
        return lay["n"] + (1 if (self.offset + 1) % r == 0 else 0)

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
            hc_mixes=self._z((2 + hc) * hc, mx.float32),
            # xall: the stacked x-projection output, sized for the widest layer
            # kind (ratio-4: q_lora + D + 2*(2D) + idx heads + 2*(2*idx_hd)).
            xall=self._z(q_lora + D + 4 * D + i_nh + 4 * ihd),
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
        c = self.cache[i] if self.cache is not None else None
        if c is not None:
            if c.ring is None:
                raise RuntimeError(f"layer {i}: cache has no ring — prefill first")
            self._reg(p + "ring", c.ring)
        else:
            self._reg(p + "ring", self._z(self.win * D))
        lay = {"ratio": ratio, "n": 0, "ring": p + "ring"}
        if ratio:
            # Compressor state is [coff*ratio, coff*head_dim] = [rows, cd]
            # (CompressorState.reset), so the buffer is rows*cd. The dsv4_comp_step
            # kernel indexes rows = coff*ratio rows; sizing at ratio*cd (missing
            # the coff row factor) under-allocates the ratio-4 (coff=2) state and
            # the kernel reads/writes 1 group past the end into adjacent buffers.
            # State updates IN PLACE (no double buffer): unwritten rows keep
            # their -inf scores, which is exactly the empty mask.
            coff, cd = (2, 2 * D) if ratio == 4 else (1, D)
            st = coff * ratio * cd
            if c is not None:
                # BORROW: the session cache's own state arrays, at the
                # session-stable capacity the compiled path also uses.
                from mlx_soloheaven.models.deepseek_v4 import _ensure_comp_capacity
                cs = c.comp
                if cs.kv_state is None:
                    cs.reset(1, ratio, coff, D)
                _ensure_comp_capacity(cs, self.model.layers[i].attn.compressor,
                                      mx.bfloat16)
                self._reg(p + "kv_a", cs.kv_state)
                self._reg(p + "sc_a", cs.score_state)
                self._reg(p + "buf", cs.cache)
                lay["n"] = int(cs.n)
            else:
                self._reg(p + "kv_a", self._z(st, mx.float32))
                self._reg(p + "sc_a", self._ninf(st))
                self._reg(p + "buf", self._z(self._cap * D))
            lay["comp"] = p
            if ratio == 4:
                icoff, icd = 2, 2 * ihd
                ist = icoff * ratio * icd
                if c is not None:
                    from mlx_soloheaven.models.deepseek_v4 import _ensure_comp_capacity
                    ics = c.idx
                    if ics.kv_state is None:
                        ics.reset(1, ratio, icoff, ihd)
                    _ensure_comp_capacity(
                        ics, self.model.layers[i].attn.indexer.compressor,
                        mx.bfloat16)
                    self._reg(p + "ikv_a", ics.kv_state)
                    self._reg(p + "isc_a", ics.score_state)
                    self._reg(p + "ibuf", ics.cache)
                    if int(ics.n) != lay["n"]:
                        raise RuntimeError(
                            f"layer {i}: comp n {lay['n']} != idx n {int(ics.n)}")
                else:
                    self._reg(p + "ikv_a", self._z(ist, mx.float32))
                    self._reg(p + "isc_a", self._ninf(ist))
                    self._reg(p + "ibuf", self._z(self._cap * ihd))
                lay["idx"] = p
        return lay

    @staticmethod
    def _cache_dicts(lay: dict):
        """comp_cache / idx_cache; state updates in place, so there is one
        buffer per state and no parity orientation."""
        if "comp" not in lay:
            return None, None
        p = lay["comp"]
        comp = {"kv_st": p + "kv_a", "sc_st": p + "sc_a", "buf": p + "buf"}
        idx = None
        if "idx" in lay:
            idx = {"kv_st": p + "ikv_a", "sc_st": p + "isc_a",
                   "buf": p + "ibuf", "i_buf": p + "ibuf"}
        return comp, idx

    def _build_plan(self, cb: P.ConstBlob, token: int):
        pl = P.Planner(self.rt, self.table, cb, self.S)
        tok_off, _ = cb.add("i", token)
        self._tok_off = tok_off
        items = P.plan_embed(pl, self.model.embed, "ha", self.hc, self.hidden, tok_off)
        cur, nxt = "ha", "hb"
        # PER-LAYER ioff blobs: the kernels read [offset, ncomp] as one
        # 8-byte constant, and ncomp (= the layer's completed-group count,
        # plan-build constant) differs per layer. A single shared blob baked
        # ncomp=0 for everyone, which silently masked the ENTIRE compressed
        # region (and the indexer's n2) on the replay path — invisible below
        # offset 128 where the ring covers everything. Only the offset half
        # is per-token; decode() patches it into every blob.
        self._ioff_offs = []
        for i, blk in enumerate(self.model.layers):
            lay = self._layers[i]
            comp, idx = self._cache_dicts(lay)
            # ioff[1] is the VISIBLE compressed-group count: the reference
            # attends to the compressor's POST-step count (`ncomp = cn`), so
            # the group this token itself completes is visible. It varies per
            # token, so it is patched in decode(); only the structural KC
            # bound below is baked.
            lioff, _ = cb.add("2i", self.offset, self._visible(lay))
            self._ioff_offs.append(lioff)
            items += P.plan_block(pl, blk, cur, lay["ring"], nxt, lioff,
                                  self.topk, self.rscale, self.limit,
                                  comp_cache=comp, idx_cache=idx,
                                  ncomp=lay["n"] + (1 if lay["ratio"] else 0),
                                  n=lay["n"], tok_off=tok_off)
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
        cached = self._plan_cache.get(0)
        if cached is None:
            cb = P.ConstBlob()
            items = self._build_plan(cb, int(token))
            if not self._barriers:
                for it in items:
                    it.barrier = 0
            cached = (items, bytearray(cb.bytes()), self._tok_off,
                      tuple(self._ioff_offs))
            self._plan_cache[0] = cached
        items, blob, tok_off, ioff_offs = cached
        struct.pack_into("<i", blob, tok_off, int(token))
        for off, lay in zip(ioff_offs, self._layers):
            struct.pack_into("<2i", blob, off, int(self.offset),
                             self._visible(lay))
        self.rt.commit(items, self.table.ptrs, bytes(blob), wait=True)
        for lay in self._layers:
            if lay["ratio"]:
                if (self.offset + 1) % lay["ratio"] == 0:
                    lay["n"] += 1
        if self.cache is not None:
            # Mirror the step onto the borrowed session caches: the arrays are
            # already current (written in place), only the Python bookkeeping
            # (offset, completed-group counts) needs to advance with us.
            for lay, c in zip(self._layers, self.cache):
                c.offset += 1
                if lay["ratio"]:
                    c.comp.n = lay["n"]
                    if c.idx is not None:
                        c.idx.n = lay["n"]
        self.offset += 1
        return self._arrays["logits"]

    def profile_kernels(self, token: int = 5, iters: int = 20) -> list:
        """Per-kernel-type GPU time for one token's plan. Groups the built plan
        by pipeline and times committing each group in isolation (the kernels
        still run; the data is stale, but the timing is what we want). Returns
        [(kernel_name, count, total_ms_per_token)] sorted slowest first — the map
        of where the decode time goes, to pick fusion targets."""
        import time

        cb = P.ConstBlob()
        items = self._build_plan(cb, int(token))
        blob = bytes(cb.bytes())
        pso_name = {i: n for n, i in self.rt._pipelines.items()}
        groups: dict[int, list] = {}
        for it in items:
            groups.setdefault(it.pso, []).append(it)
        out = []
        for pso, its in groups.items():
            for _ in range(3):
                self.rt.commit(its, self.table.ptrs, blob, wait=True)
            t0 = time.perf_counter()
            for _ in range(iters):
                self.rt.commit(its, self.table.ptrs, blob, wait=True)
            ms = (time.perf_counter() - t0) / iters * 1e3
            out.append((pso_name.get(pso, f"pso{pso}"), len(its), ms))
        out.sort(key=lambda r: -r[2])
        return out

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
