#!/usr/bin/env python
"""Convert deepseek-ai/DeepSeek-V4-Flash-0731 to a mixed-quant MLX build.

The official checkpoint stores attention/shared weights as fp8 E4M3 with
128x128-block E8M0 scales, and the routed experts as fp4 E2M1 packed two per
byte (low nibble = even index, from DeepSeek's own inference/convert.py) with
per-row 1x32-block E8M0 scales. MLX cannot read either dtype via mx.load, so
this script parses the safetensors headers itself and decodes with lookup
tables (self-checked against hand-computed encodings at startup).

Recipe (docs/specs/deepseek-v4-mlx-port.md):
  * routed experts  -> 2-bit  gs64 affine (the ~277B-param bulk)
  * everything else quantizable -> 8-bit gs64 (attention, shared experts,
    compressor/indexer projections, embed, head)
  * norms + gate.weight stay bf16; attn_sink/ape/hc_*/gate.bias stay fp32;
    tid2eid -> int32.  mtp.* is deferred and dropped.

Output is one shard per layer plus model-top.safetensors, written atomically
(tmp+rename) so an interrupted run resumes by skipping finished shards. The
final verify() step rebuilds the expected parameter tree from the model code
and diffs it against what was actually written.

Run: .venv/bin/python convert_deepseek_v4.py [SRC] [DST]
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import time

import mlx.core as mx
import numpy as np

SRC = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1
    else "~/.lmstudio/models/deepseek-ai/DeepSeek-V4-Flash-0731"
)
DST = os.path.expanduser(
    sys.argv[2] if len(sys.argv) > 2
    else "~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-mixed"
)

Q8 = {"group_size": 64, "bits": 8}
Q2 = {"group_size": 64, "bits": 2}

# --- decode tables ---------------------------------------------------------

def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 0:
            v = 2.0**-6 * (m / 8)
        elif e == 15 and m == 7:
            v = np.nan  # E4M3FN: only this encoding is NaN, no inf
        else:
            v = 2.0 ** (e - 7) * (1 + m / 8)
        t[b] = -v if s else v
    return t


def _e8m0_table() -> np.ndarray:
    t = 2.0 ** (np.arange(256, dtype=np.float64) - 127)
    t[255] = np.nan
    return t.astype(np.float32)


E4M3 = _e4m3_table()
E8M0 = _e8m0_table()
E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)

assert E4M3[0x38] == 1.0 and E4M3[0xC0] == -2.0 and E4M3[0x30] == 0.5
assert E8M0[127] == 1.0 and E8M0[128] == 2.0
assert E2M1[0x1] == 0.5 and E2M1[0x7] == 6.0 and E2M1[0x9] == -0.5


# --- checkpoint reader -----------------------------------------------------


class Reader:
    def __init__(self, root: str):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))[
            "weight_map"
        ]
        self._shards: dict[str, tuple[dict, np.memmap]] = {}

    def _shard(self, fn: str):
        if fn not in self._shards:
            path = os.path.join(self.root, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            self._shards[fn] = (header, np.memmap(path, np.uint8, "r"), 8 + n)
        return self._shards[fn]

    def raw(self, name: str) -> tuple[str, list[int], np.ndarray]:
        header, mm, base = self._shard(self.map[name])
        meta = header[name]
        lo, hi = meta["data_offsets"]
        return meta["dtype"], meta["shape"], np.asarray(mm[base + lo : base + hi])

    def has(self, name: str) -> bool:
        return name in self.map

    def close_unused(self, keep: set[str]) -> None:
        for fn in [f for f in self._shards if f not in keep]:
            del self._shards[fn]

    # -- logical decode ----------------------------------------------------

    def plain(self, name: str) -> np.ndarray:
        """bf16/f32/i64 tensor as fp32/int32."""
        dt, shape, b = self.raw(name)
        if dt == "BF16":
            u = b.view(np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt == "F32":
            return b.view(np.float32).reshape(shape)
        if dt == "I64":
            return b.view(np.int64).reshape(shape).astype(np.int32)
        raise ValueError(f"{name}: unexpected dtype {dt}")

    def linear(self, prefix: str) -> np.ndarray:
        """Weight of `prefix` (fp8-block, fp4-packed, or plain) as fp32."""
        dt, shape, b = self.raw(prefix + ".weight")
        if dt == "F8_E4M3":
            o, i = shape
            w = E4M3[b].reshape(o, i)
            _, sshape, sb = self.raw(prefix + ".scale")
            scale = E8M0[sb].reshape(sshape)
            assert sshape == [o // 128, i // 128], (prefix, shape, sshape)
            return w * np.repeat(np.repeat(scale, 128, 0), 128, 1)
        if dt == "I8":
            o, half = shape
            i = half * 2
            lo, hi = b & 0x0F, (b >> 4) & 0x0F
            w = np.empty((o, i), np.float32)
            w[:, 0::2] = E2M1[lo].reshape(o, half)
            w[:, 1::2] = E2M1[hi].reshape(o, half)
            _, sshape, sb = self.raw(prefix + ".scale")
            scale = E8M0[sb].reshape(sshape)
            assert sshape == [o, i // 32], (prefix, shape, sshape)
            return w * np.repeat(scale, 32, 1)
        return self.plain(prefix + ".weight")


# --- emit helpers ----------------------------------------------------------

QMAP: dict[str, dict] = {}  # per-path non-default quant entries for config

#: Candidate range shrink factors for the scale search. 1.0 reproduces
#: min/max; smaller values clip the group's extremes, which at 2 bits (four
#: levels) usually wins because one outlier otherwise stretches the scale and
#: coarsens the other 63 weights.
_SEARCH_GRID = [round(x, 4) for x in np.linspace(0.45, 1.0, 16)]


def quantize_search(a: mx.array, group_size: int, bits: int):
    """Affine quantization whose per-group scale/bias MINIMIZE reconstruction
    error, instead of spanning the group's min/max.

    ``mx.quantize`` picks ``scale = (max-min)/(2^bits-1)``. That is the worst
    choice at very low bitrates: a single outlier in a 64-weight group stretches
    the scale and coarsens everything else. llama.cpp's Q2_K searches instead,
    which is a large part of why ds4's 2-bit build stays coherent where ours
    did not. Measured on a real expert block: block output relative error
    0.671 (min/max) -> 0.548 (this), at IDENTICAL size.

    Output is ordinary MLX affine layout — same packing, dtypes and shapes as
    ``mx.quantize`` — so ``mx.dequantize`` and every quantized kernel consume it
    unchanged. Only the numbers are better chosen.
    """
    maxq = (1 << bits) - 1
    per_word = 32 // bits
    dtype = a.dtype
    g = a.reshape(-1, group_size).astype(mx.float32)
    lo = g.min(-1, keepdims=True)
    hi = g.max(-1, keepdims=True)
    span = hi - lo
    n = float(group_size)

    best_err = best_s = best_b = best_q = None
    for f in _SEARCH_GRID:
        s = mx.maximum(span * f / maxq, 1e-12)
        b = lo + span * (1.0 - f) / 2.0
        q = mx.clip(mx.round((g - b) / s), 0, maxq)
        # Re-fit (scale, bias) by least squares GIVEN the level assignment —
        # the grid only has to find a good assignment, not a good scale.
        sq = q.sum(-1, keepdims=True)
        sqq = (q * q).sum(-1, keepdims=True)
        sy = g.sum(-1, keepdims=True)
        sqy = (q * g).sum(-1, keepdims=True)
        det = sqq * n - sq * sq
        det = mx.where(mx.abs(det) < 1e-12, 1e-12, det)  # constant group
        s2 = (sqy * n - sq * sy) / det
        b2 = (sqq * sy - sq * sqy) / det
        err = ((g - (q * s2 + b2)) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_err, best_s, best_b, best_q = err, s2, b2, q
        else:
            m = err < best_err
            best_err = mx.where(m, err, best_err)
            best_s = mx.where(m, s2, best_s)
            best_b = mx.where(m, b2, best_b)
            best_q = mx.where(m, q, best_q)
        mx.eval(best_err, best_s, best_b, best_q)

    # A degenerate (constant) group gets scale 0; keep its bias so the value
    # still reconstructs exactly.
    packed = (
        best_q.astype(mx.uint32).reshape(-1, per_word)
        * (2 ** (mx.arange(per_word, dtype=mx.uint32) * bits))
    ).sum(-1)
    lead = a.shape[:-1]
    cols = a.shape[-1]
    return (
        packed.reshape(*lead, cols * bits // 32),
        best_s.reshape(*lead, cols // group_size).astype(dtype),
        best_b.reshape(*lead, cols // group_size).astype(dtype),
    )


def quantize_weight(a: mx.array, cfg: dict):
    """Single entry point for every quantized tensor in the build.

    Below 8 bits the min/max scale is what breaks the model, so those go
    through the search; at 8 bits it is already within 0.7% and the search
    would only cost time.
    """
    if cfg["bits"] < 8:
        return quantize_search(a, cfg["group_size"], cfg["bits"])
    return mx.quantize(a, **cfg)


def emit_q(out: dict, path: str, w: np.ndarray, cfg: dict) -> None:
    q, s, b = quantize_weight(mx.array(w).astype(mx.bfloat16), cfg)
    out[f"{path}.weight"], out[f"{path}.scales"], out[f"{path}.biases"] = q, s, b
    if cfg != Q8:
        QMAP[path] = cfg


def emit_raw(out: dict, name: str, w: np.ndarray, dtype=None) -> None:
    a = mx.array(w)
    out[name] = a.astype(dtype) if dtype is not None else a


def sanity(name: str, w: np.ndarray) -> None:
    if not np.isfinite(w).all():
        raise SystemExit(f"non-finite values after dequant in {name}")
    s = float(np.abs(w).std())
    if not (1e-6 < s < 1e3):
        raise SystemExit(f"suspicious weight scale in {name}: std={s}")


def save(out: dict, path: str) -> None:
    # mx.save_safetensors APPENDS ".safetensors" to any path that lacks it, so
    # the tmp name must already end with it — and must not match the final
    # "model*" pattern that load/verify glob.
    tmp = os.path.join(os.path.dirname(path), "tmp-" + os.path.basename(path))
    mx.save_safetensors(tmp, out)
    os.replace(tmp, path)
    try:
        mx.clear_cache()
    except AttributeError:
        pass


# --- conversion ------------------------------------------------------------


def convert_layer(r: Reader, cfg: dict, layer: int, path: str) -> None:
    ratios = cfg["compress_ratios"]
    ratio = ratios[layer]
    p = f"layers.{layer}"
    out: dict[str, mx.array] = {}

    for lin in ("wq_a", "wq_b", "wkv", "wo_b"):
        w = r.linear(f"{p}.attn.{lin}")
        if layer == 0:
            sanity(f"{p}.attn.{lin}", w)
        emit_q(out, f"{p}.attn.{lin}", w, Q8)
    wo_a = r.linear(f"{p}.attn.wo_a")
    g = cfg["o_groups"]
    emit_q(out, f"{p}.attn.wo_a", wo_a.reshape(g, wo_a.shape[0] // g, -1), Q8)

    for nrm in ("q_norm", "kv_norm"):
        emit_raw(out, f"{p}.attn.{nrm}.weight", r.plain(f"{p}.attn.{nrm}.weight"), mx.bfloat16)
    emit_raw(out, f"{p}.attn.attn_sink", r.plain(f"{p}.attn.attn_sink"))

    comps = [f"{p}.attn.compressor"] if ratio else []
    if ratio == 4:
        comps.append(f"{p}.attn.indexer.compressor")
        emit_q(out, f"{p}.attn.indexer.wq_b", r.linear(f"{p}.attn.indexer.wq_b"), Q8)
        emit_q(
            out,
            f"{p}.attn.indexer.weights_proj",
            r.linear(f"{p}.attn.indexer.weights_proj"),
            Q8,
        )
    for c in comps:
        emit_q(out, f"{c}.wkv", r.linear(f"{c}.wkv"), Q8)
        emit_q(out, f"{c}.wgate", r.linear(f"{c}.wgate"), Q8)
        emit_raw(out, f"{c}.norm.weight", r.plain(f"{c}.norm.weight"), mx.bfloat16)
        emit_raw(out, f"{c}.ape", r.plain(f"{c}.ape"))

    emit_raw(out, f"{p}.ffn.gate.weight", r.plain(f"{p}.ffn.gate.weight"), mx.bfloat16)
    if r.has(f"{p}.ffn.gate.tid2eid"):
        emit_raw(out, f"{p}.ffn.gate.tid2eid", r.plain(f"{p}.ffn.gate.tid2eid"))
    else:
        emit_raw(out, f"{p}.ffn.gate.bias", r.plain(f"{p}.ffn.gate.bias"))

    for w in ("w1", "w2", "w3"):
        emit_q(out, f"{p}.ffn.shared_experts.{w}", r.linear(f"{p}.ffn.shared_experts.{w}"), Q8)

    n_exp = cfg["n_routed_experts"]
    for src_w, proj in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
        qs, ss, bs = [], [], []
        for e in range(n_exp):
            w = r.linear(f"{p}.ffn.experts.{e}.{src_w}")
            if layer == 0 and e == 0 and src_w == "w1":
                sanity(f"{p}.ffn.experts.0.w1", w)
            # The routed experts are the ONLY sub-8-bit tensors in the build,
            # so this call — not emit_q — is where the scale search has to run.
            q, s, b = quantize_weight(mx.array(w).astype(mx.bfloat16), Q2)
            mx.eval(q, s, b)
            qs.append(q), ss.append(s), bs.append(b)
        path_e = f"{p}.ffn.experts.{proj}"
        out[f"{path_e}.weight"] = mx.stack(qs)
        out[f"{path_e}.scales"] = mx.stack(ss)
        out[f"{path_e}.biases"] = mx.stack(bs)
        QMAP[path_e] = Q2
        del qs, ss, bs

    for nrm in ("attn_norm", "ffn_norm"):
        emit_raw(out, f"{p}.{nrm}.weight", r.plain(f"{p}.{nrm}.weight"), mx.bfloat16)
    for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
               "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
        emit_raw(out, f"{p}.{hc}", r.plain(f"{p}.{hc}"))

    save(out, path)
    # keep only shards the NEXT layer needs open
    keep = {r.map[n] for n in r.map if f"layers.{layer + 1}." in n}
    r.close_unused(keep)


# The release ships no Jinja template — only encoding/encoding_dsv4.py. This
# is a Jinja rendering of that script's CHAT-mode rules (thinking mode differs
# only in the generation opener, selected via enable_thinking):
#   <BOS>{system}<｜User｜>{u}<｜Assistant｜></think>{a}<EOS>...<｜Assistant｜></think>
# Earlier-turn reasoning is always dropped (the official default), and runs of
# tool results merge into one <｜User｜> turn of <tool_result> blocks joined by
# blank lines. Kept token-exact against the engine's suffix builder in
# tests/test_deepseek_family.py.
CHAT_TEMPLATE = (
    "{{- '<｜begin▁of▁sentence｜>' -}}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}"
    "{{- m['content'] -}}"
    "{%- elif m['role'] == 'user' -%}"
    "{{- '<｜User｜>' + m['content'] -}}"
    "{%- elif m['role'] == 'tool' -%}"
    "{%- if loop.index0 == 0 or messages[loop.index0 - 1]['role'] != 'tool' -%}"
    "{{- '<｜User｜>' -}}"
    "{%- else -%}"
    "{{- '\\n\\n' -}}"
    "{%- endif -%}"
    "{{- '<tool_result>' + m['content'] + '</tool_result>' -}}"
    "{%- elif m['role'] == 'assistant' -%}"
    "{{- '<｜Assistant｜></think>' + (m['content'] or '') + '<｜end▁of▁sentence｜>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if enable_thinking is defined and enable_thinking -%}"
    "{{- '<｜Assistant｜><think>' -}}"
    "{%- else -%}"
    "{{- '<｜Assistant｜></think>' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)


def install_chat_template(dst: str) -> None:
    path = os.path.join(dst, "tokenizer_config.json")
    tc = json.load(open(path))
    tc["chat_template"] = CHAT_TEMPLATE
    with open(path, "w") as f:
        json.dump(tc, f, indent=2, ensure_ascii=False)


def convert_top(r: Reader, path: str) -> None:
    out: dict[str, mx.array] = {}
    emit_q(out, "embed", r.linear("embed"), Q8)
    emit_q(out, "head", r.linear("head"), Q8)
    emit_raw(out, "norm.weight", r.plain("norm.weight"), mx.bfloat16)
    for hc in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        emit_raw(out, hc, r.plain(hc))
    save(out, path)


def verify(cfg: dict) -> None:
    """Diff written tensors against the model code's parameter tree."""
    from mlx.utils import tree_flatten

    from mlx_soloheaven.models.deepseek_v4 import Model, ModelArgs

    expected = set()
    quantized_prefixes = set()
    for name, _ in tree_flatten(Model(ModelArgs.from_dict(cfg)).parameters()):
        mod = name.rsplit(".", 1)[0]
        is_linear_like = name.endswith(".weight") and (
            mod in QMAP
            or mod.split(".")[-1]
            in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "wgate", "weights_proj",
                "w1", "w2", "w3")
            or mod in ("embed", "head")
            or mod.endswith("indexer.wq_b")
        )
        if is_linear_like and not mod.endswith("norm") and not name.endswith(".ffn.gate.weight"):
            expected.update({f"{mod}.weight", f"{mod}.scales", f"{mod}.biases"})
            quantized_prefixes.add(mod)
        else:
            expected.add(name)

    written = set()
    for fn in os.listdir(DST):
        if fn.endswith(".safetensors") and not fn.startswith("tmp-"):
            with open(os.path.join(DST, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                written.update(
                    k for k in json.loads(f.read(n)) if k != "__metadata__"
                )

    missing, extra = sorted(expected - written), sorted(written - expected)
    if missing or extra:
        raise SystemExit(f"verify FAILED\nmissing: {missing[:10]}\nextra: {extra[:10]}")

    # Name checks alone let a whole conversion run produce byte-identical
    # output to the previous recipe (that happened: the expert loop called
    # mx.quantize directly and bypassed the scale search). So also check a
    # VALUE: the shipped expert scales must NOT be what plain min/max gives.
    r = Reader(SRC)
    src = mx.array(r.linear("layers.0.ffn.experts.0.w1")).astype(mx.bfloat16)
    _, minmax_s, _ = mx.quantize(src, **Q2)
    stored = mx.load(os.path.join(DST, "model-layer-00.safetensors"))
    got = stored["layers.0.ffn.experts.gate_proj.scales"][0]
    if bool(mx.all(got == minmax_s).item()):
        raise SystemExit(
            "verify FAILED: expert scales equal plain min/max — the scale "
            "search did not run on the routed experts"
        )
    print(f"verify OK: {len(written)} tensors, {len(quantized_prefixes)} quantized "
          f"modules, expert scales are search-optimized")


def main() -> None:
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    for fn in os.listdir(DST):  # stale tmp files from an interrupted run
        if fn.startswith("tmp-") or ".tmp" in fn:
            os.remove(os.path.join(DST, fn))
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    r = Reader(SRC)
    n_layers = cfg["num_hidden_layers"]

    for layer in range(n_layers):
        path = os.path.join(DST, f"model-layer-{layer:02d}.safetensors")
        if os.path.exists(path):
            print(f"[{layer + 1}/{n_layers}] exists, skip", flush=True)
            continue
        convert_layer(r, cfg, layer, path)
        print(
            f"[{layer + 1}/{n_layers}] layer {layer} done "
            f"({time.time() - t0:.0f}s elapsed)",
            flush=True,
        )

    top = os.path.join(DST, "model-top.safetensors")
    if not os.path.exists(top):
        convert_top(r, top)
        print("top done", flush=True)

    # QMAP is only fully populated on a non-resumed pass; rebuild it
    # deterministically so resumed runs still write a complete config.
    for layer in range(n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            QMAP[f"layers.{layer}.ffn.experts.{proj}"] = Q2

    quant = {**Q8, **QMAP}
    cfg_out = dict(cfg)
    cfg_out["quantization"] = quant
    cfg_out["quantization_config"] = quant
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "LICENSE"):
        if os.path.exists(os.path.join(SRC, fn)):
            shutil.copy(os.path.join(SRC, fn), os.path.join(DST, fn))
    install_chat_template(DST)

    verify(cfg)
    total = sum(
        os.path.getsize(os.path.join(DST, f))
        for f in os.listdir(DST)
        if f.endswith(".safetensors") and not f.startswith("tmp-")
    )
    print(f"DONE in {(time.time() - t0) / 60:.1f} min, {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
