# Porting DeepSeek-V4 to MLX (`mlx_lm/models/deepseek_v4.py`)

**Status**: ALL 9 STEPS DONE — DeepSeek-V4 runs inside our engine and serves
over both APIs. Verified live: multi-turn prefix reuse (`cache=hit
cached=33 new=8`), SSE streaming, native session API (carries a name across
turns), thinking-mode routing, per-request sampling. ds4 oracle on identical
raw-tokenized input: top-1 match, KL 0.18; teacher-forced over 32
continuation positions: top-1 27/32, ours-in-ds4-top5 31/32.

Generation quality was broken (Korean corrupted at the token level) until the
converter stopped using MLX's min/max scales — see "The quantization ceiling"
below. With the error-searched scales, teacher-forced perplexity is 3.69
overall / 6.60 Korean, down from 7.11 / 17.91, at identical size.
**Date**: 2026-08-01
**Goal**: run DeepSeek-V4-Flash *inside* our engine, so SoloHeaven's KV/prompt
machinery actually applies to it.

## Why the model must run inside our engine

Fronting an external OpenAI-compatible server (ds4-server, say) would make the
model *reachable*, but generation would happen over there, and whoever generates
owns the KV cache. Everything this project exists for — session prefix reuse,
compaction, branch/regenerate rollback, memory/disk budgets, cross-model cache
safety, the MTP/PLD speculative stack — would not apply to that model at all. A
pass-through is a delivery mechanism, not an integration.

So the model must run in our engine, and that means an MLX implementation.

## Why MLX is the right target (not vLLM directly)

vLLM's registry has `DeepseekV4ForCausalLM`, but on Apple Silicon vLLM runs as
**vLLM-Metal**, which its own docs describe as using "MLX as the compute
backend". Its repository contains **299 files and zero model implementations**
(`grep -i deepseek` → nothing). So vLLM-Metal can run exactly what mlx-lm can
run.

An `mlx_lm/models/deepseek_v4.py` therefore unlocks *both* SoloHeaven and
vLLM-Metal from one implementation. Porting into vLLM's CUDA path would help
neither, since this machine has no NVIDIA GPU.

## The port base: mlx-lm already has most of it

`mlx_lm/models/deepseek_v32.py` (654 lines) implements DeepSeek-V3.2 and is a
proper mlx-lm citizen — `__call__(x, mask, cache)`, `cache[0].update_and_fetch()`,
`make_cache()`. It already provides:

| Piece | Status in v32 |
|---|---|
| MLA (q_lora, kv_lora, qk_rope split) | ✅ |
| **DSA `Indexer`** (`index_topk`, `index_n_heads`, `index_head_dim`) | ✅ |
| MoE + `group_expert_select` + `noaux_tc` top-k | ✅ |
| YaRN rope scaling | ✅ |
| **fp8 block dequant in `sanitize()`** — reads DeepSeek's official fp8 | ✅ |
| `make_cache`, `shard`, `cast_predicate` | ✅ |

And its `ModelArgs` field names already match V4's `config.json`:
`index_head_dim`, `index_n_heads`, `index_topk`, `q_lora_rank`,
`qk_rope_head_dim`, `n_routed_experts`, `n_shared_experts`,
`num_experts_per_tok`, `routed_scaling_factor`, `topk_method: noaux_tc`,
`norm_topk_prob`.

## The deltas V4 adds

Each was located in Vontra's MLX implementation
(`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX/deepseek_v4.py`, 1238 lines) — already
MLX, so this is translation, not derivation.

| Delta | Config keys | Size | Notes |
|---|---|---|---|
| `sqrtsoftplus` routing score | `scoring_func` | 1 line | `sqrt(softplus(s))` |
| **Hash routing** | `num_hash_layers: 3` | ~2 lines | layers < 3 route by a `tid2eid[vocab, topk]` **lookup table** — no computation at inference |
| **Hyper-Connections** | `hc_mult: 4`, `hc_eps`, `hc_sinkhorn_iters: 20` | moderate | `hc_head_reduce` (sigmoid-weighted reduction over hc residual streams) + `hc_split_sinkhorn` |
| **KV compression** | `compress_ratios[]` (0/4/128), `compress_rope_theta` | moderate | `Compressor`: learned gated pooling over N consecutive tokens; this is what carries long context |
| Grouped low-rank O proj | `o_lora_rank: 1024`, `o_groups: 8` | small | `einsum bsgd,grd->bsgr` |
| Attention sinks | `attn_sink` per head | small | |
| Sliding window | `sliding_window: 128` | small | ring buffer |
| Clipped SwiGLU | `swiglu_limit: 10.0` | small | |
| MTP / DSpark | `num_nextn_predict_layers` | optional | defer |

## The cache — the part that decides whether this is worth doing

Vontra's implementation keeps state **inside the modules**
(`Attention.__call__(x, start_pos)`, `self.kv_cache`, `compressor.kv_state`),
DeepSeek-reference style. That is single-conversation, non-trimmable, and
unusable for our session machinery. The port must move all state into explicit
cache objects passed per call, exactly as v32 does.

DeepSeek's own `inference/model.py` shows the layout, and it is a **single
preallocated buffer per layer** — the compressor writes into a *view* of it:

```python
kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
self.kv_cache = zeros(max_batch, kv_cache_size, head_dim)      # [b, 128 + seq/ratio, 512]
self.compressor.kv_cache = self.kv_cache[:, win:]              # view, not a copy
```

* slots `[0, win)` — the sliding ring, written at `start_pos % win`
* slots `[win, ...)` — compressed KV, appended every `ratio` tokens

Attention is then `sparse_attn(q, kv_cache, attn_sink, topk_idxs, scale)` where
`topk_idxs` concatenates window slots and compressed/Indexer-selected slots.

Sizes (head_dim 512, bf16):

| Context | ratio-4 layers (×21) | ratio-128 layers (×20) | total |
|---|---|---|---|
| 32K | 8320 slots → 8.5 MB | 384 slots → 0.4 MB | **~187 MB** |
| 1M | 262272 slots → 268 MB | 8320 slots → 8.5 MB | ~5.8 GB |

Three consequences, all favourable:

1. **Prefix reuse works.** The compressed region is append-only and the ring is a
   deterministic function of `offset`, so a session's whole state is
   `(buffer, offset)` — exactly the shape `state` get/set needs for our disk
   cache.
2. **Rollback is exactly solvable, unlike EXAONE.** The ring is only
   `128 × 512 × 2 B = 128 KB` per layer (**5.5 MB across all 43**), so it can be
   snapshotted *by value* before a speculative round and restored bit-exactly —
   no ring-headroom gate needed. EXAONE's ring is 4096 × 8 heads × 128 dim,
   which is why value-snapshot is not affordable there and trimming past the
   wrap is lossy.
3. **KV is cheap at normal context.** ~187 MB at 32K, versus 80+ GB of weights.

So `DeepSeekV4Cache` per layer: the buffer, `offset`, `state` get/set for
persistence, `trim(n)` (truncate compressed region + restore ring snapshot), and
a cheap snapshot/restore pair for speculation.

## Validation: ds4 is a numerical oracle

This is what makes the port de-riskable, and what the EXAONE MTP attempt lacked.
`ds4` exposes:

```
--dump-logits FILE      full next-token logits as JSON
--dump-logprobs FILE    greedy continuation top-logprobs
--decode-consistency N  compare N-token decode logits vs fresh prefill
--perplexity-file FILE  teacher-forced NLL
```

Verified working:

```
$ ds4 -p "The capital of France is" --dump-logits ref.json -n 1 --temp 0
  → {"vocab": 129280, "argmax_token": {"id": 671, "text": "The"},
     "argmax_logit": 32.64, "logits": [...129280 floats...]}
```

Caveat: ds4's logits come from the IQ2_XXS build (`quant_bits: 2`), so a
different quantization will not match bit-exactly. Compare on
quantization-robust measures — top-1 agreement across many positions, top-20
overlap, KL — not equality. Structural errors (wrong layer wiring) move those
metrics to chance; quantization noise does not.

Validation ladder:
1. strict weight load (every checkpoint tensor consumed)
2. tiny synthetic config — forward/decode shape + finiteness, cache round-trip
3. top-1 agreement vs ds4 logits over a few hundred positions
4. coherent generation in Korean/English

## Weights

MLX cannot read ds4's GGUF — `mx.load` fails with `gguf_tensor_to_f16 failed`
on IQ2_XXS/Q2_K. So convert from the official fp8 checkpoint
(`deepseek-ai/DeepSeek-V4-Flash-0731`, 166.9 GB), which v32's `sanitize()`
already knows how to dequantize.

Sizing (`ds4 --inspect` reports **284.33 B logical parameters**):

| Quant | Approx size | Fits 128 GB? |
|---|---|---|
| 4-bit | ~160 GB | ✗ |
| 3-bit | ~120 GB | borderline |
| 2-bit mixed | ~85 GB | ✓ (ds4's build is 80.76 GiB) |

Target ds4's asymmetric recipe rather than uniform 2-bit — routed experts low,
attention/shared-experts/output at 8-bit — which `mlx_lm.convert` supports via
`quant_predicate`. **Quality risk to measure, not assume**: ds4's build is
imatrix-calibrated on 1.5M routed-MoE tokens; MLX affine 2-bit has no imatrix
equivalent, so step 3 above is what decides whether this is usable.

**Disk**: 177 GB free; source 167 GB + output 85 GB = 252 GB does not fit. Use a
**staged conversion** — download shard → dequant + quantize its tensors → write →
delete shard — keeping peak at ~90 GB. Quantization is per-tensor independent, so
this is sound. (Do not delete the user's models to make room.)

## The quantization ceiling (why output quality is poor)

The conversion was audited tensor-by-tensor against the source checkpoint
before blaming quantization. Everything checkable is **correct**:

| check | result |
|---|---|
| 8-bit path (attention `wq_a`/`wkv`/`wo_b`) | cos 0.99997, rel err 0.7% |
| `wo_a` grouped reshape `[8,1024,4096]` | cos 0.99997 |
| unquantized (`attn_sink`, `ape`, `hc_*`, `gate.weight`) | bit-exact |
| `tid2eid` hash-routing table | bit-exact, range [0,255] |
| expert STACKING ORDER (ours[e] vs source[e]) | cos 0.913 on the diagonal, **0.0002 off-diagonal** |

So the damage is the 2-bit step itself. Measured on a real expert block
(`layers.20.experts.3`, output relative error of the full
clipped-SwiGLU FFN under a realistic post-RMSNorm input):

| experts | out rel err | total build | fits 128 GiB? |
|---|---|---|---|
| 2-bit gs128 | 0.733 | 79.9 GiB | yes |
| **2-bit gs64 (shipped)** | **0.671** | **88.0 GiB** | yes — measured 2.3 GiB free |
| 2-bit gs32 | 0.611 | 104.1 GiB | no |
| 3-bit gs128 | **0.400** | 112.2 GiB | no |

Three findings that close off the obvious escapes:

1. **The recipe is already ds4's.** Its GGUF is named
   `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix` — routed experts ~2-bit,
   attention/shared-experts/output 8-bit. Identical to ours. The ONLY
   difference is the quantizer: IQ2_XXS (codebook/lattice) with imatrix
   calibration versus MLX affine.
2. **MLX has no sub-3-bit alternative.** Modes are `affine`, `mxfp4`,
   `mxfp8`, `nvfp4` — the fp4 ones are 4-bit (≈156 GB of experts) — and
   affine group sizes are limited to 32/64/128.
3. **AWQ-style scale folding does not help here** (measured: 0.6712 →
   0.672–0.796 across α and both foldable scale points). The usual win comes
   from equalizing outlier input channels, but DeepSeek ships the experts
   ALREADY fp4-quantized with per-32-element scales, so that structure is
   pre-flattened and there is nothing left to exploit. Recorded so nobody
   spends a day re-deriving it.

Mixed precision was measured too and is not a way out: upgrading only `w2`
to 3-bit while dropping `w1`/`w3` to gs128 moves 0.671 → 0.651 for +2.8 GB.
The error is dominated by `w1`/`w3`, and raising those is what costs the
memory we do not have.

### What actually fixed it: choosing the scale instead of taking min/max

The bits were never the real constraint. `mx.quantize` sets a group's scale to
`(max-min)/(2^bits-1)`, so at 2 bits — four levels — one outlier in a
64-weight group stretches the scale and coarsens the other 63. llama.cpp's
Q2_K searches for the scale that minimizes reconstruction error instead, and
that is a large part of why ds4's build stays coherent.

`quantize_search()` in the converter does the same: a grid over range-shrink
factors, and for each candidate a least-squares re-fit of (scale, bias) given
the resulting level assignment. The output is ORDINARY MLX affine layout —
same packing, dtypes and shapes — so `mx.dequantize` and every quantized
kernel consume it unchanged. **No memory cost, no runtime cost.** Conversion
goes from 14 to 34 minutes.

Measured, same weights, same size:

| probe | min/max scales | searched scales |
|---|---|---|
| Korean | ppl 17.91 | **6.60** |
| English | ppl 9.01 | **4.12** |
| code | ppl 1.55 | **1.46** |
| **overall** | **ppl 7.11** | **ppl 3.69** |

Korean generation goes from `대한민국의 수도는 **서ulum** (서울-ulo…)` to
`대한민국의 수도는 **서울**입니다. 서울은 대한민국의 정치, 경제, 문화, 교육
중심지이며…`.

Two process notes, both earned the hard way:

* The first rebuild produced a BYTE-IDENTICAL model. The search had been wired
  into `emit_q`, but the routed experts — the only sub-8-bit tensors — were
  quantized by a separate loop calling `mx.quantize` directly. All quantization
  now goes through one `quantize_weight()` entry point, and `verify()` checks a
  VALUE (stored expert scales must differ from plain min/max), not just tensor
  names, because name checks passed happily on the identical build.
* Ranking builds by top-1 agreement with ds4 over a 32-token continuation is
  too noisy and too indirect to use: it preferred the WORSE build (27/32 vs
  23/32) while perplexity and generation both said the opposite. Use
  `validate_deepseek_v4.py ppl`.

**Remaining headroom**: 3-bit experts would reach 0.400 block error but need
112 GiB. The only way to get there on this machine is non-resident weights —
MoE activates ~6/256 experts per token, so mmap-backed streaming is plausible
(it is what ds4's `--ssd-streaming` does), but it needs MLX to keep weights
file-backed and unwired, which is unverified.

## Sequence

1. `ModelArgs` for V4 + weight-name mapping, from the real `config.json`
2. Attention: v32's MLA/Indexer + o_lora grouping, attn sink, sliding ring —
   with state in cache objects
3. `Compressor` + per-layer `compress_ratios` routing
4. Gate: sqrtsoftplus + hash-layer `tid2eid` lookup
5. Hyper-Connections
6. `make_cache()` returning the composite cache; `sanitize()` for fp8
7. Staged converter → 2-bit mixed MLX build
8. Validate against the ds4 oracle
9. Only then: wire into SoloHeaven's session cache (prefix reuse, ring headroom
   gate)

Steps 1-6 are the engineering; 7-8 are mechanical but long-running. This is a
multi-session task, not an afternoon.
