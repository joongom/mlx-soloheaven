# Porting DeepSeek-V4 to MLX (`mlx_lm/models/deepseek_v4.py`)

**Status**: steps 1-8 DONE. Converted build (94.5 GB, experts 2-bit gs64,
rest 8-bit) loads and runs; ds4 oracle agreement on identical raw-tokenized
input: top-1 " Paris" match, KL 0.18 single-position; teacher-forced over 32
continuation positions: top-1 27/32, ours-in-ds4-top5 31/32, disagreements
all near-synonyms — structural errors would sit at chance. ds4's one-shot
mode applies a chat template (`--raw` disables it); a template-mismatched
compare reads as chance (KL 13.6) — align tokenization FIRST.
Remaining: step 9 (engine integration: dialect/template, continuation
prefill for prefix reuse, trim/rollback, launcher) and quality work — our
affine 2-bit trails ds4's imatrix-calibrated build in free generation.
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
