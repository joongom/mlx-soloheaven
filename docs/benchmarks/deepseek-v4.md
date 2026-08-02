# DeepSeek-V4-Flash-0731 — measurement ledger

Machine: Apple M1 Ultra, 128 GiB. All decode numbers: **wired working set**
(`mx.set_wired_limit(max_recommended)`), greedy, ~10-token Korean prompt,
warm average (first 4-6 tokens excluded). Un-wired numbers are invalid —
see the 0.53 tok/s entry below for why that rule exists.

Companion narrative: `docs/specs/deepseek-v4-mlx-port.md` (design, evidence,
decisions). This file is the raw numbers + how to reproduce them.

## 1. Quality

### 1.1 Teacher-forced perplexity (`validate_deepseek_v4.py ppl`)

Probes: fixed Korean / English / Python-code paragraphs (in the script).
Korean is the canary — expert-quantization damage hits it first.

| build | ko | en | code | ALL |
|---|---|---|---|---|
| 2bit-mixed (mx.quantize min/max scales) | 17.91 | 9.01 | 1.55 | 7.11 |
| **2bit-search (error-searched scales, shipped)** | **6.60** | **4.12** | **1.46** | **3.69** |

Same recipe, same 94.5 GB size — the ONLY delta is per-group scale
selection (`quantize_search` in `convert_deepseek_v4.py`).

Generation, greedy, chat template (`validate_deepseek_v4.py smoke`):

| prompt | min/max build | search build |
|---|---|---|
| 안녕하세요. | `안ulumus! …支持的可以支持的…` | `안녕하세요! 무엇을 도와드릴까요? …` |
| 대한민국의 수도는? | `**서ulum** (서울-ulo…)` | `**서울**입니다. 서울은 대한민국의 정치…` |

### 1.2 Quantization frontier (expert-block output relative error)

Measured as the output error of a real routed-expert FFN
(`layers.20.experts.3`, clipped SwiGLU, realistic post-RMSNorm input),
NOT weight-space error:

| experts config | out rel err | total build | fits 128 GiB? |
|---|---|---|---|
| 2b/gs128 min/max | 0.733 | 79.9 GiB | yes |
| 2b/gs64 min/max | 0.671 | 88.0 GiB | yes (2.3 GiB free live) |
| **2b/gs64 scale-search** | **0.543** | **88.0 GiB** | **yes — shipped** |
| 2b/gs32 min/max | 0.611 | 104.1 GiB | no |
| 3b/gs128 | 0.400 | 112.2 GiB | no |

Closed escapes (measured, do not re-derive):
* AWQ-style scale folding: 0.671 → 0.672–0.796 (no help — the source is
  already fp4 with per-32 scales; outlier structure pre-flattened).
* Mixed precision (w2↑3b only): 0.671 → 0.651 for +2.8 GB — error is
  dominated by w1/w3.
* MLX has no sub-3-bit mode besides affine; group sizes only 32/64/128.

### 1.3 Conversion integrity audit (vs source checkpoint)

| check | result |
|---|---|
| 8-bit attention (wq_a/wkv/wo_b) | cos 0.99997, rel err 0.7% |
| wo_a grouped reshape [8,1024,4096] | cos 0.99997 |
| attn_sink / ape / hc_* / gate.weight / tid2eid | bit-exact |
| expert stacking order (ours[e] vs src[e]) | diag cos 0.913, off-diag 0.0002 |

### 1.4 ds4 oracle agreement

Same raw token ids (ds4 needs `--raw`; its one-shot mode silently applies a
chat template — a mismatched compare reads as chance, KL 13.6):

* single position "The capital of France is": top-1 both " Paris", KL 0.18.
* teacher-forced over ds4's 32-token greedy continuation: 27/32 top-1,
  31/32 in-top-5 (min/max build); 23/32, 30/32 (search build).
  **Deprecated as a ranking metric**: 32 positions is noise and it measures
  similarity to another 2-bit build, not quality — it preferred the build
  that ppl and generation both rank worse. Use `ppl`.

## 2. Decode speed (chronological; every attempt, including failures)

Reference target, measured same day, same machine:
**ds4: decode 27.34 tok/s (36.6 ms), prefill 34.82 tok/s** — and its log
prints `using GPU graph generation` (command-buffer replay runtime).

Our prefill: **80.9 tok/s** (256 tokens, one chunk) — 2.3x FASTER than ds4.
That inversion is the standing diagnosis: arithmetic fine, per-token
execution overhead is the gap.

| step | tok/s | ms/tok | verdict |
|---|---|---|---|
| harness WITHOUT wired limit | 0.53 | 1892 | INVALID — page faults dominate; masked every ablation |
| baseline, wired | 8.92 | 112.1 | true starting point (matches server feel) |
| ablation: sinkhorn 20→1 | 13.64 | 73.3 | HC path = 39 ms located |
| HC path via mx.compile | 11.46 | 87.3 | **+25 ms recovered — kept** |
| + fused Metal attention kernel (`dsv4_sparse_decode`) | 11.72 | 85.4 | +1.9 ms — kept |
| + kernel input caching (lru params/scale) | 11.95 | 83.7 | +1.7 ms — kept |
| whole-layer mx.compile (functional cache state) | 11.72 | 85.3 | **no change — launch-tax theory refuted (1)**; kept for structure |
| stacked x-projections (7 matmuls → 1) | 11.61 | 86.1 | **no change — refuted (2)**; kept (harmless) |
| HC hand Metal kernel (single dispatch) | 7.53 | 132.8 | **REGRESSION — reverted.** One threadgroup starves the chip that the library GEMV saturates |

Component ablation of the 82.4 ms baseline (removing compute+dispatch):

| component | Δms |
|---|---|
| CPU graph build | 11.0 (overlapped by async_eval in serving) |
| routed MoE (gather_qmm, batch 1) | 14.0 — vs ~3 ms of pure bandwidth |
| attention gather/softmax | 10.7 |
| compressors | 5.4 |
| rope | 3.3 |
| HC (after fusion) | ~14 |
| base quantized matmuls | ~9 (near bandwidth) |

### MLX 0.32.0 upgrade regression check (2026-08-02, Stage 0)

| metric | 0.31.2 | 0.32.0 |
|---|---|---|
| ppl ALL (ko/en/code) | 3.69 (6.60/4.12/1.46) | **3.65 (6.51/4.07/1.46)** |
| compiled decode | 86.1 ms | 89.0 ms (within bench noise) |
| suite | 1371 passed | 1371 passed |

Spike finding for the external-loop plan: `mlx.metallib` (154 MiB, in the
wheel) contains FULLY SPECIALIZED kernel entry names — e.g.
`qmv_bfloat16_t_gs_128_b_2`, `steel_gather_mm_rhs_nax_nn_bfloat16_...` —
with `MTL_FC_INIT` function constants, i.e. loadable by name via MTLLibrary
and instantiable with MTLFunctionConstantValues. Path A (reuse MLX's
compiled kernels from our own encoder) is viable at the naming level;
argument layouts to be read from the matching-version .metal sources (MIT).

### Campaign Stage 1a — fused batch-1 MoE kernels (2026-08-02)

`dsv4_moe_w13`/`dsv4_moe_w2`: routed-expert FFN as two dispatches/layer,
2-bit unpack in-register, full-chip grids (expert x row / output dim),
differential-tested against `mx.dequantize` reference incl. masked slots.

Result: **90.3 ms/token — no wall-time change** (89.0 baseline on 0.32).
Third substitution in a row (whole-layer compile, stacked projections, MoE
kernels) that is correctness-clean but does not move end-to-end time, while
component DELETION does (MoE ablation: -14 ms). Interpretation shifted to:
the decode is bounded by the length of the dependent kernel CHAIN (per-
boundary latency), not by any component's compute or bandwidth. Kernel kept:
it is the Stage 3 MoE kernel (external loop needs it regardless).

### The validated cost model (2026-08-02) — read this before optimizing

Two decisive measurements:

1. **Dual-stream test**: two independent decode chains in one eval take
   1.99x one chain (90.9 → 180.7 ms/pair). The GPU is BUSY, not idle —
   latency-bound hypothesis refuted (refutation #4). Corollary: concurrent
   sessions/speculative batches are nearly free in throughput terms only if
   the chain shortens; a second stream costs full price.
2. **Kernel micro-bench** (synthetic, no model load — iterate in seconds):
   our MoE kernels 0.30 ms/layer ≈ gather_qmm 0.25 ≈ both ~6x off the
   library qmv on identical bytes (0.047 ms). And take+qmv variants LOSE
   (0.68 ms) because they add ops: 10 ops x ~60 µs each.

**The model that fits every experiment so far**: each op in the dependent
chain costs ~50 µs of GPU time regardless of its size (setup + memory
round-trip dominates at [1,1,*] shapes); a token is ~1,800 ops ≈ 90 ms.
Substituting equal-op implementations changes nothing; deleting ops helps
linearly; adding ops hurts linearly. HC fusion (+25 ms for ~10k removed),
MoE kernel (±0 for net -6 ops), stacking (±0 for -190 tiny-share ops... at
~4-6 µs their share was ~1 ms), attention kernel (+3.6 ms for ~24 removed
x 43) are all consistent within factor-2.

**Therefore the campaign lever is CHAIN OP COUNT.** Priority by removable
ops per token: attention pre/post glue (rope x3, parameterless q-norm,
concats, casts ~20/layer x 43 -> absorb into the attention kernel),
compressor step non-matmul math (~12 x 61 calls -> one small-state kernel),
HC sinkhorn tail (keep the [24,16384] GEMV in the library, fuse the rest),
cast/astype hygiene across the step.

### Campaign Stage 1 close-out (2026-08-02): the in-MLX plateau is ~87 ms

| attempt | result |
|---|---|
| v2 fused attention core (in-kernel rope table, q-RMS, window indices, inverse rope, roped-kv emission; template-cast for bf16) | 89.0 → 87.3 ms — glue was already compile-fused |
| MoE K1/K2 vectorization (threadgroup x/h staging, unrolled unpack) | 0.303 → 0.296 ms/layer — call-chain floor, not kernel time |

Micro-truths that settle the accounting (kbench2):
* big library qmv at decode shapes are AT bandwidth (x0.8-1.2) — nothing
  to win there; head costs 0.94 ms/token (x1.2);
* a bare `astype(fp32)` on 16 KB costs 67 µs — the per-op floor made
  visible; `rms_norm` at 4.5 µs shows the floor is per-DEPENDENT-op
  (independent micro-loops pipeline, chains do not — the micro harness
  measures throughput, the decode chain pays latency);
* therefore the ~87 ms ≈ (big matmuls ~15 ms) + (attention/MoE/HC/comp
  kernels ~15 ms) + (~1,000+ chained small ops x ~40-70 µs).

**Stage-1 verdict**: inside MLX's per-op execution model the decode
plateaus around ~80-87 ms (12 tok/s). The 25 tok/s goal requires the
external command-buffer loop (Stage 3), where the per-op floor does not
exist — ds4 runs its whole step in 36 ms. Every kernel built in Stage 1
(attention core, MoE pair, compressor step) is exactly what the external
loop encodes, so none of this work is lost.

### Campaign Stage 3a — replay-loop preconditions ALL PROVEN (2026-08-02)

`src/mlx_soloheaven/native/dsv4_replay_spike.py` (pure ctypes/objc, run it):

1. `mx.array.__dlpack_device__() == (8, 0)` — kDLMetal; the capsule's data
   pointer IS the live `id<MTLBuffer>` (verified: `[buf contents]` reads the
   array's actual values; class AGXG13XFamilyBuffer).
2. `mlx.metallib` loads via `newLibraryWithURL:`; the fully specialized
   entries exist for OUR quantizations (`affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0`,
   `..._b_2`, `gather_qmv_*`, steel gemm) and qmv needs NO function constants.
3. Dispatch spec extracted from mlx v0.32.0 `quantized.cpp`: buffers
   0=w 1=scales 2=biases 3=x 4=y, bytes 5=K 6=N (int32), threadgroups
   (1, ceil(N/8), 1) x (32, 2, 1); `qmv_fast` requires N%8==0 && K%512==0.
4. External command buffer dispatching that kernel on MLX-owned DLPack
   buffers returns **max abs diff 0.0** vs `mx.quantized_matmul` (bf16
   scales — T-typed kernels need scales/biases in T; fp32 scales feed NaN).
5. `mx.synchronize()` before external submission is the ordering contract.

With Stage 2's session-stable cache buffers, everything the Stage 3b
runtime needs is now demonstrated: encode the whole 43-layer step once
against fixed MLX-owned buffers (library kernels for matmuls + our custom
kernels for attention/MoE/compressor), then per token write (token_id,
offset) into a small uniform buffer and re-commit the prebuilt command
buffer(s). The per-op floor that caps MLX at ~87 ms does not exist there.

Standing conclusions:
* Per-launch overhead is ~4 µs (39 ms / ~10k launches) — the remaining ~1k
  launches cost ~3 ms. The 85 ms is a SUM of medium inefficiencies, not one
  dispatch tax. Fewer dispatches is NOT a goal when it costs parallelism.
* Upstream survey (2026-08-02, MLX 0.32.0): no Metal graph replay exists or
  is planned; Metal already batches encoding per stream; `qmv_wide` helps
  batch 2-8 only; our gather_qmm batch-1 finding is UNREPORTED upstream.
  Official hybrid paths: multi-kernel extension Primitive (days, modest) and
  0.32 DLPack zero-copy for an external ds4-style decode loop (weeks, the
  only mapped road to parity). Details in the spec.

## 3. Reproduce

```bash
# quality
DSV4_MODEL=~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-search \
  .venv/bin/python validate_deepseek_v4.py ppl     # or: smoke / logits / compare / agree
# ds4 side (from ~/workspace/numenore/ds4; --raw is required for comparisons)
./ds4 --raw -p "PROMPT" --dump-logits ref.json -n 1 --temp 0
./ds4 -p "PROMPT" -n 64 --temp 0                   # prints prefill/generation t/s
# decode bench pattern (wired! nothing else loaded!)
python - <<'PY'
import mlx.core as mx, time
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
# load via validate_deepseek_v4.load(), prefill ~10 tokens, time 24+ greedy
# decode steps with mx.eval per token, discard the first 4-6 (traces/JIT)
PY
```

Rules this ledger follows are in `docs/DOCUMENTATION.md`.
