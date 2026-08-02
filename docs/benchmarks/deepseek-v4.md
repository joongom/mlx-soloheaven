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

### Stage 3b ladder step 1 — the replay loop is numerically viable

Chained dependent dispatches of a small qmv in a RAW external command
buffer: **13-15 µs/kernel** (vs the ~40-70 µs MLX-layer floor — the replay
loop saves ~4x per op). Projection: a ~600-800-kernel step x 13 µs + ~25-30
ms of genuine compute ≈ **35-45 ms/token ≈ 22-28 tok/s** — the target is
inside the envelope, consistent with ds4's 36 ms.

Encode-cost decision — RESOLVED (2026-08-02, native/encoder.m):
the C encode loop does **0.65 ms per 1500 dispatches (0.43 µs each)**, 25x
faster than python-ctypes's 16.6 ms and negligible against a ~35-45 ms
token. **Plain re-encode wins; ICB not needed.** Per-token varying scalars
still go in a uniform buffer (rewritten in-place in unified memory — proven
by `test_uniform_write_takes_effect_without_reencode`), so plan items stay
identical across tokens and only buffer *contents* change.

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

### Stage 3b ladder steps 8b–8c — full native decoder CORRECT + the dispatch floor (2026-08-02)

The `NativeDecoder` now replays a full multi-layer decode (dense + ratio-128 +
ratio-4) as one command buffer and is **bit-for-bit deterministic** and correct
(argmax in the compiled reference's top-3 when that reference is self-consistent;
the per-layer-type plan tests are the tight proof). Getting here surfaced two
out-of-bounds bugs — found with **`MTL_SHADER_VALIDATION=1`**, the tool that
cracked a nasty in-suite-only Heisenbug (full write-up:
`docs/benchmarks/deepseek-v4-native-debugging.md`):
* **Compressor state under-allocation (real, native-path bug).** `dsv4_comp_step`
  indexes the state as `[coff*ratio, coff*head_dim]` (`CompressorState.reset`),
  but `decoder.py`/tests allocated `ratio*cd` — missing the `coff` row factor, so
  every ratio-4 (`coff==2`) layer read/wrote one group past the end into adjacent
  buffers (allocation-dependent garbage). Fixed to `coff*ratio*cd`. Only the
  native runtime under-allocated; the compiled/eager model path (which sizes via
  `CompressorState.reset`) was always correct.
* **qmv `N%8` (test-config artifact).** `affine_qmv_fast_..._batch_0` needs
  `N%8==0`; `weights_proj` has `N=index_n_heads`, and the tests used `2`. The real
  model uses `64`, so this never affected production; tests now use `8`.

The residual full-model-test flake was traced to the model's **`mx.compile`'d
decode reference being non-deterministic under pytest memory pressure**
(`ref_self_max` up to ~2.0 on identical inputs), NOT the native path
(`nat_self_max==0` always). Open question for the server: is the compiled decode
non-deterministic in production too? (weights were confirmed identical).

**Plan cache (`decoder.py`).** The per-token plan build was ~10.8 ms of Python
(1306 `_PlanItem` structs). The dispatch list depends only on each layer's
completed-group count `n` and the compressor parity `par`; token id and offset
are the only per-token-varying values and live at fixed const-blob offsets
(patched in place). Caching by `(n, par)` and rebuilding only when `n` advances
makes most tokens a byte-patch. Verified bit-identical to a per-token rebuild
over 12 tokens (par flips + n increments).

**The dispatch floor (measured, tiny 43-layer model, hidden=256).** Per token:
Python plan build ~11 ms (now cached away), C encode **0.52 ms**, GPU replay of
1306 dispatches **~26 ms**. Barriers are **free**: same plan with all barriers vs
none timed 25.88 vs 26.03 ms — the 26 ms is the GPU command processor launching
1306 dispatches, ~20 µs each, NOT barrier serialization. So on the real model the
lever for ≥25 tok/s is **dispatch count** (kernel fusion — e.g. the 8 grouped
`wo_a` qmv → 1), not barriers or Python. Tiny-model µs/dispatch does not predict
the real model (whose large kernels are compute-dominated); a real-model number
is required — run `validate_deepseek_v4.py bench`.

### Stage 3b/3d — REAL-MODEL bench: the native replay is 7x SLOWER (2026-08-02) ⚠️

First real-88 GB-model decode bench (`validate_deepseek_v4.py bench 32`, machine
wired at load, 111 GiB reclaimable, nothing else resident, prefill 8 tokens):

| path | tok/s | ms/token |
|---|---|---|
| compiled (`mx.compile` decode) | **11.9** | 84.2 |
| native replay (`NativeDecoder`) | **1.6** | **611.5** |
| speedup | **0.14x** | — |

**This REFUTES the Stage 3b projection** (ladder step 1 guessed
"35-45 ms/token ≈ 22-28 tok/s" from the numerically-viable replay). The native
runtime is **7.3x slower** than the path it was meant to beat. Prediction failed;
recorded so we do not repeat the reasoning.

Why — barriers RULED OUT (measured, same load, `bench_barrier.py`):

| native variant | ms/token |
|---|---|
| WITH a buffer barrier before every dispatch | 611.1 |
| with ALL barriers forced off (`plan_item barrier=False`) | 610.1 |

**1.00x** — the blanket barriers cost nothing; the earlier tiny-model "barriers
are free" result DID transfer. So the 611 ms is the **serial sum of ~1500
dispatches' launch+compute latency**, ~0.4 ms each. That is ~8x MLX's own
~50 µs/op floor (the compiled path is at its floor: ~1800 ops × ~47 µs ≈ 84 ms).

The gap is therefore NOT barriers and NOT the CPU encode (0.52 ms). Two survivors:
* **No overlap.** A `computeCommandEncoder` is serial-dispatch; forcing barriers
  off does not enable concurrency (that needs `MTLDispatchTypeConcurrent`), so
  every dispatch pays its full launch latency in series. MLX keeps the GPU fed by
  streaming many small command buffers with `async_eval` (CPU queues ahead).
* **Slower custom kernels.** attn_core/comp_step/hc/gate use small grids
  (e.g. attn_core = one threadgroup per head = 64) that underfill the chip on the
  real dims, vs MLX's steel/gather_qmm. The MoE micro-bench had them ~on par, but
  the others are unprofiled on real dims.

Both point the same way as the validated cost model (§ above): the floor is
per-DISPATCH, so the only real lever is **dispatch count** — ~1500 is already
MLX-parity, and even at MLX's 50 µs/op that is ~75-90 ms (≈12 tok/s), NOT 40 ms
(25 tok/s). Reaching 25 tok/s needs ds4-scale FUSION (≪1000 dispatches), not an
external loop over the same op granularity.

(This run first crashed before timing: the real model's first `num_hash_layers`
(3) route experts by `tid2eid[token]` with no gate bias, which `plan_moe` didn't
handle. Fixed by adding `dsv4_gate_hash_k` — the native path is now correct on
every real layer type, which is why the throughput number above could be taken.)

Bottom line: the ≥25 tok/s goal is **NOT met** via the replay path as designed
(1.6 tok/s, worse than the 12 tok/s compiled baseline), and the barrier ablation
shows tinkering with the encoder won't fix it. The replay loop was premised on
the per-op floor being CPU-side and removable; the cost model already found it is
GPU-side, and this real-model number confirms it. **The external loop over the
same ~1500-op granularity cannot beat the compiled path** — the only lever is
dispatch-count reduction (fusion), which is a ds4-scale rewrite. Correctness of
the native path stands (bit-deterministic, matches the reference); this is purely
a throughput verdict. Decision point for the campaign, recorded for the user.

### Stage 3d — per-kernel profile → idx_topk fix: native 611 → 199 ms (2026-08-02)

A per-kernel-type profile (`bench` now prints it: each pipeline's dispatches
timed alone against a populated buffer set) split the 611 ms cleanly — it was
NOT spread across the ~1500 dispatches, it was **one kernel**:

| kernel | count | ms | share |
|---|---|---|---|
| `dsv4_idx_topk_k` (before fix) | 21 | **405** | **67%** |
| `dsv4_hc_pre_k` | 86 | 70 | 12% |
| `dsv4_gate_k` | 40 | 70 | 12% |
| everything else (~1330 dispatches) | — | ~66 | ~11% |

`dsv4_idx_topk_k` ran a **single-thread `O(cap*topk)` selection sort** (256×512 =
131k serial iterations/layer) even when only `n2` (~8) groups are visible. Fix:
when `n2 <= topk` every valid group is selected, so emit `0..n2-1` across all
threads (order is irrelevant — attn softmaxes the set). Result:

| path | tok/s | ms/token |
|---|---|---|
| compiled | 11.9 | 84.2 |
| native (before) | 1.6 | 611.5 |
| **native (after idx_topk fix)** | **5.0** | **199.1** |

idx_topk went 405 → 0.68 ms. **3.1x native speedup from one pathological-kernel
fix** — and it validates the profile-first method. New profile after the fix:
`hc_pre_k` **73 ms** and `gate_k` **71 ms** are now 72% of 199 ms. Both are the
single-threadgroup-starves-the-chip pattern (cf. the reverted HC hand-kernel at
132 ms) — the fix is to route the HC/gate GEMV through a multi-threadgroup
library `qmv` and keep only the sinkhorn/top-k tail custom. Native is still 2.4x
the compiled 84 ms, and 25 tok/s (40 ms) still needs fewer dispatches (fusion),
but the pathological floor is gone.

### Stage 3e — single-threadgroup kernels parallelized: native OVERTAKES compiled (2026-08-02)

Three fixes in the same shape ("one threadgroup / one thread cannot hide latency
or run serial loops — give the chip the whole grid"), each verified by diff test
then measured on the real model (`bench 24`, wired, `--force`):

| step | native ms/tok | tok/s | what changed |
|---|---|---|---|
| after idx_topk fix | 199-201 | 5.0 | (previous entry) |
| + gate split (score per-expert grid + tiny top-k) | 141.7 | 7.1 | gate 70.4 → 12.7 ms |
| + hc_pre split (`dsv4_hc_mix_k` grid=24 rows) + topk staging | 90.2 | 11.1 | hc_pre 72 → 16.3 ms |
| + PARALLEL argmax rounds in gate top-k | **80.2** | **12.5** | gate_topk 10.8 → 1.26 ms |

**Native now beats the compiled path (80.2 vs 84.5 ms, 1.05x)** — the replay
premise is redeemed after the kernel fixes; cumulative 611 → 80.2 = 7.6x.

Negative result worth keeping: STAGING the gate top-k's reads into threadgroup
memory barely helped (11.7 → 10.8 ms) — the cost was ONE GPU thread executing
topk×n_exp serial iterations, not where it read from. Parallel argmax rounds
(256-thread cooperative reduction per selected expert) fixed it (→ 1.26 ms).
Single-thread serial loops on GPU are ~100x slower than they look.

Current profile (80.2 ms; isolated-group times, sum ≈ 91 inflated by per-group
commit overhead):

| kernel | count | ms | lever |
|---|---|---|---|
| library qmv | 812 | 19.8 | dispatch COUNT: reuse the model's `_xstack` (7→1 x-projections/layer, −188), grouped-wo_a kernel (8→1, −301) |
| dsv4_attn_core | 43 | 16.3 | 64 threadgroups underfill; split score/value or widen grid |
| dsv4_hc_pre_k (tail) | 86 | 13.0 | serial sinkhorn on tid0 + launch; modest |
| dsv4_comp_step | 62 | 11.4 | single threadgroup; grid over d |
| moe w13+w2 | 86 | 12.5 | already full-chip; near bandwidth |

Target ≥25 tok/s = ≤40 ms: needs roughly qmv-count (−7) + attn_core (−8) +
comp_step (−6) + hc_pre tail (−5) + misc, i.e. most of the table — grind, but
every step so far landed where the profile pointed.

### Stage 3f — x-stack qmv + parallel Sinkhorn: 71.4 ms, 14.0 tok/s (2026-08-02)

Two more steps, same discipline (profile -> fix -> diff test -> real-model bench):

| step | native ms/tok | tok/s | what changed |
|---|---|---|---|
| + stacked x-projection (reuse the model's `_x_stack`: 7/4/2 -> 1 qmv/layer) | 78.0 | 12.8 | qmv 812 -> 624 dispatches, 19.8 -> 16.8 ms |
| + PARALLEL gates+Sinkhorn in hc_pre (rows/cols, bit-identical) | **71.4** | **14.0** | hc_pre 13.0 -> 7.35 ms |

Cumulative 611 -> 71.4 ms = 8.6x; native is 1.14x the compiled path. The
single-thread-serial-loop pathology has now been found and killed in THREE
kernels (gate top-k, idx top-k slow path context, hc_pre Sinkhorn) — check for
it FIRST in any slow custom kernel.

Remaining profile (71.4 ms): qmv 17.8 (624 dispatches — grouped wo_a 8->1 is the
next count lever, -301), attn_core 16.3 (64 threadgroups underfill; split
score/value or widen the grid), comp_step 11.8 (single threadgroup; grid over
d), moe 13.4 (near bandwidth), hc_pre 7.4. Target 40 ms needs roughly
attn_core + comp_step + qmv-count together.

### Stage 3g — threadgroup widening: 62.6 ms, 16.0 tok/s (2026-08-02)

The next three bottlenecks all fell to the same lever (widen the underfilled
threadgroup, ALL dispatch sites moved together):

| step | native ms/tok | tok/s | what changed |
|---|---|---|---|
| attn_core TG 128 -> 256 -> 512 | 69.0 | 14.5 | attn_core 16.3 -> 9.5 ms |
| comp_step TG 256 -> 1024 (+wsum[32]) | **62.6** | **16.0 (1.26x)** | comp_step 11.3 -> <6 ms |

Trap recorded: bumping a kernel's compile-time TG while ANY dispatch site still
launches the old thread count leaves the simdgroup-partial array (red[]/wsum[])
partially unwritten -> NaN. The first 512 attempt NaN'd from exactly this (a
test's hand-built _PlanItem and the mx.fast twin's dispatch weren't moved);
PSO maxTotalThreadsPerThreadgroup is 1024, not the limit. Check `pipeline
maxTotalThreadsPerThreadgroup` and grep EVERY dispatch site before widening.

Cumulative: 611 -> 62.6 ms = 9.8x. Remaining (62.6 ms): qmv 16.9 (624 —
grouped wo_a 8->1 is the count lever, -301 dispatches), moe 12.4 (near
bandwidth), attn_core 9.6, hc_pre 7.4, hc_post/mix/rms ~9. Target 40 ms:
needs the wo_a count cut plus one more of attn_core/hc tail.

### Stage 3h — grouped wo_a fusion: dispatch count DOWN, wall time FLAT ⚠️ (2026-08-02)

`dsv4_wo_a_k` fuses the o_groups (8) wo_a into one grouped 8-bit qmv (one
simdgroup per output row), so the 8 separate library qmv/attention become one:
**qmv dispatch count 624 -> 280** (-344). Correct — the dense/ratio-128/ratio-4
attention plan tests diff-verify it. But end to end it did **nothing**:

| | ms/token |
|---|---|
| before (TG-widened, Stage 3g) | 62.6 |
| after wo_a fusion | 62.6 |

**Prediction FAILED** (expected ~-10 ms from -301 dispatches; measured ~0, within
noise). Isolated-group timing: qmv 16.57 -> 11.54 ms (-5.0), but the new
`dsv4_wo_a_k` adds 3.68 ms → net -1.35 ms in the isolated profile, none of which
shows in wall time. Why: on this replay path per-launch overhead is already ~4 µs
(Stage 3a), so cutting 301 launches saves ~1 ms at most; the decode wall time is
dominated by the big serial kernels, NOT dispatch count. The "50 µs/op floor"
from the in-MLX cost model is an MLX-scheduler property; the single-command-buffer
replay does NOT pay it per dispatch. Lesson: on the native replay, **dispatch
COUNT is a weak lever** once launches are ~µs — optimize the big kernels' compute
(bandwidth/occupancy), not the op count. Kept anyway: correct, and it simplifies
the plan. NOTE: commit 661c239's message wrongly credits the 72.1 -> 62.6 gain to
this fusion; that gain was the concurrent attn_core/comp_step TG widening
(Stage 3g). This ledger entry is the correction of record.

Profile after wo_a fusion (bench8, isolated-group ms; sum > wall by per-group
commit overhead):

| kernel | count | ms |
|---|---|---|
| moe w13+w2 | 86 | 13.3 |
| library qmv | 280 | 11.54 |
| dsv4_attn_core | 43 | 9.97 |
| dsv4_hc_pre_k | 86 | 7.32 |
| dsv4_comp_step | 62 | 4.89 |
| dsv4_wo_a_k | 43 | 3.68 |
| dsv4_hc_post_k | 86 | 3.43 |
| dsv4_hc_mix_k | 86 | 3.34 |
| dsv4_rms_k | 173 | 2.61 |

So the real 40 ms levers are the big kernels' compute: moe (13, near bandwidth —
hard), attn_core (10), hc_pre (7). The easy pathological-kernel wins are spent.

### Stage 3i — attn_core loops bounded to O(context): small win, model of cost REFUTED ⚠️ (2026-08-02)

Commit aad0054. Every attn_core pass iterated the full slot capacity
K = WIN + KC regardless of context; the loops now run [jlo, jhi) with
jlo = max(0, WIN-1-offset), jhi = WIN + (PLAIN ? min(NCOMP, KC) : KC).
Method: `validate_deepseek_v4.py bench` (64 tok, prefill 8), wired, 96% free
before load, bench9.

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3h) | 62.6 | 16.0 |
| after loop bounding | **60.8** | **16.4** (1.36x vs 82.9 compiled) |

**Prediction FAILED in the interesting direction.** Expected ~-7 ms (ratio-128
layers drop from 2176 to ~72 slots at bench offsets; config: sliding_window=128,
index_topk=512); measured -1.8 ms wall and attn_core isolated 9.97 -> 9.88 (flat).
The arithmetic explains it in hindsight: a skipped slot (sc[j] <= 0 branch) costs
a few cycles, so 2176-slot sweeps were only ~7 µs/dispatch of the ~230 µs
observed. **The O(K) serial loops were never the attn_core cost.** The ~230
µs/dispatch remains unexplained — candidates: exposed memory latency on the
scattered ring/comp rows, barrier stalls (12/dispatch x 64 TGs), or
per-dispatch scheduling in the isolated-profile method itself (this run's
isolated sum 67.1 > wall 60.8, so the profiler overstates per-group cost —
treat isolated numbers as ranking, not absolute). Next probe: early-return
bisection variants of attn_core on the real model to localize the 230 µs.
Change kept: strictly less work, correct (native tests diff-verify;
full suite 1396), and it matters at prefill-heavy offsets.

Profile after (bench9, isolated-group ms): qmv 11.10 / attn_core 9.88 /
hc_pre 7.45 / moe w13+w2 12.65 / comp_step 4.93 / wo_a 3.82 / hc_mix 3.30 /
hc_post 3.14 / rms 2.75.

### Stage 3j — hc_pre TG 1024 + hc_post d-split: -4.2 ms, prediction MET (2026-08-02)

Commit 8d2e4d5. hc_pre single-threadgroup widened 256 -> 1024 (red[8] ->
red[32], all dispatch sites moved — Stage 3g checklist); hc_post grid split
from hcn (4) threadgroups to hcn x NSPLIT(8) d-slices. Method: same bench,
wired, 96% free, bench10.

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3i) | 60.8 | 16.4 |
| after hc widening | **56.6** | **17.7** (1.44x vs 81.4 compiled) |

Isolated: hc_pre 7.45 -> 5.34 (-2.1), hc_post 3.14 -> 1.13 (-2.0); wall
-4.2 ms ~= the isolated sum — the first change this session where the
prediction landed exactly. Session cumulative: 611 -> 56.6 ms (10.8x).

Profile after (bench10, isolated-group ms): qmv 11.39/280 · attn_core
10.19/43 · moe w13+w2 12.72/86 · hc_pre 5.34/86 · comp_step 4.89/62 ·
wo_a 3.51/43 · hc_mix 3.25/86 · rms 2.81/173 · gate 2.49/80 · hc_post 1.13/86.
Remaining to 40 ms: -16.6. Next: attn_core early-return bisection (the
~230 us/dispatch is still unexplained — Stage 3i), then comp_step/hc_pre
single-TG tails, then moe/qmv bandwidth efficiency.

### Stage 3k — attn_core bisected and fixed: 10.2 -> 1.6 ms isolated (2026-08-02)

Commits: probe method + fix in the perf commit (see git log around this
entry). New METHOD worth keeping: early-return bisection on the real model
with ONE load — patch the kernel source string per variant and build a fresh
NativeDecoder per variant (each compiles its own runtime), then read just
that kernel's profile_kernels row. Probe results for attn_core (token=5,
offset 13):

| cut | cumulative ms | delta |
|---|---|---|
| P0 dispatch floor | 0.80 | — |
| + kv rope / q RMS+rope | 0.86 | ~0 |
| + scores loop | 6.78 | **+5.92** |
| + softmax | 6.90 | +0.12 |
| + values loop (FULL) | 10.31 | **+3.41** |

Two causes, two fixes (one commit): (1) scores gave each valid slot to ONE
thread walking a 512-dim row serially — a single thread's few outstanding
loads expose full DRAM latency per element; now one SIMDGROUP per slot,
lanes stride the dims, simd_sum reduces (the qmv access pattern). (2) traced
(indexer) layers swept jhi = WIN+KC = 640 slots in every pass; idx_topk
emits winners as a -1-padded contiguous prefix and ioff[1] is its n2, so
both comp modes now bound jhi = WIN + min(NCOMP, KC). Bench (wired, 90%+
free, bench11):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3j) | 56.6 | 17.7 |
| after attn_core fix | **53.4** | **18.7** (1.45x vs 77.5 compiled) |

attn_core isolated 10.19 -> **1.59** ms. Session cumulative 611 -> 53.4
(11.4x). Lesson recorded: when a kernel's cost defies its arithmetic,
bisect it on the real model before optimizing — Stage 3i optimized the
wrong loop property (trip count, actually ~free) where the real cost was
one-thread-per-row latency exposure.

Profile after (bench11, isolated ms): moe w13+w2 12.68/86 · qmv 11.42/280 ·
hc_pre 5.27/86 · comp_step 5.12/62 · hc_mix 3.59/86 · wo_a 3.47/43 ·
rms 2.70/173 · gate 2.37/80 · attn_core 1.59/43 · hc_post 1.15/86.
Remaining to 40 ms: -13.4. Next: the same bisection on moe_w13/w2 (12.7 ms
measured vs ~1.8 ms of 2-bit expert-weight bandwidth — 14% efficiency, the
biggest unexplained gap left).

### Stage 3l — moe bisected: staging was the enemy; 49.3 ms / 20.3 tok/s (2026-08-02)

Three probe rounds on the real model (probe_moe*.py in the session
scratchpad; one ~2-min load each), then the fix. Additive decomposition of
moe FULL (isolated ms, K1/K2): dispatch+staging floor 1.6/2.3, weight loads
1.5/~0, 2-bit unpack ALU 1.2/0.9, staged-x threadgroup reads 2.0/2.5.
Refuted along the way (recorded so nobody retries them):

* float4 vector reads of staged x: NO change — the cost is not load
  instruction count;
* uint4 (128-bit) weight loads: K1 got WORSE (6.6 -> 7.1);
* bf16 threadgroup staging (16 KB -> 8 KB): -1.0 — pointed at residency.

Winner: **no staging at all**. x (8 KB) and h (48 KB) are L2-hot across
every threadgroup; K1's 16 KB xs[] had capped residency at 2 TGs/core and
K2 serialized behind 2 barriers per active expert. Reading straight from
device is bit-exact (the stage only moved the bf16->float convert) and
measured w13 6.7 -> 5.2, w2 6.3 -> 4.9 isolated. Commit is the perf commit
preceding this entry. Bench (wired, 96% free, bench12):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3k) | 53.4 | 18.7 |
| after moe no-staging | **49.3** | **20.3** (1.55x vs 76.6 compiled) |

Session cumulative 611 -> 49.3 (12.4x). The no-barrier diagnostic now
exactly equals the barriered run — blanket barriers cost ~0; overlap is
NOT a remaining lever.

Profile after (bench12, isolated ms): qmv 11.25/280 · moe w13+w2 10.10/86 ·
hc_pre 5.18/86 · comp_step 4.70/62 · wo_a 4.04/43 · hc_mix 3.20/86 ·
rms 2.77/173 · attn_core 1.74/43 · hc_post 1.64/86 · gate 2.21/80.
Remaining to 40 ms: -9.3. Next queued: hc_pre Sinkhorn on simd shuffles
(the 20 iterations cost ~40 threadgroup barriers on a 4x4 matrix; shuffle
gather keeps the serial add order, so it stays bit-identical), then
hc_mix widening, shared-expert qmv trio fusion, moe unpack LUT.

### Stage 3m — hc_pre Sinkhorn on simd shuffles: 47.3 ms / 21.1 tok/s (2026-08-02)

The Sinkhorn loop paid 2 threadgroup barriers per iteration (~40+6 total at
20 iters) to sync a hcn x hcn matrix across a 1024-thread group. Now lane l
of ONE simdgroup holds comb element (l/hcn, l%hcn); row/col sums gather
lane-by-lane with simd_shuffle in the same serial order (bit-identical),
and a simdgroup needs no explicit sync — kernel barrier count 46 -> 3.
Bench (wired, 96% free, bench13):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3l) | 49.3 | 20.3 |
| after shuffle Sinkhorn | **47.3** | **21.1** (1.60x vs 75.6 compiled) |

hc_pre isolated 5.18 -> 2.90 (-2.3, matching the wall change). Session
cumulative 611 -> 47.3 (12.9x). General rule extracted for the ledger:
**a threadgroup barrier on a wide TG costs ~1 us; anything iterating with
barriers over tiny data belongs in one simdgroup with shuffles.**

Profile after (bench13, isolated ms): qmv 10.85/280 · moe 10.25/86 ·
comp_step 4.95/62 · wo_a 3.84/43 · hc_mix 3.30/86 · hc_pre 2.90/86 ·
rms 2.74/173 · gate 2.25/80 · attn_core 1.47/43 · hc_post 1.22/86.
Remaining to 40 ms: -7.3. Queued: hc_mix TG 1024 (committed, bench next),
comp_step bisection, shared-expert trio fusion, moe unpack LUT.

### Stage 3n — hc_mix TG 1024: 45.4 ms / 22.0 tok/s (2026-08-02)

hc_mix's 24 threadgroups underfill the 64-core chip; TG 256 -> 1024
shortens each thread's serial load chain 64 -> 16 elements. Bench (wired,
96% free, bench14):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3m) | 47.3 | 21.1 |
| after hc_mix widening | **45.4** | **22.0** (1.66x vs 75.5 compiled) |

hc_mix isolated 3.30 -> 1.77. Session cumulative 611 -> 45.4 (13.5x).
Profile after (bench14, isolated ms): qmv 11.31/280 · moe 9.78/86 ·
comp_step 5.03/62 · wo_a 3.54/43 · hc_pre 2.87/86 · rms 2.73/173 ·
hc_mix 1.77/86 · attn_core 1.35/43. Remaining to 40 ms: **-5.4**.

### Stage 3o — comp_step state in place: 41.7 ms / 24.0 tok/s (2026-08-02)

Bisection (probe_comp.py, same one-load method): comp_step's 5.0 ms =
state double-buffer copy 2.5 + softmax pooling 2.3 + rms/rope ~0. The
kernel now writes ONLY the fresh slot row in place (the pooling loop never
reads that row — it redirects to the fresh kv_row/sc_row), and the
overlap-head shift (head rows [0,ratio) take tail rows [ratio,2*ratio),
disjoint ranges) runs behind the pooling barrier upgraded to mem_device.
Side wins: the decoder's b-side state buffers are gone and the plan cache
loses its parity dimension (ONE cached plan per n, not two). Unwritten
rows keep their -inf scores — exactly the empty mask. Bench (wired, 96%
free, bench15):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3n) | 45.4 | 22.0 |
| after in-place state | **41.7** | **24.0** (1.85x vs 76.9 compiled) |

Session cumulative 611 -> 41.7 (14.7x). Profile after (bench15, isolated
ms): qmv 11.12/280 · moe 10.11/86 · comp_step 4.20/62 · wo_a 3.77/43 ·
hc_pre 2.81/86 · rms 2.66/173 · hc_mix 1.56/86 · attn_core 1.40/43.
Remaining to 40 ms: **-1.7**. The pooling column-scatter half of comp_step
(-2.3 candidate, online-softmax rewrite) stays queued if needed.

### Stage 3p — shared-expert w1/w3+SwiGLU fused: 40.7 ms / 24.6 tok/s (2026-08-02)

`dsv4_sh13_k`: one simdgroup per inter row does both 8-bit gs64 dots and
applies the clipped SwiGLU in-register — the shared expert drops from
2 library qmv + swiglu to ONE dispatch per MoE layer (qmv count 280 -> 194).
Bench16: 40.7 ms / 24.6 tok/s (1.88x vs 76.5 compiled); sh13 isolated
1.99 vs the ~2.9 the three dispatches cost.

### Stage 3q — comp_step pooling as online softmax: 40.2 ms / 24.9 tok/s (2026-08-02)

Single pass with running rescale over the row-strided state instead of
max-then-sum twice (the second half of the Stage 3o bisection). The
mn > -inf guard skips empty rows so the -inf - -inf NaN intermediate never
forms. Bench17: 40.2 ms / 24.9 tok/s; comp_step isolated 4.00 -> 3.41.

### Stage 3r — TARGET MET: 38.9 ms / 25.7 tok/s (2026-08-02) ✅

`dsv4_hc_post2_k` fuses the MoE routed+shared add into the ffn-half hc_post
(x = T(a + b) in-register; a is FLOAT32 y_routed, b bfloat shared — binding
both as bfloat reinterprets the float buffer and NaNs the block, caught by
the plan tests). -43 dispatches. Bench (wired, 96% free, bench18):

| | ms/token | tok/s |
|---|---|---|
| before (Stage 3q) | 40.2 | 24.9 |
| after add fusion | **38.9** | **25.7** (1.85x vs 72.0 compiled) — **>= 25 tok/s MET** |

**The campaign: 611 -> 38.9 ms/token, 15.7x, in one day.** Every stage above
records its method; the losers (wo_a wall-time ~0, float4/uint4 loads,
threadgroup staging) are recorded next to the winners. Where the time went,
start to finish: pathological serial loops (Stage 3d-3f), threadgroup
widening (3g), O(capacity) loop bounds (3i), one-thread-per-row latency
exposure (3k), threadgroup staging vs residency (3l), barrier stacks on tiny
data (3m), double-buffer state copy (3o), and dispatch fusion (3p-3r).

Final profile (bench18, isolated ms): qmv 9.16/194 · moe w13+w2 9.71/86 ·
wo_a 3.57/43 · comp_step 3.20/62 · hc_pre 2.64/86 · rms 2.38/173 ·
sh13 2.00/43 · hc_mix 1.47/86 · attn_core 1.17/43 · gate 2.07/80 ·
hc_post+post2 1.56/86.

Paths beyond 25 tok/s, if wanted (in expected-value order): hc_pre rms
fusion (-1.0, designed), moe unpack ALU LUT (~-1), wo_a load widening
(~-1?), gate score+topk merge (~-0.5). Still OPEN before serving defaults
to the native path: ppl through the native decoder (quality gate), the
SOLOHEAVEN_DSV4_NATIVE=1 integration with eager fallback, and a multi-turn
server check — the native/README ladder items.

### Stage 4a — serving stability: native path quality-equivalent; WEIGHTS REGRESSED ON DISK ⚠️ (2026-08-02)

Borrow-mode serving integration (commit 7efcda7) validated on the real
model. The investigation chain and its traps, in order:

1. `nativecheck` (prefill 200, greedy 24): native determinism **0.0**,
   coherent-LOOKING output; compiled looked like garbage. **Trap**: on
   repetitive synthetic text, greedy coherence is a mirage — the
   locally-pattern-matching path can LOOK right while being wrong.
2. Teacher-forced step logits vs the eager batch reference: native was the
   one drifting (P=60 max|d| 9.4 -> P=200 19.3 with an argmax flip);
   compiled CONVERGED to eager with P (6.2 -> 0.8). Lesson: judge paths on
   logits against an independent reference, never on greedy text.
3. Per-layer state diff after ONE step from identical prefill state
   (prefill determinism 0.0): NO localized break — L0 ring differs by
   0.016 and the difference amplifies smoothly to ~3 by L42. bf16 chain
   divergence between two valid kernel orders, not a seeding bug.
4. Verdict by teacher-forced ppl on identical probes and weights:
   batch 7.11 / compiled-decode 6.98 / **native-decode 6.96** (ALL) —
   the native path is quality-EQUIVALENT. Gate PASSED.

**CORRECTION (same day)**: the paragraph that stood here claimed the
`-2bit-search` weights had been overwritten and regressed. **False — it
was a measurement error on this side.** `validate_deepseek_v4.py`'s MODEL
default (DSV4_MODEL unset) pointed at `-2bit-mixed`, so every Stage 4a
quality run measured the OLD min/max build — hence the exact match with
§1.1's mixed row. Proof the code did not regress either: batch ppl at
pre-campaign commit d8b55fe, same (default) build, gives the identical
17.91/9.01/1.55/7.11. The -search shards (one conversion run 01:10-01:43,
untouched since) show no evidence of damage. Consequences: the speed
numbers stand (the builds differ only in scale VALUES — identical sizes
and kernel timing); the three-way path-equivalence verdict stands (all
paths measured on the same weights); only the "weights regressed" claim is
retracted. The default now points at the shipped `-search` build so an
unset DSV4_MODEL can't silently measure the wrong build again. Lesson:
**a quality number is meaningless without the build it was measured on —
print the resolved model path in every validate run** (validate now does).

New validate subcommands: `nativecheck` (seeded-session agreement,
determinism, mini second turn), `pplnative` / `ppldec` (teacher-forced ppl
through the native / compiled DECODE step — decode-vs-decode is the fair
native gate; batch ppl differs by prefill-vs-decode semantics).

### Stage 4b — native decode quality gap: no localized bug; path-inherent divergence (2026-08-02)

Decode-vs-decode teacher-forced ppl on the shipped build: compiled 3.65
(ko 6.58 / en 3.98 / code 1.46) vs **native 4.13 (ko 7.05 / en 4.01 /
code 1.97)** — +13% ALL, concentrated on peaked distributions. The
investigation, so the next person doesn't redo it:

* Real bug found and FIXED on the way (commit e7d9026): the decoder gave
  every layer ONE shared [offset, ncomp] blob with ncomp=0 — the whole
  compressed region and the indexer's n2 were masked on the replay path
  above offset 128. Per-layer blobs now bake each layer's n.
* Per-position nll: the deficit clusters on the FIRST native-scored peaked
  token and recurring knife-edge positions; per-layer state diff shows
  smooth ~1e-2 divergence at L0 amplifying through the 43-layer HC ladder
  (no localized break).
* Component microscopes on the real model at offset=1: attention half
  CLEAN (attn_out 0.4% of magnitude; qr/q_raw/kvn/kv_roped at 1e-3),
  dense block clean (0.047); compressed-layer blocks 4x worse (0.18) via
  legitimate amplification of ~0.008 input diffs through the 2-bit MoE's
  sharp nonlinearities. (Trap for future probes: cwkv/cwgate/i_ckv/i_cwg/
  iw scratch are LEGACY-UNUSED — the plan reads xall by byte offset;
  comparing those buffers gives false smoking guns.)
* REFUTED: restoring the reference path's bf16 rounding points at the
  fusion seams (q rms/rope, roped kv, attn out, pooled group, sg/su) made
  ppl WORSE (4.13 -> 4.37) — reverted (3a943e2). The gap is not missing
  rounding; it is the compounding cost of a numerically different (often
  higher-precision) kernel path, and matching the library kernels'
  summation orders would reinstate the serial-loop latency pathologies.
* Generation A/B (greedy, smoke prompts): the delta IS user-visible in
  Korean — a corrupted multibyte token in a greeting, one repetitive
  degeneration — while English/factual outputs stay comparable.

Serving guidance recorded: SOLOHEAVEN_DSV4_NATIVE=1 is a SPEED mode
(25.7 tok/s, 1.85x) with a measured quality delta vs the compiled path
(13.5 tok/s, ppl 3.65). Closing the gap is future work (candidate: fp32
hc streams end-to-end, which could pass the compiled path's quality
rather than chase it).

### Stage 4c-4e — the native quality gap: two real bugs fixed, precision REFUTED as the cause (2026-08-02)

Goal: native decode quality at or above the compiled path. Baseline
(decode-vs-decode ppl on the shipped build, same probes): compiled **3.65**,
native **4.41**.

**Two real native-only bugs, found and fixed** (both were invisible below
offset 128, where the sliding window covers everything):

1. `ncomp` was baked 0 for every layer (one shared ioff blob) — the whole
   compressed region and the indexer's n2 were masked. Fixed by per-layer
   blobs (e7d9026). Long-context logit error vs the eager batch: 19.3 -> 4.25
   at P=200, argmax restored.
2. The visible group count was the compressor's PRE-step count, but the
   reference attends to the POST-step count (`ncomp = cn`) — native dropped
   the group the current token had just completed, i.e. the most recent
   summary, on every ratio-th token.

**Three precision upgrades — all REFUTED as the cause** (each measured on
the real model, each kept because it is strictly more precision than the
compiled path carries, none moved quality):

| change | native ppl |
|---|---|
| baseline (both bugs fixed) | 4.412 |
| + fp32 HC residual streams + fp32 MoE-gate input | 4.440 |
| + fp32 stacked x-projection (new `dsv4_qmv8_k`) | 4.44 (argmax-agree 62.5% -> 65.0%) |
| + fp32 attention/FFN output path (acore, o_lora, attn_out, sh, shared) | 4.441 |

So the gap is **not** bf16 rounding. The hypothesis it killed, recorded so
nobody re-runs it: "an independent kernel set differs by ~1 ULP at every
bf16 buffer and the 43-layer HC ladder amplifies that ~240x." Evidence
for the shape of the error, from `probe_growth.py`: per-position error vs
the eager batch is FLAT (native 11-15, compiled 0.8-1.5; argmax agreement
65% vs 98.8%), so nothing accumulates in the session state — every step
diverges by the same amount.

**Components cleared by isolation probes on the REAL model** (identical
inputs into native and the reference, scratch probes kept in the session
scratchpad): attention half (attn_out 0.4% = 1 ULP), compressor including
the COMPLETION path (pooled row 0.36%, state bit-exact), MoE gate incl.
hash routing (expert sets identical, weights to 1e-7), hc_pre (bit-exact),
comp/indexer state seeding, borrow-mode buffer registration (byte-exact).

**The 2-bit experts are clean too** (probe_moe_real.py, real weights, same
x and same expert indices into both): hexp max 0.022 / mean 0.0009 against
|ref| 1.75; y_routed max 0.004 / mean 0.0009 against |ref| 0.46. So EVERY
component matches the reference to ~1-3 ULP, yet the assembled 43-layer
path is 21% worse in ppl and flips 35% of argmaxes.

**Reading of the evidence, and the next experiment.** What is left is not a
bug but a property: our quantized-matmul kernels (wo_a, moe_w13/w2, sh13,
qmv8, embed, gate) are each correct to ~1 ULP but not BIT-IDENTICAL to
MLX's, and this build's 2-bit experts are sensitive enough that 43 layers
of that compound into real quality loss. Raising activation precision
cannot fix it — the residual difference lives in the quantized WEIGHT
paths, where being "more precise" than MLX is not the same as agreeing
with it. Note which paths already agree bit-for-bit: everything still on
the library qmv.

**RESOLVED — it was a real bug: hc_post read the HC mixing matrix
TRANSPOSED.** The reference is `einsum("bsjk,bsjd->bskd", comb, residual)`:
the SOURCE stream is comb's first axis, the OUTPUT stream its second. Both
native hc_post kernels summed `comb[k*hcn + j]` instead of
`comb[j*hcn + k]`. Fixed; native decode ppl **4.44 -> 3.651 against the
compiled path's 3.649** (Korean 6.52 vs 6.58 — native slightly BETTER), and
the long-context logit error vs the eager batch matches exactly (P=200:
1.375 both).

Two layers of camouflage hid it:

1. **Every 2x2 doubly-stochastic matrix is symmetric.** All native test
   configs use hc_mult=2, so a transposed mixing matrix is numerically
   invisible there. The shipped model is hc_mult=4. A regression test with
   a deliberately non-symmetric comb at hc=4 now exists.
2. **The compiled decode calls `_hc_post_math` (the einsum), not the shared
   mx.fast twin**, so only the replay path carried the bug — which is
   exactly why compiled measured 3.65 and native 4.44.

How it was finally localized (the method to reuse): seed ONE block with an
input identical to the reference's and walk its intermediates in order.
hx/post/comb came out exact, xn/qr/q_raw/kvn/attn_out within 1 ULP, and
h1 off by 14.5% — every input right, the output wrong, so the mixing step
was the only candidate left.

**Correction of record.** Earlier entries in this stage concluded that the
gap was inherent — that the model amplifies rounding ~20x per block and
only bit-exactness with MLX's kernels could close it. That was WRONG, and
the way it went wrong is worth keeping: three precision experiments in a
row measured flat, which should have falsified the precision hypothesis
instead of being explained away by an amplification story built on a
probe whose two sides had different input dtypes. The decisive counter-
example was available the whole time and came from the user: **ds4 is a
fully independent implementation with its own kernels and it is accurate**,
so "different kernels cost quality" could not be true. When measurements
keep refuting a hypothesis, suspect the hypothesis.

Kept from the detour: the two other real bugs (per-layer ioff/ncomp, the
post-step visible group count) and `dsv4_qmv8_k` (unused by the plan). The
fp32 activation chain was reverted — with the actual bug fixed it buys
nothing and costs bandwidth.

## 3. Reproduce

```bash
# quality
DSV4_MODEL=~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-search \
  .venv/bin/python validate_deepseek_v4.py ppl     # or: smoke / logits / compare / agree
# ds4 side (from ~/workspace/numenore/ds4; --raw is required for comparisons)
./ds4 --raw -p "PROMPT" --dump-logits ref.json -n 1 --temp 0
./ds4 -p "PROMPT" -n 64 --temp 0                   # prints prefill/generation t/s
# decode throughput — compiled path vs native replay runtime, tok/s each
# (loads the 88 GB weights; close everything else first — needs ~100 GiB free)
DSV4_MODEL=~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-search \
  .venv/bin/python validate_deepseek_v4.py bench 64
```

Rules this ledger follows are in `docs/DOCUMENTATION.md`.
