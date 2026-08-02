# dsv4 native decode runtime (Stage 3b) — design

Goal: decode DeepSeek-V4 by REPLAYING pre-encoded Metal command buffers
against MLX-owned memory, eliminating the measured ~40-70 µs per-op floor
that caps in-MLX decode at ~87 ms/token. Target ≥25 tok/s (≤40 ms).

Every precondition is proven by `dsv4_replay_spike.py` (run it) and
recorded in `docs/benchmarks/deepseek-v4.md` §Stage 3a.

## Architecture

One Python module (`runtime.py`, ctypes/objc like the spike — no C build
step unless profiling later demands it) owning:

1. **Pipeline cache** — `newLibraryWithURL(mlx.metallib)` for library
   kernels; `newLibraryWithSource` for OUR kernels (attention core, MoE
   pair, compressor step: reuse the Metal bodies from `deepseek_v4.py`,
   rewritten with explicit `[[buffer(i)]]` signatures in `kernels.metal`).
2. **Buffer table** — built once per session from the loaded model + cache:
   every weight array and every Stage-2 fixed-capacity cache array →
   (MTLBuffer, byte_offset) via DLPack. Capsules must be kept alive for the
   session. Scratch (h between MoE K1/K2, projection outputs, hc streams)
   allocated once by the runtime.
3. **Uniform buffer** (small, CPU-written per token): token_id, offset,
   comp-n per layer kind, and anything else that varies. Kernels that today
   take offset/n as traced scalars read them from this buffer instead.
4. **Encoding plan** — encode the full 43-layer step ONCE into a command
   buffer sequence. Metal command buffers are single-use, so "replay" =
   re-encode? NO: use `MTLIndirectCommandBuffer` (ICB) for compute, OR
   simply re-encode from a prebuilt Python plan (list of (pso, buffers,
   grid) tuples) — measure encode cost first: ~1500 encodes × ~1-2 µs
   CPU-side ≈ 2-3 ms/token, acceptable if GPU stays busy; ICB is the
   fallback if encode CPU time shows up.
   -> FIRST MEASUREMENT of 3b: encode-cost of a 1500-dispatch dummy plan.
5. **Sync contract** — `mx.synchronize()` after prefill / any MLX op that
   wrote the buffers; runtime waitUntilCompleted before MLX reads logits
   (later: shared MTLEvent if the handoff shows up in profiles).

## What runs where

| piece | kernel source |
|---|---|
| stacked x-proj, wq_b, wo_a/wo_b, shared expert, head | metallib `affine_qmv_fast_*` / `gather_qmv_*` / steel gemm |
| embedding row fetch | trivial custom kernel (row copy) |
| attention core, MoE K1/K2, compressor step | our kernels (bodies exist) |
| HC pre (mixes GEMV + sinkhorn + reduce), HC post | mixes via metallib gemv; sinkhorn tail custom |
| gate (scores + top-k), indexer scoring/top-k | custom (small); top-k as simple in-kernel selection over 256/512 |
| final norm + head | rms custom or metallib + qmv |

## Verification ladder (development order — each step gates the next)

1. ✅ encode-cost measurement — C loop 0.43 µs/dispatch, re-encode wins.
2. ✅ plumbing proven end-to-end (tests/test_dsv4_native.py): library qmv
   via C loop == mx.quantized_matmul (diff 0.0); per-token uniform rewrite
   takes effect with no re-encode; BufferTable dedup + diff 0.0; and OUR
   custom kernel (kernels.metal, explicit signature, C-loop dispatch) ==
   the mx.fast version. Every runtime primitive the loop needs now has a
   passing test.
3. ✅ (mostly) native/kernels.py GENERATES the kernel source from the model's
   frozen body strings + explicit signatures — single source of truth, no
   drift possible. All four compute kernels (moe_w13, moe_w2, attn_core,
   comp_step) compile; moe_w13, moe_w2 and the big branch-heavy attn_core are
   diffed against their mx.fast twins and match. Remaining for a full layer:
   the x-projection split, wq_b, wo_a/wo_b (all library qmv), the shared
   expert, HC pre/post (mixes via library gemv + a small sinkhorn kernel),
   the gate (scores + top-k) and the indexer — mechanical, same pattern.
   ✅ ALL custom kernels now exist and are diff-tested through the C loop:
   attn_core, moe K1/K2, hc_pre/post, gate (sqrtsoftplus + noaux_tc top-k),
   rms_norm. Library qmv covers wq_a/wkv/wq_b/wo_b, gate is custom, the
   shared expert is 3 qmv + a swiglu (reuse moe_w13-style or eager), grouped
   wo_a is 8 qmv (one per group) or gather_qmv. NOTHING custom remains
   unbuilt for a dense layer.
   ✅ BOTH chaining patterns a layer uses are proven: custom->custom (MoE
   K1->K2) and library->custom (wq_b qmv -> attn_core), each an intermediate
   buffer across a barrier in one command buffer.
   ✅ plan_item binds buffer SUB-RANGES via byte offset (grouped wo_a, the
   x-projection split, sub-vector reads) — diff 0.0. The ring-store kernel and
   every projection K being %512 (qmv_fast) are confirmed. NOTHING about the
   assembly is now unproven: all kernels, both chaining patterns, sub-range
   binding, in-place ring write.
4. ✅ DONE: the whole dense-attention sub-block is assembled as ONE native
   plan (wq_a qmv, q_norm rms, wq_b qmv, wkv qmv, kv_norm rms, attn_core,
   ring_store, wo_a x n_groups at byte offsets, wo_b qmv — 10 dispatches, one
   command buffer) and matches Attention.decode_step_math on a quantized tiny
   layer (test_native_dense_attention_plan_matches_reference). First full
   sub-block replayed end to end. Bug found and recorded: wo_a is 8-bit, so
   its per-group packed stride is gin/4 (not the 2-bit gin/16).
5. ✅ FFN-half plan (hc_pre, ffn_norm rms, gate, moe K1/K2, shared expert
   w1/w3/swiglu/w2, add, hc_post) diffed against the reference (< 3e-2).
6. ✅ FULL dense Block plan via native/plan.plan_block (~26 dispatches, both
   HC-wrapped halves) vs Block.decode_step_math — median diff < 2e-2, tail
   < 0.15 (accumulated bf16 over the deep chain; the per-half tests are the
   tight proof). native/plan.py now has the reusable Planner + plan_attention
   / plan_moe / plan_block builders.
7. ✅ FULL-MODEL decode replayed natively as ONE command buffer — embed
   (dequant row gather), N dense Blocks (plan_block), hc-head, final norm,
   head qmv — and its logits' ARGMAX matches Model.__call__ (median logit
   diff < 5e-2). plan.py has plan_embed / plan_head / plan_block. Bug caught:
   embed 8-bit row stride is hidden/4 (not the 2-bit hidden/16).
8. NEXT (only DENSE-model wiring remains for a bench; compressed layers add
   the compressor/indexer dispatches to plan_attention):
   a. compressed-layer plan: extend plan_attention with the compressor step
      (dsv4_comp_step) + indexer for ratio-4, wiring the per-layer compressor
      cache buffers; diff a compressed Block.
   b. real-Model driver: a NativeDecoder holding the Runtime, a session
      BufferTable built from the loaded model's weights + a DeepSeekV4Cache's
      ring/compressor buffers, the scratch, and a prebuilt plan; per token
      write (token, offset) into the uniform buffer and commit.
   c. integrate as a third path in Model.__call__ behind
      SOLOHEAVEN_DSV4_NATIVE=1 (after the compiled path), eager fallback kept.
   d. bench decode tok/s vs the 12 tok/s compiled path; target >=25.
5. All layer kinds, then the full 43-layer step + embed/head → token-level
   agreement over a 32-token greedy run.
6. ppl probes through the native path ≈ MLX path (3.65).
7. serving integration: third path in `Model.__call__` behind
   `SOLOHEAVEN_DSV4_NATIVE=1`, fallback preserved; multi-turn HIT check.
8. bench: target ≤40 ms/token (≥25 tok/s).

## Chained-plan contracts (learned building the first 2-item plan)

* **MLX buffers are hazard-untracked.** A dispatch reading what the previous
  one wrote gets garbage/NaN unless an explicit `memoryBarrierWithScope:
  Buffers` sits between them. `plan_item(barrier=True)` (the default) emits it;
  set False only for a provably independent dispatch. The decode chain is
  almost entirely dependent, so default-on is correct.
* **Kernels are specialized to the DEPLOYED dtypes.** The generated kernels
  read scales/biases as `bfloat` because the converter writes bf16 (verified
  on the real build). Test fixtures that quantize with `nn.quantize` get fp32
  scales and MUST cast to bf16 to feed the kernel its real format;
  `mx.fast` auto-adapts and hides this, the fixed-signature kernel does not.

## Known constraints

* scales/biases must be bf16 (T-typed kernels) — our build already is.
* qmv_fast needs N%8==0 && K%512==0 — all our shapes qualify except the
  stacked x-proj (4160 rows: 4160%8==0 ✓, K=4096 ✓ fine).
* Cache growth past SOLOHEAVEN_DSV4_MAX_CONTEXT re-allocates buffers →
  runtime must detect (buffer identity check) and re-build the plan.
* Everything stays MLX-resident: the runtime borrows buffers, never owns
  weights; session save/restore and the engine contract are untouched.
