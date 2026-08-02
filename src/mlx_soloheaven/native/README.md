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

1. dummy-plan encode-cost measurement (decides re-encode vs ICB)
2. ONE layer (dense, layer 0) replayed → logits diff vs the MLX compiled
   step for the same inputs (tolerance bf16 ~1e-2)
3. layer kinds: +ratio-128 (comp step), +ratio-4 (indexer) — per-kind diff
4. full 43-layer step + embed/head → token-level agreement with the MLX
   path over a 32-token greedy run (must be exact or near-exact)
5. ppl probes through the native path ≈ MLX path (3.65)
6. serving integration: third path in `Model.__call__` behind
   `SOLOHEAVEN_DSV4_NATIVE=1`, fallback preserved; multi-turn HIT check
7. bench: target ≤40 ms/token

## Known constraints

* scales/biases must be bf16 (T-typed kernels) — our build already is.
* qmv_fast needs N%8==0 && K%512==0 — all our shapes qualify except the
  stacked x-proj (4160 rows: 4160%8==0 ✓, K=4096 ✓ fine).
* Cache growth past SOLOHEAVEN_DSV4_MAX_CONTEXT re-allocates buffers →
  runtime must detect (buffer identity check) and re-build the plan.
* Everything stays MLX-resident: the runtime borrows buffers, never owns
  weights; session save/restore and the engine contract are untouched.
