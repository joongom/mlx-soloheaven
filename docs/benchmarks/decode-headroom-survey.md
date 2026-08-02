# Which models have decode headroom? (2026-08-03)

Machine: Apple M1 Ultra, 128 GiB, ~96% free per load, wired limit set, one
model resident at a time. Every row measured the same way: mlx-lm `load()`, an
8-token prompt, 4 warm-up steps then 32 timed greedy decode steps
(`scratchpad/probe_headroom.py`). No draft model anywhere.

**The question this answers.** The DeepSeek-V4 campaign paid off because V4's
decode sat far from what memory bandwidth allows — the gap was execution
overhead, and a replay runtime could take it. Before porting that machinery to
another architecture, measure whether the same gap exists. A model already
near its floor has nothing to give, and no amount of kernel work beats physics.

`floor = active bytes per token / 800 GB/s`. Active counts routed-expert
weights at `topk/E`, because batch-1 decode touches only those.

| model | kind | active | measured | floor | **gap** |
|---|---|---|---|---:|---:|
| Qwen3.6-27B 8bit | dense | 28.6 GB | 55.4 ms | 35.7 | **1.55x** |
| gemma-4-31B 8bit | dense | 32.6 GB | 66.1 ms | 40.8 | **1.62x** |
| EXAONE-4.5-33B 8bit | dense | 34.6 GB | 61.7 ms | 43.2 | **1.43x** |
| Qwen3.6-35B-A3B 8bit | MoE 8/256 | 3.67 GB | 17.1 ms | 4.6 | **3.72x** |
| Qwen3.5-122B-A10B mxfp4 | MoE 8/256 | 5.21 GB | 25.4 ms | 6.5 | **3.89x** |
| DeepSeek-V4-Flash, compiled | MoE 6/256 | 9.95 GB | 73.5 ms | 12.4 | **5.91x** |
| DeepSeek-V4-Flash, **native** | MoE 6/256 | 9.95 GB | **37.0 ms** | 12.4 | **2.98x** |

**The split is dense vs MoE, not vendor.** The three dense models are within
1.4-1.6x of their floor: they are already bandwidth jobs, so the ceiling on
ANY execution-side work is under 1.6x and the realistic gain is a fraction of
that. Not worth an architecture port.

The two Qwen MoE models sit at 3.7-3.9x — the same signature V4 had at 5.9x,
for the same structural reason. A batch-1 MoE step reads only ~10% of its
weights but still pays full per-op and per-dispatch costs, and MLX's
`gather_qmm` is not built for a batch of one. That is precisely what
`sh_dsv4_moe_w13` / `sh_dsv4_moe_w2` replace.

## The cheap experiment, and why it is now cheap

**The fused MoE kernels no longer need a port.** They used to hardcode 2-bit
unpacking; they now derive (bits, group_size) from the weights, and
`_moe_kernel_usable` admits every width they can read. Verified against
dequantized reference math at 2, 4 and 8 bits
(`test_moe_kernel_matches_dequantized_reference_at_any_supported_width`), so
Qwen's 8-bit experts are already in range. `_moe_routed_kernel` takes a
`SwitchGLU` and a clipping limit; passing `limit <= 0` turns the clipped
SwiGLU into the plain one Qwen uses.

So the first experiment is NOT the replay runtime. It is: call
`_moe_routed_kernel` in place of mlx-lm's `switch_mlp` for a batch-1 Qwen MoE
step and measure. That tests the single biggest hypothesis behind the 3.7x
gap, needs no new kernels, no plan builder, and no cache adapter.

Only if that pays would the rest of the runtime be worth discussing — and even
then the remaining pieces are real work: a GQA attention kernel (ours is MLA
with a sliding-window ring and compressed groups), a softmax top-k gate (ours
is DeepSeek's noaux_tc plus hash routing), a Qwen plan builder, and a
preallocating adapter for mlx-lm's `KVCache`, which grows in 256-token chunks
and would hit exactly the address-stability bug that broke V4 multi-turn.

## Caveats on these numbers

* 800 GB/s is the nominal figure, not a measured STREAM number for this
  machine, so the floors are optimistic and every gap is an UPPER bound on
  the available headroom. The relative ordering is what matters here.
* Short prompt (8 tokens). Stage 4m is the standing warning that decode cost
  can change shape with context; these gaps are the short-context regime.
* GLM-4.7 and GLM-5.1 are in the start scripts but not present locally, and
  gemma-4-26B-A4B (the MoE gemma, the one that would be interesting) is
  missing too. **The gemma row above is the DENSE 31B; the MoE gemma is
  untested and, being MoE, is the one likely to look like the Qwen MoE rows.**
