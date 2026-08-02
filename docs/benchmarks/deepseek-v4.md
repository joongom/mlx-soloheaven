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
