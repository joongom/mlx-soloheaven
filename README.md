# MLX SoloHeaven

**Single-user LLM inference server with KV cache optimization for Apple Silicon.**

SoloHeaven turns your Mac into a personal AI server with sub-second response times, even on 100K+ token conversations. It exposes an OpenAI-compatible API, making it a drop-in backend for tools like [OpenCode](https://opencode.ai), [Continue](https://continue.dev), [OpenClaw](https://openclaw.com), or any OpenAI SDK client.

![SoloHeaven Demo](docs/demo.gif)

## Key Features

- **Session-based KV cache reuse** — Only process new tokens each turn, not the entire conversation
- **250x TTFT improvement** — From 126s to 0.5s at 131K context tokens
- **99.9% token savings** — Cache hit reuses all previously computed KV states
- **Multi-model support** — Load multiple models simultaneously, route requests by model name
- **Multimodal (VLM) support** — Gemma 4, Qwen3-VL, GLM4V, and other vision/omni models via mlx-vlm, with automatic mlx-lm fallback for text-only models
- **Per-model thinking control** — Enable/disable `<think>` tags per model (e.g., `:no_think_tag` for non-reasoning models)
- **Thinking budget control** — Configurable token limit for reasoning models, with per-request override
- **Prompt Lookup Decoding (PLD)** — Optional draft-model-free speculative decoding for prompt-echo-heavy workloads (+37% on copy tasks)
- **Speculative decoding (MTP)** — Gemma 4 multi-token-prediction drafter via mlx-vlm (`--backend mlx-vlm --draft-model`); ~2x decode, byte-identical output, with sliding-window wrap fixes
- **Separate-process engine (default)** — Generation runs in a child process on its main thread for ~+30% tok/s at temp>0 (the FastAPI parent holds no MLX); switch off with `--engine-mode inprocess`
- **Structured output (`response_format`)** — OpenAI-compatible JSON mode and JSON Schema constraints via logits-level FSM masking (100% schema adherence, no prompt engineering)
- **Prefill chunk tuning** — Configurable `prefill_step_size` for 1.3-1.5x long-prompt speedup
- **KV cache quantization** — Optional 4/8-bit KV quant (mlx-lm path)
- **Budget-based cache eviction** — No time-based TTL, evicts LRU only when memory/disk budget exceeded; disk files auto-pruned when exceeding `--disk-budget-gb`
- **GPU keepalive** — Optional Metal idle prevention with periodic micro-computations (`--gpu-keepalive`)
- **OpenAI-compatible API** — Streaming SSE, tool calling, `developer` role, `response_format`, `/v1/chat/completions`, `/v1/models` (note: `stop`/`seed`/`tool_choice` are accepted in the request schema but not yet enforced)
- **Built-in web UI** — Chat interface with model selector, live thinking display, TPS stats, cache hit indicators, branch/regenerate/delete controls
- **Admin dashboard** — Real-time log viewer, cache/DB overview, and reset controls at `/admin`
- **Conversation branching** — Fork any conversation at any turn; the new session's KV cache is rebuilt by reprocessing the truncated history (no checkpoint restore)
- **Regenerate & Delete** — Re-roll the last response or remove turns; the cache is rebuilt from the truncated history
- **Disk persistence** — KV caches survive server restarts via safetensors serialization
- **Client disconnect handling** — Frees the generation lock immediately, tolerates content mismatches on reconnect
- **Base cache pool** — System prompt KV caches shared across sessions for fast cold starts

## Supported Model Families

With `--backend auto` (the default), SoloHeaven is **mlx-lm-first by support,
not by multimodal-ness**: it loads via `mlx-lm` whenever mlx-lm supports the
model's `model_type`, and only falls back to `mlx-vlm` for model types mlx-lm
cannot load (or when you pass `--backend mlx-vlm` explicitly). SoloHeaven is a
TEXT-only server, so a `vision_config`/`audio_config`/`image_token_index` in
config.json does **not** force mlx-vlm. **Gemma 4 is a VLM family whose config
always carries `vision_config`, yet it loads via `mlx-lm`** — mlx-lm supports
the `gemma4` type and its output is byte-identical to LM Studio's. This keeps
text coverage on the faster mlx-lm path while preserving opt-in MTP/vision
support through mlx-vlm. (`--backend mlx-lm` forces the mlx-lm path;
`--backend mlx-vlm` forces the vlm path and is required for the MTP
`--draft-model` drafter.)

| Model family | Backend (`--backend auto`) | Cache structure | Notes |
|--------------|---------|-----------------|-------|
| **Gemma 4** (`gemma4`, e.g. 31B/26B-A4B/E4B) | mlx-lm (mlx-lm supports `gemma4`); mlx-vlm only via `--backend mlx-vlm` | 50 `RotatingKVCache` (sliding window=1024) + 10 `KVCache` (full attn) | gemma4 loads via mlx-lm by default **even though its config always carries `vision_config`** — vision_config does NOT force mlx-vlm (text-only server; output byte-identical to LM Studio). Uses `<\|channel>thought\|...\|<channel\|>` for reasoning. **MTP speculative decoding** (`--draft-model`, gemma4-only) requires `--backend mlx-vlm`; on the default mlx-lm path use `--pld`. Past 1024 cumulative tokens the sliding ring wraps — handled by append-only wrapped-cache reuse + the MTP B4 wrap fix (see notes below). |
| **Qwen3.5 MoE** (`qwen3_5_moe`, e.g. 122B/397B) | mlx-lm | `ArraysCache` (DeltaNet linear) + `KVCache` (full attn every 4th) | Uses ChatML. **PLD not applicable** — DeltaNet state is not trimmable. **MTP not applicable** (drafter is gemma4-only). Use `--kv-bits 0` (quantization won't help; only 2 KV heads per layer). No sliding window — cache prefix reuse stays valid across long chats. |
| **Qwen3.6-27B** (`qwen3_5`, dense-hybrid) | mlx-lm | `ArraysCache` (DeltaNet) + `KVCache` (full attn) | 64 layers: 48 Gated-DeltaNet + 16 full-attention, **no sliding window**. ChatML. **PLD not applicable** (DeltaNet ArraysCache not trimmable); **MTP not applicable** (gemma4-only). Dense → bandwidth-bound decode. |
| **Qwen3.6-35B-A3B** (`qwen3_5_moe`, MoE) | mlx-lm | `ArraysCache` (DeltaNet) + `KVCache` (full attn) | 40 layers, 256 experts / 8 active (~3B active), **no sliding window**. ChatML. **PLD not applicable**; **MTP not applicable** (gemma4-only). MoE → fast decode, structurally stable multi-turn TTFT (no wrap). |
| **Qwen3-VL / Qwen3-Omni** | mlx-vlm | Varies per model | Vision/omni variants. |
| **GLM-5.1 / DeepSeek-V3.2** (`glm_moe_dsa`, `deepseek_v32`) | mlx-lm | `CacheList(KVCache, KVCache)` per layer (MLA + DSA indexer) | **Multi-head Latent Attention (MLA)**: KV is pre-compressed to `kv_lora_rank=512` — cache is ~1/3 of typical. **PLD-capable** (CacheList is trimmable), but `--pld` is off by default here — acceptance was ~12% on casual/reasoning; add it back only for copy-heavy workloads. Use `--no-thinking` (keep `prefill-step-size` at the default `2048` — `8192` OOMs on >100K prompts). |
| **GLM-4.5 / 4.7** (`glm4_moe`, `glm4_moe_lite`) | mlx-lm | `KVCache` or `RotatingKVCache` mix | ChatML-ish format with `<\|user\|>`/`<\|assistant\|>`. PLD compatible on pure-KVCache variants. |
| **GLM4V / GLM-OCR** | mlx-vlm | Per-model | Vision variants. |
| **MiniMax, GPT-OSS** | mlx-lm | Standard `KVCache` | ChatML. |

**Note on mxfp4/mxfp8 quantization**: MLX currently has kernel inefficiencies for
MoE matmul under mxfp4/mxfp8 modes (see [mlx issue #3402](https://github.com/ml-explore/mlx/issues/3402)).
If a `mxfp8` quant produces garbage output (`!!!!` repeated tokens), switch to
the plain `8bit` quant of the same model.

## Research Background

SoloHeaven was born from a systematic benchmarking study on KV cache optimization for hybrid attention models (DeltaNet + Full Attention). We tested 6 different strategies on Qwen3.5-122B-A10B-bf16 running on Mac Studio M3 Ultra (512GB) to find the optimal configuration.

### What Worked

| Strategy | Result | Verdict |
|----------|--------|---------|
| **Session-based KV cache reuse** | 265x TTFT improvement (171s → 0.6s at Turn 10) | Core feature |
| **Thinking budget (logits processor)** | 100% answer completion rate, 7% TPS drop | Adopted (default: 8192) |
| **Disk persistence (safetensors)** | 0.001s session reload, survives restarts | Adopted |
| **Base cache pool** | Instant system prompt reuse across sessions | Adopted |

### What Failed

| Strategy | Result | Why It Failed |
|----------|--------|---------------|
| **RotatingKVCache (8192)** | Best TPS (28.38) but quality degrades | Long-range recall drops to 4/8 — model loses earlier context despite DeltaNet compression |
| **KV 8-bit quantization** | TPS drops 16.5%, thinking budget hit 91% | Only 2 KV heads (dim 256) per layer — cache is already small, quantization overhead exceeds bandwidth savings |
| **Thinking token trim** | Layer mismatch causes pathological behavior | ArraysCache (DeltaNet) retains thinking state but KVCache has it spliced out — bimodal inconsistency, 31% longer responses, worse recall |

### Key Insight

> **Thinking preservation in KV cache is critical.** The model references its past reasoning across turns. Removing thinking tokens from the cache (trimming) or limiting context window (rotating) both degrade quality. The optimal strategy is to keep the full KV cache including thinking tokens, with a budget to prevent infinite thinking loops.

## Benchmark Results

The systematic study and the Qwen3.5 / 397B / GLM-5.1 numbers below are
**historical (Mac Studio M3 Ultra 512GB)** and remain valid for those models.
The current dev/optimization target is **M5 Max 128GB** — see
[M5 Max 128GB (Gemma 4 / Qwen3.6)](#m5-max-128gb-gemma-4--qwen36) for the
latest measurements.

### Systematic Comparison (11 turns, Qwen3.5-122B-A10B-bf16, M3 Ultra 512GB) *(historical)*

| Strategy | Avg TTFT | Avg TPS | TPS Drop (T0→T10) | Final Cache | Quality |
|----------|----------|---------|--------------------|----|---------|
| Baseline (no cache) | 71.9s | 26.1 | 18% | N/A | Thinking loops |
| Cached (no budget) | 0.74s | 26.2 | 18% | 88K | Thinking loops |
| **Optimized (adopted)** | **0.69s** | **27.55** | **7.4%** | **32K** | **5.0/5** |
| RotatingKV (8192) | 0.49s | 28.38 | 2.4% | 8K fixed | 4.32/5 |
| KV 8-bit | 0.52s | 26.04 | 16.5% | 32K | 4.64/5 |
| ThinkTrim | 0.66s | 28.48 | 4.9% | 26.5K | Layer mismatch |
| NoCache (rebuild) | 10.46s | 28.52 | 4.8% | N/A | Good but slow |

### M5 Max 128GB (Gemma 4 / Qwen3.6)

Current-generation measurements on a **MacBook Pro M5 Max 128GB** with 8-bit
models. Methodology: a single long generation (`max_tokens=2500`,
`temp=0.6`) for flat decode tok/s, plus an 8-turn multi-turn conversation to
~14K tokens for per-turn TTFT and cold-fill counts.

| Model (M5 Max 128GB, 8-bit) | Decode tok/s (2500-tok gen, flat) | Multi-turn TTFT (8 turns, ~14K tok) | Cold-fills | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (MoE ~3B active) | ~92.5 (flat, ~94→92.5) | 41–62 ms | 0 | No sliding window → stable TTFT, no cold-fill; MoE fast decode. Best all-round. |
| Qwen3.6-27B (dense-hybrid) | ~17 (flat) | 232–276 ms | 0 | Dense → bandwidth-bound slow decode (~27 GB/token). Stable TTFT. |
| Gemma4-26B-A4B + MTP (8-bit drafter) | ~97.8 (after B4 fix) | ~60 ms (after cold-fill fix) | 0 | MTP ~2x; needs the wrap fixes (B4 + cold-fill reconcile). |
| Gemma4-26B-A4B (no drafter) | ~83 (flat) | — | — | Plain decode baseline. |

**Model-selection takeaway:** for interactive multi-turn use,
**Qwen3.6-35B-A3B** is the strongest all-round on M5 Max (fast MoE decode +
structurally stable TTFT, no wrap handling needed); **Gemma4-26B-A4B + MTP**
is competitive on single long generations but relies on the sliding-window
fixes; **Qwen3.6-27B** dense is decode-bound.

### Qwen3.6-35B-A3B Best Practices (M5 Max 128GB)

All numbers in this section were measured on a **MacBook Pro M5 Max 128GB**
(mlx-lm path, `Qwen3.6-35B-A3B-MLX-8bit`). They are machine-specific guidance,
not universals — every flag below is overridable at launch (trailing args) or
per-request. Architecture background is in
[Supported Model Families](#supported-model-families) and the
[Hybrid Attention Architecture Note](#hybrid-attention-architecture-note).

#### Recommended Launch

```bash
./start_qwen3.6_35b_a3b.sh        # plain decode — simplest recommended config
./start_qwen3.6_35b_a3b_mtp.sh    # + native MTP head (block size 1) — a wash vs plain, safe
```

- **Plain decode is already fast.** The A3B MoE (~3B active params) decodes at
  ~90–97 tok/s on this machine — `./start_qwen3.6_35b_a3b.sh` is the simplest
  recommended config and leaves nothing meaningful on the table.
- **The MTP script is safe but not a meaningful speedup here.** It uses the
  model's **native `qwen3_5_mtp` head** (5-bit, ~0.6 GB) on the default mlx-lm
  backend — this is *not* the gemma4 mlx-vlm drafter, and no `--backend` flag
  is needed. After a block-size sweep (below) it defaults to
  `--draft-block-size 1`, which is a **wash vs plain decode (mean 1.019x)**.
  The head-config default of 3 was a measured **9–13% regression** and is
  overridden.
- **Sampling needs no flags.** The engine auto-applies the model's
  `generation_config.json` (temp 1.0 / top_k 20 / top_p 0.95); the flags in
  the start scripts only mirror it for visibility. Per-request override via
  the OpenAI API works as usual.
- **Anti-loop safety net.** Both start scripts set
  `--thinking-budget 4096 --repetition-penalty 1.1` (overridable).
- **`--pld` does not apply to this arch.** The Gated-DeltaNet `ArraysCache` is
  untrimmable, which makes [PLD](#prompt-lookup-decoding-pld) incompatible —
  the engine guards this automatically.
- **Memory.** The 8-bit model is ~37 GB; `--memory-budget-gb` 20–30 is a good
  setting on 128 GB machines. Active-session LRU eviction keeps RAM bounded.

#### MTP Block-Size Sweep (2026-06)

Production sampling (temp 1.0 / top_k 20 / top_p 0.95), 500 generated tokens
per run; speedup is vs plain decode on the same prompt, acceptance rate in
parentheses. This is the condensed view — the full table (raw tok/s, seed,
greedy spot-check) lives as a comment in
[`start_qwen3.6_35b_a3b_mtp.sh`](start_qwen3.6_35b_a3b_mtp.sh).

| Block size | 152-tok prompt | 2,542-tok prompt | 9,989-tok prompt | Mean |
|---|---|---|---|---|
| no MTP | 1.000x | 1.000x | 1.000x | 1.000x |
| **1 (script default)** | 1.07x (acc .86) | 0.97x (.78) | 1.02x (.83) | **1.019x** |
| 2 | 1.07x (.75) | 0.99x (.71) | 0.98x (.70) | 1.010x |
| 3 (head-config default) | 0.91x (.60) | 0.87x (.58) | 0.91x (.61) | 0.897x |
| 4 | 0.77x (.46) | 0.84x (.55) | 0.69x (.45) | 0.766x |

Key facts:

- **10K context does not change the winner** — deep blocks get *worse* with
  context, not better: every rejected round pays a 40-layer ArraysCache
  restore + replay whose cost grows with KV size.
- **Free-form / creative content is the worst case** for MTP (block 3 fell to
  0.707x). Highly predictable content can still profit — isolated measurements
  up to +26% at acceptance ~0.81, and ~2x on very short structured turns — so
  it is content-dependent, and lossless either way (verify-then-accept keeps
  output identical to plain decode).
- **Contrast with gemma4:** dense Gemma4 is where MTP genuinely shines
  (60 → 97.8 tok/s — see
  [Speculative Decoding (MTP)](#speculative-decoding-mtp)). This A3B MoE
  simply plain-decodes too fast for the head to pay off.

#### Multi-Turn

Session KV-cache reuse works **with MTP enabled** (lazy last-pair commit):
measured turn-2 TTFT was **58 ms** (cache reuse) vs **407 ms** (cold-fill) at
1.6K session tokens, and the gap grows with session length. Output is
byte-identical with and without reuse.

#### When to Use What

- **Coding agent / structured output** → MTP block 1 is fine
  (wash-to-slight-win), or just plain decode.
- **Long free-form / creative / thinking-heavy** → plain decode (MTP off).
- **Never** set `--draft-block-size` ≥ 3 on this model — a measured 10–23%
  mean regression across all tested prompt lengths.

### Production Metrics (Qwen3.5-122B-A10B-bf16) *(historical, M3 Ultra 512GB)*

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| TTFT (Turn 10, ~131K tokens) | 126s | 0.5s | **~250x** |
| Token savings per request | 0% | 99.9% | — |
| Generation TPS | 27.5 tok/s | 27.5 tok/s | No degradation |
| Quality (5-point scale) | 4.64 | 4.64 | No degradation |

### Real-World Usage: Qwen3.5-122B-A10B-bf16 (1 hour with OpenCode) *(historical, M3 Ultra 512GB)*

**Machine:** Mac Studio M3 Ultra 512GB / 4TB

Over 191 messages in a real coding session:

- **Cache hit rate**: 89% (170/191)
- **Avg TTFT on cache hit**: 0.5s
- **Avg TTFT on cache miss**: 45s
- **Total tokens saved**: 11.8M (99.9% reduction)
- **Peak context**: 131,072 tokens

### Real-World Usage: Qwen3.5-397B-A17B-MLX-8bit (OpenClaw agent session) *(historical, M3 Ultra 512GB)*

**Machine:** Mac Studio M3 Ultra 512GB / 4TB

Qwen3.5-397B is 3.2x larger than 122B (17B active params vs 10B). KV cache per token is also 4.3x larger (32.2 KB vs ~7.5 KB), reflecting the model's deeper attention structure.

Live data from a coding agent session (OpenClaw, 266 messages, 131K context):

| Metric | Value |
|--------|-------|
| Cache hit rate | 93.8% (15/16 requests) |
| Token savings rate | 92.2% (1.72M tokens reused / 1.86M total) |
| Active sessions in memory | 7 (3 clients: OpenClaw, OpenCode, Web UI) |
| Largest session | 131,927 tokens, 266 msgs, 4.1 GB KV cache |
| Total KV cache in memory | 321,138 tokens (10.3 GB) |
| Disk cache (persistent) | 21 files, 18.0 GB |
| Base cache pool | 5 system prompts cached, 5 reuses |
| Disk save time | avg 0.79s for ~3.6 GB (background, non-blocking) |

**TTFT by suffix size (cache HIT only, 15 requests):**

> **Suffix** = the tokens that need to be actually processed after a cache hit. A typical user message is ~50-90 tokens. When tool results are included, the suffix can grow to hundreds or thousands of tokens. TTFT is proportional to suffix size.

| Suffix Size | Count | Avg TTFT | Range |
|-------------|-------|----------|-------|
| Small (<500 tokens) | 8 | **1.4s** | 1.0-3.0s |
| Large (500+ tokens) | 5 | **3.9-5.0s** | 3.9-5.0s |
| Very large (~8,800 tokens) | 2 | **55.6s** | 54-57s |
| Full miss (124K tokens) | 1 | **528s** (8.8 min) | — |

**TTFT comparison: 122B vs 397B (cache HIT, small suffix):**

| Model | Suffix ~50-90 tokens | Full miss (124K) | KV cache per token |
|-------|---------------------|------------------|-------------------|
| Qwen3.5-122B-A10B-bf16 | 0.5s | ~126s | ~7.5 KB |
| Qwen3.5-397B-A17B-MLX-8bit | 1.0-1.3s | 528s | 32.2 KB |
| Ratio | ~2-2.5x | ~4x | ~4.3x |

**Key observations:**
- 397B with cache hit (small suffix) achieves 1.0-1.3s TTFT — practical for coding agent workflows
- Large tool results (8K+ tokens) in suffix cause TTFT spikes to ~55s — tool result size optimization or caching strategies are needed
- KV cache is 4.1 GB/session, but 512GB memory can hold 7 sessions simultaneously (model weight ~160GB + KV cache ~10GB)
- Disk save runs in background without lock, no blocking on generation requests

### GLM-5.1 (MLA + DSA + MoE 256 experts, M3 Ultra 512GB) *(historical, M3 Ultra 512GB)*

GLM-5.1 (`mlx-community/GLM-5.1-MXFP4-Q8`) inherits DeepSeek-V3.2's
architecture: **Multi-head Latent Attention** (kv_lora_rank=512) +
**DeepSeek Sparse Attention** (32-head indexer, top-k=2048) + **MoE**
(256 routed experts, 8 active). 78 layers, ~378 GB on disk.

**Baseline performance**: **12–13 TPS** at short context (no thinking).

| Metric | Value |
|--------|-------|
| Model size (mxfp4 + Q8) | 378 GB on disk, ~450 GB resident |
| KV cache per token (MLA compressed) | ~90 KB (78 × 1152 bytes) |
| TTFT (cache miss, 13K tokens prefill) | ~100s |
| TTFT (cache hit, small suffix) | 400–600 ms |
| Decode TPS (short context) | 15 TPS |
| Decode TPS (30K+ context) | 12 TPS |

**Why 12–13 TPS is near the ceiling**: At M3 Ultra's 800 GB/s bandwidth
with ~30 GB of active weights read per step (MLA + MoE + DSA indexer),
the memory-bandwidth ceiling is ~26 TPS at 100% efficiency. mlx-lm
reaches about 50% of peak, matching measurements. MLA already keeps
the KV read volume very small (see table) — there's limited room for
further KV-side gains.

**Recommended launch config for GLM-5.1**:

```bash
./start_glm5.1.sh
# or:
mlx-soloheaven --model ~/models/GLM-5.1-MXFP4-Q8 \
  --memory-budget-gb 50 \
  --gpu-keepalive \
  --no-thinking
```

> **Why no `--prefill-step-size 8192` or `--pld`** (both removed from
> `start_glm5.1.sh`): `8192` causes Metal OOM on >100K-token prompts — each
> chunk's attention against the full KV cache exceeds the ~60 GB free after the
> ~450 GB model loads, so the default `2048` is kept. `--pld` was removed
> because acceptance was only ~12% on casual/reasoning workloads (verification
> overhead exceeds the gain); add it back **only** for copy-heavy workloads
> (code editing, RAG, tool-arg repetition) where acceptance typically exceeds
> 30%.

**What each flag buys**:

| Flag | Effect on GLM-5.1 |
|------|-------------------|
| `--no-thinking` | Skips `<think>` generation — in OpenClaw agent workloads, thinking turns often eat 80% of generation tokens; disabling it is a large effective speedup. |
| `--memory-budget-gb 50` | Leaves ~60 GB headroom on top of the ~450 GB model for KV growth before session LRU eviction kicks in. |
| `--gpu-keepalive` | Keeps Metal warm to avoid the idle-wakeup penalty. |
| `--prefill-step-size 8192` *(NOT recommended here)* | 1.3–1.5x faster prefill on long prompts in general, but on GLM-5.1's huge model it OOMs on >100K-token prompts — keep the default `2048`. |
| `--pld` *(workload-dependent, off by default)* | Prompt Lookup Decoding: +30–40% TPS on copy-heavy turns where the model echoes the prompt, but ~12% acceptance on GLM-5.1 casual/reasoning makes it net-negative there. Logs `[PLD] accepted X/Y (Z%)` — keep it only if Z > 30%. |
| `--disk-budget-gb 100` (default) | Disk session cache enforces this budget via LRU eviction. |

**Known limitations**:

- **PLD is workload-dependent**. Pure creative writing shows a ~15%
  slowdown. The acceptance-rate log lets you confirm per-workload
  whether to keep it enabled.
- **Disk save skipped for GLM MoE**: mlx-lm's save_prompt_cache can't
  serialize some empty arrays in GLM MoE state. Session cache is
  in-memory only for this model; restart loses the KV. (Other models
  with standard cache shapes still persist.)
- **mxfp4 MoE kernels are slower than Q8** on M3 (see [mlx #3402](https://github.com/ml-explore/mlx/issues/3402)).
  If a quant feels pathologically slow, swap to the Q8 variant of the
  same model.

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11+ (tested on 3.12 and 3.14; install via [pyenv](https://github.com/pyenv/pyenv))
- An MLX-format model (e.g., from [mlx-community on HuggingFace](https://huggingface.co/mlx-community))

### Setup from Scratch

```bash
# 1. Install pyenv (if not installed)
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 2. Install Python 3.12
pyenv install 3.12.11
pyenv local 3.12.11    # sets .python-version in project root

# 3. Clone and create venv
git clone https://github.com/joongom/mlx-soloheaven.git
cd mlx-soloheaven
python -m venv .venv
source .venv/bin/activate

# 4. Install
pip install -e .
```

### Download a Model

```bash
# Example: Qwen3.5-122B MoE (10B active params, bf16)
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen3.5-122B-A10B-bf16 --local-dir ~/models/Qwen3.5-122B-A10B-bf16
```

### Running

```bash
# Activate venv (if not already)
source .venv/bin/activate

# Single model
mlx-soloheaven --model ~/models/Qwen3.5-122B-A10B-bf16

# Single model with GPU keepalive and verbose logging
mlx-soloheaven --model ~/models/Qwen3.5-122B-A10B-bf16 --gpu-keepalive --verbose

# Multiple models
mlx-soloheaven --models ~/models/Qwen3.5-122B-A10B-8bit ~/models/Qwen3.5-9B-bf16 ~/models/Qwen3-Coder-Next-8bit:no_think_tag

# Or use the start script
./start.sh
```

The server starts on `http://localhost:8000` with:
- Web UI at `/`
- Admin dashboard at `/admin`
- OpenAI API at `/v1/chat/completions`
- Health check at `/health`

### Configuration

All settings can be passed via CLI flags or environment variables:

```bash
# CLI flags
mlx-soloheaven \
  --model ~/models/Qwen3.5-122B-A10B-bf16 \
  --port 8000 \
  --temperature 0.6 \
  --top-p 1.0 \
  --min-p 0.0 \
  --top-k 0 \
  --repetition-penalty 1.0 \
  --thinking-budget 8192 \
  --memory-budget-gb 200 \
  --gpu-keepalive \
  --verbose

# Or use environment variables (prefix: SOLOHEAVEN_)
export SOLOHEAVEN_MODEL=~/models/Qwen3.5-122B-A10B-bf16
export SOLOHEAVEN_PORT=8000
export SOLOHEAVEN_TEMPERATURE=0.6
export SOLOHEAVEN_THINKING_BUDGET=8192
mlx-soloheaven
```

You can also use a `.env` file — copy [`.env.example`](.env.example) to `.env` and edit:

```bash
cp .env.example .env
# Edit .env with your model path and preferences
mlx-soloheaven
```

Run `mlx-soloheaven --help` for all options:

```
Options:
  --model, -m           Path to MLX model directory
  --models              Multiple models: 'path' or 'path:no_think_tag' (env: SOLOHEAVEN_MODELS, comma-separated)
  --backend             {auto,mlx-lm,mlx-vlm} inference backend (default: auto;
                        mlx-lm-first BY SUPPORT — mlx-lm whenever it supports
                        the model_type, incl. gemma4; falls to mlx-vlm only for
                        types mlx-lm cannot load). 'mlx-vlm' is REQUIRED for the
                        MTP --draft-model drafter (env: SOLOHEAVEN_BACKEND)
  --host                Bind address (default: 0.0.0.0)
  --port, -p            Listen port (default: 8000)
  --temperature         Default sampling temperature (default: 0.6)
  --top-p               Nucleus sampling top-p (default: 1.0, disabled)
  --min-p               Min-p sampling threshold (default: 0.0, disabled)
  --top-k               Top-k sampling (default: 0, disabled)
  --repetition-penalty  Repetition penalty (default: 1.0, disabled)
  --max-tokens          Default max generation tokens (default: 32768)
  --thinking-budget     Max thinking tokens before forcing </think> (default: 8192, 0=unlimited)
  --memory-budget-gb    In-memory KV cache budget in GB (default: 200)
  --disk-budget-gb      On-disk KV cache budget in GB, auto-evicts oldest (default: 100)
  --max-checkpoints     Max DeltaNet checkpoints per session for branching (default: 50, 0=unlimited)
  --data-dir            Directory for SQLite DB and cache files (default: ./data)
  --no-thinking         Disable thinking mode globally
  --gpu-keepalive       Keep Metal GPU warm to avoid idle penalty (env: SOLOHEAVEN_GPU_KEEPALIVE)
  --verbose, -v         Enable verbose logging (env: SOLOHEAVEN_VERBOSE)

Engine mode:
  --engine-mode         'process' (default; run generation in a separate child
                        process on its main thread for higher decode tok/s;
                        single --model only, --models auto-falls-back to inprocess)
                        or 'inprocess' (env: SOLOHEAVEN_ENGINE_MODE)

Streaming:
  --stream-coalesce-n   Max tokens batched per SSE frame (default: 4; <=1 disables coalescing)
  --stream-coalesce-ms  Max ms to hold a partial batch before flushing (default: 30)

Speculative decoding (mlx-vlm drafter, Gemma 4) — requires --backend mlx-vlm:
  --draft-model         Path to drafter MLX model directory (enables MTP speculative
                        decoding; requires --backend mlx-vlm. On the default mlx-lm
                        backend use --pld instead)
  --draft-kind          Drafter kind: 'mtp' (Gemma 4) or 'dflash' (Qwen3); default: auto-detect
  --draft-block-size    Drafter block size (default: use drafter config)

Performance tuning (see Performance Tuning Flags section below):
  --prefill-step-size   Prefill chunk size (default: 2048; try 4096/8192)
  --pld                 Enable Prompt Lookup Decoding (speculative decoding via prompt n-grams)
  --pld-num-draft       Max draft tokens per step (default: 10)
  --pld-ngram-k         N-gram size for PLD matching (default: 3)
  --kv-bits             KV cache quantization bits (0=off, 4, 8; mlx-lm path only)
  --kv-group-size       KV quant group size (default: 64)
  --quantized-kv-start  Token offset at which KV quantization kicks in
```

### Sampling Parameters

Default sampling parameters applied to all generation requests. Each can be overridden per-request via the API.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.6 | Controls randomness. 0 = deterministic (argmax), higher = more creative |
| `top_p` | 1.0 | Nucleus sampling. 1.0 = disabled, lower values focus on high-probability tokens |
| `min_p` | 0.0 | Minimum probability threshold (scaled by top token). 0.0 = disabled |
| `top_k` | 0 | Top-k sampling. 0 = disabled, positive values limit token candidates |
| `repetition_penalty` | 1.0 | Penalizes repeated tokens. 1.0 = disabled, >1.0 discourages repetition |

Configure via CLI (`--temperature 0.6`), environment variables (`SOLOHEAVEN_TEMPERATURE=0.6`), or `.env` file.

Per-request override via the OpenAI API:

```json
{
  "model": "default",
  "messages": [...],
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 40,
  "min_p": 0.05,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3
}
```

> `frequency_penalty` and `presence_penalty` (OpenAI standard) are mapped to `repetition_penalty` when `repetition_penalty` is not explicitly set.

### Performance Tuning Flags

Optional speed knobs. All are server-level (set at launch); SoloHeaven's
session/KV/compaction logic is unaware of them and keeps working identically
whether they're on or off.

| Flag | Default | Applies to | Effect |
|------|---------|-----------|--------|
| `--prefill-step-size N` | 2048 | both mlx-vlm & mlx-lm | Chunk size for prompt prefill. Larger = fewer kernel launches = faster prefill on long prompts. Recommended: **8192** on Mac M3 Ultra for 1.3–1.5x prefill speedup. |
| `--pld` | off | mlx-lm only; auto-disabled on DeltaNet/ArraysCache | Prompt Lookup Decoding: speculative decoding using n-gram matching from the prompt instead of a separate draft model. **+37% TPS on copy-heavy workloads (code editing, RAG), -15% on novel/creative**. See [PLD section](#prompt-lookup-decoding-pld) below. |
| `--pld-num-draft N` | 10 | PLD only | Max draft tokens proposed per step. |
| `--pld-ngram-k K` | 3 | PLD only | N-gram size for the prompt-lookup match. Shorter = more matches but noisier. |
| `--kv-bits 0\|4\|8` | 0 (off) | mlx-lm path only | Quantize KV cache to reduce memory. **Skip for MLA models** (GLM-5.1, DeepSeek): KV is already compressed via `kv_lora_rank`. Skip for small-KV-head models (Qwen3.5 MoE has only 2 KV heads per layer — overhead dominates). Useful for standard dense/MoE models with large KV heads at long context. |
| `--kv-group-size N` | 64 | with `--kv-bits` | Quantization group size. |
| `--quantized-kv-start N` | 0 | with `--kv-bits` | Skip quant for first N tokens. |
| `--memory-budget-gb N` | 200 | always | In-memory KV cache budget before LRU eviction to disk. |
| `--disk-budget-gb N` | 100 | always | On-disk cache budget. **Now actually enforced**: when `_save_session_to_disk` exceeds this, oldest session files are deleted (protecting active sessions). |

#### Prompt Lookup Decoding (PLD)

PLD is a draft-model-free form of speculative decoding. On each decode step
it searches the prompt (and accumulated output) for a match to the last K
generated tokens; if found, it proposes the next N tokens as drafts, which
the main model verifies in one forward pass.

**When PLD is a win** (+30% to +40% TPS):
- Code edits (the model echoes parts of the source it's modifying)
- RAG / summarization (the model quotes the retrieved text)
- Tool-calling agents where tool arguments mirror prompt content
- Long-context Q&A that cites the document

**When PLD is a loss** (−10% to −15% TPS):
- Free-form creative writing (no prompt-to-output overlap)
- Pure reasoning / chain-of-thought
- First-turn greetings

**Compatibility**: Requires a trimmable KV cache. Automatic detection:
- ✅ GLM-5.1 (CacheList/KVCache), GLM-4.7, DeepSeek-V3/V3.2, Llama, Mistral
- ❌ Qwen3.5 / Qwen3.6 (DeltaNet ArraysCache is not trimmable — auto-falls back, zero overhead)
- ❌ mlx-vlm path (use the VLM's own speculation path if available)

**Acceptance rate**: logged after every request as `[PLD] accepted X/Y draft tokens (Z%)`.
Rule of thumb: Z > 30% is a net win; Z < 20% means PLD is hurting — disable for that workload.

#### Speculative Decoding (MTP)

In addition to PLD (which needs no draft model), SoloHeaven supports a real
**draft-model speculative decoder** for **Gemma 4 only**, via mlx-vlm's
multi-token-prediction (MTP) path. Since the mlx-lm-first migration the MTP
drafter is an **explicit mlx-vlm opt-in**: launch with `--backend mlx-vlm`
(`--backend auto`/`mlx-lm` would otherwise load gemma4 text via mlx-lm, where
the drafter is unavailable — use `--pld` there instead). Then enable it with
`--draft-model <drafter>` (plus optional `--draft-block-size`); `--draft-kind`
auto-detects as `mtp` for gemma4 assistant drafters.

**Verified setup** (`start_gemma4_26b_a4b_mtp.sh`): an 8-bit assistant drafter
(`guardiangate1775/gemma-4-26B-A4B-it-assistant-8bit`) with `--draft-block-size 3`,
against the 8-bit target. Output is **byte-identical to plain greedy** decode,
at ~2x decode throughput (≈97.8 tok/s vs ≈83 tok/s plain on M5 Max).

**The hard part — the sliding-window wrap.** Gemma 4 is 50 sliding-attention
layers (`RotatingKVCache`, window=1024) + 10 full-attention layers. The drafter
relies on the sliding-window state; past the 1024-token ring wrap the drafter's
query RoPE phase was being corrupted (by the old offset-clamp), collapsing
acceptance from ~1.2 to ~0.2 and making MTP net-negative. The **B4 RoPE-frame
fix** (feed the *true* offset + build an absolute-position drafter mask)
recovers post-wrap acceptance to ~1.1–1.2, so MTP stays ~2x even past the wrap.
A `SOLOHEAVEN_MTP_WRAP_GATE=1` env fallback can disable the drafter post-wrap
(plain decode) if a future drafter/model ever regresses.

**Compatibility**:
- ✅ Gemma 4 (mlx-vlm MTP path)
- ❌ Qwen3.5 / Qwen3.6 (DeltaNet) — not supported by the MTP drafter (see
  [Future Directions](#future-directions) for native Qwen3.6 MTP / DFlash)
- ❌ Combined with `--models` (multi-model) — drafter is single-`--model` only
- Independent of PLD — MTP is a separate, model-specific path

> **Qwen3.6-35B-A3B** ships a separate **native `qwen3_5_mtp` head** that runs on the default mlx-lm path (no mlx-vlm) — measured a wash vs plain decode; see [Qwen3.6-35B-A3B Best Practices](#qwen36-35b-a3b-best-practices-m5-max-128gb).

### Structured Output (`response_format`)

SoloHeaven implements OpenAI's `response_format` parameter for **guaranteed JSON**
output — not by adding "please output JSON" to the prompt, but by masking the
model's logits at each decode step so that only tokens that maintain a valid
JSON schema can be sampled. This is the same technique used by vLLM's
`guided_json` / OpenAI's Structured Outputs.

**No server flag required.** This is a per-request API parameter — all start
scripts support it out of the box. When a client sends `response_format`,
SoloHeaven builds a logits-level FSM constraint on demand; when it doesn't,
generation runs unconstrained.

**Three modes** (OpenAI-compatible):

```python
# Mode 1: no constraint (default, unchanged behavior)
{"type": "text"}

# Mode 2: any valid JSON object
{"type": "json_object"}

# Mode 3: strict JSON schema (what you want most of the time)
{
  "type": "json_schema",
  "json_schema": {
    "name": "person",
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"}
      },
      "required": ["name", "age"]
    }
  }
}
```

**Usage with OpenAI SDK** — drop-in, works on any model:

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

class Person(BaseModel):
    name: str
    age: int

response = client.chat.completions.create(
    model="GLM-5.1",
    messages=[{"role": "user", "content": "Alice, age 30"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": Person.model_json_schema()},
    },
)
# response.choices[0].message.content is guaranteed valid JSON matching Person
data = Person.model_validate_json(response.choices[0].message.content)
```

**How it works**:

1. The JSON schema is compiled to a regex, then to a finite-state automaton
   (via the Rust-backed `outlines-core` library).
2. At each decode step, the FSM reports which token IDs would keep the output
   a valid prefix of the schema.
3. All other tokens have their logits set to `-inf`, so sampling is forced to
   stay on the schema.
4. The model decides **content** (what to say); the FSM enforces **structure**.
5. Compiled schemas are cached across requests (second request with the same
   schema skips the 10–200 ms compile).

**Compatibility**:

| Context | Status |
|---------|--------|
| All models loaded via mlx-lm (Qwen3.5/3.6, GLM-5.1, GLM-4.7, DeepSeek, Llama, Mistral, MiniMax, …) | ✅ Works |
| mlx-vlm models (Gemma 4, Qwen3-VL, GLM4V, …) | ✅ Works (same `logits_processors` contract) |
| Streaming (`stream=True`) | ✅ Works — tokens stream normally, client buffers and parses |
| `tools` present | ⚠️ `response_format` ignored with server-side warning (OpenAI behavior — tools imply `tool_calls` output, which is not JSON-schema content) |
| `--pld` enabled | ⚠️ `response_format` disabled with warning (PLD's speculative multi-token steps are incompatible with FSM state). Either: drop `--pld` for that server, or keep `--pld` and skip `response_format` requests. |

**Validation**: SoloHeaven validates the schema at request time via
`outlines_core.json_schema.build_regex_from_schema()`. Malformed schemas
return HTTP 400 (matches OpenAI behavior) rather than silently falling
through to unconstrained generation.

**Performance cost**: ~1 ms per decode step for the mask build on a ~250K
vocab (benchmarked on M3 Ultra). Typical overhead on a 27 TPS generation is
under 3%. Content quality depends on the model — smaller models may pick
strange string values but the structure is guaranteed.

### Multi-Model Setup

Load multiple models and route requests by the `model` field:

```bash
# Via CLI
mlx-soloheaven --models /path/to/model-A /path/to/model-B /path/to/model-C:no_think_tag

# Via .env
SOLOHEAVEN_MODELS=/path/to/model-A,/path/to/model-B,/path/to/model-C:no_think_tag
```

- Model IDs are derived from directory names (e.g., `Qwen3.5-122B-A10B-8bit`)
- Requests match by exact name or substring (e.g., `model: "qwen3.5-122b"` matches `Qwen3.5-122B-A10B-8bit`)
- `:no_think_tag` suffix disables `<think>` tag injection for models that don't support thinking patterns
- All models share a single GPU lock to prevent Metal concurrency issues
- The first model is the default when no model is specified

### Admin Dashboard

Access the admin dashboard at `http://localhost:8000/admin`:

- **Logs** — Real-time server log streaming via SSE with level filtering and search
- **Models** — Loaded models with default sampling parameters, thinking config, and cache budgets
- **Cache** — Per-model session cache overview (tokens, size, age), base cache stats, disk files
- **Database** — Session/message/memory counts, DB size, session list
- **Reset** — Clear caches only, DB only, or everything (with confirmation)

## Client Integration

### OpenCode

We maintain a [modified fork of OpenCode](https://github.com/joongom/opencode-for-soloheaven) optimized for SoloHeaven:

- **Session-based cache routing** — Uses `user` field with `sessionID:agentName` format for consistent KV cache reuse
- **Stable system prompts** — Moves dynamic date info from system prompt to user message, keeping the system prompt prefix cacheable
- **Thinking display** — Renders `<think>` blocks from reasoning models in a dedicated terminal UI panel
- **Cache-aware compaction** — Appends summaries at end of conversation (not replacing system prompt) for maximum cache reuse
- **Cache hit/miss UI** — Shows cache status on the footer and sidebar

Add to your project's `opencode.json`:

```json
{
  "provider": {
    "mlx-soloheaven": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Soloheaven",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3-coder-next": {
          "id": "Qwen3-Coder-Next-8bit",
          "name": "Qwen3 Coder Next (8bit)",
          "temperature": true,
          "limit": { "context": 200000, "output": 30000 },
          "tool_call": true,
          "options": { "thinking": false }
        },
        "qwen3.5-122b": {
          "id": "Qwen3.5-122B-A10B-8bit",
          "name": "Qwen3.5 122B A10B (8bit)",
          "temperature": true,
          "limit": { "context": 200000, "output": 30000 },
          "tool_call": true
        }
      }
    }
  },
  "model": "mlx-soloheaven/qwen3.5-122b"
}
```

### OpenClaw

We also maintain a [modified fork of OpenClaw](https://github.com/joongom/openclaw-soloheaven) optimized for SoloHeaven with the same cache-aware enhancements.

### Any OpenAI SDK Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    user="my-session-id",  # optional: enables KV cache reuse
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Client Compatibility

SoloHeaven handles common quirks of OpenAI-compatible clients:
- Strips null fields from SSE chunks (`exclude_none`)
- Normalizes list-format `content` to strings
- Converts tool call `arguments` from JSON strings to dicts for Jinja templates
- Strips `<think>` tags from incoming assistant messages to prevent accumulation
- Tolerates client-side content modifications (cleared tool results, truncated assistant messages)
- Supports `developer` role (mapped to `system` for non-OpenAI models)

## Architecture

In the default **process** engine mode, the FastAPI parent holds no MLX state —
it talks to a child process that owns the model and runs generation on its main
thread, over three `multiprocessing.Pipe`s (cmd / resp / ctrl). In `inprocess`
mode the MLX Engine block lives directly inside the FastAPI process.

```
┌─────────────────────────────────────────────┐
│                 Client                       │
│  (OpenCode / Web UI / any OpenAI client)     │
└──────────────┬──────────────────────────────┘
               │ HTTP/SSE
┌──────────────▼──────────────────────────────┐
│  FastAPI Server  (parent process)            │
│  ├── /v1/chat/completions  (OpenAI compat)   │
│  ├── /v1/models            (model listing)   │
│  ├── /v1/sessions[/{id}]   (list/get/delete) │
│  ├── /api/sessions/*/chat  (Web UI)          │
│  ├── /api/admin/*          (Admin dashboard) │
│  ├── /api/sessions/*/settings  (per-session) │
│  ├── /api/sessions/*/compact   (compaction)  │
│  ├── /api/sessions/*/branch    (branching)   │
│  ├── /api/sessions/*/regenerate              │
│  ├── /api/sessions/*/delete-last             │
│  ├── /api/memories[/search]  (web memories)  │
│  └── /health                                 │
│                                              │
│  engine_mode=process → EngineProcessProxy    │
│    (no MLX in parent; looks like MLXEngine)  │
└──────────────┬───────────────────────────────┘
               │ cmd / resp / ctrl Pipes
┌──────────────▼───────────────────────────────┐
│  Child process — MLXEngine                    │
│  (execution_mode="main_thread")               │
│  ├── Generation on the MAIN thread (temp>0 ↑) │
│  ├── Session-based KV cache (in-memory)       │
│  ├── Base cache pool (system prompt reuse)    │
│  ├── Suffix injection (new turn only)         │
│  ├── Thinking budget processor (logits)       │
│  ├── Tool call parser (XML ↔ OpenAI JSON)     │
│  ├── PLD / MTP speculative decoding           │
│  ├── Branch/regen rebuild (reprocess history) │
│  ├── Client disconnect cancellation           │
│  └── Disk persistence (safetensors)           │
├───────────────────────────────────────────────┤
│  Cache Manager                                │
│  ├── Budget-based LRU eviction                │
│  ├── Memory → Disk spillover                  │
│  └── Prefix matching                          │
├───────────────────────────────────────────────┤
│  SQLite Storage                               │
│  ├── Sessions & messages                      │
│  ├── Long-term memories                       │
│  └── Compaction history                       │
└───────────────────────────────────────────────┘
```

### Engine Modes

SoloHeaven runs the MLX engine in one of two modes (`--engine-mode`, default
`process`):

| Mode | Where generation runs | When to use |
|------|----------------------|-------------|
| **`process`** (default) | A **child process**, on its **main thread** (`MLXEngine(execution_mode="main_thread")`). The FastAPI parent holds an `EngineProcessProxy` and talks to the child over cmd/resp/ctrl Pipes. | Single-model serving. Restores throughput lost to a worker-thread temp>0 penalty. |
| **`inprocess`** | Inside the FastAPI process (the original F3 worker-thread engine). | Multi-model (`--models`), or when you need GPU-keepalive (see below). |

**Why process mode is the default:** running generation on a *non-main* thread
with `temp > 0` incurs a ~25% throughput penalty (a Metal / `mx.random`
interaction). Running generation on the **child's main thread** restores
~+30% tok/s at `temp > 0`.

**Constraints of process mode:**
- **Single `--model` only.** Passing `--models` (multi-model) auto-falls back to
  in-process mode.
- **`--draft-model` is single-model only** (so it pairs with process mode).
- **GPU-keepalive is disabled** in process (main-thread) mode — the keepalive
  ping thread (and the dirty-session disk flush it drives) is not started.

> **Disk-persistence caveat for process mode.** The dirty-session disk flush is
> driven by the keepalive loop and an `atexit`/signal shutdown handler, both of
> which are registered only when keepalive starts — i.e. **not** in process
> (main-thread) mode. The child worker also just exits on shutdown without a
> final flush. So in the default process mode, dirty sessions are persisted to
> disk less reliably than in `inprocess` mode (per-request saves still happen on
> the explicit save path, but there is no periodic or shutdown flush). If you
> need the strongest disk persistence of in-flight sessions, run
> `--engine-mode inprocess --gpu-keepalive`.

### How KV Cache Reuse Works

Traditional approach: Every API call tokenizes and processes the **entire** conversation from scratch.

SoloHeaven's approach:
1. Each session stores a KV cache containing all computed attention states (including thinking tokens)
2. On a new turn, only the **suffix** is processed: `\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n`
3. The model continues generation from the cached state — TTFT drops from minutes to milliseconds

This works because OpenAI API clients always send the full conversation history, so we can detect which messages are already cached and only process the delta.

### Cache Modes

The engine reports exactly one of four `cache_mode` values per request:

| Mode | When | Action |
|------|------|--------|
| `hit` | Existing session whose stored messages are a prefix of the request (one **or more** new messages — e.g. tool results) | Reuse all cached tokens, process only the new-message suffix |
| `base_hit` | Cache miss, but the system prompt matches the base cache pool | Clone the base cache, process the remaining tokens |
| `retry` | Same session, but stored messages match exactly (nothing new) | Discard stale cache, full re-process |
| `miss` | New session (no base hit) | Full process from scratch |

> Multi-new-message turns (tool results, several queued messages) are still a
> plain `hit` — the suffix just covers all the new messages. Branching and
> regeneration are **not** cache modes; they rebuild the cache by reprocessing
> the truncated history (see [Conversation Branching](#conversation-branching--regeneration)).

### Message Matching

To maintain cache validity, SoloHeaven compares stored messages with incoming requests:

- **Exact match** — Messages must match for cache reuse
- **System prompt date normalization** — Dynamic dates like `Today's date: Wed Mar 11 2026` are normalized
- **System reminder stripping** — `<system-reminder>` tags injected by clients are removed before comparison
- **Tool result clearing** — Clients may replace old tool results with `[Old tool result content cleared]`; these are accepted
- **Assistant content tolerance** — Last stored assistant message allows content differences (client disconnect/reformatting)
- **Thinking tag stripping** — `<think>...</think>` blocks in assistant messages are stripped before comparison

### Cache Invalidation and Compaction

KV cache is a **prefix-based** data structure. The token sequence is computed linearly from the beginning, so any change to an earlier position invalidates everything after it.

**System message is the first token sequence.** If the system prompt changes (even slightly), the entire cache is invalidated — no partial reuse is possible.

This has a critical implication for **context compaction** (summarizing long conversations to free context space):

```
Wrong approach: Replace the system message with a summary
  -> Entire cache invalidated, full re-processing required

Correct approach: Append a compaction summary as a new system/user message at the END
  -> All preceding tokens remain cache-hit, only the summary + new turn are processed
```

For example, instead of rewriting the system prompt to include a conversation summary, inject a `system` message at the current position:

```
[system] Original system prompt          <- cached (unchanged)
[user] Turn 1 question                   <- cached
[assistant] Turn 1 answer                <- cached
...
[user] Turn 50 question                  <- cached
[assistant] Turn 50 answer               <- cached (everything above is a cache HIT)
[system] "Summary of turns 1-50: ..."    <- NEW: compaction summary appended here
[user] Turn 51 question                  <- NEW: processed as suffix
```

This preserves the full cache prefix while adding context compression at the boundary.

### Hybrid Attention Architecture Note

**Qwen3.5 / Qwen3.6 (DeltaNet hybrid, no sliding window).** Qwen3.5 uses a
hybrid architecture: 36 DeltaNet layers (linear attention with recurrent state)
+ 12 full attention layers (standard KV cache); Qwen3.6-27B is 48 Gated-DeltaNet
+ 16 full-attention (64 layers) and 3.6-35B-A3B is a 40-layer MoE variant. None
of them use a sliding window. Implications:

- **DeltaNet layers (ArraysCache)** store compressed recurrent state — cannot be sliced or partially reused, and the state is non-reversible (can't roll back to an arbitrary position)
- **Full attention layers (KVCache)** store standard key-value pairs — can be sliced but must stay consistent with DeltaNet state
- **No sliding window** means there is no ring-buffer wrap: the cached KV always corresponds to a contiguous prefix of the logical history, so **cache prefix reuse stays valid across long multi-turn chats** with no cold-fills

**Gemma 4 (sliding-window hybrid, ring wrap).** Gemma 4 is 50 sliding-attention
layers (`RotatingKVCache`, window=1024) + 10 full-attention layers (`KVCache`).
Past 1024 cumulative tokens the sliding ring **wraps** — the physical buffer
holds only the most-recent 1024 tokens, no longer a contiguous prefix of the
logical history. SoloHeaven handles this with two fixes:

- **Append-only wrapped-cache reuse** — strict-append turns (the cached history
  is a prefix of the new prompt) reuse the wrapped cache; the suffix is processed
  against it (`RotatingKVCache._update_concat` trims-to-window correctly), so
  TTFT stays ~60 ms instead of cold-filling to ~600 ms. Divergent / edit turns
  past the wrap still cold-fill (the ring no longer holds the needed prefix).
- **Save-time offset↔len reconcile + MTP finally-rollback** — a both-direction
  `cache.offset` ↔ `len(token_ids)` reconcile at save time, plus a
  `finally`-rollback in the MTP drafter loop, eliminated multi-turn cold-fills
  (TTFT 5–14 s → ~60 ms) caused by the cache advancing past the recorded token ids.

### Conversation Branching & Regeneration

SoloHeaven supports branching conversations at any turn, regenerating the last
response, and deleting turns. **Branching and regeneration rebuild the KV cache
by reprocessing the truncated history — there is no instant checkpoint restore.**

**How it works:** `branch_from_turn()` / `truncate_session()` /
`prepare_regenerate()` all funnel into `_rebuild_session()`, which:

1. Takes the truncated message list (history up to the branch/regen point)
2. Seeds from the base cache pool if the system-prompt prefix matches
3. Prefills the remaining tokens into a fresh KV cache
4. Stores the new session — the next user message is then a normal cache `hit`

The operation returns `method: "build"` with a `build_time` (typically a couple
of seconds for short histories, longer for long ones — it is a full prefill of
the truncated prompt, not a snapshot restore).

```
Source session: [sys, user1, asst1, user2, asst2, user3, asst3]

Branch at Turn 2:
  1. Truncate to [sys, user1, asst1, user2]
  2. Try base cache for the system-prompt prefix
  3. Prefill the remaining tokens → fresh KV cache  (method: "build")
  4. New session ready — next message is a cache HIT
```

> The SQLite DB stores branch **metadata** (parent/child session links) only;
> the KV cache itself is always reconstructed by reprocessing.

## API Reference

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming & non-streaming) |
| GET | `/v1/models` | List available models |
| GET | `/v1/sessions` | List active sessions (engine view) |
| GET | `/v1/sessions/{id}` | Get session info |
| DELETE | `/v1/sessions/{id}` | Delete a session's cache |
| POST | `/v1/sessions/{id}/compact` | Compact a session (OpenAI-namespaced) |

### Web Chat API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sessions` | Create session |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{id}` | Get session metadata |
| PATCH | `/api/sessions/{id}` | Update session (title, system_prompt) |
| DELETE | `/api/sessions/{id}` | Delete session |
| GET | `/api/sessions/{id}/messages` | Get all messages |
| POST | `/api/sessions/{id}/chat` | Send message (SSE streaming) |
| POST | `/api/sessions/{id}/branch` | Branch conversation at a specific turn |
| POST | `/api/sessions/{id}/regenerate` | Remove last turn and regenerate |
| POST | `/api/sessions/{id}/delete-last` | Delete last user+assistant turn |
| GET | `/api/cache/stats` | Cache statistics |

### Memories (Web UI)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/memories` | Create a long-term memory |
| GET | `/api/memories` | List memories |
| GET | `/api/memories/search` | Search memories |

### Settings & Compaction

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/sessions/{id}/settings` | Get session settings |
| PATCH | `/api/sessions/{id}/settings` | Update settings |
| POST | `/api/sessions/{id}/compact` | Trigger context compaction |
| GET | `/api/sessions/{id}/compaction-status` | Get context utilization |
| GET | `/api/sessions/{id}/compaction-prompt` | Preview the compaction prompt |
| GET | `/api/sessions/{id}/compactions` | List past compactions for the session |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/logs/stream` | SSE real-time log streaming |
| GET | `/api/admin/logs/recent` | Recent log entries |
| GET | `/api/admin/models` | Loaded models with default parameters |
| GET | `/api/admin/cache` | Cache overview (all models) |
| GET | `/api/admin/db` | Database overview |
| POST | `/api/admin/cache/reset` | Reset caches only |
| POST | `/api/admin/db/reset` | Reset the database only |
| POST | `/api/admin/reset-all` | Reset caches and database |

### Extra Request Fields

SoloHeaven extends the OpenAI API with optional fields:

```json
{
  "user": "session-id",
  "thinking": true,
  "thinking_budget": 4096,
  "top_k": 40,
  "min_p": 0.05,
  "repetition_penalty": 1.1
}
```

- `user` — Session ID for KV cache reuse (see below)
- `thinking` — Override server config for this request (`true`/`false`)
- `thinking_budget` — Override thinking token budget for this request
- `top_k` — Top-k sampling override for this request
- `min_p` — Min-p threshold override for this request
- `repetition_penalty` — Repetition penalty override for this request
- `frequency_penalty` / `presence_penalty` — OpenAI-standard penalties (mapped to `repetition_penalty`)

**`user` field behavior:**

| `user` value | Behavior |
|---|---|
| Unique ID (e.g., `"session-abc123"`) | Dedicated KV cache per session. Cache persists across requests and server restarts. **Recommended for all clients.** |
| Same ID across different conversations | Cache conflict — messages won't match, falls back to `retry` mode (full re-process). Use unique IDs per conversation. |
| Omitted or empty | Falls back to `"anon"` — all requests share one cache slot. Only works for single-client single-conversation use. |

For OpenCode / OpenClaw, the client typically sends a consistent session ID automatically. For custom integrations, generate a UUID per conversation and pass it as `user`.

## Project Structure

```
src/mlx_soloheaven/
├── cli.py              # CLI entry point (argparse + env fallback)
├── config.py           # Configuration dataclass (ModelConfig + Config)
├── server.py           # FastAPI app factory, multi-model setup
├── engine/
│   ├── mlx_engine.py      # Core: model loading, generation, KV cache, drafter (~3700 lines)
│   ├── thinking.py        # Thinking budget logits processor
│   ├── tool_parser.py     # XML tool calls <-> OpenAI JSON conversion
│   ├── compaction.py      # Context compaction engine
│   ├── pld.py             # Prompt Lookup Decoding (draft-model-free speculation)
│   ├── structured.py      # JSON-schema FSM logits masking (response_format)
│   ├── types.py           # mlx-free GenerationResult/CompletionResult for IPC
│   ├── process_client.py  # EngineProcessProxy (parent side of process mode)
│   ├── process_worker.py  # Child-process worker (main-thread generation)
│   └── process_protocol.py # cmd/resp/ctrl Pipe message protocol
├── api/
│   ├── openai_compat.py  # /v1/chat/completions, /v1/models
│   ├── chat.py           # /api/sessions/*/chat (web UI)
│   ├── admin.py          # /api/admin/* (admin dashboard)
│   ├── settings.py       # /api/sessions/*/settings
│   ├── compaction.py     # /api/sessions/*/compact
│   └── schemas.py        # Pydantic request/response models
├── cache/
│   └── manager.py      # Budget-based LRU cache manager
├── storage/
│   └── database.py     # SQLite: sessions, messages, memories, compactions
└── web/                # Built-in web UI
    ├── index.html      # Chat interface
    ├── admin.html      # Admin dashboard
    ├── style.css       # Dark theme, responsive design
    └── app.js          # Client-side logic (~500 lines)
```

## Future Directions

### Stateful Session Protocol

The current OpenAI API is **stateless** — every request sends the full conversation history. In a 100K token conversation, only the last user message is new, yet 99.9% of the payload is re-transmitted and re-compared on the server. A dedicated protocol could eliminate this overhead:

- **Delta-only transmission** — As long as the session is alive, send only the new user message (the server already holds previous assistant responses)
- **Mode-based prompt routing** — Use `mode: "code"`, `mode: "chat"`, etc. to select server-side system prompts, avoiding repeated transmission of long system prompts from the client
- **Mid-conversation system message injection** — API for inserting system messages at the current position (for compaction summaries, mode switches, etc.) while preserving the cache prefix
- **Server-driven compaction** — Server monitors context utilization and triggers compaction automatically, without client involvement

```
# Current (OpenAI API, stateless)
Client -> Server: [system, user1, assistant1, user2, assistant2, ..., userN]  # full history every time

# Dedicated protocol (stateful)
Client -> Server: {session: "abc", message: "new question"}  # new message only
Server -> Client: {delta: "response text", cache_tokens: 50000}
```

### GPU Concurrency

Currently uses a **Global GPU Lock** (queue mode) designed for single-user use. One request generates at a time; others wait in queue.

A future option could split GPU resources for concurrent processing:

- **Queue mode (current)** — Sequential single-request processing. Maximum throughput per request, optimal for single-user
- **Concurrent mode** — Split Metal GPU resources across multiple requests. Useful for multi-user or parallel agent scenarios. Per-request TPS is reduced, but overall wait time decreases

### Other Ideas

- **Native MTP / DFlash for Qwen3.6** — Qwen3.6 ships built-in MTP heads
  (`mtp_num_hidden_layers=1`) and z-lab publishes DFlash drafters, but mlx-vlm
  currently strips the MTP weights. Wiring native MTP (cf. mlx-lm PR #990 /
  MTPLX, ~1.5–2.2x on Qwen3.6-27B) or DFlash into the mlx-vlm path is a future
  lever. Note: native MTP is ineffective on the 35B-A3B MoE (~11% acceptance —
  a single MTP layer can't predict expert routing).

> Generic draft-model **speculative decoding is already implemented** for
> Gemma 4 (see [Speculative Decoding (MTP)](#speculative-decoding-mtp)) and for
> trimmable-cache models via [PLD](#prompt-lookup-decoding-pld).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) & [mlx-lm](https://github.com/ml-explore/mlx-examples) — Apple's ML framework that makes local LLM inference on Apple Silicon possible
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance async web framework
- [Highlight.js](https://highlightjs.org/) & [marked.js](https://marked.js.org/) — Code highlighting and Markdown rendering for the web UI
- [OpenCode](https://github.com/opencode-ai/opencode) — The terminal AI assistant that inspired the KV cache optimization work

Special thanks to [Clover Games](https://www.clovergames.com) — Lord of Heroes, It's Me (#Me), Heaven x Hells, and Ayakashi Rise forever.

## License

[MIT](LICENSE)
