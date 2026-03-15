# MLX SoloHeaven

**Single-user LLM inference server with KV cache optimization for Apple Silicon.**

SoloHeaven turns your Mac into a personal AI server with sub-second response times, even on 100K+ token conversations. It exposes an OpenAI-compatible API, making it a drop-in backend for tools like [OpenCode](https://opencode.ai), [Continue](https://continue.dev), [OpenClaw](https://openclaw.com), or any OpenAI SDK client.

![SoloHeaven Demo](docs/demo.gif)

## Key Features

- **Session-based KV cache reuse** — Only process new tokens each turn, not the entire conversation
- **250x TTFT improvement** — From 126s to 0.5s at 131K context tokens
- **99.9% token savings** — Cache hit reuses all previously computed KV states
- **Multi-model support** — Load multiple models simultaneously, route requests by model name
- **Per-model thinking control** — Enable/disable `<think>` tags per model (e.g., `:no_think_tag` for non-reasoning models)
- **Thinking budget control** — Configurable token limit for reasoning models, with per-request override
- **Budget-based cache eviction** — No time-based TTL, evicts LRU only when memory/disk budget exceeded
- **GPU keepalive** — Optional Metal idle prevention with periodic micro-computations (`--gpu-keepalive`)
- **Full OpenAI API compatibility** — Streaming SSE, tool calling, `developer` role, `/v1/chat/completions`, `/v1/models`
- **Built-in web UI** — Chat interface with model selector, live thinking display, TPS stats, cache hit indicators, branch/regenerate/delete controls
- **Admin dashboard** — Real-time log viewer, cache/DB overview, and reset controls at `/admin`
- **Conversation branching** — Fork any conversation at any turn with instant KV cache restore from DeltaNet checkpoints
- **Regenerate & Delete** — Re-roll the last response or remove turns, with cache state automatically restored
- **DeltaNet checkpoint persistence** — Turn-level snapshots saved to disk, survive server restarts
- **Disk persistence** — KV caches survive server restarts via safetensors serialization
- **Client disconnect handling** — Frees the generation lock immediately, tolerates content mismatches on reconnect
- **Base cache pool** — System prompt KV caches shared across sessions for fast cold starts

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

### Systematic Comparison (11 turns, Qwen3.5-122B-A10B-bf16, M3 Ultra 512GB)

| Strategy | Avg TTFT | Avg TPS | TPS Drop (T0→T10) | Final Cache | Quality |
|----------|----------|---------|--------------------|----|---------|
| Baseline (no cache) | 71.9s | 26.1 | 18% | N/A | Thinking loops |
| Cached (no budget) | 0.74s | 26.2 | 18% | 88K | Thinking loops |
| **Optimized (adopted)** | **0.69s** | **27.55** | **7.4%** | **32K** | **5.0/5** |
| RotatingKV (8192) | 0.49s | 28.38 | 2.4% | 8K fixed | 4.32/5 |
| KV 8-bit | 0.52s | 26.04 | 16.5% | 32K | 4.64/5 |
| ThinkTrim | 0.66s | 28.48 | 4.9% | 26.5K | Layer mismatch |
| NoCache (rebuild) | 10.46s | 28.52 | 4.8% | N/A | Good but slow |

### Production Metrics (Qwen3.5-122B-A10B-bf16)

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| TTFT (Turn 10, ~131K tokens) | 126s | 0.5s | **~250x** |
| Token savings per request | 0% | 99.9% | — |
| Generation TPS | 27.5 tok/s | 27.5 tok/s | No degradation |
| Quality (5-point scale) | 4.64 | 4.64 | No degradation |

### Real-World Usage: Qwen3.5-122B-A10B-bf16 (1 hour with OpenCode)

**Machine:** Mac Studio M3 Ultra 512GB / 4TB

Over 191 messages in a real coding session:

- **Cache hit rate**: 89% (170/191)
- **Avg TTFT on cache hit**: 0.5s
- **Avg TTFT on cache miss**: 45s
- **Total tokens saved**: 11.8M (99.9% reduction)
- **Peak context**: 131,072 tokens

### Real-World Usage: Qwen3.5-397B-A17B-MLX-8bit (OpenClaw agent session)

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

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ (recommended: 3.12 via [pyenv](https://github.com/pyenv/pyenv))
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
  --disk-budget-gb      On-disk KV cache budget in GB (default: 100)
  --data-dir            Directory for SQLite DB and cache files (default: ./data)
  --no-thinking         Disable thinking mode globally
  --gpu-keepalive       Keep Metal GPU warm to avoid idle penalty (env: SOLOHEAVEN_GPU_KEEPALIVE)
  --verbose, -v         Enable verbose logging (env: SOLOHEAVEN_VERBOSE)
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

```
┌─────────────────────────────────────────────┐
│                 Client                       │
│  (OpenCode / Web UI / any OpenAI client)     │
└──────────────┬──────────────────────────────┘
               │ HTTP/SSE
┌──────────────▼──────────────────────────────┐
│  FastAPI Server                              │
│  ├── /v1/chat/completions  (OpenAI compat)   │
│  ├── /v1/models            (model listing)   │
│  ├── /api/sessions/*/chat  (Web UI)          │
│  ├── /api/admin/*          (Admin dashboard) │
│  ├── /api/sessions/*/settings  (per-session) │
│  ├── /api/sessions/*/compact   (compaction)  │
│  ├── /api/sessions/*/branch    (branching)   │
│  ├── /api/sessions/*/regenerate              │
│  ├── /api/sessions/*/delete-last             │
│  └── /health                                 │
├──────────────────────────────────────────────┤
│  MLX Engine (per model, shared GPU lock)     │
│  ├── Session-based KV cache (in-memory)      │
│  ├── Base cache pool (system prompt reuse)   │
│  ├── Suffix injection (new turn only)        │
│  ├── Thinking budget processor (logits)      │
│  ├── Tool call parser (XML ↔ OpenAI JSON)    │
│  ├── GPU keepalive thread (optional)         │
│  ├── DeltaNet checkpoints (branch/regen)     │
│  ├── Client disconnect cancellation          │
│  └── Disk persistence (safetensors + ckpt)   │
├──────────────────────────────────────────────┤
│  Cache Manager                               │
│  ├── Budget-based LRU eviction               │
│  ├── Memory → Disk spillover                 │
│  └── Prefix matching                         │
├──────────────────────────────────────────────┤
│  SQLite Storage                              │
│  ├── Sessions & messages                     │
│  ├── Long-term memories                      │
│  └── Compaction history                      │
└──────────────────────────────────────────────┘
```

### How KV Cache Reuse Works

Traditional approach: Every API call tokenizes and processes the **entire** conversation from scratch.

SoloHeaven's approach:
1. Each session stores a KV cache containing all computed attention states (including thinking tokens)
2. On a new turn, only the **suffix** is processed: `\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n`
3. The model continues generation from the cached state — TTFT drops from minutes to milliseconds

This works because OpenAI API clients always send the full conversation history, so we can detect which messages are already cached and only process the delta.

### Cache Modes

| Mode | When | Action |
|------|------|--------|
| `hit` | Existing session, one new user message | Reuse all cached tokens, process suffix only |
| `hit_multi` | Multiple new messages (e.g., tool results) | Reuse cached prefix, process all new messages |
| `base_hit` | System prompt matches base cache pool | Clone base cache, process remaining tokens |
| `base_build` | New system prompt, no base cache | Process system tokens, register base cache |
| `branch` | Incoming shorter than stored, prefix matches | Restore from DeltaNet checkpoint, process new suffix |
| `retry` | Same session, messages don't match | Discard stale cache, full re-process |
| `miss` | New session | Full process from scratch |

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

Qwen3.5 uses a hybrid architecture: 36 DeltaNet layers (linear attention with recurrent state) + 12 full attention layers (standard KV cache). This has important implications:

- **DeltaNet layers (ArraysCache)** store compressed recurrent state — cannot be sliced or partially reused
- **Full attention layers (KVCache)** store standard key-value pairs — can be sliced but must stay consistent with DeltaNet state
- **DeltaNet state is non-reversible** — cannot be rolled back to an arbitrary position. SoloHeaven solves this with turn-level DeltaNet snapshots (checkpoints) that enable instant cache restoration at any turn boundary

### Conversation Branching & Regeneration

SoloHeaven supports branching conversations at any turn, regenerating responses, and deleting turns — all with instant KV cache restoration.

**The challenge:** In Qwen3.5's hybrid architecture, KVCache (full attention) can be sliced by offset, but DeltaNet (linear attention) is a recurrent state that cannot be reversed. To branch at turn 5 of a 10-turn conversation, you need the exact DeltaNet state at turn 5.

**The solution:** Save a DeltaNet snapshot (`deepcopy`) at every turn boundary. These checkpoints are persisted to disk alongside the main cache.

```
Turn 1: [sys, user1] → generate → checkpoint #1 (DeltaNet snapshot + KV offset)
Turn 2: [sys, user1, asst1, user2] → generate → checkpoint #2
Turn 3: [sys, user1, asst1, user2, asst2, user3] → generate → checkpoint #3
...
Branch at Turn 2:
  1. Find checkpoint #2
  2. Restore DeltaNet state (deepcopy from snapshot)
  3. Slice KVCache to checkpoint offset
  4. New session ready — next message is a cache HIT
```

**Branch modes:**

| Mode | When | Speed |
|------|------|-------|
| **COPY** | Branch at last turn (full conversation copy) | Instant (deepcopy) |
| **CHECKPOINT** | Branch at earlier turn, checkpoint exists | Instant (snapshot restore + KV slice) |
| **BUILD** | No checkpoint available (e.g., pre-existing sessions) | 2-3s (prefill from scratch) |

**Disk persistence:**

Each session stores two files:
- `session_{id}.safetensors` — KV cache (existing)
- `session_{id}_ckpt.safetensors` — DeltaNet snapshots at each turn boundary

After server restart, checkpoints are loaded from disk — branching and regeneration remain instant without needing to rebuild.

**Storage cost:** ~5MB per checkpoint (36 DeltaNet layers × fixed-size state), max 50 checkpoints per session (~250MB). Negligible compared to the KV cache itself (100-200MB per session).

## API Reference

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming & non-streaming) |
| GET | `/v1/models` | List available models |

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

### Settings & Compaction

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/sessions/{id}/settings` | Get session settings |
| PATCH | `/api/sessions/{id}/settings` | Update settings |
| POST | `/api/sessions/{id}/compact` | Trigger context compaction |
| GET | `/api/sessions/{id}/compaction-status` | Get context utilization |

### Admin

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/logs/stream` | SSE real-time log streaming |
| GET | `/api/admin/logs/recent` | Recent log entries |
| GET | `/api/admin/models` | Loaded models with default parameters |
| GET | `/api/admin/cache` | Cache overview (all models) |
| GET | `/api/admin/db` | Database overview |
| POST | `/api/admin/reset` | Reset caches, DB, or all |

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
│   ├── mlx_engine.py   # Core: model loading, generation, KV cache (~1300 lines)
│   ├── thinking.py     # Thinking budget logits processor
│   ├── tool_parser.py  # XML tool calls <-> OpenAI JSON conversion
│   └── compaction.py   # Context compaction engine
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

- **Speculative decoding** — Use a small draft model followed by verification from the large model to improve TPS

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) & [mlx-lm](https://github.com/ml-explore/mlx-examples) — Apple's ML framework that makes local LLM inference on Apple Silicon possible
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance async web framework
- [Highlight.js](https://highlightjs.org/) & [marked.js](https://marked.js.org/) — Code highlighting and Markdown rendering for the web UI
- [OpenCode](https://github.com/opencode-ai/opencode) — The terminal AI assistant that inspired the KV cache optimization work

Special thanks to [Clover Games](https://www.clovergames.com) — Lord of Heroes, It's Me (#Me), Heaven x Hells, and Ayakashi Rise forever.

## License

[MIT](LICENSE)
