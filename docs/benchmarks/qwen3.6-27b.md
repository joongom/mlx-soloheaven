# Qwen3.6-27B (MLX 8-bit) — measurement ledger

Machine: Apple M1 Ultra, 128 GiB, ~96% free at load (nothing else resident).
Model: `lmstudio-community/Qwen3.6-27B-MLX-8bit`, 28 GB on disk — dense, 64
layers, hidden 5120, **vocab 248,320**. Served by `mlx-soloheaven` through the
stock mlx-lm path (no custom kernels, no native replay runtime — those are
DeepSeek-V4 only).

Sampling held identical across every row below: `temp=0.6 top_k=20 top_p=0.95
rep_pen=1.1 --no-thinking --prefill-step-size 2048`. Workload is the stored
27-turn KV-cache research conversation from the session DB (session
`96871ffb41b3`), replayed in order through `POST /api/sessions/{id}/chat` so
that every turn after the first exercises prefix reuse.

## 1. Multi-turn to 11k tokens (MTP, block size 1)

18 turns on ONE session. `prompt` is the full context the turn was fed;
`new` is what actually had to be prefilled after the cache hit.

| turn | prompt | new | gen | tok/s |
|---|---|---|---|---|
| 1 | 38 | 38 (miss) | 395 | 17.6 |
| 5 | 2254 | 26 | 591 | 17.9 |
| 10 | 5515 | 27 | 595 | 18.5 |
| 14 | 8044 | ~30 | 555 | 18.6 |
| 18 | **10569** | 30 | 597 | 17.8 |

**Flat: 17.6-18.9 tok/s from a 38-token prompt to a 10.5k one.** Final context
11,166 tokens. Prefix reuse hit on all 17 follow-up turns — the last one reused
10,539 cached tokens and prefilled 30. Answers stayed on topic and correctly
structured throughout; turns 13 and 14 are the same question verbatim in the
source script and produced two consistent, differently-framed answers rather
than a repeat or a confusion.

No custom-kernel work is involved here, which is the point: **Qwen serves
correctly and its decode speed does not decay with context**, unlike the
DeepSeek-V4 native path before Stage 4m.

## 2. MTP speculative decoding: works, does NOT pay (2026-08-03)

The MTP head (`mlx-community/Qwen3.6-27B-MTP-4bit`, 247 MB) loads and runs —
`kind=qwen_mtp num_head_layers=1 weights=31 (strict)` — and its drafts are
genuinely accepted. It still buys nothing:

| config | tok/s (turns 1-8) | draft acceptance |
|---|---|---|
| **no MTP** | **18.2-18.5** | — |
| MTP, `--draft-block-size 1` | 17.6-18.9 | 0.63-0.76 of 1 |
| MTP, `--draft-block-size 3` | **12.1-15.1** | 1.09-1.48 of 3 |

Block size 1 is a wash; block size 3 is a **29% LOSS**. Acceptance is not the
problem — at block 1 roughly two thirds of drafts are accepted, which should
be worth ~1.65 tokens per target pass. Getting 1.0x means draft + verify
overhead is eating essentially the entire saving, and widening the block just
buys more of that overhead than it recovers.

Not yet explained, and worth stating as open rather than guessed: the draft
head is one decoder layer plus the target's lm_head (5120 x 248,320 — 1.27 GB
at 8-bit, ~4.5% of the model's total weight traffic), which accounts for only
a small part of the gap. Locating the rest needs the same per-kernel profiling
that found the DeepSeek-V4 cliff. **Until then, serve this model WITHOUT the
draft**: `./start_qwen3.6_27b.sh` (adjust its temperature — the MTP script
uses 0.6, the plain one 1.0).

Reproduce:

```bash
./start_qwen3.6_27b_mtp.sh                 # or start_qwen3.6_27b.sh for no draft
# then replay the stored conversation against http://127.0.0.1:8000
#   POST /api/sessions          -> {"id": ...}
#   POST /api/sessions/{id}/chat {"content": <turn>, "stream": true}
# per-turn truth is in the server log:
#   [KV Cache] ... HIT | reusing N cached tokens + M suffix tokens
#   [Generate] ... tps=X
```

Note when reading the stream yourself: the `start` SSE frame's `cache_hit` is
a PREFLIGHT value and reads false even on turns that hit. The authoritative
per-turn cache outcome is the `[KV Cache]` log line.
