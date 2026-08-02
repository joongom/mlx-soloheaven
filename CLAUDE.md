# mlx-soloheaven — agent guide

MLX-based LLM serving on Apple Silicon: FastAPI server (`/v1/chat/completions`
+ native session API) whose reason to exist is the session KV machinery —
prefix reuse, compaction, branch/regenerate. Engine: `src/mlx_soloheaven/engine/mlx_engine.py`.
Custom architectures (EXAONE-4.5, DeepSeek-V4) live in `src/mlx_soloheaven/models/`
and register into mlx-lm via `register_extra_architectures()`.

## Commands

```bash
.venv/bin/python -m pytest -q                       # full suite (~30 s)
.venv/bin/python -m ruff check <changed files>      # lint SCOPED — repo has legacy noise
.venv/bin/python validate_deepseek_v4.py ppl        # quality gate for V4 builds (also: smoke)
.venv/bin/python convert_deepseek_v4.py SRC DST     # fp8/fp4 -> MLX, ~34 min, resumable
./start_deepseek_v4_flash_0731.sh                   # serve V4 (94.5 GB — nothing else loaded!)
```

## Documentation system — READ THIS BEFORE CLAIMING OR MEASURING ANYTHING

Rules live in `docs/DOCUMENTATION.md`. Non-negotiables:
* numbers with method + machine state (wired vs unwired!), recorded at the
  moment of measurement, in `docs/benchmarks/`;
* negative results (refuted theories, reverted optimizations) are recorded
  with the failed prediction — several already saved days, do not delete;
* every spec keeps a Status block at the top saying done / broken / next.

Live documents:
* `docs/specs/deepseek-v4-mlx-port.md` — the V4 port: status, quantization
  ceiling, decode-speed analysis, upstream survey, remaining paths.
* `docs/benchmarks/deepseek-v4.md` — every measurement + reproduce steps.

## Hard-won facts (do not relearn these expensively)

* This machine reboots/OOMs when ~90 GB models load beside other residents
  or right after ~100 GB of file IO — check `memory_pressure` (needs ~90%+
  free) before loading; never benchmark un-wired.
* ds4 (`~/workspace/numenore/ds4`, MIT) is the numerical oracle for V4;
  its one-shot mode silently applies a chat template — use `--raw` for any
  logit comparison.
* Decode-speed work on V4: per-launch overhead is ~4 µs (not the
  bottleneck); the 85 ms/token is a sum of medium inefficiencies — see the
  benchmark ledger before optimizing anything.
* User-personal working preferences (language, push policy, scope checks)
  are NOT in this file — they live in the agent's local memory and must not
  be committed.
