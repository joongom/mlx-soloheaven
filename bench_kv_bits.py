"""Benchmark kv_bits on a loaded model.

Usage:
    .venv/bin/python bench_kv_bits.py --model /path/to/model [--context 8192]

Compares baseline (kv_bits=0), kv_bits=8, kv_bits=4 on TPS and memory.
"""

import argparse
import sys
import time
import gc

import mlx.core as mx


def measure(engine, prompt_tokens, max_tokens, label):
    """Run generation and measure TPS + memory."""
    import os, psutil
    proc = psutil.Process(os.getpid())

    mx.clear_cache()
    gc.collect()
    mem_before = proc.memory_info().rss / 1e9

    t0 = time.perf_counter()
    t_first = None
    count = 0
    text = ""
    for r in engine.generate_stream(
        prompt_tokens,
        session_id=f"bench_{label}",
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        if r.text:
            text += r.text
        if t_first is None and r.text:
            t_first = time.perf_counter()
            ttft = t_first - t0
        count += 1
        if r.finish_reason:
            break

    elapsed = time.perf_counter() - t0
    gen_time = elapsed - (t_first - t0) if t_first else elapsed
    tps = count / gen_time if gen_time > 0 else 0

    mx.eval([])  # sync
    mem_after = proc.memory_info().rss / 1e9

    print(f"\n=== {label} ===")
    print(f"  tokens generated: {count}")
    print(f"  TTFT: {ttft*1000:.0f}ms" if t_first else "  TTFT: N/A")
    print(f"  generation time: {gen_time:.2f}s")
    print(f"  TPS: {tps:.2f}")
    print(f"  memory: before={mem_before:.1f}GB, after={mem_after:.1f}GB, delta={mem_after-mem_before:.1f}GB")
    print(f"  first 100 chars: {text[:100]!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context", type=int, default=2048, help="Prompt length")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--bits", type=int, nargs="+", default=[0, 8, 4])
    args = p.parse_args()

    sys.path.insert(0, "src")
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    for bits in args.bits:
        print(f"\n{'#'*60}")
        print(f"# Loading model with kv_bits={bits}")
        print(f"{'#'*60}")

        cfg = Config(model_path=args.model, kv_bits=bits)
        cfg.gpu_keepalive = False
        eng = MLXEngine(cfg)
        eng.load_model()

        msgs = [{"role": "user", "content": "Write a " + ("long " * (args.context // 10)) + "story about AI."}]
        label = f"kv_bits={bits}"
        measure(eng, msgs, args.max_tokens, label)

        # Clean up before next
        del eng
        mx.clear_cache()
        gc.collect()


if __name__ == "__main__":
    main()
