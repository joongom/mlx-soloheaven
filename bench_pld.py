"""Benchmark PLD vs baseline on a given model.

Workloads:
  1. Prompt-echo: prompt contains a sequence; model should copy (ideal for PLD)
  2. Agent-like: long prompt + tool-output-style question (realistic)
  3. Creative: short prompt + free generation (worst case for PLD)

Usage:
  .venv/bin/python bench_pld.py --model <path> [--max-tokens 200]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, "src")

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import MLXEngine


def run_test(engine, messages, max_tokens, label):
    t0 = time.perf_counter()
    t_first = None
    count = 0
    text = ""
    for r in engine.generate_stream(
        messages,
        session_id=f"bench_{label}_{int(t0)}",
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        if r.text:
            text += r.text
            count += 1
            if t_first is None:
                t_first = time.perf_counter()
        if r.finish_reason:
            break
    elapsed = time.perf_counter() - t0
    gen_time = elapsed - (t_first - t0) if t_first else elapsed
    tps = count / gen_time if gen_time > 0 else 0
    ttft = (t_first - t0) * 1000 if t_first else 0
    return {
        "tokens": count,
        "ttft_ms": round(ttft),
        "gen_time_s": round(gen_time, 2),
        "tps": round(tps, 2),
        "preview": text[:80].replace("\n", " "),
    }


WORKLOADS = {
    "echo": {
        "messages": [{
            "role": "user",
            "content": (
                "Repeat the following list exactly: "
                "apple, banana, cherry, date, elderberry, fig, grape, honeydew, "
                "imbe, jackfruit, kiwi, lemon, mango, nectarine, orange, papaya.\n\n"
                "Now repeat:"
            ),
        }],
        "desc": "Prompt-echo (ideal PLD case)",
    },
    "summarize": {
        "messages": [{
            "role": "user",
            "content": (
                "Summarize this in one sentence:\n\n"
                "The transformer architecture, introduced in the 2017 paper "
                "'Attention is All You Need' by Vaswani et al., revolutionized "
                "natural language processing by relying entirely on attention "
                "mechanisms rather than recurrence or convolution. This allowed "
                "for much more parallelization during training and led to the "
                "current era of large language models."
            ),
        }],
        "desc": "Summarize (moderate PLD potential)",
    },
    "creative": {
        "messages": [{
            "role": "user",
            "content": "Write a haiku about AI.",
        }],
        "desc": "Creative (worst PLD case)",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--workloads", nargs="+", default=["echo", "summarize", "creative"])
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"Loading model: {os.path.basename(args.model)}")
    print(f"{'='*70}\n")

    results = {}

    for pld_on in [False, True]:
        label = "PLD" if pld_on else "baseline"
        print(f"\n### Mode: {label} ###")

        cfg = Config(
            model_path=args.model,
            pld_enabled=pld_on,
            pld_num_draft_tokens=10,
            pld_ngram_k=3,
        )
        cfg.gpu_keepalive = False
        eng = MLXEngine(cfg)
        eng.load_model()

        results[label] = {}
        for name in args.workloads:
            wl = WORKLOADS[name]
            r = run_test(eng, wl["messages"], args.max_tokens, f"{label}_{name}")
            results[label][name] = r
            print(f"  [{name}] {wl['desc']}")
            print(f"    tokens={r['tokens']}, TTFT={r['ttft_ms']}ms, "
                  f"gen={r['gen_time_s']}s, TPS={r['tps']}")
            print(f"    preview: {r['preview']}")

        del eng
        import gc, mlx.core as mx
        gc.collect()
        mx.clear_cache()

    # Comparison
    print(f"\n\n{'='*70}")
    print(f"{'Workload':<15} {'Baseline TPS':<15} {'PLD TPS':<15} {'Speedup':<10}")
    print(f"{'='*70}")
    for name in args.workloads:
        b = results["baseline"][name]["tps"]
        p_tps = results["PLD"][name]["tps"]
        sp = p_tps / b if b > 0 else 0
        print(f"{name:<15} {b:<15} {p_tps:<15} {sp:.2f}x")


if __name__ == "__main__":
    main()
