#!/usr/bin/env python3
"""Measure end-to-end decode tok/s through SoloHeaven's OpenAI-compatible
streaming endpoint (before/after the streaming-pipeline optimization).

Hits POST /v1/chat/completions with stream=true and times the steady-state
inter-token interval (excludes prefill/TTFT). Run with the SoloHeaven server
already up. Usage:
    python bench_stream_tps_client.py --url http://127.0.0.1:8000 --model <id> --n 3
"""
import argparse
import json
import time
import urllib.request


def one_run(url, model, prompt, max_tokens, temperature):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    n_tok = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tok += 1
    ttft = (t_first - t_start) if t_first else 0.0
    # Steady-state decode tps: tokens after the first, over the time between
    # first and last token (excludes prefill/TTFT).
    decode_span = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    decode_tps = ((n_tok - 1) / decode_span) if decode_span > 0 else 0.0
    return {"tokens": n_tok, "ttft_s": round(ttft, 3), "decode_tps": round(decode_tps, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=3, help="number of timed runs")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--prompt", default="한국어로 트랜스포머 어텐션과 KV 캐시 원리, 추론 최적화 기법을 단계별로 아주 자세히 설명해줘.")
    args = ap.parse_args()

    # discover model id if not given
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as r:
                model = (json.load(r).get("data") or [{}])[0].get("id", "")
        except Exception as e:
            print(f"could not auto-discover model ({e}); pass --model")
            return
    print(f"model={model} | url={args.url} | n={args.n} | max_tokens={args.max_tokens} temp={args.temperature}")
    results = []
    for i in range(args.n):
        r = one_run(args.url, model, args.prompt, args.max_tokens, args.temperature)
        print(f"  run {i+1}: tokens={r['tokens']} ttft={r['ttft_s']}s decode_tps={r['decode_tps']}")
        results.append(r["decode_tps"])
    if results:
        avg = sum(results) / len(results)
        print(f"=== avg decode_tps = {avg:.1f} (best {max(results):.1f}) ===")


if __name__ == "__main__":
    main()
