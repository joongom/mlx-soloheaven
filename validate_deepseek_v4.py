#!/usr/bin/env python
"""Validation ladder for the DeepSeek-V4 MLX port (spec step 8).

Subcommands:
  smoke                     load the model and greedy-decode the probe prompts
                            (Korean first — it degrades before English does)
  logits "PROMPT" OUT.json  dump final-position logits for a RAW (untemplated)
                            prompt, for comparison against
                            `ds4 -p "PROMPT" --dump-logits ref.json -n 1 --temp 0`
  compare OURS.json DS4.json   top-1 / top-20 overlap / KL report
  agree DS4LP.json "PROMPT"    teacher-forced top-1 agreement over a ds4
                            `--dump-logprobs` continuation — the number to rank
                            two builds on
  bench [n_tokens]          decode throughput: mx.compile'd path vs the native
                            Metal replay runtime (NativeDecoder), tok/s each

Point at a specific build with DSV4_MODEL=/path/to/build.

ds4's build is 2-bit and ours is 2-bit-mixed, so equality is not expected;
structural errors push top-1 agreement to chance, quantization noise does not.
Tokenization must match — if agreement is at chance, check BOS handling FIRST.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

MODEL = os.environ.get(
    "DSV4_MODEL",
    "~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-mixed",
)


def free_gib() -> float:
    page = 16384
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = 0
    for line in out.splitlines():
        for key in ("Pages free", "Pages inactive", "Pages purgeable"):
            if line.startswith(key):
                pages += int(line.split()[-1].rstrip("."))
    return pages * page / 2**30


def load():
    import os
    from pathlib import Path

    if free_gib() < 100 and "--force" not in sys.argv:
        raise SystemExit(
            f"only {free_gib():.0f} GiB reclaimable — something big is running; "
            "pass --force to load anyway"
        )
    import sys as _sys

    import mlx_soloheaven.models.deepseek_v4 as v4

    _sys.modules["mlx_lm.models.deepseek_v4"] = v4
    from mlx_lm.utils import load_model

    path = os.path.expanduser(MODEL)
    model, _ = load_model(Path(path))
    # AutoTokenizer consults AutoConfig for the unknown model_type and
    # transformers 5.x's generic config trips over rope_scaling; loading the
    # concrete class reads only the tokenizer files.
    from transformers import PreTrainedTokenizerFast

    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tokenizer = TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path))
    return model, tokenizer


#: Greedy probes. Korean is the sensitive case: it degrades first under
#: aggressive expert quantization while English still reads cleanly, so a
#: build that only ever gets checked in English looks better than it is.
SMOKE_PROMPTS = [
    ("ko/chat", "안녕하세요.", True),
    ("ko/chat", "대한민국의 수도는?", True),
    ("en/chat", "What is the capital of France?", True),
    ("ko/raw", "대한민국의 수도는", False),
]


def cmd_smoke(n_tokens: int = 32) -> None:
    import mlx.core as mx

    model, tokenizer = load()
    for label, prompt, templated in SMOKE_PROMPTS:
        if templated:
            # exactly what the server sends (chat mode, thinking off)
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            ids = list(ids["input_ids"] if not isinstance(ids, list) else ids)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        cache = model.make_cache()
        logits = model(mx.array([ids]), cache)
        mx.eval(logits)
        assert mx.isfinite(logits).all(), f"non-finite logits for {prompt!r}"
        out, tok = [], logits[0, -1].argmax()
        for _ in range(n_tokens):
            out.append(int(tok))
            tok = model(tok.reshape(1, 1), cache)[0, -1].argmax()
            mx.eval(tok)
        print(f"[{label}] {prompt!r}\n    -> {tokenizer.decode(out)!r}\n")


def cmd_logits(prompt: str, out_path: str) -> None:
    import mlx.core as mx

    model, tokenizer = load()
    ids = tokenizer.encode(prompt)
    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
    mx.eval(logits)
    arr = [float(x) for x in logits]
    with open(out_path, "w") as f:
        json.dump(
            {
                "prompt_ids": ids,
                "vocab": len(arr),
                "argmax_token": {"id": int(logits.argmax())},
                "logits": arr,
            },
            f,
        )
    print(f"wrote {out_path}; argmax {int(logits.argmax())}, ids {ids[:8]}...")


#: Held-out probes for perplexity. Korean is included deliberately: expert
#: quantization damage shows up there long before it shows up in English.
PPL_TEXTS = {
    "ko": (
        "대한민국의 수도는 서울이며, 인구는 약 950만 명이다. 서울은 한강을 "
        "중심으로 강북과 강남으로 나뉘고, 정치와 경제, 문화의 중심지 역할을 "
        "한다. 조선 시대에는 한양이라 불렸고, 경복궁과 창덕궁 같은 궁궐이 "
        "지금도 남아 있다. 지하철은 세계에서 가장 복잡한 노선망 가운데 "
        "하나로 꼽히며, 하루 수백만 명이 이용한다."
    ),
    "en": (
        "The Apollo program was a series of missions run by NASA between 1961 "
        "and 1972. Its stated goal was to land humans on the Moon and return "
        "them safely to Earth. Apollo 11 achieved that in July 1969, when Neil "
        "Armstrong and Buzz Aldrin walked on the lunar surface while Michael "
        "Collins remained in orbit aboard the command module."
    ),
    "code": (
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
}


def cmd_ppl() -> None:
    """Teacher-forced perplexity per probe — the metric that ranks two builds.

    Unlike top-1 agreement against ds4 (a similarity measure over a handful of
    positions, so noisy AND indirect), this scores every position against the
    text that was actually written, and needs no reference implementation.
    """
    import math

    import mlx.core as mx

    model, tokenizer = load()
    total_nll = total_n = 0.0
    for name, text in PPL_TEXTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        logits = model(mx.array([ids])).astype(mx.float32)
        lp = logits[0, :-1] - mx.logsumexp(logits[0, :-1], axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).mean()
        mx.eval(nll)
        n = len(ids) - 1
        total_nll += float(nll) * n
        total_n += n
        print(f"  {name:<5} tokens={n:<4} nll={float(nll):.4f}  ppl={math.exp(float(nll)):.2f}")
    print(f"  {'ALL':<5} tokens={int(total_n):<4} nll={total_nll / total_n:.4f}  "
          f"ppl={math.exp(total_nll / total_n):.2f}")


def cmd_agree(ds4_logprobs: str, prompt: str) -> None:
    """Teacher-forced agreement against a ds4 ``--dump-logprobs`` run.

    Feeds ds4's own greedy continuation through our model and asks, at every
    position, whether our argmax matches ds4's pick. This is the quantitative
    quality number: it is insensitive to sampling and to divergence compounding,
    unlike free generation, so two builds can be ranked on it directly.

    Produce the reference with:
      ds4 --raw -p "PROMPT" --dump-logprobs out.json --logprobs-top-k 5 -n 32 --temp 0
    """
    import mlx.core as mx

    d = json.load(open(ds4_logprobs))
    picked = [s["selected"]["id"] for s in d["steps"]]
    top5 = [[t["token"]["id"] for t in s["top_logprobs"]] for s in d["steps"]]

    model, tokenizer = load()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) != d["prompt_tokens"]:
        raise SystemExit(
            f"tokenization mismatch: ours {len(ids)} vs ds4 {d['prompt_tokens']} — "
            "the prompt must be the one ds4 was run with, and ds4 needs --raw"
        )
    logits = model(mx.array([ids + picked]))
    mx.eval(logits)

    agree = sum(
        int(logits[0, len(ids) - 1 + i].argmax()) == t for i, t in enumerate(picked)
    )
    in5 = sum(
        int(logits[0, len(ids) - 1 + i].argmax()) in top5[i] for i in range(len(picked))
    )
    n = len(picked)
    print(f"top-1 agreement:   {agree}/{n}  ({agree / n:.0%})")
    print(f"ours in ds4 top-5: {in5}/{n}  ({in5 / n:.0%})")


def cmd_bench(n_tokens: int = 64, prefill: int = 8) -> None:
    """Decode-throughput benchmark: the mx.compile'd path vs the external
    Metal replay runtime (NativeDecoder), on the REAL model. Reports tok/s for
    each. Run yourself (loads the 88 GB weights): `python validate_deepseek_v4.py
    bench [n_tokens] [prefill]`. Needs ~100 GiB reclaimable (or --force)."""
    import time

    import mlx.core as mx

    # keep the 88 GB of weights GPU-resident, else decode pages every token
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tokenizer = load()
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. "
                           "In a distant land,", add_special_tokens=False)[:prefill]

    def timed(fn, warmup=4):
        for _ in range(warmup):
            fn()
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            fn()
        mx.synchronize()
        return n_tokens / (time.perf_counter() - t0)

    # --- compiled path ---
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache)
    mx.eval(logits)
    mx.synchronize()
    tok = logits[0, -1].argmax().reshape(1, 1)

    def step_compiled():
        nonlocal tok
        tok = model(tok, cache)[0, -1].argmax().reshape(1, 1)
        mx.eval(tok)

    tps_compiled = timed(step_compiled)
    print(f"compiled decode:  {tps_compiled:5.1f} tok/s  ({1e3 / tps_compiled:.1f} ms/token)")

    # --- native replay path ---
    from mlx_soloheaven.native.decoder import NativeDecoder

    cap = int(os.environ.get("SOLOHEAVEN_DSV4_MAX_CONTEXT", "8192"))

    def native_tps(barriers: bool) -> float:
        dec = NativeDecoder(model, max_context=cap, barriers=barriers)
        dec.offset = len(ids)
        for i, c in enumerate(cache):
            if getattr(c, "ring", None) is not None:
                dec.set_ring(i, c.ring)
        seed = int(logits[0, -1].argmax())

        def step():
            nonlocal seed
            lg = dec.decode(seed)
            mx.eval(lg)
            seed = int(lg.argmax())

        return timed(step)

    tps_native = native_tps(True)
    print(f"native  decode:   {tps_native:5.1f} tok/s  ({1e3 / tps_native:.1f} ms/token)")
    # DIAGNOSTIC (not correct output): strip the per-dispatch barriers. If this is
    # much faster, the cost is blanket-barrier serialization (fix: dependency-aware
    # barriers); if it's the same, our kernels are simply slower than MLX's.
    tps_nobar = native_tps(False)
    print(f"native (no barr): {tps_nobar:5.1f} tok/s  ({1e3 / tps_nobar:.1f} ms/token)  "
          f"[diagnostic — wrong output]")
    print(f"speedup vs compiled: {tps_native / tps_compiled:.2f}x  "
          f"(target >=25 tok/s: {'MET' if tps_native >= 25 else 'not yet'})")


def cmd_compare(ours_path: str, ds4_path: str) -> None:
    import numpy as np

    a = np.array(json.load(open(ours_path))["logits"], np.float64)
    b = np.array(json.load(open(ds4_path))["logits"], np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    ta, tb = np.argsort(-a)[:20], np.argsort(-b)[:20]
    pa = np.exp(a - a.max()) / np.exp(a - a.max()).sum()
    pb = np.exp(b - b.max()) / np.exp(b - b.max()).sum()
    kl = float((pb * (np.log(pb + 1e-12) - np.log(pa + 1e-12))).sum())
    print(f"top-1 agree: {ta[0] == tb[0]}  (ours {ta[0]}, ds4 {tb[0]})")
    print(f"top-20 overlap: {len(set(ta) & set(tb))}/20")
    print(f"KL(ds4 || ours): {kl:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "smoke":
        cmd_smoke()
    elif cmd == "logits":
        cmd_logits(sys.argv[2], sys.argv[3])
    elif cmd == "ppl":
        cmd_ppl()
    elif cmd == "agree":
        cmd_agree(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    elif cmd == "bench":
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 64
        cmd_bench(n_tokens=n)
    else:
        raise SystemExit(__doc__)
