#!/usr/bin/env python
"""Validation ladder for the DeepSeek-V4 MLX port (spec step 8).

Subcommands:
  smoke                     load the converted model, check a forward pass is
                            finite, greedy-decode a short continuation
  logits "PROMPT" OUT.json  dump final-position logits for a RAW (untemplated)
                            prompt, for comparison against
                            `ds4 -p "PROMPT" --dump-logits ref.json -n 1 --temp 0`
  compare OURS.json DS4.json   top-1 / top-20 overlap / KL report

ds4's build is 2-bit and ours is 2-bit-mixed, so equality is not expected;
structural errors push top-1 agreement to chance, quantization noise does not.
Tokenization must match — if agreement is at chance, check BOS handling FIRST.
"""

from __future__ import annotations

import json
import subprocess
import sys

MODEL = (
    "~/.lmstudio/models/mlx-soloheaven/DeepSeek-V4-Flash-0731-MLX-2bit-mixed"
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

    if free_gib() < 100 and "--force" not in sys.argv:
        raise SystemExit(
            f"only {free_gib():.0f} GiB reclaimable — something big is running; "
            "pass --force to load anyway"
        )
    import sys as _sys

    import mlx_soloheaven.models.deepseek_v4 as v4

    _sys.modules["mlx_lm.models.deepseek_v4"] = v4
    from mlx_lm.utils import load as mlx_load

    return mlx_load(os.path.expanduser(MODEL))


def cmd_smoke() -> None:
    import mlx.core as mx

    model, tokenizer = load()
    ids = tokenizer.encode("대한민국의 수도는")
    print("prompt ids:", ids)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache)
    mx.eval(logits)
    assert mx.isfinite(logits).all(), "non-finite logits"
    print("prefill ok:", logits.shape, "argmax:", int(logits[0, -1].argmax()))
    out = []
    tok = logits[0, -1].argmax()
    for _ in range(24):
        out.append(int(tok))
        tok = model(tok.reshape(1, 1), cache)[0, -1].argmax()
        mx.eval(tok)
    print("continuation:", tokenizer.decode(out))


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
    elif cmd == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
