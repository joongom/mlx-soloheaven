"""Benchmark: does the Gemma 4 MTP drafter's token-acceptance HOLD or
COLLAPSE as generation crosses the RotatingKVCache sliding-window wrap?

Context
-------
- Target: gemma-4-31B-it-MLX-8bit (50 sliding layers, sliding_window=1024,
  10 full layers every 6th). RotatingKVCache wraps once cumulative tokens > 1024.
- Drafter: gemma-4-31B-it-assistant-bf16 (mlx-vlm "gemma4_assistant" MTP head).
- Historical: pre-wrap mean_accepted ~1.17; an OLD bug collapsed post-wrap
  mean_accepted to ~0.26 (net-negative, garbage loop until max_tokens).
  We patched those bugs (B1 temporal-order rewrite + B2-v2 kv_offset clamp;
  B3 removed). This bench measures the POST-FIX numbers, bucketed by cumulative
  position, to decide if "restore MTP past the wrap" is worth pursuing.

This script BYPASSES the SoloHeaven engine guards (the wrap-imminent drafter
kill-switch and the cold-fill-on-wrap path) and drives mlx-vlm DIRECTLY so the
drafter runs CONTINUOUSLY through the wrap. It still installs the B1/B2-v2
monkey-patches (engine helper `_install_mtp_wrap_patches`) BEFORE generation,
exactly as the engine worker-thread init does, so the fixes are ACTIVE.

Thread/stream affinity
----------------------
mlx-vlm 0.5.0's module-global `mlx_vlm.generate.generation_stream` is a
ThreadLocalStream created on the importing thread. Model weights have THREAD
AFFINITY: VLM load + drafter load + generation must all run on the SAME thread.
The engine pins everything to a dedicated single worker thread; here we do the
simplest equivalent: run EVERYTHING on the main thread, and re-install a fresh
thread-local `generation_stream` on the main thread (the same swap the engine's
`_vlm_worker_init` does) so every mlx-vlm call inherits a slot we registered.

Usage
-----
    .venv/bin/python bench_mtp_wrap_acceptance.py 2>&1 | tee /tmp/bench.out
    .venv/bin/python bench_mtp_wrap_acceptance.py --temp 0.6 --max-tokens 2400 \
        --log /tmp/bench_mtp_wrap_acceptance.log
"""

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, "src")

import mlx.core as mx

# mlx-vlm direct API (NOT the engine's _run_vlm wrapper).
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import stream_generate as vlm_stream_generate
from mlx_vlm.speculative import load_drafter

# The B1/B2-v2 wrap-bug monkey-patches live as a module-level helper in the
# engine. Import the function ONLY (importing the module does not load any
# model weights) and call it before generation so the fixes are active.
from mlx_soloheaven.engine.mlx_engine import _install_mtp_wrap_patches


DEFAULT_TARGET = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/gemma-4-31B-it-MLX-8bit"
)
DEFAULT_DRAFTER = os.path.expanduser(
    "~/.lmstudio/models/mlx-community/gemma-4-31B-it-assistant-bf16"
)
DEFAULT_PROMPT = (
    "다음을 아주 자세히 단계별로 설명해줘: 트랜스포머 어텐션과 KV 캐시가 동작하는 "
    "원리, 그리고 추론 속도 최적화 기법들을 가능한 길게 설명해줘."
)

# Gemma 4 RotatingKVCache max_size == sliding_window.
SLIDING_WINDOW = 1024

# Degradation thresholds for the repetition metric on the tail of the output.
UNIQUE_12GRAM_RATIO_THRESHOLD = 0.35  # below this => DEGRADED (loopy)
LONGEST_RUN_THRESHOLD = 60            # repeating-substring run (chars) => DEGRADED


class Tee:
    """Write to stdout AND a log file so the run is `tail -f`-able."""

    def __init__(self, log_path):
        self._fh = open(log_path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        self._stdout.write(line + "\n")
        self._stdout.flush()
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def install_generation_stream_on_this_thread(log):
    """Replicate the engine's `_vlm_worker_init` stream swap, but on the
    main (current) thread, since we run load + generate here.

    mlx-vlm 0.5.0 creates `generation_stream` as a ThreadLocalStream on the
    importing thread. If load/gen happen on a different thread than the one
    that imported the module, MLX raises
    `RuntimeError: There is no Stream(gpu, N) in current thread.`
    By creating a fresh thread-local stream on THIS thread and warming its
    slot, every mlx-vlm `with mx.stream(generation_stream):` block inside
    `_mtp_rounds` uses a slot we registered here.
    """
    import mlx_vlm.generate as _mvg

    old = getattr(_mvg, "generation_stream", None)
    new = mx.new_thread_local_stream(mx.default_device())
    _mvg.generation_stream = new
    with mx.stream(new):
        probe = mx.array([1.0]) * 1.0
        mx.eval(probe)
    log(f"[stream] installed generation_stream on main thread: "
        f"old={old!r} new={new!r} id={id(new)}")


def build_prompt_token_ids(processor, prompt_text, log):
    """Tokenize the user prompt via the chat template (matches engine wiring:
    apply_chat_template(tokenize=True, add_generation_prompt=True)).
    Returns a flat list[int]."""
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = [{"role": "user", "content": prompt_text}]
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except TypeError:
        # Some templates reject unknown kwargs; retry minimally.
        result = tokenizer.apply_chat_template(messages, tokenize=True)
    if hasattr(result, "input_ids"):
        ids = list(result.input_ids)
    else:
        ids = list(result)
    # Flatten in case of a nested batch dimension.
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    log(f"[prompt] token_len={len(ids)} | text[:60]={prompt_text[:60]!r}")
    return ids


# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------
#
# `draft_model.accept_lens[i]` = number of DRAFTED tokens accepted in round i.
# Each round emits `accept_lens[i] + 1` tokens total (the +1 is the always-
# emitted target "bonus" token). The first bonus token (before the very first
# round) is yielded once outside the rounds; we fold it into the prompt offset
# so the per-round cumulative arithmetic stays simple and slightly conservative.
#
#   cumulative_start_of_round_i = prompt_len + 1 + sum_{j<i} (accept_lens[j] + 1)
#
# Bucket each round by the cumulative position at the START of the round
# (i.e. the RotatingKVCache offset the drafter sees when it begins drafting):
#   PRE-WRAP : cumulative_start < 1024
#   WRAP->2048: 1024 <= cumulative_start < 2048
#   2048->4096: cumulative_start >= 2048
def bucket_rounds(accept_lens, prompt_len):
    buckets = {
        "PRE-WRAP (<1024)": [],
        "WRAP->2048 [1024,2048)": [],
        "2048->4096 (>=2048)": [],
    }
    # +1 for the first bonus token emitted before the rounds begin.
    cumulative = prompt_len + 1
    for a in accept_lens:
        start = cumulative
        if start < SLIDING_WINDOW:
            buckets["PRE-WRAP (<1024)"].append(a)
        elif start < 2 * SLIDING_WINDOW:
            buckets["WRAP->2048 [1024,2048)"].append(a)
        else:
            buckets["2048->4096 (>=2048)"].append(a)
        cumulative += a + 1
    return buckets, cumulative


# --------------------------------------------------------------------------
# Degradation detection
# --------------------------------------------------------------------------
def repetition_metrics(text, tail_chars=400):
    """Compute simple repetition metrics on the LAST `tail_chars` chars.

    Returns dict with:
      - unique_12gram_ratio: |unique char 12-grams| / |total 12-grams|
        (low => loopy / repeating).
      - longest_repeat_run: length (chars) of the longest immediately-
        repeating unit run (e.g. "이해해를이해해를..." -> a long run).
    """
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    n = len(tail)

    # 12-gram uniqueness ratio.
    if n >= 12:
        grams = [tail[i:i + 12] for i in range(n - 12 + 1)]
        ratio = len(set(grams)) / len(grams)
    else:
        ratio = 1.0

    # Longest immediately-repeating substring run: for each small unit length
    # u, find the longest run where tail[k:k+u] repeats back-to-back.
    longest_run = 0
    for u in range(1, 13):
        if n < 2 * u:
            break
        run = 0
        i = 0
        while i + u <= n:
            j = i + u
            count = 1
            while j + u <= n and tail[j:j + u] == tail[i:i + u]:
                count += 1
                j += u
            if count >= 2:
                this_run = count * u
                if this_run > run:
                    run = this_run
                i = j
            else:
                i += u
        if run > longest_run:
            longest_run = run

    degraded = (
        (n >= 12 and ratio < UNIQUE_12GRAM_RATIO_THRESHOLD)
        or longest_run >= LONGEST_RUN_THRESHOLD
    )
    return {
        "tail_len": n,
        "unique_12gram_ratio": ratio,
        "longest_repeat_run": longest_run,
        "degraded": degraded,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tokens", type=int, default=2400,
                    help="enough to cross 1024 and reach the 2048 bucket")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0.0 (greedy) or 0.6")
    ap.add_argument("--block-size", type=int, default=3,
                    help="MTP draft block size")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                    help="short (~40-100 token) Korean reasoning prompt")
    ap.add_argument("--target", type=str, default=DEFAULT_TARGET)
    ap.add_argument("--drafter", type=str, default=DEFAULT_DRAFTER)
    ap.add_argument("--log", type=str,
                    default="/tmp/bench_mtp_wrap_acceptance.log")
    args = ap.parse_args()

    log = Tee(args.log)
    log("=" * 78)
    log("Gemma 4 MTP wrap-acceptance benchmark (B1+B2-v2 patches ACTIVE, "
        "engine guards BYPASSED)")
    log("=" * 78)
    log(f"target      : {args.target}")
    log(f"drafter     : {args.drafter}")
    log(f"max_tokens  : {args.max_tokens} | temp: {args.temp} | "
        f"block_size: {args.block_size}")
    log(f"sliding_win : {SLIDING_WINDOW} (RotatingKVCache wrap point)")
    log(f"log         : {args.log}")
    log("-" * 78)

    # STEP 0: install generation_stream on THIS (main) thread BEFORE any
    # mlx-vlm work — load + generate both run here (thread affinity).
    install_generation_stream_on_this_thread(log)

    # STEP 1: load target + drafter on this thread.
    t_load0 = time.perf_counter()
    log("[load] loading target VLM model + processor ...")
    model, processor = vlm_load(args.target)
    log("[load] loading MTP drafter via mlx_vlm.speculative.load_drafter ...")
    drafter, draft_kind = load_drafter(args.drafter, kind=None)
    t_load = time.perf_counter() - t_load0
    log(f"[load] done in {t_load:.1f}s | draft_kind={draft_kind!r} | "
        f"drafter_block_size="
        f"{getattr(getattr(drafter, 'config', None), 'block_size', None)}")

    # STEP 2: install the B1/B2-v2 wrap-bug patches (idempotent; the function
    # monkey-patches mlx_vlm.generate._mtp_rounds and Gemma4TextModel.__call__).
    # Module-level _HOT_PATH_FAST defaults to False, so the patches run their
    # FULL correctness path (temporal-order rewrite) — which is exactly what we
    # want when driving the drafter through the wrap by hand.
    applied = _install_mtp_wrap_patches()
    log(f"[patch] _install_mtp_wrap_patches() -> applied_now={applied} "
        f"(False just means already-installed; patches are ACTIVE either way)")

    # STEP 3: tokenize prompt and reset acceptance bookkeeping.
    prompt_ids = build_prompt_token_ids(processor, args.prompt, log)
    prompt_len = len(prompt_ids)
    input_ids = mx.array([prompt_ids])
    drafter.accept_lens = []  # reset right before the run (per task spec)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        draft_model=drafter,
        draft_kind=draft_kind,
        draft_block_size=args.block_size,
    )

    # STEP 4: ONE continuous stream_generate through the wrap.
    log("-" * 78)
    log("[gen] starting continuous stream_generate (drafter ACTIVE, "
        "running THROUGH the wrap) ...")
    text = ""
    gen_tokens = 0
    last_result = None
    err = None
    t_gen0 = time.perf_counter()
    try:
        for r in vlm_stream_generate(model, processor, "", **gen_kwargs):
            last_result = r
            seg = getattr(r, "text", None)
            if seg:
                text += seg
            gt = getattr(r, "generation_tokens", None)
            if gt is not None:
                gen_tokens = gt
    except Exception as e:  # noqa: BLE001 — always report partial results
        err = e
        log("[gen] EXCEPTION during generation:")
        log(traceback.format_exc())
    t_gen = time.perf_counter() - t_gen0

    # STEP 5: read accept_lens (always, even on error). The OLD bug produced a
    # garbage loop that ran to max_tokens — hitting max_tokens with low
    # acceptance is itself a signal.
    accept_lens = list(getattr(drafter, "accept_lens", []) or [])
    if last_result is not None and getattr(last_result, "generation_tokens", None):
        gen_tokens = last_result.generation_tokens
    if gen_tokens == 0 and accept_lens:
        gen_tokens = sum(a + 1 for a in accept_lens) + 1  # rough fallback
    overall_tps = gen_tokens / t_gen if t_gen > 0 else 0.0

    log("-" * 78)
    log(f"[gen] finished | gen_tokens={gen_tokens} | gen_time={t_gen:.1f}s | "
        f"overall_tps={overall_tps:.2f}")
    log(f"[gen] n_rounds(accept_lens)={len(accept_lens)} | "
        f"hit_max_tokens={gen_tokens >= args.max_tokens}")
    if err is not None:
        log(f"[gen] NOTE: generation raised {type(err).__name__}: {err} "
            f"— stats below are PARTIAL")

    # STEP 6: bucket rounds by cumulative offset and report.
    buckets, final_cumulative = bucket_rounds(accept_lens, prompt_len)
    log("=" * 78)
    log("PER-BUCKET ACCEPTANCE (bucketed by cumulative offset at round START)")
    log("=" * 78)
    header = (f"{'bucket':<26} {'n_rounds':>9} {'mean_accepted':>14} "
              f"{'throughput_x':>13}")
    log(header)
    log("-" * len(header))
    overall_accepted = 0
    overall_rounds = 0
    for name, lens in buckets.items():
        n = len(lens)
        overall_rounds += n
        overall_accepted += sum(lens)
        if n == 0:
            log(f"{name:<26} {0:>9} {'-':>14} {'-':>13}")
            continue
        mean_acc = sum(lens) / n
        throughput_x = sum(a + 1 for a in lens) / n  # mean(accepted+1)
        log(f"{name:<26} {n:>9} {mean_acc:>14.3f} {throughput_x:>13.3f}")
    log("-" * len(header))
    if overall_rounds:
        log(f"{'ALL ROUNDS':<26} {overall_rounds:>9} "
            f"{overall_accepted / overall_rounds:>14.3f} "
            f"{(overall_accepted + overall_rounds) / overall_rounds:>13.3f}")
    log(f"final_cumulative_offset (approx) = {final_cumulative}")

    # Quick verdict on the wrap question.
    pre = buckets["PRE-WRAP (<1024)"]
    post = buckets["WRAP->2048 [1024,2048)"]
    if pre and post:
        pre_mean = sum(pre) / len(pre)
        post_mean = sum(post) / len(post)
        log("-" * 78)
        log(f"[verdict] pre-wrap mean_accepted={pre_mean:.3f} | "
            f"post-wrap mean_accepted={post_mean:.3f}")
        if post_mean < 0.3:
            log("[verdict] POST-WRAP COLLAPSE — acceptance fell to old-bug "
                "territory (<0.3). Drafter net-negative past the wrap.")
        elif post_mean >= 0.8 * pre_mean:
            log("[verdict] HOLDS — post-wrap acceptance retained vs pre-wrap. "
                "Restoring MTP past the wrap looks worthwhile.")
        else:
            log("[verdict] PARTIAL — post-wrap acceptance degraded but did NOT "
                "collapse. Marginal; inspect throughput_x vs 1.0.")
    else:
        log("[verdict] insufficient rounds in one of the buckets — increase "
            "--max-tokens or shorten --prompt to ensure the wrap is crossed.")

    # STEP 5 (degradation): repetition metric + tail eyeball.
    rep = repetition_metrics(text, tail_chars=400)
    log("=" * 78)
    log("DEGRADATION CHECK (last ~400 chars)")
    log("=" * 78)
    log(f"tail_len={rep['tail_len']} | unique_12gram_ratio="
        f"{rep['unique_12gram_ratio']:.3f} (thr {UNIQUE_12GRAM_RATIO_THRESHOLD}) "
        f"| longest_repeat_run={rep['longest_repeat_run']} chars "
        f"(thr {LONGEST_RUN_THRESHOLD})")
    log(f"DEGRADED={rep['degraded']}  "
        f"({'garbage-loop symptom detected' if rep['degraded'] else 'looks clean'})")
    log("-" * 78)
    log("[output tail ~300 chars] (eyeball for '이해해를 이해해를...' loops):")
    log(repr(text[-300:]))

    log("=" * 78)
    log(f"SUMMARY: load={t_load:.1f}s | gen={t_gen:.1f}s | "
        f"gen_tokens={gen_tokens} | overall_tps={overall_tps:.2f} | "
        f"rounds={len(accept_lens)} | "
        f"final_offset~={final_cumulative} | DEGRADED={rep['degraded']}"
        + (f" | ERROR={type(err).__name__}" if err else ""))
    log("=" * 78)
    log.close()


if __name__ == "__main__":
    main()
