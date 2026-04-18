"""
Benchmark: Can we offload KV quantization (demotion) to a background thread
on a secondary MLX stream without impacting main-thread attention ops?

Tests:
  A. Baseline:      main op alone.
  B. Same stream:   main op + quant on default stream (serial contention).
  C. New stream:    main op + quant on secondary stream, same Python thread
                    (via async_eval).
  D. BG thread +
     new stream:    main op in main thread, quant on secondary stream driven
                    from a Python thread (concurrent with GIL release).
  E. async_eval:    main op async_eval'd, quant on another stream async_eval'd,
                    single thread — mirrors how mlx_lm.generate does it.

We shape workloads to match realistic SoloHeaven demotion:
  - Main "attention": 2048x2048 matmul chain (stand-in for decode-time GEMM).
  - Quant work: 10x mx.quantize on a (1, 8, 1024, 128) bf16 chunk — this is
    one ~1 MB chunk per layer, which is the unit we'd demote.
"""

import os, time, threading, statistics, sys
import mlx.core as mx

# Shapes chosen so main op ≈ 80–150 ms on M3 Ultra.
MAIN_N = 2048
MAIN_ITERS = 200        # matmul chain length
QUANT_CHUNK = (1, 8, 1024, 128)  # (batch, heads, seq, head_dim) bf16
QUANT_COUNT = 10        # how many chunks the bg worker will quantize
WARMUP = 3
TRIALS = 8

DEV = mx.default_device()
print(f"device={DEV}")
print(f"gpu_stream_default={mx.default_stream(DEV)}")

# ----- workload builders ------------------------------------------------------

def make_main_inputs():
    a = mx.random.normal((MAIN_N, MAIN_N)).astype(mx.bfloat16)
    mx.eval(a)
    return a

def main_op(a, stream=None):
    """Simulated attention: long matmul chain, returns a single scalar-ish array."""
    x = a
    ctx = mx.stream(stream) if stream is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for _ in range(MAIN_ITERS):
            x = x @ x.T
            # keep numerics bounded
            x = x * mx.array(0.5, dtype=mx.bfloat16)
        return x
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

def make_quant_inputs(n):
    arrs = [mx.random.normal(QUANT_CHUNK).astype(mx.bfloat16) for _ in range(n)]
    mx.eval(arrs)
    return arrs

def quant_once(arr, stream=None):
    ctx = mx.stream(stream) if stream is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        q, s, z = mx.quantize(arr.reshape(-1, QUANT_CHUNK[-1]), bits=4, group_size=64)
        return q, s, z
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

# ----- trials -----------------------------------------------------------------

def trial_solo(a):
    t0 = time.perf_counter()
    x = main_op(a)
    mx.eval(x)
    return time.perf_counter() - t0

def trial_quant_solo(arrs):
    t0 = time.perf_counter()
    outs = [quant_once(ar) for ar in arrs]
    mx.eval(outs)
    return time.perf_counter() - t0

def trial_same_stream(a, arrs):
    """Main + quant, both on default stream, single thread. Pure serial."""
    t0 = time.perf_counter()
    x = main_op(a)
    outs = [quant_once(ar) for ar in arrs]
    mx.eval(x, outs)
    return time.perf_counter() - t0

def trial_async_eval_sep_stream(a, arrs, bg_stream):
    """Main op on default stream, quant on bg stream. Same thread.
       Kick both with async_eval then join — the mlx_lm pattern."""
    t0 = time.perf_counter()
    x = main_op(a)                        # default stream
    outs = [quant_once(ar, stream=bg_stream) for ar in arrs]
    mx.async_eval(x)
    mx.async_eval(outs)
    mx.eval(x, outs)
    return time.perf_counter() - t0

def trial_bg_thread_sep_stream(a, arrs, bg_stream, done_event):
    """Main op in main thread (default stream), quant driven by bg thread on bg stream."""
    def worker():
        outs = [quant_once(ar, stream=bg_stream) for ar in arrs]
        mx.eval(outs)
        done_event.set()
    done_event.clear()
    th = threading.Thread(target=worker)
    t0 = time.perf_counter()
    th.start()
    x = main_op(a)
    mx.eval(x)
    main_done = time.perf_counter() - t0
    th.join()
    total = time.perf_counter() - t0
    return main_done, total

# ----- run --------------------------------------------------------------------

bg_stream = mx.new_stream(DEV)
print(f"bg_stream          ={bg_stream}")

a = make_main_inputs()
arrs = make_quant_inputs(QUANT_COUNT)

# warmup everything (compile kernels)
for _ in range(WARMUP):
    trial_solo(a)
    trial_quant_solo(arrs)
    trial_same_stream(a, arrs)
    trial_async_eval_sep_stream(a, arrs, bg_stream)
    m, t = trial_bg_thread_sep_stream(a, arrs, bg_stream, threading.Event())

def stats(xs):
    return f"min={min(xs)*1000:7.2f} ms  median={statistics.median(xs)*1000:7.2f} ms  max={max(xs)*1000:7.2f} ms"

solo_main = [trial_solo(a) for _ in range(TRIALS)]
solo_quant = [trial_quant_solo(arrs) for _ in range(TRIALS)]
same_stream = [trial_same_stream(a, arrs) for _ in range(TRIALS)]
async_sep = [trial_async_eval_sep_stream(a, arrs, bg_stream) for _ in range(TRIALS)]

done = threading.Event()
bg_main_only = []
bg_total = []
for _ in range(TRIALS):
    m, t = trial_bg_thread_sep_stream(a, arrs, bg_stream, done)
    bg_main_only.append(m)
    bg_total.append(t)

print()
print("=== RESULTS ===")
print(f"A. main solo                         : {stats(solo_main)}")
print(f"   quant solo ({QUANT_COUNT} chunks)           : {stats(solo_quant)}")
print(f"B. main + quant, SAME stream (serial): {stats(same_stream)}")
print(f"C. main + quant, DIFFERENT streams   :")
print(f"     async_eval, same thread          : {stats(async_sep)}")
print(f"D. main + quant, bg THREAD + bg stream:")
print(f"     main-thread op wall time         : {stats(bg_main_only)}")
print(f"     total wall (incl. bg join)       : {stats(bg_total)}")

solo_med = statistics.median(solo_main)
bg_med = statistics.median(bg_main_only)
async_med = statistics.median(async_sep)
same_med = statistics.median(same_stream)
quant_med = statistics.median(solo_quant)
print()
print("=== INTERPRETATION ===")
print(f"Median solo main op             : {solo_med*1000:7.2f} ms")
print(f"Median solo quant               : {quant_med*1000:7.2f} ms")
print(f"Ideal serial (A_main + A_quant) : {(solo_med+quant_med)*1000:7.2f} ms")
print(f"Observed same-stream combined   : {same_med*1000:7.2f} ms  "
      f"(overhead vs ideal = {(same_med - (solo_med+quant_med))*1000:+.2f} ms)")
print(f"Observed async_eval sep-stream  : {async_med*1000:7.2f} ms  "
      f"(speedup vs same-stream = {(same_med-async_med)*1000:+.2f} ms, "
      f"{(1-async_med/same_med)*100:.1f}%)")
print(f"Observed bg-thread main-only    : {bg_med*1000:7.2f} ms  "
      f"(overhead vs solo main = {(bg_med-solo_med)*1000:+.2f} ms, "
      f"{(bg_med/solo_med-1)*100:+.1f}%)")
print()
print(f"IMPACT on main-thread generation op: {(bg_med/solo_med-1)*100:+.2f}%")
print(f"Below 5% threshold? {'YES' if abs(bg_med/solo_med-1) < 0.05 else 'NO'}")
