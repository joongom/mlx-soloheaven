"""
bench_stream_overlap2.py — stress version.

Increase quant work until it's ~50-100% of the main op's duration.
That's the realistic regime for SoloHeaven demotion: during a 50ms decode step
the bg thread might be quantizing dozens of chunks across layers.

If Metal truly parallelizes two streams, we'd expect combined time ≈ max(main, quant).
If Metal just serializes with non-blocking queue submission, combined ≈ sum.
"""

import time, threading, statistics
import mlx.core as mx

MAIN_N = 2048
MAIN_ITERS = 50               # ~40 ms — closer to realistic decode-step GEMM
QUANT_CHUNK = (8, 2048, 128)  # (heads, seq, head_dim) bf16 ≈ 4 MB
QUANT_COUNT = 400             # ~40-50ms of quant — comparable to main
WARMUP = 2
TRIALS = 6

DEV = mx.default_device()
bg_stream = mx.new_stream(DEV)
print(f"default stream: {mx.default_stream(DEV)}")
print(f"bg stream     : {bg_stream}")

# ---- workloads ----

def make_main():
    a = mx.random.normal((MAIN_N, MAIN_N)).astype(mx.bfloat16)
    mx.eval(a); return a

def main_op_inplace(a, stream=None):
    ctx = mx.stream(stream) if stream else None
    if ctx: ctx.__enter__()
    try:
        x = a
        for _ in range(MAIN_ITERS):
            x = x @ x.T
            x = x * mx.array(0.5, dtype=mx.bfloat16)
        return x
    finally:
        if ctx: ctx.__exit__(None, None, None)

def make_quant():
    arrs = [mx.random.normal(QUANT_CHUNK).astype(mx.bfloat16) for _ in range(QUANT_COUNT)]
    mx.eval(arrs); return arrs

def quant_chunk(arr, stream=None):
    ctx = mx.stream(stream) if stream else None
    if ctx: ctx.__enter__()
    try:
        q, s, z = mx.quantize(arr.reshape(-1, QUANT_CHUNK[-1]), bits=4, group_size=64)
        return (q, s, z)
    finally:
        if ctx: ctx.__exit__(None, None, None)

def quant_all(arrs, stream=None):
    return [quant_chunk(a, stream=stream) for a in arrs]

# ---- trials ----

def t_main_solo(a):
    t0 = time.perf_counter()
    x = main_op_inplace(a); mx.eval(x)
    return time.perf_counter() - t0

def t_quant_solo(arrs, stream=None):
    t0 = time.perf_counter()
    outs = quant_all(arrs, stream=stream); mx.eval(outs)
    return time.perf_counter() - t0

def t_combined_same_stream(a, arrs):
    t0 = time.perf_counter()
    x = main_op_inplace(a)
    outs = quant_all(arrs)
    mx.eval(x, outs)
    return time.perf_counter() - t0

def t_combined_async_sep(a, arrs):
    t0 = time.perf_counter()
    x = main_op_inplace(a)                    # default stream
    outs = quant_all(arrs, stream=bg_stream)  # bg stream
    mx.async_eval(x)
    mx.async_eval(outs)
    mx.eval(x); mx.eval(outs)
    return time.perf_counter() - t0

def t_combined_bgthread(a, arrs):
    main_t = [None]; total_t0 = time.perf_counter()
    def worker():
        outs = quant_all(arrs, stream=bg_stream)
        mx.eval(outs)
    th = threading.Thread(target=worker)
    th.start()
    t0 = time.perf_counter()
    x = main_op_inplace(a); mx.eval(x)
    main_t[0] = time.perf_counter() - t0
    th.join()
    total = time.perf_counter() - total_t0
    return main_t[0], total

a = make_main()
arrs = make_quant()

# warmup
for _ in range(WARMUP):
    t_main_solo(a)
    t_quant_solo(arrs)
    t_quant_solo(arrs, stream=bg_stream)
    t_combined_same_stream(a, arrs)
    t_combined_async_sep(a, arrs)
    t_combined_bgthread(a, arrs)

def med(xs): return statistics.median(xs)
def stats(xs): return f"min={min(xs)*1000:7.1f}  med={med(xs)*1000:7.1f}  max={max(xs)*1000:7.1f} ms"

main_solo = [t_main_solo(a) for _ in range(TRIALS)]
quant_solo_def = [t_quant_solo(arrs) for _ in range(TRIALS)]
quant_solo_bg = [t_quant_solo(arrs, stream=bg_stream) for _ in range(TRIALS)]
same_combined = [t_combined_same_stream(a, arrs) for _ in range(TRIALS)]
async_combined = [t_combined_async_sep(a, arrs) for _ in range(TRIALS)]

bg_main = []; bg_total = []
for _ in range(TRIALS):
    m, t = t_combined_bgthread(a, arrs)
    bg_main.append(m); bg_total.append(t)

print()
print("=== RESULTS (QUANT_COUNT=%d chunks of %s bf16) ===" % (QUANT_COUNT, QUANT_CHUNK))
print(f"solo main op              : {stats(main_solo)}")
print(f"solo quant (default strm) : {stats(quant_solo_def)}")
print(f"solo quant (bg stream)    : {stats(quant_solo_bg)}")
print(f"same-stream combined      : {stats(same_combined)}")
print(f"async_eval sep-stream     : {stats(async_combined)}")
print(f"bg-thread main-only wall  : {stats(bg_main)}")
print(f"bg-thread total wall      : {stats(bg_total)}")

M = med(main_solo); Q = med(quant_solo_def)
print()
print(f"main alone          : {M*1000:.1f} ms")
print(f"quant alone         : {Q*1000:.1f} ms")
print(f"ideal sum (serial)  : {(M+Q)*1000:.1f} ms")
print(f"ideal max (parallel): {max(M,Q)*1000:.1f} ms")
print(f"same-stream observed: {med(same_combined)*1000:.1f} ms")
print(f"async sep-stream    : {med(async_combined)*1000:.1f} ms")
print(f"bg-thread total     : {med(bg_total)*1000:.1f} ms")
print(f"bg-thread MAIN wall : {med(bg_main)*1000:.1f} ms   (impact on generation)")
print()
impact = med(bg_main)/M - 1
print(f"Main-op slowdown from bg quant: {impact*100:+.2f}%")
print(f"Below 5%? {'YES' if abs(impact) < 0.05 else 'NO'}")

# Parallelism indicator: if same-stream combined ≈ M+Q but async ≈ max(M,Q),
# Metal is genuinely overlapping.
par_indicator = (med(same_combined) - med(async_combined)) / Q
print(f"Overlap indicator (higher = more parallel): {par_indicator*100:.1f}% of quant time saved")
