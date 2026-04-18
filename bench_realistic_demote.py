"""
bench_realistic_demote.py — the question that actually matters.

SoloHeaven demotion pattern: each generation step (~40-60ms), we want to
quantize a small number of chunks in the background. What's the safe rate?

We sweep: K ∈ {0, 1, 2, 4, 8, 16, 32} chunks per step.
For each K, measure main-thread generation step time vs K=0 baseline.

Also: GIL check. Verify that the bg Python thread actually runs while the
main thread is blocked in mx.eval (i.e. MLX releases the GIL during eval).
"""

import time, threading, statistics
import mlx.core as mx

MAIN_N = 2048
MAIN_ITERS = 50           # ~40 ms main step
QUANT_CHUNK = (8, 2048, 128)  # ~4 MB bf16
WARMUP = 3
TRIALS = 10

DEV = mx.default_device()
bg_stream = mx.new_stream(DEV)

def make_main():
    a = mx.random.normal((MAIN_N, MAIN_N)).astype(mx.bfloat16)
    mx.eval(a); return a

def main_step(a):
    x = a
    for _ in range(MAIN_ITERS):
        x = x @ x.T
        x = x * mx.array(0.5, dtype=mx.bfloat16)
    return x

def make_chunks(k):
    arrs = [mx.random.normal(QUANT_CHUNK).astype(mx.bfloat16) for _ in range(k)]
    mx.eval(arrs); return arrs

def quant_all(arrs, stream):
    with mx.stream(stream):
        outs = [mx.quantize(a.reshape(-1, QUANT_CHUNK[-1]), bits=4, group_size=64)
                for a in arrs]
    return outs

def trial_with_bg(a, chunks):
    ev = threading.Event()
    tick_count = [0]
    def worker():
        if chunks:
            outs = quant_all(chunks, bg_stream)
            mx.eval(outs)
        ev.set()
    th = threading.Thread(target=worker)
    t0 = time.perf_counter()
    th.start()
    x = main_step(a); mx.eval(x)
    main_done = time.perf_counter() - t0
    th.join()
    total = time.perf_counter() - t0
    return main_done, total

a = make_main()

# baseline
for _ in range(WARMUP):
    x = main_step(a); mx.eval(x)
baseline = []
for _ in range(TRIALS):
    t0 = time.perf_counter()
    x = main_step(a); mx.eval(x)
    baseline.append(time.perf_counter() - t0)
B = statistics.median(baseline)
print(f"baseline main step: median={B*1000:.2f} ms  min={min(baseline)*1000:.2f}  max={max(baseline)*1000:.2f}")
print()
print(f"{'K':>4}  {'main_med(ms)':>13}  {'slowdown':>10}  {'total_med(ms)':>14}  {'quant_alone(ms)':>16}")
print("-"*70)

for K in [0, 1, 2, 4, 8, 16, 32, 64]:
    if K == 0:
        chunks = []
    else:
        chunks = make_chunks(K)
    # warmup this K
    for _ in range(WARMUP):
        trial_with_bg(a, chunks)
    mains = []; totals = []
    for _ in range(TRIALS):
        m, t = trial_with_bg(a, chunks)
        mains.append(m); totals.append(t)
    # measure solo quant time for reference
    if chunks:
        qsolo = []
        for _ in range(3):
            t0 = time.perf_counter()
            outs = quant_all(chunks, bg_stream); mx.eval(outs)
            qsolo.append(time.perf_counter() - t0)
        qs = statistics.median(qsolo) * 1000
    else:
        qs = 0.0
    mm = statistics.median(mains)
    tm = statistics.median(totals)
    print(f"{K:>4}  {mm*1000:>13.2f}  {(mm/B-1)*100:>+9.2f}%  {tm*1000:>14.2f}  {qs:>16.2f}")

# ---- GIL / threading check ----
print()
print("=== GIL / thread concurrency check ===")
# While main thread does mx.eval, does a bg Python-only thread tick?
ticks = [0]
stop = [False]
def ticker():
    while not stop[0]:
        ticks[0] += 1
        time.sleep(0.001)  # ~1kHz
th = threading.Thread(target=ticker, daemon=True)
th.start()
ticks_before = ticks[0]
t0 = time.perf_counter()
for _ in range(5):
    x = main_step(a); mx.eval(x)
elapsed = time.perf_counter() - t0
ticks_after = ticks[0]
stop[0] = True; th.join()
expected = elapsed / 0.001
got = ticks_after - ticks_before
print(f"During {elapsed*1000:.1f} ms of main_step+eval:")
print(f"  bg ticker expected ~{expected:.0f} ticks (1kHz)")
print(f"  actually got       {got} ticks")
print(f"  → GIL released during mx.eval: {'YES' if got > expected * 0.5 else 'NO/PARTIAL'}")

# ---- concurrent-quant thread safety ----
print()
print("=== Two Python threads, each quantizing on own stream, concurrently ===")
s1 = mx.new_stream(DEV); s2 = mx.new_stream(DEV)
c1 = make_chunks(32); c2 = make_chunks(32)
errs = []
def w(ch, st, tag):
    try:
        for _ in range(5):
            outs = quant_all(ch, st); mx.eval(outs)
    except Exception as e:
        errs.append((tag, e))
t0 = time.perf_counter()
t1 = threading.Thread(target=w, args=(c1, s1, "A"))
t2 = threading.Thread(target=w, args=(c2, s2, "B"))
t1.start(); t2.start()
t1.join(); t2.join()
dur = time.perf_counter() - t0
print(f"2 threads x 5 iters x 32 chunks each: {dur*1000:.1f} ms, errors={errs}")
print("No crash → MLX is thread-safe for concurrent quant on separate streams.")
