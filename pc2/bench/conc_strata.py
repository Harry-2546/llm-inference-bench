"""Concurrency scaling on r10 with the bench_strata.py workload (768 tok greedy, reasoning_effort none)."""
import sys, threading, time, random, statistics
sys.argv = ["x", "none"]
src = open("bench_strata_r10.py").read().split("mode = sys.argv[1]")[0]
exec(src)  # brings in BASE and stream()
P = ["Implement a Python LRU cache with TTL and full unit tests. Output complete source.",
     "Write a long, detailed essay about the history of the Rhône river.",
     "用中文详细介绍长江流域的地理、历史和经济，分多个章节。"]
W = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
def run(n, ctx_words=0):
    res = [None] * n
    def one(k):
        pre = ""
        if ctx_words:
            rng = random.Random(time.time() + k)
            pre = f"[{rng.random()}] " + " ".join(rng.choice(W) for _ in range(ctx_words)) + "\n\nIgnore the words above. "
        res[k] = stream({"messages": [{"role": "user", "content": pre + P[k % 3]}], "max_tokens": 768, "temperature": 0, "reasoning_effort": "none"})
    t0 = time.perf_counter(); th = [threading.Thread(target=one, args=(k,)) for k in range(n)]
    [t.start() for t in th]; [t.join() for t in th]; wall = time.perf_counter() - t0
    toks = sum(r["n"] for r in res); per = [r["tps"] for r in res]
    print(f"[ctx {res[0]["prompt"]:6d} tok] N={n:2d}: aggregate {toks / wall:6.0f} tok/s (end-to-end) | per-stream {statistics.mean(per):5.1f} tok/s (min {min(per):.0f}) | mean TTFT {statistics.mean(r["ttft"] for r in res)*1000:5.0f} ms | wall {wall:.1f}s", flush=True)
run(3)  # warm-up
for n in (1, 2, 4, 8, 16):
    run(n) if n > 1 else [run(1) for _ in range(3)]
for n in (1, 4, 8):
    run(n, 23000)
