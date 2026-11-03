"""Same workloads as the 27B bench scripts (bench_step / bench_hermes / bench_prefill), pointed at Strata :8080.
No vLLM /metrics here, so tok/s = completion tokens / (last chunk - first chunk)."""
import json, random, statistics, sys, time, uuid, urllib.request
BASE = "http://127.0.0.1:8036"

def stream(body):
    body = dict(body, model="vllm/qwen3.8-flash-next", stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = None; n_chunks = 0; calls = {}; content = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: {"): continue
            c = json.loads(line[6:])
            if c.get("usage"): usage = c["usage"]
            if not c.get("choices"): continue
            d = c["choices"][0].get("delta", {})
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                last = time.perf_counter(); first = first or last; n_chunks += 1
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                calls.setdefault(tc.get("index", 0), (tc.get("function") or {}).get("name") or "")
    end = time.perf_counter()
    n = usage["completion_tokens"] if usage else n_chunks
    return dict(total=end - t0, ttft=first - t0, tps=(n - 1) / (last - first) if last > first else 0, n=n,
                prompt=(usage or {}).get("prompt_tokens"), calls=calls, content=content)

mode = sys.argv[1]
if mode == "step":   # = bench_step.py 3 768 [ctx_words]
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    filler = " ".join(("alpha bravo charlie delta echo foxtrot golf hotel india juliet " * (ctx // 10 + 1)).split()[:ctx])
    filler = filler + "\n\nIgnore the words above. " if ctx else ""
    P = {"code": "Implement a Python LRU cache with TTL and full unit tests. Output complete source.",
         "prose": "Write a long, detailed essay about the history of the Rhône river.",
         "zh": "用中文详细介绍长江流域的地理、历史和经济，分多个章节。"}
    g = lambda p: stream({"messages": [{"role": "user", "content": filler + p}], "max_tokens": 768, "temperature": 0, "reasoning_effort": "none"})
    w = g(P["code"])
    if ctx: print(f"(prompt {w['prompt']} tokens, first fill {w['ttft']:.1f}s)")
    for k, p in P.items():
        rs = [g(p) for _ in range(3)]
        t = [r["tps"] for r in rs]
        print(f"[{k:5s}] tok/s {statistics.mean(t):6.1f} (min {min(t):.1f} max {max(t):.1f}) | {rs[0]['n']} tok | TTFT {statistics.mean(r['ttft'] for r in rs)*1000:.0f} ms")
elif mode == "hermes":   # = bench_hermes.py 2 (default reasoning effort, like the 27B run)
    snap = json.load(open(sys.argv[2], encoding="utf-8"))
    T = [("chat", "用中文简单介绍一下你自己能帮我做什么。"), ("chat", "给我三个周末在家放松的建议，简洁一点。"), ("tool", "帮我看一下 C 盘还剩多少空间。")]
    eff = sys.argv[3] if len(sys.argv) > 3 else None
    def turn(m):
        b = {"max_tokens": 2048, "tools": snap["tools"], "messages": [{"role": "system", "content": snap["system"]}, {"role": "user", "content": f"{m} [{uuid.uuid4().hex[:6]}]"}]}
        if eff: b["reasoning_effort"] = eff
        return stream(b)
    w = turn(T[0][1]); print(f"(warm-up: prompt {w['prompt']} tokens, cold TTFT {w['ttft']:.2f}s)")
    rows = []
    for _ in range(2):
        for kind, m in T:
            r = turn(m); leak = "<tool_call" in r["content"] or "<function" in r["content"]
            ok = "" if kind == "chat" else (" tool OK" if r["calls"] and not leak else " tool FAIL")
            rows.append((kind, r["total"], r["tps"]))
            print(f"{kind:4s}: {r['n']:4d} tok  total {r['total']:5.2f}s  TTFT {r['ttft']*1000:4.0f}ms  {r['tps']:5.1f} tok/s{ok}  calls={list(r['calls'].values())}")
    for kind in ("chat", "tool"):
        r = [x for x in rows if x[0] == kind]
        print(f"[{kind}] mean total {sum(x[1] for x in r)/len(r):.2f}s  mean {sum(x[2] for x in r)/len(r):.1f} tok/s")
elif mode == "prefill":  # = bench_prefill.py 2
    W = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    def once(n):
        rng = random.Random(); text = f"[{rng.random()}] " + " ".join(rng.choice(W) for _ in range(n)) + "\n\nSay OK."
        body = {"model": "vllm/qwen3.8-flash-next", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0, "reasoning_effort": "none"}
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter(); r = json.load(urllib.request.urlopen(req, timeout=1800)); return r["usage"]["prompt_tokens"], time.perf_counter() - t0
    once(2000)
    for n in (12000, 46000):
        res = [once(n) for _ in range(2)]; toks = res[0][0]; lat = statistics.median(r[1] for r in res)
        print(f"[prefill {toks:6d} tok] latency {lat:6.2f}s (min {min(r[1] for r in res):.2f}) | {toks/lat:7.0f} tok/s")
