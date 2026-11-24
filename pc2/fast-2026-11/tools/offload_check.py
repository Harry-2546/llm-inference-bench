#!/usr/bin/env python3
"""offload_check.py : CPU KV offload correctness on :8036.
A (~50K tokens) at temp 0 -> 13 fillers of ~50K tokens (> GPU KV pool 594K) -> A again.
Checks the second A is served from the CPU tier (external prefix cache hits rise) and compares outputs.
Note: full-compute vs GPU-prefix-hit outputs can already differ by a word or two at temp 0 (different kernel
paths), so judge the CPU-hit output against that baseline difference, not against exact equality."""
import json, random, re, time, urllib.request

B, M = "http://localhost:8036", "vllm/qwen3.8-flash-next"
WORDS = ("apple river stone cloud maple orbit lantern copper velvet harbor quartz meadow falcon ember "
         "willow cobalt prism tundra saffron canyon").split()


def text(seed, n=50000):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(n // 3))


def ask(p, n=48):
    body = {"model": M, "messages": [{"role": "user", "content": p + "\n\nSummarize the text above in one sentence."}],
            "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u["prompt_tokens"], \
        (u.get("prompt_tokens_details") or {}).get("cached_tokens"), round(time.time() - t, 2)


def ext():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(v) for v in re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.e+]+)$" % n, t, re.M))
    return g("external_prefix_cache_hits_total"), g("external_prefix_cache_queries_total")


A = text(777)
a1 = ask(A, 160); print("A1 full compute", a1[1:])
a2 = ask(A, 160); print("A2 GPU hit     ", a2[1:])
for i in range(13):
    f = ask(text(2000 + i), 4)
e0 = ext()
a3 = ask(A, 160); e1 = ext()
print("A3 CPU hit     ", a3[1:], "external hits +%d" % (e1[0] - e0[0]))
print("GPU-hit == CPU-hit:", a2[0] == a3[0], "| full == GPU-hit:", a1[0] == a2[0])
for n, x in (("A1", a1), ("A2", a2), ("A3", a3)): print(n, repr(x[0][:400]))
