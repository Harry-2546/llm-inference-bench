"""Needle-in-a-haystack on r10: hide a fact at a given depth of an N-token filler, ask for it back.
Usage: needle.py <target_tokens> [depth ...]   (depth 0..1, default 0.1 0.5 0.9)"""
import json, random, sys, time, urllib.request

B = "http://localhost:8036"
M = "vllm/qwen3.8-flash-next"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
TOK_PER_WORD = 1.4  # measured: 23,000 filler words = 32,221 tokens


def ask(text, max_tokens=4000):
    body = {"model": M, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d["choices"][0]["message"].get("content") or "", d["usage"]["prompt_tokens"], time.time() - t0


target = int(sys.argv[1])
depths = [float(x) for x in sys.argv[2:]] or [0.1, 0.5, 0.9]
for depth in depths:
    rng = random.Random()
    code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
    city = rng.choice(["Lisbon", "Osaka", "Quito", "Tallinn", "Hobart", "Windhoek", "Reykjavik", "Cusco"])
    n = int(target / TOK_PER_WORD)
    words = [rng.choice(WORDS) for _ in range(n)]
    needle = f" The secret vault code is {code} and the vault is in {city}. "
    words.insert(int(n * depth), needle)
    q = ("\n\nSomewhere in the text above there is one sentence about a secret vault. "
         "What is the vault code, and in which city is the vault? Answer in one short line.")
    ans, ptoks, dt = ask(" ".join(words) + q)
    ok = code in ans and city in ans
    print(f"[{ptoks:7d} tok, depth {depth:.0%}] {'PASS' if ok else 'FAIL'} in {dt:5.1f}s | expected {code} / {city} | got: {ans.strip()[:120]!r}", flush=True)
