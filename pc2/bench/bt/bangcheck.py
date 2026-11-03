import json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8037/v1/chat/completions"
METRICS = "http://127.0.0.1:8037/metrics"
TOPICS = ["a red-black tree in C", "an LRU cache in Python", "a CSV reader in Go", "a rate limiter in Rust",
          "TCP congestion control", "how B-trees split pages", "the French Revolution", "photosynthesis",
          "general relativity for students", "the history of the printing press", "hash table resizing",
          "a tokenizer in TypeScript"]
FORMS = ["Write a long, detailed explanation with code where useful: {}.",
         "Write a step by step tutorial about {}.",
         "用中文详细讲解：{}，并举例。",
         "Write an essay of at least 800 words about {}."]
PROMPTS = [f.format(t) for t in TOPICS for f in FORMS]
BANG = re.compile(r"!{6,}")


def running():
    text = urllib.request.urlopen(urllib.request.Request(METRICS, headers={"Accept": "text/plain"}), timeout=10).read().decode()
    m = re.search(r"vllm:num_requests_running\{[^}]*\} ([0-9.]+)", text)
    return float(m.group(1)) if m else -1


def one(i):
    body = {"model": "vllm/qwen3.8-flash-next", "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": PROMPTS[i]}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    msg = d["choices"][0]["message"]
    text = ((msg.get("reasoning") or msg.get("reasoning_content")) or "") + (msg.get("content") or "")
    hit = BANG.search(text)
    return i, d["usage"]["completion_tokens"], time.time() - t, (hit.start() if hit else -1), text[max(0, hit.start() - 80):hit.start() + 40] if hit else ""


busy = running()
print("running before:", busy)
if busy > 0:
    raise SystemExit("service busy, not testing")
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, range(len(PROMPTS))))
wall = time.time() - t0
bad = [r for r in res if r[3] >= 0]
toks = sum(r[1] for r in res)
print(f"generations {len(res)}  corrupt {len(bad)}  tokens {toks}  wall {wall:.0f}s  total {toks / wall:.0f} tok/s")
print("short generations:", sum(r[1] < 1024 for r in res))
for r in bad:
    print("BAD", r[0], repr(PROMPTS[r[0]][:50]), "at char", r[3], repr(r[4]))
