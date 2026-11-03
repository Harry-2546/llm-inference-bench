"""Client-side truth for the panel: one stream, then a large uncached prompt."""
import json, random, time, urllib.request
B = "http://localhost:8036/v1/chat/completions"
def call(msg, n):
    body = {"model": "vllm/qwen3.8-flash-next", "temperature": 0, "max_tokens": n, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": msg}], "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(B, json.dumps(body).encode(), {"Content-Type": "application/json"})
    a = time.time(); first = None; u = None
    for line in urllib.request.urlopen(r, timeout=180):
        if not line.startswith(b"data: {"): continue
        d = json.loads(line[6:])
        if d.get("usage"): u = d["usage"]
        if first is None and d.get("choices"): first = time.time()
    e = time.time()
    return u, first - a, e - first
u, p, g = call("Write a complete red-black tree implementation in C with insert, delete and search, explaining each function.", 2500)
print(f"single: {u['completion_tokens']} tok, ttft {p:.2f}s, {u['completion_tokens'] / g:.0f} tok/s after first token")
time.sleep(3)
words = [str(random.randrange(10**6)) for _ in range(60000)]
u, p, g = call(" ".join(words) + "\nReply with OK.", 8)
print(f"prefill: prompt {u['prompt_tokens']} tok, ttft {p:.2f}s, {u['prompt_tokens'] / p:.0f} tok/s")
