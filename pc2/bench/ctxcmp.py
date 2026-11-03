"""Decode speed at real agent context lengths, temperature 1 with thinking. Usage: ctxcmp.py PORT"""
import json, random, sys, time, urllib.request
PORT = sys.argv[1]
URL = f"http://localhost:{PORT}/v1/chat/completions"
W = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
     "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
Q = {"code": "Write a complete red-black tree implementation in C with insert, delete and search, explaining each function.",
     "prose": "Write a long essay about how the printing press changed European society."}


def call(text, temp=1.0):
    body = {"model": "vllm/qwen3.8-flash-next", "messages": [{"role": "user", "content": text}], "max_tokens": 768, "temperature": temp,
            "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    a = time.time(); first = None; last = None; u = None
    for line in urllib.request.urlopen(r, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            u = d["usage"]
        if d.get("choices"):
            dl = d["choices"][0].get("delta", {})
            if dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content"):
                last = time.time()
                first = first or last
    return u["prompt_tokens"], u["completion_tokens"] / (last - first), first - a


for words in (71000, 143000, 214000):
    random.seed(words)
    filler = " ".join(random.choice(W) for _ in range(words))
    for name, q in Q.items():
        rs = [call(f"Background notes, ignore them:\n{filler}\n\nTask: {q}") for _ in range(4)]
        print(f"ctx {rs[0][0]:7d} {name:5} tok/s {' '.join('%.0f' % r[1] for r in rs[1:])} mean {sum(r[1] for r in rs[1:]) / 3:.0f} | first fill {rs[0][2]:.1f}s then {rs[-1][2]:.2f}s", flush=True)
