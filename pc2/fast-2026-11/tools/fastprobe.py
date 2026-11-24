#!/usr/bin/env python3
"""fastprobe.py NAME: quick speed probe against :8036.
sanity (19*23, one tool call) -> single stream x2 (think off, ignore_eos, 1024) -> 8-way (16 x 1024)
-> one cold ~60K-token prompt TTFT. Appends one JSON line to $PROBE_DIR/results.jsonl (default: current dir)."""
import json, os, random, re, subprocess, sys, time, urllib.request
import concurrent.futures as cf

B = "http://localhost:8036"
MODEL = "vllm/qwen3.8-flash-next"
name = sys.argv[1]
res = {"name": name, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def post(path, body, timeout=900):
    req = urllib.request.Request(B + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(m) for m in re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.e+]+)$" % n, t, re.M))
    return g("spec_decode_num_accepted_tokens_total"), g("spec_decode_num_drafts_total")


def acc(m0, m1):
    d = m1[1] - m0[1]
    return round(1 + (m1[0] - m0[0]) / d, 2) if d > 0 else None


def chat(content, n, think=False, temp=0.6, ignore_eos=False, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": temp, "chat_template_kwargs": {"enable_thinking": think}}
    if ignore_eos:
        body["ignore_eos"] = True
    body.update(kw)
    return post("/v1/chat/completions", body)


# --- sanity
ok_math = ok_tool = False
try:
    r = chat("What is 19*23? Reply with only the number.", 32, temp=0)
    txt = r["choices"][0]["message"].get("content") or ""
    ok_math = "437" in txt
    res["math_reply"] = txt[:60]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = chat("What's the weather in Paris right now?", 256, temp=0, tools=tools, tool_choice="auto")
    tc = r["choices"][0]["message"].get("tool_calls") or []
    ok_tool = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"]
except Exception as e:
    res["sanity_error"] = repr(e)[:200]
# --- vision check (not part of the sanity gate): 64x64 PNG, left half red / right half blue


def _png(w, h, px):
    import struct, zlib
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


try:
    import base64
    img = base64.b64encode(_png(64, 64, lambda x, y: (220, 20, 20) if x < 32 else (20, 20, 220))).decode()
    r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 32, "temperature": 0,
                                      "chat_template_kwargs": {"enable_thinking": False},
                                      "messages": [{"role": "user", "content": [
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                          {"type": "text", "text": "What two colors are in this image? Answer with two words."}]}]})
    vt = (r["choices"][0]["message"].get("content") or "").lower()
    res["vision_reply"] = vt[:60]
    res["vision_ok"] = "red" in vt and "blue" in vt
except Exception as e:
    res["vision_ok"] = False
    res["vision_error"] = repr(e)[:160]

res["sanity"] = ok_math and ok_tool
res["sanity_math"], res["sanity_tool"] = ok_math, ok_tool

# --- single stream
P = "Write a detailed, well-structured explanation of TCP congestion control for a university student."
# warm-up: one full-length run, discarded (lazy kernel prep, first-shape JIT, cache effects)
# warm-up count / measured runs: $PROBE_DIR/probe_cfg "WARM NSINGLE" (default "1 3")
try:
    WARM, NSINGLE = (int(x) for x in open(os.path.join(os.environ.get("PROBE_DIR", "."), "probe_cfg")).read().split()[:2])
except Exception:
    WARM, NSINGLE = 1, 3
res["probe_cfg"] = [WARM, NSINGLE]
warms = []
for _ in range(WARM):
    t = time.time(); r = chat(P, 1024, ignore_eos=True)
    warms.append(round(r["usage"]["completion_tokens"] / (time.time() - t), 1))
res["single_warm"] = warms[-1]
res["single_warms"] = warms
singles = []
m0 = metrics()
for _ in range(NSINGLE):
    t = time.time()
    r = chat(P, 1024, ignore_eos=True)
    singles.append(r["usage"]["completion_tokens"] / (time.time() - t))
m1 = metrics()
res["single_tok_s"] = round(sum(singles) / len(singles), 1)
res["single_runs"] = [round(x, 1) for x in singles]
res["single_accept"] = acc(m0, m1)

# --- 8-way
TOPICS = ["TCP congestion control", "photosynthesis", "the French Revolution", "binary search trees", "how vaccines work",
          "the Roman aqueducts", "quantum entanglement", "compound interest", "plate tectonics", "the history of jazz",
          "garbage collection in Java", "the water cycle", "Byzantine fault tolerance", "black holes", "supply and demand",
          "the immune system"]


def one(i):
    r = chat("Write a detailed, well-structured explanation of %s for a university student." % TOPICS[i % 16], 1024,
             ignore_eos=True)
    return r["usage"]["completion_tokens"]


print("partial", json.dumps(res), flush=True)
try:
    with cf.ThreadPoolExecutor(8) as ex:   # warm-up round at 8-way, discarded
        list(ex.map(lambda i: chat("Write a short note about topic %d." % i, 256, ignore_eos=True), range(8)))
    c8 = []
    m0 = metrics()
    for _ in range(2):
        t = time.time()
        with cf.ThreadPoolExecutor(8) as ex:
            toks = sum(ex.map(one, range(16)))
        c8.append(toks / (time.time() - t))
    m1 = metrics()
    res["c8_tok_s"] = round(sum(c8) / len(c8))
    res["c8_runs"] = [round(x) for x in c8]
    res["c8_accept"] = acc(m0, m1)
except Exception as e:
    res["c8_error"] = repr(e)[:200]

# --- cold prefill ~60K tokens, twice (first includes any lazy kernel prep; second = steady state)
words = "alpha beta gamma delta river stone cloud number system kernel matrix ocean forest signal vector memory buffer engine".split()


def cold(tag):
    rr = random.Random(time.time())
    out = []
    while len(out) < 60000 * 0.26:
        out.append(rr.choice(words) + str(rr.randint(0, 999)))
    text = "[%d] " % rr.randint(0, 10**9) + " ".join(out) + "\nSummarize the above in one word."
    k = post("/tokenize", {"model": MODEL, "prompt": text})["count"]
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 8,
                       "stream": True}).encode()
    t0 = time.time()
    for line in urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=900):
        if line.startswith(b"data:"):
            ttft = time.time() - t0
            break
    res[tag + "_tokens"], res[tag + "_ttft_s"], res[tag + "_tok_s"] = k, round(ttft, 2), round(k / ttft)


for tag in ("cold", "cold2"):
    try:
        if "c8_error" in res:
            raise RuntimeError("skip cold: server dead")
        cold(tag)
    except Exception as e:
        res[tag + "_error"] = repr(e)[:200]

try:
    res["gpu_mem_mib"] = [int(x) for x in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()]
except Exception:
    pass
line = json.dumps(res)
print(line, flush=True)
with open(os.path.join(os.environ.get("PROBE_DIR", "."), "results.jsonl"), "a") as f:
    f.write(line + "\n")
