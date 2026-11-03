import hashlib, json, time, random, re, sys, urllib.request
B = "http://localhost:8037"
M = "vllm/qwen3.8-flash-next"

def metrics():
    t = urllib.request.urlopen(urllib.request.Request(B + "/metrics", headers={"Accept": "text/plain"})).read().decode()
    out = {}
    for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
              "vllm:spec_decode_num_draft_tokens_total"):
        out[k] = sum(float(m) for m in re.findall(r"^" + k + r"\{[^}]*\} ([0-9.e+]+)", t, re.M))
    return out

def stream(messages, max_tokens):
    body = json.dumps({"model": M, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(B + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = []
    for line in urllib.request.urlopen(req):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("usage"): usage = d["usage"]
        if d.get("choices"):
            dl = d["choices"][0]["delta"]
            piece = dl.get("content") or dl.get("reasoning_content") or (dl.get("reasoning") or dl.get("reasoning_content"))
            if piece:
                text.append(piece)
                if first is None: first = time.time()
    stream.text = "".join(text)
    return t0, first, time.time(), usage

def stream_local(messages, max_tokens):
    r = stream(messages, max_tokens)
    return r

def decode(n=3, max_tokens=2048):
    p = "Write a complete Python implementation of a red-black tree with insert, delete, search, in-order iteration, and a pytest test suite. Output only code."
    for i in range(n):
        m0 = metrics()
        t0, f, t1, u = stream([{"role": "user", "content": p}], max_tokens)
        m1 = metrics()
        toks = u["completion_tokens"]; dt = t1 - f
        steps = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
        acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
        drf = m1["vllm:spec_decode_num_draft_tokens_total"] - m0["vllm:spec_decode_num_draft_tokens_total"]
        print(f"decode run {i+1}: {toks} tok in {dt:.1f}s = {toks/dt:.0f} tok/s | steps/s {steps/dt:.1f} | accept {acc/max(drf,1):.2f} | TTFT {f-t0:.2f}s | md5 {hashlib.md5(stream.text.encode()).hexdigest()[:8]}", flush=True)

def prefill(target):
    words = "alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river stone cloud ember".split()
    rnd = random.Random(time.time())
    text = " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(int(target / 2.6)))
    t0, f, t1, u = stream([{"role": "user", "content": text + "\n\nReply with OK."}], 1)
    pt = u["prompt_tokens"]
    print(f"prefill {pt} tok: TTFT {f - t0:.2f}s = {pt/(f-t0):.0f} tok/s", flush=True)

def check():
    rnd = random.Random(42); ok = 0
    qs = [(rnd.randint(100, 9999), rnd.randint(100, 9999)) for _ in range(8)]
    for a, b in qs:
        stream([{"role": "user", "content": f"Compute {a}*{b}. Reply with only the integer, no commas."}], 3000)
        ans = re.findall(r"-?\d+", stream.text.split("</think>")[-1].replace(",", ""))
        ok += bool(ans) and ans[-1] == str(a * b)
    print(f"check: {ok}/8 correct", flush=True)

def conc(n=8, ctx=16000, out=1500):
    import threading
    words = "alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river stone cloud ember".split()
    res = [None] * n
    def one(k):
        rnd = random.Random(time.time() + k)
        text = " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(int(ctx / 3.6)))
        msg = text + "\n\nIgnore the noise above. Write a complete Python implementation of a B-tree with insert, delete, search, range queries, and a pytest suite. Output only code."
        res[k] = stream_local([{"role": "user", "content": msg}], out)
    m0 = metrics(); t0 = time.time()
    th = [threading.Thread(target=one, args=(k,)) for k in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0; m1 = metrics()
    toks = sum(r[3]["completion_tokens"] for r in res)
    ptoks = sum(r[3]["prompt_tokens"] for r in res)
    dec = sum(r[3]["completion_tokens"] / (r[2] - r[1]) for r in res)
    acc = (m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]) / max(1, m1["vllm:spec_decode_num_draft_tokens_total"] - m0["vllm:spec_decode_num_draft_tokens_total"])
    print(f"conc {n}x: prompt {ptoks//n} tok each, out {toks} tok, wall {wall:.1f}s | aggregate decode {dec:.0f} tok/s | end-to-end {toks/wall:.0f} tok/s | accept {acc:.2f}", flush=True)

for a in sys.argv[1:]:
    if a == "decode": decode()
    elif a == "check": check()
    elif a == "conc": conc()
    else: prefill(int(a))
